"""Goal-pose readout for the Insert-Peg bank.

The physical goal metric uses the hand and peg poses in the goal image, not
the current arm. The wrist origin is not a rendered geom, and rebuilding it
from estimated joints was the previous failure. This script predicts those
metric poses directly from images at the same 25-step offset as the goals.
Episodes 0–49 are never used to fit or select the model. Simulator-state
labels are supervision SMWM does not receive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit
import anygoal_dynamics_factor as factor
import anygoal_proprio_baseline as proprio
import anygoal_state_estimator as estimator

GOAL_OFFSET = 25
NEIGHBOR_OFFSETS = (10, 20, 25, 30, 40)
OUTPUT_DIM = 8


def metric_targets(states: np.ndarray) -> np.ndarray:
    """Hand and object position and sin/cos of their planar angles."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != audit.STATE_DIM:
        raise ValueError("Expected (N, 44) states.")
    hand_angle = estimator.planar_angles(states[:, 31:33])
    object_angle = estimator.planar_angles(states[:, 35:37])
    return np.concatenate(
        [
            states[:, 29:31],
            np.sin(hand_angle)[:, None],
            np.cos(hand_angle)[:, None],
            states[:, 33:35],
            np.sin(object_angle)[:, None],
            np.cos(object_angle)[:, None],
        ],
        axis=1,
    )


def goal_state_from_prediction(prediction: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Write the metric pose into a goal vector. Other entries are unused."""
    raw = np.asarray(prediction, dtype=np.float64).reshape(-1) * std + mean
    if raw.shape != (OUTPUT_DIM,):
        raise ValueError("Goal prediction must have 8 values.")
    state = np.zeros(audit.STATE_DIM, dtype=np.float64)
    state[29:31] = raw[0:2]
    hand_angle = float(np.arctan2(raw[2], raw[3]))
    state[31] = np.cos(0.5 * hand_angle)
    state[32] = np.sin(0.5 * hand_angle)
    state[33:35] = raw[4:6]
    object_angle = float(np.arctan2(raw[6], raw[7]))
    state[35] = np.cos(0.5 * object_angle)
    state[36] = np.sin(0.5 * object_angle)
    return state


def goal_decision(hand_median: float, object_median: float, regret: dict | None) -> dict[str, str]:
    competent = hand_median <= 0.02 and object_median <= 0.02
    if not competent:
        return {
            "decision": "INCONCLUSIVE",
            "reason": (
                f"Held-out goal hand error is {hand_median:.4f} m and object error "
                f"is {object_median:.4f} m. The goal-pose baseline is untested below 2 cm."
            ),
        }
    if regret is None:
        return {"decision": "INCONCLUSIVE", "reason": "The goal estimate met 2 cm, but ranking was not run."}
    mean, low, high = float(regret["mean"]), float(regret["ci_low"]), float(regret["ci_high"])
    if mean <= 0.12 and high <= 0.20:
        return {
            "decision": "NO-GO",
            "reason": (
                f"Measured joints, a visual object, and this goal estimate have regret "
                f"{mean:.3f} [{low:.3f}, {high:.3f}]. Conventional state plus physics "
                "closes the ranking gap."
            ),
        }
    if mean >= 0.20 and low > 0.10:
        return {
            "decision": "DIAGNOSIS",
            "reason": (
                f"The goal estimate is inside 2 cm and regret remains {mean:.3f} "
                f"[{low:.3f}, {high:.3f}]."
            ),
        }
    return {
        "decision": "INCONCLUSIVE",
        "reason": f"Regret is {mean:.3f} [{low:.3f}, {high:.3f}] and does not clearly close or keep the gap.",
    }


def _collect(data_root: Path, episodes: range, offsets: tuple[int, ...]) -> dict[str, np.ndarray]:
    path = data_root / audit.TRAIN_FILENAME
    records = {int(record["episode_id"]): record for record in audit.index_h5_episodes(path)}
    images, states, ids, offs = [], [], [], []
    handle = audit.open_h5(path)
    try:
        for episode_id in episodes:
            record = records[int(episode_id)]
            for offset in offsets:
                if offset >= int(record["length"]):
                    continue
                row = int(record["offset"]) + int(offset)
                images.append(np.asarray(handle["pixels"][row]))
                states.append(np.asarray(handle["observation"][row], dtype=np.float32))
                ids.append(int(episode_id))
                offs.append(int(offset))
            if episode_id % 200 == 0:
                print(f"cached episode {episode_id}", flush=True)
    finally:
        handle.close()
    episode_ids = np.asarray(ids, dtype=np.int64)
    split = "train" if episodes == estimator.TRAIN_EPISODES else "val"
    estimator.assert_fit_episodes(episode_ids, split)
    return {
        "images": np.stack(images).astype(np.uint8),
        "states": np.stack(states).astype(np.float32),
        "episode_id": episode_ids,
        "offset": np.asarray(offs, dtype=np.int64),
    }


def build_cache(data_root: Path, output: Path, split: str) -> dict:
    episodes = estimator.TRAIN_EPISODES if split == "train" else estimator.VAL_EPISODES
    offsets = NEIGHBOR_OFFSETS if split == "train" else (GOAL_OFFSET,)
    payload = _collect(data_root, episodes, offsets)
    destination = output / split
    destination.mkdir(parents=True, exist_ok=True)
    for key, value in payload.items():
        np.save(destination / f"{key}.npy", value)
    summary = {
        "split": split,
        "n": int(len(payload["episode_id"])),
        "offsets": list(offsets),
        "episode_min": int(payload["episode_id"].min()),
        "episode_max": int(payload["episode_id"].max()),
    }
    factor.write_json(destination / "summary.json", summary)
    return summary


def _pose_errors(states: np.ndarray, predictions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    hand, obj, hand_ang, obj_ang = [], [], [], []
    for label, pred in zip(states, predictions):
        goal = goal_state_from_prediction(pred, mean, std)
        errors = insert_peg_goal_errors(label, goal)
        hand.append(errors["hand_position"])
        obj.append(errors["object_position"])
        hand_ang.append(errors["hand_angle"])
        obj_ang.append(errors["object_angle"])
    return {
        "hand_m": np.asarray(hand),
        "object_m": np.asarray(obj),
        "hand_rad": np.asarray(hand_ang),
        "object_rad": np.asarray(obj_ang),
    }


def _report(errors: dict[str, np.ndarray]) -> dict:
    report = {name: estimator.percentile_summary(values) for name, values in errors.items()}
    report["meets_2cm"] = bool(report["hand_m"]["median"] <= 0.02 and report["object_m"]["median"] <= 0.02)
    return report


def _load_goal_model(checkpoint: Path, device: str):
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = estimator._load_encoder(device)
    encoder.load_state_dict(saved["encoder"])
    head = estimator._head(1536, OUTPUT_DIM).to(device)
    head.load_state_dict(saved["head"])
    return encoder, head, np.asarray(saved["mean"]), np.asarray(saved["std"])


def train_goal(cache: Path, init_encoder: Path, output: Path, device: str, epochs: int = 8, resume: Path | None = None) -> dict:
    import os
    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")
    started = time.perf_counter()
    train_images = np.load(cache / "train" / "images.npy", mmap_mode="r")
    train_states = np.load(cache / "train" / "states.npy")
    val_images = np.load(cache / "val" / "images.npy", mmap_mode="r")
    val_states = np.load(cache / "val" / "states.npy")
    estimator.assert_fit_episodes(np.load(cache / "train" / "episode_id.npy"), "train")
    estimator.assert_fit_episodes(np.load(cache / "val" / "episode_id.npy"), "val")
    raw = metric_targets(train_states)
    if resume is not None and resume.is_file():
        encoder, head, mean, std = _load_goal_model(resume, device)
    else:
        mean = raw.mean(axis=0)
        std = np.where(raw.std(axis=0) < 1e-6, 1.0, raw.std(axis=0))
        encoder = estimator._load_encoder(device)
        if init_encoder.is_file():
            saved = torch.load(init_encoder, map_location="cpu", weights_only=False)
            encoder.load_state_dict(saved["encoder"])
        head = estimator._head(1536, OUTPUT_DIM).to(device)
    targets = ((raw - mean) / std).astype(np.float32)
    estimator._set_trainable(encoder, head, blocks=4)
    groups = [
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": list(head.parameters()), "lr": 1e-3},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
    order = np.arange(len(targets))
    rng = np.random.default_rng(0)
    best, best_state, losses = None, None, []
    for epoch in range(epochs):
        rng.shuffle(order)
        encoder.train()
        head.train()
        epoch_loss = []
        for start in range(0, len(order), 64):
            index = order[start : start + 64]
            images = estimator._normalize_images(train_images[index], device)
            target = torch.as_tensor(targets[index], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(estimator._features(encoder, images))
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        raw_pred = estimator._predict_raw(encoder, head, val_images, device)
        report = _report(_pose_errors(val_states, raw_pred, mean, std))
        score = 0.5 * (report["hand_m"]["median"] + report["object_m"]["median"])
        losses.append(float(np.mean(epoch_loss)))
        print(
            f"epoch {epoch} loss {losses[-1]:.5f} hand {report['hand_m']['median']:.4f} "
            f"object {report['object_m']['median']:.4f} "
            f"hand_rad {report['hand_rad']['median']:.3f} obj_rad {report['object_rad']['median']:.3f}",
            flush=True,
        )
        if best is None or score < best["score"]:
            best = {"score": score, "report": report, "epoch": epoch}
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
            }
        if report["meets_2cm"]:
            break
    head.load_state_dict(best_state["head"])
    encoder.load_state_dict(best_state["encoder"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "encoder": encoder.state_dict(), "mean": mean, "std": std}, output / "goal_pose.pt")
    payload = {"seconds": time.perf_counter() - started, "losses": losses, "validation": best["report"], "epoch": best["epoch"]}
    factor.write_json(output / "train.json", payload)
    return payload


def _predict_goals(checkpoint: Path, images: np.ndarray, device: str) -> np.ndarray:
    encoder, head, mean, std = _load_goal_model(checkpoint, device)
    raw = estimator._predict_raw(encoder, head, images, device)
    return np.stack([goal_state_from_prediction(row, mean, std) for row in raw])


def rank(anchor: Path, goal_checkpoint: Path, start_checkpoint: Path, output: Path, device: str, validation: Path) -> dict:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    bank = np.load(anchor / "bank.npz")
    success = np.load(anchor / "bank" / "evaluator_10.npz")["success"]
    predicted = np.load(anchor / "scores" / "evaluator_10.npz")["predicted_cost"]
    true_start = np.asarray(bank["start_state"], dtype=np.float64)
    true_goal = np.asarray(bank["goal_state"], dtype=np.float64)
    actions = np.asarray(bank["actions"])
    start_hat = proprio._estimate(start_checkpoint, bank["start_image"], device)
    goal_hat = _predict_goals(goal_checkpoint, bank["goal_image"], device)
    measured = proprio.apply_current_proprio(start_hat, true_start, zero_object_velocity=True)
    visual = proprio_summary(estimator._rank_from_states(measured, goal_hat, actions, success))
    true_goal_row = proprio_summary(estimator._rank_from_states(measured, true_goal, actions, success))
    oracle = proprio_summary(estimator._rank_from_states(true_start, true_goal, actions, success))
    substitutions = {}
    for name, hand_true, object_true in (
        ("true_goal_hand", True, False),
        ("true_goal_object", False, True),
    ):
        mixed = goal_hat.copy()
        if hand_true:
            mixed[:, 29:33] = true_goal[:, 29:33]
        if object_true:
            mixed[:, 33:37] = true_goal[:, 33:37]
        substitutions[name] = proprio_summary(estimator._rank_from_states(measured, mixed, actions, success))
    smwm_index = np.argmin(predicted, axis=1).astype(np.int64)
    smwm_scores = factor.per_episode_scores(success, smwm_index)
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    errors = [insert_peg_goal_errors(label, pred) for label, pred in zip(true_goal, goal_hat)]
    anchor_errors = {
        "hand_m": estimator.percentile_summary([row["hand_position"] for row in errors]),
        "object_m": estimator.percentile_summary([row["object_position"] for row in errors]),
        "hand_rad": estimator.percentile_summary([row["hand_angle"] for row in errors]),
        "object_rad": estimator.percentile_summary([row["object_angle"] for row in errors]),
        "n_object_over_20cm": int(sum(row["object_position"] > 0.20 for row in errors)),
        "n_hand_over_20cm": int(sum(row["hand_position"] > 0.20 for row in errors)),
    }
    heldout = json.loads(validation.read_text())["validation"]
    payload = {
        "supervision": "simulator hand and object poses on goal-offset frames; not used by SMWM",
        "validation": heldout,
        "anchor_goal_errors": anchor_errors,
        "oracle_true_state": oracle,
        "measured_arm_visual_object_true_goal": true_goal_row,
        "measured_arm_visual_object_estimated_goal": visual,
        "substitutions": substitutions,
        "smwm_predicted": {
            "success_rate": float(smwm_scores["chosen_success"].mean()),
            "regret": audit.bootstrap_mean(smwm_scores["regret"], seed=51),
        },
    }
    payload["decision"] = goal_decision(
        heldout["hand_m"]["median"], heldout["object_m"]["median"], visual["regret"]
    )
    output.mkdir(parents=True, exist_ok=True)
    factor.write_json(output / "ranking.json", payload)
    return payload


def proprio_summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "selected"}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache")
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--split", choices=("train", "val"), required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--init-encoder", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--resume", type=Path, default=None)
    evaluate = subparsers.add_parser("rank")
    evaluate.add_argument("--anchor", type=Path, required=True)
    evaluate.add_argument("--goal-checkpoint", type=Path, required=True)
    evaluate.add_argument("--start-checkpoint", type=Path, required=True)
    evaluate.add_argument("--validation", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "cache":
        print(json.dumps(build_cache(args.data_root, args.output, args.split)))
    elif args.command == "train":
        print(json.dumps(
            train_goal(args.cache, args.init_encoder, args.output, args.device, epochs=args.epochs, resume=args.resume),
            default=audit._json_default,
        ))
    elif args.command == "rank":
        print(json.dumps(
            rank(args.anchor, args.goal_checkpoint, args.start_checkpoint, args.output, args.device, args.validation),
            default=audit._json_default,
        ))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
