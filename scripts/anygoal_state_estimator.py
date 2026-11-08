"""Supervised image-to-state estimator for the Insert-Peg ranking bank.

Episodes 0–49 are the existing evaluation anchor and are never used to fit,
normalize, or select the estimator. Episodes 50–1299 train. Episodes
1300–1460 validate. The readout is a DINOv2-B encoder plus an MLP. State
labels are simulator observations; that supervision is not available to SMWM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit
import anygoal_dynamics_factor as factor

EVAL_EPISODES = range(0, 50)
TRAIN_EPISODES = range(50, 1300)
VAL_EPISODES = range(1300, 1461)
POSITION_LIMIT_M = 0.02
CONTACT_DISTANCE_M = 0.08
FRAMES_PER_EPISODE = 8
DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Groups written into MuJoCo by InsertPegEnv.set_state. Touch is a sensor
# output and is not a controlled initial condition. Hand pose is recomputed
# from the arm joints.
STATE_GROUPS = (
    ("arm_sincos", 0, 16),
    ("arm_vel", 16, 24),
    ("obj_xz", 33, 35),
    ("obj_quat", 35, 37),
    ("obj_vel", 37, 40),
    ("target_xz", 40, 42),
    ("target_quat", 42, 44),
)
UNIT_GROUPS = {"arm_sincos", "obj_quat", "target_quat"}


def assign_split(episode_id: int) -> str:
    episode_id = int(episode_id)
    if episode_id < 0 or episode_id > 1460:
        raise ValueError(f"Episode id {episode_id} is outside 0–1460.")
    if episode_id in EVAL_EPISODES:
        return "eval"
    if episode_id in TRAIN_EPISODES:
        return "train"
    if episode_id in VAL_EPISODES:
        return "val"
    raise AssertionError(episode_id)


def assert_fit_episodes(episode_ids: np.ndarray, split: str) -> None:
    """Refuse evaluation episodes in any array used to fit or select the model."""
    if split not in {"train", "val"}:
        raise ValueError("Fit data must be train or val.")
    ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("Episode id list is empty.")
    forbidden = [int(episode) for episode in ids if assign_split(int(episode)) == "eval"]
    if forbidden:
        raise RuntimeError(f"Evaluation episodes leaked into {split}: {forbidden[:5]}.")
    wrong = [int(episode) for episode in np.unique(ids) if assign_split(int(episode)) != split]
    if wrong:
        raise RuntimeError(f"{split} data contains other splits: {wrong[:5]}.")


def joint_angles(state: np.ndarray) -> np.ndarray:
    pose = np.asarray(state, dtype=np.float64).reshape(-1)[:16].reshape(8, 2)
    return np.arctan2(pose[:, 0], pose[:, 1])


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def normalize_unit_pairs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    if values.size % 2 != 0:
        raise ValueError("Unit-pair values must have even length.")
    pairs = values.reshape(-1, 2)
    norm = np.linalg.norm(pairs, axis=1, keepdims=True)
    pairs /= np.clip(norm, 1e-6, None)
    return pairs.reshape(values.shape)


def pack_state(prediction: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Unstandardize a 35-d head output into a 44-d simulator observation."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    if prediction.shape != (35,) or mean.shape != (35,) or std.shape != (35,):
        raise ValueError("Expected 35-d prediction statistics.")
    raw = prediction * std + mean
    state = np.zeros(audit.STATE_DIM, dtype=np.float64)
    state[:16] = normalize_unit_pairs(raw[:16])
    state[16:24] = raw[16:24]
    state[33:35] = raw[24:26]
    object_angle = float(np.arctan2(raw[26], raw[27]))
    state[35] = np.cos(0.5 * object_angle)
    state[36] = np.sin(0.5 * object_angle)
    state[37:40] = raw[28:31]
    state[40:42] = raw[31:33]
    target_angle = float(np.arctan2(raw[33], raw[34]))
    state[42] = np.cos(0.5 * target_angle)
    state[43] = np.sin(0.5 * target_angle)
    if not np.all(np.isfinite(state)):
        raise ValueError("Packed state is non-finite.")
    return state


def group_loss(prediction, target):
    """Mean of per-group MSEs, so two position channels are not lost in sixteen joints."""
    import torch

    losses = []
    for _name, start, stop in TARGET_GROUPS:
        losses.append(torch.mean((prediction[:, start:stop] - target[:, start:stop]) ** 2))
    return torch.stack(losses).mean()


def group_slices() -> list[tuple[str, int, int]]:
    cursor = 0
    slices = []
    for name, start, stop in STATE_GROUPS:
        width = stop - start
        slices.append((name, cursor, cursor + width))
        cursor += width
    return slices


def planar_angles(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return 2.0 * np.arctan2(quat[..., 1], quat[..., 0])


def regression_targets(states: np.ndarray) -> np.ndarray:
    """Pose targets. Orientations are sin/cos of the planar angle.

    The stored object quaternion and its negation are the same rotation, and
    about 10% of nearby frames flip that sign. Regressing the quaternion
    components directly makes those frames contradict each other.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != audit.STATE_DIM:
        raise ValueError("Expected (N, 44) states.")
    object_angle = planar_angles(states[:, 35:37])
    target_angle = planar_angles(states[:, 42:44])
    return np.concatenate(
        [
            states[:, 0:16],
            states[:, 16:24],
            states[:, 33:35],
            np.sin(object_angle)[:, None],
            np.cos(object_angle)[:, None],
            states[:, 37:40],
            states[:, 40:42],
            np.sin(target_angle)[:, None],
            np.cos(target_angle)[:, None],
        ],
        axis=1,
    ).astype(np.float64)


def fit_target_standardizer(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 35:
        raise ValueError("Expected (N, 35) regression targets.")
    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


TARGET_GROUPS = (
    ("arm_sincos", 0, 16),
    ("arm_vel", 16, 24),
    ("obj_xz", 24, 26),
    ("obj_angle", 26, 28),
    ("obj_vel", 28, 31),
    ("target_xz", 31, 33),
    ("target_angle", 33, 35),
)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Error values must be non-empty and finite.")
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def estimator_decision(
    hand_median_m: float,
    object_median_m: float,
    regret: dict | None,
) -> dict[str, str]:
    """Map validation accuracy and bank regret onto the three gates.

    A sub-2 cm median is required before estimated-state regret can reject or
    accept the mechanics baseline. Regret at most 0.10, with the upper end of
    the interval at most 0.15, closes the SMWM gap.
    """
    competent = float(hand_median_m) <= POSITION_LIMIT_M and float(object_median_m) <= POSITION_LIMIT_M
    if not competent:
        return {
            "decision": "INCONCLUSIVE",
            "reason": (
                f"Held-out median hand error is {hand_median_m:.4f} m and object "
                f"error is {object_median_m:.4f} m. The 2 cm checkpoint was not met, "
                "so estimated-state physics was not accepted as a test of the simulator."
            ),
        }
    if regret is None:
        return {
            "decision": "INCONCLUSIVE",
            "reason": "The estimator met 2 cm, but the action-bank ranking was not run.",
        }
    mean = float(regret["mean"])
    high = float(regret["ci_high"])
    low = float(regret["ci_low"])
    if mean <= 0.10 and high <= 0.15:
        return {
            "decision": "NO-GO",
            "reason": (
                f"Estimated-state physics regret is {mean:.3f} [{low:.3f}, {high:.3f}]. "
                "It closes the SMWM ranking gap. No new learning objective is indicated."
            ),
        }
    if mean >= 0.20 and low > 0.10:
        return {
            "decision": "DIAGNOSIS",
            "reason": (
                f"The estimator meets the 2 cm checkpoint, and regret remains {mean:.3f} "
                f"[{low:.3f}, {high:.3f}]. The residual is not an under-trained pose net."
            ),
        }
    return {
        "decision": "INCONCLUSIVE",
        "reason": (
            f"Regret is {mean:.3f} [{low:.3f}, {high:.3f}]. The point estimate or the "
            "interval does not meet the 0.10 close-the-gap rule, and it is not a "
            "clearly remaining failure above 0.20."
        ),
    }


def choose_frames(length: int, contact_index: int) -> list[int]:
    if length <= 0:
        raise ValueError("Episode length must be positive.")
    contact_index = int(np.clip(contact_index, 0, length - 1))
    if length <= FRAMES_PER_EPISODE:
        return list(range(length))
    uniform = np.linspace(0, length - 1, FRAMES_PER_EPISODE - 1, dtype=np.int64)
    frames = sorted({int(frame) for frame in uniform} | {contact_index})
    return frames[:FRAMES_PER_EPISODE]


def _read_frames(pixels, offset: int, length: int, frames: list[int]) -> np.ndarray:
    images = np.empty((len(frames), 224, 224, 3), dtype=np.uint8)
    by_chunk: dict[int, list[int]] = {}
    for index, frame in enumerate(frames):
        by_chunk.setdefault(int(frame) // 100, []).append(index)
    for chunk, indexes in by_chunk.items():
        start = offset + chunk * 100
        stop = min(offset + length, start + 100)
        block = np.asarray(pixels[start:stop])
        for index in indexes:
            local = frames[index] - chunk * 100
            images[index] = block[local]
    return images


def build_cache(data_root: Path, output: Path, split: str, starts_only: bool = False) -> dict:
    episodes = TRAIN_EPISODES if split == "train" else VAL_EPISODES
    path = data_root / audit.TRAIN_FILENAME
    records = {int(record["episode_id"]): record for record in audit.index_h5_episodes(path)}
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    ids: list[int] = []
    contact_flags: list[bool] = []
    handle = audit.open_h5(path)
    try:
        pixels = handle["pixels"]
        for episode_id in episodes:
            record = records[int(episode_id)]
            observation = np.asarray(
                handle["observation"][record["offset"] : record["offset"] + record["length"]],
                dtype=np.float32,
            )
            distance = np.linalg.norm(observation[:, 29:31] - observation[:, 33:35], axis=1)
            contact_index = int(np.argmin(distance))
            frames = [0] if starts_only else choose_frames(int(record["length"]), contact_index)
            batch = _read_frames(pixels, int(record["offset"]), int(record["length"]), frames)
            for local, frame in enumerate(frames):
                images.append(batch[local])
                states.append(observation[frame])
                ids.append(int(episode_id))
                contact_flags.append(bool(distance[frame] <= CONTACT_DISTANCE_M))
            if episode_id % 100 == 0:
                print(f"cached {split} episode {episode_id}", flush=True)
    finally:
        handle.close()
    image_array = np.stack(images).astype(np.uint8)
    state_array = np.stack(states).astype(np.float32)
    id_array = np.asarray(ids, dtype=np.int64)
    assert_fit_episodes(id_array, split)
    destination = output / split
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "images.npy", image_array)
    np.save(destination / "states.npy", state_array)
    np.save(destination / "episode_id.npy", id_array)
    np.save(destination / "contact.npy", np.asarray(contact_flags, dtype=bool))
    summary = {
        "split": split,
        "n_frames": int(len(id_array)),
        "n_episodes": int(len(np.unique(id_array))),
        "n_contact": int(np.sum(contact_flags)),
        "episode_min": int(id_array.min()),
        "episode_max": int(id_array.max()),
    }
    factor.write_json(destination / "summary.json", summary)
    return summary


def _load_encoder(device: str):
    import torch

    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
    if str(hub) not in sys.path:
        sys.path.insert(0, str(hub))
    from dinov2.hub.backbones import dinov2_vitb14

    encoder = dinov2_vitb14(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder.to(device)


def _head(width: int, out_dim: int):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, 512),
        nn.GELU(),
        nn.Linear(512, out_dim),
    )


def _features(encoder, images):
    features = encoder.forward_features(images)
    cls = features["x_norm_clstoken"]
    patches = features["x_norm_patchtokens"].mean(dim=1)
    return torch_cat(cls, patches)


def torch_cat(cls, patches):
    import torch

    return torch.cat((cls, patches), dim=1)


def _normalize_images(images, device: str):
    import torch

    copied = np.array(images, dtype=np.uint8, copy=True)
    tensor = torch.as_tensor(copied, device=device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.as_tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _predict_raw(encoder, head, images, device: str, batch_size: int = 64) -> np.ndarray:
    import torch

    encoder.eval()
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = _normalize_images(images[start : start + batch_size], device)
            outputs.append(head(_features(encoder, batch)).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _physical_errors(env, states: np.ndarray, predictions: np.ndarray, mean, std) -> dict[str, np.ndarray]:
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    hand, obj, hand_ang, obj_ang, joints, arm_vel, obj_vel, target = [], [], [], [], [], [], [], []
    valid = []
    for label, raw in zip(states, predictions):
        estimate = pack_state(raw, mean, std)
        env.reset(seed=0, options={"state": estimate})
        restored = env.get_state()
        errors = insert_peg_goal_errors(label, restored)
        hand.append(errors["hand_position"])
        obj.append(errors["object_position"])
        hand_ang.append(errors["hand_angle"])
        obj_ang.append(errors["object_angle"])
        joints.append(float(np.max(np.abs(wrap_angle(joint_angles(label) - joint_angles(restored))))))
        arm_vel.append(float(np.linalg.norm(label[16:24] - restored[16:24])))
        obj_vel.append(float(np.linalg.norm(label[37:40] - restored[37:40])))
        target.append(float(np.linalg.norm(label[40:42] - restored[40:42])))
        valid.append(bool(np.all(np.isfinite(restored)) and errors["hand_position"] < 0.05 and errors["object_position"] < 0.05))
    return {
        "hand_m": np.asarray(hand),
        "object_m": np.asarray(obj),
        "hand_rad": np.asarray(hand_ang),
        "object_rad": np.asarray(obj_ang),
        "joint_rad": np.asarray(joints),
        "arm_vel": np.asarray(arm_vel),
        "object_vel": np.asarray(obj_vel),
        "target_m": np.asarray(target),
        "valid": np.asarray(valid, dtype=bool),
    }


def _error_report(errors: dict[str, np.ndarray], contact: np.ndarray) -> dict:
    report = {
        name: percentile_summary(values)
        for name, values in errors.items()
        if name != "valid" and values.ndim == 1
    }
    report["physically_valid_rate"] = float(np.mean(errors["valid"]))
    if np.any(contact):
        report["contact_hand_m"] = percentile_summary(errors["hand_m"][contact])
        report["contact_object_m"] = percentile_summary(errors["object_m"][contact])
    report["meets_2cm"] = bool(
        report["hand_m"]["median"] <= POSITION_LIMIT_M and report["object_m"]["median"] <= POSITION_LIMIT_M
    )
    return report


def _set_trainable(encoder, head, blocks: int) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if blocks > 0:
        for block in encoder.blocks[-blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in encoder.norm.parameters():
            parameter.requires_grad_(True)
    for parameter in head.parameters():
        parameter.requires_grad_(True)


def _run_epochs(encoder, head, train_images, train_targets, val_images, val_states, mean, std, device, epochs, backbone_lr, env) -> tuple[dict, list[float]]:
    import torch

    groups = []
    encoder_params = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": backbone_lr})
    groups.append({"params": list(head.parameters()), "lr": 1e-3 if backbone_lr == 0.0 else 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
    losses = []
    best = None
    best_state = None
    generator = np.random.default_rng(0)
    order = np.arange(len(train_images))
    for epoch in range(epochs):
        generator.shuffle(order)
        encoder.train(any(parameter.requires_grad for parameter in encoder.parameters()))
        head.train()
        epoch_loss = []
        for start in range(0, len(order), 64):
            batch_index = order[start : start + 64]
            images = _normalize_images(train_images[batch_index], device)
            target = torch.as_tensor(train_targets[batch_index], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(_features(encoder, images))
            loss = group_loss(prediction, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        raw = _predict_raw(encoder, head, val_images, device)
        errors = _physical_errors(env, val_states, raw, mean, std)
        report = _error_report(errors, np.zeros(len(val_states), dtype=bool))
        score = 0.5 * (report["hand_m"]["median"] + report["object_m"]["median"])
        losses.append(float(np.mean(epoch_loss)))
        print(
            f"epoch {epoch} loss {losses[-1]:.5f} hand {report['hand_m']['median']:.4f} "
            f"object {report['object_m']['median']:.4f}",
            flush=True,
        )
        if best is None or score < best["score"]:
            best = {"score": score, "report": report, "epoch": epoch}
            best_state = {
                "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                "encoder": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
                if backbone_lr > 0 else None,
            }
        if report["hand_m"]["median"] <= 0.005 and report["object_m"]["median"] <= 0.005:
            break
    head.load_state_dict(best_state["head"])
    if best_state["encoder"] is not None:
        encoder.load_state_dict(best_state["encoder"])
    return best, losses


def train_estimator(cache: Path, output: Path, device: str) -> dict:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    started = time.perf_counter()
    train_images = np.load(cache / "train" / "images.npy", mmap_mode="r")
    train_states = np.load(cache / "train" / "states.npy")
    train_ids = np.load(cache / "train" / "episode_id.npy")
    val_images = np.load(cache / "val" / "images.npy", mmap_mode="r")
    val_states = np.load(cache / "val" / "states.npy")
    val_ids = np.load(cache / "val" / "episode_id.npy")
    assert_fit_episodes(train_ids, "train")
    assert_fit_episodes(val_ids, "val")
    raw_targets = regression_targets(train_states)
    mean, std = fit_target_standardizer(raw_targets)
    targets = ((raw_targets - mean) / std).astype(np.float32)
    encoder = _load_encoder(device)
    head = _head(1536, targets.shape[1]).to(device)
    from anygoal.envs.insert_peg.env import InsertPegEnv

    env = InsertPegEnv(terminate_at_goal=False)
    stages = []
    try:
        _set_trainable(encoder, head, blocks=0)
        frozen, frozen_losses = _run_epochs(
            encoder, head, train_images, targets, val_images, val_states, mean, std,
            device, epochs=2, backbone_lr=0.0, env=env,
        )
        stages.append({"name": "frozen", "losses": frozen_losses, **frozen["report"], "epoch": frozen["epoch"]})
        frozen_tight = (
            frozen["report"]["hand_m"]["median"] <= 0.005 and frozen["report"]["object_m"]["median"] <= 0.005
        )
        if not frozen_tight:
            _set_trainable(encoder, head, blocks=4)
            tuned, tuned_losses = _run_epochs(
                encoder, head, train_images, targets, val_images, val_states, mean, std,
                device, epochs=8, backbone_lr=1e-5, env=env,
            )
            stages.append({"name": "last4", "losses": tuned_losses, **tuned["report"], "epoch": tuned["epoch"]})
        raw = _predict_raw(encoder, head, val_images, device)
        errors = _physical_errors(env, val_states, raw, mean, std)
        contact = np.load(cache / "val" / "contact.npy")
        final = _error_report(errors, contact)
    finally:
        env.close()
    import torch

    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head": head.state_dict(), "encoder": encoder.state_dict(), "mean": mean, "std": std},
        output / "estimator.pt",
    )
    payload = {
        "seconds": time.perf_counter() - started,
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "stages": stages,
        "validation": final,
        "supervision": "simulator state labels from insert episodes 50–1299; not used by SMWM",
        "normalization": "train episodes only",
    }
    factor.write_json(output / "train.json", payload)
    return payload


def _load_trained(checkpoint: Path, device: str):
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = _load_encoder(device)
    encoder.load_state_dict(saved["encoder"])
    head = _head(1536, len(saved["mean"])).to(device)
    head.load_state_dict(saved["head"])
    encoder.eval()
    head.eval()
    return encoder, head, np.asarray(saved["mean"]), np.asarray(saved["std"])


def derived_observation(states: np.ndarray) -> np.ndarray:
    """Fill hand pose from the arm joints. ``set_state`` does not read hand coordinates."""
    from anygoal.envs.insert_peg.env import InsertPegEnv

    env = InsertPegEnv(terminate_at_goal=False)
    derived = np.empty_like(states, dtype=np.float64)
    try:
        for index, state in enumerate(states):
            env.reset(seed=0, options={"state": np.asarray(state, dtype=np.float64)})
            derived[index] = env.get_state()
    finally:
        env.close()
    return derived


def _rank_from_states(starts: np.ndarray, goals: np.ndarray, actions: np.ndarray, success: np.ndarray) -> dict:
    from anygoal.envs.insert_peg.env import InsertPegEnv, insert_peg_goal_errors

    env = InsertPegEnv(terminate_at_goal=False)
    selected = []
    try:
        for episode in range(len(starts)):
            costs = []
            for candidate in range(audit.N_CANDIDATES):
                env.reset(seed=0, options={"state": starts[episode], "goal_state": goals[episode]})
                for action in np.asarray(actions[episode, candidate], dtype=np.float64):
                    env.step(action)
                errors = insert_peg_goal_errors(goals[episode], env.get_state())
                costs.append(sum(errors[name] / audit.TOLERANCES[name] for name in audit.ERROR_NAMES))
            selected.append(int(np.argmin(costs)))
    finally:
        env.close()
    scores = factor.per_episode_scores(success, np.asarray(selected, dtype=np.int64))
    return {
        "selected": np.asarray(selected, dtype=np.int64),
        "success_rate": float(scores["chosen_success"].mean()),
        "regret": audit.bootstrap_mean(scores["regret"], seed=31),
    }


def continue_full(cache: Path, checkpoint: Path, output: Path, device: str, epochs: int = 4) -> dict:
    """One full-backbone continuation after the last-four-block stage plateaus."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    started = time.perf_counter()
    train_images = np.load(cache / "train" / "images.npy", mmap_mode="r")
    train_states = np.load(cache / "train" / "states.npy")
    val_images = np.load(cache / "val" / "images.npy", mmap_mode="r")
    val_states = np.load(cache / "val" / "states.npy")
    assert_fit_episodes(np.load(cache / "train" / "episode_id.npy"), "train")
    assert_fit_episodes(np.load(cache / "val" / "episode_id.npy"), "val")
    encoder, head, mean, std = _load_trained(checkpoint, device)
    raw_targets = regression_targets(train_states)
    targets = ((raw_targets - mean) / std).astype(np.float32)
    _set_trainable(encoder, head, blocks=12)
    from anygoal.envs.insert_peg.env import InsertPegEnv

    env = InsertPegEnv(terminate_at_goal=False)
    try:
        tuned, losses = _run_epochs(
            encoder, head, train_images, targets, val_images, val_states, mean, std,
            device, epochs=epochs, backbone_lr=5e-6, env=env,
        )
        raw = _predict_raw(encoder, head, val_images, device)
        final = _error_report(_physical_errors(env, val_states, raw, mean, std), np.load(cache / "val" / "contact.npy"))
    finally:
        env.close()
    import torch

    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head": head.state_dict(), "encoder": encoder.state_dict(), "mean": mean, "std": std},
        output / "estimator_full.pt",
    )
    payload = {
        "seconds": time.perf_counter() - started,
        "stage": {"name": "all_blocks", "losses": losses, **tuned["report"], "epoch": tuned["epoch"]},
        "validation": final,
    }
    factor.write_json(output / "train_full.json", payload)
    return payload


def rank_anchor(checkpoint: Path, anchor: Path, device: str, output: Path, validation: Path) -> dict:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    bank = np.load(anchor / "bank.npz")
    rollout = np.load(anchor / "bank" / "evaluator_10.npz")
    score = np.load(anchor / "scores" / "evaluator_10.npz")
    ids = np.asarray(bank["episode_id"])
    if np.any(np.asarray([assign_split(int(episode)) for episode in ids]) != "eval"):
        raise RuntimeError("The anchor is not the reserved evaluation episodes.")
    encoder, head, mean, std = _load_trained(checkpoint, device)
    start_raw = _predict_raw(encoder, head, bank["start_image"], device)
    goal_raw = _predict_raw(encoder, head, bank["goal_image"], device)
    start_hat = np.stack([pack_state(row, mean, std) for row in start_raw])
    goal_hat = derived_observation(np.stack([pack_state(row, mean, std) for row in goal_raw]))
    true_start = np.asarray(bank["start_state"], dtype=np.float64)
    true_goal = np.asarray(bank["goal_state"], dtype=np.float64)
    actions = np.asarray(bank["actions"])
    success = np.asarray(rollout["success"])
    estimated = _rank_from_states(start_hat, goal_hat, actions, success)
    start_only = _rank_from_states(start_hat, true_goal, actions, success)
    true_pose_zero_vel = true_start.copy()
    true_pose_zero_vel[:, 37:40] = 0.0
    zero_velocity = _rank_from_states(true_pose_zero_vel, true_goal, actions, success)
    smwm_index = np.argmin(score["predicted_cost"], axis=1).astype(np.int64)
    smwm_scores = factor.per_episode_scores(success, smwm_index)
    smwm = {
        "success_rate": float(smwm_scores["chosen_success"].mean()),
        "regret": audit.bootstrap_mean(smwm_scores["regret"], seed=32),
    }
    oracle = _rank_from_states(true_start, true_goal, actions, success)
    # Controlled substitutions: keep the estimated state and restore one true group.
    substitutions = {}
    groups = {
        "arm": (slice(0, 24),),
        "object_pose": (slice(33, 37),),
        "object_velocity": (slice(37, 40),),
        "target": (slice(40, 44),),
    }
    for name, slices in groups.items():
        mixed = start_hat.copy()
        mixed_goal = goal_hat.copy()
        for piece in slices:
            mixed[:, piece] = true_start[:, piece]
            mixed_goal[:, piece] = true_goal[:, piece]
        mixed_goal = derived_observation(mixed_goal)
        substitutions[name] = {
            key: value for key, value in _rank_from_states(mixed, mixed_goal, actions, success).items() if key != "selected"
        }
    from anygoal.envs.insert_peg.env import InsertPegEnv

    env = InsertPegEnv(terminate_at_goal=False)
    try:
        anchor_errors = _physical_errors(env, true_start, start_raw, mean, std)
    finally:
        env.close()
    payload = {
        "anchor_start_errors": _error_report(anchor_errors, np.zeros(len(true_start), dtype=bool)),
        "estimated_start_and_goal": {key: value for key, value in estimated.items() if key != "selected"},
        "estimated_start_true_goal": {key: value for key, value in start_only.items() if key != "selected"},
        "true_state_zero_object_velocity": {key: value for key, value in zero_velocity.items() if key != "selected"},
        "oracle_true_state": {key: value for key, value in oracle.items() if key != "selected"},
        "smwm_predicted": smwm,
        "substitutions": substitutions,
        "information": {
            "estimator": "DINOv2-B supervised with simulator state labels",
            "smwm": "frozen visual checkpoint, no state labels",
            "oracle": "true simulator state",
        },
    }
    heldout = json.loads(validation.read_text())["validation"]
    payload["decision"] = estimator_decision(
        heldout["hand_m"]["median"],
        heldout["object_m"]["median"],
        payload["estimated_start_and_goal"]["regret"],
    )
    payload["competence_source"] = "validation episodes 1300–1460, not the 50 evaluation episodes"
    output.mkdir(parents=True, exist_ok=True)
    factor.write_json(output / "ranking.json", payload)
    np.savez_compressed(output / "anchor_estimates.npz", start=start_hat, goal=goal_hat, selected=estimated["selected"])
    return payload


def sensitivity(anchor: Path, output: Path) -> dict:
    """How much true-state noise the same bank ranking can tolerate."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    bank = np.load(anchor / "bank.npz")
    success = np.load(anchor / "bank" / "evaluator_10.npz")["success"]
    base = np.asarray(bank["start_state"], dtype=np.float64)
    goals = np.asarray(bank["goal_state"], dtype=np.float64)
    actions = np.asarray(bank["actions"])
    rng = np.random.default_rng(0)
    rows = []
    for sigma in (0.0, 0.005, 0.01, 0.02, 0.05):
        starts = base.copy()
        if sigma > 0:
            starts[:, 33:35] += rng.normal(0.0, sigma, size=(len(starts), 2))
        result = _rank_from_states(starts, goals, actions, success)
        rows.append({
            "perturbation": f"object position sigma {sigma:.3f} m",
            "success_rate": result["success_rate"],
            "regret": result["regret"],
        })
        print(rows[-1]["perturbation"], result["success_rate"], result["regret"]["mean"], flush=True)
    factor.write_json(output / "sensitivity.json", {"rows": rows})
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache")
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--split", choices=("train", "val"), required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    rank = subparsers.add_parser("rank")
    rank.add_argument("--checkpoint", type=Path, required=True)
    rank.add_argument("--anchor", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--device", default="cuda")
    rank.add_argument("--validation", type=Path, required=True)
    full = subparsers.add_parser("continue-full")
    full.add_argument("--cache", type=Path, required=True)
    full.add_argument("--checkpoint", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--device", default="cuda")
    full.add_argument("--epochs", type=int, default=4)
    cache.add_argument("--starts-only", action="store_true")
    sense = subparsers.add_parser("sensitivity")
    sense.add_argument("--anchor", type=Path, required=True)
    sense.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        print(json.dumps(build_cache(args.data_root, args.output, args.split, starts_only=args.starts_only)))
    elif args.command == "train":
        print(json.dumps(train_estimator(args.cache, args.output, args.device), default=audit._json_default))
    elif args.command == "continue-full":
        print(json.dumps(
            continue_full(args.cache, args.checkpoint, args.output, args.device, epochs=args.epochs),
            default=audit._json_default,
        ))
    elif args.command == "rank":
        print(json.dumps(
            rank_anchor(args.checkpoint, args.anchor, args.device, args.output, args.validation),
            default=audit._json_default,
        ))
    elif args.command == "sensitivity":
        print(json.dumps(sensitivity(args.anchor, args.output), default=audit._json_default))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
