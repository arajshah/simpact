"""Standard baselines for the supported Insert-Peg action bank.

The Extract-Peg goal is not solvable under the training integrator. These
baselines are therefore scored on the insert anchor, where that integrator
generated the goals. Episode ids 0-49 are the anchor and are never training
rows. Numerical helpers do not import torch or MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit
import anygoal_dynamics_factor as factor

ANCHOR_EPISODES = 50
TRAIN_EPISODE_START = 50
TRAIN_EPISODE_STOP = 130
VALIDATION_EPISODE_START = 130
VALIDATION_EPISODE_STOP = 140
PAIRS_PER_EPISODE = 16
FRAME_STRIDE = 10
POSE_ERROR_LIMIT_M = 0.02
TRAIN_STEPS = 2000
BATCH_SIZE = 64


def closes_eighty_percent(baseline_regret: float, reference_regret: float) -> bool:
    """True when the baseline removes at least 80% of the reference regret gap."""
    baseline_regret = float(baseline_regret)
    reference_regret = float(reference_regret)
    if not np.isfinite(baseline_regret) or not np.isfinite(reference_regret):
        raise ValueError("Regret values must be finite.")
    if reference_regret < 0.0 or baseline_regret < -1e-9:
        raise ValueError("Regret cannot be negative.")
    if reference_regret == 0.0:
        return bool(baseline_regret <= 1e-9)
    return bool(baseline_regret <= 0.20 * reference_regret)


def nearest_neighbor_indices(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Index of the nearest pool row for each query. Rows are flat features."""
    query = np.asarray(query, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    if query.ndim != 2 or pool.ndim != 2 or query.shape[1] != pool.shape[1]:
        raise ValueError("Query and pool must be 2-d with the same width.")
    if len(pool) == 0:
        raise ValueError("The retrieval pool is empty.")
    # Squared distance without forming the full cross matrix twice.
    query_norm = np.sum(query * query, axis=1, keepdims=True)
    pool_norm = np.sum(pool * pool, axis=1)
    dots = query @ pool.T
    distance = query_norm + pool_norm - 2.0 * dots
    return np.argmin(distance, axis=1).astype(np.int64)


def rank_by_action_distance(bank: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Pick the candidate whose raw action sequence is closest to a prototype."""
    bank = np.asarray(bank, dtype=np.float64)
    prototype = np.asarray(prototype, dtype=np.float64)
    if bank.shape[1:] != prototype.shape:
        raise ValueError("Prototype action shape does not match the bank.")
    delta = bank - prototype
    distance = np.sum(delta * delta, axis=(1, 2))
    return distance


def downsample(images: np.ndarray, size: int = 32) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    if images.shape[-1] != 3:
        raise ValueError("Expected HWC uint8 images.")
    step_y = images.shape[1] // size
    step_x = images.shape[2] // size
    if step_y < 1 or step_x < 1:
        raise ValueError("Images are smaller than the downsample grid.")
    pooled = images[:, : size * step_y, : size * step_x].reshape(
        images.shape[0], size, step_y, size, step_x, 3
    ).mean(axis=(2, 4))
    flat = pooled.reshape(images.shape[0], -1).astype(np.float32)
    flat /= 255.0
    return flat


def _episode_windows(records: list[dict]) -> list[tuple[int, int, int]]:
    windows = []
    for record in records:
        length = int(record["length"])
        offset = int(record["offset"])
        last = min(length - audit.HORIZON_STEPS, PAIRS_PER_EPISODE * FRAME_STRIDE)
        starts = list(range(0, last, FRAME_STRIDE))[:PAIRS_PER_EPISODE]
        for start in starts:
            windows.append((int(record["episode_id"]), offset, start))
    if not windows:
        raise RuntimeError("No training windows were available.")
    return windows


def _load_windows(data_root: Path, episode_start: int, episode_stop: int) -> dict[str, np.ndarray]:
    path = data_root / audit.TRAIN_FILENAME
    records = [
        record for record in audit.index_h5_episodes(path)
        if episode_start <= int(record["episode_id"]) < episode_stop
    ]
    if not records:
        raise RuntimeError(f"No episodes in [{episode_start}, {episode_stop}).")
    windows = _episode_windows(records)
    starts = np.empty((len(windows), 224, 224, 3), dtype=np.uint8)
    goals = np.empty_like(starts)
    actions = np.empty((len(windows), audit.HORIZON_STEPS, audit.ACTION_DIM), dtype=np.float32)
    states = np.empty((len(windows), audit.STATE_DIM), dtype=np.float32)
    goal_states = np.empty_like(states)
    ids = np.empty(len(windows), dtype=np.int64)
    handle = audit.open_h5(path)
    try:
        for index, (episode_id, offset, start) in enumerate(windows):
            rows = audit.read_rows(
                handle,
                offset,
                start,
                start + audit.HORIZON_STEPS + 1,
                ("pixels", "action", "observation"),
            )
            starts[index] = audit.as_hwc_uint8(rows["pixels"][0])
            goals[index] = audit.as_hwc_uint8(rows["pixels"][audit.HORIZON_STEPS])
            actions[index] = np.asarray(rows["action"][: audit.HORIZON_STEPS], dtype=np.float32)
            states[index] = np.asarray(rows["observation"][0], dtype=np.float32)
            goal_states[index] = np.asarray(rows["observation"][audit.HORIZON_STEPS], dtype=np.float32)
            ids[index] = episode_id
    finally:
        handle.close()
    return {
        "start_image": starts,
        "goal_image": goals,
        "action": actions,
        "state": states,
        "goal_state": goal_states,
        "episode_id": ids,
    }


def _anchor_arrays(anchor: Path) -> dict[str, np.ndarray]:
    bank = np.load(anchor / "bank.npz")
    rollout = np.load(anchor / "bank" / "evaluator_10.npz")
    score = np.load(anchor / "scores" / "evaluator_10.npz")
    return {
        "actions": np.asarray(bank["actions"]),
        "start_image": np.asarray(bank["start_image"]),
        "goal_image": np.asarray(bank["goal_image"]),
        "start_state": np.asarray(bank["start_state"]),
        "goal_state": np.asarray(bank["goal_state"]),
        "success": np.asarray(rollout["success"]),
        "errors": np.asarray(rollout["errors"]),
        "predicted_cost": np.asarray(score["predicted_cost"]),
    }


def _selection_summary(success: np.ndarray, selected: np.ndarray, *, seed: int) -> dict:
    scores = factor.per_episode_scores(success, selected)
    solvable = scores["solvable"]
    conditional = factor.conditional_bootstrap(scores["regret"], solvable, seed=seed)
    return {
        "success_rate": float(scores["chosen_success"].mean()),
        "regret_unconditional": audit.bootstrap_mean(scores["regret"], seed=seed),
        "regret_given_solvable": conditional,
        "solvability": float(solvable.mean()),
    }


def _cached_windows(data_root: Path, output: Path, start: int, stop: int, name: str) -> dict[str, np.ndarray]:
    path = output / f"{name}.npz"
    if path.is_file():
        loaded = np.load(path)
        return {key: loaded[key] for key in loaded.files}
    windows = _load_windows(data_root, start, stop)
    factor._atomic_npz(path, **windows)
    return windows


def run_retrieval(data_root: Path, anchor: Path, output: Path) -> dict:
    started = time.perf_counter()
    pool = _cached_windows(data_root, output, TRAIN_EPISODE_START, TRAIN_EPISODE_STOP, "train_windows")
    anchor_rows = _anchor_arrays(anchor)
    if np.intersect1d(pool["episode_id"], np.arange(ANCHOR_EPISODES)).size:
        raise RuntimeError("Retrieval pool overlaps the anchor episodes.")
    query = np.concatenate(
        [downsample(anchor_rows["start_image"]), downsample(anchor_rows["goal_image"])], axis=1
    )
    library = np.concatenate(
        [downsample(pool["start_image"]), downsample(pool["goal_image"])], axis=1
    )
    neighbor = nearest_neighbor_indices(query, library)
    selected = []
    for episode, index in enumerate(neighbor):
        distance = rank_by_action_distance(anchor_rows["actions"][episode], pool["action"][index])
        selected.append(int(np.argmin(distance)))
    selected_arr = np.asarray(selected)
    summary = _selection_summary(anchor_rows["success"], selected_arr, seed=11)
    state_query = np.concatenate([anchor_rows["start_state"], anchor_rows["goal_state"]], axis=1)
    state_pool = np.concatenate([pool["state"], pool["goal_state"]], axis=1)
    state_neighbor = nearest_neighbor_indices(state_query, state_pool)
    state_selected = []
    for episode, index in enumerate(state_neighbor):
        distance = rank_by_action_distance(anchor_rows["actions"][episode], pool["action"][index])
        state_selected.append(int(np.argmin(distance)))
    state_summary = _selection_summary(anchor_rows["success"], np.asarray(state_selected), seed=12)
    reference = _selection_summary(
        anchor_rows["success"], np.argmin(anchor_rows["predicted_cost"], axis=1), seed=13
    )
    payload = {
        "method": "nearest neighbor goal-conditioned retrieval",
        "pool_episodes": [TRAIN_EPISODE_START, TRAIN_EPISODE_STOP],
        "n_pool": int(len(pool["action"])),
        "anchor_episodes_excluded": ANCHOR_EPISODES,
        "image_retrieval": summary,
        "privileged_state_retrieval": {
            **state_summary,
            "information": "true start and goal state; not available to SMWM",
        },
        "smwm_on_same_bank": reference,
        "image_closes_eighty_percent_of_smwm_regret": closes_eighty_percent(
            summary["regret_unconditional"]["mean"],
            reference["regret_unconditional"]["mean"],
        ),
        "seconds": time.perf_counter() - started,
    }
    factor.write_json(output / "retrieval.json", payload)
    return payload


def _torch_models():
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 6, stride=4),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 5 * 5, 256),
                nn.ReLU(),
            )

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.net(pixels.float().div(255.0))

    class PoseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.head = nn.Linear(256, audit.STATE_DIM)

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(pixels))

    class GoalPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, audit.HORIZON_STEPS * audit.ACTION_DIM),
            )

        def forward(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            features = torch.cat([self.encoder(start), self.encoder(goal)], dim=1)
            return self.head(features).reshape(-1, audit.HORIZON_STEPS, audit.ACTION_DIM)

    return torch, PoseNet, GoalPolicy


def _train_loop(model, batches, steps: int, lr: float, device: str) -> list[float]:
    import torch

    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    for step in range(steps):
        choice = torch.randint(0, len(batches), (1,), generator=generator).item()
        batch = batches[int(choice)]
        optimizer.zero_grad(set_to_none=True)
        loss = batch(model)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 200 == 0:
            print(f"step {step} loss {losses[-1]:.5f}", flush=True)
    model.eval()
    return losses


def _make_pose_batches(images, states, mean, std, device: str):
    import torch

    images_t = torch.as_tensor(np.transpose(images, (0, 3, 1, 2)), device=device)
    target = torch.as_tensor((states - mean) / std, dtype=torch.float32, device=device)
    batches = []
    for start in range(0, len(images), BATCH_SIZE):
        batch_images = images_t[start : start + BATCH_SIZE]
        batch_target = target[start : start + BATCH_SIZE]

        def run(model, batch_images=batch_images, batch_target=batch_target):
            return torch.mean((model(batch_images) - batch_target) ** 2)

        batches.append(run)
    if not batches:
        raise RuntimeError("Pose training produced no batches.")
    return batches


def _make_policy_batches(starts, goals, actions, device: str):
    import torch

    start_t = torch.as_tensor(np.transpose(starts, (0, 3, 1, 2)), device=device)
    goal_t = torch.as_tensor(np.transpose(goals, (0, 3, 1, 2)), device=device)
    action_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    batches = []
    for start in range(0, len(starts), BATCH_SIZE):
        batch_start = start_t[start : start + BATCH_SIZE]
        batch_goal = goal_t[start : start + BATCH_SIZE]
        batch_action = action_t[start : start + BATCH_SIZE]

        def run(model, batch_start=batch_start, batch_goal=batch_goal, batch_action=batch_action):
            return torch.mean((model(batch_start, batch_goal) - batch_action) ** 2)

        batches.append(run)
    if not batches:
        raise RuntimeError("Policy training produced no batches.")
    return batches


def _predict_pose(model, images, mean, std, device: str) -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        tensor = torch.as_tensor(np.transpose(images, (0, 3, 1, 2)), device=device)
        for start in range(0, len(images), BATCH_SIZE):
            pred = model(tensor[start : start + BATCH_SIZE]).detach().cpu().numpy()
            outputs.append(pred * std + mean)
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _predict_actions(model, starts, goals, device: str) -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        start_t = torch.as_tensor(np.transpose(starts, (0, 3, 1, 2)), device=device)
        goal_t = torch.as_tensor(np.transpose(goals, (0, 3, 1, 2)), device=device)
        for start in range(0, len(starts), BATCH_SIZE):
            pred = model(start_t[start : start + BATCH_SIZE], goal_t[start : start + BATCH_SIZE])
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _physical_pose_error(env, predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    hand, obj = [], []
    for estimate, target in zip(predicted, truth):
        estimate = np.asarray(estimate, dtype=np.float64).copy()
        estimate[24:29] = 0.0
        if not np.all(np.isfinite(estimate)):
            raise RuntimeError("Pose prediction is non-finite.")
        env.reset(seed=0, options={"state": estimate})
        errors = insert_peg_goal_errors(target, env.get_state())
        hand.append(errors["hand_position"])
        obj.append(errors["object_position"])
    return {
        "n": int(len(predicted)),
        "hand_median_m": float(np.median(hand)),
        "object_median_m": float(np.median(obj)),
        "competent": bool(
            np.median(hand) <= POSE_ERROR_LIMIT_M and np.median(obj) <= POSE_ERROR_LIMIT_M
        ),
        "limit_m": POSE_ERROR_LIMIT_M,
    }


def run_learned(data_root: Path, anchor: Path, output: Path, device: str) -> dict:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    started = time.perf_counter()
    train = _cached_windows(data_root, output, TRAIN_EPISODE_START, TRAIN_EPISODE_STOP, "train_windows")
    validation = _cached_windows(
        data_root, output, VALIDATION_EPISODE_START, VALIDATION_EPISODE_STOP, "validation_windows"
    )
    anchor_rows = _anchor_arrays(anchor)
    forbidden = set(range(ANCHOR_EPISODES)) | set(range(VALIDATION_EPISODE_START, VALIDATION_EPISODE_STOP))
    if forbidden.intersection(set(np.asarray(train["episode_id"]).tolist())):
        raise RuntimeError("Training windows overlap the anchor or the pose validation episodes.")
    torch, PoseNet, GoalPolicy = _torch_models()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    mean = train["state"].mean(axis=0)
    std = train["state"].std(axis=0)
    std[std < 1e-6] = 1.0
    pose = PoseNet()
    pose_losses = _train_loop(
        pose, _make_pose_batches(train["start_image"], train["state"], mean, std, device),
        TRAIN_STEPS, 1e-3, device,
    )
    policy = GoalPolicy()
    policy_losses = _train_loop(
        policy, _make_policy_batches(train["start_image"], train["goal_image"], train["action"], device),
        TRAIN_STEPS, 1e-3, device,
    )
    from anygoal.envs.insert_peg.env import InsertPegEnv

    env = InsertPegEnv(terminate_at_goal=False)
    try:
        validation_prediction = _predict_pose(pose, validation["start_image"], mean, std, device)
        validation_error = _physical_pose_error(env, validation_prediction, validation["state"])
        anchor_prediction = _predict_pose(pose, anchor_rows["start_image"], mean, std, device)
        anchor_goal_prediction = _predict_pose(pose, anchor_rows["goal_image"], mean, std, device)
        anchor_error = _physical_pose_error(env, anchor_prediction, anchor_rows["start_state"])
        simulated = _simulate_bank(env, anchor_prediction, anchor_rows["actions"], anchor_rows["goal_state"])
        simulated_estimated_goal = _simulate_bank(
            env, anchor_prediction, anchor_rows["actions"], anchor_goal_prediction
        )
    finally:
        env.close()
    action_hat = _predict_actions(policy, anchor_rows["start_image"], anchor_rows["goal_image"], device)
    policy_selected = np.asarray([
        int(np.argmin(rank_by_action_distance(anchor_rows["actions"][episode], action_hat[episode])))
        for episode in range(len(action_hat))
    ])
    reference = _selection_summary(
        anchor_rows["success"], np.argmin(anchor_rows["predicted_cost"], axis=1), seed=14
    )
    physics_summary = _selection_summary(anchor_rows["success"], simulated["selected"], seed=15)
    estimated_goal_summary = _selection_summary(
        anchor_rows["success"], simulated_estimated_goal["selected"], seed=16
    )
    policy_summary = _selection_summary(anchor_rows["success"], policy_selected, seed=17)
    payload = {
        "budget": {
            "steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "train_episodes": [TRAIN_EPISODE_START, TRAIN_EPISODE_STOP],
            "validation_episodes": [VALIDATION_EPISODE_START, VALIDATION_EPISODE_STOP],
            "pairs_per_episode": PAIRS_PER_EPISODE,
            "frame_stride": FRAME_STRIDE,
            "n_train": int(len(train["action"])),
            "anchor_excluded": True,
        },
        "pose_validation": validation_error,
        "pose_on_anchor_starts": anchor_error,
        "privileged_goal_physics": {
            **physics_summary,
            "information": "image-estimated start state, true goal state, Insert-Peg physics",
        },
        "estimated_goal_physics": {
            **estimated_goal_summary,
            "information": "image-estimated start and goal state, Insert-Peg physics",
        },
        "goal_conditioned_regression": {
            **policy_summary,
            "information": "start image, goal image, and training actions only",
            "final_loss": policy_losses[-1],
        },
        "smwm_on_same_bank": reference,
        "pose_final_loss": pose_losses[-1],
        "policy_closes_eighty_percent": closes_eighty_percent(
            policy_summary["regret_unconditional"]["mean"],
            reference["regret_unconditional"]["mean"],
        ),
        "estimated_goal_physics_closes_eighty_percent": closes_eighty_percent(
            estimated_goal_summary["regret_unconditional"]["mean"],
            reference["regret_unconditional"]["mean"],
        ),
        "seconds": time.perf_counter() - started,
    }
    factor.write_json(output / "learned_baselines.json", payload)
    return payload


def _simulate_bank(env, starts: np.ndarray, actions: np.ndarray, goals: np.ndarray) -> dict:
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    selected = []
    for episode in range(len(starts)):
        costs = []
        for candidate in range(audit.N_CANDIDATES):
            state = np.asarray(starts[episode], dtype=np.float64).copy()
            state[24:29] = 0.0
            env.reset(seed=0, options={"state": state, "goal_state": goals[episode]})
            for action in np.asarray(actions[episode, candidate], dtype=np.float64):
                env.step(action)
            errors = insert_peg_goal_errors(goals[episode], env.get_state())
            costs.append(sum(errors[name] / audit.TOLERANCES[name] for name in audit.ERROR_NAMES))
        selected.append(int(np.argmin(costs)))
    return {"selected": np.asarray(selected)}


def run_short_horizon(anchor: Path, checkpoint: Path, anygoal_root: Path, output: Path) -> dict:
    """One-block prediction on the insert anchor, matched to the image after five actions.

    The goal for this probe is the realized five-step outcome of the recorded
    action, so the horizon of the goal matches one model block. It is not the
    25-step dataset goal.
    """
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from anygoal.envs.insert_peg.env import InsertPegEnv, insert_peg_goal_errors, insert_peg_goal_reached

    bank = np.load(anchor / "bank.npz")
    env = InsertPegEnv(render_mode="rgb_array", width=224, height=224, terminate_at_goal=False)
    model, transform = factor._load_model(checkpoint, anygoal_root)
    n_keep = int(bank["actions"].shape[0])
    success = np.zeros((n_keep, audit.N_CANDIDATES), dtype=bool)
    pred_error = np.empty((n_keep, audit.N_CANDIDATES), dtype=np.float64)
    p_cost = np.empty_like(pred_error)
    o_cost = np.empty_like(pred_error)
    try:
        for episode in range(n_keep):
            images = []
            states = []
            start = np.asarray(bank["start_state"][episode], dtype=np.float64)
            for candidate in range(audit.N_CANDIDATES):
                env.reset(seed=0, options={"state": start})
                for action in np.asarray(bank["actions"][episode, candidate, : audit.BLOCK_STEPS], dtype=np.float64):
                    env.step(action)
                states.append(env.get_state().copy())
                images.append(audit.as_hwc_uint8(env.render()))
            goal_state = states[0]
            for candidate, state in enumerate(states):
                errors = insert_peg_goal_errors(goal_state, state)
                success[episode, candidate] = insert_peg_goal_reached(errors, audit.TOLERANCES)
            blocked = audit.block_normalized_actions(
                np.asarray(bank["actions"][episode], dtype=np.float64),
                np.asarray(bank["action_mean"]),
                np.asarray(bank["action_scale"]),
            )
            import torch

            current = audit._policy_image(bank["start_image"][episode], transform)
            current = current.to(next(model.parameters()).device)
            actions = audit.float_actions(blocked, current).unsqueeze(0)
            info = {
                "pixels": current[:, None, None].expand(1, audit.N_CANDIDATES, 1, *current.shape[1:]).contiguous(),
            }
            rolled = model.rollout(info, actions)
            predicted = rolled["predicted_emb"]
            if predicted.shape[2] < 2:
                raise RuntimeError(f"Rollout has no one-block future: {tuple(predicted.shape)}")
            one_block = predicted[0, :, 1, :]
            observed = model.encode({
                "pixels": torch.cat([audit._policy_image(frame, transform) for frame in images], dim=0).to(current.device)
            })["emb"][:, -1, :]
            goal = observed[:1]
            one_block_np = one_block.detach().cpu().numpy()
            observed_np = observed.detach().cpu().numpy()
            goal_np = goal.detach().cpu().numpy()
            pred_error[episode] = factor.latent_sse(one_block_np, observed_np)
            p_cost[episode] = factor.latent_sse(one_block_np, goal_np)
            o_cost[episode] = factor.latent_sse(observed_np, goal_np)
            print(f"horizon episode {int(bank['episode_id'][episode])}", flush=True)
    finally:
        env.close()
    p_selected = np.argmin(p_cost, axis=1)
    o_selected = np.argmin(o_cost, axis=1)
    payload = {
        "definition": (
            "Insert anchor, training integrator, five raw actions. The goal is the "
            "recorded action's own five-step image and pose. Prediction is the first "
            "model block, not the 25-step rollout."
        ),
        "solvability": float(success.any(axis=1).mean()),
        "recorded_is_success": float(success[:, 0].mean()),
        "prediction_sse": float(pred_error.mean()),
        "predicted": _selection_summary(success, p_selected, seed=21),
        "observed": _selection_summary(success, o_selected, seed=22),
    }
    factor.write_json(output / "short_horizon.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieval = subparsers.add_parser("retrieval")
    retrieval.add_argument("--data-root", type=Path, required=True)
    retrieval.add_argument("--anchor", type=Path, required=True)
    retrieval.add_argument("--output", type=Path, required=True)
    learned = subparsers.add_parser("learned")
    learned.add_argument("--data-root", type=Path, required=True)
    learned.add_argument("--anchor", type=Path, required=True)
    learned.add_argument("--output", type=Path, required=True)
    learned.add_argument("--device", default="cuda")
    horizon = subparsers.add_parser("horizon")
    horizon.add_argument("--anchor", type=Path, required=True)
    horizon.add_argument("--checkpoint", type=Path, required=True)
    horizon.add_argument("--anygoal-root", type=Path, required=True)
    horizon.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "retrieval":
        print(json.dumps(run_retrieval(args.data_root, args.anchor, args.output), default=audit._json_default))
    elif args.command == "learned":
        print(json.dumps(run_learned(args.data_root, args.anchor, args.output, args.device), default=audit._json_default))
    elif args.command == "horizon":
        print(json.dumps(
            run_short_horizon(args.anchor, args.checkpoint, args.anygoal_root, args.output),
            default=audit._json_default,
        ))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
