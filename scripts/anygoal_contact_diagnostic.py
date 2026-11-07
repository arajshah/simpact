"""AnyGoal Section 8 contact-action diagnostic.

Stage A reproduces the released Sensorimotor World Model planner and audits
its evaluation split. Stage B scores one fixed 16-sequence action bank with
the predicted-latent cost, the same cost on observed futures, and the
evaluator's physical success criterion.

Numerical helpers at the top do not import AnyGoal, MuJoCo, or torch. The
runners import those only inside the command that needs them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

PAPER_SMWM_SUCCESS = {
    "insertpeg": 70.0,
    "avoid_peg": 74.0,
    "extractpeg": 24.0,
}
REPRODUCTION_AUDIT_PP = 10.0
N_STATES = 50
N_CANDIDATES = 16
HORIZON_STEPS = 25
BLOCK_STEPS = 5
N_BLOCKS = HORIZON_STEPS // BLOCK_STEPS
START_INDEX = 10
BANK_SEED = 0
NOISE_STD_SCALE = 0.1
N_PERTURBATIONS = 7
N_TRAIN_CHUNKS = 8
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
ACTION_DIM = 5
STATE_DIM = 44
PROPRIO_DIM = 33

# Released InsertPegEnv evaluation tolerances.
TOLERANCES = {
    "hand_position": 0.04,
    "hand_angle": 0.10,
    "object_position": 0.04,
    "object_angle": 0.10,
}
ERROR_NAMES = (
    "hand_position",
    "hand_angle",
    "object_position",
    "object_angle",
)
FULL_SUCCESS_KEYS = ERROR_NAMES
OBJECT_SUCCESS_KEYS = ("object_position", "object_angle")
RENDER_RELEVANT = np.r_[0:24, 29:44]

MODEL_REVISION = "b27f9dac9c750410d1331d258a95eff0d56570c2"
DATA_REVISION = "1719e7dab2fbeb07a0d5c3cf9cd881bc49510c88"
ANYGOAL_REVISION = "f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e"
SMWM_REVISION = "9d22bcc15c6ad8af8cc93337a950df2b49a866b9"

DATASETS = ("insertpeg", "avoid_peg", "extractpeg")
TRAIN_FILENAME = "manipulator_insert_peg_weak_expert_train.h5"
TEST_FILENAMES = {
    "insertpeg": TRAIN_FILENAME,
    "avoid_peg": "avoid_peg_test.h5",
    "extractpeg": "extract_peg_test.h5",
}


def goal_errors(goal_state: np.ndarray, state: np.ndarray) -> dict[str, float]:
    """Hand and peg errors. Same quantities as InsertPegEnv.insert_peg_goal_errors."""
    goal = np.asarray(goal_state, dtype=np.float64).reshape(STATE_DIM)
    current = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
    if not np.all(np.isfinite(goal)) or not np.all(np.isfinite(current)):
        raise ValueError("Goal and state must be finite.")
    return {
        "hand_position": float(np.linalg.norm(goal[29:31] - current[29:31])),
        "hand_angle": _angle_error(goal[31:33], current[31:33]),
        "object_position": float(np.linalg.norm(goal[33:35] - current[33:35])),
        "object_angle": _angle_error(goal[35:37], current[35:37]),
    }


def _planar_angle(quaternion_wy: np.ndarray) -> float:
    quaternion_wy = np.asarray(quaternion_wy, dtype=np.float64).reshape(2)
    return float(2.0 * np.arctan2(quaternion_wy[1], quaternion_wy[0]))


def _angle_error(goal_q: np.ndarray, current_q: np.ndarray) -> float:
    delta = _planar_angle(goal_q) - _planar_angle(current_q)
    return float(abs((delta + np.pi) % (2.0 * np.pi) - np.pi))


def reached(errors: dict[str, float], keys: tuple[str, ...]) -> bool:
    return all(errors[name] <= TOLERANCES[name] for name in keys)


def full_success(errors: dict[str, float]) -> bool:
    return reached(errors, FULL_SUCCESS_KEYS)


def object_success(errors: dict[str, float]) -> bool:
    return reached(errors, OBJECT_SUCCESS_KEYS)


def normalized_error_sum(errors: dict[str, float]) -> float:
    total = 0.0
    for name in ERROR_NAMES:
        total += errors[name] / TOLERANCES[name]
    return float(total)


def select_extract_episodes(
    records: list[dict],
    *,
    n_states: int = N_STATES,
    start_index: int = START_INDEX,
    horizon: int = HORIZON_STEPS,
) -> tuple[list[dict], list[dict]]:
    """Take the first ``n_states`` episodes that can host the fixed start.

    ``records`` must already be in the recorded deterministic order. An
    episode is eligible when frame ``start_index + horizon`` exists.
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive.")
    need = start_index + horizon + 1
    chosen: list[dict] = []
    excluded: list[dict] = []
    seen: set[int] = set()
    for record in records:
        episode_id = int(record["episode_id"])
        if episode_id in seen:
            raise ValueError(f"Duplicate episode id {episode_id}.")
        seen.add(episode_id)
        length = int(record["length"])
        if length < need:
            excluded.append(
                {
                    "episode_id": episode_id,
                    "length": length,
                    "reason": f"length {length} < {need} for start {start_index}",
                }
            )
            continue
        if len(chosen) < n_states:
            chosen.append(dict(record))
    if len(chosen) < n_states:
        raise RuntimeError(
            f"Only {len(chosen)} Extract-Peg episodes have a frame at "
            f"index {start_index} and a {horizon}-step continuation."
        )
    return chosen, excluded


def build_action_bank(
    recorded: np.ndarray,
    train_episodes: list[np.ndarray],
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Build the 16-sequence bank for one restored state.

    Candidate 0 is the recorded continuation. Candidates 1–7 perturb it in
    raw action units. Candidates 8–15 are training-file chunks, sorted by
    episode and start after the random draw so the set, not the RNG visit
    order, determines their positions.
    """
    recorded = np.asarray(recorded, dtype=np.float64)
    if recorded.shape != (HORIZON_STEPS, ACTION_DIM):
        raise ValueError(f"Recorded continuation must be {(HORIZON_STEPS, ACTION_DIM)}.")
    if not np.all(np.isfinite(recorded)):
        raise ValueError("Recorded continuation has non-finite actions.")
    low = np.asarray(action_low, dtype=np.float64).reshape(ACTION_DIM)
    high = np.asarray(action_high, dtype=np.float64).reshape(ACTION_DIM)
    std = np.asarray(action_std, dtype=np.float64).reshape(ACTION_DIM)
    if np.any(std <= 0.0) or not np.all(np.isfinite(std)):
        raise ValueError("Action standard deviations must be positive and finite.")

    bank = np.empty((N_CANDIDATES, HORIZON_STEPS, ACTION_DIM), dtype=np.float64)
    meta: list[dict] = []
    bank[0] = np.clip(recorded, low, high)
    meta.append({"kind": "recorded", "source": "evaluation_episode"})

    noise = rng.standard_normal((N_PERTURBATIONS, HORIZON_STEPS, ACTION_DIM))
    noise *= NOISE_STD_SCALE * std
    perturbed = np.clip(recorded + noise, low, high)
    bank[1 : 1 + N_PERTURBATIONS] = perturbed
    for index in range(N_PERTURBATIONS):
        meta.append(
            {
                "kind": "perturbation",
                "noise_scale": NOISE_STD_SCALE,
                "max_abs_delta": float(np.max(np.abs(perturbed[index] - recorded))),
            }
        )

    spans: list[tuple[int, int]] = []
    for episode_number, actions in enumerate(train_episodes):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError("Each training episode must have shape (T, 5).")
        n_starts = actions.shape[0] - HORIZON_STEPS + 1
        for start in range(max(n_starts, 0)):
            spans.append((episode_number, start))
    if len(spans) < N_TRAIN_CHUNKS:
        raise RuntimeError(
            f"Training file has {len(spans)} contiguous {HORIZON_STEPS}-step "
            f"chunks; need {N_TRAIN_CHUNKS}."
        )
    picked = np.sort(rng.choice(len(spans), size=N_TRAIN_CHUNKS, replace=False))
    ordered = sorted(spans[int(item)] for item in picked)
    for slot, (episode_number, start) in enumerate(ordered):
        chunk = np.asarray(train_episodes[episode_number][start : start + HORIZON_STEPS])
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Training chunk has non-finite actions.")
        bank[1 + N_PERTURBATIONS + slot] = np.clip(chunk, low, high)
        meta.append(
            {
                "kind": "train_chunk",
                "train_episode_list_index": int(episode_number),
                "start": int(start),
            }
        )
    if bank.shape != (N_CANDIDATES, HORIZON_STEPS, ACTION_DIM):
        raise AssertionError(bank.shape)
    return bank, meta


def block_normalized_actions(raw_actions: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalize raw actions with the training scaler and group each 5-step block."""
    raw = np.asarray(raw_actions, dtype=np.float64)
    if raw.shape != (N_CANDIDATES, HORIZON_STEPS, ACTION_DIM):
        raise ValueError(f"Unexpected action bank shape {raw.shape}.")
    mean = np.asarray(mean, dtype=np.float64).reshape(ACTION_DIM)
    scale = np.asarray(scale, dtype=np.float64).reshape(ACTION_DIM)
    if np.any(scale == 0.0):
        raise ValueError("Action scale contains a zero.")
    normalized = (raw - mean) / scale
    blocked = normalized.reshape(N_CANDIDATES, N_BLOCKS, BLOCK_STEPS, ACTION_DIM)
    return blocked.reshape(N_CANDIDATES, N_BLOCKS, BLOCK_STEPS * ACTION_DIM)


def select_lowest_cost(costs: np.ndarray) -> tuple[int, int]:
    costs = np.asarray(costs, dtype=np.float64).reshape(N_CANDIDATES)
    if not np.all(np.isfinite(costs)):
        raise ValueError("Costs must be finite.")
    selected = int(np.argmin(costs))
    ties = int(np.sum(costs == costs[selected]))
    return selected, ties


def select_true_utility(success: np.ndarray, error_sums: np.ndarray) -> tuple[int, int]:
    """Prefer any true success. Break remaining ties by pose error, then index."""
    success = np.asarray(success, dtype=bool).reshape(N_CANDIDATES)
    error_sums = np.asarray(error_sums, dtype=np.float64).reshape(N_CANDIDATES)
    if not np.all(np.isfinite(error_sums)):
        raise ValueError("Error sums must be finite.")
    rank = np.where(success, 0.0, 1.0)
    order = np.lexsort((np.arange(N_CANDIDATES), error_sums, rank))
    selected = int(order[0])
    n_success = int(success.sum())
    return selected, n_success


def opportunity_regret(success: np.ndarray, selected: int) -> int:
    success = np.asarray(success, dtype=bool).reshape(N_CANDIDATES)
    best = int(success.any())
    chosen = int(success[selected])
    return best - chosen


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_mean(
    values: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Episode-level percentile interval for a mean. Episodes are the resampled unit."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap values must be non-empty and finite.")
    if n_resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def section_8_checks(
    *,
    replay_pass_rate: float,
    solvability: float,
    regret_p: dict,
    regret_reduction: dict,
) -> dict[str, bool]:
    return {
        "replay_at_least_90": bool(replay_pass_rate >= 0.90),
        "bank_solvability_at_least_80": bool(solvability >= 0.80),
        "predicted_regret_mean_at_least_0.20": bool(regret_p["mean"] >= 0.20),
        "predicted_regret_ci_low_above_0.10": bool(regret_p["ci_low"] > 0.10),
        "observed_replacement_reduces_regret_by_at_least_0.15": bool(
            regret_reduction["mean"] >= 0.15
        ),
        "observed_replacement_interval_excludes_zero": bool(
            regret_reduction["ci_low"] > 0.0
        ),
    }


def interpret_stage_b(summary: dict) -> dict[str, str]:
    """Map Stage B onto the memorandum's GO / NO-GO criteria.

    Stage C is not claimed here. A prediction-shaped result stays pending
    those inference controls and does not authorize a new training method.
    """
    checks = summary["checks"]
    if not checks["replay_at_least_90"]:
        return {
            "decision": "NO-GO",
            "reason": (
                "Recorded-action replay reached the recorded goal on fewer than "
                "90% of states. This is a restoration, dynamics, or goal-alignment "
                "failure. Predicted-latent scores are not interpreted as evidence "
                "about the world model."
            ),
        }
    if not checks["bank_solvability_at_least_80"]:
        return {
            "decision": "NO-GO",
            "reason": (
                "The common 16-sequence bank contains a successful continuation on "
                "fewer than 80% of states, so it cannot support the ranking claim."
            ),
        }
    prediction_failure = (
        checks["predicted_regret_mean_at_least_0.20"]
        and checks["predicted_regret_ci_low_above_0.10"]
    )
    prediction_explains = (
        checks["observed_replacement_reduces_regret_by_at_least_0.15"]
        and checks["observed_replacement_interval_excludes_zero"]
    )
    if prediction_failure and prediction_explains:
        return {
            "decision": "PENDING-STAGE-C",
            "reason": (
                "Hypothesis H is supported on this bank: predicted-latent selection "
                "has high opportunity regret, and replacing the rollout with the "
                "actual future under the same terminal latent cost reduces that "
                "regret. Section 8.8 still requires the inference-only controls "
                "before a GO to standard-baseline diagnosis. This does not "
                "authorize a new representation method."
            ),
        }
    if prediction_failure and not prediction_explains:
        return {
            "decision": "NO-GO",
            "reason": (
                "Predicted-latent selection fails, but scoring the actual futures "
                "with the same latent goal cost does not remove the regret. The "
                "failure is consistent with goal scoring or with terminal-cost "
                "timing, not with an isolated dynamics-prediction error."
            ),
        }
    return {
        "decision": "NO-GO",
        "reason": (
            "Hypothesis H fails: predicted-latent opportunity regret is not above "
            "the pre-registered bar. This test does not support a predictive-learning "
            "bottleneck."
        ),
    }


def assert_visual_checkpoint_interface(state_dim: int, include_velocities: bool) -> None:
    """Refuse a 44-d vector into a checkpoint trained without velocities."""
    expected = STATE_DIM if include_velocities else PROPRIO_DIM
    if int(state_dim) == expected:
        return
    if not include_velocities:
        raise RuntimeError(
            f"Refusing to feed a {int(state_dim)}-d vector into a checkpoint "
            f"trained on {expected} features."
        )
    raise RuntimeError(
        f"State checkpoint declares state_dim={state_dim}, but "
        f"include_velocities=True implies {expected}."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def index_h5_episodes(path: Path) -> list[dict]:
    import hdf5plugin  # noqa: F401  # registers HDF5 compressors used by the release
    import h5py

    with h5py.File(path, "r") as handle:
        if "ep_len" not in handle or "ep_offset" not in handle:
            raise KeyError(f"{path} is missing ep_len/ep_offset.")
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        id_key = "episode_idx" if "episode_idx" in handle else "ep_idx"
        if id_key not in handle:
            raise KeyError(f"{path} has no episode index column.")
        episode_ids = np.asarray(handle[id_key][:])
        columns = sorted(key for key in handle.keys() if key not in {"ep_len", "ep_offset"})
    if len(lengths) != len(offsets):
        raise RuntimeError("ep_len and ep_offset disagree.")
    records = []
    for store_index, (length, offset) in enumerate(zip(lengths, offsets)):
        window = episode_ids[int(offset) : int(offset) + int(length)]
        unique = np.unique(window)
        if unique.size != 1:
            raise RuntimeError(
                f"Episode store {store_index} in {path} mixes ids {unique.tolist()}."
            )
        records.append(
            {
                "store_index": int(store_index),
                "episode_id": int(unique[0]),
                "length": int(length),
                "offset": int(offset),
            }
        )
    records.sort(key=lambda item: (item["episode_id"], item["store_index"]))
    duplicate_ids = [
        records[index]["episode_id"]
        for index in range(1, len(records))
        if records[index]["episode_id"] == records[index - 1]["episode_id"]
    ]
    if duplicate_ids:
        raise RuntimeError(f"Duplicate episode ids in {path}: {duplicate_ids[:5]}")
    records[0]["columns"] = columns  # type: ignore[index]
    return records


def open_h5(path: Path):
    import hdf5plugin  # noqa: F401
    import h5py

    return h5py.File(path, "r")


def read_rows(handle, offset: int, start: int, stop: int, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(handle[key][int(offset) + start : int(offset) + stop])
        for key in keys
    }


def as_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected an image, got shape {array.shape}.")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def load_training_actions(path: Path, records: list[dict]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    episodes: list[np.ndarray] = []
    with open_h5(path) as handle:
        all_actions = np.asarray(handle["action"][:], dtype=np.float64)
    for record in records:
        actions = all_actions[record["offset"] : record["offset"] + record["length"]]
        if actions.shape != (record["length"], ACTION_DIM):
            raise RuntimeError(
                f"Training episode {record['episode_id']} has actions {actions.shape}."
            )
        if not np.all(np.isfinite(actions)):
            raise RuntimeError(
                f"Non-finite training actions in episode {record['episode_id']}."
            )
        episodes.append(actions)
    stacked = np.concatenate(episodes, axis=0)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return episodes, scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64)


def record_environment(output: Path) -> dict:
    payload: dict = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "anygoal_revision_expected": ANYGOAL_REVISION,
        "smwm_revision_expected": SMWM_REVISION,
        "model_revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
        "env": {
            key: os.environ.get(key)
            for key in (
                "ANYGOAL_HOME",
                "ANYGOAL_DATA_ROOT",
                "ANYGOAL_CHECKPOINT_ROOT",
                "ANYGOAL_SOURCE_ROOT",
                "MUJOCO_GL",
            )
        },
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(0)
            payload["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as error:  # noqa: BLE001 - provenance should record the failure
        payload["torch_error"] = str(error)
    for module_name in ("mujoco", "numpy", "h5py", "gymnasium", "huggingface_hub"):
        try:
            module = __import__(module_name)
            payload[module_name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # noqa: BLE001
            payload[module_name] = f"unavailable: {error}"
    try:
        payload["anygoal_version"] = subprocess.check_output(
            ["anygoal", "--version"], text=True
        ).strip()
    except Exception as error:  # noqa: BLE001
        payload["anygoal_version"] = f"unavailable: {error}"
    write_json(output, payload)
    return payload


def parse_bool_vector(value) -> list[bool]:
    """Read a JSON list or the string form produced by json default=str on a NumPy array."""
    if isinstance(value, list):
        return [bool(item) for item in value]
    if isinstance(value, str):
        tokens = value.replace("[", " ").replace("]", " ").replace("\n", " ").split()
        parsed = []
        for token in tokens:
            if token in {"True", "true"}:
                parsed.append(True)
            elif token in {"False", "false"}:
                parsed.append(False)
            else:
                raise ValueError(f"Cannot parse success token {token!r}.")
        return parsed
    raise TypeError(f"Expected a success vector, got {type(value).__name__}.")


def _metrics_success(metrics: dict) -> dict[str, float]:
    dataset_goal = metrics.get("dataset_goal_criterion", {})
    native = metrics.get("native_insert_peg_criterion", {})
    final = metrics.get("insert_peg_final_errors", {})
    object_hits = None
    if final and not isinstance(next(iter(final.values())), str):
        count = len(next(iter(final.values())))
        object_hits = [
            object_success({name: float(final[name][index]) for name in ERROR_NAMES})
            for index in range(count)
        ]
    return {
        "dataset_goal_success_percent": float(
            dataset_goal.get("success_rate", metrics.get("success_rate"))
        ),
        "native_insert_success_percent": float(native.get("success_rate"))
        if native
        else float("nan"),
        "final_object_success_percent": (
            float(np.mean(object_hits) * 100.0) if object_hits is not None else float("nan")
        ),
        "n": len(parse_bool_vector(metrics["episode_successes"]))
        if "episode_successes" in metrics
        else 0,
    }


def audit_stage_a(results: Path, data_root: Path) -> dict:
    """Join native metrics with the evaluator's own start sampler and split hashes."""
    from anygoal.runners.plan.planning_sensorimotor import sample_start_indices
    import stable_worldmodel as swm

    report: dict = {"datasets": {}, "paper_reference_percent": PAPER_SMWM_SUCCESS}
    train_path = data_root / TRAIN_FILENAME
    train_records = index_h5_episodes(train_path)
    train_hashes = _episode_hashes(train_path, train_records)
    for dataset_name in DATASETS:
        metrics_path = results / "reproduction" / "smwm" / dataset_name / "seed_0" / "metrics.json"
        if not metrics_path.is_file():
            report["datasets"][dataset_name] = {"status": "missing", "path": str(metrics_path)}
            continue
        saved = json.loads(metrics_path.read_text())
        metrics = saved["metrics"]
        config = saved["config"]
        filename = TEST_FILENAMES[dataset_name]
        dataset_path = data_root / filename
        dataset = swm.data.HDF5Dataset(
            path=str(dataset_path),
            keys_to_cache=["action", "observation"],
        )
        episodes, start_steps = sample_start_indices(
            dataset,
            int(config["eval"]["num_eval"]),
            int(config["eval"]["goal_offset_steps"]),
            int(saved["task_seed"]),
            str(config["eval"].get("start_selection", "random")),
        )
        successes = parse_bool_vector(metrics["episode_successes"])
        if len(successes) != len(episodes):
            raise RuntimeError(
                f"{dataset_name}: {len(successes)} successes vs {len(episodes)} sampled starts."
            )
        rates = _metrics_success(metrics)
        reference = PAPER_SMWM_SUCCESS[dataset_name]
        delta = rates["dataset_goal_success_percent"] - reference
        evaluated = index_h5_episodes(dataset_path)
        evaluated_hashes = (
            train_hashes
            if dataset_path.resolve() == train_path.resolve()
            else _episode_hashes(dataset_path, evaluated)
        )
        train_digest_set = set(train_hashes.values())
        overlap = [
            evaluated_hashes[int(episode_id)] in train_digest_set
            for episode_id in episodes
        ]
        stats_name = str(config["dataset"].get("stats", config["eval"]["dataset_name"]))
        report["datasets"][dataset_name] = {
            "status": "scored",
            "metrics_path": str(metrics_path),
            "elapsed_seconds": saved.get("elapsed_seconds"),
            "task_seed": saved.get("task_seed"),
            "policy_seed": saved.get("policy_seed"),
            "eval_dataset_name": config["eval"]["dataset_name"],
            "stats_dataset_name": stats_name,
            "stats_are_training_file": stats_name == Path(TRAIN_FILENAME).stem,
            "label": (
                "base-task reference on training source"
                if dataset_name == "insertpeg"
                else "held-out goal file"
            ),
            "rates": rates,
            "paper_reference_percent": reference,
            "dataset_goal_minus_paper_pp": delta,
            "audit_trigger": abs(delta) > REPRODUCTION_AUDIT_PP,
            "episodes": [int(item) for item in episodes],
            "start_steps": [int(item) for item in start_steps],
            "episode_successes": successes,
            "final_errors": metrics.get("insert_peg_final_errors"),
            "trajectory_hash_overlap_with_train": int(sum(overlap)),
            "n_evaluated": len(episodes),
            "sampler_note": (
                "sample_start_indices draws with np.random.default_rng and "
                "choice(len(valid_indices) - 1), so the last valid row is excluded. "
                "This audit calls that function; it does not repair it."
            ),
        }
        dataset.h5_file.close()
    write_json(results / "stage_a_audit.json", report)
    return report


def _episode_hashes(path: Path, records: list[dict]) -> dict[int, str]:
    with open_h5(path) as handle:
        observations = np.asarray(handle["observation"][:])
        actions = np.asarray(handle["action"][:])
    hashes = {}
    for record in records:
        start = record["offset"]
        stop = start + record["length"]
        hashes[int(record["episode_id"])] = _hash_rows(
            {"observation": observations[start:stop], "action": actions[start:stop]}
        )
    return hashes


def _hash_rows(rows: dict[str, np.ndarray]) -> str:
    observation = np.round(np.asarray(rows["observation"], dtype=np.float64), 6)
    action = np.round(np.asarray(rows["action"], dtype=np.float64), 6)
    digest = hashlib.sha256()
    digest.update(observation.tobytes())
    digest.update(action.tobytes())
    return digest.hexdigest()


def _policy_image(frame_hwc: np.ndarray, transform):
    import torch
    from torchvision.tv_tensors import Image

    array = np.transpose(as_hwc_uint8(frame_hwc), (2, 0, 1))
    return torch.stack([transform(Image(array))])


def _terminal_mse(predicted_emb, goal_emb):
    import torch.nn.functional as F

    if goal_emb.ndim + 1 == predicted_emb.ndim:
        goal_emb = goal_emb.unsqueeze(1)
    goal_emb = goal_emb[..., -1:, :].expand_as(predicted_emb)
    cost = F.mse_loss(
        predicted_emb[..., -1:, :],
        goal_emb[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, predicted_emb.ndim)))
    return cost


def score_bank(model, transform, start_hwc, goal_hwc, final_hwc, blocked_actions) -> tuple[np.ndarray, np.ndarray, str]:
    """Return predicted costs, observed terminal costs, and which cost path ran."""
    import torch

    current = _policy_image(start_hwc, transform).to(next(model.parameters()).device)
    goal = _policy_image(goal_hwc, transform).to(current.device)
    finals = torch.cat(
        [_policy_image(frame, transform) for frame in final_hwc],
        dim=0,
    ).to(current.device)
    actions = torch.as_tensor(blocked_actions, dtype=current.dtype, device=current.device)
    actions = actions.unsqueeze(0)
    n_samples = actions.shape[1]
    info = {
        "pixels": current[:, None, None].expand(1, n_samples, 1, *current.shape[1:]).contiguous(),
        "goal": goal[:, None, None].expand(1, n_samples, 1, *goal.shape[1:]).contiguous(),
    }
    mode = "native_get_cost"
    try:
        predicted = model.get_cost(info, actions)
    except RuntimeError as error:
        if "expand" not in str(error).lower() and "match" not in str(error).lower():
            raise
        mode = "terminal_mse_after_broadcast_repair"
        predicted_info = model.rollout(
            {key: value for key, value in info.items()},
            actions,
        )
        goal_encoded = model.encode({"pixels": goal})["emb"]
        predicted = _terminal_mse(predicted_info["predicted_emb"], goal_encoded)
    observed_emb = model.encode({"pixels": finals})["emb"].unsqueeze(0)
    goal_emb = model.encode({"pixels": goal})["emb"]
    observed = _terminal_mse(observed_emb, goal_emb)
    predicted_np = predicted.detach().cpu().numpy().reshape(N_CANDIDATES)
    observed_np = observed.detach().cpu().numpy().reshape(N_CANDIDATES)
    if not np.all(np.isfinite(predicted_np)) or not np.all(np.isfinite(observed_np)):
        raise RuntimeError("Non-finite latent costs.")
    return predicted_np, observed_np, mode


def replay_recorded_continuations(path: Path, records: list[dict]) -> list[dict]:
    """Replay each recorded 25-step continuation with the evaluator's env.step."""
    from anygoal.envs.insert_peg.env import (
        InsertPegEnv,
        insert_peg_goal_errors,
        insert_peg_goal_reached,
    )

    env = InsertPegEnv(terminate_at_goal=False)
    rows: list[dict] = []
    try:
        tolerances = {name: float(env._tolerances[name]) for name in ERROR_NAMES}
        if tolerances != TOLERANCES:
            raise RuntimeError(f"Evaluator tolerances changed: {tolerances}")
        with open_h5(path) as handle:
            for record in records:
                window = read_rows(
                    handle,
                    record["offset"],
                    START_INDEX,
                    START_INDEX + HORIZON_STEPS + 1,
                    ("action", "observation"),
                )
                actions = np.asarray(window["action"][:HORIZON_STEPS], dtype=np.float64)
                states = np.asarray(window["observation"], dtype=np.float64)
                if actions.shape != (HORIZON_STEPS, ACTION_DIM) or states.shape[0] != HORIZON_STEPS + 1:
                    raise RuntimeError(
                        f"Episode {record['episode_id']} does not have a 25-step continuation."
                    )
                _state, _info = env.reset(
                    seed=0,
                    options={"state": states[0], "goal_state": states[HORIZON_STEPS]},
                )
                restored = env.get_state()
                restore_pose = float(
                    np.max(np.abs(restored[np.r_[0:16, 29:37]] - states[0][np.r_[0:16, 29:37]]))
                )
                for action in actions:
                    env.step(action)
                errors = insert_peg_goal_errors(states[HORIZON_STEPS], env.get_state())
                rows.append(
                    {
                        "episode_id": int(record["episode_id"]),
                        "start_index": START_INDEX,
                        "restore_pose_max_abs": restore_pose,
                        "replay_goal_reached": bool(insert_peg_goal_reached(errors, env._tolerances)),
                        "replay_object_reached": bool(object_success(errors)),
                        **{name: float(errors[name]) for name in ERROR_NAMES},
                    }
                )
    finally:
        env.close()
    return rows


def _replay_summary(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("Replay audit produced no episodes.")
    errors = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in ERROR_NAMES}
    return {
        "n": len(rows),
        "pass_rate": float(np.mean([row["replay_goal_reached"] for row in rows])),
        "object_pass_rate": float(np.mean([row["replay_object_reached"] for row in rows])),
        "restore_pose_max_abs": float(np.max([row["restore_pose_max_abs"] for row in rows])),
        "median_errors": {name: float(np.median(values)) for name, values in errors.items()},
    }


def replay_named_starts(path: Path, starts: list[tuple[int, int]]) -> list[dict]:
    """Replay the recorded actions at evaluator-sampled (episode, start) pairs."""
    from anygoal.envs.insert_peg.env import (
        InsertPegEnv,
        insert_peg_goal_errors,
        insert_peg_goal_reached,
    )

    records = {record["episode_id"]: record for record in index_h5_episodes(path)}
    env = InsertPegEnv(terminate_at_goal=False)
    rows: list[dict] = []
    try:
        with open_h5(path) as handle:
            for episode_id, start in starts:
                record = records[int(episode_id)]
                if int(start) < 0 or int(start) + HORIZON_STEPS >= record["length"]:
                    raise RuntimeError(
                        f"Episode {episode_id} start {start} has no {HORIZON_STEPS}-step continuation."
                    )
                window = read_rows(
                    handle,
                    record["offset"],
                    int(start),
                    int(start) + HORIZON_STEPS + 1,
                    ("action", "observation"),
                )
                actions = np.asarray(window["action"][:HORIZON_STEPS], dtype=np.float64)
                states = np.asarray(window["observation"], dtype=np.float64)
                env.reset(
                    seed=0,
                    options={"state": states[0], "goal_state": states[HORIZON_STEPS]},
                )
                for action in actions:
                    env.step(action)
                errors = insert_peg_goal_errors(states[HORIZON_STEPS], env.get_state())
                rows.append(
                    {
                        "episode_id": int(episode_id),
                        "start_index": int(start),
                        "replay_goal_reached": bool(
                            insert_peg_goal_reached(errors, env._tolerances)
                        ),
                        "replay_object_reached": bool(object_success(errors)),
                        **{name: float(errors[name]) for name in ERROR_NAMES},
                    }
                )
    finally:
        env.close()
    return rows


def run_stage_b(results: Path, data_root: Path, checkpoint: Path, anygoal_root: Path) -> dict:
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(anygoal_root))
    import anygoal  # noqa: F401
    from anygoal.envs.insert_peg.env import InsertPegEnv
    from anygoal.resources import resource_path
    from anygoal.runners.plan.planning_sensorimotor import image_transform, load_model
    from anygoal.sources import source_path
    import torch

    extract_path = data_root / "extract_peg_test.h5"
    train_path = data_root / TRAIN_FILENAME
    extract_records = index_h5_episodes(extract_path)
    chosen, excluded = select_extract_episodes(extract_records)
    replay_rows = replay_recorded_continuations(extract_path, chosen)
    replay = _replay_summary(replay_rows)
    references = {}
    for name, filename in (
        ("insertpeg_train", TRAIN_FILENAME),
        ("avoid_peg", "avoid_peg_test.h5"),
    ):
        reference_path = data_root / filename
        reference_records = index_h5_episodes(reference_path)
        reference_chosen, _reference_excluded = select_extract_episodes(reference_records)
        references[name] = _replay_summary(
            replay_recorded_continuations(reference_path, reference_chosen)
        )
    if replay["pass_rate"] < 0.90:
        summary = {
            "n_states": replay["n"],
            "n_excluded_short": len(excluded),
            "excluded": excluded,
            "replay_pass_rate": replay["pass_rate"],
            "replay": replay,
            "replay_references": references,
            "solvability_full": None,
            "model_scoring": "not_run",
            "scoring_note": (
                "Recorded continuations were replayed with InsertPegEnv.step, "
                "the evaluator used by the released planner. The 90% goal-reach "
                "check failed, so predicted-latent and observed-latent costs were "
                "not computed. The training-file reference uses the same player."
            ),
            "checks": {"replay_at_least_90": False},
            "checks_not_reached": [
                "bank_solvability_at_least_80",
                "predicted_regret_mean_at_least_0.20",
                "predicted_regret_ci_low_above_0.10",
                "observed_replacement_reduces_regret_by_at_least_0.15",
                "observed_replacement_interval_excludes_zero",
            ],
        }
        summary["interpretation"] = interpret_stage_b(summary)
        output = results / "stage_b"
        output.mkdir(parents=True, exist_ok=True)
        _write_csv(output / "replay_episodes.csv", replay_rows)
        write_json(output / "summary.json", summary)
        print(
            f"Replay pass rate {replay['pass_rate']:.3f}. "
            "Stopping before latent scoring, as required when replay is below 90%.",
            flush=True,
        )
        return summary
    if not torch.cuda.is_available():
        raise RuntimeError("Stage B requires CUDA. Refusing a silent CPU fallback.")
    train_records = index_h5_episodes(train_path)
    train_actions, action_mean, action_scale = load_training_actions(train_path, train_records)
    env = InsertPegEnv(render_mode="rgb_array", width=224, height=224, terminate_at_goal=False)
    extract_handle = None
    try:
        if {name: float(env._tolerances[name]) for name in ERROR_NAMES} != TOLERANCES:
            raise RuntimeError(f"Env tolerances changed: {env._tolerances}")
        low = np.asarray(env.action_space.low, dtype=np.float64)
        high = np.asarray(env.action_space.high, dtype=np.float64)
        source = source_path("sensorimotor")
        planning_source = source / "planning" if (source / "planning" / "jepa.py").is_file() else source
        model = load_model(
            planning_source,
            planning_source,
            "cuda",
            train_config_path=resource_path("train", "sensorimotor", "insert_peg.yaml"),
            checkpoint_path=checkpoint,
        )
        transform = image_transform(224)
        rng = np.random.default_rng(BANK_SEED)
        rows = []
        costs_p = []
        costs_o = []
        success_full = []
        success_terminal = []
        success_object = []
        cost_mode = None
        extract_handle = open_h5(extract_path)
        for record in chosen:
            window = read_rows(
                extract_handle,
                record["offset"],
                START_INDEX,
                START_INDEX + HORIZON_STEPS + 1,
                ("action", "observation", "pixels"),
            )
            recorded_actions = np.asarray(window["action"][:HORIZON_STEPS], dtype=np.float64)
            start_state = np.asarray(window["observation"][0], dtype=np.float32)
            goal_state = np.asarray(window["observation"][HORIZON_STEPS], dtype=np.float32)
            start_image = as_hwc_uint8(window["pixels"][0])
            goal_image = as_hwc_uint8(window["pixels"][HORIZON_STEPS])
            bank, bank_meta = build_action_bank(
                recorded_actions,
                train_actions,
                low,
                high,
                action_scale,
                rng,
            )
            episode_success = []
            episode_terminal = []
            episode_object = []
            episode_errors = []
            episode_terminal_errors = []
            final_images = []
            replay_error = None
            for candidate, actions in enumerate(bank):
                _state, info = env.reset(
                    seed=0,
                    options={"state": start_state, "goal_state": goal_state},
                )
                restored = env.get_state()
                if candidate == 0:
                    render = as_hwc_uint8(env.render())
                    replay_pixel_mae = float(np.mean(np.abs(render.astype(np.int16) - start_image.astype(np.int16))))
                ever_full = bool(info["evaluation_success"])
                ever_object = object_success(goal_errors(goal_state, env.get_state()))
                ever_native = bool(info["task_success"])
                min_object = goal_errors(goal_state, env.get_state())["object_position"]
                for _step, action in enumerate(actions):
                    _state, _reward, _terminated, _truncated, info = env.step(action)
                    errors_now = goal_errors(goal_state, env.get_state())
                    ever_full = ever_full or bool(info["evaluation_success"])
                    ever_object = ever_object or object_success(errors_now)
                    ever_native = ever_native or bool(info["task_success"])
                    min_object = min(min_object, errors_now["object_position"])
                terminal_errors = goal_errors(goal_state, env.get_state())
                if candidate == 0:
                    replay_error = float(np.max(np.abs(env.get_state()[RENDER_RELEVANT] - goal_state[RENDER_RELEVANT])))
                    replay_full = full_success(terminal_errors)
                    replay_pixel_mae_out = replay_pixel_mae
                final_images.append(as_hwc_uint8(env.render()))
                episode_success.append(ever_full)
                episode_terminal.append(full_success(terminal_errors))
                episode_object.append(ever_object)
                episode_errors.append(normalized_error_sum(terminal_errors))
                episode_terminal_errors.append(terminal_errors)
            blocked = block_normalized_actions(bank, action_mean, action_scale)
            predicted, observed, cost_mode = score_bank(
                model,
                transform,
                start_image,
                goal_image,
                final_images,
                blocked,
            )
            p_index, p_ties = select_lowest_cost(predicted)
            o_index, o_ties = select_lowest_cost(observed)
            success_array = np.asarray(episode_success, dtype=bool)
            t_index, n_success = select_true_utility(success_array, np.asarray(episode_errors))
            p_regret = opportunity_regret(success_array, p_index)
            o_regret = opportunity_regret(success_array, o_index)
            t_regret = opportunity_regret(success_array, t_index)
            if int(success_array.any()) != (1 - t_regret):
                raise RuntimeError("True-utility selection did not realize the bank's best success.")
            terminal_array = np.asarray(episode_terminal, dtype=bool)
            rows.append(
                {
                    "episode_id": record["episode_id"],
                    "start_index": START_INDEX,
                    "goal_index": START_INDEX + HORIZON_STEPS,
                    "length": record["length"],
                    "replay_goal_reached": bool(replay_full),
                    "replay_render_relevant_max_abs": replay_error,
                    "replay_start_pixel_mae": replay_pixel_mae_out,
                    "bank_solvable_full": bool(success_array.any()),
                    "bank_solvable_terminal": bool(terminal_array.any()),
                    "bank_solvable_object": bool(np.any(episode_object)),
                    "n_success_full": n_success,
                    "p_index": p_index,
                    "o_index": o_index,
                    "t_index": t_index,
                    "p_ties": p_ties,
                    "o_ties": o_ties,
                    "p_regret": p_regret,
                    "o_regret": o_regret,
                    "t_regret": t_regret,
                    "p_regret_terminal": opportunity_regret(terminal_array, p_index),
                    "o_regret_terminal": opportunity_regret(terminal_array, o_index),
                    "p_success": bool(success_array[p_index]),
                    "o_success": bool(success_array[o_index]),
                    "t_success": bool(success_array[t_index]),
                    "spearman_p_o": spearman(predicted, observed),
                    "same_argmin": int(p_index == o_index),
                    "h_indicator": int(p_regret == 1 and o_regret == 0),
                    "goal_score_indicator": int(
                        p_regret == 1 and o_regret == 1 and success_array.any()
                    ),
                    "min_object_position": float(min(item["object_position"] for item in episode_terminal_errors)),
                    **{
                        f"p_{name}": episode_terminal_errors[p_index][name]
                        for name in ERROR_NAMES
                    },
                    **{
                        f"o_{name}": episode_terminal_errors[o_index][name]
                        for name in ERROR_NAMES
                    },
                    "bank_meta": bank_meta,
                }
            )
            costs_p.append(predicted)
            costs_o.append(observed)
            success_full.append(success_array)
            success_terminal.append(terminal_array)
            success_object.append(np.asarray(episode_object, dtype=bool))
            print(
                f"episode {record['episode_id']}: replay={bool(replay_full)} "
                f"solvable={bool(success_array.any())} "
                f"regret P/O/T={p_regret}/{o_regret}/{t_regret}",
                flush=True,
            )
    finally:
        if extract_handle is not None:
            extract_handle.close()
        env.close()

    regret_p = bootstrap_mean([row["p_regret"] for row in rows])
    regret_o = bootstrap_mean([row["o_regret"] for row in rows])
    reduction = bootstrap_mean(
        [row["p_regret"] - row["o_regret"] for row in rows],
        seed=BOOTSTRAP_SEED + 1,
    )
    h_rate = bootstrap_mean([row["h_indicator"] for row in rows], seed=BOOTSTRAP_SEED + 2)
    replay_rate = float(np.mean([row["replay_goal_reached"] for row in rows]))
    solvability = float(np.mean([row["bank_solvable_full"] for row in rows]))
    summary = {
        "n_states": len(rows),
        "n_excluded_short": len(excluded),
        "excluded": excluded,
        "cost_mode": cost_mode,
        "action_mean": action_mean.tolist(),
        "action_scale": action_scale.tolist(),
        "replay_pass_rate": replay_rate,
        "solvability_full": solvability,
        "solvability_terminal": float(np.mean([row["bank_solvable_terminal"] for row in rows])),
        "solvability_object": float(np.mean([row["bank_solvable_object"] for row in rows])),
        "regret_p": regret_p,
        "regret_o": regret_o,
        "regret_reduction_p_minus_o": reduction,
        "hypothesis_h_rate": h_rate,
        "goal_score_rate": bootstrap_mean(
            [row["goal_score_indicator"] for row in rows],
            seed=BOOTSTRAP_SEED + 3,
        ),
        "argmin_agreement": float(np.mean([row["same_argmin"] for row in rows])),
        "mean_spearman_p_o": float(np.nanmean([row["spearman_p_o"] for row in rows])),
        "mean_replay_state_error": float(
            np.mean([row["replay_render_relevant_max_abs"] for row in rows])
        ),
        "checks": section_8_checks(
            replay_pass_rate=replay_rate,
            solvability=solvability,
            regret_p=regret_p,
            regret_reduction=reduction,
        ),
        "population_note": (
            "Stage B uses the first 50 episode ids that contain frame 35, "
            "started at index 10. Stage A samples random valid rows from seed 0. "
            "These populations are not the same experiment."
        ),
        "scoring_note": (
            "P is the model terminal latent cost of the 25-step sequence. "
            "O encodes the simulated frame after all 25 actions and applies the "
            "same terminal MSE against the dataset goal image. T is ever-achieved "
            "dataset-goal success during those 25 actions. A terminal-only regret "
            "is also stored because the native cost does not score intermediate frames."
        ),
    }
    summary["interpretation"] = interpret_stage_b(summary)
    output = results / "stage_b"
    output.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        item = dict(row)
        item.pop("bank_meta", None)
        serializable.append(item)
    _write_csv(output / "episodes.csv", serializable)
    write_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "banks.npz",
        predicted_cost=np.asarray(costs_p),
        observed_cost=np.asarray(costs_o),
        success_full=np.asarray(success_full),
        success_terminal=np.asarray(success_terminal),
        success_object=np.asarray(success_object),
        episode_id=np.asarray([row["episode_id"] for row in rows]),
    )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty CSV.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_with_interval(ax, labels, means, lows, highs, ylabel: str, title: str) -> None:
    x = np.arange(len(labels))
    ax.bar(x, means, color="#3c6e71", width=0.7)
    yerr = np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)])
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, max(highs) * 1.15))


def _success_rates_from_text(text: str) -> tuple[float, float]:
    import re

    rates = [float(value) for value in re.findall(r"'success_rate': ([0-9.]+)", text)]
    if len(rates) < 3:
        raise RuntimeError("A planning result file does not contain three success rates.")
    return rates[0], rates[2]


def state_predictor_rates(results: Path) -> list[dict]:
    rows = []
    for model in ("toy-adaln", "toy-action-duplicate"):
        for dataset in DATASETS:
            path = results / "reproduction" / model / dataset / "results.txt"
            if not path.is_file():
                continue
            goal, native = _success_rates_from_text(path.read_text())
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "dataset_goal_success_percent": goal,
                    "native_insert_success_percent": native,
                    "state_dim": 33,
                    "include_velocities": False,
                }
            )
    return rows


def _state_predictor_lines(results: Path) -> list[str]:
    rows = state_predictor_rates(results)
    if not rows:
        return []
    lines = [
        "Released 33-dimensional state predictors, velocities excluded, seed 0, 50 evaluations:",
        "",
        "| Model | Dataset | Dataset-goal success | Native insert success |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{row['dataset_goal_success_percent']:.1f} | "
            f"{row['native_insert_success_percent']:.1f} |"
        )
    lines.append("")
    return lines


def write_figures(results: Path, docs: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs.mkdir(parents=True, exist_ok=True)
    audit_path = results / "stage_a_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        labels = []
        measured = []
        reference = []
        for name in DATASETS:
            item = audit["datasets"].get(name, {})
            if item.get("status") != "scored":
                continue
            labels.append(name)
            measured.append(item["rates"]["dataset_goal_success_percent"])
            reference.append(item["paper_reference_percent"])
        if labels:
            state_rows = state_predictor_rates(results)
            series = [("This SMWM", measured), ("Paper SMWM", reference)]
            for model, color_name in (
                ("toy-adaln", "AdaLN 33-d"),
                ("toy-action-duplicate", "Action-duplicate 33-d"),
            ):
                values = []
                for name in labels:
                    match = [
                        row["dataset_goal_success_percent"]
                        for row in state_rows
                        if row["model"] == model and row["dataset"] == name
                    ]
                    values.append(match[0] if match else float("nan"))
                series.append((color_name, values))
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            x = np.arange(len(labels))
            width = 0.18
            colors = ["#3c6e71", "#d9d9d9", "#8d6a9f", "#c4a35a"]
            for index, (label, values) in enumerate(series):
                ax.bar(x + (index - 1.5) * width, values, width=width, label=label, color=colors[index])
            ax.set_xticks(x, labels)
            ax.set_ylabel("Dataset-goal success (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Stage A: seed 0, 50 evaluations")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(docs / "stage_a_success.png", dpi=140)
            plt.close(fig)
        replay_rates = []
        replay_labels = []
        for name in labels:
            replay = audit["datasets"][name].get("recorded_action_replay")
            if not replay:
                continue
            replay_labels.append(name)
            replay_rates.append(100.0 * replay["pass_rate"])
        if replay_labels:
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            ax.bar(np.arange(len(replay_labels)), replay_rates, color="#3c6e71")
            ax.axhline(90, color="#9a3b3b", linestyle="--", linewidth=1, label="90% replay gate")
            ax.set_xticks(np.arange(len(replay_labels)), replay_labels)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Recorded 25-step actions reaching the goal (%)")
            ax.set_title("Same Stage A starts, evaluator env.step")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(docs / "recorded_action_replay.png", dpi=140)
            plt.close(fig)
    summary_path = results / "stage_b" / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    if summary.get("model_scoring") == "not_run":
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    series = [
        ("P regret", summary["regret_p"]),
        ("O regret", summary["regret_o"]),
        ("H rate", summary["hypothesis_h_rate"]),
    ]
    _bar_with_interval(
        ax,
        [item[0] for item in series],
        [item[1]["mean"] for item in series],
        [item[1]["ci_low"] for item in series],
        [item[1]["ci_high"] for item in series],
        "Episode mean",
        "Stage B: regret and hypothesis H",
    )
    fig.tight_layout()
    fig.savefig(docs / "stage_b_regret.png", dpi=140)
    plt.close(fig)

    episodes = list(csv.DictReader((results / "stage_b" / "episodes.csv").open()))
    counts = {
        "H: P fails, O works": sum(int(row["h_indicator"]) for row in episodes),
        "P and O fail, bank works": sum(int(row["goal_score_indicator"]) for row in episodes),
        "Unsolvable bank": sum(int(row["bank_solvable_full"]) == 0 for row in episodes),
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.barh(list(counts.keys()), list(counts.values()), color="#3c6e71")
    ax.set_xlabel(f"Episodes out of {len(episodes)}")
    ax.set_title("Stage B failure attribution")
    fig.tight_layout()
    fig.savefig(docs / "stage_b_attribution.png", dpi=140)
    plt.close(fig)

    banks = np.load(results / "stage_b" / "banks.npz")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(
        banks["predicted_cost"].reshape(-1),
        banks["observed_cost"].reshape(-1),
        s=8,
        alpha=0.35,
        color="#3c6e71",
    )
    ax.set_xlabel("Predicted-latent cost")
    ax.set_ylabel("Observed-latent cost")
    ax.set_title("Same 800 action sequences")
    fig.tight_layout()
    fig.savefig(docs / "stage_b_cost_scatter.png", dpi=140)
    plt.close(fig)


def write_report(results: Path, docs: Path) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    lines = ["# AnyGoal contact-action diagnostic", ""]
    audit_path = results / "stage_a_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        lines.append("## Stage A")
        lines.append("")
        lines.append("| Dataset | Dataset-goal success | Paper SMWM | Native insert success | Final object success | Recorded-action replay | Train-hash overlap | Label |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for name in DATASETS:
            item = audit["datasets"].get(name, {})
            if item.get("status") != "scored":
                lines.append(f"| {name} | missing | {PAPER_SMWM_SUCCESS[name]:.1f} |  |  |  |  | {item.get('status', 'missing')} |")
                continue
            rates = item["rates"]
            replay = item.get("recorded_action_replay", {})
            replay_text = (
                f"{100.0 * replay['pass_rate']:.1f}" if replay else ""
            )
            lines.append(
                f"| {name} | {rates['dataset_goal_success_percent']:.1f} | "
                f"{item['paper_reference_percent']:.1f} | "
                f"{rates['native_insert_success_percent']:.1f} | "
                f"{rates['final_object_success_percent']:.1f} | "
                f"{replay_text} | "
                f"{item['trajectory_hash_overlap_with_train']} / {item['n_evaluated']} | "
                f"{item['label']} |"
            )
        lines.append("")
        lines.extend(_state_predictor_lines(results))
    summary_path = results / "stage_b" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        lines.extend(
            [
                "## Stage B",
                "",
                f"Decision: **{summary['interpretation']['decision']}**",
                "",
                summary["interpretation"]["reason"],
                "",
            ]
        )
        if summary.get("model_scoring") == "not_run":
            replay = summary["replay"]
            lines.extend(
                [
                    "| Replay set | Full goal reached | Object pose reached | Median hand position | Median object position |",
                    "|---|---:|---:|---:|---:|",
                    (
                        f"| Extract-Peg, first 50 at index 10 | {replay['pass_rate']:.3f} | "
                        f"{replay['object_pass_rate']:.3f} | "
                        f"{replay['median_errors']['hand_position']:.4f} | "
                        f"{replay['median_errors']['object_position']:.4f} |"
                    ),
                ]
            )
            for name, item in summary.get("replay_references", {}).items():
                lines.append(
                    f"| {name} | {item['pass_rate']:.3f} | {item['object_pass_rate']:.3f} | "
                    f"{item['median_errors']['hand_position']:.4f} | "
                    f"{item['median_errors']['object_position']:.4f} |"
                )
            lines.extend(["", summary.get("scoring_note", ""), ""])
            (docs / "summary.md").write_text("\n".join(lines) + "\n")
            return
        lines.extend(
            [
                "| Quantity | Mean | 95% interval |",
                "|---|---:|---:|",
            ]
        )
        for label, key in (
            ("P opportunity regret", "regret_p"),
            ("O opportunity regret", "regret_o"),
            ("P minus O regret", "regret_reduction_p_minus_o"),
            ("H rate (P fails and O succeeds)", "hypothesis_h_rate"),
            ("P and O fail while the bank succeeds", "goal_score_rate"),
        ):
            item = summary[key]
            lines.append(
                f"| {label} | {item['mean']:.3f} | [{item['ci_low']:.3f}, {item['ci_high']:.3f}] |"
            )
        lines.extend(
            [
                "",
                f"Replay pass rate: {summary['replay_pass_rate']:.3f}.",
                f"Bank solvability (ever full success): {summary['solvability_full']:.3f}.",
                f"Terminal-frame solvability: {summary['solvability_terminal']:.3f}.",
                f"Object-only solvability: {summary['solvability_object']:.3f}.",
                f"Argmin agreement: {summary['argmin_agreement']:.3f}.",
                f"Mean Spearman P vs O: {summary['mean_spearman_p_o']:.3f}.",
                f"Cost path: {summary['cost_mode']}.",
                "",
                "Checks:",
                "",
            ]
        )
        for name, value in summary["checks"].items():
            lines.append(f"- {name}: {value}")
        lines.append("")
    (docs / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    env_parser = subparsers.add_parser("record-env")
    env_parser.add_argument("--output", type=Path, required=True)
    audit_parser = subparsers.add_parser("stage-a-audit")
    audit_parser.add_argument("--results", type=Path, required=True)
    audit_parser.add_argument("--data-root", type=Path, required=True)
    replay_parser = subparsers.add_parser("stage-a-replay")
    replay_parser.add_argument("--results", type=Path, required=True)
    replay_parser.add_argument("--data-root", type=Path, required=True)
    stage_parser = subparsers.add_parser("stage-b")
    stage_parser.add_argument("--results", type=Path, required=True)
    stage_parser.add_argument("--data-root", type=Path, required=True)
    stage_parser.add_argument("--checkpoint", type=Path, required=True)
    stage_parser.add_argument("--anygoal-root", type=Path, required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--results", type=Path, required=True)
    report_parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record-env":
        record_environment(args.output)
    elif args.command == "stage-a-audit":
        audit_stage_a(args.results, args.data_root)
    elif args.command == "stage-a-replay":
        audit_path = args.results / "stage_a_audit.json"
        audit = json.loads(audit_path.read_text())
        for name in DATASETS:
            item = audit["datasets"][name]
            starts = list(zip(item["episodes"], item["start_steps"]))
            rows = replay_named_starts(args.data_root / TEST_FILENAMES[name], starts)
            item["recorded_action_replay"] = {
                "n": len(rows),
                "pass_rate": float(np.mean([row["replay_goal_reached"] for row in rows])),
                "object_pass_rate": float(np.mean([row["replay_object_reached"] for row in rows])),
                "median_errors": {
                    key: float(np.median([row[key] for row in rows])) for key in ERROR_NAMES
                },
            }
        write_json(audit_path, audit)
    elif args.command == "stage-b":
        run_stage_b(args.results, args.data_root, args.checkpoint, args.anygoal_root)
    elif args.command == "report":
        write_figures(args.results, args.docs)
        write_report(args.results, args.docs)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
