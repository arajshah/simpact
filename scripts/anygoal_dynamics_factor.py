"""Separate contact mechanics, action duration, and goal identity for Extract-Peg.

The corrected diagnostic scored one dynamics setting: a hand-peg weld and 25
physics substeps. This script keeps the same 50 episodes, the same 16-sequence
bank, the same initial observations, and the same recorded goal, and changes
only the integrator. Numerical helpers do not import MuJoCo or torch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit

PRIMARY_CONDITIONS = (
    "evaluator_10",
    "noweld_25",
    "weld_10",
    "weld_25",
)
CONFOUND_CONDITIONS = (
    "extract_noweld_10",
    "extract_noweld_25",
)
SOLVABILITY_FLOOR = 0.80


@dataclass(frozen=True)
class DynamicsCondition:
    """One physically specified integrator.

    ``base`` selects the MuJoCo model. Insert-Peg has no weld equality and no
    extract-specific slot contact edits. Extract-Peg has both. ``use_env_step``
    is the published evaluator path (dm_control's 10-substep control step).
    """

    name: str
    weld: bool
    nstep: int
    base: str
    use_env_step: bool
    zero_velocity: bool
    role: str
    label: str

    def __post_init__(self) -> None:
        if self.base not in {"insert", "extract"}:
            raise ValueError(f"Unknown physics base {self.base!r}.")
        if self.nstep <= 0:
            raise ValueError("nstep must be positive.")
        if self.weld and self.base != "extract":
            raise ValueError("The insert model has no grasp weld.")
        if self.use_env_step and (self.weld or self.nstep != 10 or self.zero_velocity):
            raise ValueError("env.step is only the published no-weld 10-substep integrator.")
        if self.zero_velocity and not (self.weld and self.nstep == 25):
            raise ValueError("The velocity ablation is defined on the corrected weld/25 dynamics.")


CONDITIONS: dict[str, DynamicsCondition] = {
    item.name: item
    for item in (
        DynamicsCondition(
            "evaluator_10", False, 10, "insert", True, False,
            "training_evaluator", "No weld, 10 substeps (training/evaluator)",
        ),
        DynamicsCondition(
            "noweld_25", False, 25, "insert", False, False,
            "duration", "No weld, 25 substeps",
        ),
        DynamicsCondition(
            "weld_10", True, 10, "extract", False, False,
            "contact", "Weld, 10 substeps",
        ),
        DynamicsCondition(
            "weld_25", True, 25, "extract", False, False,
            "corrected_extract", "Weld, 25 substeps (corrected Extract-Peg)",
        ),
        DynamicsCondition(
            "extract_noweld_10", False, 10, "extract", False, False,
            "contact_confound", "Extract slot contacts, no weld, 10 substeps",
        ),
        DynamicsCondition(
            "extract_noweld_25", False, 25, "extract", False, False,
            "contact_confound", "Extract slot contacts, no weld, 25 substeps",
        ),
        DynamicsCondition(
            "weld_25_zerovel", True, 25, "extract", False, True,
            "velocity_ablation", "Weld, 25 substeps, initial velocity zeroed",
        ),
    )
}


def condition_by_name(name: str) -> DynamicsCondition:
    try:
        return CONDITIONS[name]
    except KeyError as error:
        known = ", ".join(CONDITIONS)
        raise KeyError(f"Unknown dynamics condition {name!r}. Known: {known}.") from error


def per_episode_scores(success: np.ndarray, selected: np.ndarray) -> dict[str, np.ndarray]:
    """Opportunity regret for one selected index per episode.

    Regret is zero when the bank contains no success. That zero is not evidence
    that the scorer succeeded, so callers must keep solvability beside it.
    """
    success = np.asarray(success, dtype=bool)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if success.ndim != 2 or success.shape[1] != audit.N_CANDIDATES:
        raise ValueError(f"Success must have shape (episodes, {audit.N_CANDIDATES}).")
    if selected.shape[0] != success.shape[0]:
        raise ValueError("Selection and success episode counts differ.")
    if np.any((selected < 0) | (selected >= audit.N_CANDIDATES)):
        raise ValueError("Selected index is outside the bank.")
    solvable = success.any(axis=1)
    chosen = success[np.arange(success.shape[0]), selected]
    regret = solvable.astype(np.float64) - chosen.astype(np.float64)
    return {"solvable": solvable, "chosen_success": chosen, "regret": regret}


def conditional_bootstrap(values: np.ndarray, mask: np.ndarray, *, seed: int) -> dict:
    """Bootstrap a mean on the masked episodes. An empty mask is undefined."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("Values and mask must have the same length.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap values must be finite.")
    if int(mask.sum()) == 0:
        return {
            "n": 0,
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "defined": False,
            "reason": "undefined: the mask selects no episodes",
        }
    stats = audit.bootstrap_mean(values[mask], seed=seed)
    stats["defined"] = True
    return stats


def regret_is_comparable(solvability: float, *, minimum: float = SOLVABILITY_FLOOR) -> bool:
    """The memorandum ranks actions only when the bank is usually solvable."""
    solvability = float(solvability)
    if not np.isfinite(solvability) or not 0.0 <= solvability <= 1.0:
        raise ValueError("Solvability must be a finite rate in [0, 1].")
    return bool(solvability >= minimum)


def latent_sse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sum of squared error over the embedding axis. Matches JEPA terminal cost."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if right.shape != left.shape:
        try:
            right = np.broadcast_to(right, left.shape)
        except ValueError as error:
            raise ValueError(f"Cannot broadcast {right.shape} to {left.shape}.") from error
    delta = left - right
    if not np.all(np.isfinite(delta)):
        raise ValueError("Latent vectors must be finite.")
    return np.sum(delta * delta, axis=-1)


def alternate_candidate(object_distance_m: np.ndarray, within_reference: np.ndarray) -> int | None:
    """Farthest realized terminal that is not already the reference goal.

    Candidate 0 is the reference outcome. Returns None when every other
    terminal is inside the dataset tolerance of that outcome, because the two
    goals would not be distinct.
    """
    distance = np.asarray(object_distance_m, dtype=np.float64).reshape(audit.N_CANDIDATES)
    within = np.asarray(within_reference, dtype=bool).reshape(audit.N_CANDIDATES)
    if not np.all(np.isfinite(distance)):
        raise ValueError("Object distances must be finite.")
    eligible = ~within
    eligible[0] = False
    if not bool(eligible.any()):
        return None
    scores = np.where(eligible, distance, -np.inf)
    return int(np.argmax(scores))


def _prediction_failure(row: dict) -> bool:
    return bool(
        row["solvability"] >= SOLVABILITY_FLOOR
        and row["p_regret_defined"]
        and row["p_regret_mean"] is not None
        and row["p_regret_mean"] >= 0.20
        and row["p_regret_ci_low"] is not None
        and row["p_regret_ci_low"] > 0.10
        and row["reduction_mean"] is not None
        and row["reduction_mean"] >= 0.15
        and row["reduction_ci_low"] is not None
        and row["reduction_ci_low"] > 0.0
    )


def interpret_dynamics(conditions: dict) -> dict:
    """Decide whether the corrected-bank failure survives the training integrator.

    ``BASELINES`` means a prediction-shaped gap remains where the goal is
    bank-solvable under the training integrator. That still does not authorize
    a new learning objective. ``NO-GO`` means the gap is carried by a
    transition or a goal the training dynamics do not make solvable.
    """
    for name in PRIMARY_CONDITIONS:
        if name not in conditions:
            raise KeyError(f"Missing condition {name}.")
    corrected = conditions["weld_25"]
    training = conditions["evaluator_10"]
    corrected_failure = _prediction_failure(corrected)
    training_can_rank = regret_is_comparable(training["solvability"])
    training_failure = _prediction_failure(training)
    if not corrected_failure:
        decision = "NO-GO"
        reason = (
            "The corrected weld and 25-substep bank does not meet the "
            "pre-registered prediction-failure gates, so there is no residual "
            "Extract-Peg ranking failure to explain."
        )
    elif training_failure:
        decision = "BASELINES"
        reason = (
            "Predicted-latent opportunity regret stays high, and observed-latent "
            "scoring removes it, when the same goal and action bank are executed "
            "with the training integrator. The weld and the longer action do not "
            "by themselves explain the failure. Standard baselines are required. "
            "This is not a decision to invent a new objective."
        )
    elif not training_can_rank:
        decision = "NO-GO"
        reason = (
            "The fixed Extract-Peg goal is bank-solvable under the welded "
            "25-substep dynamics. Under the training integrator, bank solvability "
            f"is {float(training['solvability']):.3f}, below the 0.80 floor, so "
            "opportunity regret there is not a ranking result. The demonstrated "
            "decision failure sits on a transition the Insert-Peg training data "
            "do not support."
        )
    else:
        decision = "NO-GO"
        reason = (
            "The same goal and action bank do not show a prediction-shaped "
            "regret gap under the training integrator. The gap appears with the "
            "welded and longer-action transition."
        )
    return {
        "decision": decision,
        "corrected_prediction_failure": corrected_failure,
        "training_prediction_failure": training_failure,
        "training_bank_can_rank": training_can_rank,
        "reason": reason,
    }


def interpret_baselines(dynamics_decision: str, oracle_regret: float, deployable_closes: bool | None) -> dict:
    """Apply the memorandum's stop rule after the dynamics isolation.

    An oracle that knows the weld is a ceiling, not an equal-information
    baseline. ``deployable_closes`` is true only when a baseline limited to
    training information removes at least 80% of the corrected regret gap.
    """
    if dynamics_decision not in {"NO-GO", "BASELINES"}:
        raise ValueError(f"Unexpected dynamics decision {dynamics_decision!r}.")
    if not np.isfinite(oracle_regret) or oracle_regret < -1e-9:
        raise ValueError("Oracle regret must be a finite non-negative rate.")
    if dynamics_decision == "NO-GO":
        return {
            "decision": "NO-GO",
            "reason": (
                "No further learning objective is indicated. The ranking gap does "
                "not survive a controlled training-dynamics comparison, or the "
                "fixed goal is not solvable there. "
                f"A privileged simulator that uses the corrected dynamics has "
                f"opportunity regret {oracle_regret:.3f} on that bank."
            ),
        }
    if oracle_regret > 0.05:
        return {
            "decision": "NO-GO",
            "reason": (
                "The privileged true-state simulator does not rank this bank. "
                "The residual is not evidence that learned world models lack a "
                "dynamics other methods possess."
            ),
        }
    if deployable_closes is True:
        return {
            "decision": "NO-GO",
            "reason": (
                "A baseline restricted to the training information closes at least "
                "80% of the corrected regret gap. Existing methods account for the "
                "failure."
            ),
        }
    if deployable_closes is None:
        return {
            "decision": "NO-GO",
            "reason": (
                "No fair deployable baseline could be fit well enough to support "
                "a claim that standard methods fail. The privileged physics "
                "ceiling still ranks the corrected bank. Do not treat the missing "
                "baseline as a new-objective result."
            ),
        }
    return {
        "decision": "DIAGNOSIS",
        "reason": (
            "A prediction-shaped failure remains under the training integrator, "
            "the privileged physics ceiling ranks the bank, and the available "
            "equal-information baselines do not close 80% of the gap. Further "
            "diagnosis is warranted. This does not by itself justify a new method."
        ),
    }


def zero_velocity_state(state: np.ndarray) -> np.ndarray:
    """Drop arm and object velocity. Positions and the target are unchanged."""
    state = np.asarray(state, dtype=np.float64).copy()
    if state.shape != (audit.STATE_DIM,):
        raise ValueError(f"State must have shape {(audit.STATE_DIM,)}, got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("State must be finite.")
    state[16:24] = 0.0
    state[37:40] = 0.0
    return state


def _fmt_rate(stats: dict) -> str:
    if not stats.get("defined", True) or stats.get("mean") is None:
        return "undefined"
    if "ci_low" in stats and stats["ci_low"] is not None:
        return f"{stats['mean']:.3f} [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"
    return f"{stats['mean']:.3f}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=audit._json_default) + "\n")


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _prepare_bank(data_root: Path, output: Path, *, source: str, holdout_scored: bool) -> dict:
    if source not in {"extract", "insert"}:
        raise ValueError("source must be extract or insert.")
    if source == "extract":
        path = data_root / "extract_peg_test.h5"
    else:
        path = data_root / audit.TRAIN_FILENAME
    train_path = data_root / audit.TRAIN_FILENAME
    records = audit.index_h5_episodes(path)
    chosen, excluded = audit.select_extract_episodes(records, start_index=0)
    train_records = audit.index_h5_episodes(train_path)
    scored_ids = {int(record["episode_id"]) for record in chosen}
    if holdout_scored:
        train_records = [
            record for record in train_records if int(record["episode_id"]) not in scored_ids
        ]
        if len(train_records) < 10:
            raise RuntimeError("Holding out the scored episodes removed the training pool.")
    train_actions, action_mean, action_scale = audit.load_training_actions(train_path, train_records)
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")
    from anygoal.envs.insert_peg.env import InsertPegEnv

    env = InsertPegEnv(render_mode="rgb_array", width=224, height=224, terminate_at_goal=False)
    try:
        low = np.asarray(env.action_space.low, dtype=np.float64)
        high = np.asarray(env.action_space.high, dtype=np.float64)
        physics_timestep = float(env.physics.timestep())
    finally:
        env.close()
    banks = np.empty((len(chosen), audit.N_CANDIDATES, audit.HORIZON_STEPS, audit.ACTION_DIM), dtype=np.float32)
    starts = np.empty((len(chosen), audit.STATE_DIM), dtype=np.float32)
    goals = np.empty_like(starts)
    start_images = np.empty((len(chosen), 224, 224, 3), dtype=np.uint8)
    goal_images = np.empty_like(start_images)
    next_states = np.empty_like(starts)
    rng = np.random.default_rng(audit.BANK_SEED)
    handle = audit.open_h5(path)
    try:
        for index, record in enumerate(chosen):
            window = audit.read_rows(
                handle,
                record["offset"],
                0,
                audit.HORIZON_STEPS + 1,
                ("action", "observation", "pixels"),
            )
            recorded = np.asarray(window["action"][: audit.HORIZON_STEPS], dtype=np.float64)
            states = np.asarray(window["observation"], dtype=np.float64)
            bank, _meta = audit.build_action_bank(
                recorded, train_actions, low, high, action_scale, rng
            )
            banks[index] = bank
            starts[index] = states[0]
            goals[index] = states[audit.HORIZON_STEPS]
            next_states[index] = states[1]
            start_images[index] = audit.as_hwc_uint8(window["pixels"][0])
            goal_images[index] = audit.as_hwc_uint8(window["pixels"][audit.HORIZON_STEPS])
    finally:
        handle.close()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output / "bank.npz",
        actions=banks,
        start_state=starts,
        goal_state=goals,
        next_state=next_states,
        start_image=start_images,
        goal_image=goal_images,
        episode_id=np.asarray([int(record["episode_id"]) for record in chosen], dtype=np.int64),
        action_mean=action_mean.astype(np.float64),
        action_scale=action_scale.astype(np.float64),
        action_low=low,
        action_high=high,
    )
    manifest = {
        "source": source,
        "n_episodes": len(chosen),
        "n_excluded_short": len(excluded),
        "holdout_scored_from_train_chunks": holdout_scored,
        "physics_timestep_s": physics_timestep,
        "goal_definition": (
            "Recorded state and image 25 steps after the first stored frame. "
            "The same arrays are the goal under every dynamics condition."
        ),
        "bank": "candidate 0 recorded continuation; 1-7 perturbations; 8-15 training chunks",
        "bank_seed": audit.BANK_SEED,
    }
    write_json(output / "bank_manifest.json", manifest)
    return manifest


def _make_env(spec: DynamicsCondition):
    if spec.base == "insert":
        from anygoal.envs.insert_peg.env import InsertPegEnv

        return InsertPegEnv(render_mode="rgb_array", width=224, height=224, terminate_at_goal=False)
    from anygoal.envs.extract_peg.env import ExtractPegEnv

    return ExtractPegEnv(render_mode="rgb_array", width=224, height=224, terminate_at_goal=False)


def _set_weld(env, active: bool) -> None:
    from anygoal.envs.extract_peg.policy import set_grasp_constraint

    set_grasp_constraint(env.physics, bool(active))


def _native_reward(env) -> float:
    return float(env.dmc_env.task.get_reward(env.physics))


def _rollout_sequence(env, state, goal, actions, spec: DynamicsCondition):
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors, insert_peg_goal_reached

    state = zero_velocity_state(state) if spec.zero_velocity else np.asarray(state, dtype=np.float64)
    env.reset(seed=0, options={"state": state, "goal_state": goal})
    if spec.base == "extract":
        _set_weld(env, spec.weld)
    one_step = None
    ever = False
    for index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        if spec.use_env_step:
            env.step(action)
        else:
            clipped = np.clip(action, env.action_space.low, env.action_space.high)
            env.physics.set_control(np.asarray(clipped, dtype=np.float64))
            env.physics.step(nstep=int(spec.nstep))
        current = env.get_state().copy()
        if index == 0:
            one_step = current
        ever = ever or insert_peg_goal_reached(
            insert_peg_goal_errors(goal, current), audit.TOLERANCES
        )
    final = env.get_state().copy()
    image = audit.as_hwc_uint8(env.render())
    reward = _native_reward(env)
    return final, image, one_step, bool(ever), reward


def _errors_of(goal: np.ndarray, state: np.ndarray) -> np.ndarray:
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    errors = insert_peg_goal_errors(goal, state)
    return np.asarray([errors[name] for name in audit.ERROR_NAMES], dtype=np.float32)


def run_rollout(output: Path, name: str, *, recorded_only: bool, episodes: int | None) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    spec = condition_by_name(name)
    bank_path = output / "bank.npz"
    if not bank_path.is_file():
        raise FileNotFoundError(f"Missing {bank_path}. Run prepare first.")
    bank = np.load(bank_path)
    n_all = int(bank["actions"].shape[0])
    n_keep = n_all if episodes is None else min(int(episodes), n_all)
    if n_keep <= 0:
        raise ValueError("episodes must be positive.")
    env = _make_env(spec)
    started = time.perf_counter()
    try:
        low = np.asarray(env.action_space.low, dtype=np.float64)
        high = np.asarray(env.action_space.high, dtype=np.float64)
        if not np.allclose(low, bank["action_low"]) or not np.allclose(high, bank["action_high"]):
            raise RuntimeError(f"{name} action limits do not match the prepared bank.")
        physics_timestep = float(env.physics.timestep())
        n_candidates = 1 if recorded_only else audit.N_CANDIDATES
        finals = np.empty((n_keep, n_candidates, audit.STATE_DIM), dtype=np.float32)
        images = np.empty((n_keep, n_candidates, 224, 224, 3), dtype=np.uint8)
        errors = np.empty((n_keep, n_candidates, len(audit.ERROR_NAMES)), dtype=np.float32)
        ever = np.empty((n_keep, n_candidates), dtype=bool)
        native = np.empty((n_keep, n_candidates), dtype=np.float32)
        one_step_errors = np.empty((n_keep, len(audit.ERROR_NAMES)), dtype=np.float32)
        start_native = np.empty(n_keep, dtype=np.float32)
        goal_native = np.empty(n_keep, dtype=np.float32)
        for episode in range(n_keep):
            start = np.asarray(bank["start_state"][episode], dtype=np.float64)
            goal = np.asarray(bank["goal_state"][episode], dtype=np.float64)
            actions = np.asarray(bank["actions"][episode, :n_candidates], dtype=np.float64)
            probe = zero_velocity_state(start) if spec.zero_velocity else start
            env.reset(seed=0, options={"state": probe, "goal_state": goal})
            if spec.base == "extract":
                _set_weld(env, spec.weld)
            start_native[episode] = _native_reward(env)
            env.reset(seed=0, options={"state": goal, "goal_state": goal})
            goal_native[episode] = _native_reward(env)
            for candidate in range(n_candidates):
                final, image, one_step, hit, reward = _rollout_sequence(
                    env, start, goal, actions[candidate], spec
                )
                finals[episode, candidate] = final
                images[episode, candidate] = image
                errors[episode, candidate] = _errors_of(goal, final)
                ever[episode, candidate] = hit
                native[episode, candidate] = reward
                if candidate == 0:
                    one_step_errors[episode] = _errors_of(bank["next_state"][episode], one_step)
            print(
                f"{name} episode {int(bank['episode_id'][episode])}: "
                f"recorded_goal={bool(np.all(errors[episode, 0] <= _tolerance_vector()))}",
                flush=True,
            )
    finally:
        env.close()
    success = np.all(errors <= _tolerance_vector(), axis=-1)
    destination = output / ("recorded" if recorded_only else "bank") / f"{name}.npz"
    _atomic_npz(
        destination,
        final_state=finals,
        final_image=images,
        errors=errors,
        success=success,
        ever_success=ever,
        native_reward=native,
        one_step_errors=one_step_errors,
        start_native_reward=start_native,
        goal_native_reward=goal_native,
        episode_id=np.asarray(bank["episode_id"][:n_keep]),
    )
    summary = {
        "condition": asdict(spec),
        "recorded_only": recorded_only,
        "n_episodes": n_keep,
        "physics_timestep_s": physics_timestep,
        "action_duration_s": physics_timestep * int(spec.nstep),
        "recorded_goal_rate": float(success[:, 0].mean()),
        "recorded_object_error_median_m": float(np.median(errors[:, 0, 2])),
        "recorded_hand_error_median_m": float(np.median(errors[:, 0, 0])),
        "one_step_hand_median_m": float(np.median(one_step_errors[:, 0])),
        "one_step_object_median_m": float(np.median(one_step_errors[:, 2])),
        "seconds": time.perf_counter() - started,
    }
    if not recorded_only:
        summary["solvability"] = float(success.any(axis=1).mean())
        summary["native_insert_rate_best"] = float((native >= 1.0 - 1e-7).any(axis=1).mean())
    write_json(destination.with_suffix(".json"), summary)
    return summary


def _tolerance_vector() -> np.ndarray:
    return np.asarray([audit.TOLERANCES[name] for name in audit.ERROR_NAMES], dtype=np.float32)


def _load_model(checkpoint: Path, anygoal_root: Path):
    sys.path.insert(0, str(anygoal_root))
    import torch
    from anygoal.resources import resource_path
    from anygoal.runners.plan.planning_sensorimotor import image_transform, load_model
    from anygoal.sources import source_path

    if not torch.cuda.is_available():
        raise RuntimeError("SMWM scoring requires CUDA.")
    source = source_path("sensorimotor")
    planning = source / "planning" if (source / "planning" / "jepa.py").is_file() else source
    model = load_model(
        planning,
        planning,
        "cuda",
        train_config_path=resource_path("train", "sensorimotor", "insert_peg.yaml"),
        checkpoint_path=checkpoint,
    )
    return model, image_transform(224)


def _terminal_embeddings(model, transform, start_hwc, goal_hwc, final_hwc, blocked):
    """Native predicted cost plus terminal embeddings for prediction error."""
    import torch

    predicted, observed, mode = audit.score_bank(
        model, transform, start_hwc, goal_hwc, final_hwc, blocked
    )
    current = audit._policy_image(start_hwc, transform).to(next(model.parameters()).device)
    goal = audit._policy_image(goal_hwc, transform).to(current.device)
    finals = torch.cat(
        [audit._policy_image(frame, transform) for frame in final_hwc], dim=0
    ).to(current.device)
    actions = torch.as_tensor(blocked, dtype=current.dtype, device=current.device).unsqueeze(0)
    n_samples = actions.shape[1]
    info = {
        "pixels": current[:, None, None].expand(1, n_samples, 1, *current.shape[1:]).contiguous(),
        "goal": goal[:, None, None].expand(1, n_samples, 1, *goal.shape[1:]).contiguous(),
    }
    goal_encoded = model.encode({"pixels": goal})["emb"][:, -1, :]
    rolled = model.rollout({key: value for key, value in info.items()}, actions)
    predicted_terminal = rolled["predicted_emb"][0, :, -1, :]
    observed_terminal = model.encode({"pixels": finals})["emb"][:, -1, :]
    recomputed = latent_sse(
        predicted_terminal.detach().cpu().numpy(),
        goal_encoded.detach().cpu().numpy(),
    )
    if not np.allclose(recomputed, predicted, rtol=1e-4, atol=1e-4):
        mode = mode + "+prediction_path_disagrees"
    return (
        predicted,
        observed,
        predicted_terminal.detach().cpu().numpy().astype(np.float32),
        observed_terminal.detach().cpu().numpy().astype(np.float32),
        mode,
    )


def _score_episode(model, transform, bank, rollout, episode: int):
    actions = _as_bank(np.asarray(bank["actions"][episode], dtype=np.float64))
    blocked = audit.block_normalized_actions(
        actions,
        np.asarray(bank["action_mean"]),
        np.asarray(bank["action_scale"]),
    )
    return _terminal_embeddings(
        model,
        transform,
        bank["start_image"][episode],
        bank["goal_image"][episode],
        rollout["final_image"][episode],
        blocked,
    )


def _as_bank(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (audit.N_CANDIDATES, audit.HORIZON_STEPS, audit.ACTION_DIM):
        raise ValueError(f"Expected one full action bank, got {actions.shape}.")
    return actions


def run_score(output: Path, checkpoint: Path, anygoal_root: Path, names: list[str]) -> dict:  # noqa: F811
    os.environ.setdefault("MUJOCO_GL", "egl")
    bank = np.load(output / "bank.npz")
    model, transform = _load_model(checkpoint, anygoal_root)
    written = []
    for name in names:
        condition_by_name(name)
        path = output / "bank" / f"{name}.npz"
        rollout = np.load(path)
        n_keep = int(rollout["success"].shape[0])
        if int(bank["actions"].shape[0]) < n_keep:
            raise RuntimeError("Bank is shorter than the rollout.")
        predicted = np.empty((n_keep, audit.N_CANDIDATES), dtype=np.float32)
        observed = np.empty_like(predicted)
        pred_emb = np.empty((n_keep, audit.N_CANDIDATES, 0), dtype=np.float32)
        obs_emb = None
        mode = None
        for episode in range(n_keep):
            pred, obs, pred_terminal, obs_terminal, mode = _score_episode(
                model, transform, bank, rollout, episode
            )
            if obs_emb is None:
                width = int(pred_terminal.shape[-1])
                pred_emb = np.empty((n_keep, audit.N_CANDIDATES, width), dtype=np.float32)
                obs_emb = np.empty_like(pred_emb)
            predicted[episode] = pred
            observed[episode] = obs
            pred_emb[episode] = pred_terminal
            obs_emb[episode] = obs_terminal
            print(f"scored {name} episode {int(rollout['episode_id'][episode])}", flush=True)
        if "prediction_path_disagrees" in str(mode):
            raise RuntimeError(f"{name} native cost and rollout cost disagree: {mode}")
        destination = output / "scores" / f"{name}.npz"
        _atomic_npz(
            destination,
            predicted_cost=predicted,
            observed_cost=observed,
            predicted_emb=pred_emb,
            observed_emb=obs_emb,
            episode_id=np.asarray(rollout["episode_id"]),
        )
        written.append({"condition": name, "cost_mode": mode, "n": n_keep})
    write_json(output / "scores" / "manifest.json", {"conditions": written})
    return {"conditions": written}


def _row_from_arrays(name: str, success, predicted, observed, pred_error, errors, native, one_step) -> dict:
    p_index = np.argmin(predicted, axis=1).astype(np.int64)
    o_index = np.argmin(observed, axis=1).astype(np.int64)
    p_scores = per_episode_scores(success, p_index)
    o_scores = per_episode_scores(success, o_index)
    solvable = p_scores["solvable"]
    p_regret = conditional_bootstrap(p_scores["regret"], solvable, seed=audit.BOOTSTRAP_SEED)
    o_regret = conditional_bootstrap(o_scores["regret"], solvable, seed=audit.BOOTSTRAP_SEED + 1)
    both = conditional_bootstrap(
        p_scores["regret"] - o_scores["regret"], solvable, seed=audit.BOOTSTRAP_SEED + 2
    )
    unconditional_p = audit.bootstrap_mean(p_scores["regret"], seed=audit.BOOTSTRAP_SEED + 3)
    best_object = errors[:, :, 2].min(axis=1)
    recorded_object = errors[:, 0, 2]
    return {
        "condition": name,
        "label": CONDITIONS[name].label,
        "n": int(success.shape[0]),
        "solvability": float(solvable.mean()),
        "ranking_comparable": regret_is_comparable(float(solvable.mean())),
        "recorded_goal_rate": float(success[:, 0].mean()),
        "recorded_ever_rate": None,
        "p_success_rate": float(p_scores["chosen_success"].mean()),
        "o_success_rate": float(o_scores["chosen_success"].mean()),
        "p_regret_unconditional": unconditional_p,
        "p_regret_defined": bool(p_regret["defined"]),
        "p_regret_mean": p_regret["mean"],
        "p_regret_ci_low": p_regret["ci_low"],
        "p_regret_ci_high": p_regret["ci_high"],
        "p_regret_n": p_regret["n"],
        "o_regret_mean": o_regret["mean"],
        "o_regret_ci_low": o_regret["ci_low"],
        "o_regret_ci_high": o_regret["ci_high"],
        "reduction_mean": both["mean"],
        "reduction_ci_low": both["ci_low"],
        "reduction_ci_high": both["ci_high"],
        "prediction_error_mean": float(np.mean(pred_error)),
        "prediction_error_recorded_mean": float(np.mean(pred_error[:, 0])),
        "recorded_object_error_median_m": float(np.median(recorded_object)),
        "recorded_hand_error_median_m": float(np.median(errors[:, 0, 0])),
        "best_object_error_median_m": float(np.median(best_object)),
        "one_step_object_median_m": float(np.median(one_step[:, 2])),
        "one_step_hand_median_m": float(np.median(one_step[:, 0])),
        "native_insert_recorded_rate": float(np.mean(native[:, 0] >= 1.0 - 1e-7)),
        "native_insert_bank_rate": float(np.mean(np.any(native >= 1.0 - 1e-7, axis=1))),
        "p_index": p_index,
        "o_index": o_index,
        "p_regret_episode": p_scores["regret"],
        "o_regret_episode": o_scores["regret"],
        "solvable_episode": solvable,
        "prediction_error_episode": np.mean(pred_error, axis=1),
    }


def _paired_goal_rows(states, observed_emb, predicted_emb) -> list[dict]:
    rows = []
    for episode in range(states.shape[0]):
        reference = states[episode, 0]
        within = np.array(
            [
                audit.full_success(audit.goal_errors(reference, states[episode, candidate]))
                for candidate in range(audit.N_CANDIDATES)
            ]
        )
        distance = np.linalg.norm(states[episode, :, 33:35] - reference[33:35], axis=1)
        alternate = alternate_candidate(distance, within)
        if alternate is None:
            rows.append(
                {
                    "episode": episode,
                    "distinct": False,
                    "alternate": -1,
                    "object_distance_m": 0.0,
                    "goal_embedding_sse": 0.0,
                    "reference_p_regret": 0,
                    "alternate_p_regret": 0,
                    "reference_o_regret": 0,
                    "alternate_o_regret": 0,
                    "reference_p_index": -1,
                    "alternate_p_index": -1,
                    "reference_solvable": False,
                    "alternate_solvable": False,
                    "p_index_changes": 0,
                }
            )
            continue
        goals = {"reference": 0, "alternate": int(alternate)}
        entry = {
            "episode": episode,
            "distinct": True,
            "alternate": int(alternate),
            "object_distance_m": float(distance[alternate]),
            "goal_embedding_sse": float(latent_sse(observed_emb[episode, 0], observed_emb[episode, alternate])),
        }
        for goal_name, goal_index in goals.items():
            success = np.array(
                [
                    audit.full_success(audit.goal_errors(states[episode, goal_index], states[episode, candidate]))
                    for candidate in range(audit.N_CANDIDATES)
                ]
            )
            p_cost = latent_sse(predicted_emb[episode], observed_emb[episode, goal_index])
            o_cost = latent_sse(observed_emb[episode], observed_emb[episode, goal_index])
            p_index = int(np.argmin(p_cost))
            o_index = int(np.argmin(o_cost))
            entry[f"{goal_name}_p_regret"] = audit.opportunity_regret(success, p_index)
            entry[f"{goal_name}_o_regret"] = audit.opportunity_regret(success, o_index)
            entry[f"{goal_name}_p_index"] = p_index
            entry[f"{goal_name}_solvable"] = bool(success.any())
        entry["p_index_changes"] = int(entry["reference_p_index"] != entry["alternate_p_index"])
        rows.append(entry)
    return rows


def _goal_probe_summary(rows: list[dict]) -> dict:
    distinct = [row for row in rows if row["distinct"]]
    if not distinct:
        return {
            "n_distinct_pairs": 0,
            "reason": (
                "No episode has a second bank outcome outside the dataset tolerance "
                "of the recorded outcome. A goal-only comparison is not supported."
            ),
        }
    def mean_ci(key):
        return audit.bootstrap_mean(
            np.asarray([row[key] for row in distinct], dtype=np.float64),
            seed=audit.BOOTSTRAP_SEED + 4,
        )
    return {
        "n_distinct_pairs": len(distinct),
        "n_episodes": len(rows),
        "dynamics": "weld_25",
        "definition": (
            "Both goals are realized terminals of the same start, the same weld/25 "
            "integrator, and the same action bank. The reference is candidate 0. "
            "The alternate is the farthest other terminal outside the dataset tolerance."
        ),
        "reference_p_regret": mean_ci("reference_p_regret"),
        "alternate_p_regret": mean_ci("alternate_p_regret"),
        "reference_o_regret": mean_ci("reference_o_regret"),
        "alternate_o_regret": mean_ci("alternate_o_regret"),
        "p_index_change_rate": mean_ci("p_index_changes"),
        "goal_embedding_sse_median": float(np.median([row["goal_embedding_sse"] for row in distinct])),
        "not_a_benchmark_insert_goal": (
            "These pairs do not replace Insert-Peg versus Extract-Peg. Those "
            "benchmark goals were generated under different integrators."
        ),
    }


def summarize(output: Path, docs: Path) -> dict:
    names = [name for name in (*PRIMARY_CONDITIONS, *CONFOUND_CONDITIONS, "weld_25_zerovel")
             if (output / "scores" / f"{name}.npz").is_file() or (output / "bank" / f"{name}.npz").is_file()]
    rows = []
    cache = {}
    for name in names:
        rollout_path = output / "bank" / f"{name}.npz"
        score_path = output / "scores" / f"{name}.npz"
        if not rollout_path.is_file():
            continue
        rollout = np.load(rollout_path)
        if not score_path.is_file():
            continue
        score = np.load(score_path)
        pred_error = latent_sse(score["predicted_emb"], score["observed_emb"])
        row = _row_from_arrays(
            name,
            rollout["success"],
            score["predicted_cost"],
            score["observed_cost"],
            pred_error,
            rollout["errors"],
            rollout["native_reward"],
            rollout["one_step_errors"],
        )
        row["recorded_ever_rate"] = float(rollout["ever_success"][:, 0].mean())
        row["start_native_insert_rate"] = float(np.mean(rollout["start_native_reward"] >= 1.0 - 1e-7))
        row["goal_native_insert_rate"] = float(np.mean(rollout["goal_native_reward"] >= 1.0 - 1e-7))
        row["action_duration_s"] = json.loads((output / "bank" / f"{name}.json").read_text()).get(
            "action_duration_s"
        ) if (output / "bank" / f"{name}.json").is_file() else None
        cache[name] = row
        rows.append(row)
    primary = {name: cache[name] for name in PRIMARY_CONDITIONS if name in cache}
    interpretation = interpret_dynamics(primary) if len(primary) == len(PRIMARY_CONDITIONS) else None
    goal_summary = None
    if "weld_25" in cache and (output / "scores" / "weld_25.npz").is_file():
        score = np.load(output / "scores" / "weld_25.npz")
        rollout = np.load(output / "bank" / "weld_25.npz")
        goal_rows = _paired_goal_rows(rollout["final_state"], score["observed_emb"], score["predicted_emb"])
        goal_summary = _goal_probe_summary(goal_rows)
        audit._write_csv(
            output / "goal_pairs.csv",
            [{key: value for key, value in row.items()} for row in goal_rows],
        )
    paired = {}
    if "weld_25" in cache and "evaluator_10" in cache:
        paired["prediction_error_weld25_minus_evaluator10"] = audit.bootstrap_mean(
            cache["weld_25"]["prediction_error_episode"] - cache["evaluator_10"]["prediction_error_episode"],
            seed=audit.BOOTSTRAP_SEED + 5,
        )
        paired["recorded_goal_weld25_minus_evaluator10"] = audit.bootstrap_mean(
            np.load(output / "bank" / "weld_25.npz")["success"][:, 0].astype(np.float64)
            - np.load(output / "bank" / "evaluator_10.npz")["success"][:, 0].astype(np.float64),
            seed=audit.BOOTSTRAP_SEED + 6,
        )
    if "weld_25" in cache and "weld_10" in cache:
        paired["prediction_error_weld25_minus_weld10"] = audit.bootstrap_mean(
            cache["weld_25"]["prediction_error_episode"] - cache["weld_10"]["prediction_error_episode"],
            seed=audit.BOOTSTRAP_SEED + 7,
        )
    if "weld_25" in cache and "extract_noweld_25" in cache:
        paired["prediction_error_weld25_minus_extract_noweld25"] = audit.bootstrap_mean(
            cache["weld_25"]["prediction_error_episode"] - cache["extract_noweld_25"]["prediction_error_episode"],
            seed=audit.BOOTSTRAP_SEED + 8,
        )
    oracle = None
    if "weld_25" in cache:
        success = np.load(output / "bank" / "weld_25.npz")["success"]
        errors = np.load(output / "bank" / "weld_25.npz")["errors"]
        chosen = []
        for episode in range(success.shape[0]):
            index, _n = audit.select_true_utility(
                success[episode],
                np.asarray([audit.normalized_error_sum(
                    dict(zip(audit.ERROR_NAMES, errors[episode, candidate]))
                ) for candidate in range(audit.N_CANDIDATES)]),
            )
            chosen.append(index)
        scores = per_episode_scores(success, np.asarray(chosen))
        oracle = {
            "information": "privileged true state, known weld, 25 substeps",
            "regret_mean": float(scores["regret"].mean()),
            "success_rate": float(scores["chosen_success"].mean()),
            "note": (
                "Selecting the candidate with the best true outcome has zero "
                "opportunity regret on every solvable episode by construction."
            ),
        }
    public_rows = []
    for row in rows:
        public = {
            key: value for key, value in row.items()
            if not isinstance(value, np.ndarray)
        }
        public_rows.append(public)
    payload = {
        "conditions": public_rows,
        "paired": paired,
        "interpretation": interpretation,
        "goal_probe": goal_summary,
        "oracle_corrected_dynamics": oracle,
        "fixed": {
            "episodes": "first 50 Extract-Peg episodes, start at the first stored frame",
            "bank_seed": audit.BANK_SEED,
            "goal": "recorded 25-step state and image",
            "model": "sensorimotor_insertpeg.ckpt",
        },
    }
    write_json(output / "summary.json", payload)
    audit._write_csv(output / "conditions.csv", [
        {
            "condition": row["condition"],
            "solvability": row["solvability"],
            "recorded_goal_rate": row["recorded_goal_rate"],
            "p_success_rate": row["p_success_rate"],
            "o_success_rate": row["o_success_rate"],
            "p_regret_conditional": row["p_regret_mean"],
            "o_regret_conditional": row["o_regret_mean"],
            "prediction_error_mean": row["prediction_error_mean"],
            "recorded_object_error_median_m": row["recorded_object_error_median_m"],
            "best_object_error_median_m": row["best_object_error_median_m"],
            "native_insert_bank_rate": row["native_insert_bank_rate"],
            "ranking_comparable": row["ranking_comparable"],
        }
        for row in public_rows
    ])
    _write_report(docs, payload)
    _write_figure(docs, public_rows)
    return payload


def _write_report(docs: Path, payload: dict) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Extract-Peg dynamics factorial",
        "",
        "Model weights, the 16-sequence bank, the initial state and image, and the recorded 25-step goal are fixed. Only the integrator changes.",
        "",
        "Conditional opportunity regret is computed on episodes whose bank contains a success. It is undefined when that set is empty, and it is not a cross-condition ranking comparison when solvability is below 0.80.",
        "",
        "| Condition | Solvability | Recorded goal | P success | O success | P regret given solvable | O regret given solvable | Latent prediction SSE | Recorded object error (m) | Best object error (m) | Native insert in bank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["conditions"]:
        p_regret = "undefined" if row["p_regret_mean"] is None else (
            f"{row['p_regret_mean']:.3f} [{row['p_regret_ci_low']:.3f}, {row['p_regret_ci_high']:.3f}]"
        )
        o_regret = "undefined" if row["o_regret_mean"] is None else (
            f"{row['o_regret_mean']:.3f} [{row['o_regret_ci_low']:.3f}, {row['o_regret_ci_high']:.3f}]"
        )
        lines.append(
            f"| {row['label']} | {row['solvability']:.3f} | {row['recorded_goal_rate']:.3f} | "
            f"{row['p_success_rate']:.3f} | {row['o_success_rate']:.3f} | {p_regret} | {o_regret} | "
            f"{row['prediction_error_mean']:.4f} | {row['recorded_object_error_median_m']:.4f} | "
            f"{row['best_object_error_median_m']:.4f} | {row['native_insert_bank_rate']:.3f} |"
        )
    lines.extend(["", "## Paired differences", ""])
    for name, stats in payload["paired"].items():
        lines.append(f"- {name}: {_fmt_rate(stats)}")
    if payload.get("interpretation"):
        lines.extend(["", "## Dynamics decision", "", f"**{payload['interpretation']['decision']}.** {payload['interpretation']['reason']}", "",
            "This decision concerns the fixed Extract-Peg goal only. A low unconditional regret under the training integrator is not a ranking success when solvability is below 0.80. Whether the same model ranks goals generated by that integrator is a separate insert-anchor measurement and is not in this table."])
    if payload.get("goal_probe"):
        lines.extend(["", "## Goal-only probe", "", "```json", json.dumps(payload["goal_probe"], indent=2, default=audit._json_default), "```"])
    if payload.get("oracle_corrected_dynamics"):
        oracle = payload["oracle_corrected_dynamics"]
        lines.extend([
            "",
            "## Privileged physics ceiling",
            "",
            f"Information: {oracle['information']}. Opportunity regret {oracle['regret_mean']:.3f}. "
            f"Selected-candidate success {oracle['success_rate']:.3f}. {oracle['note']}",
        ])
    (docs / "summary.md").write_text("\n".join(lines) + "\n")


def _write_figure(docs: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in rows if row["condition"] in PRIMARY_CONDITIONS]
    if not primary:
        return
    labels = [row["condition"] for row in primary]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _bars(axes[0], labels, [row["solvability"] for row in primary], "Bank solvability", "Fixed goal reachable in the bank")
    _bars(
        axes[1],
        labels,
        [row["prediction_error_mean"] for row in primary],
        "Mean terminal latent SSE",
        "SMWM prediction versus the rendered future",
    )
    figure.tight_layout()
    figure.savefig(docs / "dynamics_factor.png", dpi=140)
    plt.close(figure)


def _bars(ax, labels, values, ylabel, title) -> None:
    ax.bar(np.arange(len(labels)), values, color="#3c6e71")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--source", choices=("extract", "insert"), default="extract")
    prepare.add_argument("--holdout-scored", action="store_true")
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--condition", required=True)
    rollout.add_argument("--recorded-only", action="store_true")
    rollout.add_argument("--episodes", type=int, default=None)
    score = subparsers.add_parser("score")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--anygoal-root", type=Path, required=True)
    score.add_argument("--condition", action="append", required=True)
    report = subparsers.add_parser("summarize")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = _prepare_bank(
            args.data_root, args.output, source=args.source, holdout_scored=args.holdout_scored
        )
        print(json.dumps(manifest), flush=True)
    elif args.command == "rollout":
        summary = run_rollout(
            args.output, args.condition, recorded_only=args.recorded_only, episodes=args.episodes
        )
        print(json.dumps(summary, default=audit._json_default), flush=True)
    elif args.command == "score":
        summary = run_score(args.output, args.checkpoint, args.anygoal_root, args.condition)
        print(json.dumps(summary), flush=True)
    elif args.command == "summarize":
        summarize(args.output, args.docs)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
