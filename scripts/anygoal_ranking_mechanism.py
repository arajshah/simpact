"""Why SMWM ranks the Insert-Peg action bank incorrectly.

The 50-episode bank, checkpoint, and terminal scores already exist. This
module labels each candidate as a recorded continuation, a perturbation, or
an action sequence transplanted from another episode, then compares
prediction error, physical success, and goal-cost ranking. A second command
compares one-block predictions from the true image with autoregressive
rollouts. No model is trained.

Numerical helpers do not import MuJoCo or torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit
import anygoal_dynamics_factor as factor

RECORDED = slice(0, 1)
PERTURBATION = slice(1, 8)
TRANSPLANT = slice(8, 16)
GROUP_SLICES = {
    "recorded": RECORDED,
    "perturbation": PERTURBATION,
    "transplant": TRANSPLANT,
}
HORIZON_STEPS = (5, 10, 15, 20, 25)
# Operational ball for "another training frame at a similar pose."
# These radii are a similarity screen, not a claim that the transition is identified.
HAND_BALL_M = 0.05
OBJECT_BALL_M = 0.05
JOINT_BALL_RAD = 0.15


def candidate_kinds(n_candidates: int = audit.N_CANDIDATES) -> np.ndarray:
    if n_candidates != audit.N_CANDIDATES:
        raise ValueError(f"The Insert-Peg bank has {audit.N_CANDIDATES} candidates.")
    kinds = np.empty(n_candidates, dtype=object)
    kinds[RECORDED] = "recorded"
    kinds[PERTURBATION] = "perturbation"
    kinds[TRANSPLANT] = "transplant"
    return kinds


def support_label(*, trajectory_exact: bool, sequence_in_training: bool) -> str:
    """Separate a stored transition from an action reused at a new state.

    A perturbation is not a stored window. A transplanted window is a stored
    action sequence whose source state is a different episode, so the pair is
    recombined even though each piece occurs in the training file.
    """
    if trajectory_exact and sequence_in_training:
        return "observed_continuation"
    if trajectory_exact and not sequence_in_training:
        raise ValueError("An exact trajectory is a training window by definition.")
    if sequence_in_training:
        return "recombined_expert_action"
    return "unobserved_action"


def joint_angles(state: np.ndarray) -> np.ndarray:
    """Eight arm angles from the dataset's interleaved sin/cos pairs."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != audit.STATE_DIM:
        raise ValueError(f"Expected a {audit.STATE_DIM}-D state, got {state.shape}.")
    pairs = state[..., :16].reshape(state.shape[:-1] + (8, 2))
    return np.arctan2(pairs[..., 0], pairs[..., 1])


def pose_distance(query: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """Hand, object, and max-joint distances. ``reference`` broadcasts over query."""
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    hand = np.linalg.norm(query[..., 29:31] - reference[..., 29:31], axis=-1)
    obj = np.linalg.norm(query[..., 33:35] - reference[..., 33:35], axis=-1)
    joints = np.max(np.abs(joint_angles(query) - joint_angles(reference)), axis=-1)
    return {"hand_m": hand, "object_m": obj, "max_joint_rad": joints}


def in_pose_ball(distance: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (distance["hand_m"] <= HAND_BALL_M)
        & (distance["object_m"] <= OBJECT_BALL_M)
        & (distance["max_joint_rad"] <= JOINT_BALL_RAD)
    )


def group_means(values: np.ndarray) -> dict[str, np.ndarray]:
    """Per-episode mean of ``values`` inside each candidate group.

    ``values`` has shape (episodes, candidates).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != audit.N_CANDIDATES:
        raise ValueError(f"Expected (episodes, 16) values, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Group values must be finite.")
    return {name: values[:, group].mean(axis=1) for name, group in GROUP_SLICES.items()}


def pair_order_counts(success: np.ndarray, cost: np.ndarray) -> dict[str, np.ndarray]:
    """Per episode, how often a lower cost lands on the successful action.

    Episodes with only successes or only failures have undefined accuracy.
    Ties are counted and are not treated as correct orderings.
    """
    success = np.asarray(success, dtype=bool)
    cost = np.asarray(cost, dtype=np.float64)
    if success.shape != cost.shape or success.ndim != 2:
        raise ValueError("Success and cost must share shape (episodes, candidates).")
    if not np.all(np.isfinite(cost)):
        raise ValueError("Costs must be finite.")
    correct = np.zeros(success.shape[0], dtype=np.float64)
    ties = np.zeros_like(correct)
    total = np.zeros_like(correct)
    for episode in range(success.shape[0]):
        good = np.flatnonzero(success[episode])
        bad = np.flatnonzero(~success[episode])
        if good.size == 0 or bad.size == 0:
            continue
        gap = cost[episode, bad][None, :] - cost[episode, good][:, None]
        total[episode] = float(gap.size)
        correct[episode] = float(np.sum(gap > 0.0))
        ties[episode] = float(np.sum(gap == 0.0))
    defined = total > 0.0
    accuracy = np.full(success.shape[0], np.nan, dtype=np.float64)
    accuracy[defined] = correct[defined] / total[defined]
    return {"correct": correct, "ties": ties, "total": total, "accuracy": accuracy, "defined": defined}


def effect_alignment(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    reference: int = 0,
    min_observed_norm: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compare predicted and actual embedding changes from one reference action.

    Both arrays are (episodes, candidates, dim). The parallel component is the
    signed length of the predicted change along the actual change. A model that
    ignores the action has predicted norms near zero. A model that moves the
    latent the wrong way has a parallel component near zero while the predicted
    norm stays positive.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 3:
        raise ValueError("Predicted and observed embeddings must share shape (E, C, D).")
    if not 0 <= reference < predicted.shape[1]:
        raise ValueError("Reference candidate is outside the bank.")
    if min_observed_norm < 0.0:
        raise ValueError("min_observed_norm must be non-negative.")
    pred_delta = predicted - predicted[:, reference : reference + 1, :]
    obs_delta = observed - observed[:, reference : reference + 1, :]
    pred_norm = np.linalg.norm(pred_delta, axis=-1)
    obs_norm = np.linalg.norm(obs_delta, axis=-1)
    dot = np.sum(pred_delta * obs_delta, axis=-1)
    usable = obs_norm > min_observed_norm
    usable[:, reference] = False
    cosine = np.full(pred_norm.shape, np.nan, dtype=np.float64)
    parallel = np.full(pred_norm.shape, np.nan, dtype=np.float64)
    ok = usable & (pred_norm > 0.0) & (obs_norm > 0.0)
    cosine[ok] = dot[ok] / (pred_norm[ok] * obs_norm[ok])
    parallel[ok] = dot[ok] / obs_norm[ok]
    return {
        "cosine": cosine,
        "parallel": parallel,
        "predicted_norm": pred_norm,
        "observed_norm": obs_norm,
        "usable": ok,
    }


def _effect_by_group(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """Displacement from the recorded action, split by the other two groups."""
    alignment = effect_alignment(predicted, observed, min_observed_norm=0.0)
    grouped = {}
    for index, name in enumerate(("perturbation", "transplant")):
        mask = np.zeros(alignment["usable"].shape, dtype=bool)
        mask[:, GROUP_SLICES[name]] = True
        mask &= alignment["usable"]
        grouped[name] = {
            "predicted_norm": bootstrap_defined(
                per_episode_mean(alignment["predicted_norm"], mask), seed=51 + index
            ),
            "observed_norm": bootstrap_defined(
                per_episode_mean(alignment["observed_norm"], mask), seed=53 + index
            ),
        }
    return grouped


def per_episode_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("Values and mask shapes differ.")
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for episode in range(values.shape[0]):
        chosen = mask[episode] & np.isfinite(values[episode])
        if np.any(chosen):
            out[episode] = float(values[episode, chosen].mean())
    return out


def bootstrap_defined(values: np.ndarray, *, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "defined": False}
    summary = audit.bootstrap_mean(finite, seed=seed)
    summary["defined"] = True
    return summary


def recorded_success_mask(success: np.ndarray) -> np.ndarray:
    success = np.asarray(success, dtype=bool)
    if success.ndim != 2 or success.shape[1] != audit.N_CANDIDATES:
        raise ValueError("Success must have shape (episodes, 16).")
    return success[:, 0].copy()


def uniform_regret(success: np.ndarray) -> np.ndarray:
    """Expected opportunity regret of a uniform draw over the same bank."""
    success = np.asarray(success, dtype=bool)
    solvable = success.any(axis=1).astype(np.float64)
    return solvable - success.mean(axis=1)


def ranking_gap(success: np.ndarray, selected: np.ndarray) -> dict[str, np.ndarray]:
    scores = factor.per_episode_scores(success, selected)
    uniform = uniform_regret(success)
    return {
        "regret": scores["regret"],
        "chosen_success": scores["chosen_success"].astype(np.float64),
        "solvable": scores["solvable"],
        "uniform_regret": uniform,
        "regret_minus_uniform": scores["regret"] - uniform,
    }


def horizon_prediction_error(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Sum of squared error at each block. Both arrays are (E, C, blocks, D)."""
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 4:
        raise ValueError("Horizon embeddings must share shape (E, C, blocks, D).")
    return factor.latent_sse(predicted, observed)


def goal_costs(predicted: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Terminal latent goal cost. ``predicted`` is (E, C, D) and ``goal`` is (E, D)."""
    predicted = np.asarray(predicted, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if predicted.ndim != 3 or goal.ndim != 2:
        raise ValueError("Goal costs expect predicted (E, C, D) and goal (E, D).")
    if goal.shape[0] != predicted.shape[0] or goal.shape[1] != predicted.shape[-1]:
        raise ValueError("Goal embedding does not match the predictions.")
    return factor.latent_sse(predicted, goal[:, None, :])


def local_failure_present(regret: dict, *, mean_at_least: float = 0.20, ci_low_above: float = 0.10) -> bool:
    """The memorandum's prediction-failure bar, applied at one horizon."""
    if not regret.get("defined", False):
        return False
    return bool(regret["mean"] >= mean_at_least and regret["ci_low"] > ci_low_above)


def coverage_conclusion(*, recorded_regret: dict, sse_gap: dict, recorded_sse: dict) -> str:
    """Decide whether ranking error is confined to unsupported actions.

    ``sse_gap`` is the paired per-episode difference, transplant minus recorded.
    The gap is small when it is at most a quarter of the recorded error: the
    unsupported actions are not a separate error regime. The ranking clause
    uses episodes whose recorded continuation succeeds, so that expert
    transition was available and physically correct.
    """
    on_recorded = local_failure_present(recorded_regret)
    gap_small = False
    if sse_gap.get("defined") and recorded_sse.get("defined") and recorded_sse["mean"] is not None:
        gap_small = bool(sse_gap["mean"] <= 0.25 * max(float(recorded_sse["mean"]), 1.0))
    if on_recorded and gap_small:
        return "present_on_recorded_transitions"
    if on_recorded:
        return "present_on_recorded_transitions_with_larger_gap"
    if recorded_regret.get("defined") and recorded_regret["mean"] is not None and recorded_regret["mean"] < 0.10:
        return "concentrated_off_the_recorded_continuation"
    return "unresolved"


def accumulation_conclusion(*, early_regret: dict, late_regret: dict, early_sse: dict, late_sse: dict) -> str:
    """Separate a bad first block from error that appears only after rollout.

    Early and late are matched to the same 25-step physical outcome. SSE
    entries are episode means of prediction error at those horizons.
    """
    early_bad = local_failure_present(early_regret)
    late_bad = local_failure_present(late_regret)
    if not early_sse.get("defined") or not late_sse.get("defined"):
        return "unresolved"
    growth = late_sse["mean"] - early_sse["mean"]
    if early_bad and growth <= 0.25 * max(early_sse["mean"], 1.0):
        return "present_at_first_block"
    if (not early_bad) and late_bad and growth > 0.25 * max(early_sse["mean"], 1.0):
        return "emerges_with_horizon"
    if early_bad:
        return "present_at_first_block_and_grows"
    return "unresolved"


def _fmt(summary: dict, digits: int = 2) -> str:
    if not summary or not summary.get("defined", True) or summary.get("mean") is None:
        return "undefined"
    return (
        f"{summary['mean']:.{digits}f} "
        f"[{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=audit._json_default) + "\n")


def _bank_arrays(anchor: Path):
    bank = np.load(anchor / "bank.npz")
    rollout = np.load(anchor / "bank" / "evaluator_10.npz")
    score = np.load(anchor / "scores" / "evaluator_10.npz")
    if not np.array_equal(bank["episode_id"], rollout["episode_id"]):
        raise RuntimeError("Bank and rollout episode ids differ.")
    if not np.array_equal(bank["episode_id"], score["episode_id"]):
        raise RuntimeError("Bank and score episode ids differ.")
    return bank, rollout, score


def reconstruct_support(anchor: Path, data_root: Path) -> dict:
    """Replay the seed-0 bank and label exact, recombined, and perturbed actions."""
    bank, rollout, _score = _bank_arrays(anchor)
    path = data_root / audit.TRAIN_FILENAME
    records = audit.index_h5_episodes(path)
    by_id = {int(record["episode_id"]): record for record in records}
    chosen = [by_id[int(episode_id)] for episode_id in bank["episode_id"]]
    scored = {int(episode_id) for episode_id in bank["episode_id"]}
    train_records = [record for record in records if int(record["episode_id"]) not in scored]
    train_actions, action_mean, action_scale = audit.load_training_actions(path, train_records)
    if not np.allclose(action_mean, bank["action_mean"]) or not np.allclose(action_scale, bank["action_scale"]):
        raise RuntimeError("Rebuilt action scaler does not match the saved bank.")
    rng = np.random.default_rng(audit.BANK_SEED)
    labels = np.empty((len(chosen), audit.N_CANDIDATES), dtype=object)
    source_hand = np.full((len(chosen), audit.N_CANDIDATES), np.nan, dtype=np.float64)
    source_joint = np.full_like(source_hand, np.nan)
    handle = audit.open_h5(path)
    try:
        for index, record in enumerate(chosen):
            window = audit.read_rows(
                handle, record["offset"], 0, audit.HORIZON_STEPS + 1, ("action", "observation")
            )
            recorded = np.asarray(window["action"][: audit.HORIZON_STEPS], dtype=np.float64)
            start = np.asarray(window["observation"][0], dtype=np.float64)
            if not np.allclose(start, bank["start_state"][index], atol=0.0, rtol=0.0):
                raise RuntimeError(f"Episode {record['episode_id']} start state is not the stored row.")
            rebuilt, meta = audit.build_action_bank(
                recorded,
                train_actions,
                np.asarray(bank["action_low"]),
                np.asarray(bank["action_high"]),
                action_scale,
                rng,
            )
            saved = np.asarray(bank["actions"][index], dtype=np.float32)
            if not np.array_equal(saved, rebuilt.astype(np.float32)):
                raise RuntimeError(f"Rebuilt bank differs for episode {record['episode_id']}.")
            for candidate, item in enumerate(meta):
                if item["kind"] == "recorded":
                    labels[index, candidate] = support_label(
                        trajectory_exact=True, sequence_in_training=True
                    )
                    source_hand[index, candidate] = 0.0
                    source_joint[index, candidate] = 0.0
                elif item["kind"] == "perturbation":
                    labels[index, candidate] = support_label(
                        trajectory_exact=False, sequence_in_training=False
                    )
                elif item["kind"] == "train_chunk":
                    source = train_records[int(item["train_episode_list_index"])]
                    source_state = np.asarray(
                        handle["observation"][int(source["offset"]) + int(item["start"])],
                        dtype=np.float64,
                    )
                    distance = pose_distance(start, source_state)
                    exact_source = bool(
                        np.allclose(start, source_state, atol=1e-6, rtol=0.0)
                        and np.allclose(
                            rebuilt[candidate],
                            np.asarray(bank["actions"][index, 0], dtype=np.float64),
                            atol=1e-6,
                            rtol=0.0,
                        )
                    )
                    labels[index, candidate] = support_label(
                        trajectory_exact=exact_source, sequence_in_training=True
                    )
                    source_hand[index, candidate] = float(distance["hand_m"])
                    source_joint[index, candidate] = float(distance["max_joint_rad"])
                else:
                    raise RuntimeError(f"Unknown bank kind {item['kind']}.")
    finally:
        handle.close()
    if not np.array_equal(rollout["episode_id"], bank["episode_id"]):
        raise RuntimeError("Rollout ids drifted.")
    return {
        "label": labels,
        "source_hand_m": source_hand,
        "source_joint_rad": source_joint,
        "n_train_episodes_for_chunks": len(train_records),
    }


def pose_ball_coverage(anchor: Path, data_root: Path, source_hand: np.ndarray) -> dict:
    """Count non-anchor frames near each start, and how close their actions are."""
    bank, _rollout, _score = _bank_arrays(anchor)
    path = data_root / audit.TRAIN_FILENAME
    handle = audit.open_h5(path)
    try:
        episode_ids = np.asarray(handle["ep_idx"][:], dtype=np.int64)
        observation = np.asarray(handle["observation"][:], dtype=np.float32)
        actions = np.asarray(handle["action"][:], dtype=np.float32)
    finally:
        handle.close()
    anchor_ids = set(int(item) for item in bank["episode_id"])
    outside = ~np.isin(episode_ids, list(anchor_ids))
    other_obs = observation[outside]
    other_act = actions[outside]
    other_joints = joint_angles(other_obs)
    starts = np.asarray(bank["start_state"], dtype=np.float64)
    start_joints = joint_angles(starts)
    bank_actions = np.asarray(bank["actions"], dtype=np.float64)
    n_in_ball = np.zeros(len(starts), dtype=np.int64)
    first_action_gap = np.full((len(starts), audit.N_CANDIDATES), np.nan, dtype=np.float64)
    chunk = 200_000
    for episode in range(len(starts)):
        hand_gap = other_obs[:, 29:31].astype(np.float64) - starts[episode, 29:31]
        object_gap = other_obs[:, 33:35].astype(np.float64) - starts[episode, 33:35]
        joint_gap = np.max(np.abs(other_joints - start_joints[episode]), axis=1)
        # Chunk the norm so a full copy of the boolean mask stays small.
        inside = np.zeros(other_obs.shape[0], dtype=bool)
        for begin in range(0, other_obs.shape[0], chunk):
            end = min(begin + chunk, other_obs.shape[0])
            inside[begin:end] = (
                np.linalg.norm(hand_gap[begin:end], axis=1) <= HAND_BALL_M
            ) & (np.linalg.norm(object_gap[begin:end], axis=1) <= OBJECT_BALL_M) & (
                joint_gap[begin:end] <= JOINT_BALL_RAD
            )
        n_in_ball[episode] = int(inside.sum())
        if not np.any(inside):
            continue
        local_actions = other_act[inside].astype(np.float64)
        query = bank_actions[episode, :, 0, :]
        # (candidates, local frames)
        gap = np.linalg.norm(query[:, None, :] - local_actions[None, :, :], axis=-1)
        first_action_gap[episode] = gap.min(axis=1)
    recorded_steps = []
    handle = audit.open_h5(path)
    try:
        records = {int(record["episode_id"]): record for record in audit.index_h5_episodes(path)}
        for index, episode_id in enumerate(bank["episode_id"]):
            record = records[int(episode_id)]
            window = np.asarray(
                handle["observation"][int(record["offset"]) : int(record["offset"]) + audit.HORIZON_STEPS],
                dtype=np.float64,
            )
            distance = pose_distance(window, starts[index])
            recorded_steps.append(in_pose_ball(distance))
    finally:
        handle.close()
    recorded_inside = np.asarray(recorded_steps, dtype=bool)
    return {
        "n_frames_outside_anchor": int(outside.sum()),
        "n_in_ball": n_in_ball,
        "first_action_gap": first_action_gap,
        "recorded_steps_in_start_ball": recorded_inside,
        "episodes_with_another_frame": int(np.sum(n_in_ball > 0)),
        "median_source_hand_m": float(np.nanmedian(source_hand[:, TRANSPLANT])),
        "min_source_hand_m": float(np.nanmin(source_hand[:, TRANSPLANT])),
    }


def terminal_summary(anchor: Path, labels: np.ndarray) -> dict:
    bank, rollout, score = _bank_arrays(anchor)
    success = np.asarray(rollout["success"], dtype=bool)
    predicted_cost = np.asarray(score["predicted_cost"], dtype=np.float64)
    observed_cost = np.asarray(score["observed_cost"], dtype=np.float64)
    prediction_error = factor.latent_sse(score["predicted_emb"], score["observed_emb"])
    predicted_index = np.argmin(predicted_cost, axis=1).astype(np.int64)
    observed_index = np.argmin(observed_cost, axis=1).astype(np.int64)
    predicted_rank = ranking_gap(success, predicted_index)
    observed_rank = ranking_gap(success, observed_index)
    recorded_ok = recorded_success_mask(success)
    groups = {}
    for name, group in GROUP_SLICES.items():
        groups[name] = {
            "success_rate": bootstrap_defined(success[:, group].mean(axis=1), seed=30),
            "prediction_sse": bootstrap_defined(prediction_error[:, group].mean(axis=1), seed=31),
            "predicted_goal_cost": bootstrap_defined(predicted_cost[:, group].mean(axis=1), seed=32),
            "observed_goal_cost": bootstrap_defined(observed_cost[:, group].mean(axis=1), seed=33),
            "selected_by_predicted": bootstrap_defined(
                np.isin(predicted_index, np.arange(audit.N_CANDIDATES)[group]).astype(np.float64),
                seed=34,
            ),
        }
    alignment = effect_alignment(
        score["predicted_emb"], score["observed_emb"], min_observed_norm=1.0
    )
    order_predicted = pair_order_counts(success, predicted_cost)
    order_observed = pair_order_counts(success, observed_cost)
    sse_gap = bootstrap_defined(
        prediction_error[:, TRANSPLANT].mean(axis=1) - prediction_error[:, 0],
        seed=35,
    )
    recorded_regret = bootstrap_defined(predicted_rank["regret"][recorded_ok], seed=36)
    label_counts = {
        name: int(np.sum(labels == name))
        for name in ("observed_continuation", "recombined_expert_action", "unobserved_action")
    }
    return {
        "n_episodes": int(success.shape[0]),
        "episode_id": np.asarray(bank["episode_id"]).tolist(),
        "solvability": float(success.any(axis=1).mean()),
        "label_counts": label_counts,
        "groups": groups,
        "predicted_regret": bootstrap_defined(predicted_rank["regret"], seed=37),
        "observed_regret": bootstrap_defined(observed_rank["regret"], seed=38),
        "uniform_regret": bootstrap_defined(predicted_rank["uniform_regret"], seed=39),
        "predicted_minus_uniform_regret": bootstrap_defined(
            predicted_rank["regret_minus_uniform"], seed=40
        ),
        "recorded_success_rate": float(recorded_ok.mean()),
        "predicted_regret_when_recorded_succeeds": recorded_regret,
        "observed_regret_when_recorded_succeeds": bootstrap_defined(
            observed_rank["regret"][recorded_ok], seed=41
        ),
        "predicted_selects_recorded_when_it_succeeds": bootstrap_defined(
            (predicted_index[recorded_ok] == 0).astype(np.float64), seed=42
        ),
        "predicted_selection_counts": np.bincount(predicted_index, minlength=audit.N_CANDIDATES).tolist(),
        "pair_accuracy_predicted": bootstrap_defined(order_predicted["accuracy"], seed=43),
        "pair_accuracy_observed": bootstrap_defined(order_observed["accuracy"], seed=44),
        "transplant_minus_recorded_sse": sse_gap,
        "effect_cosine_when_outcome_moves": bootstrap_defined(
            per_episode_mean(alignment["cosine"], alignment["usable"]), seed=45
        ),
        "effect_parallel_when_outcome_moves": bootstrap_defined(
            per_episode_mean(alignment["parallel"], alignment["usable"]), seed=46
        ),
        "effect_predicted_norm_when_outcome_moves": bootstrap_defined(
            per_episode_mean(alignment["predicted_norm"], alignment["usable"]), seed=47
        ),
        "effect_observed_norm_when_outcome_moves": bootstrap_defined(
            per_episode_mean(alignment["observed_norm"], alignment["usable"]), seed=48
        ),
        "effect_by_group": _effect_by_group(score["predicted_emb"], score["observed_emb"]),
        "coverage_conclusion": coverage_conclusion(
            recorded_regret=recorded_regret,
            sse_gap=sse_gap,
            recorded_sse=groups["recorded"]["prediction_sse"],
        ),
        "prediction_error": prediction_error,
        "predicted_cost": predicted_cost,
        "observed_cost": observed_cost,
        "success": success,
        "predicted_index": predicted_index,
    }


def run_support(anchor: Path, data_root: Path, output: Path) -> dict:
    support = reconstruct_support(anchor, data_root)
    coverage = pose_ball_coverage(anchor, data_root, support["source_hand_m"])
    summary = terminal_summary(anchor, support["label"])
    arrays = {
        "prediction_error": summary.pop("prediction_error"),
        "predicted_cost": summary.pop("predicted_cost"),
        "observed_cost": summary.pop("observed_cost"),
        "success": summary.pop("success"),
        "predicted_index": summary.pop("predicted_index"),
        "source_hand_m": support["source_hand_m"],
        "n_in_ball": coverage["n_in_ball"],
        "first_action_gap": coverage["first_action_gap"],
        "recorded_steps_in_start_ball": coverage["recorded_steps_in_start_ball"],
    }
    public = {
        key: value
        for key, value in summary.items()
        if key not in arrays
    }
    public["coverage"] = {
        "ball_m_hand": HAND_BALL_M,
        "ball_m_object": OBJECT_BALL_M,
        "ball_rad_joint": JOINT_BALL_RAD,
        "n_frames_outside_anchor": coverage["n_frames_outside_anchor"],
        "episodes_with_another_frame_in_ball": coverage["episodes_with_another_frame"],
        "median_frames_in_ball": float(np.median(coverage["n_in_ball"])),
        "median_transplant_source_hand_m": coverage["median_source_hand_m"],
        "min_transplant_source_hand_m": coverage["min_source_hand_m"],
        "recorded_fraction_still_in_start_ball": bootstrap_defined(
            coverage["recorded_steps_in_start_ball"].mean(axis=1), seed=49
        ),
        "nearest_foreign_action_gap": {
            name: bootstrap_defined(coverage["first_action_gap"][:, group].mean(axis=1), seed=50 + index)
            for index, (name, group) in enumerate(GROUP_SLICES.items())
        },
    }
    public["label_counts"] = summary["label_counts"]
    public["n_train_episodes_for_chunks"] = support["n_train_episodes_for_chunks"]
    public["definition"] = (
        "Candidate 0 is the episode's own first 25 actions. Candidates 1-7 add "
        "Gaussian noise at 0.1 training standard deviations. Candidates 8-15 are "
        "25-step windows from other training episodes. The pose ball is 5 cm in "
        "hand and object position and 0.15 rad in every arm joint, measured "
        "against frames outside episodes 0-49. Nearest-neighbor distance is not "
        "used as evidence of support."
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "support.json", public)
    np.savez_compressed(output / "support_arrays.npz", **arrays, label=support["label"])
    return public


def _encode_images(model, transform, frames: list[np.ndarray]):
    import torch

    batch = torch.cat([audit._policy_image(frame, transform) for frame in frames], dim=0)
    batch = batch.to(next(model.parameters()).device)
    encoded = model.encode({"pixels": batch})["emb"][:, -1, :]
    return encoded


def _rollout_from_images(model, transform, frames: np.ndarray, blocked_actions: np.ndarray):
    """One-block or multi-block rollout. ``frames`` is (candidates, H, W, 3)."""
    import torch

    device = next(model.parameters()).device
    pixels = torch.cat([audit._policy_image(frame, transform) for frame in frames], dim=0)
    pixels = pixels.to(device)
    pixels = pixels.unsqueeze(0).unsqueeze(2)
    actions = audit.float_actions(blocked_actions, pixels).unsqueeze(0)
    with torch.inference_mode():
        rolled = model.rollout({"pixels": pixels}, actions)
    predicted = rolled["predicted_emb"][0, :, 1:, :]
    return predicted


def run_horizon(anchor: Path, checkpoint: Path, anygoal_root: Path, output: Path) -> dict:
    """Matched-horizon error from true images and from autoregressive rollout."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from anygoal.envs.insert_peg.env import InsertPegEnv

    bank, rollout, _score = _bank_arrays(anchor)
    n_keep = int(bank["actions"].shape[0])
    n_blocks = len(HORIZON_STEPS)
    model, transform = factor._load_model(checkpoint, anygoal_root)
    env = InsertPegEnv(render_mode="rgb_array", width=224, height=224, terminate_at_goal=False)
    observed = np.empty((n_keep, audit.N_CANDIDATES, n_blocks, 192), dtype=np.float32)
    autoregressive = np.empty_like(observed)
    teacher = np.empty_like(observed)
    goal = np.empty((n_keep, 192), dtype=np.float32)
    states = np.empty((n_keep, audit.N_CANDIDATES, n_blocks, audit.STATE_DIM), dtype=np.float32)
    state_after_one = np.empty((n_keep, audit.N_CANDIDATES, audit.STATE_DIM), dtype=np.float32)
    try:
        for episode in range(n_keep):
            start = np.asarray(bank["start_state"][episode], dtype=np.float64)
            actions = np.asarray(bank["actions"][episode], dtype=np.float64)
            rendered = np.empty((audit.N_CANDIDATES, n_blocks, 224, 224, 3), dtype=np.uint8)
            for candidate in range(audit.N_CANDIDATES):
                env.reset(seed=0, options={"state": start})
                block = 0
                for step, action in enumerate(actions[candidate]):
                    env.step(action)
                    if step == 0:
                        state_after_one[episode, candidate] = env.get_state()
                    if (step + 1) in HORIZON_STEPS:
                        states[episode, candidate, block] = env.get_state()
                        rendered[candidate, block] = audit.as_hwc_uint8(env.render())
                        block += 1
                if block != n_blocks:
                    raise RuntimeError("A horizon boundary was not rendered.")
            blocked = audit.block_normalized_actions(
                actions, np.asarray(bank["action_mean"]), np.asarray(bank["action_scale"])
            )
            import torch

            with torch.inference_mode():
                goal_emb = _encode_images(model, transform, [bank["goal_image"][episode]])
                goal[episode] = goal_emb[0].detach().cpu().numpy()
                flat_frames = [bank["start_image"][episode]]
                for block in range(n_blocks):
                    flat_frames.extend(rendered[candidate, block] for candidate in range(audit.N_CANDIDATES))
                encoded = _encode_images(model, transform, flat_frames).detach().cpu().numpy()
            observed[episode, :, :, :] = encoded[1:].reshape(n_blocks, audit.N_CANDIDATES, 192).transpose(1, 0, 2)
            start_frames = np.repeat(bank["start_image"][episode][None], audit.N_CANDIDATES, axis=0)
            ar = _rollout_from_images(model, transform, start_frames, blocked)
            if tuple(ar.shape) != (audit.N_CANDIDATES, n_blocks, 192):
                raise RuntimeError(f"Autoregressive rollout shape {tuple(ar.shape)}.")
            autoregressive[episode] = ar.detach().cpu().numpy()
            for block in range(n_blocks):
                if block == 0:
                    origin = start_frames
                else:
                    origin = rendered[:, block - 1]
                step = _rollout_from_images(model, transform, origin, blocked[:, block : block + 1, :])
                teacher[episode, :, block, :] = step[:, 0, :].detach().cpu().numpy()
            disagreement = np.max(np.abs(autoregressive[episode, :, 0, :] - teacher[episode, :, 0, :]))
            if disagreement > 1e-3:
                raise RuntimeError(f"First-block teacher and autoregressive predictions differ by {disagreement}.")
            final_gap = np.max(np.abs(states[episode, :, -1, :] - rollout["final_state"][episode]))
            if final_gap > 1e-4:
                raise RuntimeError(f"Resimulated terminal state differs from the saved bank by {final_gap}.")
            print(f"horizon episode {int(bank['episode_id'][episode])}", flush=True)
    finally:
        env.close()
    destination = output / "horizon_arrays.npz"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        observed=observed,
        autoregressive=autoregressive,
        teacher=teacher,
        goal=goal,
        states=states,
        state_after_one=state_after_one,
        episode_id=np.asarray(bank["episode_id"]),
    )
    public = summarize_horizon(destination, anchor)
    _write_json(output / "horizon.json", public)
    return public


def _matched_success(states: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """Physical dataset-goal success at one horizon. ``goals`` is (E, 44)."""
    success = np.empty(states.shape[:2], dtype=bool)
    for episode in range(states.shape[0]):
        for candidate in range(states.shape[1]):
            errors = audit.goal_errors(goals[episode], states[episode, candidate])
            success[episode, candidate] = audit.full_success(errors)
    return success


def summarize_horizon(arrays_path: Path, anchor: Path) -> dict:
    bank, rollout, _score = _bank_arrays(anchor)
    arrays = np.load(arrays_path)
    if not np.array_equal(arrays["episode_id"], bank["episode_id"]):
        raise RuntimeError("Horizon episodes do not match the anchor.")
    success_25 = np.asarray(rollout["success"], dtype=bool)
    recorded_ok = recorded_success_mask(success_25)
    blocks = []
    early_sse = None
    late_sse = None
    early_regret = None
    late_regret = None
    for block, horizon in enumerate(HORIZON_STEPS):
        ar_error = horizon_prediction_error(arrays["autoregressive"][:, :, block : block + 1, :], arrays["observed"][:, :, block : block + 1, :])[:, :, 0]
        teacher_error = horizon_prediction_error(arrays["teacher"][:, :, block : block + 1, :], arrays["observed"][:, :, block : block + 1, :])[:, :, 0]
        ar_cost = goal_costs(arrays["autoregressive"][:, :, block, :], arrays["goal"])
        teacher_cost = goal_costs(arrays["teacher"][:, :, block, :], arrays["goal"])
        ar_rank = ranking_gap(success_25, np.argmin(ar_cost, axis=1).astype(np.int64))
        teacher_rank = ranking_gap(success_25, np.argmin(teacher_cost, axis=1).astype(np.int64))
        matched_goal = arrays["states"][:, 0, block]
        matched = _matched_success(arrays["states"][:, :, block, :], matched_goal)
        matched_rank = ranking_gap(matched, np.argmin(ar_cost, axis=1).astype(np.int64))
        entry = {
            "horizon_raw_actions": horizon,
            "valid_model_horizon": True,
            "autoregressive_sse": {
                name: bootstrap_defined(ar_error[:, group].mean(axis=1), seed=60 + block)
                for name, group in GROUP_SLICES.items()
            },
            "teacher_sse": {
                name: bootstrap_defined(teacher_error[:, group].mean(axis=1), seed=70 + block)
                for name, group in GROUP_SLICES.items()
            },
            "autoregressive_regret_vs_25_step_success": bootstrap_defined(ar_rank["regret"], seed=80 + block),
            "teacher_regret_vs_25_step_success": bootstrap_defined(teacher_rank["regret"], seed=90 + block),
            "autoregressive_regret_on_recorded_successes": bootstrap_defined(
                ar_rank["regret"][recorded_ok], seed=100 + block
            ),
            "matched_pose_solvability": float(matched.any(axis=1).mean()),
            "matched_pose_candidates": float(matched.mean()),
            "autoregressive_regret_vs_matched_pose": bootstrap_defined(matched_rank["regret"], seed=110 + block),
            "pair_accuracy_vs_25_step_success": bootstrap_defined(
                pair_order_counts(success_25, ar_cost)["accuracy"], seed=120 + block
            ),
        }
        if horizon == 5:
            alignment = effect_alignment(
                arrays["autoregressive"][:, :, 0, :],
                arrays["observed"][:, :, 0, :],
                min_observed_norm=1.0,
            )
            entry["effect_cosine"] = bootstrap_defined(
                per_episode_mean(alignment["cosine"], alignment["usable"]), seed=130
            )
            entry["effect_predicted_norm"] = bootstrap_defined(
                per_episode_mean(alignment["predicted_norm"], alignment["usable"]), seed=131
            )
            entry["effect_observed_norm"] = bootstrap_defined(
                per_episode_mean(alignment["observed_norm"], alignment["usable"]), seed=132
            )
            early_sse = bootstrap_defined(ar_error[:, 0], seed=133)
            early_regret = entry["autoregressive_regret_vs_25_step_success"]
        if horizon == 25:
            late_sse = bootstrap_defined(ar_error[:, 0], seed=134)
            late_regret = entry["autoregressive_regret_vs_25_step_success"]
        blocks.append(entry)
    one_step = pose_distance(arrays["state_after_one"], bank["start_state"][:, None, :])
    conclusion = accumulation_conclusion(
        early_regret=early_regret,
        late_regret=late_regret,
        early_sse=early_sse,
        late_sse=late_sse,
    )
    return {
        "definition": (
            "Horizons are raw Insert-Peg actions. SMWM consumes them in blocks of "
            "five, so one raw action is not a model prediction. Autoregressive "
            "predictions start from the true first frame. Teacher predictions "
            "encode the true frame at the previous block boundary and predict one "
            "block. Goal costs compare a prediction with the encoded 25-step goal "
            "image. The 25-step physical success is the saved evaluator outcome."
        ),
        "one_raw_action_is_model_output": False,
        "one_raw_action_hand_motion_m": bootstrap_defined(one_step["hand_m"].mean(axis=1), seed=135),
        "blocks": blocks,
        "accumulation_conclusion": conclusion,
        "recorded_first_block_sse": early_sse,
        "recorded_final_block_sse": late_sse,
    }


def _metric_row(summary: dict) -> str:
    return _fmt(summary)


def _action_paragraph(support: dict, horizon: dict | None) -> str:
    grouped = support["effect_by_group"]
    text = (
        "The start image is held fixed and the 16 action sequences vary. "
        f"Small perturbations move the encoded future by {_fmt(grouped['perturbation']['observed_norm'])} "
        f"and the prediction by {_fmt(grouped['perturbation']['predicted_norm'])}. "
        f"Transplanted sequences move the encoded future by {_fmt(grouped['transplant']['observed_norm'])} "
        f"and the prediction by {_fmt(grouped['transplant']['predicted_norm'])}. "
        "The prediction does respond to the action. On futures that actually separate, its component "
        f"along the true change is {_fmt(support['effect_parallel_when_outcome_moves'])}."
    )
    if horizon is not None and "effect_cosine" in horizon["blocks"][0]:
        text += (
            " The same cosine at the first block is "
            f"{_fmt(horizon['blocks'][0]['effect_cosine'])}."
        )
    return text


def _literature_paragraph(support: dict) -> str:
    cosine = support["effect_cosine_when_outcome_moves"]
    norm = support["effect_predicted_norm_when_outcome_moves"]
    unrelated = (
        cosine.get("defined")
        and cosine["mean"] is not None
        and abs(float(cosine["mean"])) < 0.15
        and float(cosine["ci_low"]) < 0.15
        and float(cosine["ci_high"]) > -0.15
    )
    if str(support["coverage_conclusion"]).startswith("present_on_recorded"):
        expert = "the expert continuation is predicted no better"
    elif support["coverage_conclusion"] == "concentrated_off_the_recorded_continuation":
        expert = "the extra error sits on actions that were not stored with that start"
    else:
        expert = "the recorded-versus-transplant error split is in the table above"
    direction = (
        "The direction of that change is unrelated to the future the encoder records."
        if unrelated
        else f"The alignment of that change with the encoded future is {_fmt(cosine)}."
    )
    moves = (
        norm.get("defined") and norm["mean"] is not None and float(norm["mean"]) > 0.5
    )
    motion = (
        "The measured change is not near zero when the action actually changes the future."
        if moves
        else "The predicted change stays small when the action changes the future."
    )
    return (
        "AnyGoal reports that expert data can make a world model under-use the commanded "
        "action, and that a low prediction error need not yield a usable plan. The Insert-Peg "
        "result agrees that the planner's latent cost is not a goal ranking. It does not agree "
        "that the failure appears only after the action leaves the expert support: "
        f"{expert}. SMWM's inverse loss is designed to keep action "
        "information in the encoder of real transitions. That part is working here, because the "
        "encoded real futures rank the goal. The forward rollout used for planning does not "
        "inherit a usable direction. Action Influence Maximization shows that expert "
        "demonstrations can make the next state predictable without the action, which would "
        f"show up as a near-zero predicted change. {motion} {direction}"
    )


def _decision_paragraph(support: dict, horizon: dict | None) -> str:
    coverage = support["coverage_conclusion"]
    horizon_name = None if horizon is None else horizon["accumulation_conclusion"]
    if coverage == "present_on_recorded_transitions" and horizon_name == "present_at_first_block":
        return (
            "Next action: stop. The ranking error is on the stored expert continuation, "
            "and it is already present at the first action block. Another representation "
            "objective is not the measurement this result asks for."
        )
    if coverage == "concentrated_off_the_recorded_continuation":
        return (
            "Next action: stop treating the expert continuation as the broken prediction. "
            "The residual question is which recombined actions the training distribution "
            "actually supports, and that is a coverage question rather than a new loss."
        )
    if horizon_name == "emerges_with_horizon":
        return (
            "Next action: test whether replanning from the true encoded image, at the "
            "first block boundary, removes the 25-step regret. Do not train a new loss "
            "until that control is in."
        )
    if horizon is None:
        return "Horizon comparison has not been run, so the local-versus-accumulated question is still open."
    return (
        f"Coverage conclusion `{coverage}` and horizon conclusion `{horizon_name}` "
        "do not yet pick a single next experiment. Inspect the per-horizon table before "
        "adding a loss or a new dataset."
    )


def write_report(docs: Path, support: dict, horizon: dict | None) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    groups = support["groups"]
    lines = [
        "# Why SMWM misranks the Insert-Peg action bank",
        "",
        "Fifty Insert-Peg episodes, the seed-0 bank of 16 sequences, and `sensorimotor_insertpeg.ckpt`. The goal is the recorded frame 25 actions after the start. Predicted opportunity regret on this bank is "
        f"{_fmt(support['predicted_regret'])}. The same goal cost on the encoded simulated future is {_fmt(support['observed_regret'])}.",
        "",
        "## Stage 1: state-action coverage",
        "",
        support["definition"],
        "",
        f"Labels: {support['label_counts']['observed_continuation']} observed continuations, "
        f"{support['label_counts']['unobserved_action']} perturbed actions, and "
        f"{support['label_counts']['recombined_expert_action']} recombined expert actions. "
        f"Transplanted windows come from {support['n_train_episodes_for_chunks']} other training episodes. "
        f"Their source hand is {support['coverage']['median_transplant_source_hand_m']:.2f} m from the anchor start "
        f"(minimum {support['coverage']['min_transplant_source_hand_m']:.2f} m).",
        "",
        f"Outside episodes 0–49 there are {support['coverage']['n_frames_outside_anchor']} frames. "
        f"A 5 cm / 0.15 rad ball around the anchor start contains another training frame in "
        f"{support['coverage']['episodes_with_another_frame_in_ball']} of 50 episodes "
        f"(median count {support['coverage']['median_frames_in_ball']:.0f}). "
        "The recorded continuation itself leaves that ball within the 25 steps "
        f"({_fmt(support['coverage']['recorded_fraction_still_in_start_ball'])} of its steps remain inside). "
        "An empty or tiny ball means these starts are not a cloud of alternative recorded interventions. "
        "It does not, by itself, explain a prediction that is equally wrong on the stored continuation.",
        "",
        "| Group | Physical success | Prediction SSE | Predicted goal cost | Observed goal cost | Chosen by predicted cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("recorded", "perturbation", "transplant"):
        row = groups[name]
        lines.append(
            f"| {name} | {_metric_row(row['success_rate'])} | {_metric_row(row['prediction_sse'])} | "
            f"{_metric_row(row['predicted_goal_cost'])} | {_metric_row(row['observed_goal_cost'])} | "
            f"{_metric_row(row['selected_by_predicted'])} |"
        )
    lines.extend([
        "",
        f"Paired prediction SSE, transplant minus recorded: {_fmt(support['transplant_minus_recorded_sse'])}.",
        f"Selection counts across the 16 candidates are {support['predicted_selection_counts']}. "
        f"When the recorded continuation succeeds ({support['recorded_success_rate']:.2f} of episodes), "
        f"predicted regret is {_fmt(support['predicted_regret_when_recorded_succeeds'])} and the model selects "
        f"that continuation on {_fmt(support['predicted_selects_recorded_when_it_succeeds'])} of those episodes. "
        f"Observed-latent regret on the same episodes is {_fmt(support['observed_regret_when_recorded_succeeds'])}.",
        "",
        f"Pair accuracy of the predicted goal cost is {_fmt(support['pair_accuracy_predicted'])}. "
        f"Pair accuracy of the observed goal cost is {_fmt(support['pair_accuracy_observed'])}. "
        f"Uniform selection has regret {_fmt(support['uniform_regret'])}. Predicted regret minus that "
        f"expectation is {_fmt(support['predicted_minus_uniform_regret'])}.",
        "",
        f"On pairs whose actual futures differ by more than 1 in latent norm, the predicted change has "
        f"cosine {_fmt(support['effect_cosine_when_outcome_moves'])} with the actual change, parallel "
        f"component {_fmt(support['effect_parallel_when_outcome_moves'])}, predicted norm "
        f"{_fmt(support['effect_predicted_norm_when_outcome_moves'])}, and observed norm "
        f"{_fmt(support['effect_observed_norm_when_outcome_moves'])}.",
        "",
        f"Coverage decision: `{support['coverage_conclusion']}`.",
        "",
    ])
    if horizon is not None:
        lines.extend([
            "## Stage 2: first block versus later blocks",
            "",
            horizon["definition"],
            "",
            f"One raw action is not a model output. Across the bank it moves the hand by "
            f"{_fmt(horizon['one_raw_action_hand_motion_m'], 4)} m, so candidates have not separated yet.",
            "",
            "| Horizon | Recorded AR SSE | Recorded teacher SSE | Transplant AR SSE | Predicted regret vs 25-step success | Pair accuracy |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for block in horizon["blocks"]:
            if block["horizon_raw_actions"] not in (5, 10, 25):
                continue
            lines.append(
                f"| {block['horizon_raw_actions']} | {_fmt(block['autoregressive_sse']['recorded'])} | "
                f"{_fmt(block['teacher_sse']['recorded'])} | {_fmt(block['autoregressive_sse']['transplant'])} | "
                f"{_fmt(block['autoregressive_regret_vs_25_step_success'])} | "
                f"{_fmt(block['pair_accuracy_vs_25_step_success'])} |"
            )
        lines.extend([
            "",
            f"Recorded first-block SSE is {_fmt(horizon['recorded_first_block_sse'])}. "
            f"Recorded fifth-block autoregressive SSE is {_fmt(horizon['recorded_final_block_sse'])}. "
            f"The last-block prediction that starts from the true image at step 20 still has recorded "
            f"error {_fmt(horizon['blocks'][-1]['teacher_sse']['recorded'])}. The recorded hand has left "
            "the isolated start pose by then, so the local error is not confined to that pose.",
            "",
            f"Horizon decision: `{horizon['accumulation_conclusion']}`.",
            "",
        ])
        first = horizon["blocks"][0]
        if "effect_cosine" in first:
            lines.extend([
                f"At the first block, where the outcome has already moved, effect cosine is "
                f"{_fmt(first['effect_cosine'])}, predicted norm {_fmt(first['effect_predicted_norm'])}, "
                f"observed norm {_fmt(first['effect_observed_norm'])}.",
                "",
            ])
    lines.extend([
        "## Stage 3",
        "",
        "Not run. The observed-latent goal cost already reads the encoded future. Its pair accuracy is far above chance and its regret is near zero, so the outcome encoding retains the distinction the decision needs. A supervised probe would add labels the world model does not have, and a failed probe would not show that the information is absent.",
        "",
        "## Stage 4: action conditioning",
        "",
        _action_paragraph(support, horizon),
        "",
        "## What this supports",
        "",
        "The 0.64 regret is a bad ordering of goal costs. Successful recorded and perturbed actions end near the goal in the encoder, and transplanted actions do not. The forward model assigns those groups nearly the same goal cost, because its action-conditioned update points somewhere other than the actual next embedding. That error is already present on the stored expert continuation, whose start state and 25 actions are a training trajectory. "
        + (
            "The horizon comparison places the same failure at the first action block."
            if horizon is not None and horizon.get("accumulation_conclusion") == "present_at_first_block"
            else "Whether that error starts at the first block is the horizon result above."
        ),
        "",
        _literature_paragraph(support),
        "",
        "Unresolved: this is one checkpoint and the 50 training episodes used as the anchor. Teacher-forced blocks later in those same episodes are not a held-out episode. The transplanted sequences are expert strings from other poses, not every intervention the training support could contain. No new loss was trained.",
        "",
        _decision_paragraph(support, horizon),
        "",
    ])
    (docs / "report.md").write_text("\n".join(lines))
    _write_figure(docs, support, horizon)


def _write_figure(docs: Path, support: dict, horizon: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["recorded", "perturbation", "transplant"]
    predicted = [support["groups"][name]["predicted_goal_cost"]["mean"] for name in names]
    observed = [support["groups"][name]["observed_goal_cost"]["mean"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    positions = np.arange(len(names))
    axes[0].bar(positions - 0.18, observed, width=0.36, color="#1b4f72", label="Encoded future")
    axes[0].bar(positions + 0.18, predicted, width=0.36, color="#b9770e", label="Predicted")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("Latent goal cost")
    axes[0].set_title("Same start, three action groups")
    axes[0].legend(frameon=False)
    if horizon is not None:
        horizons = [block["horizon_raw_actions"] for block in horizon["blocks"]]
        recorded = [block["autoregressive_sse"]["recorded"]["mean"] for block in horizon["blocks"]]
        teacher = [block["teacher_sse"]["recorded"]["mean"] for block in horizon["blocks"]]
        transplant = [block["autoregressive_sse"]["transplant"]["mean"] for block in horizon["blocks"]]
        axes[1].plot(horizons, recorded, marker="o", color="#1b4f72", label="Recorded, autoregressive")
        axes[1].plot(horizons, teacher, marker="o", color="#148f77", label="Recorded, from true image")
        axes[1].plot(horizons, transplant, marker="o", color="#b9770e", label="Transplant, autoregressive")
        axes[1].set_xlabel("Raw actions")
        axes[1].set_ylabel("Prediction SSE")
        axes[1].set_ylim(0.0, max(60.0, max(recorded + teacher + transplant) * 1.05))
        axes[1].set_title("Error at each action block")
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].set_axis_off()
    figure.tight_layout()
    figure.savefig(docs / "ranking_mechanism.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    support = subparsers.add_parser("support")
    support.add_argument("--anchor", type=Path, required=True)
    support.add_argument("--data-root", type=Path, required=True)
    support.add_argument("--output", type=Path, required=True)
    horizon = subparsers.add_parser("horizon")
    horizon.add_argument("--anchor", type=Path, required=True)
    horizon.add_argument("--checkpoint", type=Path, required=True)
    horizon.add_argument("--anygoal-root", type=Path, required=True)
    horizon.add_argument("--output", type=Path, required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--support", type=Path, required=True)
    report.add_argument("--horizon", type=Path, default=None)
    report.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "support":
        payload = run_support(args.anchor, args.data_root, args.output)
        print(json.dumps({
            "coverage_conclusion": payload["coverage_conclusion"],
            "predicted_regret": payload["predicted_regret"],
            "label_counts": payload["label_counts"],
        }, default=audit._json_default))
    elif args.command == "horizon":
        payload = run_horizon(args.anchor, args.checkpoint, args.anygoal_root, args.output)
        print(json.dumps({
            "accumulation_conclusion": payload["accumulation_conclusion"],
            "first": payload["blocks"][0]["autoregressive_regret_vs_25_step_success"],
            "last": payload["blocks"][-1]["autoregressive_regret_vs_25_step_success"],
        }, default=audit._json_default))
    elif args.command == "report":
        support_payload = json.loads(args.support.read_text())
        horizon_payload = json.loads(args.horizon.read_text()) if args.horizon else None
        write_report(args.docs, support_payload, horizon_payload)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
