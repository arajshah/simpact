"""Proprioceptive arm baseline for the Insert-Peg action bank.

Joint positions, joint velocities, and the hand pose from forward kinematics
are in the environment's proprioception. SMWM's planner does not read them.
Object pose, object velocity, and the slot target are privileged in the
dataset and are estimated from images. Future joint positions are not a
current encoder reading; the goal image still has to supply the goal pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit
import anygoal_dynamics_factor as factor
import anygoal_state_estimator as estimator

ARM_SLICE = slice(0, 24)
HAND_SLICE = slice(29, 33)
OBJECT_SLICE = slice(33, 40)
TARGET_SLICE = slice(40, 44)


def apply_current_proprio(estimated: np.ndarray, measured: np.ndarray, *, zero_object_velocity: bool) -> np.ndarray:
    """Replace the estimated arm with the measured one. Leave the object estimate."""
    estimated = np.asarray(estimated, dtype=np.float64).copy()
    measured = np.asarray(measured, dtype=np.float64)
    if estimated.shape != measured.shape or estimated.shape[-1] != audit.STATE_DIM:
        raise ValueError("Estimated and measured states must share shape (N, 44).")
    estimated[..., ARM_SLICE] = measured[..., ARM_SLICE]
    estimated[..., HAND_SLICE] = measured[..., HAND_SLICE]
    if zero_object_velocity:
        estimated[..., 37:40] = 0.0
    return estimated


def proprio_decision(regret: dict) -> dict[str, str]:
    mean = float(regret["mean"])
    low = float(regret["ci_low"])
    high = float(regret["ci_high"])
    if mean <= 0.10 and high <= 0.16:
        return {
            "decision": "NO-GO",
            "reason": (
                f"Current joint readings plus a visual object and goal estimate "
                f"have regret {mean:.3f} [{low:.3f}, {high:.3f}]. Conventional "
                "state estimation plus physics closes the ranking gap."
            ),
        }
    if mean >= 0.20 and low > 0.10:
        return {
            "decision": "DIAGNOSIS",
            "reason": (
                f"Proprioceptive arm state still leaves regret {mean:.3f} "
                f"[{low:.3f}, {high:.3f}]. The residual is not the visual arm estimate."
            ),
        }
    return {
        "decision": "INCONCLUSIVE",
        "reason": (
            f"Proprioceptive regret is {mean:.3f} [{low:.3f}, {high:.3f}]. "
            "The interval does not clearly close the gap and does not clearly remain large."
        ),
    }


def _summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "selected"}


def _estimate(checkpoint: Path, images: np.ndarray, device: str) -> np.ndarray:
    encoder, head, mean, std = estimator._load_trained(checkpoint, device)
    raw = estimator._predict_raw(encoder, head, images, device)
    packed = np.stack([estimator.pack_state(row, mean, std) for row in raw])
    return estimator.derived_observation(packed)


def _errors(estimate: np.ndarray, truth: np.ndarray) -> dict:
    from anygoal.envs.insert_peg.env import insert_peg_goal_errors

    hand, obj, hand_ang, obj_ang, joints = [], [], [], [], []
    for pred, label in zip(estimate, truth):
        errors = insert_peg_goal_errors(label, pred)
        hand.append(errors["hand_position"])
        obj.append(errors["object_position"])
        hand_ang.append(errors["hand_angle"])
        obj_ang.append(errors["object_angle"])
        joints.append(float(np.max(np.abs(estimator.wrap_angle(
            estimator.joint_angles(label) - estimator.joint_angles(pred)
        )))))
    return {
        "hand_m": estimator.percentile_summary(np.asarray(hand)),
        "object_m": estimator.percentile_summary(np.asarray(obj)),
        "hand_rad": estimator.percentile_summary(np.asarray(hand_ang)),
        "object_rad": estimator.percentile_summary(np.asarray(obj_ang)),
        "joint_rad": estimator.percentile_summary(np.asarray(joints)),
    }


def run(anchor: Path, start_checkpoint: Path, goal_checkpoint: Path, output: Path, device: str) -> dict:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    bank = np.load(anchor / "bank.npz")
    success = np.load(anchor / "bank" / "evaluator_10.npz")["success"]
    predicted = np.load(anchor / "scores" / "evaluator_10.npz")["predicted_cost"]
    true_start = np.asarray(bank["start_state"], dtype=np.float64)
    true_goal = np.asarray(bank["goal_state"], dtype=np.float64)
    actions = np.asarray(bank["actions"])
    start_hat = _estimate(start_checkpoint, bank["start_image"], device)
    goal_hat = _estimate(goal_checkpoint, bank["goal_image"], device)
    proprio = apply_current_proprio(start_hat, true_start, zero_object_velocity=False)
    proprio_zero_vel = apply_current_proprio(start_hat, true_start, zero_object_velocity=True)
    rows = {
        "oracle_true_state": estimator._rank_from_states(true_start, true_goal, actions, success),
        "proprio_arm_visual_object_and_goal": estimator._rank_from_states(proprio, goal_hat, actions, success),
        "proprio_arm_visual_object_zero_object_velocity": estimator._rank_from_states(
            proprio_zero_vel, goal_hat, actions, success
        ),
        "proprio_arm_true_object_visual_goal": estimator._rank_from_states(
            _with_true_object(proprio, true_start), goal_hat, actions, success
        ),
        "proprio_arm_visual_object_true_goal": estimator._rank_from_states(proprio, true_goal, actions, success),
    }
    smwm_index = np.argmin(predicted, axis=1).astype(np.int64)
    smwm_scores = factor.per_episode_scores(success, smwm_index)
    payload = {
        "information": {
            "smwm": "goal and current images only; the planner transform has no proprio key",
            "proprio": (
                "current joint positions, joint velocities, and hand pose from the "
                "environment proprio field; not the future arm and not the object"
            ),
            "visual_object_and_goal": (
                "episode-start checkpoint for the current object; mixed-frame checkpoint "
                "for the goal image 25 steps later. Both were fixed before this ranking."
            ),
            "oracle": "full simulator state, including the object and the future",
        },
        "start_visual_errors": _errors(start_hat, true_start),
        "goal_visual_errors": _errors(goal_hat, true_goal),
        "rankings": {name: _summary(result) for name, result in rows.items()},
        "smwm_predicted": {
            "success_rate": float(smwm_scores["chosen_success"].mean()),
            "regret": audit.bootstrap_mean(smwm_scores["regret"], seed=41),
        },
    }
    payload["decision"] = proprio_decision(payload["rankings"]["proprio_arm_visual_object_and_goal"]["regret"])
    output.mkdir(parents=True, exist_ok=True)
    factor.write_json(output / "proprio_ranking.json", payload)
    return payload


def _with_true_object(state: np.ndarray, truth: np.ndarray) -> np.ndarray:
    mixed = np.asarray(state, dtype=np.float64).copy()
    mixed[:, OBJECT_SLICE] = truth[:, OBJECT_SLICE]
    mixed[:, TARGET_SLICE] = truth[:, TARGET_SLICE]
    return mixed


def audit_start_shift(cache: Path, output: Path) -> dict:
    mixed = np.load(cache / "train" / "states.npy")
    starts = np.load(cache.parent / "cache_starts" / "train" / "states.npy")
    mixed_joints = np.stack([estimator.joint_angles(row) for row in mixed])
    start_joints = np.stack([estimator.joint_angles(row) for row in starts])
    low, high = np.quantile(mixed_joints, [0.05, 0.95], axis=0)
    outside = np.mean(np.any((start_joints < low) | (start_joints > high), axis=1))
    payload = {
        "train_start_joint_outside_mixed_5_95": float(outside),
        "mixed_hand_z_median_m": float(np.median(mixed[:, 30])),
        "start_hand_z_median_m": float(np.median(starts[:, 30])),
        "mixed_hand_object_distance_median_m": float(np.median(np.linalg.norm(mixed[:, 29:31] - mixed[:, 33:35], axis=1))),
        "start_hand_object_distance_median_m": float(np.median(np.linalg.norm(starts[:, 29:31] - starts[:, 33:35], axis=1))),
        "note": (
            "Episode starts put at least one joint outside the central 90% of the "
            "mixed-frame training sample. The hand is also higher and farther from the peg."
        ),
    }
    factor.write_json(output / "start_shift.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    rank = subparsers.add_parser("rank")
    rank.add_argument("--anchor", type=Path, required=True)
    rank.add_argument("--start-checkpoint", type=Path, required=True)
    rank.add_argument("--goal-checkpoint", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--device", default="cuda")
    shift = subparsers.add_parser("shift")
    shift.add_argument("--cache", type=Path, required=True)
    shift.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rank":
        print(json.dumps(
            run(args.anchor, args.start_checkpoint, args.goal_checkpoint, args.output, args.device),
            default=audit._json_default,
        ))
    elif args.command == "shift":
        print(json.dumps(audit_start_shift(args.cache, args.output)))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
