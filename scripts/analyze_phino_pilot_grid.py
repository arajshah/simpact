#!/usr/bin/env python3
"""Probe synthetic mass/friction recovery from calibrated SIMPACT trajectories.

Run from the SIMPACT repository root (no MuJoCo or PHINO import required):
    python scripts/analyze_phino_pilot_grid.py --self-test
    python scripts/analyze_phino_pilot_grid.py
    python scripts/analyze_phino_pilot_grid.py --also-object-holdout
    python scripts/analyze_phino_pilot_grid.py --feature-ablation

The feature-ablation option compares three fixed trajectory feature sets on
identical object and joint-physics holdouts, and saves separate CSV/JSON files.
The object-holdout option also reports per-carton errors, out-of-range
predictions, train-to-test feature shifts, first-saved-frame orientation,
and paired object motion scales. It does NOT retrain or regenerate trajectories.

Inputs from scripts/phino_simpact_pilot_v2.py:
    results/phino_pilot_grid_v1/manifest.json
    results/phino_pilot_grid_v1/action_trajectories/*.npz

Primary evaluation: leave one (mass, friction) setting out, with BOTH object
meshes in that test fold. Compare a fixed-alpha ridge model to the training-
mean baseline for EACH target. Never split 50 time steps into separate samples.
Only valid individual trajectories contribute to each action combination.
The lift is reported separately: missing runs at 0.25 kg are not imputed.

These are synthetic, deterministic simulation trajectories for multiple
geometries. Object holdout tests transfer to a different geometry, but any
missing action trajectories reduce the eligible test population. The test
does not establish real-world physics or DINOv2 performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ACTIONS = ("push_x", "push_y", "grasp_lift_release", "lateral_flick")
NON_LIFT = ("push_x", "push_y", "lateral_flick")
ACTION_SETS = (
    ("push_x",),
    ("push_y",),
    ("lateral_flick",),
    ("push_x", "push_y"),
    NON_LIFT,
)
TARGETS = (("mass_kg", "kg"), ("contact_friction", "coefficient"))
SAMPLE_SHAPE = (50, 13)
# Indices in trajectory_features()'s documented 11-value output. All features
# are derived exclusively from the trajectory, not labels or validity flags.
# The reduced subsets are predeclared, not selected using held-out performance.
FEATURE_SETS = {
    "original": tuple(range(11)),
    "no_vertical_position": (0, 1, 3, 4, 6, 7, 8, 9, 10),
    "motion_magnitudes_only": (6, 7, 8, 9, 10),
}


def _scalar(x: Any) -> float:
    n = float(x)
    if not math.isfinite(n):
        raise ValueError(f"Non-finite numeric value: {x}")
    return n


def _key(row: dict[str, Any]) -> tuple[float, float]:
    return (round(_scalar(row["mass_kg"]), 9),
            round(_scalar(row["contact_friction"]), 9))


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Sign-invariant orientation change from first saved frame, in degrees."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    dot = np.clip(np.abs(q1 @ q0), 0., 1.)
    return np.degrees(2 * np.arccos(dot))


def trajectory_features(trajectory: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """11 compact, label-free features from each valid (50,13) action trace.

    PHINO channels: position relative to start (3), quaternion wxyz (4),
    linear velocity (3), angular velocity (3). No mass/friction/status inputs.
    Position at frame 10 and 50, path length and peak displacement/speed,
    plus sign-invariant maximum orientation change are measured.
    """
    a = np.asarray(trajectory, dtype=np.float64)
    if a.shape != SAMPLE_SHAPE or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected finite {SAMPLE_SHAPE} trajectory, got {a.shape}")
    p, q, v, w = a[:, :3], a[:, 3:7], a[:, 7:10], a[:, 10:13]
    qlength = np.linalg.norm(q, axis=1)
    if np.any(qlength < 1e-8):
        raise ValueError("Zero quaternion in trajectory")
    values = np.r_[p[9], p[-1],
                    np.max(np.linalg.norm(p, axis=1)),
                    np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)),
                    np.max(np.linalg.norm(v, axis=1)),
                    np.max(np.linalg.norm(w, axis=1)),
                    np.max(_quat_angle_deg(q[0], q))]
    names = ([f"frame10_position_{c}_m" for c in "xyz"]
             + [f"final_position_{c}_m" for c in "xyz"]
             + ["peak_displacement_m", "path_length_m", "peak_linear_speed_m_s",
                "peak_angular_speed_rad_s", "max_orientation_change_deg"])
    assert len(values) == len(names) == 11
    return values.astype(np.float64), names


def first_saved_frame_orientation(trace: np.ndarray) -> dict[str, float]:
    """Orientation of first SAVED frame (after the first simulation interval).

    This is NOT the exact pre-action pose. The local body axes can differ between
    reconstructed objects; local-z-to-world-up is not a toppling measurement.
    """
    q = np.asarray(trace[0, 3:7], dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = map(float, q)
    local_z_dot_world_up = float(np.clip(1. - 2. * (x*x + y*y), -1., 1.))
    return {
        "first_frame_quat_w": w, "first_frame_quat_x": x,
        "first_frame_quat_y": y, "first_frame_quat_z": z,
        "first_frame_local_z_vs_world_up_deg":
            float(np.degrees(np.arccos(local_z_dot_world_up))),
        "first_frame_quat_vs_identity_deg":
            float(np.degrees(2. * np.arccos(np.clip(abs(w), 0., 1.)))),
    }


def _path_from_record(grid_dir: Path, record: dict[str, Any], action: str) -> Path:
    status = record["action_status"][action]
    given = status.get("trajectory_path")
    if not given:
        raise ValueError(f"Valid {record['object']} {action} has no trajectory_path")
    p = Path(given)
    possibilities = [p, grid_dir / p,
                     grid_dir / "action_trajectories" / p.name]
    for candidate in possibilities:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Trajectory for {record['object']}, {_key(record)}, {action} missing. "
        f"Tried original path and {grid_dir}/action_trajectories/{p.name}")


def load_dataset(grid_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected {manifest_path}; generate grid first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("action_order") != list(ACTIONS):
        raise ValueError("Unexpected action_order in manifest; refusing to mislabel traces")
    if int(manifest.get("frames_per_action", -1)) != SAMPLE_SHAPE[0]:
        raise ValueError("Expected exactly 50 frames per action")
    if manifest.get("feature_clipping") is not False:
        raise ValueError("Expected raw, unclipped trajectories")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest has no configuration records")
    rows = []
    visited = set()
    for rec in records:
        name = str(rec["object"])
        mass, mu = _key(rec)
        identifier = (name, mass, mu)
        if identifier in visited:
            raise ValueError(f"Duplicate configuration {identifier}")
        visited.add(identifier)
        row = {"object": name, "mass_kg": mass, "contact_friction": mu,
               "features": {}, "valid_actions": [], "action_status": rec["action_status"],
               "action_metrics": rec.get("action_metrics", {}),
               "first_saved_frame_orientation": {}}
        for action in ACTIONS:
            state = rec["action_status"].get(action)
            if state is None:
                raise ValueError(f"Missing status for {identifier}, {action}")
            if state.get("valid") is not True:
                continue
            path = _path_from_record(grid_dir, rec, action)
            with np.load(path, allow_pickle=False) as arr:
                if "trajectory" not in arr:
                    raise ValueError(f"Expected 'trajectory' in {path}")
                trace = np.asarray(arr["trajectory"])
                # Check redundant metadata to catch accidental file mixing.
                metadata = {"object_name": name, "action_name": action,
                            "mass_kg": mass, "contact_friction": mu}
                for field, expected in metadata.items():
                    if field not in arr:
                        raise ValueError(f"Missing '{field}' metadata in {path}")
                    actual = arr[field].item()
                    if isinstance(expected, str):
                        if str(actual) != expected:
                            raise ValueError(f"Mismatched '{field}' in {path}")
                    elif not np.isclose(float(actual), expected, atol=1e-9, rtol=0):
                        raise ValueError(f"Mismatched '{field}' in {path}")
            features, _ = trajectory_features(trace)
            row["features"][action] = features
            row["first_saved_frame_orientation"][action] = first_saved_frame_orientation(trace)
            row["valid_actions"].append(action)
        rows.append(row)
    return rows, manifest


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, alpha: float) -> np.ndarray:
    """Train-only standardization + ridge with a free intercept, NumPy only."""
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.
    ztrain = (x_train - center) / scale
    ztest = (x_test - center) / scale
    ymean = y_train.mean(axis=0)
    # Dual-form ridge is stable with far more features than samples.
    kernel = ztrain @ ztrain.T
    weights = np.linalg.solve(kernel + alpha * np.eye(len(ztrain)),
                              y_train - ymean)
    return ymean + (ztest @ ztrain.T) @ weights


def _score(y: np.ndarray, yhat: np.ndarray) -> dict[str, float | None]:
    diff = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1. - np.sum(diff ** 2) / denom) if denom > 1e-15 else None
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _split_keys(rows: list[dict[str, Any]], split: str) -> list[Any]:
    if split == "physics":
        return sorted({_key(r) for r in rows})
    if split == "object":
        return sorted({r["object"] for r in rows})
    raise ValueError(split)


def _membership(row: dict[str, Any], split: str) -> Any:
    return _key(row) if split == "physics" else row["object"]


def _selected_feature_columns(action_set: tuple[str, ...], feature_set: str) -> np.ndarray:
    """Select the exact same indices, independently for each action."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    n = len(trajectory_features(_fake_trace(.3, .3, 0, 0))[1])
    return np.array([action_ix * n + feature_ix
                     for action_ix in range(len(action_set))
                     for feature_ix in FEATURE_SETS[feature_set]], dtype=int)


def evaluate(rows: list[dict[str, Any]], action_set: tuple[str, ...],
             alpha: float, split: str,
             feature_set: str = "original") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = [r for r in rows if all(a in r["features"] for a in action_set)]
    excluded = len(rows) - len(available)
    if len(available) < 6:
        raise ValueError(f"Only {len(available)} rows available for {action_set}")
    X = np.array([np.concatenate([r["features"][a] for a in action_set])
                  for r in available], dtype=float)
    X = X[:, _selected_feature_columns(action_set, feature_set)]
    Y = np.array([[r["mass_kg"], r["contact_friction"]]
                  for r in available], dtype=float)
    folds = _split_keys(available, split)
    predictions = []
    if len(folds) < 2:
        raise ValueError(f"Need >=2 {split} groups")
    for fold in folds:
        held = np.array([_membership(row, split) == fold for row in available], dtype=bool)
        train = ~held
        if not np.any(held) or np.sum(train) < 2:
            raise ValueError(f"Insufficient training data for holdout {fold}")
        # A whole joint mass/friction configuration across all objects is test.
        if split == "physics":
            assert not any(_key(available[i]) == fold for i in np.flatnonzero(train))
        p_ridge = _ridge_fit_predict(X[train], Y[train], X[held], alpha)
        p_mean = np.repeat(Y[train].mean(axis=0)[None, :], int(np.sum(held)), axis=0)
        for j, idx in enumerate(np.flatnonzero(held)):
            row = available[idx]
            predictions.append({
                "split": split, "actions": "+".join(action_set),
                "holdout": (f"m{fold[0]:.3f}_mu{fold[1]:.3f}"
                            if split == "physics" else str(fold)),
                "object": row["object"],
                "mass_kg_actual": float(Y[idx, 0]),
                "mass_kg_ridge": float(p_ridge[j, 0]),
                "mass_kg_baseline": float(p_mean[j, 0]),
                "contact_friction_actual": float(Y[idx, 1]),
                "contact_friction_ridge": float(p_ridge[j, 1]),
                "contact_friction_baseline": float(p_mean[j, 1]),
            })
    if len(predictions) != len(available):
        raise AssertionError("Each usable configuration must be predicted exactly once")
    metrics = {"split": split, "actions": "+".join(action_set),
               "n_rows": len(available), "excluded_rows": excluded,
               "n_objects_evaluable": len({r["object"] for r in available}),
               "objects_evaluable": sorted({r["object"] for r in available}),
               "n_folds": len(folds), "alpha": alpha, "targets": {}}
    for target, unit in TARGETS:
        actual = np.array([p[f"{target}_actual"] for p in predictions])
        ridge = np.array([p[f"{target}_ridge"] for p in predictions])
        baseline = np.array([p[f"{target}_baseline"] for p in predictions])
        metrics["targets"][target] = {"unit": unit,
                                       "ridge": _score(actual, ridge),
                                       "baseline": _score(actual, baseline)}
    return predictions, metrics


def object_holdout_diagnostics(
        rows: list[dict[str, Any]],
        predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Explain leave-one-object-out performance without fitting a new model.

    Each feature shift is measured against the *training objects* distribution,
    never pooled across the train/test boundary. No label is a model feature.
    Works with two or more objects; each object is held out entirely.
    """
    if not any(p["split"] == "object" for p in predictions):
        raise ValueError("Object predictions required; use --also-object-holdout")
    feature_names = trajectory_features(_fake_trace(.3, .3, 0, 0))[1]
    errors = []
    shift = []
    shift_summary = []
    for action_set in ACTION_SETS:
        action_label = "+".join(action_set)
        available = [r for r in rows if all(a in r["features"] for a in action_set)]
        combined = [f"{a}:{feature}" for a in action_set for feature in feature_names]
        matrix = np.array([np.concatenate([r["features"][a] for a in action_set])
                           for r in available], dtype=float)
        objects = sorted({r["object"] for r in available})
        if len(objects) < 2:
            raise ValueError(f"Expected >=2 objects for object shift, found {objects}")
        for obj in objects:
            train_objects_label = ",".join(o for o in objects if o != obj)
            test_mask = np.array([r["object"] == obj for r in available], dtype=bool)
            train_x, test_x = matrix[~test_mask], matrix[test_mask]
            if len(train_x) < 2 or len(test_x) < 2:
                raise ValueError(f"Not enough object rows for {action_label}, {obj}")
            train_mean = train_x.mean(axis=0)
            train_std = train_x.std(axis=0)  # model uses population SD
            train_scale = train_std.copy()
            replaced = train_scale < 1e-12
            train_scale[replaced] = 1.  # EXACTLY the ridge model's rule
            test_z = (test_x - train_mean) / train_scale
            train_min, train_max = train_x.min(axis=0), train_x.max(axis=0)
            tol = 1e-12
            outside = ((test_x < (train_min - tol)) |
                       (test_x > (train_max + tol)))
            for j, fname in enumerate(combined):
                shift.append({
                    "heldout_object": obj, "train_object": train_objects_label,
                    "actions": action_label, "feature": fname,
                    "n_train": len(train_x), "n_test": len(test_x),
                    "train_mean": float(train_mean[j]),
                    "train_std": float(train_std[j]),
                    "train_min": float(train_min[j]),
                    "train_max": float(train_max[j]),
                    "test_mean": float(test_x[:, j].mean()),
                    "test_std": float(test_x[:, j].std()),
                    "test_min": float(test_x[:, j].min()),
                    "test_max": float(test_x[:, j].max()),
                    "test_mean_z_vs_train": float(test_z[:, j].mean()),
                    "max_abs_test_z_vs_train": float(np.max(np.abs(test_z[:, j]))),
                    "test_outside_train_minmax_fraction": float(outside[:, j].mean()),
                    "train_std_replaced_with_one": bool(replaced[j]),
                })
            shift_summary.append({
                "heldout_object": obj,
                "train_object": train_objects_label,
                "actions": action_label,
                "feature_count": len(combined),
                "mean_abs_test_mean_z": float(np.abs(test_z.mean(axis=0)).mean()),
                "max_abs_test_z": float(np.abs(test_z).max()),
                "feature_count_test_mean_abs_z_gt_3": int(np.sum(np.abs(test_z.mean(axis=0)) > 3)),
                "fraction_of_test_feature_values_outside_train_range": float(outside.mean()),
                "train_std_replaced_feature_count": int(np.sum(replaced)),
            })
            fold_predictions = [p for p in predictions if
                                p["split"] == "object" and p["actions"] == action_label
                                and p["holdout"] == obj]
            if len(fold_predictions) != len(test_x):
                raise AssertionError(f"Missing object predictions for {obj}, {action_label}")
            for target, unit in TARGETS:
                actual = np.array([p[f"{target}_actual"] for p in fold_predictions])
                predicted = np.array([p[f"{target}_ridge"] for p in fold_predictions])
                baseline = np.array([p[f"{target}_baseline"] for p in fold_predictions])
                train_labels = np.array([r[target] for r, is_test in zip(available, test_mask)
                                         if not is_test])
                min_train, max_train = float(train_labels.min()), float(train_labels.max())
                errors.append({
                    "heldout_object": obj,
                    "train_object": train_objects_label,
                    "actions": action_label, "target": target, "unit": unit,
                    "n_test": len(actual),
                    "ridge_mae": _score(actual, predicted)["mae"],
                    "baseline_mae": _score(actual, baseline)["mae"],
                    "ridge_rmse": _score(actual, predicted)["rmse"],
                    "ridge_mean_signed_error": float(np.mean(predicted - actual)),
                    "actual_min": float(actual.min()), "actual_max": float(actual.max()),
                    "ridge_prediction_min": float(predicted.min()),
                    "ridge_prediction_max": float(predicted.max()),
                    "train_target_min": min_train, "train_target_max": max_train,
                    "prediction_outside_train_target_range_count": int(np.sum(
                        (predicted < (min_train - 1e-12)) |
                        (predicted > (max_train + 1e-12)))),
                })

    orientation = []
    for row in rows:
        for action in NON_LIFT:
            if action not in row["first_saved_frame_orientation"]:
                continue
            orientation.append({
                "object": row["object"], "mass_kg": row["mass_kg"],
                "contact_friction": row["contact_friction"], "action": action,
                **row["first_saved_frame_orientation"][action],
            })

    # Compare motion magnitudes for the SAME physics values on each geometry.
    # Differences can come from geometry, contact configuration or frames;
    # they do not isolate any one causal explanation.
    objects = sorted({row["object"] for row in rows})
    if len(objects) < 2:
        raise ValueError("Matched comparison requires at least two object geometries")
    by_key = {(r["object"], *_key(r)): r for r in rows}
    selected_indices = [6, 7, 8, 9, 10]  # displacement, path, speed, angular speed, rotation
    matched = []
    for mass, mu in sorted({_key(row) for row in rows}):
        for i, obj_a in enumerate(objects):
            first = by_key.get((obj_a, mass, mu))
            if first is None:
                continue
            for obj_b in objects[i+1:]:
                second = by_key.get((obj_b, mass, mu))
                if second is None:
                    continue
                for action in NON_LIFT:
                    if action not in first["features"] or action not in second["features"]:
                        continue
                    for j in selected_indices:
                        a = float(first["features"][action][j])
                        b = float(second["features"][action][j])
                        matched.append({
                            "mass_kg": mass, "contact_friction": mu,
                            "action": action, "feature": feature_names[j],
                            "reference_object": obj_a, "other_object": obj_b,
                            "reference_value": a, "other_value": b,
                            "other_minus_reference": b - a,
                            "other_over_reference_if_reference_gt_1e-6":
                                b / a if abs(a) > 1e-6 else None,
                        })
    return {
        "object_transfer_errors": errors,
        "object_feature_shift": shift,
        "object_feature_shift_summary": shift_summary,
        "first_saved_frame_orientation": orientation,
        "matched_object_motion": matched,
    }


def coverage_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive validity, not a model feature or supervised prediction."""
    result = []
    masses = sorted({r["mass_kg"] for r in rows})
    for mass in masses:
        sub = [r for r in rows if r["mass_kg"] == mass]
        for action in ACTIONS:
            valid = [r for r in sub if action in r["features"]]
            directed = [float(r["action_metrics"][action]["directed_m"])
                        for r in valid if "directed_m" in r["action_metrics"].get(action, {})]
            reached = sum(bool(r["action_status"][action].get("motion_target_met", False))
                          for r in valid)
            result.append({"mass_kg": mass, "action": action,
                           "valid": len(valid), "total": len(sub),
                           "target_met": reached,
                           "mean_directed_m_if_valid":
                               float(np.mean(directed)) if directed else None})
    return result


def object_action_set_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Valid complete-case input coverage for every geometry and predictor set.

    An object lacking an action is not an object-transfer test for that set.
    Do not impute rejected trajectories, or count missing objects as zero error.
    """
    report = []
    for obj in sorted({r["object"] for r in rows}):
        sub = [r for r in rows if r["object"] == obj]
        for action_set in (*ACTION_SETS, ("grasp_lift_release",)):
            valid = sum(all(a in r["features"] for a in action_set) for r in sub)
            report.append({
                "object": obj,
                "actions": "+".join(action_set),
                "n_available": valid,
                "n_total": len(sub),
                "n_missing": len(sub) - valid,
                "object_holdout_evaluable": valid > 0,
                "note": ("evaluable" if valid > 0 else
                         "not_evaluable: no valid trajectory for required actions"),
            })
    return report


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot save empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_flat(m: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target, _ in TARGETS:
        entry = m["targets"][target]
        for model in ("ridge", "baseline"):
            rows.append({"split": m["split"], "actions": m["actions"],
                         "target": target, "unit": entry["unit"], "model": model,
                         "n_rows": m["n_rows"], "excluded_rows": m["excluded_rows"],
                         "n_objects_evaluable": m["n_objects_evaluable"],
                         "n_folds": m["n_folds"], "ridge_alpha": m["alpha"],
                         **entry[model]})
    return rows


def _run(rows: list[dict[str, Any]], alpha: float,
         splits: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for split in splits:
        for action_set in ACTION_SETS:
            pred, metric = evaluate(rows, action_set, alpha, split)
            predictions.extend(pred)
            scores.append(metric)
    return predictions, scores


def feature_ablation(rows: list[dict[str, Any]], alpha: float
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                list[dict[str, Any]]]:
    """Identical folds, ridge alpha, and mean baseline for all feature sets.

    These feature sets were chosen after inspecting the first two objects.
    Treat their scores on those objects as exploratory, not independent tests.
    """
    preds: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for split in ("physics", "object"):
            for action_set in ACTION_SETS:
                fold_preds, metric = evaluate(rows, action_set, alpha, split,
                                              feature_set=feature_set)
                preds.extend([{"feature_set": feature_set, **p} for p in fold_preds])
                metrics.append({"feature_set": feature_set, **metric,
                                "n_features_per_action": len(FEATURE_SETS[feature_set])})

    object_breakdown: list[dict[str, Any]] = []
    base_scores: dict[tuple[str, str, str], float] = {}
    for feature_set in FEATURE_SETS:
        for action_set in ACTION_SETS:
            label = "+".join(action_set)
            for obj in sorted({r["object"] for r in rows}):
                sub = [p for p in preds if p["feature_set"] == feature_set
                       and p["split"] == "object" and p["actions"] == label
                       and p["holdout"] == obj]
                # Leave the geometry visible in the report even if every
                # required trajectory was rejected. No prediction is possible.
                for target, unit in TARGETS:
                    if sub:
                        y = np.array([p[f"{target}_actual"] for p in sub])
                        yhat = np.array([p[f"{target}_ridge"] for p in sub])
                        baseline = np.array([p[f"{target}_baseline"] for p in sub])
                        mae = _score(y, yhat)["mae"]
                        mae_base = _score(y, baseline)["mae"]
                    else:
                        mae = None
                        mae_base = None
                    if feature_set == "original":
                        base_scores[(label, obj, target)] = mae
                    baseline_original = base_scores[(label, obj, target)]
                    object_breakdown.append({
                        "feature_set": feature_set,
                        "actions": label,
                        "heldout_object": obj,
                        "target": target,
                        "unit": unit,
                        "n_test": len(sub),
                        "n_features_total": len(FEATURE_SETS[feature_set]) * len(action_set),
                        "status": ("evaluated" if sub else "not_evaluable_missing_actions"),
                        "ridge_mae": mae,
                        "baseline_mae": mae_base,
                        "ridge_minus_baseline_mae": (
                            mae - mae_base if sub else None),
                        "ridge_minus_original_mae": (
                            mae - baseline_original if sub and baseline_original is not None
                            else None),
                    })
    return preds, metrics, object_breakdown


def save_feature_ablation(output_dir: Path, rows: list[dict[str, Any]],
                          alpha: float) -> dict[str, Any]:
    """Save dedicated files; do not overwrite established regression outputs."""
    predictions, metrics, breakdown = feature_ablation(rows, alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "feature_ablation_predictions.csv", predictions)
    save_csv(output_dir / "feature_ablation_scores.csv", [
        {"feature_set": m["feature_set"],
         "features_per_action": m["n_features_per_action"], **flat}
        for m in metrics for flat in _summary_flat(m)])
    save_csv(output_dir / "feature_ablation_object_breakdown.csv", breakdown)
    save_csv(output_dir / "action_set_coverage_by_object.csv", object_action_set_coverage(rows))
    names = trajectory_features(_fake_trace(.3, .3, 0, 0))[1]
    description = {
        key: [names[j] for j in indices] for key, indices in FEATURE_SETS.items()
    }
    report = {
        "protocol": (f"fixed-feature ablation; same "
                     f"{len({_key(r) for r in rows})} joint-physics folds and "
                     f"{len({r['object'] for r in rows})} object folds"),
        "alpha": alpha,
        "baseline": "training-fold target mean",
        "feature_sets": description,
        "n_configurations": len(rows),
        "n_source_objects": len({r["object"] for r in rows}),
        "note": ("No additional simulations or model training beyond the ridge fits. "
                 "Feature sets were chosen after observing object-transfer failures. "
                 "Deterministic synthetic diagnostic, not causal proof or a "
                 "real-world generalization claim."),
        "scores": metrics,
        "object_breakdown": breakdown,
        "action_set_coverage_by_object": object_action_set_coverage(rows),
        "missingness_policy": ("Scores use only configurations with ALL required "
                               "valid action traces; ineligible objects get N/A "
                               "rather than silently receiving an error score. "
                               "Effective sample size depends on the action set."),
    }
    (output_dir / "feature_ablation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report




def sphere_identifier(rows: list[dict[str, Any]]) -> str:
    """Identify the one sphere geometry by its explicit mesh name."""
    names = sorted({r["object"] for r in rows
                    if r["object"].endswith("__mesh_sphere") or r["object"] == "mesh_sphere"})
    if len(names) != 1:
        raise ValueError(f"Expected exactly one sphere geometry, found {names}")
    return names[0]


def compare_sphere_training(rows: list[dict[str, Any]], alpha: float):
    """Matched held-out NON-sphere cases; vary only sphere inclusion in training.

    Test rows, feature subsets, ridge penalty, and model family remain fixed.
    Scaling is always fitted on training data only. Rejected actions are excluded.
    """
    sphere = sphere_identifier(rows)
    other_objects = sorted({r["object"] for r in rows if r["object"] != sphere})
    if len(other_objects) < 2:
        raise ValueError("Need at least two non-sphere geometries")
    predictions = []
    for feature_set in FEATURE_SETS:
        for actions in ACTION_SETS:
            available = [r for r in rows if all(a in r["features"] for a in actions)]
            cols = _selected_feature_columns(actions, feature_set)
            for heldout in other_objects:
                test_rows = [r for r in available if r["object"] == heldout]
                if not test_rows:
                    continue
                train_base = [r for r in available if r["object"] not in (heldout, sphere)]
                train_sphere = [r for r in available if r["object"] == sphere]
                if len(train_base) < 2:
                    raise ValueError(f"Insufficient training data for {heldout}, {actions}")

                def xy(items):
                    x = np.array([np.concatenate([r["features"][a] for a in actions])
                                  for r in items], dtype=float)[:, cols]
                    y = np.array([[r["mass_kg"], r["contact_friction"]]
                                  for r in items], dtype=float)
                    return x, y

                x_test, y_test = xy(test_rows)
                for mode, train_rows in (("without_sphere", train_base),
                                         ("with_sphere", train_base + train_sphere)):
                    x_train, y_train = xy(train_rows)
                    ridge = _ridge_fit_predict(x_train, y_train, x_test, alpha)
                    mean = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
                    for i, r in enumerate(test_rows):
                        predictions.append({
                            "feature_set": feature_set, "actions": "+".join(actions),
                            "heldout_object": heldout, "mass_kg": r["mass_kg"],
                            "contact_friction": r["contact_friction"],
                            "training_mode": mode, "n_training": len(train_rows),
                            "n_sphere_training": len(train_sphere) if mode == "with_sphere" else 0,
                            "mass_kg_actual": float(y_test[i, 0]),
                            "mass_kg_ridge": float(ridge[i, 0]),
                            "mass_kg_baseline": float(mean[i, 0]),
                            "contact_friction_actual": float(y_test[i, 1]),
                            "contact_friction_ridge": float(ridge[i, 1]),
                            "contact_friction_baseline": float(mean[i, 1]),
                        })
    paired = defaultdict(set)
    for p in predictions:
        key = (p["feature_set"], p["actions"], p["heldout_object"],
               p["mass_kg"], p["contact_friction"])
        paired[key].add(p["training_mode"])
    if not paired or any(v != {"without_sphere", "with_sphere"} for v in paired.values()):
        raise AssertionError("Sphere comparison has unmatched test cases")
    summary, by_object = [], []
    for feature_set in FEATURE_SETS:
        for actions in ACTION_SETS:
            label = "+".join(actions)
            for target, unit in TARGETS:
                entries = {}
                for mode in ("without_sphere", "with_sphere"):
                    ps = [p for p in predictions if p["feature_set"] == feature_set
                          and p["actions"] == label and p["training_mode"] == mode]
                    if not ps:
                        continue
                    actual = np.array([p[f"{target}_actual"] for p in ps])
                    ridge = np.array([p[f"{target}_ridge"] for p in ps])
                    baseline = np.array([p[f"{target}_baseline"] for p in ps])
                    entries[mode] = (_score(actual, ridge), _score(actual, baseline))
                    for obj in other_objects:
                        op = [p for p in ps if p["heldout_object"] == obj]
                        if op:
                            oa = np.array([p[f"{target}_actual"] for p in op])
                            oridge = np.array([p[f"{target}_ridge"] for p in op])
                            obase = np.array([p[f"{target}_baseline"] for p in op])
                            by_object.append({
                                "feature_set": feature_set, "actions": label,
                                "target": target, "unit": unit,
                                "heldout_object": obj, "training_mode": mode,
                                "n_test": len(op),
                                "ridge_mae": _score(oa, oridge)["mae"],
                                "baseline_mae": _score(oa, obase)["mae"],
                            })
                if len(entries) != 2:
                    raise AssertionError("Missing matched model comparison")
                before, after = entries["without_sphere"], entries["with_sphere"]
                n_test = sum(p["feature_set"] == feature_set and p["actions"] == label
                             and p["training_mode"] == "without_sphere" for p in predictions)
                summary.append({
                    "feature_set": feature_set, "actions": label, "target": target,
                    "unit": unit, "n_test": n_test,
                    "n_test_objects": len({p["heldout_object"] for p in predictions
                                          if p["feature_set"] == feature_set
                                          and p["actions"] == label}),
                    "ridge_mae_without_sphere": before[0]["mae"],
                    "ridge_mae_with_sphere": after[0]["mae"],
                    "delta_mae_with_minus_without": after[0]["mae"] - before[0]["mae"],
                    "baseline_mae_without_sphere": before[1]["mae"],
                    "baseline_mae_with_sphere": after[1]["mae"],
                })
    coverage = []
    for actions in ACTION_SETS:
        for obj in [*other_objects, sphere]:
            obj_rows = [r for r in rows if r["object"] == obj]
            n = sum(all(a in r["features"] for a in actions) for r in obj_rows)
            coverage.append({
                "object": obj, "actions": "+".join(actions),
                "n_valid": n, "n_total": len(obj_rows),
                "n_missing": len(obj_rows) - n,
                "role": "training_only_sphere" if obj == sphere else "evaluation_object",
            })
    return sphere, predictions, summary, by_object, coverage


def rolling_speed_features(trace: np.ndarray, radius_m: float) -> dict[str, Any]:
    """Translation and angular-speed magnitudes, not a no-slip contact test.

    Angular speed norm is frame invariant. The v/(r*omega) ratio is approximate
    and cannot establish the contact point's velocity or rolling without slip.
    """
    a = np.asarray(trace, dtype=float)
    if a.shape != SAMPLE_SHAPE or not np.isfinite(a).all():
        raise ValueError(f"Expected finite {SAMPLE_SHAPE}, got {a.shape}")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive and finite")
    vxy = np.linalg.norm(a[:, 7:9], axis=1)
    omega = np.linalg.norm(a[:, 10:13], axis=1)
    active = (vxy > 0.005) & (radius_m * omega > 0.005)
    ratios = vxy[active] / (radius_m * omega[active])
    p = a[:, :3]
    return {
        "final_xy_displacement_m": float(np.linalg.norm(p[-1, :2])),
        "max_xy_displacement_m": float(np.max(np.linalg.norm(p[:, :2], axis=1))),
        "xy_path_length_m": float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))),
        "z_range_m": float(np.ptp(p[:, 2])),
        "mean_xy_speed_m_s": float(np.mean(vxy)),
        "peak_xy_speed_m_s": float(np.max(vxy)),
        "mean_angular_speed_rad_s": float(np.mean(omega)),
        "peak_angular_speed_rad_s": float(np.max(omega)),
        "active_ratio_frames": int(np.sum(active)),
        "median_speed_ratio_v_over_r_omega": (
            float(np.median(ratios)) if len(ratios) else None),
        "fraction_active_frames_ratio_0p5_to_2": (
            float(np.mean((ratios >= .5) & (ratios <= 2.))) if len(ratios) else None),
    }


def save_sphere_transfer(output_dir: Path, grid_dir: Path, rows, manifest,
                         alpha: float, radius_m: float):
    """Write separate sphere-training ablation and rotational-motion reports."""
    sphere, preds, scores, per_object, coverage = compare_sphere_training(rows, alpha)
    frame_hz = float(manifest.get("sampling_Hz", 20.))
    if not np.isfinite(frame_hz) or frame_hz <= 0:
        raise ValueError("Invalid sampling frequency")
    sphere_motion = []
    for rec in manifest["records"]:
        if rec["object"] != sphere:
            continue
        for action in ("push_x", "push_y"):
            status = rec["action_status"].get(action, {})
            if status.get("valid") is not True:
                sphere_motion.append({
                    "mass_kg": float(rec["mass_kg"]),
                    "contact_friction": float(rec["contact_friction"]),
                    "action": action, "valid": False,
                    "reason": status.get("reason", "not valid"),
                })
                continue
            path = _path_from_record(grid_dir, rec, action)
            with np.load(path, allow_pickle=False) as data:
                trace = np.asarray(data["trajectory"])
            sphere_motion.append({
                "mass_kg": float(rec["mass_kg"]),
                "contact_friction": float(rec["contact_friction"]),
                "action": action, "valid": True, "reason": "",
                **rolling_speed_features(trace, radius_m),
            })
    fields = list(dict.fromkeys(k for r in sphere_motion for k in r))
    for row in sphere_motion:
        for field in fields:
            row.setdefault(field, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "matched_predictions.csv", preds)
    save_csv(output_dir / "matched_scores.csv", scores)
    save_csv(output_dir / "per_object_scores.csv", per_object)
    save_csv(output_dir / "coverage.csv", coverage)
    if sphere_motion:
        save_csv(output_dir / "sphere_motion.csv", sphere_motion)
    report = {
        "analysis": "Non-sphere object holdouts, sphere included/excluded only in training",
        "sphere": sphere, "alpha": alpha,
        "n_test_cases_per_action": {
            "+".join(a): sum(p["training_mode"] == "without_sphere"
                             and p["feature_set"] == "original"
                             and p["actions"] == "+".join(a) for p in preds)
            for a in ACTION_SETS},
        "sphere_radius_assumed_m": radius_m, "frame_rate_hz": frame_hz,
        "scores": scores, "per_object_scores": per_object,
        "coverage": coverage, "sphere_motion": sphere_motion,
        "interpretation_limits": [
            "Ridge alpha and feature sets fixed; test configurations matched between training modes.",
            "Sphere training adds rows, so training sample size differs; no reweighting applied.",
            "Missing actions are never imputed; eligible samples vary across action sets.",
            "v/(r*|omega|) uses angular-speed magnitude, not contact-point velocity; it cannot prove no-slip rolling.",
            "Sphere radius is approximate; collision and contact physics are synthetic.",
            "Exploratory seven-geometry evaluation; no untouched final test geometries.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nMatched transfer to six non-sphere geometries "
          "(MAE; positive delta means sphere hurts):")
    print(f"{'Feature set':<24} {'Actions':<30} {'Target':<17} "
          f"{'Without':>9} {'With':>9} {'Delta':>9} {'Cases':>5}")
    for entry in scores:
        print(f"{entry['feature_set']:<24} {entry['actions']:<30} "
              f"{entry['target']:<17} {entry['ridge_mae_without_sphere']:>9.4f} "
              f"{entry['ridge_mae_with_sphere']:>9.4f} "
              f"{entry['delta_mae_with_minus_without']:>+9.4f} {entry['n_test']:>5}")
    valid_motion = [r for r in sphere_motion if r["valid"]]
    if valid_motion:
        print("\nSphere translation/rotation diagnostic "
              "(coarse; not proof of no-slip rolling):")
        for action in ("push_x", "push_y"):
            sub = [r for r in valid_motion if r["action"] == action]
            if sub:
                ratios = [r["median_speed_ratio_v_over_r_omega"] for r in sub
                          if r["median_speed_ratio_v_over_r_omega"] is not None]
                print(f"  {action}: n={len(sub)} "
                      f"median peak_xy_speed="
                      f"{np.median([r['peak_xy_speed_m_s'] for r in sub]):.3f}m/s "
                      f"median peak_angular_speed="
                      f"{np.median([r['peak_angular_speed_rad_s'] for r in sub]):.3f}rad/s "
                      f"median v/(r|omega|)={np.median(ratios) if ratios else float('nan'):.3f}")
    print(f"SPHERE_TRANSFER_DIR={output_dir.resolve()}")
    print("PHINO_SPHERE_TRANSFER_OK")
    return report


def _fake_trace(mass: float, friction: float, obj: int, action: int) -> np.ndarray:
    """Synthetic dynamics used only to test the analysis path, not score models."""
    t = np.linspace(0, 1, 50)
    a = np.zeros(SAMPLE_SHAPE, dtype=np.float32)
    amplitude = (0.2 + .15 * obj + .08 * action) / (mass + friction)
    a[:, 0] = amplitude * t * .1
    a[:, 1] = amplitude * .5 * t ** 2 * .1
    a[:, 2] = 0.01 * np.sin(np.pi * t)
    a[:, 3] = 1
    a[:, 7] = np.gradient(a[:, 0], 0.05)
    a[:, 8] = np.gradient(a[:, 1], 0.05)
    return a


def self_test() -> None:
    q = np.tile(np.array([1., 0., 0., 0.]), (50, 1))
    q[-1] = -q[0]
    assert abs(_quat_angle_deg(q[0], q)[-1]) < 1e-6
    assert abs(_quat_angle_deg(q[0], np.array([[0., 0., 0., 1.]]))[0] - 180) < 1e-6
    tr = _fake_trace(.3, .2, 0, 0)
    assert abs(first_saved_frame_orientation(tr)["first_frame_local_z_vs_world_up_deg"]) < 1e-8
    sideways = tr.copy()
    sideways[0, 3:7] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
    assert abs(first_saved_frame_orientation(sideways)["first_frame_local_z_vs_world_up_deg"] - 90.) < 1e-5
    pure_yaw = tr.copy()
    pure_yaw[0, 3:7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    assert abs(first_saved_frame_orientation(pure_yaw)["first_frame_local_z_vs_world_up_deg"]) < 1e-5
    feats, names = trajectory_features(tr)
    assert feats.shape == (11,) and len(names) == 11
    assert [names[i] for i in FEATURE_SETS["no_vertical_position"]] == [
        n for n in names if n not in ("frame10_position_z_m", "final_position_z_m")]
    assert [names[i] for i in FEATURE_SETS["motion_magnitudes_only"]] == [
        "peak_displacement_m", "path_length_m", "peak_linear_speed_m_s",
        "peak_angular_speed_rad_s", "max_orientation_change_deg"]
    assert len(_selected_feature_columns(NON_LIFT, "original")) == 33
    assert len(_selected_feature_columns(NON_LIFT, "no_vertical_position")) == 27
    assert len(_selected_feature_columns(NON_LIFT, "motion_magnitudes_only")) == 15
    # Exercise the actual I/O format and CV logic without MuJoCo/scikit-learn.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "action_trajectories").mkdir()
        records = []
        for obj_ix, name in enumerate(("test_white", "test_blue")):
            for mass in (.25, .3, .35):
                for mu in (.2, .3, .4):
                    rec = {"object": name, "mass_kg": mass, "contact_friction": mu,
                           "action_metrics": {}, "action_status": {}}
                    for action_ix, action in enumerate(ACTIONS):
                        if action == "grasp_lift_release" and mass == .25:
                            rec["action_status"][action] = {
                                "valid": False, "status": "rejected", "reason": "overshoot"}
                            continue
                        p = root / "action_trajectories" / f"{name}_{mass}_{mu}__{action}.npz"
                        np.savez_compressed(p,
                            trajectory=_fake_trace(mass, mu, obj_ix, action_ix),
                            object_name=name, action_name=action,
                            mass_kg=mass, contact_friction=mu)
                        rec["action_status"][action] = {
                            "valid": True, "status": "valid",
                            "motion_target_met": True, "trajectory_path": str(p)}
                        rec["action_metrics"][action] = {"directed_m": .03}
                    records.append(rec)
        (root / "manifest.json").write_text(json.dumps({
            "action_order": list(ACTIONS), "frames_per_action": 50,
            "feature_clipping": False, "records": records}), encoding="utf-8")
        loaded, _ = load_dataset(root)
        assert len(loaded) == 18
        assert sum("grasp_lift_release" in r["features"] for r in loaded) == 12
        preds, scores = _run(loaded, alpha=10., splits=("physics", "object"))
        assert len(scores) == 10
        for metric in scores:
            assert metric["n_rows"] == 18 and metric["excluded_rows"] == 0
            assert metric["n_folds"] == (9 if metric["split"] == "physics" else 2)
        assert len(preds) == 18 * 5 * 2
        assert len(coverage_table(loaded)) == 12
        diagnostic = object_holdout_diagnostics(loaded, preds)
        assert len(diagnostic["object_transfer_errors"]) == 20
        assert len(diagnostic["object_feature_shift"]) == 176
        assert len(diagnostic["object_feature_shift_summary"]) == 10
        assert len(diagnostic["first_saved_frame_orientation"]) == 54
        assert len(diagnostic["matched_object_motion"]) == 135
        assert all(np.isfinite(d["test_mean_z_vs_train"])
                   for d in diagnostic["object_feature_shift"])
        ablation_dir = root / "feature_ablation"
        report = save_feature_ablation(ablation_dir, loaded, alpha=10.)
        assert len(report["scores"]) == 3 * 2 * len(ACTION_SETS)
        assert len(report["object_breakdown"]) == 3 * len(ACTION_SETS) * 2 * len(TARGETS)
        for name in ("feature_ablation_predictions.csv", "feature_ablation_scores.csv",
                     "feature_ablation_object_breakdown.csv", "feature_ablation_report.json"):
            assert (ablation_dir / name).is_file()
        # All feature sets have identical split membership, labels and baselines.
        ab_preds, _, _ = feature_ablation(loaded, alpha=10.)
        fold_keys = {}
        for pr in ab_preds:
            key = (pr["split"], pr["actions"], pr["holdout"], pr["object"],
                   pr["mass_kg_actual"], pr["contact_friction_actual"])
            fold_keys.setdefault(pr["feature_set"], set()).add(key)
        assert all(fold_keys[s] == fold_keys["original"] for s in FEATURE_SETS)
        by_record = {}
        for pr in ab_preds:
            key = (pr["split"], pr["actions"], pr["holdout"], pr["object"],
                   pr["mass_kg_actual"], pr["contact_friction_actual"])
            by_record.setdefault(key, {})[pr["feature_set"]] = (
                pr["mass_kg_baseline"], pr["contact_friction_baseline"])
        assert all(len(v) == len(FEATURE_SETS) and len(set(v.values())) == 1
                   for v in by_record.values())
        for filename, entries in diagnostic.items():
            out = root / (filename + ".csv")
            save_csv(out, entries)
            assert out.exists() and out.stat().st_size > 100
        # Test portable fallback for a moved manifest with an old absolute path.
        records[0]["action_status"]["push_x"]["trajectory_path"] = (
            "/other/machine/" + Path(records[0]["action_status"]["push_x"]["trajectory_path"]).name)
        (root / "manifest.json").write_text(json.dumps({
            "action_order": list(ACTIONS), "frames_per_action": 50,
            "feature_clipping": False, "records": records}), encoding="utf-8")
        assert len(load_dataset(root)[0]) == 18
        # A third distinct object must participate in physics folds as a third
        # test row and in object folds as its own entirely held-out group.
        import copy
        third = []
        for old in loaded[:9]:
            row3 = copy.deepcopy(old)
            row3["object"] = "test_third"
            for action in NON_LIFT:
                row3["features"][action] = row3["features"][action] * 1.07
            third.append(row3)
        expanded = loaded + third
        p3, scores3 = _run(expanded, 10., ("physics", "object"))
        assert len(scores3) == 10
        assert sum(p["split"] == "physics" for p in p3) == 27 * len(ACTION_SETS)
        assert sum(p["split"] == "object" for p in p3) == 27 * len(ACTION_SETS)
        assert all(m["n_folds"] == (9 if m["split"] == "physics" else 3)
                   for m in scores3)
        diag3 = object_holdout_diagnostics(expanded, p3)
        assert len(diag3["object_transfer_errors"]) == 3 * len(ACTION_SETS) * 2
        assert len(diag3["matched_object_motion"]) == 3 * 135
        ab3 = feature_ablation(expanded, 10.)
        assert len(ab3[1]) == 3 * 2 * len(ACTION_SETS)
        sphere_test = copy.deepcopy(expanded)
        for rec in sphere_test:
            if rec["object"] == "test_third":
                rec["object"] = "object_test__mesh_sphere"
        sid, matched, checks, per_geom, cov = compare_sphere_training(sphere_test, 10.)
        assert sid == "object_test__mesh_sphere"
        assert len(checks) == len(FEATURE_SETS) * len(ACTION_SETS) * len(TARGETS)
        assert all(e["n_test_objects"] == 2 and e["n_test"] == 18 for e in checks)
        assert all(p["training_mode"] in ("without_sphere", "with_sphere") for p in matched)
        assert len(per_geom) > 0 and len(cov) == len(ACTION_SETS) * 3
        rf = rolling_speed_features(_fake_trace(.3, .2, 2, 0), .05)
        assert rf["max_xy_displacement_m"] > 0
        assert rf["peak_xy_speed_m_s"] > 0
        # Simulate the real mesh-grid failure: the third object has no valid
        # push_x or push_y trajectories but retains flick trajectories.
        missing = copy.deepcopy(expanded)
        for r in missing:
            if r["object"] == "test_third":
                r["features"].pop("push_x", None)
                r["features"].pop("push_y", None)
        am, bm, br = feature_ablation(missing, 10.)
        assert len(am) > 0 and len(bm) == len(ab3[1])
        for entry in br:
            if entry["heldout_object"] == "test_third":
                if "push_" in entry["actions"]:
                    assert entry["status"] == "not_evaluable_missing_actions"
                    assert entry["n_test"] == 0 and entry["ridge_mae"] is None
                else:
                    assert entry["status"] == "evaluated" and entry["n_test"] == 9
        coverage3 = object_action_set_coverage(missing)
        assert any(c["object"] == "test_third" and c["actions"] == "push_x"
                   and c["n_available"] == 0 for c in coverage3)
        # Also exercise saved reports for missing-action geometry.
        report_missing = save_feature_ablation(root / "missing_ablation", missing, 10.)
        assert any(e["status"] == "not_evaluable_missing_actions"
                   for e in report_missing["object_breakdown"])
    print("PHINO_PROBE_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", type=Path,
                        default=Path("results/phino_pilot_grid_v1"),
                        help="Directory containing manifest.json and action_trajectories/")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: GRID_DIR/recovery_analysis")
    parser.add_argument("--alpha", type=float, default=10.,
                        help="Fixed ridge penalty; not tuned on held-out folds (default 10)")
    parser.add_argument("--also-object-holdout", action="store_true",
                        help="Leave one geometry out, with feature-shift and orientation diagnostics")
    parser.add_argument("--feature-ablation", action="store_true",
                        help="Compare original, no-vertical-position, and motion-only "
                             "features on identical physics and object holdouts; "
                             "prints both split results")
    parser.add_argument("--sphere-transfer", action="store_true",
                        help="Compare non-sphere object holdouts with vs without sphere training; "
                             "also report sphere translational and angular speeds")
    parser.add_argument("--sphere-radius-m", type=float, default=0.05,
                        help="Approximate sphere radius for coarse v/(r*omega) diagnostic")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be positive and finite")
    if args.output_dir is None:
        args.output_dir = args.grid_dir / "recovery_analysis"
    rows, manifest = load_dataset(args.grid_dir)
    if args.sphere_transfer:
        if args.feature_ablation or args.also_object_holdout:
            parser.error("--sphere-transfer already compares fixed feature sets and "
                         "object holdouts; omit other analysis flags")
        save_sphere_transfer(args.output_dir / "sphere_transfer", args.grid_dir,
                             rows, manifest, args.alpha, args.sphere_radius_m)
        return
    n_objects = len({r["object"] for r in rows})
    n_physics = len({_key(r) for r in rows})
    print(f"Loaded {len(rows)} configuration rows from {args.grid_dir}/manifest.json")
    print("Predictors: 11 trajectory-derived features per action; no labels, paths, or motion-target flags")
    print(f"Primary split: leave one (mass, friction) PAIR out across all {n_objects} objects")
    print("Training-fold-only feature scaling; ridge alpha=" + str(args.alpha))
    splits = ("physics", "object") if args.also_object_holdout else ("physics",)
    predictions, scores = _run(rows, args.alpha, splits)
    coverage = coverage_table(rows)
    object_coverage = object_action_set_coverage(rows)
    missing_objects = [r for r in object_coverage
                       if r["actions"] in {"push_x", "push_y", "lateral_flick",
                                           "+".join(NON_LIFT)}
                       and r["n_available"] == 0]
    if missing_objects:
        print("\nCoverage warning: some object/action combinations have NO valid inputs.")
        for c in missing_objects:
            print(f"  NOT EVALUABLE object={c['object']} actions={c['actions']} "
                  f"valid={c['n_available']}/{c['n_total']}")
        print("These objects are absent from those action-set scores; no imputation applied.")
    object_diagnostics = (object_holdout_diagnostics(rows, predictions)
                          if args.also_object_holdout else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "heldout_predictions.csv", predictions)
    save_csv(args.output_dir / "recovery_scores.csv", [r for m in scores for r in _summary_flat(m)])
    save_csv(args.output_dir / "action_coverage_by_mass.csv", coverage)
    save_csv(args.output_dir / "action_set_coverage_by_object.csv", object_coverage)
    if object_diagnostics is not None:
        for filename, entries in object_diagnostics.items():
            save_csv(args.output_dir / (filename + ".csv"), entries)
    ablation_report = (save_feature_ablation(args.output_dir / "feature_ablation",
                                              rows, args.alpha)
                       if args.feature_ablation else None)
    payload = {
        "protocol": "leave-one-joint-physics-combination-out, across all objects",
        "secondary_protocol": ("leave-one-object-out" if args.also_object_holdout else None),
        "n_source_objects": len({r["object"] for r in rows}),
        "n_physics_combinations": len({_key(r) for r in rows}),
        "n_rows": len(rows), "n_frames_per_action": SAMPLE_SHAPE[0],
        "feature_channels": "relative_position_xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz",
        "features_per_action": trajectory_features(_fake_trace(.3, .3, 0, 0))[1],
        "model": "ridge; training-only z-score; unregularized intercept",
        "alpha_fixed_without_test_tuning": args.alpha,
        "baseline": "training fold target mean",
        "lift_policy": "not used in predictive models; report validity and motion separately",
        "metadata_used_as_model_input": False,
        "source_scene_sha256": manifest.get("source_scene_sha256"),
        "source_scenes": manifest.get("source_scenes"),
        "scores": scores, "action_coverage_by_mass": coverage,
        "action_set_coverage_by_object": object_coverage,
        "missingness_policy": "Complete-case by action set; zero-valid objects reported as N/A",
        "object_diagnostics": ({
            "object_transfer_errors": object_diagnostics["object_transfer_errors"],
            "object_feature_shift_summary": object_diagnostics["object_feature_shift_summary"],
            "note": ("Feature shifts use training-carton mean and SD as in ridge. "
                     "First-frame pose is recorded AFTER the first simulation step; "
                     "object-local axes may not be physically comparable. "
                     "Matched motion contrasts do not identify a causal reason for shift."),
        } if object_diagnostics is not None else None),
        "limitations": [
            f"Only {n_objects} simulation geometries and {n_physics} synthetic mass-friction combinations.",
            "Held-out pair evaluation is not extrapolation to new mass/friction values.",
            "Per-object and per-configuration observations are correlated, not 18 independent experiments.",
            "Some actions (especially lift and sphere pushes) were rejected by displacement/tilt thresholds; missingness is systematic.",
            "Action-set evaluations are complete-case, and may exclude whole object geometries.",
            "This does not establish independent identifiability in real scenes or image-based PHINO gains.",
        ],
    }
    (args.output_dir / "recovery_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nRecovery performance (MAE; smaller is better)")
    print(f"{'Split':<8} {'Actions':<32} {'Rows':>4} {'Mass ridge/mean (kg)':>25} {'Friction ridge/mean':>24}")
    print("-" * 102)
    for m in scores:
        mass = m["targets"]["mass_kg"]
        fric = m["targets"]["contact_friction"]
        print(f"{m['split']:<8} {m['actions']:<32} {m['n_rows']:>4} "
              f"{mass['ridge']['mae']:.4f} / {mass['baseline']['mae']:.4f}".rjust(25)
              + f" {fric['ridge']['mae']:.4f} / {fric['baseline']['mae']:.4f}".rjust(24))
    print("  Note: rows and eligible objects vary by action set; see action_set_coverage_by_object.csv")
    if object_diagnostics is not None:
        print("\nObject-transfer breakdown by held-out object (MAE; ridge vs mean baseline):")
        print(f"{'Object':<28} {'Actions':<32} {'Mass (kg)':>16} {'Friction':>16}")
        transfer = object_diagnostics["object_transfer_errors"]
        for obj in sorted({r["object"] for r in rows}):
            if not any(r["heldout_object"] == obj for r in transfer):
                print(f"{obj:<28} N/A: no valid trajectories for evaluated action sets")
                continue
            for aset in ACTION_SETS:
                label = "+".join(aset)
                matches = [r for r in transfer if r["heldout_object"] == obj
                           and r["actions"] == label]
                if not matches:
                    print(f"{obj:<28} {label:<32} N/A (zero valid inputs)")
                    continue
                m = next(r for r in matches if r["target"] == "mass_kg")
                f = next(r for r in matches if r["target"] == "contact_friction")
                print(f"{obj:<28} {label:<32} "
                      f"{m['ridge_mae']:.4f}/{m['baseline_mae']:.4f} "
                      f"{f['ridge_mae']:.4f}/{f['baseline_mae']:.4f}")
            full = "+".join(NON_LIFT)
            shifted = [r for r in object_diagnostics["object_feature_shift"]
                       if r["heldout_object"] == obj and r["actions"] == full]
            shifted = sorted(shifted, key=lambda r: abs(r["test_mean_z_vs_train"]),
                             reverse=True)
            print(f"  Largest combined-action feature shifts when {obj} is held out:")
            for r in shifted[:4]:
                print(f"    {r['feature']}: test_mean_z={r['test_mean_z_vs_train']:+.2f}, "
                      f"outside_train_range={r['test_outside_train_minmax_fraction']:.0%}")
        print("  Warning: extreme z-scores can arise from tiny training variance.")
        print("  First-saved-frame quaternions are not pre-action poses or proof of tipping.")
        print(f"OBJECT_TRANSFER_ERRORS_CSV={(args.output_dir / 'object_transfer_errors.csv').resolve()}")
        print(f"OBJECT_FEATURE_SHIFT_CSV={(args.output_dir / 'object_feature_shift.csv').resolve()}")
        print(f"OBJECT_FEATURE_SHIFT_SUMMARY_CSV={(args.output_dir / 'object_feature_shift_summary.csv').resolve()}")
        print(f"FIRST_FRAME_ORIENTATION_CSV={(args.output_dir / 'first_saved_frame_orientation.csv').resolve()}")
        print(f"MATCHED_OBJECT_MOTION_CSV={(args.output_dir / 'matched_object_motion.csv').resolve()}")
    if ablation_report is not None:
        print("\nFeature ablation: physics and object holdout MAE "
              "(ridge / training-mean baseline; smaller is better)")
        print(f"{'Split':<8} {'Feature set':<23} {'Actions':<32} "
              f"{'Mass ridge / mean':>20} {'Friction ridge / mean':>22}")
        print("-" * 113)
        for metric in ablation_report["scores"]:
            m = metric['targets']['mass_kg']
            f = metric['targets']['contact_friction']
            print(f"{metric['split']:<8} {metric['feature_set']:<23} {metric['actions']:<32} "
                  f"{m['ridge']['mae']:.4f} / {m['baseline']['mae']:.4f}".rjust(20)
                  + f" {f['ridge']['mae']:.4f} / {f['baseline']['mae']:.4f}".rjust(22))
        print("\nPer-object breakdown for all-three-action model (ridge / mean):")
        for feature_set in FEATURE_SETS:
            for obj in sorted({r["object"] for r in rows}):
                selected = [r for r in ablation_report["object_breakdown"]
                            if r["feature_set"] == feature_set
                            and r["actions"] == "+".join(NON_LIFT)
                            and r["heldout_object"] == obj]
                mass = next(r for r in selected if r["target"] == "mass_kg")
                friction = next(r for r in selected if r["target"] == "contact_friction")
                if mass["n_test"] == 0:
                    print(f"  {feature_set} -> {obj}: N/A (no complete valid "
                          f"push_x+push_y+lateral_flick trajectory)")
                else:
                    print(f"  {feature_set} -> {obj}: "
                          f"mass {mass['ridge_mae']:.4f}/{mass['baseline_mae']:.4f}; "
                          f"friction {friction['ridge_mae']:.4f}/{friction['baseline_mae']:.4f}")
        ablation_dir = (args.output_dir / "feature_ablation").resolve()
        print(f"FEATURE_ABLATION_DIR={ablation_dir}")
        print(f"FEATURE_ABLATION_SCORES_CSV={ablation_dir / 'feature_ablation_scores.csv'}")
        print(f"FEATURE_ABLATION_OBJECT_CSV={ablation_dir / 'feature_ablation_object_breakdown.csv'}")
        print(f"FEATURE_ABLATION_COVERAGE_CSV={ablation_dir / 'action_set_coverage_by_object.csv'}")
    print("\nLift coverage by mass (descriptive only; no imputation):")
    for row in coverage:
        if row["action"] == "grasp_lift_release":
            print(f"  m={row['mass_kg']:.3f} kg: valid={row['valid']}/{row['total']}, "
                  f"target_met={row['target_met']}/{row['total']}")
    print(f"PREDICTIONS_CSV={(args.output_dir / 'heldout_predictions.csv').resolve()}")
    print(f"SCORES_CSV={(args.output_dir / 'recovery_scores.csv').resolve()}")
    print(f"COVERAGE_CSV={(args.output_dir / 'action_coverage_by_mass.csv').resolve()}")
    print(f"OBJECT_COVERAGE_CSV={(args.output_dir / 'action_set_coverage_by_object.csv').resolve()}")
    print(f"ANALYSIS_JSON={(args.output_dir / 'recovery_analysis.json').resolve()}")
    print("NOTE: Synthetic trajectory-based diagnostic only; no PHINO/DINOv2 model trained")
    print("PHINO_PROPERTY_RECOVERY_ANALYSIS_OK")


if __name__ == "__main__":
    main()

