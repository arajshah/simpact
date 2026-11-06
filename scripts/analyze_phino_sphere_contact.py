#!/usr/bin/env python3
"""Check whether sphere motion is rolling, sliding, or yaw spinning.

The PHINO trajectory stores freejoint linear velocity in WORLD coordinates
and freejoint angular velocity in BODY coordinates (MuJoCo convention).
This script rotates angular velocity into world coordinates before forming the
nominal bottom-contact velocity v + omega x [0,0,-R].

A zero nominal contact speed is consistent with ideal no-slip rolling, but
does NOT prove contact or physical fidelity. The radius is approximate.

Run:
  python scripts/analyze_phino_sphere_contact.py --self-test
  python scripts/analyze_phino_sphere_contact.py \
      --grid-dir results/phino_mesh_grid_cutoff_1m_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def rotate_body_vectors_to_world(quaternion_wxyz, vectors_body):
    """Vectorized unit-quaternion rotation R(q) @ v, no SciPy needed."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    v = np.asarray(vectors_body, dtype=np.float64)
    if q.shape != v.shape[:-1] + (4,) or v.shape[-1] != 3:
        raise ValueError(f"Incompatible quaternion/vector shapes {q.shape}, {v.shape}")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norms < 1e-9) or not np.all(np.isfinite(v)):
        raise ValueError("Nonfinite or zero quaternion, or nonfinite vector")
    q = q / norms
    axis = q[..., 1:]
    uv = np.cross(axis, v)
    return v + 2.0 * (q[..., :1] * uv + np.cross(axis, uv))


def contact_metrics(trace, radius_m, frame_dt_s):
    """Kinematic diagnostics for a nominal spherical bottom point on flat z=0."""
    a = np.asarray(trace, dtype=np.float64)
    if a.shape != (50, 13) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected finite (50,13) sphere trajectory, got {a.shape}")
    if not (math.isfinite(radius_m) and radius_m > 0 and math.isfinite(frame_dt_s)
            and frame_dt_s > 0):
        raise ValueError("Radius and frame time must be positive and finite")
    pos = a[:, :3]
    vel_world = a[:, 7:10]
    omega_world = rotate_body_vectors_to_world(a[:, 3:7], a[:, 10:13])

    vxy = np.linalg.norm(vel_world[:, :2], axis=1)
    omega_xy = np.linalg.norm(omega_world[:, :2], axis=1)
    omega_z = np.abs(omega_world[:, 2])
    omega_total_sq = np.sum(omega_world ** 2, axis=1)

    # Velocity at nominal bottom contact relative to the stationary floor.
    # omega x (0,0,-R) = (-R*omega_y, +R*omega_x, 0).
    slip_xy = np.column_stack([
        vel_world[:, 0] - radius_m * omega_world[:, 1],
        vel_world[:, 1] + radius_m * omega_world[:, 0],
    ])
    slip_speed = np.linalg.norm(slip_xy, axis=1)
    roll_speed = radius_m * omega_xy
    active = (vxy > 0.01) | (roll_speed > 0.01)
    denom_ok = (vxy > 0.01) & (roll_speed > 0.01)
    ratio = vxy[denom_ok] / roll_speed[denom_ok]
    spin_frac = (float(np.sum(omega_z ** 2) / np.sum(omega_total_sq))
                 if np.sum(omega_total_sq) > 1e-12 else None)
    ratio_median = float(np.median(ratio)) if len(ratio) else None
    result = {
        "frames": 50,
        "radius_assumed_m": radius_m,
        "final_xy_displacement_m": float(np.linalg.norm(pos[-1, :2])),
        "max_xy_displacement_m": float(np.max(np.linalg.norm(pos[:, :2], axis=1))),
        "xy_path_length_m": float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))),
        "peak_xy_speed_m_s": float(np.max(vxy)),
        "peak_horizontal_angular_speed_rad_s": float(np.max(omega_xy)),
        "peak_yaw_angular_speed_rad_s": float(np.max(omega_z)),
        "yaw_angular_speed_squared_fraction": spin_frac,
        "active_frames": int(np.sum(active)),
        "mean_nominal_contact_slip_m_s_active": (
            float(np.mean(slip_speed[active])) if np.any(active) else None),
        "median_nominal_contact_slip_m_s_active": (
            float(np.median(slip_speed[active])) if np.any(active) else None),
        "fraction_active_contact_slip_under_0p05_m_s": (
            float(np.mean(slip_speed[active] < 0.05)) if np.any(active) else None),
        "median_v_over_R_omega_horizontal": ratio_median,
        "ratio_defined_frames": int(np.sum(denom_ok)),
    }
    traces = []
    for i in range(len(a)):
        traces.append({
            "frame": i, "time_from_first_frame_s": float(i * frame_dt_s),
            "xy_displacement_m": float(np.linalg.norm(pos[i, :2])),
            "v_x_world_m_s": float(vel_world[i, 0]),
            "v_y_world_m_s": float(vel_world[i, 1]),
            "omega_x_world_rad_s": float(omega_world[i, 0]),
            "omega_y_world_rad_s": float(omega_world[i, 1]),
            "omega_z_world_rad_s": float(omega_world[i, 2]),
            "omega_horizontal_rad_s": float(omega_xy[i]),
            "nominal_bottom_contact_slip_x_m_s": float(slip_xy[i, 0]),
            "nominal_bottom_contact_slip_y_m_s": float(slip_xy[i, 1]),
            "nominal_bottom_contact_slip_speed_m_s": float(slip_speed[i]),
        })
    return result, traces


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    q = np.tile(np.array([1., 0., 0., 0.]), (50, 1))
    a = np.zeros((50, 13))
    a[:, 3:7] = q
    a[:, 7] = .1
    a[:, 11] = 2.0  # +Y rotation -> bottom contact velocity -0.1 X
    m, _ = contact_metrics(a, .05, .05)
    assert m["mean_nominal_contact_slip_m_s_active"] < 1e-10
    assert np.isclose(m["median_v_over_R_omega_horizontal"], 1)
    assert m["yaw_angular_speed_squared_fraction"] == 0
    a[:, 10:13] = (0., 0., 10.)
    m, _ = contact_metrics(a, .05, .05)
    assert np.isclose(m["mean_nominal_contact_slip_m_s_active"], .1)
    assert np.isclose(m["yaw_angular_speed_squared_fraction"], 1)
    assert m["median_v_over_R_omega_horizontal"] is None
    a[:, 10:13] = 0
    m, _ = contact_metrics(a, .05, .05)
    assert np.isclose(m["mean_nominal_contact_slip_m_s_active"], .1)
    assert m["yaw_angular_speed_squared_fraction"] is None
    # 90 degree yaw turns body +X rotation into world +Y.
    q90 = np.tile(np.array([2 ** -.5, 0., 0., 2 ** -.5]), (50, 1))
    a[:, 3:7] = q90
    a[:, 10] = 2.
    m, _ = contact_metrics(a, .05, .05)
    assert m["mean_nominal_contact_slip_m_s_active"] < 1e-10
    print("PHINO_SPHERE_CONTACT_SELF_TEST_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path,
                        default=Path("results/phino_mesh_grid_cutoff_1m_v1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--radius-m", type=float, default=.05)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    grid = args.grid_dir.resolve()
    manifest_path = grid / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("action_order") != [
        "push_x", "push_y", "grasp_lift_release", "lateral_flick"]:
        parser.error("Unexpected action_order")
    if manifest.get("frames_per_action") != 50:
        parser.error("Expected 50-frame trajectories")
    frame_dt = 1. / float(manifest.get("sampling_Hz", 20.))
    if not (math.isfinite(frame_dt) and frame_dt > 0):
        parser.error("Invalid frame rate")
    if not (math.isfinite(args.radius_m) and args.radius_m > 0):
        parser.error("--radius-m must be finite and positive")
    spheres = sorted({rec["object"] for rec in manifest["records"]
                      if rec["object"].endswith("__mesh_sphere")})
    if len(spheres) != 1:
        parser.error(f"Expected one sphere, found {spheres}")
    sphere = spheres[0]
    rows = []
    per_frame = []
    for rec in manifest["records"]:
        if rec["object"] != sphere:
            continue
        for action in ("push_x", "push_y"):
            meta = {"object": sphere, "mass_kg": float(rec["mass_kg"]),
                    "contact_friction": float(rec["contact_friction"]), "action": action}
            status = rec["action_status"].get(action, {})
            if status.get("valid") is not True:
                rows.append({**meta, "valid": False,
                             "rejection_reason": status.get("reason", "missing"),
                             "max_tilt_deg_from_generator": None})
                continue
            rel = status.get("trajectory_path")
            if not rel:
                raise ValueError(f"No trajectory path: {meta}")
            path = grid / rel
            if not path.is_file():
                # Portability fallback for manifests from different machines.
                path = grid / "action_trajectories" / Path(rel).name
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                trace = np.asarray(data["trajectory"])
                if "mass_kg" in data and not np.isclose(
                        float(data["mass_kg"]), meta["mass_kg"]):
                    raise ValueError(f"Mass metadata mismatch {path}")
                if "contact_friction" in data and not np.isclose(
                        float(data["contact_friction"]), meta["contact_friction"]):
                    raise ValueError(f"Friction metadata mismatch {path}")
            measures, sequence = contact_metrics(trace, args.radius_m, frame_dt)
            rows.append({**meta, "valid": True, "rejection_reason": "",
                         "max_tilt_deg_from_generator": rec.get(
                             "action_metrics", {}).get(action, {}).get("max_tilt_deg"),
                         **measures})
            per_frame.extend({**meta, **frame} for frame in sequence)

    if not rows:
        raise ValueError("No sphere trials in the grid")
    # Ensure uniform CSV fields even if one sphere action was rejected.
    columns = list(dict.fromkeys(key for entry in rows for key in entry))
    for row in rows:
        for key in columns:
            row.setdefault(key, None)
    output = (args.output_dir or
              grid / "recovery_analysis" / "sphere_contact_diagnostic").resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "summary.csv", rows)
    if per_frame:
        save_csv(output / "frames.csv", per_frame)
    report = {
        "purpose": "Separate nominal sphere rolling versus yaw-spinning diagnostic",
        "sphere": sphere, "grid": str(grid), "radius_assumed_m": args.radius_m,
        "angular_velocity": "MuJoCo qvel angular BODY frame rotated using qpos quaternion",
        "linear_velocity": "MuJoCo qvel linear WORLD frame",
        "contact_velocity": "v_world + cross(omega_world, [0,0,-R])",
        "limitations": [
            "Nominal bottom point assumes sphere radius and centered contact.",
            "No actual MuJoCo contact or contact forces are inspected.",
            "A low nominal slip speed is consistent with, not proof of, rolling.",
            "A high vertical spin fraction indicates yaw spin, not necessarily rolling.",
            "Finite speed and complete trajectories do not prove physical realism.",
        ],
        "trials": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                        encoding="utf-8")
    for action in ("push_x", "push_y"):
        sub = [r for r in rows if r["action"] == action and r["valid"]]
        if sub:
            spin = [r["yaw_angular_speed_squared_fraction"] for r in sub
                    if r["yaw_angular_speed_squared_fraction"] is not None]
            slip = [r["median_nominal_contact_slip_m_s_active"] for r in sub
                    if r["median_nominal_contact_slip_m_s_active"] is not None]
            tilt = [r["max_tilt_deg_from_generator"] for r in sub
                    if r["max_tilt_deg_from_generator"] is not None]
            print(f"{action}: valid={len(sub)}/9, "
                  f"median yaw-spin fraction={np.median(spin) if spin else float('nan'):.3f}, "
                  f"median nominal slip={np.median(slip) if slip else float('nan'):.3f} m/s, "
                  f"median max tilt={np.median(tilt) if tilt else float('nan'):.1f} deg")
    print(f"SPHERE_CONTACT_REPORT={output / 'summary.csv'}")
    print("PHINO_SPHERE_CONTACT_DIAGNOSTIC_OK")


if __name__ == "__main__":
    main()
