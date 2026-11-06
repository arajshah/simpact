  #!/usr/bin/env python3
"""Analyze PHINO/SIMPACT mass x friction sensitivity, with optional MuJoCo follow-up.

Reads the physics sweep produced by calibrate_phino_interventions.py --mode physics.
Only compares simulations of the *same object* while one physical parameter
changes and the other stays fixed. The intervention remains unchanged.

Outputs a human-readable table, matched_contrasts.csv and analysis.json.
This is a deterministic sensitivity diagnostic, NOT a trained physics probe.

Without flags: preserve original offline analysis. With --diagnose-actions: reuse
existing calibration helpers to test zero-torque flicks and revised pushes.
With --flick-focus, run a finer torque sweep; --render-flick saves inspectable GIFs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PARAMETERS = {"mass": "mass_kg", "friction": "contact_friction"}
MOTION_FEATURES = ("position", "linear_velocity", "angular_velocity")


def median_or_none(numbers):
    return float(np.median(numbers)) if numbers else None


def label(x):
    return "n/a" if x is None else f"{x:.4f}"


def position_rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a[:, :3] - b[:, :3]) ** 2, axis=1))))


def vector_rms(a, b, start, stop):
    return float(np.sqrt(np.mean(np.sum((a[:, start:stop] - b[:, start:stop]) ** 2, axis=1))))


def orientation_angle_rms_deg(a, b):
    # q and -q represent the same orientation. Quaternion convention wxyz.
    qa, qb = a[:, 3:7].astype(float), b[:, 3:7].astype(float)
    qa /= np.maximum(np.linalg.norm(qa, axis=1, keepdims=True), 1e-12)
    qb /= np.maximum(np.linalg.norm(qb, axis=1, keepdims=True), 1e-12)
    dots = np.clip(np.abs(np.sum(qa * qb, axis=1)), 0.0, 1.0)
    angles_deg = np.degrees(2 * np.arccos(dots))
    return float(np.sqrt(np.mean(angles_deg ** 2)))


def normalized_trial(r):
    for field in ("object", "action_id", "mass_kg", "contact_friction", "status"):
        if field not in r:
            raise ValueError(f"Sweep record missing field {field}: {r}")
    return {
        **r,
        "mass_kg": float(r["mass_kg"]),
        "contact_friction": float(r["contact_friction"]),
    }


def load_sweep(report_file, trajectories_file):
    if not report_file.is_file():
        raise FileNotFoundError(f"Sweep report not found: {report_file}")
    if not trajectories_file.is_file():
        raise FileNotFoundError(f"Trajectories not found: {trajectories_file}")

    report = json.loads(report_file.read_text())
    if report.get("kind") != "physics_sensitivity_validation_not_training":
        raise ValueError("Unexpected report kind; expected a PHINO physics sweep")
    records = [normalized_trial(r) for r in report["results"]]
    traces = {}
    with np.load(trajectories_file, allow_pickle=False) as npz:
        for r in records:
            if r["status"] != "ok":
                continue
            key = r.get("trajectory_key")
            if not key or key not in npz:
                raise ValueError(f"Missing stored trajectory for {r}")
            trace = np.asarray(npz[key], dtype=np.float64)
            if trace.ndim != 2 or trace.shape[1] != 13 or trace.shape[0] < 2:
                raise ValueError(f"Unexpected trajectory shape {trace.shape} for {key}")
            if not np.isfinite(trace).all():
                raise ValueError(f"Nonfinite trajectory {key}")
            traces[key] = trace

    duplicates = Counter((r["object"], r["action_id"], r["mass_kg"], r["contact_friction"])
                         for r in records)
    if any(n > 1 for n in duplicates.values()):
        raise ValueError("Duplicate (object, action, mass, friction) trials; expected one each")
    return report, records, traces


def compare_one_parameter(records, traces, report):
    """Only *adjacent grid levels* are compared, within one object and action.

    A rejected trial is never imputed or counted as a valid observation.
    """
    valid = {(r["object"], r["action_id"], r["mass_kg"], r["contact_friction"]): r
             for r in records if r["status"] == "ok"}
    objects = sorted({r["object"] for r in records})
    actions = sorted({r["action_id"] for r in records})
    mass_levels = sorted({float(v) for v in report.get("masses_kg", [])}
                         or {r["mass_kg"] for r in records})
    friction_levels = sorted({float(v) for v in report.get("contact_friction_values", [])}
                             or {r["contact_friction"] for r in records})
    contrasts = []
    coverage = defaultdict(lambda: {"possible": 0, "valid": 0})

    for obj in objects:
        for action in actions:
            for target in ("mass", "friction"):
                if target == "mass":
                    varying_levels, held_levels = mass_levels, friction_levels
                else:
                    varying_levels, held_levels = friction_levels, mass_levels

                for held in held_levels:
                    for low, high in zip(varying_levels, varying_levels[1:]):
                        cov = coverage[(obj, action, target)]
                        cov["possible"] += 1
                        if target == "mass":
                            k1 = (obj, action, low, held)
                            k2 = (obj, action, high, held)
                        else:
                            k1 = (obj, action, held, low)
                            k2 = (obj, action, held, high)
                        if k1 not in valid or k2 not in valid:
                            continue
                        cov["valid"] += 1
                        a = traces[valid[k1]["trajectory_key"]]
                        b = traces[valid[k2]["trajectory_key"]]
                        contrasts.append({
                            "object": obj, "action_id": action, "varied": target,
                            "held_constant_value": held,
                            "varied_low": low, "varied_high": high,
                            "position_rms_m": position_rms(a, b),
                            "linear_velocity_rms_m_per_s": vector_rms(a, b, 7, 10),
                            "angular_velocity_rms_rad_per_s": vector_rms(a, b, 10, 13),
                            "orientation_rms_deg": orientation_angle_rms_deg(a, b),
                        })
    return contrasts, coverage, actions, objects


def aggregate(records, contrasts, cov, actions, objects):
    summary = []
    for action in actions:
        action_records = [r for r in records if r["action_id"] == action]
        accepted = [r for r in action_records if r["status"] == "ok"]
        flips = sum(float(r.get("metrics", {}).get("max_rotation_deg", 0)) > 120
                    for r in accepted)
        clipped = sum(bool(r.get("metrics", {}).get("would_clip_PHINO_velocities", False))
                      for r in accepted)
        motion_target = sum(bool(r.get("motion_target_met", False)) for r in accepted)
        rejection_reasons = Counter(
            (r.get("reason") or "unknown").split(" at t=")[0]
            for r in action_records if r["status"] != "ok")
        out = {
            "action_id": action,
            "accepted": len(accepted), "total": len(action_records),
            "accepted_fraction": len(accepted) / len(action_records) if action_records else 0,
            "motion_target_met": motion_target,
            "flips_over_120deg": flips,
            "velocity_clipping_warnings": clipped,
            "rejection_reasons": dict(rejection_reasons),
        }
        for target in ("mass", "friction"):
            subset = [c for c in contrasts if c["action_id"] == action and c["varied"] == target]
            expected = sum(cov[(obj, action, target)]["possible"] for obj in objects)
            out[target] = {
                "valid_adjacent_contrasts": len(subset),
                "possible_adjacent_contrasts": expected,
                "median_position_rms_m": median_or_none([c["position_rms_m"] for c in subset]),
                "median_orientation_rms_deg": median_or_none([c["orientation_rms_deg"] for c in subset]),
                "median_linear_velocity_rms_m_per_s": median_or_none(
                    [c["linear_velocity_rms_m_per_s"] for c in subset]),
            }
        summary.append(out)
    return summary


def print_results(summary, objects, records):
    n = len(records)
    n_ok = sum(r["status"] == "ok" for r in records)
    print(f"Objects: {', '.join(objects)}")
    print(f"Trials accepted: {n_ok}/{n}; rejected: {n-n_ok}/{n}")
    print("Matched differences: adjacent mass values at fixed friction, and vice versa.")
    print("Each contrast uses the same object and same applied action schedule.\n")
    print("Action             Valid   Target   Flips  Mass RMS (pairs)   Friction RMS (pairs)")
    print("-" * 88)
    for a in summary:
        m, f = a["mass"], a["friction"]
        print(f"{a['action_id']:18s} {a['accepted']:2d}/{a['total']:<2d} "
              f" {a['motion_target_met']:2d}/{a['accepted']:<2d}  "
              f" {a['flips_over_120deg']:3d} "
              f"  {label(m['median_position_rms_m']):>6s} "
              f"({m['valid_adjacent_contrasts']:2d}/{m['possible_adjacent_contrasts']:2d})"
              f"    {label(f['median_position_rms_m']):>6s} "
              f"({f['valid_adjacent_contrasts']:2d}/{f['possible_adjacent_contrasts']:2d})")
    print("\nCautions:")
    print("- RMS is in metres and compares complete 50-frame position trajectories, not static images.")
    print("- A nonzero RMS means sensitivity, NOT independent physical parameter identifiability.")
    print("- Missing/rejected trajectories bias sensitivity comparisons; pair counts matter.")
    print("- A >120-degree rotation may be real tipping or an undesirable collision/contact artefact.")
    print("- Synthetic mass/friction are not measured real-world labels; only two source objects.")
    print("- No held-out generalization, repeatability, statistical significance, or DINOv2 training is tested.")


def write_csv(path, contrasts):
    fieldnames = ["object", "action_id", "varied", "held_constant_value",
                  "varied_low", "varied_high", "position_rms_m",
                  "linear_velocity_rms_m_per_s", "angular_velocity_rms_rad_per_s",
                  "orientation_rms_deg"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=Path("results/phino_calibration/physics_sweep.json"))
    parser.add_argument("--trajectories", type=Path,
                        default=Path("results/phino_calibration/physics_sweep_trajectories.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phino_calibration/sensitivity_analysis"))
    args = parser.parse_args()
    report, records, traces = load_sweep(args.report, args.trajectories)
    contrasts, cov, actions, objects = compare_one_parameter(records, traces, report)
    summary = aggregate(records, contrasts, cov, actions, objects)
    print_results(summary, objects, records)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "matched_contrasts.csv"
    json_path = args.out / "analysis.json"
    write_csv(csv_path, contrasts)
    result = {
        "kind": "diagnostic_analysis_not_training",
        "source_report": str(args.report),
        "source_trajectories": str(args.trajectories),
        "method": "Adjacent one-parameter matched trajectory contrasts within object/action",
        "summary": summary,
        "per_object_coverage": [{"object": obj, "action_id": act, "varied": param, **cov[(obj, act, param)]}
                                for obj in objects for act in actions for param in PARAMETERS],
        "contrasts": contrasts,
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nANALYSIS_JSON={json_path}")
    print(f"MATCHED_CONTRASTS_CSV={csv_path}")
    print("SENSITIVITY_ANALYSIS_OK")



# ---------------------------------------------------------------------------
# Optional MuJoCo follow-up: isolate flick torque and refine push schedules.
# This mode leaves the original offline sensitivity analysis unchanged.
# ---------------------------------------------------------------------------

def _load_simulation_helpers(script_dir: Path):
    """Reuse the proven isolated-scene and physics setup from calibration."""
    import importlib.util

    choices = (
        script_dir / "calibrate_phino_interventions.py",
        script_dir / "calibrate_phino_interactions.py",  # local renamed copy
    )
    source = next((p for p in choices if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "Place calibrate_phino_interventions.py (or your existing "
            "calibrate_phino_interactions.py) alongside this analysis script. "
            "The diagnostic mode reuses its physics/scene helpers."
        )
    spec = importlib.util.spec_from_file_location("phino_calibration_helpers", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import calibration helpers from {source}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for method in ("isolate_objects", "apply_physics_to_object", "assert_valid", "RejectedTrial"):
        if not hasattr(mod, method):
            raise RuntimeError(f"Calibration script {source.name} lacks {method}")
    return mod


def _quat_mul(a, b):
    """Quaternion multiplication; w,x,y,z convention."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=float)


def _quat_rotate(a, v):
    """Rotate vector v by wxyz quaternion a."""
    a = np.asarray(a, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-12)
    xyz = a[1:]
    t = 2.0 * np.cross(xyz, v)
    return np.asarray(v, dtype=float) + a[0] * t + np.cross(xyz, t)


def _motion_and_tilt(q_initial, q_current):
    """Total orientation difference and net tilt w.r.t. world vertical.

    Net tilt is the angle through which the *world-up vector* is rotated by
    the pose change; pure yaw contributes zero. This is an orientation diagnostic,
    not a direct measure of contact, buckling, or tipping mechanism.
    """
    q0 = np.asarray(q_initial, dtype=float)
    q1 = np.asarray(q_current, dtype=float)
    q0 /= max(np.linalg.norm(q0), 1e-12)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    angle = float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0., 1.))))
    q_relative = _quat_mul(q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
    new_up = _quat_rotate(q_relative, (0., 0., 1.))
    tilt = float(np.degrees(np.arccos(np.clip(new_up[2], -1., 1.))))
    return angle, tilt


def _world_vertical_twist_deg(q_initial, q_current):
    """Signed yaw/twist component of the *relative* wxyz rotation.

    This is a world-vertical swing-twist decomposition, NOT a cumulative turn
    counter. It is undefined for some 180-degree horizontal-axis rotations.
    Use together with max_tilt_deg and the rendered motion, not in isolation.
    """
    q0 = np.asarray(q_initial, dtype=float)
    q1 = np.asarray(q_current, dtype=float)
    q0 /= max(np.linalg.norm(q0), 1e-12)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    qrel = _quat_mul(q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
    w, z = float(qrel[0]), float(qrel[3])
    if np.hypot(w, z) < 1e-8:
        return None
    return float(np.degrees(2 * np.arctan2(z, w) + np.pi) % 360 - 180)


def _diagnostic_trial(obj, action, *, force_scale, torque_multiplier,
                      duration_frames, engine, safety_limit, helpers, mujoco):
    """Run a *fresh* 50-frame trial; never clip positions/velocities.

    Push: same PHINO force/torque direction; vary force scale and duration.
    Flick: same calibrated force, but vary the *torque multiplier alone*.
    """
    m, bid, qadr, vadr = (obj[k] for k in ("model", "bid", "qadr", "vadr"))
    d = mujoco.MjData(m)
    d.qpos[:] = obj["q0"]
    d.qvel[:] = obj["v0"]
    mujoco.mj_forward(m, d)
    helpers.assert_valid(d)
    q_initial = d.qpos[qadr+3:qadr+7].copy()
    T = engine.timesteps
    traj = np.empty((T, 13), dtype=np.float32)
    max_raw = max_directed = max_rot = max_tilt = 0.0
    max_linear_speed = max_angular_speed = 0.0
    max_yaw_twist_deg = 0.0
    undefined_twist_frames = 0
    for t in range(T):
        force = np.zeros(3, dtype=float)
        torque = np.zeros(3, dtype=float)
        if action.name in ("push_x", "push_y"):
            if t < duration_frames:
                force = action.direction * action.force * force_scale
                # Match the original calibrated push's force/torque coupling.
                torque[2] = action.torque * force_scale
        elif action.name == "lateral_flick":
            if t < 4:
                force = action.direction * action.force * force_scale
                torque[2] = action.torque * force_scale * torque_multiplier
        else:
            raise ValueError(f"Unexpected diagnostic action {action.name}")
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[bid, :3] = force
        d.xfrc_applied[bid, 3:] = torque
        for _ in range(engine.substeps):
            mujoco.mj_step(m, d)
            helpers.assert_valid(d)
            delta = d.qpos[qadr:qadr+3] - obj["start"]
            raw = float(np.linalg.norm(delta))
            if not np.isfinite(raw) or raw > safety_limit:
                raise helpers.RejectedTrial(
                    f"raw displacement {raw:.4f} m exceeds {safety_limit:.3f} m "
                    f"at t={d.time:.3f}s")
            max_raw = max(max_raw, raw)
            max_directed = max(max_directed, float(np.dot(delta, action.direction)))
            angle, tilt = _motion_and_tilt(q_initial, d.qpos[qadr+3:qadr+7])
            max_rot = max(max_rot, angle)
            max_tilt = max(max_tilt, tilt)
            yaw = _world_vertical_twist_deg(q_initial, d.qpos[qadr+3:qadr+7])
            if yaw is None:
                undefined_twist_frames += 1
            else:
                max_yaw_twist_deg = max(max_yaw_twist_deg, abs(yaw))
            max_linear_speed = max(max_linear_speed,
                                   float(np.linalg.norm(d.qvel[vadr:vadr+3])))
            max_angular_speed = max(max_angular_speed,
                                    float(np.linalg.norm(d.qvel[vadr+3:vadr+6])))
        traj[t, :3] = d.qpos[qadr:qadr+3] - obj["start"]
        traj[t, 3:7] = d.qpos[qadr+3:qadr+7]
        traj[t, 7:10] = d.qvel[vadr:vadr+3]
        traj[t, 10:13] = d.qvel[vadr+3:vadr+6]
    if not np.isfinite(traj).all():
        raise helpers.RejectedTrial("nonfinite 13-D trajectory")
    return {
        "directed_m": max_directed,
        "max_raw_m": max_raw,
        "max_orientation_change_deg": max_rot,
        "max_tilt_deg": max_tilt,
        "max_world_yaw_twist_deg": max_yaw_twist_deg,
        "undefined_twist_steps": undefined_twist_frames,
        "max_linear_speed_m_s": max_linear_speed,
        "max_angular_speed_rad_s": max_angular_speed,
        "would_clip_PHINO_velocities": bool(
            max_linear_speed > engine.lin_vel_clip or
            max_angular_speed > engine.ang_vel_clip),
    }, traj


def _parse_positive_csv(s: str, *, permit_zero=False, integer=False):
    result = []
    for value in s.split(","):
        try:
            n = int(value.strip()) if integer else float(value.strip())
        except ValueError as ex:
            raise ValueError(f"Invalid number in grid: {value!r}") from ex
        if not np.isfinite(n) or (n < 0 if permit_zero else n <= 0):
            raise ValueError(f"Invalid grid value: {n}")
        result.append(n)
    if not result:
        raise ValueError("Empty grid")
    return sorted(set(result))


def _summarize_candidates(rows, trajectories, candidate_specs, mass_levels, friction_levels,
                          min_motion, max_motion, max_tilt):
    """Aggregate coverage and matched contrasts per candidate, not per object."""
    from collections import defaultdict as _defaultdict
    by_candidate = _defaultdict(list)
    for r in rows:
        by_candidate[r["candidate_id"]].append(r)
    summaries = []
    for candidate_id, desc in candidate_specs.items():
        candidates = by_candidate[candidate_id]
        n = len(candidates)
        valid = [r for r in candidates if r["status"] == "ok"]
        good = [r for r in valid if r["target_and_tilt_met"]]
        large_orientation = sum(r["metrics"]["max_orientation_change_deg"] > 120
                                for r in valid)
        large_tilt = sum(r["metrics"]["max_tilt_deg"] > max_tilt for r in valid)
        big_tipping = sum(
            r["metrics"]["max_orientation_change_deg"] > 120
            and r["metrics"]["max_tilt_deg"] > max_tilt for r in valid)
        big_spin_dominant = sum(
            r["metrics"]["max_orientation_change_deg"] > 120
            and r["metrics"]["max_tilt_deg"] <= max_tilt for r in valid)
        lookup = {(r["object"], r["mass_kg"], r["contact_friction"]):r
                  for r in valid}
        contrast = {"mass": [], "friction": []}
        possible = {"mass":0, "friction":0}
        for obj in sorted({r["object"] for r in candidates}):
            for mu in friction_levels:
                for m0, m1 in zip(mass_levels, mass_levels[1:]):
                    possible["mass"] += 1
                    a, b = lookup.get((obj,m0,mu)), lookup.get((obj,m1,mu))
                    if a is not None and b is not None:
                        contrast["mass"].append(position_rms(
                            trajectories[a["trajectory_key"]], trajectories[b["trajectory_key"]]))
            for mass in mass_levels:
                for f0, f1 in zip(friction_levels, friction_levels[1:]):
                    possible["friction"] += 1
                    a, b = lookup.get((obj,mass,f0)), lookup.get((obj,mass,f1))
                    if a is not None and b is not None:
                        contrast["friction"].append(position_rms(
                            trajectories[a["trajectory_key"]], trajectories[b["trajectory_key"]]))
        def pack(vals, total):
            return {"valid":len(vals), "possible":total, "median_position_rms_m":median_or_none(vals)}
        summaries.append({
            "candidate_id":candidate_id, "spec":desc,
            "valid":len(valid), "total":n,
            "meets_motion_target":sum(r["motion_target_met"] for r in valid),
            "meets_target_and_tilt":len(good),
            "large_orientation_gt_120deg":large_orientation,
            "tilt_gt_limit":large_tilt,
            "large_orientation_and_tilt":big_tipping,
            "large_orientation_low_tilt":big_spin_dominant,
            "median_max_tilt_deg":median_or_none(
                [r["metrics"]["max_tilt_deg"] for r in valid]),
            "median_max_world_yaw_twist_deg":median_or_none(
                [r["metrics"]["max_world_yaw_twist_deg"] for r in valid]),
            "mass_matched":pack(contrast["mass"],possible["mass"]),
            "friction_matched":pack(contrast["friction"],possible["friction"]),
        })
    return summaries




def _render_representative_flick(obj, action, *, force_scale, torque_multiplier,
                                 engine, helpers, mujoco, output_path: Path,
                                 image_size: int = 256, stride: int = 3):
    """Re-run nominal flick and save a fixed-camera, two-angle diagnostic GIF.

    Left and right cameras are fixed in world space. A moving follow camera can
    hide sliding or tipping, so keep the focus on the initial settled object.
    """
    from PIL import Image, ImageDraw

    m, bid, qadr, vadr = (obj[k] for k in ("model", "bid", "qadr", "vadr"))
    d = mujoco.MjData(m)
    d.qpos[:] = obj["q0"]
    d.qvel[:] = obj["v0"]
    mujoco.mj_forward(m, d)
    helpers.assert_valid(d)
    focus = d.xpos[bid].copy()
    cameras = []
    for azimuth in (45, 135):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = focus
        cam.distance = 0.85
        cam.azimuth = azimuth
        cam.elevation = -25
        cameras.append(cam)

    frames = []
    safe_stride = max(int(stride), 1)
    renderer = mujoco.Renderer(m, height=image_size, width=image_size)
    try:
        for t in range(engine.timesteps):
            force = np.zeros(3, dtype=float)
            torque = np.zeros(3, dtype=float)
            if t < 4:
                force = action.direction * action.force * force_scale
                torque[2] = action.torque * force_scale * torque_multiplier
            d.xfrc_applied[:] = 0
            d.xfrc_applied[bid, :3] = force
            d.xfrc_applied[bid, 3:] = torque
            for _ in range(engine.substeps):
                mujoco.mj_step(m, d)
                helpers.assert_valid(d)
            if t % safe_stride and t != engine.timesteps-1:
                continue
            views = []
            for cam in cameras:
                renderer.update_scene(d, camera=cam)
                views.append(Image.fromarray(renderer.render().copy()).convert("RGB"))
            panel = Image.new("RGB", (2 * image_size, image_size + 28), "white")
            for i, view in enumerate(views):
                panel.paste(view, (i * image_size, 28))
            draw = ImageDraw.Draw(panel)
            draw.text((5, 6), f"t={t*engine.frame_dt:.2f}s, torque x{torque_multiplier:.2f}",
                      fill="black")
            frames.append(panel)
    finally:
        renderer.close()
    if not frames:
        raise RuntimeError("No rendered frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=max(80, int(1000 * engine.frame_dt * safe_stride)), loop=0)
    # A static contact strip is easy to inspect when opening GIFs is inconvenient.
    picks = [frames[k] for k in sorted({0, len(frames)//3, 2*len(frames)//3, len(frames)-1})]
    sheet = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)), "white")
    for k, frame in enumerate(picks):
        sheet.paste(frame, (0, k * frame.height))
    sheet.save(output_path.with_suffix(".png"))
    return {"gif":str(output_path), "contact_sheet":str(output_path.with_suffix(".png"))}


def run_action_diagnostics(argv=None):
    import sys
    import mujoco

    p = argparse.ArgumentParser(
        description="Extend offline PHINO sensitivity analysis with fresh MuJoCo action tests")
    p.add_argument("--scene", type=Path,
                   default=Path("results/rigid_0103/scene.xml"))
    p.add_argument("--calibration", type=Path,
                   default=Path("results/phino_calibration/calibration.json"))
    p.add_argument("--prior-sweep", type=Path,
                   default=Path("results/phino_calibration/physics_sweep.json"))
    p.add_argument("--phino-root", type=Path,
                   default=Path("/nbu/people/ashah264/phino"))
    p.add_argument("--out", type=Path, default=None,
                   help="Custom output folder. By default, flick-focus gets its own subfolder.")
    p.add_argument("--flick-focus", action="store_true",
                   help="Run a dense torque-only flick sweep (no repeated push runs)")
    p.add_argument("--render-flick", action="store_true",
                   help="Render nominal-mass/friction GIFs and contact sheets at selected torques")
    p.add_argument("--render-torque-multipliers", default="0,0.5,1",
                   help="Representative flick torque fractions for visualization")
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--render-stride", type=int, default=3)
    p.add_argument("--push-multipliers", default="0.70,0.85,1.0,1.15",
                   help="Multipliers of each calibrated push force (and coupled torque)")
    p.add_argument("--push-durations", default="4,6,8,10",
                   help="Candidate push application durations in 20 Hz frames")
    p.add_argument("--flick-torque-multipliers", default=None,
                   help="Fractions of calibrated flick torque; default is 0,.25,1 "
                        "(or a dense 0..1 grid with --flick-focus)")
    p.add_argument("--min-motion", type=float, default=0.02)
    p.add_argument("--max-motion", type=float, default=0.15)
    p.add_argument("--safety-limit", type=float, default=0.25)
    p.add_argument("--max-tilt-deg", type=float, default=60.,
                   help="Diagnostic quality threshold, not a physical instability condition")
    args = p.parse_args(argv)
    if args.out is None:
        base = Path("results/phino_calibration/sensitivity_analysis/action_diagnostics")
        args.out = base / "flick_focus" if args.flick_focus else base
    if args.flick_torque_multipliers is None:
        args.flick_torque_multipliers = (
            "0,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1" if args.flick_focus
            else "0,0.25,1")
    if args.render_size < 64 or args.render_stride < 1:
        p.error("Expected --render-size >=64 and --render-stride >=1")
    if not (0 < args.min_motion < args.max_motion < args.safety_limit):
        p.error("Expected 0 < min-motion < max-motion < safety-limit")
    if not (0 < args.max_tilt_deg <= 180):
        p.error("Expected 0 < max-tilt-deg <= 180")
    try:
        push_mult = _parse_positive_csv(args.push_multipliers)
        push_durations = _parse_positive_csv(args.push_durations, integer=True)
        flick_torque_mult = _parse_positive_csv(args.flick_torque_multipliers,
                                                permit_zero=True)
        render_torque_mult = _parse_positive_csv(args.render_torque_multipliers,
                                                permit_zero=True)
    except ValueError as e:
        p.error(str(e))
    if not args.scene.is_file() or not args.calibration.is_file() or not args.prior_sweep.is_file():
        p.error("Missing scene/calibration/physics_sweep input; run earlier calibration first")
    helpers = _load_simulation_helpers(Path(__file__).resolve().parent)
    prior = json.loads(args.prior_sweep.read_text())
    if prior.get("kind") != "physics_sensitivity_validation_not_training":
        p.error("Unexpected prior physics sweep format")
    masses = sorted(float(x) for x in prior["masses_kg"])
    frictions = sorted(float(x) for x in prior["contact_friction_values"])
    calib = json.loads(args.calibration.read_text())
    if Path(calib["scene"]).resolve() != args.scene.resolve():
        p.error("Calibration and scene paths differ. Use their matching --scene.")

    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.data.mujoco_engine import DIAGNOSTIC_ACTIONS, MuJoCoPhysicsEngine
    engine = MuJoCoPhysicsEngine()
    if any(duration > engine.timesteps//2 for duration in push_durations):
        p.error("Push durations cannot exceed the original push phase (25 frames)")
    actions = {a.name:a for a in DIAGNOSTIC_ACTIONS}
    for action in ("push_x", "push_y", "lateral_flick"):
        if action not in actions or not calib["actions"][action]["chosen"]:
            raise RuntimeError(f"Missing action/calibration {action}")
    calibrated = {a:calib["actions"][a]["chosen"] for a in ("push_x","push_y","lateral_flick")}

    candidates = {}
    if not args.flick_focus:
        for action in ("push_x", "push_y"):
            original_scale = float(calibrated[action]["scale"])
            for mult in push_mult:
                for duration in push_durations:
                    scale = original_scale * mult
                    cid = f"{action}_scale{scale:.6f}_duration{duration:02d}"
                    candidates[cid] = {
                        "action":action, "force_scale":scale, "duration_frames":duration,
                        "torque_multiplier":1.,
                        "force_N":float(actions[action].force*scale),
                        "external_torque_Nm":float(actions[action].torque*scale),
                    }
    flick_scale = float(calibrated["lateral_flick"]["scale"])
    for tau_mult in flick_torque_mult:
        cid = f"lateral_flick_torque_fraction{tau_mult:.4f}"
        candidates[cid] = {
            "action":"lateral_flick", "force_scale":flick_scale,
            "duration_frames":4, "torque_multiplier":tau_mult,
            "force_N":float(actions["lateral_flick"].force*flick_scale),
            "external_torque_Nm":float(actions["lateral_flick"].torque*flick_scale*tau_mult),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    isolated_dir = args.out / "isolated_scenes"
    isolated_dir.mkdir(parents=True, exist_ok=True)
    isolated = helpers.isolate_objects(args.scene.resolve(), isolated_dir)
    print("ACTION DIAGNOSTICS: same actions on each mass/friction setting", flush=True)
    print(f"Objects: {', '.join(x[0] for x in isolated)}", flush=True)
    print(f"Mass (kg): {masses}; contact friction: {frictions}", flush=True)
    print(f"Candidate schedules: {len(candidates)}; trials: "
          f"{len(candidates)*len(isolated)*len(masses)*len(frictions)}", flush=True)
    print("Flick external force is fixed; only its applied torque changes.", flush=True)
    print("Rotation report separates world-vertical yaw/twist from tilt.", flush=True)
    rows, traces = [], {}
    for obj_name, isolated_path in isolated:
        for mass in masses:
            for friction in frictions:
                try:
                    obj = helpers.apply_physics_to_object(
                        isolated_path, obj_name, engine, mass, friction)
                    err = None
                except helpers.RejectedTrial as exc:
                    obj, err = None, f"settle rejected: {exc}"
                for cid, spec in candidates.items():
                    row = {
                        "object":obj_name,
                        "mass_kg":mass,
                        "contact_friction":friction,
                        "candidate_id":cid,
                        "action_id":spec["action"],
                    }
                    if err is not None:
                        row.update(status="rejected", reason=err)
                    else:
                        try:
                            metrics, trace = _diagnostic_trial(
                                obj, actions[spec["action"]],
                                force_scale=spec["force_scale"],
                                duration_frames=spec["duration_frames"],
                                torque_multiplier=spec["torque_multiplier"],
                                engine=engine, safety_limit=args.safety_limit,
                                helpers=helpers, mujoco=mujoco)
                            key = f"trace_{len(traces):06d}"
                            traces[key] = trace
                            motion_good = (
                                args.min_motion <= metrics["directed_m"] <= args.max_motion
                                and metrics["max_raw_m"] <= args.max_motion)
                            quality_good = (
                                motion_good and metrics["max_tilt_deg"] <= args.max_tilt_deg
                                and not metrics["would_clip_PHINO_velocities"])
                            row.update(status="ok", trajectory_key=key,
                                       motion_target_met=bool(motion_good),
                                       target_and_tilt_met=bool(quality_good),
                                       metrics=metrics)
                        except helpers.RejectedTrial as exc:
                            row.update(status="rejected", reason=str(exc))
                    rows.append(row)
        print(f"Completed {obj_name}", flush=True)
    summary = _summarize_candidates(rows, traces, candidates, masses, frictions,
                                    args.min_motion,args.max_motion,args.max_tilt_deg)
    print("\nCandidate results (accepted / target+tilt / orientation>120deg):")
    for action in ("push_x", "push_y", "lateral_flick"):
        subset = [s for s in summary if s["spec"]["action"] == action]
        if not subset:
            continue
        for s in subset:
            spec = s["spec"]
            print(f"  {s['candidate_id']}: {s['valid']}/{s['total']} valid, "
                  f"{s['meets_target_and_tilt']}/{s['total']} target+tilt, "
                  f"{s['large_orientation_gt_120deg']} large orientations "
                  f"({s['large_orientation_and_tilt']} high-tilt / "
                  f"{s['large_orientation_low_tilt']} low-tilt); "
                  f"median tilt={label(s['median_max_tilt_deg'])}deg, "
                  f"yaw/twist={label(s['median_max_world_yaw_twist_deg'])}deg; "
                  f"mass_pairs={s['mass_matched']['valid']}, "
                  f"friction_pairs={s['friction_matched']['valid']}")
        # Prefer safe and task-relevant motion across the whole grid, then full coverage.
        preferred = max(subset, key=lambda s:(s["meets_target_and_tilt"],
                                             s["valid"],
                                             -s["tilt_gt_limit"]))
        if preferred["meets_target_and_tilt"]:
            print(f"  BEST_DIAGNOSTIC_{action}={preferred['candidate_id']}")
        else:
            print(f"  BEST_DIAGNOSTIC_{action}=NONE; revise action design")
    rendered = []
    if args.render_flick:
        nominal_mass = min(masses, key=lambda x:abs(x-0.30))
        nominal_mu = min(frictions, key=lambda x:abs(x-0.30))
        render_dir = args.out / "flick_renders"
        print(f"\nRendering nominal m={nominal_mass:.3f}kg, mu={nominal_mu:.3f} "
              f"in {render_dir}", flush=True)
        for obj_name, isolated_path in isolated:
            obj = helpers.apply_physics_to_object(isolated_path, obj_name, engine,
                                                   nominal_mass, nominal_mu)
            for fraction in render_torque_mult:
                file_name = f"{obj_name}_torque_{fraction:.2f}.gif"
                try:
                    result = _render_representative_flick(
                        obj, actions["lateral_flick"], force_scale=flick_scale,
                        torque_multiplier=fraction, engine=engine, helpers=helpers,
                        mujoco=mujoco, output_path=render_dir/file_name,
                        image_size=args.render_size, stride=args.render_stride)
                    rendered.append({"object":obj_name,"torque_fraction":fraction, **result})
                    print(f"  RENDERED {result['gif']}", flush=True)
                except Exception as exc:
                    # Optional rendering never invalidates the trajectory sensitivity analysis.
                    print(f"  RENDER_FAILED {obj_name} torque={fraction}: {exc}", flush=True)
    summary_path = args.out/"action_diagnostics.json"
    summary_path.write_text(json.dumps({
        "kind":"action_diagnostics_not_training",
        "reference_scene":str(args.scene.resolve()),
        "calibration_file":str(args.calibration.resolve()),
        "prior_sweep":str(args.prior_sweep.resolve()),
        "settings":{"mass_kg":masses,"contact_friction":frictions,
                    "min_motion_m":args.min_motion,"max_motion_m":args.max_motion,
                    "safety_limit_m":args.safety_limit,
                    "max_tilt_deg":args.max_tilt_deg},
        "candidate_specs":candidates,
        "summary":summary,
        "trials":rows,
        "renders":rendered,
        "note":"External world-frame wrench acts at body center of mass. Contact forces can still induce tipping. World-vertical twist and tilt are diagnostic decompositions; neither alone establishes the contact mechanism. All labels are synthetic.",
    },indent=2)+"\n")
    np.savez_compressed(args.out/"action_diagnostic_trajectories.npz", **traces)
    print(f"\nACTION_DIAGNOSTICS_JSON={summary_path}")
    print(f"ACTION_DIAGNOSTICS_NPZ={args.out/'action_diagnostic_trajectories.npz'}")
    if rendered:
        print(f"FLICK_RENDER_COUNT={len(rendered)}")
    print("NO_TRAINING_FINGERPRINTS_WRITTEN")
    print("ACTION_DIAGNOSTICS_OK")


if __name__ == "__main__":
    import sys
    if "--diagnose-actions" in sys.argv:
        sys.argv.remove("--diagnose-actions")
        run_action_diagnostics()
    else:
        main()
  
