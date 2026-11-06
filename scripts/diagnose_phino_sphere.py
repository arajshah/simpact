#!/usr/bin/env python3
"""Inspect sphere overtravel without modifying PHINO training datasets.

Run from the SIMPACT repository root:
  MUJOCO_GL=egl python scripts/diagnose_phino_sphere.py --render

This is a diagnostic only. It preserves the original 25 cm rejection threshold
as a *flag*, not a stopping rule. It uses exactly the generator's force schedule,
PHINO engine, settled initial states, and 50x13 trajectory format.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import phino_simpact_pilot_v2 as pilot


def simulate_diagnostic(obj, action, originals, engine, mj, *, cutoff, tilt_cap,
                        hard_stop, renderer=None, camera=None):
    m, bid, qadr, vadr = (obj[k] for k in ("model", "bid", "qadr", "vadr"))
    d = mj.MjData(m)
    d.qpos[:] = obj["q0"]
    d.qvel[:] = obj["v0"]
    mj.mj_forward(m, d)
    pilot.assert_state(d, mj, "diagnostic initial")
    q0 = d.qpos[qadr + 3:qadr + 7].copy()
    frames, times, raw_steps, tilt_steps, images = [], [], [], [], []
    max_raw = max_tilt = max_linear = max_angular = 0.0
    first_crossing_s = None
    error = None
    if renderer is not None:
        renderer.update_scene(d, camera=camera)
        images.append(np.asarray(renderer.render()).copy())
    for t in range(int(engine.timesteps)):
        force, torque = pilot.applied_wrench(action, t, originals)
        d.xfrc_applied[:] = 0
        d.xfrc_applied[bid, :3] = force
        d.xfrc_applied[bid, 3:] = torque
        try:
            for _ in range(int(engine.substeps)):
                mj.mj_step(m, d)
                pilot.assert_state(d, mj, f"{action} frame={t}")
                pos = d.qpos[qadr:qadr + 3] - obj["start"]
                raw = float(np.linalg.norm(pos))
                tilt = pilot.relative_world_tilt_deg(q0, d.qpos[qadr + 3:qadr + 7])
                linear = float(np.linalg.norm(d.qvel[vadr:vadr + 3]))
                angular = float(np.linalg.norm(d.qvel[vadr + 3:vadr + 6]))
                if not all(map(math.isfinite, (raw, tilt, linear, angular))):
                    raise pilot.RejectedTrial("nonfinite diagnostic state")
                max_raw = max(max_raw, raw)
                max_tilt = max(max_tilt, tilt)
                max_linear = max(max_linear, linear)
                max_angular = max(max_angular, angular)
                if first_crossing_s is None and raw > cutoff:
                    first_crossing_s = float(d.time)
                if raw > hard_stop:
                    raise pilot.RejectedTrial(
                        f"diagnostic hard stop {hard_stop:.2f} m at t={d.time:.3f}s")
        except pilot.RejectedTrial as exc:
            error = str(exc)
            break
        trace = np.concatenate((d.qpos[qadr:qadr + 3] - obj["start"],
                                d.qpos[qadr + 3:qadr + 7], d.qvel[vadr:vadr + 6]))
        frames.append(trace.astype(np.float32))
        times.append(float(d.time))
        raw_steps.append(float(np.linalg.norm(trace[:3])))
        tilt_steps.append(float(pilot.relative_world_tilt_deg(q0, trace[3:7])))
        if renderer is not None:
            renderer.update_scene(d, camera=camera)
            images.append(np.asarray(renderer.render()).copy())
    trajectory = np.asarray(frames, dtype=np.float32).reshape((-1, 13))
    results = {
        "action": action,
        "frames_saved": int(len(trajectory)),
        "completed": error is None and len(trajectory) == int(engine.timesteps),
        "diagnostic_error": error,
        "max_raw_displacement_m": round(max_raw, 6),
        "first_crossing_time_s": None if first_crossing_s is None else round(first_crossing_s, 6),
        "max_tilt_deg": round(max_tilt, 4),
        "max_linear_speed_m_s": round(max_linear, 5),
        "max_angular_speed_rad_s": round(max_angular, 5),
        "would_reject_displacement": first_crossing_s is not None,
        "would_reject_tilt": max_tilt > tilt_cap,
        "would_reject_velocity": (max_linear > engine.lin_vel_clip or
                                  max_angular > engine.ang_vel_clip),
    }
    return results, trajectory, np.asarray(times), np.asarray(raw_steps), np.asarray(tilt_steps), images


def write_gif(images, path):
    if not images:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Rendering requires Pillow; install it or omit --render") from exc
    pictures = [Image.fromarray(im) for im in images]
    pictures[0].save(path, save_all=True, append_images=pictures[1:],
                     duration=100, loop=0, optimize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path,
                    default=Path("results/phino_mesh_scenes_v1/object_sphere/scene.xml"))
    ap.add_argument("--phino-root", type=Path, default=Path("/nbu/people/ashah264/phino"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/phino_sphere_diagnostics_v1"))
    ap.add_argument("--masses", type=pilot.parse_levels, default=[0.25, 0.30, 0.35])
    ap.add_argument("--frictions", type=pilot.parse_levels, default=[0.20, 0.30, 0.40])
    ap.add_argument("--max-displacement", type=float, default=0.25,
                    help="Existing rejection cutoff; track crossing without stopping")
    ap.add_argument("--max-tilt-deg", type=float, default=60.)
    ap.add_argument("--hard-stop", type=float, default=3.0,
                    help="Absolute diagnostic emergency stop in metres")
    ap.add_argument("--render", action="store_true",
                    help="Save nominal-mass/friction GIFs; requires MUJOCO_GL=egl headlessly")
    ap.add_argument("--render-mass", type=float, default=0.30)
    ap.add_argument("--render-friction", type=float, default=0.30)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        pilot.self_test()
        assert pilot.ACTION_ORDER[:2] == ("push_x", "push_y")
        assert args.hard_stop > args.max_displacement > 0
        print("SPHERE_DIAGNOSTIC_SELF_TEST_OK")
        return
    if not (0 < args.max_displacement < args.hard_stop):
        ap.error("Require 0 < max-displacement < hard-stop")
    if not args.phino_root.is_dir():
        ap.error(f"PHINO not found: {args.phino_root}")
    scene = args.scene.resolve()
    if not scene.is_file():
        ap.error(f"Sphere scene not found: {scene}")
    dest = args.output_dir.resolve()
    if dest.exists():
        ap.error(f"Output directory already exists: {dest}; choose a new --output-dir")
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.stage-", dir=dest.parent))
    try:
        import mujoco as mj
        sys.path.insert(0, str(args.phino_root.resolve()))
        from train.data.mujoco_engine import DIAGNOSTIC_ACTIONS, MuJoCoPhysicsEngine
        engine = MuJoCoPhysicsEngine()
        if int(engine.timesteps) != 50 or not np.isclose(engine.frame_dt, 0.05):
            raise RuntimeError("PHINO schedule changed: expected 50 frames at 20 Hz")
        originals = {a.name: a for a in DIAGNOSTIC_ACTIONS}
        objects = pilot.isolate_objects(scene, stage / "isolated_scenes")
        if len(objects) != 1 or "sphere" not in objects[0]["body_name"].lower():
            raise RuntimeError(f"Expected one sphere, found {[x['body_name'] for x in objects]}")
        entry = objects[0]
        (stage / "trajectories").mkdir()
        (stage / "renders").mkdir()
        rows = []
        for mass in args.masses:
            for mu in args.frictions:
                obj = pilot.build_model(entry["isolated_scene"], entry["body_name"],
                                        mass, mu, engine, mj)
                for action in ("push_x", "push_y"):
                    should_render = (args.render and
                                     np.isclose(mass, args.render_mass) and
                                     np.isclose(mu, args.render_friction))
                    renderer = camera = None
                    if should_render:
                        renderer = mj.Renderer(obj["model"], height=360, width=480)
                        camera = mj.MjvCamera()
                        camera.type = mj.mjtCamera.mjCAMERA_FREE
                        camera.lookat[:] = obj["start"] + np.array([0.20, 0., 0.])
                        camera.distance = 1.8
                        camera.azimuth = 135
                        camera.elevation = -40
                    try:
                        metrics, trace, times, raw, tilts, images = simulate_diagnostic(
                            obj, action, originals, engine, mj,
                            cutoff=args.max_displacement, tilt_cap=args.max_tilt_deg,
                            hard_stop=args.hard_stop, renderer=renderer, camera=camera)
                    finally:
                        if renderer is not None:
                            renderer.close()
                    stem = f"sphere_m{mass:.3f}_mu{mu:.3f}_{action}"
                    npz = stage / "trajectories" / f"{stem}.npz"
                    np.savez_compressed(npz, trajectory=trace, time_s=times,
                                        raw_displacement_m=raw, tilt_deg=tilts,
                                        mass_kg=mass, contact_friction=mu, action=action,
                                        frames_expected=int(engine.timesteps))
                    metrics.update({"object": entry["body_name"], "mass_kg": mass,
                                    "contact_friction": mu,
                                    "trajectory_path": f"trajectories/{npz.name}"})
                    if should_render:
                        gif = stage / "renders" / f"{stem}.gif"
                        write_gif(images, gif)
                        metrics["render_path"] = f"renders/{gif.name}"
                    rows.append(metrics)
                    print(f"{stem}: max={metrics['max_raw_displacement_m']:.3f}m "
                          f"cross={metrics['first_crossing_time_s']}s "
                          f"frames={metrics['frames_saved']}/50 "
                          f"error={metrics['diagnostic_error'] or 'none'}")
        manifest = {
            "purpose": "diagnostic only; NOT valid PHINO training fingerprints",
            "scene": str(scene), "object": entry["body_name"],
            "actions": ["push_x", "push_y"], "fixed_action_settings": pilot.SETTINGS,
            "mass_grid_kg": args.masses, "friction_grid": args.frictions,
            "original_rejection_displacement_m": args.max_displacement,
            "diagnostic_emergency_stop_m": args.hard_stop,
            "note": "Crossing 25 cm is recorded, not automatically treated as instability."
                    " Partial sequences may result from warnings or emergency stop.",
            "trials": rows,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (stage / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "object", "mass_kg", "contact_friction", "action", "frames_saved",
                "completed", "max_raw_displacement_m", "first_crossing_time_s",
                "max_tilt_deg", "max_linear_speed_m_s", "max_angular_speed_rad_s",
                "would_reject_displacement", "would_reject_tilt", "would_reject_velocity",
                "diagnostic_error", "trajectory_path", "render_path"])
            writer.writeheader()
            for rec in rows:
                writer.writerow(rec)
        stage.rename(dest)
        print(f"SPHERE_DIAGNOSTIC_MANIFEST={dest / 'manifest.json'}")
        print(f"SPHERE_DIAGNOSTIC_SUMMARY={dest / 'summary.csv'}")
        print("PHINO_SPHERE_DIAGNOSTICS_OK")
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
