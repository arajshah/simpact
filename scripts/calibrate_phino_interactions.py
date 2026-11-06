#!/usr/bin/env python3
"""Calibrate and validate PHINO interventions on SIMPACT rigid-body scenes.

Default: validate the four calibrated actions over a fixed mass x contact-friction
physics grid, with three *fixed* lifting forces shared by every object/config.
Reads results/phino_calibration/calibration.json from the previous calibration.
Writes physics_sweep.json and physics_sweep_trajectories.npz without modifying
calibration.json or writing training fingerprints.

Original force/duration search is still available with --mode forces.
The downward press is deferred from this rigid-body baseline.

Run from the SIMPACT repository root in the existing (simpact) environment:
    python scripts/calibrate_phino_interventions.py
    python scripts/calibrate_phino_interventions.py --mode forces

Important: varying hidden physics while keeping the input image unchanged is a
SIMULATOR SENSITIVITY test, not evidence that an image-only DINOv2 model can
recover those hidden parameters.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# Original PHINO push forces/torques are multiplied by these scale values.
# Previous test at .050 barely moved either object, while .075 for 25 frames
# moved the white carton beyond the safety limit. Search finer force levels
# AND a shorter application duration, while keeping the 50-frame observation.
DEFAULT_PUSH_SCALES = [
    0.0500, 0.0525, 0.0550, 0.0575, 0.0600, 0.0625,
    0.0650, 0.0675, 0.0700, 0.0725, 0.0750, 0.0800,
    0.0900, 0.1000,
]
DEFAULT_PUSH_DURATIONS = [2, 4, 6, 8, 10, 15, 20, 25]  # 20 Hz frames
DEFAULT_FLICK_SCALES = [0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]
DEFAULT_LIFT_EXCESS_N = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080]
ACTIVE_ACTION_NAMES = ("push_x", "push_y", "grasp_lift_release", "lateral_flick")
OMITTED_ACTION_NAMES = ("slow_press_down",)


class RejectedTrial(Exception):
    """A trial has numerically invalid or out-of-workspace dynamics."""


def assert_valid(d: mujoco.MjData) -> None:
    for name, which in (
        ("BADQPOS", mujoco.mjtWarning.mjWARN_BADQPOS),
        ("BADQVEL", mujoco.mjtWarning.mjWARN_BADQVEL),
        ("BADQACC", mujoco.mjtWarning.mjWARN_BADQACC),
    ):
        if d.warning[which].number:
            raise RejectedTrial(f"MuJoCo {name} at t={d.time:.3f}s")
    if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
        raise RejectedTrial(f"nonfinite state at t={d.time:.3f}s")


def isolate_objects(scene_path: Path, output_dir: Path):
    source = ET.parse(scene_path).getroot()
    worldbody = source.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Scene has no worldbody")
    names = [b.get("name") for b in worldbody.findall("body")
             if b.find("freejoint") is not None]
    if not names or any(not name for name in names):
        raise RuntimeError(f"No named free-jointed bodies found: {names}")
    scenes = []
    for name in names:
        root = copy.deepcopy(source)
        wb = root.find("worldbody")
        for b in list(wb.findall("body")):
            if b.find("freejoint") is not None and b.get("name") != name:
                wb.remove(b)
        output = output_dir / f"{name}.xml"
        ET.ElementTree(root).write(output, encoding="unicode")
        scenes.append((name, output))
    return scenes


def load_object(name, scene, engine):
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = engine.sim_timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = max(int(model.opt.iterations), 100)
    model.opt.ls_iterations = max(int(model.opt.ls_iterations), 50)
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0 or model.body_jntnum[bid] != 1:
        raise RuntimeError(f"Expected one joint for {name}")
    jid = int(model.body_jntadr[bid])
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Expected a free joint for {name}")
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    mujoco.mj_resetData(model, data)
    for _ in range(engine.settle_steps):
        mujoco.mj_step(model, data)
        assert_valid(data)
    q0, v0 = data.qpos.copy(), data.qvel.copy()
    return {
        "name": name, "model": model, "bid": bid, "qadr": qadr, "vadr": vadr,
        "q0": q0, "v0": v0, "start": q0[qadr:qadr+3].copy(),
        "mass_kg": float(model.body_mass[bid]),
    }


def simulate_trial(obj, action, *, scale, duration_frames, lift_force_n,
                   engine, reference_mass, safety_limit):
    """Simulate full 50-frame observation, varying only the force schedule."""
    model, bid, qadr = obj["model"], obj["bid"], obj["qadr"]
    data = mujoco.MjData(model)  # fresh warnings and applied-force array
    data.qpos[:] = obj["q0"]
    data.qvel[:] = obj["v0"]
    mujoco.mj_forward(model, data)
    assert_valid(data)
    T = engine.timesteps
    max_raw = 0.0
    max_directed = 0.0
    max_rot_deg = 0.0
    initial_q = data.qpos[qadr+3:qadr+7].copy()  # wxyz

    for t in range(T):
        force = np.zeros(3)
        torque = np.zeros(3)
        if action.name in ("push_x", "push_y"):
            # Force and torque have PHINO's original directions/magnitudes,
            # but are applied only for duration_frames, then set to zero.
            if t < duration_frames:
                force = action.direction * action.force * scale
                torque[2] = action.torque * scale
        elif action.name == "lateral_flick":
            orig_force, orig_torque = engine._wrench(
                action, t, T, reference_mass)
            force = orig_force * scale
            torque = orig_torque * scale
        elif action.name == "grasp_lift_release":
            # Fixed forces shared across objects. Avoid using each object's
            # true mass as a hidden per-object input to the action controller.
            if t < int(0.3 * T):
                force[2] = lift_force_n
            elif t < int(0.6 * T):
                force[2] = reference_mass * 9.81
            # Remaining frames are unforced (release).
        else:
            raise ValueError(f"Action not selected for rigid baseline: {action.name}")

        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[bid, :3] = force
        data.xfrc_applied[bid, 3:] = torque
        for _ in range(engine.substeps):
            mujoco.mj_step(model, data)
            assert_valid(data)
            dx = data.qpos[qadr:qadr+3] - obj["start"]
            raw = float(np.linalg.norm(dx))
            if not np.isfinite(raw) or raw > safety_limit:
                raise RejectedTrial(
                    f"raw displacement {raw:.3f}m exceeds {safety_limit:.3f}m "
                    f"at t={data.time:.3f}s")
            max_raw = max(max_raw, raw)
            directed = (float(dx[2]) if action.name == "grasp_lift_release"
                        else float(np.dot(dx, action.direction)))
            max_directed = max(max_directed, directed)
            q = data.qpos[qadr+3:qadr+7]
            dot = float(np.clip(abs(np.dot(initial_q, q)), 0.0, 1.0))
            max_rot_deg = max(max_rot_deg, float(np.degrees(2*np.arccos(dot))))

    return {
        "directed_m": round(max_directed, 6),
        "max_raw_m": round(max_raw, 6),
        "max_rotation_deg": round(max_rot_deg, 3),
    }


def assess_candidate(objects, action, *, scale, duration_frames, lift_force_n,
                     engine, reference_mass, min_motion, max_motion,
                     safety_limit):
    entries = {}
    accepted = True
    for o in objects:
        try:
            result = simulate_trial(
                o, action, scale=scale, duration_frames=duration_frames,
                lift_force_n=lift_force_n, engine=engine,
                reference_mass=reference_mass, safety_limit=safety_limit)
            good = (min_motion <= result["directed_m"]
                    and result["max_raw_m"] <= max_motion)
            result["meets_target"] = bool(good)
            entries[o["name"]] = result
            accepted = accepted and good
        except RejectedTrial as exc:
            entries[o["name"]] = {"rejected": str(exc), "meets_target": False}
            accepted = False
    return {"objects": entries, "meets_target_for_all_objects": bool(accepted)}


def score_candidate(candidate, target_motion, duration_frames):
    values = list(candidate["objects"].values())
    d = [x["directed_m"] for x in values]
    # Prefer closer to a moderate target; then similar response across objects;
    # then a shorter force application, when otherwise equivalent.
    return (sum(abs(x - target_motion) for x in d) / len(d)
            + 0.10 * (max(d) - min(d))
            + 0.0001 * duration_frames)


def parse_grid(raw: str, *, integers: bool = False):
    try:
        values = ([int(v.strip()) if integers else float(v.strip())
                   for v in raw.split(",")])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("comma-separated numeric values expected") from exc
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    return sorted(set(values))


def calibrate_forces(argv=None):
    ap = argparse.ArgumentParser(description="Force/duration calibration (original stage)")
    ap.add_argument("--scene", type=Path,
                    default=Path("results/rigid_0103/scene.xml"))
    ap.add_argument("--phino-root", type=Path,
                    default=Path("/nbu/people/ashah264/phino"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/phino_calibration"))
    ap.add_argument("--reference-mass", type=float, default=0.3)
    ap.add_argument("--min-motion", type=float, default=0.02)
    ap.add_argument("--max-motion", type=float, default=0.15)
    ap.add_argument("--safety-limit", type=float, default=0.25)
    ap.add_argument("--push-scales", default=",".join(map(str, DEFAULT_PUSH_SCALES)),
                    help="Comma-separated PHINO push force/torque multipliers")
    ap.add_argument("--push-durations", default=",".join(map(str, DEFAULT_PUSH_DURATIONS)),
                    help="Comma-separated applied durations in 20 Hz frames")
    args = ap.parse_args(argv)
    if not (0 < args.min_motion < args.max_motion < args.safety_limit):
        ap.error("Require 0 < min-motion < max-motion < safety-limit")
    if args.reference_mass <= 0:
        ap.error("--reference-mass must be positive")
    try:
        push_scales = parse_grid(args.push_scales)
        push_durations = parse_grid(args.push_durations, integers=True)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))

    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.data.mujoco_engine import DIAGNOSTIC_ACTIONS, MuJoCoPhysicsEngine
    action_by_name = {a.name: a for a in DIAGNOSTIC_ACTIONS}
    missing = [x for x in ACTIVE_ACTION_NAMES if x not in action_by_name]
    if missing:
        raise RuntimeError(f"PHINO missing expected actions: {missing}")
    engine = MuJoCoPhysicsEngine()
    if any(d > engine.timesteps // 2 for d in push_durations):
        ap.error(f"Push duration must be <= {engine.timesteps // 2} frames "
                 "to stay within PHINO's original push interval")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.scene.is_file():
        raise FileNotFoundError(f"Missing SIMPACT scene: {args.scene}")
    objects = [load_object(name, path, engine) for name, path in
               isolate_objects(args.scene.resolve(), out)]
    print("Loaded:", ", ".join(f"{o['name']} ({o['mass_kg']:.3f} kg)"
                               for o in objects), flush=True)
    print(f"Four-action baseline: {', '.join(ACTIVE_ACTION_NAMES)}", flush=True)
    print(f"Motion target {args.min_motion:.3f}..{args.max_motion:.3f}m; "
          f"safety {args.safety_limit:.3f}m; "
          f"push grid {len(push_scales)} x {len(push_durations)} per direction",
          flush=True)
    report = {
        "scene": str(args.scene.resolve()),
        "reference_mass_kg": args.reference_mass,
        "rigid_baseline_actions": list(ACTIVE_ACTION_NAMES),
        "deferred_actions": ["slow_press_down"],
        "downward_press_note": "Rigid-body action retained conceptually for future deformable/contact observables; excluded from this four-action calibration.",
        "fingerprint_shape_when_generated": [len(ACTIVE_ACTION_NAMES) * engine.timesteps, 13],
        "frame_dt_s": engine.frame_dt,
        "timesteps_per_action": engine.timesteps,
        "min_motion_m": args.min_motion,
        "max_motion_m": args.max_motion,
        "safety_limit_m": args.safety_limit,
        "force_schedule_note": "Push: PHINO original force and torque scaled for chosen duration_frames; no force afterward. Lift: fixed-force initial/hold/release. Flick: original 4-frame impulse scaled.",
        "actions": {},
    }

    for name in ACTIVE_ACTION_NAMES:
        action = action_by_name[name]
        print(f"\n=== {name} ===", flush=True)
        trials = []
        passing = []
        if name.startswith("push_"):
            for scale in push_scales:
                n_pass = 0
                for duration in push_durations:
                    outcome = assess_candidate(
                        objects, action, scale=scale, duration_frames=duration,
                        lift_force_n=None, engine=engine,
                        reference_mass=args.reference_mass,
                        min_motion=args.min_motion, max_motion=args.max_motion,
                        safety_limit=args.safety_limit)
                    outcome.update({"scale": scale,
                                    "duration_frames": duration,
                                    "duration_seconds": round(duration * engine.frame_dt, 5)})
                    trials.append(outcome)
                    if outcome["meets_target_for_all_objects"]:
                        passing.append(outcome)
                        n_pass += 1
                print(f"  force/torque scale={scale:.4f}: "
                      f"{n_pass}/{len(push_durations)} durations passed for ALL objects",
                      flush=True)
            if passing:
                target = (args.min_motion + args.max_motion) / 2
                best = min(passing, key=lambda x: score_candidate(
                    x, target, x["duration_frames"]))
                chosen = {"scale": best["scale"],
                          "duration_frames": best["duration_frames"],
                          "duration_seconds": best["duration_seconds"],
                          "PHINO_original_force_N": float(action.force),
                          "PHINO_original_torque_Nm": float(action.torque),
                          "applied_force_N": round(float(action.force * best["scale"]), 6),
                          "applied_torque_Nm": round(float(action.torque * best["scale"]), 6),
                          "results": best["objects"]}
                print(f"  CHOSEN: scale={chosen['scale']:.4f}, "
                      f"duration={chosen['duration_frames']} frames "
                      f"({chosen['duration_seconds']:.2f}s), "
                      f"force={chosen['applied_force_N']:.4f}N", flush=True)
            else:
                chosen = None
                print("  CHOSEN: NONE. Inspect calibration.json or expand grid.", flush=True)
            kind = "push_scale_and_duration_frames"
        else:
            values = (DEFAULT_FLICK_SCALES if name == "lateral_flick" else
                      [args.reference_mass * 9.81 + x for x in DEFAULT_LIFT_EXCESS_N])
            for value in values:
                outcome = assess_candidate(
                    objects, action,
                    scale=value if name == "lateral_flick" else None,
                    duration_frames=None,
                    lift_force_n=value if name == "grasp_lift_release" else None,
                    engine=engine, reference_mass=args.reference_mass,
                    min_motion=args.min_motion, max_motion=args.max_motion,
                    safety_limit=args.safety_limit)
                outcome["parameter"] = round(value, 6)
                trials.append(outcome)
                if outcome["meets_target_for_all_objects"]:
                    passing.append(outcome)
                displays = []
                for obj_name, result in outcome["objects"].items():
                    if "rejected" in result:
                        displays.append(f"{obj_name}: REJECTED {result['rejected']}")
                    else:
                        displays.append(f"{obj_name}: {result['directed_m']:.3f}m "
                                        f"(raw {result['max_raw_m']:.3f}m)")
                unit = "N" if name == "grasp_lift_release" else "scale"
                print(f"  {value:.4f} {unit}: " + "; ".join(displays), flush=True)
            if passing:
                best = min(passing, key=lambda x: abs(
                    np.mean([o["directed_m"] for o in x["objects"].values()])
                    - (args.min_motion + args.max_motion)/2))
                chosen = ({"lift_force_N": best["parameter"],
                           "hold_force_N": round(args.reference_mass * 9.81, 6)}
                          if name == "grasp_lift_release" else
                          {"scale": best["parameter"],
                           "applied_force_N": round(float(action.force * best["parameter"]), 6),
                           "applied_torque_Nm": round(float(action.torque * best["parameter"]), 6)})
                chosen["results"] = best["objects"]
                print(f"  CHOSEN: {chosen}", flush=True)
            else:
                chosen = None
                print("  CHOSEN: NONE; needs further calibration", flush=True)
            kind = ("fixed_lift_and_hold_forces_N" if name == "grasp_lift_release"
                    else "PHINO_flick_force_and_torque_scale")
        report["actions"][name] = {
            "parameter_kind": kind,
            "chosen": chosen,
            "passing_candidates": len(passing),
            "trials": trials,
        }

    path = out / "calibration.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nCALIBRATION_REPORT={path}", flush=True)
    print("NO_TRAINING_FINGERPRINTS_WRITTEN", flush=True)
    print("DOWNWARD_PRESS_DEFERRED: four-action fingerprint will be 200x13", flush=True)
    if all(report["actions"][name]["chosen"] is not None
           for name in ACTIVE_ACTION_NAMES):
        print("FOUR_ACTIONS_CALIBRATED: validate with varied mass/friction next", flush=True)
    else:
        print("CALIBRATION_INCOMPLETE: inspect grid or redesign action", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: sensitivity of fixed action settings to mass and contact friction
# ---------------------------------------------------------------------------

DEFAULT_MASSES_KG = (0.25, 0.30, 0.35)
DEFAULT_CONTACT_FRICTIONS = (0.20, 0.30, 0.40)
# Three *absolute* forces, identical for every object and every mass setting.
# They straddle the nominal ~2.97 N force calibrated at 0.300 kg.
DEFAULT_LIFT_FORCES_N = (2.75, 2.973, 3.20)


def apply_physics_to_object(scene: Path, name: str, engine,
                            target_mass_kg: float, contact_friction: float):
    """Compile fresh isolated scene, modify physics, then settle from reset state.

    Both sides of each contact receive the same sliding-friction coefficient.
    This tests *object-table contact friction*, not an independently measured
    intrinsic material coefficient. Mass and inertia are scaled together.
    """
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = engine.sim_timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = max(int(model.opt.iterations), 100)
    model.opt.ls_iterations = max(int(model.opt.ls_iterations), 50)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0 or model.body_jntnum[bid] != 1:
        raise RuntimeError(f"Expected one joint for {name}")
    jid = int(model.body_jntadr[bid])
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Expected free joint for {name}")
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])

    original_mass = float(model.body_mass[bid])
    if original_mass <= 0:
        raise RuntimeError(f"Nonpositive original mass for {name}")
    ratio = target_mass_kg / original_mass
    model.body_mass[bid] *= ratio
    model.body_inertia[bid, :] *= ratio
    # Contact friction is combined across a contacting pair. Vary object,
    # tabletop and ground together so static tabletop friction does not mask
    # intended variations. Only the sliding component [0] is varied here.
    affected = 0
    for g in range(model.ngeom):
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            model.geom_friction[g, 0] = contact_friction
            affected += 1
    if not affected:
        raise RuntimeError(f"No collision geoms found for {name}")

    data = mujoco.MjData(model)
    # MuJoCo's derived mass/inertia constants need updating after model edits.
    mujoco.mj_setConst(model, data)
    mujoco.mj_resetData(model, data)
    for _ in range(engine.settle_steps):
        mujoco.mj_step(model, data)
        assert_valid(data)

    q0, v0 = data.qpos.copy(), data.qvel.copy()
    if not np.isfinite(q0).all() or not np.isfinite(v0).all():
        raise RejectedTrial("nonfinite settled state")
    return {
        "name": name, "model": model, "bid": bid, "qadr": qadr,
        "vadr": vadr, "q0": q0, "v0": v0,
        "start": q0[qadr:qadr+3].copy(),
        "mass_kg": float(model.body_mass[bid]),
        "contact_friction": float(contact_friction),
    }


def simulate_fixed_action(obj, action, settings: dict, *, lift_force_n,
                          engine, safety_limit):
    """Return (metrics, raw 50x13 trajectory) without feature clipping.

    Uses one fixed externally applied force schedule per action for *all*
    objects/physics configs. Upward lift is a scripted wrench, not a grasp.
    """
    model, bid, qadr, vadr = (obj[k] for k in ("model", "bid", "qadr", "vadr"))
    d = mujoco.MjData(model)
    d.qpos[:] = obj["q0"]
    d.qvel[:] = obj["v0"]
    mujoco.mj_forward(model, d)
    assert_valid(d)
    T = engine.timesteps
    initial_quat = d.qpos[qadr+3:qadr+7].copy()
    trace = np.empty((T, 13), dtype=np.float32)
    max_raw, max_directed, max_rotation = 0., 0., 0.
    max_speed, max_angular_speed = 0., 0.

    for t in range(T):
        force = np.zeros(3, dtype=float)
        torque = np.zeros(3, dtype=float)
        if action.name in ("push_x", "push_y"):
            if t < settings["duration_frames"]:
                force = action.direction * action.force * settings["scale"]
                torque[2] = action.torque * settings["scale"]
        elif action.name == "lateral_flick":
            # Original flick: 4-frame impulse, scaled uniformly.
            raw_force, raw_torque = engine._wrench(action, t, T, obj["mass_kg"])
            force = raw_force * settings["scale"]
            torque = raw_torque * settings["scale"]
        elif action.name == "grasp_lift_release":
            if t < int(0.3 * T):
                force[2] = lift_force_n
            elif t < int(0.6 * T):
                force[2] = settings["hold_force_N"]
            # Last phase: no force, simulates release of a virtual wrench.
        else:
            raise RuntimeError(f"Unexpected action {action.name}")

        d.xfrc_applied[:] = 0.
        d.xfrc_applied[bid, :3] = force
        d.xfrc_applied[bid, 3:] = torque
        for _ in range(engine.substeps):
            mujoco.mj_step(model, d)
            assert_valid(d)
            dx = d.qpos[qadr:qadr+3] - obj["start"]
            raw = float(np.linalg.norm(dx))
            if not np.isfinite(raw) or raw > safety_limit:
                raise RejectedTrial(
                    f"raw displacement {raw:.3f}m exceeds {safety_limit:.3f}m "
                    f"at t={d.time:.3f}s")
            max_raw = max(max_raw, raw)
            directed = float(dx[2] if action.name == "grasp_lift_release"
                             else np.dot(dx, action.direction))
            max_directed = max(max_directed, directed)
            quat = d.qpos[qadr+3:qadr+7]
            dot = np.clip(abs(float(np.dot(initial_quat, quat))), 0., 1.)
            max_rotation = max(max_rotation, float(np.degrees(2 * np.arccos(dot))))
            max_speed = max(max_speed, float(np.linalg.norm(d.qvel[vadr:vadr+3])))
            max_angular_speed = max(
                max_angular_speed, float(np.linalg.norm(d.qvel[vadr+3:vadr+6])))

        # Do NOT clip, because clipped outputs conceal safety/identifiability issues.
        trace[t, :3] = d.qpos[qadr:qadr+3] - obj["start"]
        trace[t, 3:7] = d.qpos[qadr+3:qadr+7]
        trace[t, 7:10] = d.qvel[vadr:vadr+3]
        trace[t, 10:13] = d.qvel[vadr+3:vadr+6]
    if not np.isfinite(trace).all():
        raise RejectedTrial("nonfinite recorded trajectory")
    metrics = {
        "directed_m": round(max_directed, 6),
        "max_raw_m": round(max_raw, 6),
        "max_rotation_deg": round(max_rotation, 3),
        "max_linear_speed_m_s": round(max_speed, 5),
        "max_angular_speed_rad_s": round(max_angular_speed, 5),
        "would_clip_PHINO_velocities": bool(
            max_speed > engine.lin_vel_clip or
            max_angular_speed > engine.ang_vel_clip),
    }
    return metrics, trace


def summarize_sensitivity(records, traces):
    """Compare position trajectories under one-parameter changes.

    Conditioning on the other physics variable avoids conflating mass and
    friction changes. These distances do not prove parameter identifiability.
    """
    groups = defaultdict(list)
    for r in records:
        if r["status"] != "ok":
            continue
        groups[(r["object"], r["action_id"])].append(r)
    summary = {}
    for (obj, action_id), grp in sorted(groups.items()):
        mass_dists, friction_dists = [], []
        for i, a in enumerate(grp):
            for b in grp[i+1:]:
                pa = traces[a["trajectory_key"]][:, :3]
                pb = traces[b["trajectory_key"]][:, :3]
                rms_m = float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))
                same_mass = np.isclose(a["mass_kg"], b["mass_kg"])
                same_fr = np.isclose(a["contact_friction"], b["contact_friction"])
                if same_fr and not same_mass:
                    mass_dists.append(rms_m)
                if same_mass and not same_fr:
                    friction_dists.append(rms_m)
        def pack(vals):
            return {"pairs": len(vals),
                    "mean_position_trajectory_RMS_m": (round(float(np.mean(vals)), 6)
                                                     if vals else None),
                    "max_position_trajectory_RMS_m": (round(float(np.max(vals)), 6)
                                                    if vals else None)}
        summary[f"{obj}/{action_id}"] = {
            "valid_configs": len(grp),
            "mass_sensitivity_fixed_friction": pack(mass_dists),
            "friction_sensitivity_fixed_mass": pack(friction_dists),
            "rotation_warning_count_gt_120deg": sum(
                r["metrics"]["max_rotation_deg"] > 120 for r in grp),
        }
    return summary


def validate_physics(argv=None):
    ap = argparse.ArgumentParser(
        description="Physics sensitivity grid using prior calibrated PHINO actions")
    ap.add_argument("--scene", type=Path,
                    default=Path("results/rigid_0103/scene.xml"))
    ap.add_argument("--phino-root", type=Path,
                    default=Path("/nbu/people/ashah264/phino"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/phino_calibration"))
    ap.add_argument("--calibration", type=Path, default=None,
                    help="Default: <output-dir>/calibration.json")
    ap.add_argument("--masses", default=",".join(map(str, DEFAULT_MASSES_KG)),
                    help="Comma-separated masses in kg")
    ap.add_argument("--frictions", default=",".join(map(str, DEFAULT_CONTACT_FRICTIONS)),
                    help="Comma-separated MuJoCo object-table sliding frictions")
    ap.add_argument("--lift-forces", default=",".join(map(str, DEFAULT_LIFT_FORCES_N)),
                    help="Exactly three fixed lifting forces in N: low,medium,high")
    ap.add_argument("--safety-limit", type=float, default=0.25,
                    help="Stop trial above this raw displacement in meters")
    ap.add_argument("--min-motion", type=float, default=0.02)
    ap.add_argument("--max-motion", type=float, default=0.15)
    args = ap.parse_args(argv)
    try:
        masses = parse_grid(args.masses)
        frictions = parse_grid(args.frictions)
        lifts = parse_grid(args.lift_forces)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    if len(lifts) != 3:
        ap.error("--lift-forces needs exactly three distinct positive values")
    if any(not 0 < f < 5 for f in frictions):
        ap.error("Contact-friction values must be in (0, 5)")
    if not (0 < args.min_motion < args.max_motion < args.safety_limit):
        ap.error("Require 0 < min-motion < max-motion < safety-limit")
    if not args.scene.is_file():
        raise FileNotFoundError(f"Missing SIMPACT scene: {args.scene}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    calib_path = (args.calibration or out / "calibration.json").resolve()
    if not calib_path.is_file():
        raise FileNotFoundError(
            f"Prior calibration report not found: {calib_path}. "
            "Run --mode forces to generate it first.")
    calib = json.loads(calib_path.read_text())
    if Path(calib["scene"]).resolve() != args.scene.resolve():
        raise RuntimeError("Calibration was made for a different scene. "
                           "Use the matching --scene or recalibrate.")
    actions = calib["actions"]
    chosen = {}
    for name in ACTIVE_ACTION_NAMES:
        chosen[name] = actions[name]["chosen"]
        if not chosen[name]:
            raise RuntimeError(f"No calibrated setting for {name}")
    # Preserve chosen low/mid/high forces in one file so they are reproducible.
    fixed_schedule = {
        "push_x": {"scale": float(chosen["push_x"]["scale"]),
                   "duration_frames": int(chosen["push_x"]["duration_frames"])},
        "push_y": {"scale": float(chosen["push_y"]["scale"]),
                   "duration_frames": int(chosen["push_y"]["duration_frames"])},
        "lateral_flick": {"scale": float(chosen["lateral_flick"]["scale"])},
        "grasp_lift_release": {
            "hold_force_N": float(chosen["grasp_lift_release"]["hold_force_N"]),
            "lift_forces_N": dict(zip(("low", "medium", "high"), lifts)),
        },
    }

    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.data.mujoco_engine import DIAGNOSTIC_ACTIONS, MuJoCoPhysicsEngine
    action_by_name = {a.name: a for a in DIAGNOSTIC_ACTIONS}
    if set(ACTIVE_ACTION_NAMES) - set(action_by_name):
        raise RuntimeError("PHINO does not have all four expected actions")
    engine = MuJoCoPhysicsEngine()
    T = engine.timesteps
    for name in ("push_x", "push_y"):
        if not 0 < fixed_schedule[name]["duration_frames"] <= T // 2:
            raise RuntimeError(f"Invalid calibrated duration for {name}")

    iso_dir = out / "isolated_physics"
    iso_dir.mkdir(parents=True, exist_ok=True)
    isolated = isolate_objects(args.scene.resolve(), iso_dir)
    print("Physics sweep: mass x contact friction x 3 shared lift forces")
    print("Forces are NEVER adjusted based on the current object's hidden mass.")
    print(f"Objects: {', '.join(n for n, _ in isolated)}")
    print(f"Masses (kg): {masses}; contact friction: {frictions}")
    print(f"Lift forces (N): low={lifts[0]}, medium={lifts[1]}, high={lifts[2]}")
    print("Baseline: push_x, push_y, lateral_flick and 3 lift probes; "
          "downward press deferred.", flush=True)

    results, trajectories = [], {}
    action_ids = ["push_x", "push_y", "lateral_flick",
                  "lift_low", "lift_medium", "lift_high"]
    runs = [("push_x", "push_x", None), ("push_y", "push_y", None),
            ("lateral_flick", "lateral_flick", None)] + [
                (f"lift_{label}", "grasp_lift_release", force)
                for label, force in zip(("low", "medium", "high"), lifts)]
    for object_name, scene in isolated:
        for mass in masses:
            for friction in frictions:
                try:
                    obj = apply_physics_to_object(
                        scene, object_name, engine, mass, friction)
                except RejectedTrial as exc:
                    print(f"{object_name} m={mass:.3f} mu={friction:.3f}: "
                          f"SETTLE REJECTED ({exc})", flush=True)
                    for action_id in action_ids:
                        results.append({"object": object_name,
                                        "mass_kg": mass,
                                        "contact_friction": friction,
                                        "action_id": action_id,
                                        "status": "rejected",
                                        "reason": f"settle: {exc}"})
                    continue
                labels = []
                for action_id, name, lift_force in runs:
                    try:
                        metrics, trace = simulate_fixed_action(
                            obj, action_by_name[name], fixed_schedule[name],
                            lift_force_n=lift_force, engine=engine,
                            safety_limit=args.safety_limit)
                        key = f"traj_{len(trajectories):05d}"
                        trajectories[key] = trace
                        informative_motion = (
                            args.min_motion <= metrics["directed_m"] <= args.max_motion
                            and metrics["max_raw_m"] <= args.max_motion)
                        results.append({
                            "object": object_name,
                            "mass_kg": mass, "contact_friction": friction,
                            "action_id": action_id,
                            "applied_lift_force_N": lift_force,
                            "status": "ok", "trajectory_key": key,
                            "motion_target_met": informative_motion,
                            "metrics": metrics,
                        })
                        marker = "+" if informative_motion else "~"
                        labels.append(f"{action_id}:{metrics['directed_m']:.3f}m{marker}")
                    except RejectedTrial as exc:
                        results.append({
                            "object": object_name, "mass_kg": mass,
                            "contact_friction": friction,
                            "action_id": action_id,
                            "applied_lift_force_N": lift_force,
                            "status": "rejected", "reason": str(exc),
                        })
                        labels.append(f"{action_id}:REJECTED")
                print(f"{object_name} m={mass:.3f} mu={friction:.3f}: "
                      + " | ".join(labels), flush=True)

    sensitivity = summarize_sensitivity(results, trajectories)
    report = {
        "kind": "physics_sensitivity_validation_not_training",
        "calibration_file": str(calib_path),
        "scene": str(args.scene.resolve()),
        "masses_kg": masses, "contact_friction_values": frictions,
        "fixed_schedule": fixed_schedule,
        "friction_semantics": "object, tabletop and ground collision geoms receive the same sliding friction; this controls contact-pair friction",
        "physics_semantics": "synthetic assumed mass and friction; same input RGB image across variations; no real-world physics measurement",
        "force_semantics": "external scripted body wrench, not a physical gripper/contact actuation",
        "restitution": "not varied; SIMPACT scene default retained",
        "actions_excluded": ["slow_press_down"],
        "observation": "50 raw 13D state rows per *successful action*, no clipping",
        "note": "A failed motion target is not a simulation failure. Rejections record numerical instability or >safety-limit raw displacement.",
        "safety_limit_m": args.safety_limit,
        "motion_target_m": [args.min_motion, args.max_motion],
        "results": results,
        "sensitivity": sensitivity,
        "valid_actions": sum(r["status"] == "ok" for r in results),
        "rejected_actions": sum(r["status"] != "ok" for r in results),
    }
    path = out / "physics_sweep.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    traj_path = out / "physics_sweep_trajectories.npz"
    np.savez_compressed(traj_path, **trajectories)
    print("\nSENSITIVITY (position trajectory RMS between configs, m):")
    for key, v in sensitivity.items():
        m = v["mass_sensitivity_fixed_friction"]
        f = v["friction_sensitivity_fixed_mass"]
        m_text = (f"{m['mean_position_trajectory_RMS_m']:.4f}"
                  if m["pairs"] else "n/a")
        f_text = (f"{f['mean_position_trajectory_RMS_m']:.4f}"
                  if f["pairs"] else "n/a")
        print(f"  {key}: valid={v['valid_configs']}, "
              f"mass_RMS={m_text} ({m['pairs']} pairs), "
              f"friction_RMS={f_text} ({f['pairs']} pairs), "
              f">120deg flips={v['rotation_warning_count_gt_120deg']}")
    print(f"\nPHYSICS_SWEEP_REPORT={path}")
    print(f"PHYSICS_SWEEP_TRAJECTORIES={traj_path}")
    print(f"VALID_TRIALS={report['valid_actions']} "
          f"REJECTED_TRIALS={report['rejected_actions']}")
    print("NOT_TRAINING_DATA: verify physics sensitivity and coverage first.")
    print("IMAGE_ONLY_LIMIT: same pixels cannot encode different hidden physics.")


def main():
    mode_arg = argparse.ArgumentParser(add_help=False)
    mode_arg.add_argument("--mode", choices=("physics", "forces"),
                          default="physics")
    mode, remaining = mode_arg.parse_known_args()
    if mode.mode == "forces":
        calibrate_forces(remaining)
    else:
        validate_physics(remaining)


if __name__ == "__main__":
    main()

