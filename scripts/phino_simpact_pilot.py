"""Test PHINO's five diagnostic interventions on SIMPACT-reconstructed objects.

Run in the SIMPACT environment, from the SIMPACT repo root:
    python scripts/phino_simpact_pilot.py

Use --force-scale 0.1 to test weaker interventions after inspecting failures.
The pilot intentionally fails on instability or excessive *unclipped* motion.
"""

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path,
                        default=Path("results/rigid_0103/scene.xml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phino_pilot"))
    parser.add_argument("--phino-root", type=Path,
                        default=Path("/nbu/people/ashah264/phino"))
    parser.add_argument("--force-scale", type=float, default=1.0,
                        help="Scale PHINO's forces and torques (1.0 = original)")
    parser.add_argument("--max-displacement", type=float, default=0.5,
                        help="Maximum raw 3D displacement, in meters")
    args = parser.parse_args()

    if args.force_scale <= 0 or args.max_displacement <= 0:
        parser.error("--force-scale and --max-displacement must be positive")

    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.data.mujoco_engine import DIAGNOSTIC_ACTIONS, MuJoCoPhysicsEngine

    source = args.scene.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing SIMPACT scene: {source}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Scene has no worldbody")
    objects = [body.get("name") for body in world.findall("body")
               if body.find("freejoint") is not None]
    if not objects or any(not name for name in objects):
        raise RuntimeError(f"No named free-jointed objects found: {objects}")
    print(f"Found {len(objects)} dynamic objects: {', '.join(objects)}", flush=True)

    engine = MuJoCoPhysicsEngine()
    bad_warnings = (
        ("BADQPOS", mujoco.mjtWarning.mjWARN_BADQPOS),
        ("BADQVEL", mujoco.mjtWarning.mjWARN_BADQVEL),
        ("BADQACC", mujoco.mjtWarning.mjWARN_BADQACC),
    )

    def check_state(data, *, name, action, frame):
        for warning_name, warning_id in bad_warnings:
            if data.warning[warning_id].number:
                raise RuntimeError(
                    f"MuJoCo {warning_name}: object={name}, action={action}, "
                    f"frame={frame}, time={data.time:.4f} s"
                )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(
                f"Nonfinite state: object={name}, action={action}, "
                f"frame={frame}, time={data.time:.4f} s"
            )

    # Remove old invalid fingerprints so a failed rerun cannot leave stale results.
    for name in objects:
        (out_dir / f"{name}_fingerprint.npz").unlink(missing_ok=True)

    for name in objects:
        # Isolate one object, retaining the table, ground, cameras, and assets.
        scene = copy.deepcopy(root)
        wb = scene.find("worldbody")
        for body in list(wb.findall("body")):
            if body.find("freejoint") is not None and body.get("name") != name:
                wb.remove(body)
        scene_path = out_dir / f"{name}.xml"
        ET.ElementTree(scene).write(scene_path, encoding="unicode")

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        model.opt.timestep = engine.sim_timestep
        data = mujoco.MjData(model)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0 or model.body_jntnum[bid] != 1:
            raise RuntimeError(f"Expected one joint on object {name}")
        jid = int(model.body_jntadr[bid])
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"Expected free joint on object {name}")
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        mass = float(model.body_mass[bid])

        # Settle the isolated object before interventions.
        mujoco.mj_resetData(model, data)
        for step in range(engine.settle_steps):
            mujoco.mj_step(model, data)
            check_state(data, name=name, action="settle", frame=step)
        qpos0, qvel0 = data.qpos.copy(), data.qvel.copy()
        start_pos = qpos0[qadr:qadr + 3].copy()

        traces = []
        for action in DIAGNOSTIC_ACTIONS:
            d = mujoco.MjData(model)
            d.qpos[:] = qpos0
            d.qvel[:] = qvel0
            mujoco.mj_forward(model, d)
            check_state(d, name=name, action=action.name, frame=-1)

            traj = np.empty((engine.timesteps, 13), dtype=np.float32)
            max_raw_displacement = 0.0
            for t in range(engine.timesteps):
                force, torque = engine._wrench(action, t, engine.timesteps, mass)
                scale = 0.005 if action.name == "lateral_flick" else args.force_scale
                d.xfrc_applied[bid, :3] = force * scale
                d.xfrc_applied[bid, 3:] = torque * scale

                for _ in range(engine.substeps):
                    mujoco.mj_step(model, d)
                    check_state(d, name=name, action=action.name, frame=t)

                    # Check the raw displacement BEFORE PHINO's clipping.
                    displacement = float(np.linalg.norm(
                        d.qpos[qadr:qadr + 3] - start_pos
                    ))
                    if not np.isfinite(displacement) or displacement > args.max_displacement:
                        raise RuntimeError(
                            f"Excessive displacement: object={name}, "
                            f"action={action.name}, frame={t}, "
                            f"time={d.time:.4f} s, raw_displacement={displacement:.3f} m "
                            f"(limit={args.max_displacement:.3f} m)"
                        )
                    max_raw_displacement = max(max_raw_displacement, displacement)

                # PHINO's original 13-D state representation.
                traj[t, :3] = np.clip(d.qpos[qadr:qadr + 3] - start_pos,
                                      -engine.pos_clip, engine.pos_clip)
                traj[t, 3:7] = d.qpos[qadr + 3:qadr + 7]
                traj[t, 7:10] = np.clip(d.qvel[vadr:vadr + 3],
                                        -engine.lin_vel_clip, engine.lin_vel_clip)
                traj[t, 10:13] = np.clip(d.qvel[vadr + 3:vadr + 6],
                                         -engine.ang_vel_clip, engine.ang_vel_clip)

            if not np.isfinite(traj).all():
                raise RuntimeError(f"Nonfinite trajectory: {name}/{action.name}")
            traces.append(traj)
            print(f"{name:27s} {action.name:18s} "
                  f"max raw displacement={max_raw_displacement:.3f} m", flush=True)

        fingerprint = np.concatenate(traces, axis=0)
        expected_shape = (len(DIAGNOSTIC_ACTIONS) * engine.timesteps, 13)
        assert fingerprint.shape == expected_shape
        output = out_dir / f"{name}_fingerprint.npz"
        np.savez_compressed(
            output,
            flat_trajectory=fingerprint,
            action_names=np.array([a.name for a in DIAGNOSTIC_ACTIONS]),
            nominal_mass_kg=mass,
            force_scale=args.force_scale,
        )
        print(f"SAVED {output} shape={fingerprint.shape} "
              f"nominal_mass={mass:.3f} kg", flush=True)

    print("PHINO_SIMPACT_PILOT_OK", flush=True)


if __name__ == "__main__":
    main()

