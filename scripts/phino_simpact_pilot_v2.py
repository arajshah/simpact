#!/usr/bin/env python3
"""Run a calibrated four-action SIMPACT/PHINO mass-friction grid pilot.

From the SIMPACT repository root, with the existing (simpact) environment:
    python scripts/phino_simpact_pilot_v2.py --self-test
    python scripts/phino_simpact_pilot_v2.py

Default grid: both isolated cartons x 3 masses x 3 friction coefficients.
Additional SIMPACT scenes may be supplied with --additional-scene or --scene-glob.
Use --masses and --frictions to change the physics grid.
Multiple scenes use a separate default output directory and scene-qualified IDs.

IMPORTANT: Per-action validity is recorded independently; an invalid lift
must not prevent the three other actions from being evaluated and saved.
Complete (200,13) fingerprints are saved ONLY when all four actions are valid.
Valid individual (50,13) trajectories are saved even for incomplete cases.
A motion-target failure is a separate flag, NOT an invalid trajectory.

Simulation-only data, not real measured mass/friction or PHINO training data.
An entire run is staged and only published after its manifest and files exist.
The default multi-scene output is separate from the original grid.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import copy
import fcntl
import glob
import os
import re
import shutil
import tempfile
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# These schedules were chosen from the Nov 5, 2026 action diagnostics.
# Every object/configuration receives IDENTICAL external forces and durations.
# No action is conditioned on the object's hidden mass or friction.
ACTION_ORDER = ("push_x", "push_y", "grasp_lift_release", "lateral_flick")
SETTINGS = {
    "push_x": {"force_scale": 0.1000, "duration_frames": 4},
    "push_y": {"force_scale": 0.1035, "duration_frames": 4},
    "grasp_lift_release": {
        "lift_force_N": 2.973,
        "hold_force_N": 2.943,
        "lift_frames": 15,
        "hold_frames": 15,
    },
    "lateral_flick": {
        "force_scale": 0.0100,
        "torque_fraction": 1.0000,
        "duration_frames": 4,
    },
}


class RejectedTrial(Exception):
    pass


def parse_levels(text: str) -> list[float]:
    try:
        vals = [float(v.strip()) for v in text.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from e
    if not vals or any(not math.isfinite(v) or v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("Levels must be positive and finite")
    return list(dict.fromkeys(vals))


def applied_wrench(name: str, t: int, originals: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """World-frame external force/torque; independent of hidden physics."""
    a = originals[name]
    cfg = SETTINGS[name]
    f = np.zeros(3, dtype=float)
    tau = np.zeros(3, dtype=float)
    if name in ("push_x", "push_y"):
        if t < cfg["duration_frames"]:
            f[:] = np.asarray(a.direction) * float(a.force) * cfg["force_scale"]
            tau[2] = float(a.torque) * cfg["force_scale"]
    elif name == "grasp_lift_release":
        if t < cfg["lift_frames"]:
            f[2] = cfg["lift_force_N"]
        elif t < cfg["lift_frames"] + cfg["hold_frames"]:
            f[2] = cfg["hold_force_N"]
    elif name == "lateral_flick":
        if t < cfg["duration_frames"]:
            f[:] = np.asarray(a.direction) * float(a.force) * cfg["force_scale"]
            tau[2] = (float(a.torque) * cfg["force_scale"]
                      * cfg["torque_fraction"])
    else:
        raise ValueError(f"Unknown action {name}")
    return f, tau


def self_test() -> None:
    @dataclass
    class Action:
        force: float
        torque: float
        direction: np.ndarray
    d = {
        "push_x": Action(8, 0.2, np.array([1., 0., 0.])),
        "push_y": Action(8, 0.2, np.array([0., 1., 0.])),
        "grasp_lift_release": Action(6, 0, np.array([0., 0., 1.])),
        "lateral_flick": Action(25, 5, np.array([1., 0.5, 0.]) / np.sqrt(1.25)),
    }
    fx, tx = applied_wrench("push_x", 0, d)
    assert np.allclose(fx, [0.8, 0, 0]) and np.allclose(tx, [0, 0, 0.02])
    assert np.allclose(applied_wrench("push_x", 4, d)[0], 0)
    fy, ty = applied_wrench("push_y", 0, d)
    assert np.allclose(fy, [0, 0.828, 0]) and np.allclose(ty, [0, 0, 0.0207])
    assert np.allclose(applied_wrench("grasp_lift_release", 14, d)[0], [0, 0, 2.973])
    assert np.allclose(applied_wrench("grasp_lift_release", 15, d)[0], [0, 0, 2.943])
    assert np.allclose(applied_wrench("grasp_lift_release", 30, d)[0], 0)
    ff, tf = applied_wrench("lateral_flick", 0, d)
    assert np.isclose(np.linalg.norm(ff), .25) and np.allclose(tf, [0, 0, .05])
    assert np.allclose(applied_wrench("lateral_flick", 4, d)[0], 0)
    assert tuple(ACTION_ORDER) == ("push_x", "push_y", "grasp_lift_release", "lateral_flick")
    # Regression test: a 180-degree yaw does not create world-vertical tilt,
    # even when the source object's local z-axis initially lies sideways.
    q_initial = np.array([np.sqrt(.5), np.sqrt(.5), 0., 0.])  # initial 90deg X
    q_world_yaw = np.array([0., 0., 0., 1.])                  # 180deg Z
    q_world_tilt = np.array([np.cos(np.pi/6), np.sin(np.pi/6), 0., 0.])
    assert np.isclose(relative_world_tilt_deg(
        q_initial, quaternion_multiply(q_world_yaw, q_initial)), 0., atol=1e-7)
    assert np.isclose(relative_world_tilt_deg(
        q_initial, quaternion_multiply(q_world_tilt, q_initial)), 60., atol=1e-7)
    assert np.isclose(relative_world_tilt_deg(q_initial, -q_initial), 0., atol=1e-7)
    # Stable scene discovery, unique scene IDs and asset directory anchoring.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for tag in ("rigid_one", "rigid_two"):
            scene = root / tag / "scene.xml"
            scene.parent.mkdir(parents=True)
            scene.write_text('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 1"/><body name="shared"><freejoint/><geom type="box" size=".1 .1 .1" mass=".3"/></body></worldbody></mujoco>' , encoding="utf-8")
        files = discover_scenes(root / "rigid_one" / "scene.xml",
                                [root / "rigid_two" / "scene.xml"], [])
        assert len(files) == 2
        isolated = [entry for scene in files
                    for entry in isolate_objects(scene, root / "isolated", qualify=True)]
        assert len({entry["object_id"] for entry in isolated}) == 2
        assert all(entry["body_name"] == "shared" for entry in isolated)
        assert all(ET.parse(entry["isolated_scene"]).getroot()
                   .find("compiler").get("assetdir") == str(Path(entry["source_scene"]).parent)
                   for entry in isolated)
        assert len(isolate_objects(files[0], root / "single")) == 1
        final = root / "published"
        final.mkdir()
        (final / "sentinel").write_text("old")
        new = root / "staged"
        new.mkdir()
        (new / "sentinel").write_text("new")
        publish_staged_output(new, final)
        assert (final / "sentinel").read_text() == "new"
        assert not (root / "published.previous").exists()
    print("SCHEDULE_SELF_TEST_OK")


def _slug(text: str) -> str:
    """Use stable, safe strings for filenames; preserve original body names separately."""
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    if len(result) > 70:
        result = result[:58] + "_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return result or "object"


def _scene_id(scene: Path) -> str:
    """Path identity prevents collisions even when scenes use identical body names."""
    digest = hashlib.sha256(str(scene.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{_slug(scene.parent.name)}_{digest}"


def discover_scenes(scene: Path, extra: list[Path], patterns: list[str]) -> list[Path]:
    """Resolve and de-duplicate scenes deterministically (first scene stays first)."""
    candidates = [scene, *extra]
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"Scene glob found no files: {pattern!r}")
        candidates.extend(matches)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        p = path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SIMPACT scene not found: {p}")
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _anchor_relative_assets(root: ET.Element, original_scene: Path) -> None:
    """Preserve mesh/texture asset lookup after isolated XML is moved elsewhere.

    SIMPACT's original scene may use compiler assetdir/meshdir/texturedir. MuJoCo
    otherwise resolves filenames relative to the XML's *new* location.
    """
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    original_dir = original_scene.parent.resolve()
    directories = ("assetdir", "meshdir", "texturedir")
    had_directory = False
    for attr in directories:
        value = compiler.get(attr)
        if value:
            had_directory = True
            if not Path(value).is_absolute():
                compiler.set(attr, str((original_dir / value).resolve()))
    if not had_directory:
        compiler.set("assetdir", str(original_dir))


def isolate_objects(scene_file: Path, directory: Path, *, qualify: bool = False
                    ) -> list[dict[str, Any]]:
    """Create independent object MJCFs, with unique IDs across source scenes."""
    root = ET.parse(scene_file).getroot()
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError(f"Scene has no worldbody: {scene_file}")
    names = [b.get("name") for b in wb.findall("body")
             if b.find("freejoint") is not None]
    if not names or any(not x for x in names):
        raise RuntimeError(f"No named free-jointed objects in scene: {scene_file}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate free-jointed body names in {scene_file}")
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    scene_tag = _scene_id(scene_file)
    scene_hash = hashlib.sha256(scene_file.read_bytes()).hexdigest()
    for name in names:
        one = copy.deepcopy(root)
        target = one.find("worldbody")
        assert target is not None
        for b in list(target.findall("body")):
            if b.find("freejoint") is not None and b.get("name") != name:
                target.remove(b)
        _anchor_relative_assets(one, scene_file)
        object_id = f"{scene_tag}__{_slug(name)}" if qualify else name
        dest = directory / f"{_slug(object_id)}.xml"
        ET.ElementTree(one).write(dest, encoding="unicode")
        result.append({
            "object_id": object_id, "body_name": name, "scene_id": scene_tag,
            "source_scene": str(scene_file), "source_scene_sha256": scene_hash,
            "isolated_scene": dest,
        })
    return result


def publish_staged_output(stage: Path, final: Path) -> None:
    """Replace a completed dataset, preserving the old dataset on failure.

    Old files cannot be mixed with new files. An interrupted *simulation* leaves
    the previously published output intact. Paths in the manifest are relative.
    """
    final = final.resolve()
    backup = final.with_name(final.name + ".previous")
    if backup.exists():
        raise RuntimeError(f"Existing backup {backup}; inspect it before retrying")
    if final.exists():
        final.rename(backup)
    try:
        stage.rename(final)
    except Exception:
        if backup.exists():
            backup.rename(final)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def assert_state(data, mujoco, context: str) -> None:
    for code in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC"):
        w = getattr(mujoco.mjtWarning, code)
        if data.warning[w].number:
            raise RejectedTrial(f"MuJoCo {code} at {context}, t={data.time:.4f}s")
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RejectedTrial(f"Nonfinite qpos/qvel at {context}, t={data.time:.4f}s")


def build_model(scene: Path, name: str, mass: float, friction: float, engine, mujoco):
    m = mujoco.MjModel.from_xml_path(str(scene))
    m.opt.timestep = engine.sim_timestep
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.iterations = max(int(m.opt.iterations), 100)
    m.opt.ls_iterations = max(int(m.opt.ls_iterations), 50)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0 or int(m.body_jntnum[bid]) != 1:
        raise RuntimeError(f"Expected one joint on body {name}")
    jid = int(m.body_jntadr[bid])
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Expected freejoint for {name}")
    qadr, vadr = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
    original_mass = float(m.body_mass[bid])
    if original_mass <= 0:
        raise RuntimeError(f"Nonpositive body mass for {name}")
    ratio = mass / original_mass
    m.body_mass[bid] *= ratio
    m.body_inertia[bid, :] *= ratio
    # This changes sliding contact friction on every collision geom, including
    # both surfaces of a contacting pair; it is NOT a measured material property.
    n_collision = 0
    for g in range(m.ngeom):
        if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0:
            m.geom_friction[g, 0] = friction
            n_collision += 1
    if not n_collision:
        raise RuntimeError(f"No collision geoms in {scene}")
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    mujoco.mj_resetData(m, d)
    for step in range(engine.settle_steps):
        mujoco.mj_step(m, d)
        assert_state(d, mujoco, f"{name} settling step={step}")
    return dict(model=m, bid=bid, qadr=qadr, vadr=vadr,
                q0=d.qpos.copy(), v0=d.qvel.copy(),
                start=d.qpos[qadr:qadr+3].copy())


def quat_orientation_delta_deg(q0: np.ndarray, q: np.ndarray) -> float:
    # Quaternion sign is immaterial: q and -q represent the same orientation.
    dot = float(np.clip(abs(np.dot(q0, q)), 0, 1))
    return float(np.degrees(2*np.arccos(dot)))


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=float)


def relative_world_tilt_deg(q_initial: np.ndarray, q_current: np.ndarray) -> float:
    """Tilt of relative orientation applied to world vertical, excluding pure world yaw.

    Match analyze_phino_physics_sweep.py::_motion_and_tilt. The older
    comparison of each pose's *local body z-axis* was misleading if an object's
    initial local z-axis was not aligned with world vertical: pure world-yaw
    could register as a large 'tilt'. This diagnostic does not by itself
    establish whether an object's reconstructed geometry physically toppled.
    """
    q0 = np.asarray(q_initial, dtype=float)
    q1 = np.asarray(q_current, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    qrel = quaternion_multiply(q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
    qrel /= np.linalg.norm(qrel)
    # Rotating world [0,0,1] by qrel produces z = 1 - 2*(x*x+y*y).
    # The angle to world up is 2*asin(sqrt(x*x+y*y)).
    xy_norm = float(np.hypot(qrel[1], qrel[2]))
    return float(np.degrees(2 * np.arcsin(np.clip(xy_norm, 0.0, 1.0))))


def simulate_action(obj, name: str, originals: dict, engine, mujoco,
                    max_displacement: float, max_tilt_deg: float):
    m = obj["model"]
    bid, qadr, vadr = obj["bid"], obj["qadr"], obj["vadr"]
    d = mujoco.MjData(m)
    d.qpos[:] = obj["q0"]
    d.qvel[:] = obj["v0"]
    mujoco.mj_forward(m, d)
    assert_state(d, mujoco, f"{name} initial")
    original_q = d.qpos[qadr+3:qadr+7].copy()
    T = int(engine.timesteps)
    trace = np.empty((T, 13), dtype=np.float32)
    mx_raw = mx_directed = mx_orientation = mx_tilt = 0.
    mx_speed = mx_angular_speed = 0.
    for t in range(T):
        f, tau = applied_wrench(name, t, originals)
        d.xfrc_applied[:] = 0
        d.xfrc_applied[bid, :3] = f
        d.xfrc_applied[bid, 3:] = tau
        for _ in range(engine.substeps):
            mujoco.mj_step(m, d)
            assert_state(d, mujoco, f"{name}, frame={t}")
            dx = d.qpos[qadr:qadr+3] - obj["start"]
            raw = float(np.linalg.norm(dx))
            if not np.isfinite(raw) or raw > max_displacement:
                raise RejectedTrial(f"raw displacement {raw:.4f}m exceeds "
                                    f"{max_displacement:.3f}m at frame={t}, t={d.time:.3f}s")
            mx_raw = max(mx_raw, raw)
            directed = float(dx[2] if name == "grasp_lift_release"
                             else np.dot(dx, originals[name].direction))
            mx_directed = max(mx_directed, directed)
            q = d.qpos[qadr+3:qadr+7]
            mx_orientation = max(mx_orientation, quat_orientation_delta_deg(original_q,q))
            mx_tilt = max(mx_tilt, relative_world_tilt_deg(original_q, q))
            mx_speed = max(mx_speed, float(np.linalg.norm(d.qvel[vadr:vadr+3])))
            mx_angular_speed = max(mx_angular_speed,
                                   float(np.linalg.norm(d.qvel[vadr+3:vadr+6])))
        trace[t, :3] = d.qpos[qadr:qadr+3] - obj["start"]
        trace[t, 3:7] = d.qpos[qadr+3:qadr+7]
        trace[t, 7:10] = d.qvel[vadr:vadr+3]
        trace[t, 10:13] = d.qvel[vadr+3:vadr+6]
    if not np.isfinite(trace).all():
        raise RejectedTrial("Nonfinite saved trajectory")
    metrics = dict(
        directed_m=round(mx_directed,6),
        max_raw_m=round(mx_raw,6),
        max_orientation_deg=round(mx_orientation,3),
        max_tilt_deg=round(mx_tilt,3),
        max_speed_m_s=round(mx_speed,5),
        max_angular_speed_rad_s=round(mx_angular_speed,5),
        would_clip_phino_velocities=bool(mx_speed > engine.lin_vel_clip
                                         or mx_angular_speed > engine.ang_vel_clip),
    )
    if mx_tilt > max_tilt_deg:
        raise RejectedTrial(f"tilt {mx_tilt:.1f} degrees exceeds {max_tilt_deg:.1f}")
    if metrics["would_clip_phino_velocities"]:
        raise RejectedTrial("Velocity would be clipped by PHINO")
    return metrics, trace


def _summarize(records: list[dict], n_objects: int, masses: list[float],
               frictions: list[float]) -> dict[str, Any]:
    """Summarize actual independent action outcomes without imputing failures."""
    count = n_objects * len(masses) * len(frictions)
    if len(records) != count:
        raise RuntimeError(f"Expected {count} object-physics records; got {len(records)}")
    action_counts: dict[str, dict[str, int]] = {}
    for action_name in ACTION_ORDER:
        statuses = [r["action_status"][action_name] for r in records]
        action_counts[action_name] = {
            "expected": count,
            "valid": sum(a["valid"] for a in statuses),
            "invalid": sum(a["status"] == "rejected" for a in statuses),
            "skipped": sum(a["status"] == "skipped" for a in statuses),
            "in_motion_target": sum(a.get("motion_target_met", False) for a in statuses),
        }
        if (action_counts[action_name]["valid"]
                + action_counts[action_name]["invalid"]
                + action_counts[action_name]["skipped"] != count):
            raise AssertionError(f"Bad action accounting: {action_name}")
    complete = sum(r["status"] == "accepted" for r in records)
    return {
        "object_count": n_objects,
        "physics_configurations_per_object": len(masses) * len(frictions),
        "expected_configurations": count,
        "complete_fingerprints": complete,
        "incomplete_fingerprints": count - complete,
        "action_counts": action_counts,
    }


def _write_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    """One row per physical configuration and action, including failures."""
    columns = [
        "object", "mass_kg", "contact_friction", "action", "status", "valid",
        "motion_target_met", "directed_m", "max_raw_m", "max_tilt_deg",
        "max_orientation_deg", "max_speed_m_s", "max_angular_speed_rad_s",
        "would_clip_phino_velocities", "trajectory_path", "rejection_reason",
        "complete_fingerprint_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for record in records:
            for action_name in ACTION_ORDER:
                state = record["action_status"][action_name]
                metrics = record["action_metrics"].get(action_name, {})
                row = {
                    "object": record["object"],
                    "mass_kg": record["mass_kg"],
                    "contact_friction": record["contact_friction"],
                    "action": action_name,
                    "status": state["status"],
                    "valid": state["valid"],
                    "motion_target_met": state.get("motion_target_met", ""),
                    "trajectory_path": state.get("trajectory_path", ""),
                    "rejection_reason": state.get("reason", ""),
                    "complete_fingerprint_path": record.get("fingerprint_path", ""),
                }
                for metric in columns:
                    if metric in metrics:
                        row[metric] = metrics[metric]
                w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=Path("results/rigid_0103/scene.xml"))
    ap.add_argument("--additional-scene", type=Path, action="append", default=[],
                    help="Additional SIMPACT scene.xml; repeat for multiple sources")
    ap.add_argument("--scene-glob", action="append", default=[],
                    help="Glob for more reconstructions, e.g. 'results/rigid_*/scene.xml'")
    # Use a distinct output directory so the successful nominal pilot is preserved.
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Default: grid_v1 for one scene, multi_object_grid_v1 for multiple scenes")
    ap.add_argument("--phino-root", type=Path,
                    default=Path("/nbu/people/ashah264/phino"))
    ap.add_argument("--masses", type=parse_levels, default=[0.25, 0.30, 0.35],
                    help="Comma-separated kg values; default 0.25,0.30,0.35")
    ap.add_argument("--frictions", type=parse_levels, default=[0.20, 0.30, 0.40],
                    help="Comma-separated contact friction; default 0.20,0.30,0.40")
    ap.add_argument("--max-displacement", type=float, default=0.25)
    ap.add_argument("--max-tilt-deg", type=float, default=60.)
    ap.add_argument("--min-motion", type=float, default=0.02)
    ap.add_argument("--max-motion", type=float, default=0.15)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        # Test the accounting and CSV helpers without MuJoCo.
        synthetic = []
        for m in (0.25, 0.30, 0.35):
            for mu in (0.2, 0.3, 0.4):
                synthetic.append({
                    "object": "example", "mass_kg": m, "contact_friction": mu,
                    "status": "rejected", "action_metrics": {},
                    "action_status": {
                        a: {"status": "rejected" if a == "grasp_lift_release" else "skipped",
                            "valid": False, "reason": "synthetic test"}
                        for a in ACTION_ORDER
                    },
                })
        summary = _summarize(synthetic, 1, [0.25, 0.30, 0.35], [0.2, 0.3, 0.4])
        assert summary["expected_configurations"] == 9
        assert summary["action_counts"]["grasp_lift_release"]["invalid"] == 9
        print("GRID_ACCOUNTING_SELF_TEST_OK")
        return

    if not (0 < args.min_motion < args.max_motion < args.max_displacement):
        ap.error("Expected 0 < min-motion < max-motion < max-displacement")
    if not (0 < args.max_tilt_deg <= 180):
        ap.error("max-tilt-deg must be within (0,180]")
    if len(set(round(x, 3) for x in args.masses)) != len(args.masses):
        ap.error("Masses must be unique at 0.001 kg filename precision")
    if len(set(round(x, 3) for x in args.frictions)) != len(args.frictions):
        ap.error("Frictions must be unique at 0.001 filename precision")

    import mujoco  # Delay import so --self-test works without MuJoCo.
    scenes = discover_scenes(args.scene, args.additional_scene, args.scene_glob)
    multi_scene = len(scenes) > 1
    final_output_dir = (args.output_dir if args.output_dir is not None else
                        Path("results/phino_multi_object_grid_v1" if multi_scene
                             else "results/phino_pilot_grid_v1"))
    final_output_dir = final_output_dir.expanduser().resolve()
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Avoid simultaneous jobs publishing into the same dataset directory.
    lock_path = final_output_dir.with_name(f".{final_output_dir.name}.lock")
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error(f"Another generator is already writing {final_output_dir}")
    if not args.phino_root.is_dir():
        raise FileNotFoundError(f"PHINO repository not found: {args.phino_root}")
    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.data.mujoco_engine import DIAGNOSTIC_ACTIONS, MuJoCoPhysicsEngine
    originals = {a.name: a for a in DIAGNOSTIC_ACTIONS}
    if not set(ACTION_ORDER).issubset(originals):
        raise RuntimeError("PHINO is missing one or more required actions")
    engine = MuJoCoPhysicsEngine()
    if int(engine.timesteps) != 50 or not np.isclose(engine.frame_dt, 0.05):
        raise RuntimeError("Expected PHINO 50 frames at 20 Hz")

    # Build into a fresh sibling directory. A previous successful run survives
    # a simulation crash, unlike deleting output NPZ files at run start.
    stage = Path(tempfile.mkdtemp(prefix=f".{final_output_dir.name}.stage-",
                                  dir=str(final_output_dir.parent))).resolve()
    args.output_dir = stage
    atexit.register(shutil.rmtree, stage, ignore_errors=True)
    isolation = [entry for scene in scenes
                 for entry in isolate_objects(scene, stage / "isolated_scenes", qualify=multi_scene)]
    if len({obj["object_id"] for obj in isolation}) != len(isolation):
        raise RuntimeError("Object ID collision across scenes")
    if len({_slug(obj["object_id"]) for obj in isolation}) != len(isolation):
        raise RuntimeError("Object filename collision after sanitization")
    out_fps = stage / "fingerprints"
    out_actions = stage / "action_trajectories"
    out_fps.mkdir(exist_ok=True)
    out_actions.mkdir(exist_ok=True)

    scene_sha = hashlib.sha256(scenes[0].read_bytes()).hexdigest()
    settings_record = {
        "format_version": "phino_simpact_four_action_physics_grid_v2_multi_scene",
        "source_scene": str(scenes[0]), "source_scene_sha256": scene_sha,
        "source_scenes": [{"path": str(scene), "scene_id": _scene_id(scene),
                           "sha256": hashlib.sha256(scene.read_bytes()).hexdigest()}
                          for scene in scenes],
        "source_object_index": [
            {key: entry[key] for key in ("object_id", "body_name", "scene_id",
                                        "source_scene", "source_scene_sha256")}
            for entry in isolation],
        "sampling_Hz": 1. / engine.frame_dt,
        "frames_per_action": int(engine.timesteps),
        "action_order": list(ACTION_ORDER),
        "action_settings": SETTINGS,
        "action_physics_semantics": "Fixed external world-frame wrench; not a physical robot contact/grasp",
        "feature_order": "[relative_position_xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz]",
        "feature_clipping": False,
        "physics_grid_is_synthetic": True,
        "nominal_mass_is_not_measured": True,
        "contact_friction_semantics": "Sliding friction applied to all contact geoms; not a measured material coefficient",
        "supervision_scope": "simulation dynamics only; not image-based hidden parameter inference",
        "max_raw_displacement_m": args.max_displacement,
        "max_tilt_deg": args.max_tilt_deg,
        "motion_target_min_m": args.min_motion,
        "motion_target_max_m": args.max_motion,
        "mass_grid_kg": args.masses,
        "friction_grid": args.frictions,
        "validity_semantics": (
            "Valid: numerically finite, within raw displacement and relative tilt caps, "
            "and no PHINO velocity clipping. Motion target is separate from validity. "
            "Rejected actions have no complete saved 50-frame trajectory; "
            "other valid actions from the same configuration are retained."
        ),
    }
    (args.output_dir / "action_settings.json").write_text(
        json.dumps(settings_record, indent=2) + "\n", encoding="utf-8")

    records = []
    print(f"Loaded {len(isolation)} isolated objects from {len(scenes)} source scene(s): "
          + ", ".join(entry["object_id"] for entry in isolation))
    print(f"Output (published only after completion): {final_output_dir}")
    print(f"Actions: {', '.join(ACTION_ORDER)}; each 50x13; full fingerprint 200x13")
    print(f"Physics mass x friction: {args.masses} x {args.frictions}")
    print(f"Expected configurations: {len(isolation) * len(args.masses) * len(args.frictions)}")
    for entry in isolation:
        name = entry["body_name"]
        object_id = entry["object_id"]
        one_scene = entry["isolated_scene"]
        for mass in args.masses:
            for friction in args.frictions:
                slug = f"{_slug(object_id)}_m{mass:.3f}_mu{friction:.3f}"
                row = {
                    "object": object_id, "source_body_name": name,
                    "source_scene": entry["source_scene"],
                    "source_scene_sha256": entry["source_scene_sha256"],
                    "mass_kg": mass,
                    "contact_friction": friction,
                    "action_metrics": {}, "action_status": {},
                    "status": "rejected",
                }
                try:
                    obj = build_model(one_scene, name, mass, friction, engine, mujoco)
                except RejectedTrial as exc:
                    row["model_error"] = str(exc)
                    for action_name in ACTION_ORDER:
                        row["action_status"][action_name] = {
                            "status": "skipped", "valid": False,
                            "reason": f"Model settling failed: {exc}",
                        }
                    print(f"MODEL_REJECT {slug}: {exc}")
                    records.append(row)
                    continue

                valid_traces: dict[str, np.ndarray] = {}
                for action_name in ACTION_ORDER:
                    try:
                        metrics, trajectory = simulate_action(
                            obj, action_name, originals, engine, mujoco,
                            args.max_displacement, args.max_tilt_deg)
                    except RejectedTrial as exc:
                        row["action_status"][action_name] = {
                            "status": "rejected", "valid": False,
                            "reason": str(exc),
                        }
                        print(f"  REJECT {slug} {action_name}: {exc}")
                        continue

                    if trajectory.shape != (50, 13):
                        raise RuntimeError(f"Unexpected {slug} {action_name} shape {trajectory.shape}")
                    meets_target = (args.min_motion <= metrics["directed_m"] <= args.max_motion
                                    and metrics["max_raw_m"] <= args.max_motion)
                    dest = out_actions / f"{slug}__{action_name}.npz"
                    np.savez_compressed(
                        dest, trajectory=trajectory,
                        object_name=object_id, source_body_name=name,
                        action_name=action_name,
                        mass_kg=mass, contact_friction=friction,
                        frames_per_action=int(engine.timesteps),
                    )
                    row["action_metrics"][action_name] = {
                        **metrics, "motion_target_met": bool(meets_target),
                    }
                    row["action_status"][action_name] = {
                        "status": "valid", "valid": True,
                        "motion_target_met": bool(meets_target),
                        "trajectory_path": f"action_trajectories/{dest.name}",
                    }
                    valid_traces[action_name] = trajectory
                    print(f"  VALID  {slug} {action_name}: "
                          f"directed={metrics['directed_m']:.3f}m "
                          f"target={'yes' if meets_target else 'no'}")

                if len(valid_traces) == len(ACTION_ORDER):
                    fp = np.concatenate([valid_traces[a] for a in ACTION_ORDER], axis=0)
                    if fp.shape != (200, 13):
                        raise RuntimeError(f"Unexpected {slug} fingerprint shape {fp.shape}")
                    dest = out_fps / f"{slug}.npz"
                    np.savez_compressed(
                        dest, flat_trajectory=fp,
                        action_names=np.asarray(ACTION_ORDER),
                        mass_kg=mass, contact_friction=friction,
                        frames_per_action=int(engine.timesteps),
                    )
                    row["status"] = "accepted"
                    row["fingerprint_path"] = f"fingerprints/{dest.name}"
                    row["all_actions_in_motion_target"] = all(
                        row["action_status"][a]["motion_target_met"] for a in ACTION_ORDER)
                    print(f"ACCEPT {slug} full (200,13) "
                          f"all_targets={row['all_actions_in_motion_target']}")
                else:
                    failed = [a for a in ACTION_ORDER if not row["action_status"][a]["valid"]]
                    row["failed_actions"] = failed
                    print(f"INCOMPLETE {slug} valid_actions={len(valid_traces)}/4 "
                          f"failed={','.join(failed)}")
                records.append(row)

    summary = _summarize(records, len(isolation), args.masses, args.frictions)
    manifest = {**settings_record, "records": records, "summary": summary,
                "accepted_count": summary["complete_fingerprints"],
                "rejected_count": summary["incomplete_fingerprints"],
                "total_count": summary["expected_configurations"],
                "training_ready": False}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "action_trials.csv"
    _write_csv(records, csv_path)
    print("\nPer-action validity (valid / expected, in motion target):")
    for action_name in ACTION_ORDER:
        c = summary["action_counts"][action_name]
        print(f"  {action_name:<20} {c['valid']:>2}/{c['expected']:<2} "
              f"target={c['in_motion_target']:<2} rejected={c['invalid']:<2} "
              f"skipped={c['skipped']:<2}")
    # Only publish the complete manifest and all NPZ files together.
    publish_staged_output(stage, final_output_dir)
    print(f"MANIFEST={final_output_dir / 'manifest.json'}")
    print(f"ACTION_TRIALS_CSV={final_output_dir / 'action_trials.csv'}")
    print(f"COMPLETE_FINGERPRINTS={summary['complete_fingerprints']}/{summary['expected_configurations']}")
    print("SIMULATION_GRID_ONLY: synthetic physical-property labels; no PHINO training")
    print("PHINO_PHYSICS_GRID_OK")


if __name__ == "__main__":
    main()

