#!/usr/bin/env python3
"""SIMPACT v3 simulation primitives (docs/phino_v3_identifiability_spec.md).

Library used by the Stage 3 identifiability pilot. It implements:
  * S1 canonical at-rest pose per geometry (reference physics, damped relaxation),
  * S2 per-configuration start from that pose with an undamped hold check,
  * S3 zero-action control,
  * v3 interventions (pure centre-of-mass pushes, toss) plus the unchanged v2 push,
  * v2 validity rules, and start-relative recording at 20 Hz and 100 Hz.

The v2 generator is not modified; its model construction and validity helpers
are imported. A full-grid CLI is deliberately absent until the Stage 3 gate is
approved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import phino_simpact_pilot_v2 as pilot
from audit_phino_settle_confound import qconj, qmul, quat_to_mat

REFERENCE_MASS = 0.30
REFERENCE_FRICTION = 0.30
RELAX_DAMPING = 0.9
RELAX_MAX_LINEAR = 1e-4
RELAX_MAX_ANGULAR = 1e-3
RELAX_CALM_S = 0.5
RELAX_MAX_S = 10.0
HOLD_S = 0.5
S2_MAX_OFFSET_M = 1e-4
S2_MAX_ROTATION_DEG = 0.1
S2_MAX_LINEAR = 1e-3
S2_MAX_ANGULAR = 1e-2
S3_MAX_DISPLACEMENT_M = 5e-4
S3_MAX_ROTATION_DEG = 0.5
HIGH_RATE_PER_FRAME = 5          # 100 Hz samples per 20 Hz frame


@dataclass(frozen=True)
class Action:
    """Fixed world-frame wrench schedule; never depends on hidden mass or friction."""
    name: str
    family: str
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frames: int = 0
    v2_reference: bool = False

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.force))


ACTIONS = (
    Action("push_x_v2", "push_x_v2", frames=4, v2_reference=True),
    Action("push_lo_1.6N", "push_lo", (1.6, 0.0, 0.0), 4),
    Action("push_lo_2.0N", "push_lo", (2.0, 0.0, 0.0), 4),
    Action("push_hi_3.2N", "push_hi", (3.2, 0.0, 0.0), 2),
    Action("push_hi_4.0N", "push_hi", (4.0, 0.0, 0.0), 2),
    Action("toss_4.0N", "toss", (0.0, 0.0, 4.0), 4),
    Action("toss_4.5N", "toss", (0.0, 0.0, 4.5), 4),
)
ZERO_ACTION = Action("zero_action", "none")
ACTION_BY_NAME = {a.name: a for a in ACTIONS}


def wrench(action: Action, frame: int, originals: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if action.v2_reference:
        return pilot.applied_wrench("push_x", frame, originals)
    force = np.zeros(3)
    if frame < action.frames:
        force[:] = action.force
    return force, np.zeros(3)


@dataclass
class Sim:
    model: Any
    bid: int
    qadr: int
    vadr: int
    object_geoms: np.ndarray
    substeps: int
    settle_q0: np.ndarray = field(repr=False)
    settle_v0: np.ndarray = field(repr=False)


def make_sim(scene, body: str, mass: float, friction: float, engine, mujoco,
             timestep_scale: float = 1.0) -> Sim:
    """v2 parameter application (mass/inertia scale, friction on all collision geoms)."""
    obj = pilot.build_model(scene, body, mass, friction, engine, mujoco)
    model = obj["model"]
    substeps = int(round(engine.substeps / timestep_scale))
    if timestep_scale != 1.0:
        model.opt.timestep = engine.sim_timestep * timestep_scale
    if substeps % HIGH_RATE_PER_FRAME:
        raise ValueError(f"{substeps} substeps per frame not divisible by {HIGH_RATE_PER_FRAME}")
    bid = obj["bid"]

    def in_subtree(b: int) -> bool:
        while b > 0:
            if b == bid:
                return True
            b = int(model.body_parentid[b])
        return False

    geoms = np.array([g for g in range(model.ngeom) if in_subtree(int(model.geom_bodyid[g]))])
    return Sim(model, bid, obj["qadr"], obj["vadr"], geoms, substeps, obj["q0"], obj["v0"])


def _speeds(sim: Sim, data) -> tuple[float, float]:
    return (float(np.linalg.norm(data.qvel[sim.vadr:sim.vadr + 3])),
            float(np.linalg.norm(data.qvel[sim.vadr + 3:sim.vadr + 6])))


def canonical_rest_pose(sim: Sim, mujoco) -> dict[str, Any]:
    """S1: damped relaxation from the v2 settle at reference physics (damping only here).

    ``converged`` is the approved speed criterion. MuJoCo's resting mesh contacts
    chatter at speeds near those thresholds, so the position drift over the last
    ``RELAX_CALM_S`` is also reported (``final_window_drift_m``).
    """
    m = sim.model
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = sim.settle_q0, sim.settle_v0
    mujoco.mj_forward(m, d)
    dt = float(m.opt.timestep)
    need, calm = int(round(RELAX_CALM_S / dt)), 0
    history = []
    converged, steps = False, int(round(RELAX_MAX_S / dt))
    for step in range(steps):
        d.xfrc_applied[:] = 0
        mujoco.mj_step(m, d)
        pilot.assert_state(d, mujoco, f"canonical relaxation step={step}")
        d.qvel[:] *= RELAX_DAMPING
        history.append(d.qpos[sim.qadr:sim.qadr + 3].copy())
        lin, ang = _speeds(sim, d)
        calm = calm + 1 if (lin < RELAX_MAX_LINEAR and ang < RELAX_MAX_ANGULAR) else 0
        if calm >= need:
            converged, steps = True, step + 1
            break
    window = np.array(history[-need:])
    drift = float(np.linalg.norm(window - window[0], axis=1).max())
    return {"q_star": d.qpos.copy(), "converged": converged, "relax_time_s": steps * dt,
            "final_window_drift_m": drift}


def start_state(sim: Sim, q_star: np.ndarray, engine, mujoco) -> dict[str, Any]:
    """S2: place at q*, zero velocity, undamped zero-action hold; start = post-hold state."""
    m = sim.model
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = q_star, 0.0
    mujoco.mj_forward(m, d)
    q_ref = q_star[sim.qadr + 3:sim.qadr + 7]
    for _ in range(int(round(HOLD_S / engine.frame_dt)) * sim.substeps):
        d.xfrc_applied[:] = 0
        mujoco.mj_step(m, d)
        pilot.assert_state(d, mujoco, "S2 hold")
    offset = float(np.linalg.norm(d.qpos[sim.qadr:sim.qadr + 3] - q_star[sim.qadr:sim.qadr + 3]))
    rotation = pilot.quat_orientation_delta_deg(q_ref, d.qpos[sim.qadr + 3:sim.qadr + 7])
    lin, ang = _speeds(sim, d)
    ok = (offset < S2_MAX_OFFSET_M and rotation < S2_MAX_ROTATION_DEG
          and lin < S2_MAX_LINEAR and ang < S2_MAX_ANGULAR)
    return {"q0": d.qpos.copy(), "v0": d.qvel.copy(), "ok": bool(ok), "offset_m": offset,
            "rotation_deg": rotation, "linear_speed": lin, "angular_speed": ang}


def start_relative(raw: np.ndarray, start: np.ndarray, q_start: np.ndarray) -> np.ndarray:
    """raw [pos, quat, linvel(world), angvel(body)] -> [pos-start, rotation since start, linvel, angvel(world)]."""
    out = np.empty_like(raw)
    out[:, :3] = raw[:, :3] - start
    q_rel = qmul(raw[:, 3:7], qconj(q_start)[None, :])
    sign = np.sign(q_rel[0, 0]) or 1.0
    q_rel[0] *= sign
    for k in range(1, len(q_rel)):
        if np.dot(q_rel[k], q_rel[k - 1]) < 0:
            q_rel[k] *= -1
    out[:, 3:7] = q_rel
    out[:, 7:10] = raw[:, 7:10]
    out[:, 10:13] = np.einsum("tij,tj->ti", quat_to_mat(raw[:, 3:7]), raw[:, 10:13])
    return out


def rollout(sim: Sim, q0: np.ndarray, v0: np.ndarray, action: Action, engine, mujoco,
            originals: dict[str, Any], max_displacement: float, max_tilt_deg: float
            ) -> dict[str, Any]:
    """One 50-frame action with v2 validity rules; start-relative traces at 20 and 100 Hz."""
    m = sim.model
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_forward(m, d)
    qa, va = sim.qadr, sim.vadr
    start, q_start = q0[qa:qa + 3].copy(), q0[qa + 3:qa + 7].copy()
    frames = int(engine.timesteps)
    per_sample = sim.substeps // HIGH_RATE_PER_FRAME
    raw20 = np.empty((frames, 13))
    raw100 = np.empty((frames * HIGH_RATE_PER_FRAME, 13))
    max_raw = max_tilt = max_lin = max_ang = 0.0
    airborne = 0

    def state() -> np.ndarray:
        return np.r_[d.qpos[qa:qa + 7], d.qvel[va:va + 6]]

    try:
        for t in range(frames):
            force, torque = wrench(action, t, originals)
            touching = False
            for s in range(sim.substeps):
                d.xfrc_applied[:] = 0
                d.xfrc_applied[sim.bid, :3] = force
                d.xfrc_applied[sim.bid, 3:] = torque
                mujoco.mj_step(m, d)
                pilot.assert_state(d, mujoco, f"{action.name} frame={t}")
                raw = float(np.linalg.norm(d.qpos[qa:qa + 3] - start))
                if not np.isfinite(raw) or raw > max_displacement:
                    raise pilot.RejectedTrial(f"raw displacement {raw:.4f}m exceeds "
                                              f"{max_displacement:.3f}m at frame={t}")
                max_raw = max(max_raw, raw)
                max_tilt = max(max_tilt, pilot.relative_world_tilt_deg(q_start, d.qpos[qa + 3:qa + 7]))
                lin, ang = _speeds(sim, d)
                max_lin, max_ang = max(max_lin, lin), max(max_ang, ang)
                if d.ncon and (np.isin(d.contact.geom1[:d.ncon], sim.object_geoms).any()
                               or np.isin(d.contact.geom2[:d.ncon], sim.object_geoms).any()):
                    touching = True
                if (s + 1) % per_sample == 0:
                    raw100[t * HIGH_RATE_PER_FRAME + (s + 1) // per_sample - 1] = state()
            raw20[t] = state()
            airborne += 0 if touching else 1
        if max_tilt > max_tilt_deg:
            raise pilot.RejectedTrial(f"tilt {max_tilt:.1f} degrees exceeds {max_tilt_deg:.1f}")
        if max_lin > engine.lin_vel_clip or max_ang > engine.ang_vel_clip:
            raise pilot.RejectedTrial("Velocity would be clipped by PHINO")
    except pilot.RejectedTrial as exc:
        return {"valid": False, "reason": str(exc)}
    return {"valid": True, "reason": "",
            "trace20": start_relative(raw20, start, q_start).astype(np.float32),
            "trace100": start_relative(raw100, start, q_start).astype(np.float32),
            "airborne_frames": int(airborne), "max_raw_m": max_raw, "max_tilt_deg": max_tilt}


def object_radius(sim: Sim, qpos: np.ndarray, mujoco) -> float:
    d = mujoco.MjData(sim.model)
    d.qpos[:] = qpos
    mujoco.mj_forward(sim.model, d)
    center = d.xpos[sim.bid]
    return max(float(np.linalg.norm(d.geom_xpos[g] - center)) + float(sim.model.geom_rbound[g])
               for g in sim.object_geoms)
