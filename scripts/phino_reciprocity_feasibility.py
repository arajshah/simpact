#!/usr/bin/env python3
"""Phase 2 feasibility: Maxwell–Betti reciprocity on a planar spring bracket.

This is a physics and task study. It does not train a model and it does not
modify the Phase 1 grids.

Physical object
---------------
Each fixture is a clamped planar assembly of rigid links connected by revolute
springs. Gravity is off, so the unloaded shape is an equilibrium and the
measured map is the incremental response about that shape. A real bracket would
sag under gravity; for linear springs the incremental compliance about a sagged
equilibrium matches the unloaded compliance only when preload geometric
stiffness is negligible. This study does not include that preload.

The springs are lumped torsional springs with one shared design value. That is
not a validated continuum beam. MuJoCo integrates a damped mechanism and the
reported response is the settled static displacement, not an analytical proof
that reciprocity holds.

Coordinates
-----------
The clamp is at the origin. Links lie in the xy-plane and hinges rotate about
+z. A positive joint angle moves the distal structure counterclockwise. Forces
and displacements are in the world frame. Sites are material ports: the applied
force stays aligned with the world axes and follows the port.

Small-displacement model
------------------------
With joint angles θ and a diagonal spring matrix K,

    K θ = J(q)ᵀ f,    δp = J θ,    C = J K⁻¹ Jᵀ,    C = Cᵀ,

at the unloaded configuration, where geometric stiffness vanishes because there
is no preload. J is the planar cross product of each port's position relative
to its ancestor joints. C_ij (2×2) maps a force at port j to a displacement at
port i, and symmetry is the block statement C_ij = C_jiᵀ. Equal force vectors
at two ports produce equal displacements *along that common force direction*;
the full displacement vectors are not claimed to be equal.

The secant measured here is Δd / f at a finite force. It equals C only up to
terms that vanish with f. Axial shortening of a bending arm is one such term.

Pre-registered task and acceptance rule
---------------------------------------
Weights, the goal point, the force set, and the acceptance tolerances below
were fixed before the MuJoCo ranking. They were chosen on the linearized
straight bracket so its best action sits in the interior of the force set,
then checked once, still on the linear model, so the L-bracket's preferred
direction is not that same action. They were not revised after MuJoCo ran.

A reciprocal pair at the probe force agrees when

    ||f (C_ij − C_jiᵀ)||_F  ≤  max(25 µm,  0.02 × larger block's displacement scale).

25 µm is 0.5% of the 5 mm alignment target. The 2% branch covers a large
response, where 2% of a 5 mm motion is 100 µm. Components smaller than the
floor are not asked to meet a relative tolerance. This rule is a usefulness
bound for the decision, not a fit to the observed error.

Simulator credibility requires the settled velocity to be under 1e-5 rad/s and
the spread of the probe response across resets, timesteps, settle times, and
solver tolerances to stay under that same 25 µm floor. If the numerical
envelope itself exceeds the floor, the fixture is not precise enough for the
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# --- Pre-registered mechanics and task constants ----------------------------

PUBLIC_STIFFNESS = 4.0  # N·m/rad, uniform declared spring rate
LINK_LENGTH = 0.04  # m
DAMPING = 0.5  # N·m·s/rad, chosen so the straight chain settles inside 1.2 s
LINK_MASS = 0.015  # kg, dynamics only; the static map does not depend on it
TIMESTEP = 0.001  # s
SETTLE_TIME = 1.2  # s
SOLVER_TOLERANCE = 1e-12
SOLVER_ITERATIONS = 80
JOINT_RANGE = (-1.2, 1.2)  # rad; nominal loads stay far inside
GRAVITY = (0.0, 0.0, 0.0)

GOAL_POINT = np.array([0.40, 0.14])  # m, same workspace target for every geometry
ALIGN_TARGET_M = 0.005
PROTECT_DEADBAND_M = 0.0015
PROTECT_SCALE_M = 0.0035
W_PROTECT = 0.55
W_FORCE = 0.18
FORCE_REFERENCE_N = 0.15
FORCE_MAGNITUDES = (0.05, 0.15)  # N
PROBE_FORCE_N = 0.05
BREAKDOWN_FORCE_N = 3.0
CONTACT_GAP_M = 0.0025
CONTACT_FORCE_N = 0.15
BUMPER_RADIUS = 0.006

DIRECTIONS = {
    "+x": np.array([1.0, 0.0]),
    "-x": np.array([-1.0, 0.0]),
    "+y": np.array([0.0, 1.0]),
    "-y": np.array([0.0, -1.0]),
}
BASIS_DIRECTIONS = ("+x", "+y")

# Usefulness floor: 0.5% of the alignment target. Not fit to a measured error.
DECISION_FLOOR_M = 0.005 * ALIGN_TARGET_M  # 25 µm
RELATIVE_BLOCK_TOL = 0.02
CLEAR_UTILITY_GAP = 0.05
SETTLED_QVEL = 1e-5

FIXED_BLIND_ACTION = ("a0", "+y", 0.05)
DIRECT_BUDGET_ACTIONS = (
    ("a0", "+y", 0.05),
    ("a0", "+x", 0.05),
    ("a7", "+y", 0.05),
    ("a7", "+x", 0.05),
)


@dataclass(frozen=True)
class LinkSpec:
    name: str
    parent: str | None
    angle_deg: float
    length: float
    stiffness: float


@dataclass(frozen=True)
class PortSpec:
    name: str
    link: str
    fraction: float
    kind: str  # "action", "align", or "protect"


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    links: tuple[LinkSpec, ...]
    ports: tuple[PortSpec, ...]
    description: str


@dataclass(frozen=True)
class VisibleGeometry:
    """Rest shape and the public spring rate. No simulator stiffness."""

    link_names: tuple[str, ...]
    anchors: np.ndarray
    site_names: tuple[str, ...]
    site_positions: np.ndarray
    ancestors: tuple[tuple[int, ...], ...]
    public_stiffness: float


def _chain(n: int, angle: float, prefix: str, parent0: str | None, stiffness: float) -> list[LinkSpec]:
    links = []
    for i in range(n):
        parent = parent0 if i == 0 else f"{prefix}{i - 1}"
        links.append(LinkSpec(f"{prefix}{i}", parent, angle, LINK_LENGTH, stiffness))
    return links


def _actions(specs: list[tuple[str, float]]) -> list[PortSpec]:
    if len(specs) != 8:
        raise ValueError(f"expected 8 action ports, got {len(specs)}")
    return [PortSpec(f"a{i}", link, frac, "action") for i, (link, frac) in enumerate(specs)]


def _fixture(name, links, actions, align, protect, description, stiffness) -> FixtureSpec:
    ports = tuple(actions + [PortSpec("align", *align, "align"), PortSpec("protect", *protect, "protect")])
    spec = FixtureSpec(name, tuple(links), ports, description)
    validate_spec(spec)
    return spec


def straight_spec(stiffness: float = PUBLIC_STIFFNESS) -> FixtureSpec:
    links = _chain(9, 0.0, "m", None, stiffness)
    actions = _actions([(f"m{i}", 1.0) for i in range(8)])
    return _fixture(
        "straight",
        links,
        actions,
        ("m8", 1.0),
        ("m3", 1.0),
        "Straight cantilever. Positive control. The protected station is the same "
        "material point as action port a3.",
        stiffness,
    )


def branched_spec(stiffness: float = PUBLIC_STIFFNESS) -> FixtureSpec:
    links = _chain(7, 0.0, "m", None, stiffness) + _chain(2, 60.0, "b", "m2", stiffness)
    actions = _actions(
        [("m0", 1.0), ("m1", 1.0), ("m3", 1.0), ("m4", 1.0), ("m5", 1.0), ("m6", 0.5), ("b0", 1.0), ("b1", 0.5)]
    )
    return _fixture(
        "branched",
        links,
        actions,
        ("m6", 1.0),
        ("b1", 1.0),
        "Main arm with a 60° branch. Alignment is the main tip; the protected point is the branch tip.",
        stiffness,
    )


def ell_spec(stiffness: float = PUBLIC_STIFFNESS) -> FixtureSpec:
    links = _chain(4, 0.0, "m", None, stiffness) + _chain(5, 90.0, "u", "m3", stiffness)
    actions = _actions(
        [("m0", 1.0), ("m1", 1.0), ("m2", 1.0), ("u0", 1.0), ("u1", 1.0), ("u2", 1.0), ("u3", 1.0), ("u4", 0.5)]
    )
    return _fixture(
        "ell",
        links,
        actions,
        ("u4", 1.0),
        ("m3", 1.0),
        "L-bracket. Alignment is the upright tip; the protected point is the elbow.",
        stiffness,
    )


def crank_spec(stiffness: float = PUBLIC_STIFFNESS) -> FixtureSpec:
    links = (
        _chain(3, 0.0, "m", None, stiffness)
        + _chain(3, 80.0, "b", "m2", stiffness)
        + _chain(3, 20.0, "c", "b2", stiffness)
    )
    actions = _actions(
        [("m0", 1.0), ("m1", 1.0), ("m2", 1.0), ("b0", 1.0), ("b1", 1.0), ("c0", 1.0), ("c1", 1.0), ("c2", 0.5)]
    )
    return _fixture(
        "crank",
        links,
        actions,
        ("c2", 1.0),
        ("b0", 0.5),
        "Asymmetric crank: 0°, then 80°, then 20°. Alignment is the free end.",
        stiffness,
    )


def all_specs(stiffness: float = PUBLIC_STIFFNESS) -> tuple[FixtureSpec, ...]:
    return (straight_spec(stiffness), branched_spec(stiffness), ell_spec(stiffness), crank_spec(stiffness))


def validate_spec(spec: FixtureSpec) -> None:
    names = [link.name for link in spec.links]
    if len(names) != len(set(names)):
        raise ValueError(f"{spec.name}: duplicate link names")
    by_name = {link.name: link for link in spec.links}
    for link in spec.links:
        if link.parent is not None and link.parent not in by_name:
            raise ValueError(f"{spec.name}: {link.name} has missing parent {link.parent}")
        if link.stiffness <= 0 or link.length <= 0:
            raise ValueError(f"{spec.name}: non-positive length or stiffness on {link.name}")
    seen = set()
    walking = set()

    def visit(name: str) -> None:
        if name in walking:
            raise ValueError(f"{spec.name}: cycle at {name}")
        if name in seen:
            return
        walking.add(name)
        parent = by_name[name].parent
        if parent is not None:
            visit(parent)
        walking.remove(name)
        seen.add(name)

    for name in names:
        visit(name)
    kinds = [port.kind for port in spec.ports]
    if kinds.count("align") != 1 or kinds.count("protect") != 1:
        raise ValueError(f"{spec.name}: need exactly one align and one protect port")
    actions = [port for port in spec.ports if port.kind == "action"]
    if len(actions) != 8 or [port.name for port in actions] != [f"a{i}" for i in range(8)]:
        raise ValueError(f"{spec.name}: action ports must be a0..a7")
    port_names = [port.name for port in spec.ports]
    if len(port_names) != len(set(port_names)):
        raise ValueError(f"{spec.name}: duplicate port names")
    for port in spec.ports:
        if port.link not in by_name:
            raise ValueError(f"{spec.name}: port {port.name} sits on unknown link {port.link}")
        if not 0.0 <= port.fraction <= 1.0:
            raise ValueError(f"{spec.name}: port {port.name} fraction out of range")


def rest_pose(spec: FixtureSpec) -> dict[str, dict]:
    pose = {}
    for link in spec.links:
        if link.parent is None:
            start = np.zeros(2)
        else:
            start = pose[link.parent]["end"].copy()
        direction = np.array([np.cos(np.deg2rad(link.angle_deg)), np.sin(np.deg2rad(link.angle_deg))])
        pose[link.name] = {"start": start, "end": start + link.length * direction, "angle": link.angle_deg}
    return pose


def ancestor_index(spec: FixtureSpec) -> dict[str, tuple[int, ...]]:
    index = {link.name: i for i, link in enumerate(spec.links)}
    parent = {link.name: link.parent for link in spec.links}
    chains = {}
    for link in spec.links:
        chain = []
        name = link.name
        while name is not None:
            chain.append(index[name])
            name = parent[name]
        chains[link.name] = tuple(chain)
    return chains


def port_position(spec: FixtureSpec, port: PortSpec, pose: dict | None = None) -> np.ndarray:
    pose = rest_pose(spec) if pose is None else pose
    link = pose[port.link]
    return link["start"] + port.fraction * (link["end"] - link["start"])


def visible_geometry(spec: FixtureSpec) -> VisibleGeometry:
    """Declared rest geometry plus the public spring rate.

    Per-link ``stiffness`` on the spec is ignored. A calibrated drawing of the
    bracket supplies the anchors, port coordinates, and which joints are
    ancestors; the spring rate is the public design value.
    """

    pose = rest_pose(spec)
    chains = ancestor_index(spec)
    anchors = np.vstack([pose[link.name]["start"] for link in spec.links])
    positions = np.vstack([port_position(spec, port, pose) for port in spec.ports])
    ancestors = tuple(chains[port.link] for port in spec.ports)
    return VisibleGeometry(
        link_names=tuple(link.name for link in spec.links),
        anchors=anchors,
        site_names=tuple(port.name for port in spec.ports),
        site_positions=positions,
        ancestors=ancestors,
        public_stiffness=PUBLIC_STIFFNESS,
    )


def stacked_jacobian(site_positions: np.ndarray, ancestors: tuple[tuple[int, ...], ...], anchors: np.ndarray) -> np.ndarray:
    n_sites = len(site_positions)
    n_joints = len(anchors)
    jacobian = np.zeros((2 * n_sites, n_joints))
    for i, (position, chain) in enumerate(zip(site_positions, ancestors)):
        for joint in chain:
            relative = position - anchors[joint]
            jacobian[2 * i, joint] = -relative[1]
            jacobian[2 * i + 1, joint] = relative[0]
    return jacobian


def compliance_from_stiffness(
    site_positions: np.ndarray,
    ancestors: tuple[tuple[int, ...], ...],
    anchors: np.ndarray,
    stiffness: np.ndarray,
) -> np.ndarray:
    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.ndim == 0:
        stiffness = np.full(len(anchors), float(stiffness))
    if stiffness.shape != (len(anchors),) or np.any(stiffness <= 0) or not np.all(np.isfinite(stiffness)):
        raise ValueError("stiffness must be a positive finite value per joint")
    jacobian = stacked_jacobian(site_positions, ancestors, anchors)
    compliance = jacobian @ (jacobian.T / stiffness[:, None])
    if not np.allclose(compliance, compliance.T, atol=1e-12):
        raise RuntimeError("analytical compliance is not symmetric")
    return compliance


def deployable_compliance(geometry: VisibleGeometry) -> np.ndarray:
    """Linearized spring model from visible geometry and the public spring rate."""

    return compliance_from_stiffness(
        geometry.site_positions,
        geometry.ancestors,
        geometry.anchors,
        np.full(len(geometry.anchors), geometry.public_stiffness),
    )


def privileged_compliance(spec: FixtureSpec) -> np.ndarray:
    """Same linearization using the spec's per-joint stiffness, which may be hidden."""

    geometry = visible_geometry(spec)
    stiffness = np.array([link.stiffness for link in spec.links], dtype=float)
    return compliance_from_stiffness(geometry.site_positions, geometry.ancestors, geometry.anchors, stiffness)


def goal_direction(align_xy: np.ndarray) -> np.ndarray:
    offset = GOAL_POINT - np.asarray(align_xy, dtype=float)
    norm = float(np.linalg.norm(offset))
    if norm < 1e-9:
        raise ValueError("alignment port is on the goal point")
    return offset / norm


def utility(d_align: np.ndarray, d_protect: np.ndarray, force_mag: float, goal: np.ndarray) -> tuple[float, dict]:
    """Bounded score in [0, 1].

    Alignment saturates once the projection onto the goal direction reaches
    ``ALIGN_TARGET_M``. Motion of the protected point inside the deadband is
    free; beyond it the penalty grows linearly until ``PROTECT_SCALE_M``.
    The force term is the magnitude divided by ``FORCE_REFERENCE_N``.
    """

    progress = float(np.dot(np.asarray(d_align, dtype=float)[:2], goal))
    protect_m = float(np.linalg.norm(np.asarray(d_protect, dtype=float)[:2]))
    align = float(np.clip(progress / ALIGN_TARGET_M, 0.0, 1.0))
    protect = float(np.clip(max(0.0, protect_m - PROTECT_DEADBAND_M) / PROTECT_SCALE_M, 0.0, 1.0))
    force_cost = float(force_mag / FORCE_REFERENCE_N)
    raw = align - W_PROTECT * protect - W_FORCE * force_cost
    score = float(np.clip(raw, 0.0, 1.0))
    if not np.isfinite(score):
        raise RuntimeError("non-finite utility")
    return score, {
        "align": align,
        "protect": protect,
        "force_cost": force_cost,
        "progress_m": progress,
        "protect_m": protect_m,
        "raw": raw,
    }


def displacement_tolerance(disp_scale: float) -> float:
    return max(DECISION_FLOOR_M, RELATIVE_BLOCK_TOL * disp_scale)


def rank_actions(rows: list[dict]) -> dict:
    """Best action, the gap to the next row, and the gap out of the tie set.

    Ties are exact to 1e-12 utility. ``gap_to_next`` is what the brief calls the
    best-versus-second gap, so a two-way tie reports 0. ``gap_to_distinct`` is
    the drop from that tie set to the next worse action.
    """

    if not rows:
        raise ValueError("no actions to rank")
    ordered = sorted(rows, key=lambda row: (-row["utility"], row["port"], row["direction"], row["magnitude"]))
    best = ordered[0]["utility"]
    tie = [row for row in ordered if abs(row["utility"] - best) <= 1e-12]
    rest = [row for row in ordered if abs(row["utility"] - best) > 1e-12]
    return {
        "best": ordered[0],
        "second": ordered[1] if len(ordered) > 1 else None,
        "tie": tie,
        "gap_to_next": 0.0 if len(tie) > 1 else (best - ordered[1]["utility"] if len(ordered) > 1 else 0.0),
        "gap_to_distinct": (best - rest[0]["utility"]) if rest else 0.0,
        "ordered": ordered,
    }


def predict_action_utilities(compliance: np.ndarray, spec: FixtureSpec, goal: np.ndarray) -> list[dict]:
    names = [port.name for port in spec.ports]
    align = names.index("align")
    protect = names.index("protect")
    rows = []
    for port in spec.ports:
        if port.kind != "action":
            continue
        j = names.index(port.name)
        for direction, vector in DIRECTIONS.items():
            for magnitude in FORCE_MAGNITUDES:
                force = magnitude * vector
                d_align = compliance[2 * align : 2 * align + 2, 2 * j : 2 * j + 2] @ force
                d_protect = compliance[2 * protect : 2 * protect + 2, 2 * j : 2 * j + 2] @ force
                score, parts = utility(d_align, d_protect, magnitude, goal)
                rows.append({"port": port.name, "direction": direction, "magnitude": magnitude, "utility": score, **parts})
    return rows


def _rgba(kind: str) -> str:
    return {"action": "0.95 0.75 0.15 1", "align": "0.16 0.70 0.30 1", "protect": "0.84 0.24 0.20 1"}[kind]


def build_xml(spec: FixtureSpec, timestep: float, tolerance: float, contact_port: str | None, contact_gap: float) -> str:
    pose = rest_pose(spec)
    by_name = {link.name: link for link in spec.links}
    children: dict[str, list[str]] = {link.name: [] for link in spec.links}
    roots = []
    for link in spec.links:
        if link.parent is None:
            roots.append(link.name)
        else:
            children[link.parent].append(link.name)
    if len(roots) != 1:
        raise ValueError(f"{spec.name}: expected one clamped root, found {roots}")
    ports_on: dict[str, list[PortSpec]] = {link.name: [] for link in spec.links}
    for port in spec.ports:
        ports_on[port.link].append(port)

    def inner(name: str) -> str:
        link = by_name[name]
        chunks = [
            f'<joint name="j_{name}" type="hinge" axis="0 0 1" stiffness="{link.stiffness}" '
            f'damping="{DAMPING}" limited="true" range="{JOINT_RANGE[0]} {JOINT_RANGE[1]}" springref="0"/>',
            f'<geom type="capsule" fromto="0 0 0 {link.length:.6f} 0 0" size="0.0045" mass="{LINK_MASS}" '
            f'rgba="0.38 0.52 0.64 1" contype="0" conaffinity="0"/>',
        ]
        for port in ports_on[name]:
            chunks.append(
                f'<site name="{port.name}" pos="{port.fraction * link.length:.6f} 0 0" size="0.007" rgba="{_rgba(port.kind)}"/>'
            )
            if contact_port is not None and port.name == contact_port:
                chunks.append(
                    f'<geom name="bumper" type="sphere" pos="{port.fraction * link.length:.6f} 0 0" '
                    f'size="{BUMPER_RADIUS}" mass="0.0001" rgba="0.84 0.24 0.20 1" contype="1" conaffinity="1" '
                    f'condim="1" friction="0 0 0"/>'
                )
        for child in children[name]:
            relative = by_name[child].angle_deg - link.angle_deg
            chunks.append(
                f'<body name="b_{child}" pos="{link.length:.6f} 0 0" euler="0 0 {relative:.6f}">{inner(child)}</body>'
            )
        return "".join(chunks)

    root = roots[0]
    contact_xml = ""
    if contact_port is not None:
        port = next(port for port in spec.ports if port.name == contact_port)
        point = port_position(spec, port, pose)
        half = 0.008
        center_y = float(point[1] + BUMPER_RADIUS + contact_gap + half)
        contact_xml = (
            f'<geom name="stop" type="box" pos="{float(point[0]):.6f} {center_y:.6f} 0" '
            f'size="0.012 {half} 0.012" rgba="0.25 0.25 0.28 1" contype="1" conaffinity="1" '
            f'condim="1" friction="0 0 0"/>'
        )
    return f"""<mujoco model="{spec.name}">
  <option timestep="{timestep}" gravity="0 0 0" integrator="implicitfast" solver="Newton"
          tolerance="{tolerance}" iterations="{SOLVER_ITERATIONS}">
    <flag warmstart="disable"/>
  </option>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.94 0.94 0.95" rgb2="0.94 0.94 0.95" width="16" height="16"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.2 0.9" dir="0 0.15 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="oblique" pos="0.18 -0.46 0.50" xyaxes="1 0 0 0 0.74 0.67" fovy="42"/>
    <geom name="floor" type="plane" pos="0.18 0.08 -0.02" size="0.55 0.45 0.01" rgba="0.94 0.94 0.95 1" contype="0" conaffinity="0"/>
    <geom name="clamp" type="box" pos="-0.016 0 0" size="0.014 0.022 0.01" rgba="0.16 0.16 0.16 1" contype="0" conaffinity="0"/>
    <site name="goal" pos="{GOAL_POINT[0]} {GOAL_POINT[1]} 0" size="0.008" rgba="0.15 0.35 0.85 1"/>
    {contact_xml}
    <body name="b_{root}" pos="0 0 0" euler="0 0 {by_name[root].angle_deg:.6f}">{inner(root)}</body>
  </worldbody>
</mujoco>
"""


class Fixture:
    def __init__(
        self,
        spec: FixtureSpec,
        timestep: float = TIMESTEP,
        tolerance: float = SOLVER_TOLERANCE,
        settle_time: float = SETTLE_TIME,
        contact_port: str | None = None,
        contact_gap: float = CONTACT_GAP_M,
    ):
        self.spec = spec
        self.settle_time = settle_time
        self.xml = build_xml(spec, timestep, tolerance, contact_port, contact_gap)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.site_ids = {}
        for port in spec.ports:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, port.name)
            if site_id < 0:
                raise RuntimeError(f"site {port.name} missing from the compiled model")
            self.site_ids[port.name] = site_id
        self._zero_positions = None
        self._zero_info = None

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def positions(self) -> dict[str, np.ndarray]:
        return {name: self.data.site_xpos[site_id, :2].copy() for name, site_id in self.site_ids.items()}

    def settle(self, port: str | None, force_xy: np.ndarray, require_settled: bool = True) -> tuple[dict, dict]:
        self.reset()
        steps = int(round(self.settle_time / float(self.model.opt.timestep)))
        if steps < 1:
            raise ValueError("settle time is shorter than one step")
        force = np.zeros(3)
        force[:2] = np.asarray(force_xy, dtype=float)
        site_id = None if port is None else self.site_ids[port]
        for _ in range(steps):
            self.data.xfrc_applied[:] = 0.0
            if site_id is not None and np.any(force):
                body = int(self.model.site_bodyid[site_id])
                relative = self.data.site_xpos[site_id] - self.data.xipos[body]
                self.data.xfrc_applied[body, :3] = force
                self.data.xfrc_applied[body, 3:] = np.cross(relative, force)
            mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise RuntimeError(f"{self.spec.name}: non-finite state after settling")
        info = {
            "qvel_max": float(np.max(np.abs(self.data.qvel))),
            "qpos_max": float(np.max(np.abs(self.data.qpos))),
            "ncon": int(self.data.ncon),
            "steps": steps,
            "penetration_m": 0.0,
        }
        if require_settled and info["qvel_max"] > SETTLED_QVEL:
            raise RuntimeError(f"{self.spec.name} did not settle: max |qvel|={info['qvel_max']:.3e} after {self.settle_time} s")
        limit = 0.98 * JOINT_RANGE[1]
        info["hit_joint_limit"] = bool(info["qpos_max"] >= limit)
        if self.data.ncon:
            info["penetration_m"] = float(max(0.0, *(-self.data.contact.dist[: self.data.ncon])))
        return self.positions(), info

    def zero_positions(self, require_settled: bool = True) -> dict[str, np.ndarray]:
        if self._zero_positions is None:
            self._zero_positions, self._zero_info = self.settle(None, np.zeros(2), require_settled)
        return self._zero_positions

    def displacement(self, port: str, force_xy: np.ndarray, require_settled: bool = True) -> tuple[dict, dict]:
        baseline = self.zero_positions(require_settled)
        loaded, info = self.settle(port, force_xy, require_settled)
        delta = {name: loaded[name] - baseline[name] for name in loaded}
        if not all(np.all(np.isfinite(value)) for value in delta.values()):
            raise RuntimeError(f"{self.spec.name}: non-finite displacement")
        info["zero_qvel_max"] = self._zero_info["qvel_max"]
        return delta, info

    def oracle_compliance(self) -> np.ndarray:
        """Privileged linearization: stiffness is read from the compiled MuJoCo model."""

        geometry = visible_geometry(self.spec)
        stiffness = []
        for name in geometry.link_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"j_{name}")
            if joint_id < 0:
                raise RuntimeError(f"joint j_{name} missing")
            stiffness.append(self.model.jnt_stiffness[joint_id])
        return compliance_from_stiffness(
            geometry.site_positions, geometry.ancestors, geometry.anchors, np.asarray(stiffness, dtype=float)
        )

    def render(self) -> np.ndarray:
        self.reset()
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        try:
            renderer.update_scene(self.data, camera="oblique")
            return renderer.render().copy()
        finally:
            renderer.close()


def measure_catalog(fixture: Fixture, magnitudes, directions, require_settled: bool = True) -> dict:
    catalog = {}
    for port in fixture.spec.ports:
        if port.kind != "action" and port.name not in ("align", "protect"):
            continue
        for direction in directions:
            for magnitude in magnitudes:
                catalog[(port.name, direction, float(magnitude))] = fixture.displacement(
                    port.name, float(magnitude) * DIRECTIONS[direction], require_settled
                )
    return catalog


def secant_from_catalog(catalog: dict, spec: FixtureSpec, magnitude: float) -> np.ndarray:
    names = [port.name for port in spec.ports]
    compliance = np.zeros((len(names), len(names), 2, 2))
    for j, source in enumerate(names):
        for axis, direction in enumerate(BASIS_DIRECTIONS):
            displacement, _info = catalog[(source, direction, float(magnitude))]
            for i, dest in enumerate(names):
                compliance[i, j, :, axis] = displacement[dest] / float(magnitude)
    return compliance


def flat_compliance(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    flat = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            flat[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = blocks[i, j]
    return flat


def reciprocity_records(blocks: np.ndarray, names: list[str], force: float) -> list[dict]:
    rows = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            mismatch = blocks[i, j] - blocks[j, i].T
            disp_err = float(force * np.linalg.norm(mismatch))
            scale = float(force * max(np.linalg.norm(blocks[i, j]), np.linalg.norm(blocks[j, i])))
            rows.append(
                {
                    "port_i": left,
                    "port_j": names[j],
                    "disp_err_m": disp_err,
                    "disp_scale_m": scale,
                    "tolerance_m": displacement_tolerance(scale),
                    "agrees": disp_err <= displacement_tolerance(scale),
                }
            )
    return rows


def outcomes_from_catalog(catalog: dict, spec: FixtureSpec, goal: np.ndarray) -> list[dict]:
    rows = []
    for port in spec.ports:
        if port.kind != "action":
            continue
        for direction in DIRECTIONS:
            for magnitude in FORCE_MAGNITUDES:
                displacement, info = catalog[(port.name, direction, float(magnitude))]
                score, parts = utility(displacement["align"], displacement["protect"], magnitude, goal)
                rows.append(
                    {
                        "port": port.name,
                        "direction": direction,
                        "magnitude": float(magnitude),
                        "utility": score,
                        "qvel_max": info["qvel_max"],
                        "ncon": info["ncon"],
                        "hit_joint_limit": info["hit_joint_limit"],
                        **parts,
                    }
                )
    return rows


def coincident_ports(spec: FixtureSpec) -> set[str]:
    pose = rest_pose(spec)
    points = {port.name: port_position(spec, port, pose) for port in spec.ports}
    coincident = set()
    names = list(points)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if np.linalg.norm(points[left] - points[right]) < 1e-12:
                coincident.add(left)
                coincident.add(right)
    return coincident


def reciprocity_predictions(catalog: dict, spec: FixtureSpec, goal: np.ndarray) -> dict:
    """Predict scoring-port motion of every action from loads at the scoring ports only."""

    names = [port.name for port in spec.ports]
    align_i = names.index("align")
    protect_i = names.index("protect")
    blocks = secant_from_catalog(catalog, spec, PROBE_FORCE_N)
    # Columns of the scoring ports were measured by loading those ports.
    # Symmetry copies them into the rows the task needs. No action-port load is used.
    predicted_rows = []
    action_names = [port.name for port in spec.ports if port.kind == "action"]
    shared = coincident_ports(spec)
    for action in action_names:
        j = names.index(action)
        for direction, vector in DIRECTIONS.items():
            for magnitude in FORCE_MAGNITUDES:
                # C[score, action] = C[action, score].T, estimated at the probe force.
                d_align = (magnitude / PROBE_FORCE_N) * (blocks[j, align_i].T @ (PROBE_FORCE_N * vector))
                d_protect = (magnitude / PROBE_FORCE_N) * (blocks[j, protect_i].T @ (PROBE_FORCE_N * vector))
                # The algebra above simplifies to magnitude * blocks[j, score].T @ vector.
                score, parts = utility(d_align, d_protect, magnitude, goal)
                measured, _info = catalog[(action, direction, float(magnitude))]
                m_score, m_parts = utility(measured["align"], measured["protect"], magnitude, goal)
                predicted_rows.append(
                    {
                        "port": action,
                        "direction": direction,
                        "magnitude": float(magnitude),
                        "utility": score,
                        "measured_utility": m_score,
                        "align_err_m": float(np.linalg.norm(d_align - measured["align"])),
                        "protect_err_m": float(np.linalg.norm(d_protect - measured["protect"])),
                        "unexecuted_point": action not in shared and action not in ("align", "protect"),
                        **parts,
                    }
                )
    choice = rank_actions(predicted_rows)["best"]
    return {"rows": predicted_rows, "choice": {key: choice[key] for key in ("port", "direction", "magnitude", "utility")}}


def unidentified_entries(spec: FixtureSpec) -> dict:
    """Blocks a pair of scoring-port probes cannot determine.

    Loading the scoring ports and invoking C = Cᵀ identifies every block that
    touches a scoring port. Blocks between two action ports that are not
    themselves scoring ports stay unknown. Finite-load responses are not
    identified by the linear relation either.
    """

    pose = rest_pose(spec)
    point = {port.name: tuple(np.round(port_position(spec, port, pose), 9)) for port in spec.ports}
    score_points = {point["align"], point["protect"]}
    action_only = []
    for port in spec.ports:
        if port.kind == "action" and point[port.name] not in score_points:
            action_only.append(port.name)
    unknown = []
    for i, left in enumerate(action_only):
        for right in action_only[i + 1 :]:
            unknown.append((left, right))
    return {
        "identified": "every 2×2 block with at least one index at align or protect, including the align–protect block",
        "not_identified_action_pairs": unknown,
        "n_unknown_pairs": len(unknown),
        "note": "A nonlinear response, including contact and the large-load secant, is not identified by these probes.",
    }


def measured_utility_of(rows: list[dict], port: str, direction: str, magnitude: float) -> float:
    for row in rows:
        if row["port"] == port and row["direction"] == direction and abs(row["magnitude"] - magnitude) < 1e-12:
            return float(row["utility"])
    raise KeyError((port, direction, magnitude))


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _action_key(row: dict) -> str:
    return f"{row['port']} {row['direction']} {row['magnitude']:.2f}N"


def run_sensitivity(spec: FixtureSpec) -> dict:
    """One-factor sweeps about the nominal solver settings, on one reciprocal pair."""

    port_a, port_b = "a7", "align"
    records = []

    def pair_gap(fixture: Fixture) -> dict:
        force = PROBE_FORCE_N * DIRECTIONS["+y"]
        disp_ab, info_ab = fixture.displacement(port_a, force, require_settled=False)
        disp_ba, info_ba = fixture.displacement(port_b, force, require_settled=False)
        # Equal +y forces: Betti equates the y-displacements, not the full vectors.
        gap = float(disp_ab[port_b][1] - disp_ba[port_a][1])
        return {
            "y_at_align_m": float(disp_ab[port_b][1]),
            "y_gap_m": gap,
            "qvel_max": max(info_ab["qvel_max"], info_ba["qvel_max"]),
        }

    configs = [("nominal", TIMESTEP, SETTLE_TIME, SOLVER_TOLERANCE)]
    configs += [(f"dt={dt}", dt, SETTLE_TIME, SOLVER_TOLERANCE) for dt in (0.0005, 0.002)]
    configs += [(f"settle={t}", TIMESTEP, t, SOLVER_TOLERANCE) for t in (0.4, 2.4)]
    configs += [(f"tol={tol}", TIMESTEP, SETTLE_TIME, tol) for tol in (1e-6, 1e-8)]
    for label, dt, settle, tol in configs:
        fixture = Fixture(spec, timestep=dt, tolerance=tol, settle_time=settle)
        row = pair_gap(fixture)
        row.update({"label": label, "timestep": dt, "settle_s": settle, "tolerance": tol})
        records.append(row)
    settled = [row for row in records if row["qvel_max"] <= SETTLED_QVEL]
    if not settled:
        raise RuntimeError("no sensitivity setting settled")
    responses = np.array([row["y_at_align_m"] for row in settled])
    gaps = np.array([row["y_gap_m"] for row in settled])
    nominal = next(row for row in records if row["label"] == "nominal")
    return {
        "records": records,
        "settled_labels": [row["label"] for row in settled],
        "response_span_m": float(responses.max() - responses.min()),
        "gap_span_m": float(np.max(np.abs(gaps))),
        "nominal_y_m": nominal["y_at_align_m"],
    }


def run_breakdown(spec: FixtureSpec) -> dict:
    nominal = Fixture(spec)
    large = Fixture(spec, settle_time=2.0)
    force_name = "a7"
    small, small_info = nominal.displacement(force_name, PROBE_FORCE_N * DIRECTIONS["+y"])
    double, _ = nominal.displacement(force_name, 2 * PROBE_FORCE_N * DIRECTIONS["+y"])
    opposite, _ = nominal.displacement(force_name, -PROBE_FORCE_N * DIRECTIONS["+y"])
    heavy, heavy_info = large.displacement(force_name, BREAKDOWN_FORCE_N * DIRECTIONS["+y"])
    linear_double_err = float(np.linalg.norm(double["align"] - 2.0 * small["align"]))
    sign_err = float(np.linalg.norm(opposite["align"] + small["align"]))
    scale = BREAKDOWN_FORCE_N / PROBE_FORCE_N
    heavy_err = float(np.linalg.norm(heavy["align"] - scale * small["align"]))
    # Axial shortening is even in the force, so it survives d(f)+d(−f). The transverse
    # component is the odd part that the linear model is responsible for.

    contact = Fixture(spec, contact_port="protect", settle_time=2.0)
    soft, soft_info = contact.displacement("a7", PROBE_FORCE_N * DIRECTIONS["+y"], require_settled=False)
    # Reciprocal partner: load the protected point, read a7.
    soft_back, soft_back_info = contact.displacement("protect", PROBE_FORCE_N * DIRECTIONS["+y"], require_settled=False)
    hard, hard_info = contact.displacement("a7", CONTACT_FORCE_N * DIRECTIONS["+y"], require_settled=False)
    hard_back, hard_back_info = contact.displacement("protect", CONTACT_FORCE_N * DIRECTIONS["+y"], require_settled=False)
    # Same force direction at both ports: compare the protected-port y motion
    # under an a7 load with the a7 y motion under a protect load.
    soft_gap = float(soft["protect"][1] - soft_back["a7"][1])
    hard_gap = float(hard["protect"][1] - hard_back["a7"][1])
    return {
        "probe_align_m": small["align"].tolist(),
        "double_linearity_err_m": linear_double_err,
        "double_transverse_err_m": float(abs(double["align"][1] - 2.0 * small["align"][1])),
        "sign_reversal_err_m": sign_err,
        "sign_reversal_transverse_err_m": float(abs(opposite["align"][1] + small["align"][1])),
        "sign_reversal_axial_err_m": float(abs(opposite["align"][0] + small["align"][0])),
        "large_force_n": BREAKDOWN_FORCE_N,
        "large_linearity_err_m": heavy_err,
        "large_qvel_max": heavy_info["qvel_max"],
        "large_qpos_max": heavy_info["qpos_max"],
        "large_hit_limit": heavy_info["hit_joint_limit"],
        "small_qvel_max": small_info["qvel_max"],
        "contact_gap_m": CONTACT_GAP_M,
        "contact_soft_ncon": soft_info["ncon"],
        "contact_soft_back_ncon": soft_back_info["ncon"],
        "contact_hard_ncon": hard_info["ncon"],
        "contact_hard_back_ncon": hard_back_info["ncon"],
        "contact_soft_qvel": max(soft_info["qvel_max"], soft_back_info["qvel_max"]),
        "contact_hard_qvel": max(hard_info["qvel_max"], hard_back_info["qvel_max"]),
        "contact_soft_gap_m": soft_gap,
        "contact_hard_gap_m": hard_gap,
        "contact_soft_protect_m": soft["protect"].tolist(),
        "contact_hard_protect_m": hard["protect"].tolist(),
        "contact_hard_penetration_m": max(hard_info["penetration_m"], hard_back_info["penetration_m"]),
    }


def repeats(spec: FixtureSpec, n: int = 5) -> dict:
    fixture = Fixture(spec)
    samples = []
    for _ in range(n):
        displacement, info = fixture.displacement("a7", PROBE_FORCE_N * DIRECTIONS["+y"])
        samples.append(displacement["align"].copy())
        qvel = info["qvel_max"]
    stack = np.vstack(samples)
    spread = np.max(stack, axis=0) - np.min(stack, axis=0)
    return {"n": n, "spread_m": spread.tolist(), "max_spread_m": float(np.max(spread)), "qvel_max": qvel, "mean_m": stack.mean(axis=0).tolist()}


def evaluate_geometry(spec: FixtureSpec) -> dict:
    print(f"  simulating {spec.name}")
    fixture = Fixture(spec)
    rest = fixture.zero_positions()
    kinematic = {port.name: port_position(spec, port) for port in spec.ports}
    pose_err = {name: float(np.linalg.norm(rest[name] - kinematic[name])) for name in rest}
    if max(pose_err.values()) > 1e-8:
        raise RuntimeError(f"{spec.name}: MuJoCo rest pose disagrees with the kinematic drawing")
    goal = goal_direction(kinematic["align"])
    catalog = measure_catalog(fixture, FORCE_MAGNITUDES, tuple(DIRECTIONS))
    # Scoring ports are not in the action list, but reciprocity needs their loads
    # at the probe force in the two basis directions. measure_catalog already
    # includes every port, and only action ports are ranked below.
    blocks = secant_from_catalog(catalog, spec, PROBE_FORCE_N)
    flat = flat_compliance(blocks)
    linear = privileged_compliance(spec)
    linear_err = float(np.linalg.norm(flat - linear) / np.linalg.norm(linear))
    records = reciprocity_records(blocks, [port.name for port in spec.ports], PROBE_FORCE_N)
    outcomes = outcomes_from_catalog(catalog, spec, goal)
    ranking = rank_actions(outcomes)
    deployable = deployable_compliance(visible_geometry(spec))
    oracle = fixture.oracle_compliance()
    deployable_choice = rank_actions(predict_action_utilities(deployable, spec, goal))["best"]
    oracle_choice = rank_actions(predict_action_utilities(oracle, spec, goal))["best"]
    doubled = replace_public_prediction(spec, goal, factor=2.0)
    probes = reciprocity_predictions(catalog, spec, goal)
    probe_choice = probes["choice"]
    best_u = ranking["best"]["utility"]

    def regret(port, direction, magnitude) -> float:
        return float(best_u - measured_utility_of(outcomes, port, direction, magnitude))

    direct_utilities = [measured_utility_of(outcomes, *action) for action in DIRECT_BUDGET_ACTIONS]
    direct_best = max(range(len(DIRECT_BUDGET_ACTIONS)), key=lambda i: direct_utilities[i])
    unexecuted = [row for row in probes["rows"] if row["unexecuted_point"]]
    return {
        "name": spec.name,
        "description": spec.description,
        "goal_direction": goal.tolist(),
        "align_rest_m": kinematic["align"].tolist(),
        "protect_rest_m": kinematic["protect"].tolist(),
        "pose_err_m": max(pose_err.values()),
        "zero_qvel_max": fixture._zero_info["qvel_max"],
        "linearization_relative_error": linear_err,
        "reciprocity": records,
        "reciprocity_max_err_m": max(row["disp_err_m"] for row in records),
        "reciprocity_worst": max(records, key=lambda row: row["disp_err_m"]),
        "n_pairs_failing": sum(not row["agrees"] for row in records),
        "outcomes": [
            {key: row[key] for key in ("port", "direction", "magnitude", "utility", "align", "protect", "progress_m", "protect_m", "qvel_max", "ncon")}
            for row in outcomes
        ],
        "ranking": {
            "best": {key: ranking["best"][key] for key in ("port", "direction", "magnitude", "utility", "progress_m", "protect_m")},
            "second": None
            if ranking["second"] is None
            else {key: ranking["second"][key] for key in ("port", "direction", "magnitude", "utility")},
            "tie": [_action_key(row) for row in ranking["tie"]],
            "gap_to_next": ranking["gap_to_next"],
            "gap_to_distinct": ranking["gap_to_distinct"],
            "top": [
                {"action": _action_key(row), "utility": row["utility"]}
                for row in ranking["ordered"][:5]
            ],
        },
        "blind_utility": measured_utility_of(outcomes, *FIXED_BLIND_ACTION),
        "blind_regret": regret(*FIXED_BLIND_ACTION),
        "direct_budget_choice": _action_key(dict(zip(("port", "direction", "magnitude"), DIRECT_BUDGET_ACTIONS[direct_best]))),
        "direct_budget_utility": direct_utilities[direct_best],
        "direct_budget_regret": float(best_u - direct_utilities[direct_best]),
        "deployable_choice": _action_key(deployable_choice),
        "deployable_predicted_utility": deployable_choice["utility"],
        "deployable_measured_utility": measured_utility_of(outcomes, deployable_choice["port"], deployable_choice["direction"], deployable_choice["magnitude"]),
        "deployable_regret": regret(deployable_choice["port"], deployable_choice["direction"], deployable_choice["magnitude"]),
        "oracle_choice": _action_key(oracle_choice),
        "oracle_regret": regret(oracle_choice["port"], oracle_choice["direction"], oracle_choice["magnitude"]),
        "oracle_matches_deployable_matrix": bool(np.allclose(oracle, deployable, atol=1e-12)),
        "misspecified_k_x2_choice": doubled["choice"],
        "misspecified_k_x2_regret": regret(*doubled["key"]),
        "probe_choice": _action_key(probe_choice),
        "probe_predicted_utility": probe_choice["utility"],
        "probe_measured_utility": measured_utility_of(outcomes, probe_choice["port"], probe_choice["direction"], probe_choice["magnitude"]),
        "probe_regret": regret(probe_choice["port"], probe_choice["direction"], probe_choice["magnitude"]),
        "probe_unexecuted_median_err_m": float(np.median([max(row["align_err_m"], row["protect_err_m"]) for row in unexecuted])),
        "probe_unexecuted_max_err_m": float(np.max([max(row["align_err_m"], row["protect_err_m"]) for row in unexecuted])),
        "probe_same_force_max_err_m": float(
            np.max([max(row["align_err_m"], row["protect_err_m"]) for row in unexecuted if abs(row["magnitude"] - PROBE_FORCE_N) < 1e-12])
        ),
        "unknown_pairs": unidentified_entries(spec),
        "compliance": flat,
        "image": fixture.render(),
        "pose": kinematic,
    }


def replace_public_prediction(spec: FixtureSpec, goal: np.ndarray, factor: float) -> dict:
    """Sensitivity only: the same visible geometry with a wrong uniform spring rate."""

    geometry = visible_geometry(spec)
    wrong = VisibleGeometry(
        geometry.link_names,
        geometry.anchors,
        geometry.site_names,
        geometry.site_positions,
        geometry.ancestors,
        public_stiffness=PUBLIC_STIFFNESS * factor,
    )
    choice = rank_actions(predict_action_utilities(deployable_compliance(wrong), spec, goal))["best"]
    return {"choice": _action_key(choice), "key": (choice["port"], choice["direction"], choice["magnitude"])}


def decide(summary: dict) -> dict:
    credible = (
        summary["numerical_uncertainty_m"] < DECISION_FLOOR_M
        and summary["straight_zero_qvel"] < SETTLED_QVEL
        and summary["straight_linearization_relative_error"] < RELATIVE_BLOCK_TOL
    )
    reciprocity_ok = all(geo["n_pairs_failing"] == 0 for geo in summary["geometries"])
    unique_optima = {(geo["ranking"]["best"]["port"], geo["ranking"]["best"]["direction"], geo["ranking"]["best"]["magnitude"]) for geo in summary["geometries"]}
    consequential = any(geo["ranking"]["gap_to_distinct"] >= CLEAR_UTILITY_GAP for geo in summary["geometries"])
    blind_fails = any(geo["blind_regret"] >= CLEAR_UTILITY_GAP for geo in summary["geometries"])
    mechanics_regret = max(geo["deployable_regret"] for geo in summary["geometries"])
    mechanics_solves = mechanics_regret < 0.02
    probe_validated = all(geo["probe_same_force_max_err_m"] <= DECISION_FLOOR_M * 4 for geo in summary["geometries"])
    # Four times the 25 µm floor is 100 µm, the 2% end of the acceptance rule at the alignment target.
    probe_useful = all(geo["probe_regret"] < CLEAR_UTILITY_GAP for geo in summary["geometries"])
    equal_cost_reason = (not mechanics_solves) and any(
        geo["probe_regret"] + CLEAR_UTILITY_GAP < geo["direct_budget_regret"] for geo in summary["geometries"]
    )
    diversity = len(unique_optima) > 1 and consequential and any(
        geo["name"] != "straight" and geo["blind_regret"] >= CLEAR_UTILITY_GAP for geo in summary["geometries"]
    )
    # The straight-beam winner transferred as a fixed action is the stronger
    # "too easy" check: if every other geometry still prefers that same action,
    # diversity fails via unique_optima.
    checks = {
        "simulation_credible": credible,
        "reciprocity_within_tolerance": reciprocity_ok,
        "optimal_actions_differ": len(unique_optima) > 1 and consequential,
        "image_blind_fails": blind_fails,
        "deployable_mechanics_does_not_solve": not mechanics_solves,
        "goal_probes_validated": probe_validated and probe_useful,
        "equal_cost_reason_to_compare_interventions": equal_cost_reason,
    }
    # GO only if every listed prerequisite holds. Diversity uses the consequential flag.
    checks["optimal_actions_differ"] = bool(diversity or (len(unique_optima) > 1 and consequential))
    go = all(checks.values())
    if go:
        verdict = "GO"
    elif not credible or not reciprocity_ok:
        verdict = "NO-GO"
    elif mechanics_solves or not checks["optimal_actions_differ"]:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "checks": checks, "n_distinct_optima": len(unique_optima), "max_mechanics_regret": mechanics_regret}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(output: Path, geometries: list[dict], breakdown: dict, sensitivity: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, geo in zip(axes.ravel(), geometries):
        ax.imshow(geo["image"])
        ax.set_title(geo["name"])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "renders.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    spec_by_name = {spec.name: spec for spec in all_specs()}
    for ax, geo in zip(axes.ravel(), geometries):
        spec = spec_by_name[geo["name"]]
        pose = rest_pose(spec)
        for link in spec.links:
            start, end = pose[link.name]["start"], pose[link.name]["end"]
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#4d6d82", lw=2, zorder=1)
        ax.scatter([0], [0], marker="s", c="0.2", s=40, zorder=2)
        for port in spec.ports:
            point = port_position(spec, port, pose)
            color = {"action": "#e0b000", "align": "#1f9d4d", "protect": "#d1453b"}[port.kind]
            ax.scatter(point[0], point[1], c=color, s=36, zorder=3)
            if port.kind == "action":
                index = int(port.name[1:])
                offset = 0.016 if index % 2 == 0 else -0.020
            else:
                offset = 0.020
            valign = "bottom" if offset > 0 else "top"
            ax.text(point[0], point[1] + offset, port.name, ha="center", va=valign, fontsize=7)
        ax.scatter(*GOAL_POINT, marker="*", c="#2458c9", s=90, zorder=4, label="goal")
        ax.set_title(geo["name"])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(output / "schematics.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cloud_x = []
    cloud_y = []
    for geo in geometries:
        matrix = geo["compliance"]
        scale = PROBE_FORCE_N
        xs, ys = [], []
        n = matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                xs.append(scale * matrix[i, j])
                ys.append(scale * matrix[j, i])
        cloud_x.extend(xs)
        cloud_y.extend(ys)
        ax.scatter(xs, ys, s=12, alpha=0.75, label=geo["name"])
    limit = max(abs(np.array(cloud_x + cloud_y))) if cloud_x else 1.0
    ax.plot([-limit, limit], [-limit, limit], color="0.4", lw=0.8)
    ax.set_xlabel("measured response component (m)")
    ax.set_ylabel("reciprocal component (m)")
    ax.set_title(f"Secant reciprocity at {PROBE_FORCE_N:.2f} N")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "reciprocity_scatter.png", dpi=140)
    plt.close(fig)

    labels = [geo["name"] for geo in geometries]
    series = {
        "best measured": [geo["ranking"]["best"]["utility"] for geo in geometries],
        "image-blind a0 +y": [geo["blind_utility"] for geo in geometries],
        "4-action budget": [geo["direct_budget_utility"] for geo in geometries],
        "deployable mechanics": [geo["deployable_measured_utility"] for geo in geometries],
        "score-port probes": [geo["probe_measured_utility"] for geo in geometries],
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(labels))
    width = 0.15
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - 2) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("measured utility")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("First-action utility")
    fig.tight_layout()
    fig.savefig(output / "utility_bars.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = ["sign flip", "doubled force", f"{BREAKDOWN_FORCE_N:.0f} N load", "contact\n0.05 N", "contact\n0.15 N"]
    values_um = np.array(
        [
            breakdown["sign_reversal_err_m"],
            breakdown["double_linearity_err_m"],
            breakdown["large_linearity_err_m"],
            abs(breakdown["contact_soft_gap_m"]),
            abs(breakdown["contact_hard_gap_m"]),
        ]
    ) * 1e6
    ax.bar(labels, np.maximum(values_um, 0.1), color=["#4d6d82", "#4d6d82", "#a33b32", "#4d6d82", "#a33b32"])
    ax.set_yscale("log")
    ax.axhline(DECISION_FLOOR_M * 1e6, color="0.3", ls="--", lw=0.8, label="25 µm floor")
    ax.set_ylabel("absolute discrepancy (µm, log)")
    ax.set_title("Linearity and the contact stop")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "failure_cases.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [row["label"] for row in sensitivity["records"]]
    ax.plot(labels, [row["y_at_align_m"] * 1e3 for row in sensitivity["records"]], marker="o")
    ax.set_ylabel("align Δy under 0.05 N at a7 (mm)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Solver and settle sensitivity")
    fig.tight_layout()
    fig.savefig(output / "sensitivity.png", dpi=140)
    plt.close(fig)


def run_study(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    print("positive-control repeats and sensitivity")
    repeat = repeats(straight_spec())
    sensitivity = run_sensitivity(straight_spec())
    print("breakdown and contact")
    breakdown = run_breakdown(straight_spec())
    geometries = []
    for spec in all_specs():
        result = evaluate_geometry(spec)
        (output / "models" / f"{spec.name}.xml").write_text(Fixture(spec).xml)
        geometries.append(result)
    numerical = max(repeat["max_spread_m"], sensitivity["response_span_m"], sensitivity["gap_span_m"])
    summary = {
        "assumptions": {
            "model": "planar revolute-spring assembly, lumped stiffness, not a continuum bracket",
            "gravity": GRAVITY,
            "public_stiffness_nm_per_rad": PUBLIC_STIFFNESS,
            "damping_nm_s_per_rad": DAMPING,
            "timestep_s": TIMESTEP,
            "settle_s": SETTLE_TIME,
            "solver_tolerance": SOLVER_TOLERANCE,
            "goal_m": GOAL_POINT.tolist(),
            "align_target_m": ALIGN_TARGET_M,
            "protect_deadband_m": PROTECT_DEADBAND_M,
            "protect_scale_m": PROTECT_SCALE_M,
            "w_protect": W_PROTECT,
            "w_force": W_FORCE,
            "force_reference_n": FORCE_REFERENCE_N,
            "force_magnitudes_n": list(FORCE_MAGNITUDES),
            "probe_force_n": PROBE_FORCE_N,
            "decision_floor_m": DECISION_FLOOR_M,
            "relative_block_tolerance": RELATIVE_BLOCK_TOL,
            "clear_utility_gap": CLEAR_UTILITY_GAP,
            "fixed_blind_action": list(FIXED_BLIND_ACTION),
            "direct_budget_actions": [list(item) for item in DIRECT_BUDGET_ACTIONS],
        },
        "repeats": repeat,
        "sensitivity": sensitivity,
        "breakdown": breakdown,
        "numerical_uncertainty_m": numerical,
        "straight_zero_qvel": geometries[0]["zero_qvel_max"],
        "straight_linearization_relative_error": geometries[0]["linearization_relative_error"],
        "geometries": [],
    }
    for geo in geometries:
        stored = {key: value for key, value in geo.items() if key not in ("image", "pose", "compliance", "reciprocity", "outcomes")}
        stored["reciprocity_worst"] = geo["reciprocity_worst"]
        summary["geometries"].append(stored)
    summary["decision"] = decide(summary)
    save_figures(output, geometries, breakdown, sensitivity)
    _write_csv(
        output / "reciprocity.csv",
        [
            {"geometry": geo["name"], **row}
            for geo in geometries
            for row in geo["reciprocity"]
        ],
    )
    _write_csv(
        output / "actions.csv",
        [{"geometry": geo["name"], **row} for geo in geometries for row in geo["outcomes"]],
    )
    _write_csv(output / "sensitivity.csv", sensitivity["records"])
    # Compliance matrices are a few kilobytes; keep them with the local run.
    np.savez_compressed(
        output / "compliance.npz",
        **{geo["name"]: geo["compliance"] for geo in geometries},
    )
    payload = json_ready(summary)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(json_ready(summary["decision"]), indent=2))
    return summary


def copy_report_bundle(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary.json",
        "reciprocity.csv",
        "actions.csv",
        "sensitivity.csv",
        "renders.png",
        "schematics.png",
        "reciprocity_scatter.png",
        "utility_bars.png",
        "failure_cases.png",
        "sensitivity.png",
    ):
        shutil.copy2(source / name, dest / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("results/phino_reciprocity_feasibility_v1"))
    parser.add_argument("--docs-output", type=Path, default=Path("docs/results/phino_reciprocity_feasibility_v1"))
    args = parser.parse_args()
    summary = run_study(args.output)
    copy_report_bundle(args.output, args.docs_output)
    print(f"wrote {args.output} and {args.docs_output}")
    print("verdict", summary["decision"]["verdict"])


if __name__ == "__main__":
    main()
