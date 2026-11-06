#!/usr/bin/env python3
"""Do action-conditioned images distinguish SIMPACT mass/friction configurations?

From the SIMPACT repository root, with the project environment:
    MUJOCO_GL=egl python scripts/phino_visual_observability_pilot.py \
        --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
        --phino-root /nbu/people/ashah264/phino \
        --output-dir results/phino_visual_observability_v1

No physics is regenerated. Every image is a kinematic replay of a SAVED valid
trajectory in its isolated scene. The settled start comes from the generator's
own deterministic settle (``build_model``); a one-config re-simulation per
geometry checks that this start reproduces the saved trace exactly.

The grid's settle is friction-dependent for some geometries (the object slides
or rocks while settling), so simulated initial states are not always matched.
Two render modes are therefore produced:
  faithful  object at (its own settled start + saved relative position, saved quaternion)
  aligned   the same saved relative motion re-anchored to ONE reference start pose,
            so every configuration has an identical initial image. This removes the
            static settle offset only; any dynamic effect of a different start remains.
Camera, lighting, and action are identical across configurations of a geometry; the
camera is framed once on the union of that geometry's trajectories.

Outputs: renders (initial and final frame per configuration), per-frame
image-space object centroids, state metrics, pairwise separability tables under
several observation models, comparison figures, and summary.json.
Synthetic, noiseless renders give an UPPER BOUND on visual distinguishability.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import export_phino_dataset as export
import phino_simpact_pilot_v2 as pilot

MODES = ("faithful", "aligned")
RENDER_ACTION = "push_x"
SECOND_ACTION = "lateral_flick"
IMG_SIZE = 224
CAMERA_AZIMUTH = 90.0      # looking along +y: push_x moves left -> right in the image
CAMERA_ELEVATION = -60.0
REST_SPEED_M_S = 0.01
PIXEL_THRESHOLDS = (1.0, 3.0)
STATE_THRESHOLDS_M = (0.001, 0.005)
REPLAY_TOLERANCE = 1e-5


@dataclass
class Config:
    mass: float
    friction: float
    traces: dict[str, np.ndarray]   # action -> (50, 13) saved trace

    @property
    def key(self) -> str:
        return f"m{self.mass:.3f}_mu{self.friction:.3f}"


# ---------------------------------------------------------------------------
# State-space analysis (no rendering)
# ---------------------------------------------------------------------------
def relative_yaw_deg(q0: np.ndarray, q: np.ndarray) -> float:
    qrel = pilot.quaternion_multiply(np.asarray(q, float),
                                     np.array([q0[0], -q0[1], -q0[2], -q0[3]], float))
    w, x, y, z = qrel / np.linalg.norm(qrel)
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def stop_frame(trace: np.ndarray, threshold: float = REST_SPEED_M_S) -> int | None:
    """First frame after which linear speed stays below threshold; None if still moving."""
    moving = np.linalg.norm(trace[:, 7:10], axis=1) >= threshold
    if moving[-1]:
        return None
    idx = np.flatnonzero(moving)
    return 0 if idx.size == 0 else int(idx[-1]) + 1


def state_metrics(trace: np.ndarray, q0: np.ndarray, dt: float) -> dict[str, Any]:
    speed = np.linalg.norm(trace[:, 7:10], axis=1)
    stop = stop_frame(trace)
    return {
        "final_dx_m": float(trace[-1, 0]), "final_dy_m": float(trace[-1, 1]),
        "final_xy_displacement_m": float(np.linalg.norm(trace[-1, :2])),
        "max_xy_displacement_m": float(np.linalg.norm(trace[:, :2], axis=1).max()),
        "final_yaw_change_deg": relative_yaw_deg(q0, trace[-1, 3:7]),
        "peak_speed_m_s": float(speed.max()), "final_speed_m_s": float(speed[-1]),
        "stop_time_s": None if stop is None else (stop + 1) * dt,
        "at_rest_at_end": stop is not None,
    }


def response_exponents(configs: list[Config], action: str) -> dict[str, Any] | None:
    """Least-squares log d = c + a log m + b log mu over a geometry's valid traces.

    The (a, b) direction is the mass/friction combination one action's final
    displacement responds to; configurations along the orthogonal direction
    produce (nearly) the same displacement. Local fit over the grid range only.
    """
    rows = [(c.mass, c.friction, float(np.linalg.norm(c.traces[action][-1, :2])))
            for c in configs if action in c.traces]
    rows = [r for r in rows if r[2] > 1e-4]
    if len(rows) < 4:
        return None
    m, mu, d = (np.array(x) for x in zip(*rows))
    X = np.column_stack([np.ones_like(m), np.log(m), np.log(mu)])
    coef, *_ = np.linalg.lstsq(X, np.log(d), rcond=None)
    pred = X @ coef
    ss_tot = float(np.sum((np.log(d) - np.log(d).mean()) ** 2))
    r2 = 1.0 - float(np.sum((np.log(d) - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"mass_exponent": float(coef[1]), "friction_exponent": float(coef[2]),
            "r2": r2, "n": len(rows)}


def direction_angle_deg(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    u = np.array([a["mass_exponent"], a["friction_exponent"]])
    v = np.array([b["mass_exponent"], b["friction_exponent"]])
    cos = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def final_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[-1] - b[-1]))


def path_rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def pairwise(configs: list[Config], features: dict[str, np.ndarray], fn) -> list[dict[str, Any]]:
    rows = []
    for a, b in itertools.combinations(configs, 2):
        if a.key not in features or b.key not in features:
            continue
        kind = ("mass_only" if np.isclose(a.friction, b.friction) else
                "friction_only" if np.isclose(a.mass, b.mass) else "both")
        rows.append({"a": a.key, "b": b.key, "kind": kind,
                     "distance": fn(features[a.key], features[b.key])})
    return rows


def summarize_pairs(rows: list[dict[str, Any]], thresholds: tuple[float, ...],
                    configs: list[Config]) -> dict[str, Any]:
    if not rows:
        return {"pairs": 0}
    by_key = {c.key: c for c in configs}

    def adjacent(r: dict[str, Any], attr: str, step: float) -> bool:
        return np.isclose(abs(getattr(by_key[r["a"]], attr) - getattr(by_key[r["b"]], attr)), step)

    d = np.array([r["distance"] for r in rows])
    closest = rows[int(d.argmin())]
    levels_m = sorted({c.mass for c in configs})
    levels_f = sorted({c.friction for c in configs})
    m_step = float(np.min(np.diff(levels_m))) if len(levels_m) > 1 else np.nan
    f_step = float(np.min(np.diff(levels_f))) if len(levels_f) > 1 else np.nan
    m_adj = [r["distance"] for r in rows if r["kind"] == "mass_only" and adjacent(r, "mass", m_step)]
    f_adj = [r["distance"] for r in rows if r["kind"] == "friction_only" and adjacent(r, "friction", f_step)]
    return {
        "pairs": len(rows), "min": float(d.min()), "median": float(np.median(d)),
        "closest_pair": [closest["a"], closest["b"]], "closest_kind": closest["kind"],
        "below": {str(t): int((d < t).sum()) for t in thresholds},
        "mean_one_mass_step": float(np.mean(m_adj)) if m_adj else None,
        "mean_one_friction_step": float(np.mean(f_adj)) if f_adj else None,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def object_geoms(model, bid: int) -> np.ndarray:
    def in_subtree(body: int) -> bool:
        while body > 0:
            if body == bid:
                return True
            body = int(model.body_parentid[body])
        return False
    geoms = np.array([g for g in range(model.ngeom) if in_subtree(int(model.geom_bodyid[g]))])
    if geoms.size == 0:
        raise RuntimeError("Object body has no geoms")
    return geoms


def frame_camera(model, mujoco, centers_xy: np.ndarray, z: float, radius: float):
    """One fixed camera covering every object position of every configuration."""
    lo, hi = centers_xy.min(axis=0), centers_xy.max(axis=0)
    half = float(max(hi - lo)) / 2 + radius
    fovy = float(model.vis.global_.fovy)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z]
    cam.distance = max(0.25, 1.4 * half / np.tan(np.radians(fovy) / 2))
    cam.azimuth, cam.elevation = CAMERA_AZIMUTH, CAMERA_ELEVATION
    return cam


def render_state(renderer, model, data, mujoco, cam, qadr: int, qpos7: np.ndarray,
                 geoms: np.ndarray, rgb: bool) -> tuple[np.ndarray | None, np.ndarray]:
    data.qpos[qadr:qadr + 7] = qpos7
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    image = None
    if rgb:
        renderer.update_scene(data, camera=cam)
        image = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], geoms)
    return image, mask


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError("Object left the camera view")
    if (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
        raise RuntimeError("Object touches the image border; framing is too tight")
    return np.array([xs.mean(), ys.mean()])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def geometry_figure(path: Path, geom_label: str, configs: list[Config], initial: np.ndarray,
                    initial_mask: np.ndarray, finals: dict[str, np.ndarray],
                    paths_px: dict[str, np.ndarray], metrics: dict[str, dict[str, Any]],
                    initial_max_diff: float, dt: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masses = sorted({c.mass for c in configs})
    fricts = sorted({c.friction for c in configs})
    fig = plt.figure(figsize=(3.0 * (len(fricts) + 1), 3.1 * len(masses)))
    grid = fig.add_gridspec(len(masses), len(fricts) + 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(initial)
    ax.set_title(f"start-aligned initial (all {len(configs)})\nmax pixel diff = "
                 f"{initial_max_diff:.0f}/255", fontsize=8)
    ax.axis("off")
    ax = fig.add_subplot(grid[1:, 0])
    for c in configs:
        speed = np.linalg.norm(c.traces[RENDER_ACTION][:, 7:10], axis=1)
        ax.plot((np.arange(len(speed)) + 1) * dt, speed,
                color=plt.cm.viridis((c.mass - masses[0]) / max(1e-9, masses[-1] - masses[0])),
                linestyle=["-", "--", ":"][fricts.index(c.friction) % 3], lw=1.2)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("speed (m/s)", fontsize=8)
    ax.set_title("speed: color=mass (dark light)\nline=friction (- -- :)", fontsize=8)
    ax.tick_params(labelsize=7)
    for c in configs:
        r, k = masses.index(c.mass), fricts.index(c.friction)
        ax = fig.add_subplot(grid[r, k + 1])
        ax.imshow(finals[c.key])
        ax.contour(initial_mask, levels=[0.5], colors="cyan", linewidths=0.7)
        p = paths_px[c.key]
        ax.plot(p[:, 0], p[:, 1], color="yellow", lw=0.8)
        ax.plot(p[-1, 0], p[-1, 1], "r.", ms=4)
        m = metrics[c.key]
        ax.set_title(f"m={c.mass:.2f} kg  mu={c.friction:.1f}\n"
                     f"d={1000 * m['final_xy_displacement_m']:.0f} mm  "
                     f"yaw={m['final_yaw_change_deg']:.0f} deg", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{geom_label}\n{RENDER_ACTION} final frame (t={(len(p) - 1) * dt:.2f} s); "
                 "cyan = initial outline, yellow = image-space path", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def settling_figure(path: Path, panels: list[tuple[str, np.ndarray, dict[float, np.ndarray],
                                                     dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.4), squeeze=False)
    colors = ["red", "lime", "blue"]
    for ax, (object_id, image, masks, settle) in zip(axes[0], panels):
        ax.imshow(image)
        for k, (friction, mask) in enumerate(sorted(masks.items())):
            ax.contour(mask, levels=[0.5], colors=colors[k % 3], linewidths=0.8)
        ax.set_title(f"{object_id.split('__')[-1]}\n"
                     f"{1000 * settle['max_start_offset_m']:.1f} mm, "
                     f"{settle['max_start_rotation_deg']:.1f} deg", fontsize=7)
        ax.axis("off")
    fig.suptitle("Simulated initial (pre-push) outlines by friction: red 0.2, green 0.3, blue 0.4 "
                 "(mass has no effect)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def heatmap_figure(path: Path, table: dict[str, dict[str, dict[str, float]]],
                   actions: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geoms = list(table)
    fig, axes = plt.subplots(len(actions), len(geoms), figsize=(2.3 * len(geoms), 2.4 * len(actions)),
                             squeeze=False)
    for r, action in enumerate(actions):
        for k, geom in enumerate(geoms):
            ax = axes[r, k]
            cells = table[geom].get(action, {})
            masses = sorted({float(key.split("_")[0][1:]) for key in cells})
            fricts = sorted({float(key.split("_mu")[1]) for key in cells})
            grid = np.full((len(masses), len(fricts)), np.nan)
            for key, value in cells.items():
                m, f = float(key.split("_")[0][1:]), float(key.split("_mu")[1])
                grid[masses.index(m), fricts.index(f)] = 1000 * value
            ax.imshow(grid, cmap="magma", aspect="auto")
            mid = np.nanmean([np.nanmin(grid), np.nanmax(grid)])
            for (i, j), v in np.ndenumerate(grid):
                ax.text(j, i, "-" if np.isnan(v) else f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if v > mid else "white")
            ax.set_xticks(range(len(fricts)), [f"{f:.1f}" for f in fricts], fontsize=6)
            ax.set_yticks(range(len(masses)), [f"{m:.2f}" for m in masses], fontsize=6)
            if r == 0:
                ax.set_title(geom.split("__")[-1], fontsize=7)
            if k == 0:
                ax.set_ylabel(f"{action}\nmass (kg)", fontsize=7)
            if r == len(actions) - 1:
                ax.set_xlabel("friction", fontsize=7)
    fig.suptitle("Final xy displacement (mm) from saved states; '-' = rejected trace", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def load_valid_traces(inv: export.GridInventory, object_id: str) -> list[Config]:
    configs = []
    for record in inv.manifest["records"]:
        if record["object"] != object_id:
            continue
        traces = {}
        for action, status in record["action_status"].items():
            if status.get("valid") is True:
                data = export._load_npz(inv.grid_dir / status["trajectory_path"])
                if export._scalar_str(data["action_name"]) != action:
                    raise RuntimeError(f"Trace identity mismatch: {status['trajectory_path']}")
                traces[action] = data["trajectory"]
        configs.append(Config(float(record["mass_kg"]), float(record["contact_friction"]), traces))
    return sorted(configs, key=lambda c: (c.mass, c.friction))


def resolve_geometries(inv: export.GridInventory, names: list[str] | None) -> list[export.Geometry]:
    if not names:
        return list(inv.geometries)
    chosen = []
    for name in names:
        hits = [g for g in inv.geometries if name in g.object_id]
        if len(hits) != 1:
            raise SystemExit(f"--geometry {name!r} matches {[g.object_id for g in hits]}")
        chosen.append(hits[0])
    return chosen


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    inv = export.load_grid(args.grid_dir)
    engine_mod = export.load_phino_module(args.phino_root, "train.data.mujoco_engine")
    engine = engine_mod.MuJoCoPhysicsEngine()
    originals = {a.name: a for a in engine_mod.DIAGNOSTIC_ACTIONS}
    dt = float(engine.frame_dt)

    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"Output directory already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    try:
        (stage / "figures").mkdir()
        state_rows, pair_rows, settle_panels = [], [], []
        heat: dict[str, dict[str, dict[str, float]]] = {}
        exponents: dict[str, dict[str, Any]] = {}
        summary_geoms: dict[str, Any] = {}
        for geom in inv.geometries:
            heat[geom.object_id] = {}
            all_configs = load_valid_traces(inv, geom.object_id)
            for c in all_configs:
                for action in (RENDER_ACTION, SECOND_ACTION):
                    if action in c.traces:
                        heat[geom.object_id].setdefault(action, {})[c.key] = float(
                            np.linalg.norm(c.traces[action][-1, :2]))
            fits = {a: response_exponents(all_configs, a)
                    for a in ("push_x", "push_y", "lateral_flick")}
            exponents[geom.object_id] = {
                **fits, f"angle_{RENDER_ACTION}_vs_{SECOND_ACTION}_deg":
                    direction_angle_deg(fits[RENDER_ACTION], fits[SECOND_ACTION])}

        for geom in resolve_geometries(inv, args.geometry):
            configs = [c for c in load_valid_traces(inv, geom.object_id) if RENDER_ACTION in c.traces]
            if len(configs) < 4:
                raise SystemExit(f"{geom.object_id}: only {len(configs)} valid {RENDER_ACTION} traces")
            gdir = stage / "renders" / geom.object_id
            gdir.mkdir(parents=True)

            starts = {}
            for c in configs:
                obj = pilot.build_model(geom.isolated_scene, geom.body_name, c.mass, c.friction,
                                        engine, mujoco)
                starts[c.key] = obj
            ref = configs[len(configs) // 2]
            replay = None
            if not args.skip_replay_check:
                _, resim = pilot.simulate_action(
                    starts[ref.key], RENDER_ACTION, originals, engine, mujoco,
                    float(inv.manifest["max_raw_displacement_m"]), float(inv.manifest["max_tilt_deg"]))
                replay = float(np.abs(resim - ref.traces[RENDER_ACTION]).max())
                if replay > REPLAY_TOLERANCE:
                    raise RuntimeError(f"{geom.object_id}: re-simulation differs from saved trace "
                                       f"by {replay:.2e}; settled start is not reproducible")

            model = mujoco.MjModel.from_xml_path(str(geom.isolated_scene))
            data = mujoco.MjData(model)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, geom.body_name)
            qadr = int(model.jnt_qposadr[int(model.body_jntadr[bid])])
            geoms = object_geoms(model, bid)
            ref_obj = starts[ref.key]
            ref_q0 = ref_obj["q0"][qadr + 3:qadr + 7]

            def pose(mode: str, c: Config, t: int) -> np.ndarray:
                obj = starts[c.key]
                q0 = obj["q0"][qadr + 3:qadr + 7]
                trace = c.traces[RENDER_ACTION]
                rel, q = (np.zeros(3), q0) if t < 0 else (trace[t, :3], trace[t, 3:7])
                if mode == "faithful":
                    return np.concatenate([obj["start"] + rel, q])
                qrel = pilot.quaternion_multiply(q, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
                q_aligned = pilot.quaternion_multiply(qrel, ref_q0)
                return np.concatenate([ref_obj["start"] + rel, q_aligned / np.linalg.norm(q_aligned)])

            data.qpos[:] = ref_obj["q0"]
            mujoco.mj_forward(model, data)
            center = data.xpos[bid].copy()
            radius = max(float(np.linalg.norm(data.geom_xpos[g] - center)) + float(model.geom_rbound[g])
                         for g in geoms)
            centers = np.array([pose(mode, c, t)[:2] for mode in MODES for c in configs
                                for t in range(-1, len(c.traces[RENDER_ACTION]))])
            cam = frame_camera(model, mujoco, centers, float(center[2]), radius)

            from PIL import Image
            views: dict[str, dict[str, dict[str, np.ndarray]]] = {m: {} for m in MODES}
            renderer = mujoco.Renderer(model, height=IMG_SIZE, width=IMG_SIZE)
            try:
                for mode in MODES:
                    for c in configs:
                        n = len(c.traces[RENDER_ACTION])
                        img0, mask0 = render_state(renderer, model, data, mujoco, cam, qadr,
                                                   pose(mode, c, -1), geoms, rgb=True)
                        path = [mask_centroid(mask0)]
                        for t in range(n):
                            img, mask = render_state(renderer, model, data, mujoco, cam, qadr,
                                                     pose(mode, c, t), geoms, rgb=(t == n - 1))
                            path.append(mask_centroid(mask))
                        views[mode][c.key] = {"initial": img0, "final": img, "mask0": mask0,
                                              "path": np.array(path)}
                        Image.fromarray(img0).save(gdir / f"{mode}_{c.key}_initial.png")
                        Image.fromarray(img).save(gdir / f"{mode}_{c.key}_final.png")
            finally:
                renderer.close()

            metrics = {c.key: state_metrics(c.traces[RENDER_ACTION],
                                            starts[c.key]["q0"][qadr + 3:qadr + 7], dt)
                       for c in configs}
            vadr = ref_obj["vadr"]
            settle = {
                "max_start_offset_m": float(max(np.linalg.norm(starts[c.key]["start"] - ref_obj["start"])
                                                for c in configs)),
                "max_start_rotation_deg": float(max(pilot.quat_orientation_delta_deg(
                    starts[c.key]["q0"][qadr + 3:qadr + 7], ref_q0) for c in configs)),
                "max_start_linear_speed_m_s": float(max(np.linalg.norm(starts[c.key]["v0"][vadr:vadr + 3])
                                                        for c in configs)),
                "max_start_angular_speed_rad_s": float(max(np.linalg.norm(
                    starts[c.key]["v0"][vadr + 3:vadr + 6]) for c in configs)),
                "max_start_offset_between_masses_m": float(max(
                    np.linalg.norm(starts[a.key]["start"] - starts[b.key]["start"])
                    for a, b in itertools.combinations(configs, 2) if np.isclose(a.friction, b.friction))),
            }
            settle["matched_initial_state"] = bool(
                settle["max_start_offset_m"] < 1e-3 and settle["max_start_rotation_deg"] < 1.0
                and settle["max_start_linear_speed_m_s"] < REST_SPEED_M_S
                and settle["max_start_angular_speed_rad_s"] < 0.1)
            init_diff = {mode: float(np.abs(np.stack([views[mode][c.key]["initial"].astype(int)
                                                       for c in configs])
                                            - views[mode][ref.key]["initial"].astype(int)).max())
                         for mode in MODES}
            for c in configs:
                v = views["aligned"][c.key]
                state_rows.append({"object_id": geom.object_id, "action": RENDER_ACTION,
                                   "mass_kg": c.mass, "contact_friction": c.friction,
                                   **metrics[c.key],
                                   "aligned_final_centroid_px_x": float(v["path"][-1, 0]),
                                   "aligned_final_centroid_px_y": float(v["path"][-1, 1]),
                                   "aligned_image_motion_px": float(np.linalg.norm(
                                       v["path"][-1] - v["path"][0]))})

            state_xy = {c.key: c.traces[RENDER_ACTION][:, :2] for c in configs}
            two_action = {c.key: np.concatenate([c.traces[RENDER_ACTION][-1, :2],
                                                 c.traces[SECOND_ACTION][-1, :2]])[None, :]
                          for c in configs if SECOND_ACTION in c.traces}
            mad = lambda a, b: float(np.abs(a - b).mean())  # noqa: E731
            observations = {
                "state_final_position_m": (state_xy, final_distance, STATE_THRESHOLDS_M),
                "state_path_rms_m": (state_xy, path_rms_distance, STATE_THRESHOLDS_M),
                f"state_final_position_{RENDER_ACTION}+{SECOND_ACTION}_m":
                    (two_action, final_distance, STATE_THRESHOLDS_M),
            }
            for mode in MODES:
                vm = views[mode]
                observations.update({
                    f"image_{mode}_initial_mean_abs_diff": (
                        {k: v["initial"].astype(float) for k, v in vm.items()}, mad, PIXEL_THRESHOLDS),
                    f"image_{mode}_final_centroid_px": (
                        {k: v["path"] for k, v in vm.items()}, final_distance, PIXEL_THRESHOLDS),
                    f"image_{mode}_path_rms_px": (
                        {k: v["path"] for k, v in vm.items()}, path_rms_distance, PIXEL_THRESHOLDS),
                    f"image_{mode}_final_mean_abs_diff": (
                        {k: v["final"].astype(float) for k, v in vm.items()}, mad, PIXEL_THRESHOLDS),
                })
            seps = {}
            for name, (feats, fn, thresholds) in observations.items():
                rows = pairwise(configs, feats, fn)
                seps[name] = summarize_pairs(rows, thresholds, configs)
                pair_rows += [{"object_id": geom.object_id, "observation": name, **r} for r in rows]

            aligned = views["aligned"]
            px_per_m = float(np.median([
                np.linalg.norm(aligned[c.key]["path"][-1] - aligned[c.key]["path"][0])
                / np.linalg.norm(c.traces[RENDER_ACTION][-1, :2])
                for c in configs if np.linalg.norm(c.traces[RENDER_ACTION][-1, :2]) > 0.01]))
            summary_geoms[geom.object_id] = {
                "configs": [c.key for c in configs], "replay_max_abs_diff": replay,
                "reference_config": ref.key, "settling": settle,
                "initial_image_max_pixel_diff": init_diff,
                "camera": {"lookat": [float(x) for x in cam.lookat], "distance": float(cam.distance),
                           "azimuth": CAMERA_AZIMUTH, "elevation": CAMERA_ELEVATION},
                "approx_px_per_m_along_push": px_per_m,
                "all_at_rest_at_end": all(m["at_rest_at_end"] for m in metrics.values()),
                "separability": seps,
            }
            note = ("matched initial state" if settle["matched_initial_state"] else
                    f"SETTLE-CONFOUNDED: start differs by up to {1000 * settle['max_start_offset_m']:.1f} mm"
                    f" / {settle['max_start_rotation_deg']:.1f} deg across friction")
            geometry_figure(stage / "figures" / f"{geom.object_id}_{RENDER_ACTION}.png",
                            f"{geom.object_id} [{note}]", configs, aligned[ref.key]["initial"],
                            aligned[ref.key]["mask0"], {k: v["final"] for k, v in aligned.items()},
                            {k: v["path"] for k, v in aligned.items()}, metrics,
                            init_diff["aligned"], dt)
            by_friction = {}
            for c in configs:
                if np.isclose(c.mass, ref.mass):
                    by_friction[c.friction] = views["faithful"][c.key]["mask0"]
            settle_panels.append((geom.object_id, views["faithful"][ref.key]["initial"],
                                  by_friction, settle))

        heatmap_figure(stage / "figures" / "final_displacement_heatmaps.png", heat,
                       [RENDER_ACTION, SECOND_ACTION])
        settling_figure(stage / "figures" / "settling_confound.png", settle_panels)
        for name, rows in (("state_metrics.csv", state_rows), ("pairwise_separability.csv", pair_rows)):
            with open(stage / name, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = {
            "format": "simpact_phino_visual_observability_v1",
            "command": sys.argv,
            "simpact_git": export._git_state(Path(__file__).resolve().parent.parent),
            "phino_git": export._git_state(args.phino_root.expanduser().resolve()),
            "source_grid": str(inv.grid_dir), "manifest_sha256": inv.manifest_sha256,
            "render_action": RENDER_ACTION, "second_action": SECOND_ACTION,
            "image_size": IMG_SIZE, "frame_dt_s": dt, "rest_speed_m_s": REST_SPEED_M_S,
            "method": ("Kinematic replay of saved valid traces in each isolated scene; no new "
                       "physics. 'faithful' = simulated states; 'aligned' = saved relative "
                       "motion re-anchored to the reference config's settled start. Camera "
                       "fixed per geometry, framed on the union of both modes' trajectories. "
                       "Centroids from MuJoCo segmentation masks."),
            "final_displacement_m": heat, "response_exponents": exponents,
            "geometries": summary_geoms,
            "caveats": [
                "Noiseless synthetic renders with known camera and action: an upper bound on "
                "what images can reveal.",
                "Per-geometry camera framing uses that geometry's trajectory extent.",
                "Mass and friction are synthetic simulator assignments.",
            ],
        }
        export._write_json(stage / "summary.json", summary)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, default=Path("results/phino_mesh_grid_cutoff_1m_v1"))
    ap.add_argument("--phino-root", type=Path, default=export.DEFAULT_PHINO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("results/phino_visual_observability_v1"))
    ap.add_argument("--geometry", action="append", default=None,
                    help="Unique substring of an object ID (repeatable); default: all geometries")
    ap.add_argument("--skip-replay-check", action="store_true",
                    help="Skip the one-config re-simulation check (synthetic fixtures only)")
    args = ap.parse_args()
    summary = run(args)
    for object_id, g in summary["geometries"].items():
        s = g["separability"]
        print(f"{object_id}: matched_start={g['settling']['matched_initial_state']} "
              f"replay diff={g['replay_max_abs_diff']} "
              f"aligned min final-centroid sep={s['image_aligned_final_centroid_px']['min']:.2f}px "
              f"({s['image_aligned_final_centroid_px']['closest_pair']}), "
              f"min path sep={s['image_aligned_path_rms_px']['min']:.2f}px")
    print(f"OUTPUT={args.output_dir.resolve()}")
    print("VISUAL_OBSERVABILITY_PILOT_OK")


if __name__ == "__main__":
    main()
