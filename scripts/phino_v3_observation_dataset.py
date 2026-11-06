#!/usr/bin/env python3
"""Build action-conditioned before/after image clips from the Stage 3 v3 pilot.

    MUJOCO_GL=egl python scripts/phino_v3_observation_dataset.py \
        --pilot-dir results/phino_v3_identifiability_pilot_v1 \
        --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
        --output-dir data/phino_simpact_v3obs_v1

No new physics: every image is a kinematic replay of a saved Stage 3 trace
(start-relative position and rotation) from that configuration's common start.
Starts are recomputed with the deterministic v3 protocol and each geometry is
gated by re-simulating one action and requiring an exact match with the saved
trace. Only configurations whose start passed S2 and S3 are used, so before
images do not leak friction through the settle.

One camera per intervention family, fixed relative to the object's canonical
start and identical across geometries and physics configurations:
  push cameras (push_x_v2, push_lo, push_hi): overhead-oblique
  toss camera: near-side view
Each clip = before image + post-onset frames at FRAME_TIMES. Segmentation
masks give object centroid/area for the image-position baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import export_phino_dataset as export
import phino_simpact_grid_v3 as v3
import phino_v3_identifiability_pilot as stage3

ACTIONS = ("push_x_v2", "push_lo_1.6N", "push_hi_3.2N", "toss_4.0N")
CAMERA_OF = {"push_x_v2": "push", "push_lo_1.6N": "push", "push_hi_3.2N": "push", "toss_4.0N": "toss"}
CAMERAS = {"push": {"azimuth": 90.0, "elevation": -55.0}, "toss": {"azimuth": 90.0, "elevation": -10.0}}
FRAME_INDICES = (2, 4, 6, 9, 49)           # 20 Hz trace indices -> 0.15, 0.25, 0.35, 0.50, 2.50 s
IMG = 224
MIN_START_OK_CONFIGS = 20
REPLAY_TOLERANCE = 1e-5
FRAMING_MARGIN = 1.25


def frame_time(index: int, dt: float = 0.05) -> float:
    return (index + 1) * dt


def short_name(object_id: str) -> str:
    return object_id.split("__")[-1]


def camera_spec(extents: dict[str, np.ndarray], radius: float, kind: str, fovy_deg: float
                ) -> dict[str, Any]:
    """Offset (relative to the canonical start) and distance covering the union extent."""
    lo, hi = extents["lo"], extents["hi"]
    if kind == "push":
        half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + radius
        offset = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0]
    else:
        half = max(hi[0] - lo[0], hi[2] - lo[2] + radius) / 2 + radius
        offset = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2 + radius / 2]
    distance = FRAMING_MARGIN * half / np.tan(np.radians(fovy_deg) / 2)
    return {"offset": [float(x) for x in offset], "distance": float(distance), **CAMERAS[kind]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", type=Path, default=Path("results/phino_v3_identifiability_pilot_v1"))
    ap.add_argument("--grid-dir", type=Path, default=Path("results/phino_mesh_grid_cutoff_1m_v1"))
    ap.add_argument("--phino-root", type=Path, default=export.DEFAULT_PHINO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("data/phino_simpact_v3obs_v2"))
    args = ap.parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        sys.exit(f"Output directory already exists: {out}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image

    summary_path = args.pilot_dir / "summary.json"
    traces_path = args.pilot_dir / "traces.npz"
    pilot_summary = json.loads(summary_path.read_text())
    traces = np.load(traces_path)
    inv = export.load_grid(args.grid_dir)
    if inv.manifest_sha256 != pilot_summary["manifest_sha256"]:
        sys.exit("Grid manifest differs from the one used by the Stage 3 pilot")
    engine_mod = export.load_phino_module(args.phino_root, "train.data.mujoco_engine")
    engine = engine_mod.MuJoCoPhysicsEngine()
    originals = {a.name: a for a in engine_mod.DIAGNOSTIC_ACTIONS}
    configs = stage3.build_configs()
    cfg_by_key = {c["key"]: c for c in configs}
    max_disp = float(inv.manifest["max_raw_displacement_m"])
    max_tilt = float(inv.manifest["max_tilt_deg"])

    # 1. Recompute common starts; gate each geometry on exact replay.
    geo: dict[str, dict[str, Any]] = {}
    for g in inv.geometries:
        ref = v3.make_sim(g.isolated_scene, g.body_name, *stage3.CENTER, engine, mujoco)
        canon = v3.canonical_rest_pose(ref, mujoco)
        starts = {}
        for c in configs:
            try:
                sim = v3.make_sim(g.isolated_scene, g.body_name, c["mass"], c["friction"], engine, mujoco)
                st = v3.start_state(sim, canon["q_star"], engine, mujoco)
                z = v3.rollout(sim, st["q0"], st["v0"], v3.ZERO_ACTION, engine, mujoco, originals,
                               max_disp, max_tilt)
                s3 = z["valid"] and float(np.linalg.norm(z["trace20"][:, :3], axis=1).max()) \
                    < v3.S3_MAX_DISPLACEMENT_M
                starts[c["key"]] = {"ok": bool(st["ok"] and s3), "q0": st["q0"], "v0": st["v0"]}
            except v3.pilot.RejectedTrial:
                starts[c["key"]] = {"ok": False}
        n_ok = sum(s["ok"] for s in starts.values())
        info = {"object_id": g.object_id, "start_ok_configs": n_ok, "included": n_ok >= MIN_START_OK_CONFIGS,
                "q_star": canon["q_star"], "starts": starts, "geometry": g, "sim": ref}
        if info["included"]:
            center = stage3.config_key(*stage3.CENTER)
            key = next(k for k in [center] + [c["key"] for c in configs] if starts[k]["ok"]
                       and f"{g.object_id}|{k}|toss_4.0N|20" in traces.files)
            s, c = starts[key], cfg_by_key[key]
            sim = v3.make_sim(g.isolated_scene, g.body_name, c["mass"], c["friction"], engine, mujoco)
            r = v3.rollout(sim, s["q0"], s["v0"], v3.ACTION_BY_NAME["toss_4.0N"], engine, mujoco,
                           originals, max_disp, max_tilt)
            saved = traces[f"{g.object_id}|{key}|toss_4.0N|20"]
            diff = float(np.abs(r["trace20"] - saved).max()) if r["valid"] else float("inf")
            info["replay_check"] = {"config": key, "action": "toss_4.0N", "max_abs_diff": diff}
            if diff > REPLAY_TOLERANCE:
                sys.exit(f"{g.object_id}: replay differs from saved Stage 3 trace by {diff:.2e}")
        geo[g.object_id] = info
        print(f"{short_name(g.object_id)}: start-ok {n_ok}/{len(configs)} included={info['included']} "
              f"replay={info.get('replay_check', {}).get('max_abs_diff')}")
    included = [gid for gid, v in geo.items() if v["included"]]

    # 2. Samples: (geometry, start-ok config, valid action).
    cfg = {c["key"]: c for c in configs}
    samples = []
    for gid in included:
        for key, st in geo[gid]["starts"].items():
            if not st["ok"]:
                continue
            for a in ACTIONS:
                if f"{gid}|{key}|{a}|20" in traces.files:
                    samples.append((gid, key, a))

    # 3. Cameras from the union extent of used traces over included geometries.
    radii = {gid: v3.object_radius(geo[gid]["sim"], geo[gid]["q_star"], mujoco) for gid in included}
    cams = {}
    for kind in CAMERAS:
        pts = np.concatenate([traces[f"{g}|{k}|{a}|20"][:, :3] for g, k, a in samples if CAMERA_OF[a] == kind])
        cams[kind] = camera_spec({"lo": pts.min(0), "hi": pts.max(0)}, max(radii.values()), kind, 45.0)

    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=str(out.parent)))
    try:
        rows, traj = [], {}
        shared_before: dict[tuple[str, str], tuple[list[float], int]] = {}
        for gid in included:
            info = geo[gid]
            sim0 = info["sim"]
            model = sim0.model
            fovy = float(model.vis.global_.fovy)
            if not np.isclose(fovy, 45.0):
                sys.exit(f"{gid}: fovy {fovy} != 45; camera framing would differ across scenes")
            data = mujoco.MjData(model)
            qa = sim0.qadr
            p_star = info["q_star"][qa:qa + 3]
            renderer = mujoco.Renderer(model, height=IMG, width=IMG)
            try:
                for (g, key, a) in [s for s in samples if s[0] == gid]:
                    # Anchor every clip to the canonical rest pose: each configuration's verified
                    # start still differs by < 0.1 mm in a friction-dependent way, which DINOv2
                    # features plus standardized ridge can detect.
                    q0 = info["q_star"][qa:qa + 7]
                    tr = traces[f"{g}|{key}|{a}|20"]
                    spec = cams[CAMERA_OF[a]]
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = p_star + np.array(spec["offset"])
                    cam.distance, cam.azimuth, cam.elevation = spec["distance"], spec["azimuth"], spec["elevation"]
                    poses = [q0] + [np.concatenate([q0[:3] + tr[t, :3],
                                                   _compose(tr[t, 3:7], q0[3:7])]) for t in FRAME_INDICES]
                    rel_dir = Path("images") / short_name(gid) / key / a
                    (stage / rel_dir).mkdir(parents=True, exist_ok=True)
                    paths, cents, areas, border = [], [], [], False
                    for i, q in enumerate(poses):
                        shared = Path("images") / short_name(gid) / f"before_{CAMERA_OF[a]}.png"
                        if i == 0 and shared_before.get((gid, CAMERA_OF[a])):
                            # one rendered before image per geometry and camera (EGL output is not
                            # bit-deterministic, so re-rendering would add per-clip noise)
                            paths.append(str(shared))
                            c0, a0 = shared_before[(gid, CAMERA_OF[a])]
                            cents.append(c0)
                            areas.append(a0)
                            continue
                        data.qpos[:] = info["q_star"]
                        data.qpos[qa:qa + 7] = q
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=cam)
                        rgb = renderer.render().copy()
                        renderer.enable_segmentation_rendering()
                        renderer.update_scene(data, camera=cam)
                        seg = renderer.render()
                        renderer.disable_segmentation_rendering()
                        mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], sim0.object_geoms)
                        ys, xs = np.nonzero(mask)
                        if xs.size == 0:
                            border = True
                            cents.append([np.nan, np.nan])
                        else:
                            cents.append([float(xs.mean()), float(ys.mean())])
                            border |= bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
                        areas.append(int(mask.sum()))
                        p = shared if i == 0 else rel_dir / f"f{i}.png"
                        Image.fromarray(rgb).save(stage / p)
                        paths.append(str(p))
                        if i == 0:
                            shared_before[(gid, CAMERA_OF[a])] = (cents[-1], areas[-1])
                    c = cfg[key]
                    sid = f"{short_name(gid)}|{key}|{a}"
                    traj[sid] = tr
                    rows.append({"sample_id": sid, "geometry": gid, "geometry_short": short_name(gid),
                                 "config": key, "kind": c["kind"], "cluster": c.get("center", key),
                                 "mass": c["mass"], "friction": c["friction"], "action": a,
                                 "camera": CAMERA_OF[a], "frame_times_s": [0.0] + [frame_time(t) for t in FRAME_INDICES],
                                 "images": paths, "centroid_px": cents, "mask_area_px": areas,
                                 "object_in_frame": not border})
            finally:
                renderer.close()
            for kind in CAMERAS:
                befores = {r["images"][0] for r in rows if r["geometry"] == gid and r["camera"] == kind}
                if len(befores) > 1:
                    sys.exit(f"{gid}/{kind}: clips do not share one before image")
            print(f"rendered {short_name(gid)}: {sum(r['geometry'] == gid for r in rows)} clips")
        np.savez_compressed(stage / "trajectories.npz", **traj)
        (stage / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        prov = {
            "format": "simpact_phino_v3_action_conditioned_observations_v1",
            "command": sys.argv,
            "simpact_git": export._git_state(Path(__file__).resolve().parent.parent),
            "phino_git": export._git_state(args.phino_root.expanduser().resolve()),
            "pilot_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "pilot_traces_sha256": export.sha256_file(traces_path),
            "manifest_sha256": inv.manifest_sha256,
            "actions": list(ACTIONS), "frame_indices_20hz": list(FRAME_INDICES),
            "frame_times_s": [0.0] + [frame_time(t) for t in FRAME_INDICES],
            "cameras": cams, "image_size": IMG,
            "inclusion_rule": (f"configs passing S2 and S3 of the recomputed v3 common start; geometries with "
                               f">= {MIN_START_OK_CONFIGS} such configs"),
            "geometries": {gid: {"start_ok_configs": v["start_ok_configs"], "included": v["included"],
                                 "replay_check": v.get("replay_check"),
                                 "radius_m": radii.get(gid)} for gid, v in geo.items()},
            "n_samples": len(rows),
            "n_object_out_of_frame": sum(not r["object_in_frame"] for r in rows),
            "labels": "mass (kg) and contact friction are synthetic simulator assignments",
            "anchor": ("all frames rendered as canonical rest pose q* composed with the saved "
                       "start-relative motion; one shared before image per geometry and camera "
                       "(EGL rendering varies by up to 1/255 in a few pixels between identical renders)"),
            "notes": ["Kinematic replay of saved Stage 3 traces; no new physics.",
                      "v1 of this dataset anchored clips at each configuration's own verified start; "
                      "its sub-0.1 mm friction-dependent offsets leaked friction into static images "
                      "(within-geometry static probe nMAE 0.39) and it must not be used.",
                      "Camera framing uses the union motion extent over included geometries (global, "
                      "label-independent within a camera).",
                      "Bunny and sphere fail the common-start protocol and are excluded."],
        }
        export._write_json(stage / "provenance.json", prov)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"OUTPUT={out} samples={len(rows)}")


def _compose(q_rel: np.ndarray, q0: np.ndarray) -> np.ndarray:
    from audit_phino_settle_confound import qmul
    q = qmul(np.asarray(q_rel, float), np.asarray(q0, float))
    return q / np.linalg.norm(q)


if __name__ == "__main__":
    main()
