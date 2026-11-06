#!/usr/bin/env python3
"""Audit how friction-dependent settling affects earlier SIMPACT/PHINO results.

From the SIMPACT repository root:
    MUJOCO_GL=egl python scripts/audit_phino_settle_confound.py \
        --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
        --export-dir data/phino_simpact_v0 \
        --phino-root /nbu/people/ashah264/phino \
        --output-dir results/phino_settle_audit_v1

No grid data are regenerated. MuJoCo is used only to (a) recompute each
configuration's deterministic settled start with the generator's own
``build_model`` and (b) run a zero-action control from that start, measuring
motion that is not caused by any action. Existing grids and the v0 export are
read-only.

Parts:
  1. Settle diagnostics: start pose spread across friction, residual velocity,
     zero-action drift. Geometries are classified as matched-start ("clean") or
     settle-confounded from these measurements, not by name.
  2. PHINO similarity audit on the v0 export: original fingerprints vs
     start-relative fingerprints (quaternion relative to the settled start,
     world-frame angular velocity), and a counterfactual that swaps ONLY the
     settled orientation between two configurations while keeping the response
     identical.
  3. Ridge audit: reproduce the published recovery_scores.csv numbers exactly,
     then partition the same predictions by clean vs confounded geometry and
     retrain on clean geometries only. Also partitions the matched
     with/without-sphere comparison.
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
from pathlib import Path
from typing import Any

import numpy as np

import analyze_phino_pilot_grid as apg
import export_phino_dataset as export
import phino_simpact_pilot_v2 as pilot

CLEAN_MAX_OFFSET_M = 1e-3
CLEAN_MAX_ROTATION_DEG = 1.0
CLEAN_MAX_LINEAR_SPEED = 0.01
CLEAN_MAX_ANGULAR_SPEED = 0.1
RIDGE_ALPHA = 10.0
KEY_ACTION_SETS = (("push_x", "push_y"), apg.NON_LIFT)


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz, vectorized over leading axes)
# ---------------------------------------------------------------------------
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, float) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _relative_rotation(segment: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """World-frame rotation since the settled start, sign-continuous from ~identity."""
    q_rel = qmul(segment[:, 3:7], qconj(q0)[None, :])
    return q_rel * np.sign(q_rel[0, 0] or 1.0)


def start_relative_fingerprint(flat: np.ndarray, q0: np.ndarray, frames: int = 50) -> np.ndarray:
    """Replace absolute quaternion with rotation since settle; body -> world angular velocity."""
    out = np.array(flat, dtype=np.float64, copy=True)
    for k in range(flat.shape[0] // frames):
        seg = out[k * frames:(k + 1) * frames]
        world_w = np.einsum("tij,tj->ti", quat_to_mat(seg[:, 3:7]), seg[:, 10:13])
        seg[:, 3:7] = _relative_rotation(seg, q0)
        seg[:, 10:13] = world_w
    return out.astype(np.float32)


def swap_settle_orientation(flat: np.ndarray, q0_src: np.ndarray, q0_dst: np.ndarray,
                            frames: int = 50) -> np.ndarray:
    """Same world-frame response, started from another configuration's settled orientation."""
    out = np.array(flat, dtype=np.float64, copy=True)
    for k in range(flat.shape[0] // frames):
        seg = out[k * frames:(k + 1) * frames]
        world_w = np.einsum("tij,tj->ti", quat_to_mat(seg[:, 3:7]), seg[:, 10:13])
        q_cf = qmul(_relative_rotation(seg, q0_src), q0_dst[None, :])
        seg[:, 3:7] = q_cf
        seg[:, 10:13] = np.einsum("tji,tj->ti", quat_to_mat(q_cf), world_w)
    return out.astype(np.float32)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Part 1: settle diagnostics
# ---------------------------------------------------------------------------
def zero_action_control(obj: dict[str, Any], engine, mujoco) -> dict[str, float]:
    """Free evolution from the settled start with no applied wrench."""
    m, bid, qadr, vadr = obj["model"], obj["bid"], obj["qadr"], obj["vadr"]
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = obj["q0"], obj["v0"]
    mujoco.mj_forward(m, d)
    q_start = d.qpos[qadr + 3:qadr + 7].copy()
    max_disp = max_speed = 0.0
    for _ in range(int(engine.timesteps)):
        for _ in range(engine.substeps):
            d.xfrc_applied[:] = 0
            mujoco.mj_step(m, d)
        max_disp = max(max_disp, float(np.linalg.norm(d.qpos[qadr:qadr + 3] - obj["start"])))
        max_speed = max(max_speed, float(np.linalg.norm(d.qvel[vadr:vadr + 3])))
    return {"zero_action_max_displacement_m": max_disp,
            "zero_action_final_rotation_deg": pilot.quat_orientation_delta_deg(
                q_start, d.qpos[qadr + 3:qadr + 7]),
            "zero_action_max_speed_m_s": max_speed}


def settle_diagnostics(inv: export.GridInventory, engine, mujoco
                       ) -> tuple[dict[tuple[str, float, float], dict[str, Any]], list[dict[str, Any]],
                                  dict[str, dict[str, Any]]]:
    starts: dict[tuple[str, float, float], dict[str, Any]] = {}
    rows = []
    for rec in inv.manifest["records"]:
        geom = inv.geometry(rec["object"])
        mass, mu = float(rec["mass_kg"]), float(rec["contact_friction"])
        obj = pilot.build_model(geom.isolated_scene, geom.body_name, mass, mu, engine, mujoco)
        starts[(geom.object_id, mass, mu)] = obj
        vadr = obj["vadr"]
        rows.append({"object_id": geom.object_id, "mass_kg": mass, "contact_friction": mu,
                     "start_x": float(obj["start"][0]), "start_y": float(obj["start"][1]),
                     "start_z": float(obj["start"][2]),
                     "start_linear_speed_m_s": float(np.linalg.norm(obj["v0"][vadr:vadr + 3])),
                     "start_angular_speed_rad_s": float(np.linalg.norm(obj["v0"][vadr + 3:vadr + 6])),
                     "push_x_max_raw_displacement_m": (
                         float(rec["action_metrics"]["push_x"]["max_raw_m"])
                         if "push_x" in rec.get("action_metrics", {}) else None),
                     **zero_action_control(obj, engine, mujoco)})
    per_geom: dict[str, dict[str, Any]] = {}
    for geom in inv.geometries:
        keys = [k for k in starts if k[0] == geom.object_id]
        ref = starts[(geom.object_id, 0.3, 0.3)] if (geom.object_id, 0.3, 0.3) in starts \
            else starts[keys[len(keys) // 2]]
        qadr = ref["qadr"]
        ref_q = ref["q0"][qadr + 3:qadr + 7]
        g_rows = [r for r in rows if r["object_id"] == geom.object_id]
        offset = max(float(np.linalg.norm(starts[k]["start"] - ref["start"])) for k in keys)
        rotation = max(pilot.quat_orientation_delta_deg(starts[k]["q0"][qadr + 3:qadr + 7], ref_q)
                       for k in keys)
        mass_offset = max(float(np.linalg.norm(starts[a]["start"] - starts[b]["start"]))
                          for a, b in itertools.combinations(keys, 2) if np.isclose(a[2], b[2]))
        lin = max(r["start_linear_speed_m_s"] for r in g_rows)
        ang = max(r["start_angular_speed_rad_s"] for r in g_rows)
        drift = max(r["zero_action_max_displacement_m"] for r in g_rows)
        clean = (offset < CLEAN_MAX_OFFSET_M and rotation < CLEAN_MAX_ROTATION_DEG
                 and lin < CLEAN_MAX_LINEAR_SPEED and ang < CLEAN_MAX_ANGULAR_SPEED)
        per_geom[geom.object_id] = {
            "start_offset_across_friction_m": offset, "start_rotation_across_friction_deg": rotation,
            "start_offset_across_mass_m": mass_offset,
            "max_start_linear_speed_m_s": lin, "max_start_angular_speed_rad_s": ang,
            "max_zero_action_displacement_m": drift,
            "max_zero_action_rotation_deg": max(r["zero_action_final_rotation_deg"] for r in g_rows),
            "classification": "clean" if clean else "settle_confounded"}
    return starts, rows, per_geom


# ---------------------------------------------------------------------------
# Part 2: PHINO similarity audit
# ---------------------------------------------------------------------------
def similarity_audit(export_dir: Path, starts, normalizer_cls, classes: dict[str, str],
                     seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = {}
    for split in export.SPLITS:
        meta = json.loads((export_dir / split / "metadata.json").read_text(encoding="utf-8"))
        fps = [export._load_npz(export_dir / split / e["fingerprint_path"])["flat_trajectory"]
               for e in meta]
        q0s = []
        for e in meta:
            obj = starts[(e["asset_id"], float(e["mass"]), float(e["static_friction"]))]
            q0s.append(obj["q0"][obj["qadr"] + 3:obj["qadr"] + 7])
        splits[split] = (meta, fps, q0s)

    meta_tr, fps_tr, q0_tr = splits["train"]
    shipped = normalizer_cls.load(str(export_dir / "train" / "traj_norm.npz"))
    norm_o = normalizer_cls(n_actions=4, timesteps=50).fit(fps_tr, rng=np.random.default_rng(seed))
    if not (np.allclose(norm_o.mean, shipped.mean) and np.allclose(norm_o.std, shipped.std)
            and np.isclose(norm_o.sigma, shipped.sigma)):
        raise RuntimeError("Refit normalizer does not reproduce the exported traj_norm.npz")
    rel = {s: [start_relative_fingerprint(f, q) for f, q in zip(fps, q0s)]
           for s, (_, fps, q0s) in splits.items()}
    for s, (_, fps, q0s) in splits.items():
        for f, q, r in zip(fps, q0s, rel[s]):
            back = swap_settle_orientation(f, q, q)
            if not np.allclose(back, f, atol=1e-5):
                raise RuntimeError("Settle-swap identity check failed")
    norm_r = normalizer_cls(n_actions=4, timesteps=50).fit(rel["train"],
                                                            rng=np.random.default_rng(seed))

    pair_rows, summary = [], {"normalizer_reproduced": True,
                              "sigma_original": float(norm_o.sigma),
                              "sigma_start_relative": float(norm_r.sigma)}
    for split, (meta, fps, q0s) in splits.items():
        so, sr, cats = [], [], []
        for i, j in itertools.combinations(range(len(meta)), 2):
            a, b = meta[i]["asset_id"], meta[j]["asset_id"]
            cat = ("cross_geometry" if a != b else f"within_{classes[a]}")
            s_o = norm_o.similarity(fps[i], fps[j])
            s_r = norm_r.similarity(rel[split][i], rel[split][j])
            row = {"split": split, "i": i, "j": j, "asset_i": a, "asset_j": b, "category": cat,
                   "mass_i": meta[i]["mass"], "mass_j": meta[j]["mass"],
                   "friction_i": meta[i]["static_friction"], "friction_j": meta[j]["static_friction"],
                   "sim_original": s_o, "sim_start_relative": s_r, "sim_settle_swap_only": None}
            if a == b:
                cf = swap_settle_orientation(fps[i], q0s[i], q0s[j])
                row["sim_settle_swap_only"] = norm_o.similarity(fps[i], cf)
            pair_rows.append(row)
            so.append(s_o)
            sr.append(s_r)
            cats.append(cat)
        so, sr, cats = np.array(so), np.array(sr), np.array(cats)
        per_cat = {}
        for cat in sorted(set(cats)):
            sel = cats == cat
            rows_c = [r for r in pair_rows if r["split"] == split and r["category"] == cat]
            swap = np.array([r["sim_settle_swap_only"] for r in rows_c
                             if r["sim_settle_swap_only"] is not None
                             and r["friction_i"] != r["friction_j"]])
            actual = np.array([r["sim_original"] for r in rows_c
                               if r["sim_settle_swap_only"] is not None
                               and r["friction_i"] != r["friction_j"]])
            entry = {"pairs": int(sel.sum()),
                     "mean_abs_change": float(np.abs(so[sel] - sr[sel]).mean()),
                     "max_abs_change": float(np.abs(so[sel] - sr[sel]).max()),
                     "rank_corr": rank_corr(so[sel], sr[sel]) if sel.sum() > 2 else None}
            if swap.size:
                # Distances in PHINO's normalized-MSE units: mse = -sigma * log(sim).
                d_swap = -norm_o.sigma * np.log(np.clip(swap, 1e-12, 1))
                d_act = -norm_o.sigma * np.log(np.clip(actual, 1e-12, 1))
                entry.update({
                    "friction_pairs": int(swap.size),
                    "settle_swap_only_similarity_min": float(swap.min()),
                    "settle_swap_only_similarity_mean": float(swap.mean()),
                    "median_settle_distance_share": float(np.median(d_swap / np.maximum(d_act, 1e-12))),
                })
            per_cat[cat] = entry
        summary[split] = {"pairs": int(len(so)), "rank_corr": rank_corr(so, sr),
                          "pearson": float(np.corrcoef(so, sr)[0, 1]),
                          "mean_abs_change": float(np.abs(so - sr).mean()), "by_category": per_cat}
    per_geometry = {}
    for asset in sorted({r["asset_i"] for r in pair_rows if r["sim_settle_swap_only"] is not None}):
        sel = [r for r in pair_rows if r["asset_i"] == asset and r["asset_j"] == asset
               and r["friction_i"] != r["friction_j"]]
        if sel:
            swap = np.array([r["sim_settle_swap_only"] for r in sel])
            act = np.array([r["sim_original"] for r in sel])
            per_geometry[asset] = {
                "class": classes[asset], "friction_pairs": len(sel),
                "settle_swap_only_similarity_mean": float(swap.mean()),
                "settle_swap_only_similarity_min": float(swap.min()),
                "actual_similarity_mean": float(act.mean()),
                "median_settle_distance_share": float(np.median(
                    np.log(np.clip(swap, 1e-12, 1)) / np.minimum(np.log(np.clip(act, 1e-12, 1)), -1e-12))),
            }
    summary["settle_swap_by_geometry"] = per_geometry
    return summary, pair_rows


# ---------------------------------------------------------------------------
# Part 3: ridge audit
# ---------------------------------------------------------------------------
def _mae(preds: list[dict[str, Any]], target: str, model: str) -> float | None:
    if not preds:
        return None
    return float(np.mean([abs(p[f"{target}_actual"] - p[f"{target}_{model}"]) for p in preds]))


def _score_row(scope: str, preds: list[dict[str, Any]], **keys) -> dict[str, Any]:
    out = {"scope": scope, **keys, "n_test": len(preds),
           "n_test_objects": len({p.get("object", p.get("heldout_object")) for p in preds})}
    for target in ("mass_kg", "contact_friction"):
        r, b = _mae(preds, target, "ridge"), _mae(preds, target, "baseline")
        out[f"{target}_ridge_mae"], out[f"{target}_baseline_mae"] = r, b
        out[f"{target}_ridge_beats_baseline"] = None if r is None else bool(r < b)
    return out


def ridge_audit(grid_dir: Path, classes: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, _ = apg.load_dataset(grid_dir)
    published = {}
    pub_path = grid_dir / "recovery_analysis" / "recovery_scores.csv"
    if pub_path.is_file():
        for r in csv.DictReader(open(pub_path)):
            published[(r["split"], r["actions"], r["target"], r["model"])] = float(r["mae"])
    clean = {o for o, c in classes.items() if c == "clean"}
    out: list[dict[str, Any]] = []
    reproduced = []
    for split in ("physics", "object"):
        for actions in apg.ACTION_SETS:
            label = "+".join(actions)
            for feature_set in apg.FEATURE_SETS:
                preds, metrics = apg.evaluate(rows, actions, RIDGE_ALPHA, split, feature_set)
                if feature_set == "original" and published:
                    for target in ("mass_kg", "contact_friction"):
                        for model in ("ridge", "baseline"):
                            mine = metrics["targets"][target][model]["mae"]
                            ref = published.get((split, label, target, model))
                            reproduced.append(ref is not None and np.isclose(mine, ref, atol=1e-12))
                keys = {"split": split, "actions": label, "feature_set": feature_set,
                        "training": "all_7"}
                out.append(_score_row("all", preds, **keys))
                out.append(_score_row("clean_test_objects",
                                      [p for p in preds if p["object"] in clean], **keys))
                out.append(_score_row("confounded_test_objects",
                                      [p for p in preds if p["object"] not in clean], **keys))
                clean_rows = [r for r in rows if r["object"] in clean]
                try:
                    cpreds, _ = apg.evaluate(clean_rows, actions, RIDGE_ALPHA, split, feature_set)
                    out.append(_score_row("clean_only", cpreds, **{**keys, "training": "clean_only"}))
                except ValueError as exc:
                    out.append({"scope": "clean_only", **keys, "training": "clean_only",
                                "error": str(exc)})
    sphere, sp_preds, _, _, _ = apg.compare_sphere_training(rows, RIDGE_ALPHA)
    for feature_set in apg.FEATURE_SETS:
        for actions in apg.ACTION_SETS:
            label = "+".join(actions)
            for mode in ("without_sphere", "with_sphere"):
                base = [p for p in sp_preds if p["feature_set"] == feature_set
                        and p["actions"] == label and p["training_mode"] == mode]
                keys = {"split": "sphere_matched_object", "actions": label,
                        "feature_set": feature_set, "training": mode}
                out.append(_score_row("all", base, **keys))
                out.append(_score_row("clean_test_objects",
                                      [p for p in base if p["heldout_object"] in clean], **keys))
                out.append(_score_row("confounded_test_objects",
                                      [p for p in base if p["heldout_object"] not in clean], **keys))
    return out, {"published_scores_reproduced": bool(reproduced) and all(reproduced),
                 "n_reproduced_cells": len(reproduced), "clean_geometries": sorted(clean),
                 "sphere": sphere}


# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, default=Path("results/phino_mesh_grid_cutoff_1m_v1"))
    ap.add_argument("--export-dir", type=Path, default=Path("data/phino_simpact_v0"))
    ap.add_argument("--phino-root", type=Path, default=export.DEFAULT_PHINO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("results/phino_settle_audit_v1"))
    ap.add_argument("--seed", type=int, default=0, help="Must match the export's --seed")
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    if out.exists():
        sys.exit(f"Output directory already exists: {out}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    inv = export.load_grid(args.grid_dir)
    engine_mod = export.load_phino_module(args.phino_root, "train.data.mujoco_engine")
    engine = engine_mod.MuJoCoPhysicsEngine()

    starts, settle_rows, per_geom = settle_diagnostics(inv, engine, mujoco)
    classes = {g: v["classification"] for g, v in per_geom.items()}
    sim_summary, sim_rows = similarity_audit(args.export_dir.resolve(), starts,
                                             engine_mod.TrajectoryNormalizer, classes, args.seed)
    ridge_rows, ridge_info = ridge_audit(inv.grid_dir, classes)

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    try:
        _write_csv(stage / "settle_diagnostics.csv", settle_rows)
        _write_csv(stage / "phino_similarity_pairs.csv", sim_rows)
        _write_csv(stage / "ridge_audit.csv", ridge_rows)
        export._write_json(stage / "summary.json", {
            "format": "simpact_phino_settle_audit_v1", "command": sys.argv,
            "simpact_git": export._git_state(Path(__file__).resolve().parent.parent),
            "phino_git": export._git_state(args.phino_root.expanduser().resolve()),
            "source_grid": str(inv.grid_dir), "manifest_sha256": inv.manifest_sha256,
            "export_dir": str(args.export_dir.resolve()),
            "clean_thresholds": {"offset_m": CLEAN_MAX_OFFSET_M,
                                 "rotation_deg": CLEAN_MAX_ROTATION_DEG,
                                 "linear_speed_m_s": CLEAN_MAX_LINEAR_SPEED,
                                 "angular_speed_rad_s": CLEAN_MAX_ANGULAR_SPEED},
            "settle_by_geometry": per_geom, "phino_similarity": sim_summary,
            "ridge": ridge_info,
        })
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    for g, v in per_geom.items():
        print(f"{g}: {v['classification']} offset={1000 * v['start_offset_across_friction_m']:.2f}mm "
              f"rot={v['start_rotation_across_friction_deg']:.2f}deg "
              f"drift={1000 * v['max_zero_action_displacement_m']:.2f}mm")
    print(f"PUBLISHED_RIDGE_REPRODUCED={ridge_info['published_scores_reproduced']} "
          f"({ridge_info['n_reproduced_cells']} cells)")
    print(f"OUTPUT={out}")
    print("SETTLE_AUDIT_OK")


if __name__ == "__main__":
    main()
