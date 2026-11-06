#!/usr/bin/env python3
"""Spatial-readout diagnostic from completed PHINO probe outputs.

Reads the existing pooled, frozen-4x4, and fine-tuned-4x4 probe results. It does
not train, extract features, or write into an existing probe directory. Per-geometry
position-decoding R^2 is computed from the saved feature banks because the probe
files store only the pooled R^2.

    python scripts/phino_spatial_diagnostic.py \
        --output-dir docs/results/phino_action_learning_v2/spatial_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import phino_action_probe as probe

ROOT = Path("results/phino_action_learning_v2")
DATA = Path("data/phino_simpact_v3obs_v2")
SOURCES = {
    "pooled": ROOT / "probes_all" / "results.json",
    "grid_frozen": ROOT / "probes_grid_frozen" / "results.json",
    "grid_ft": ROOT / "probes_grid_ft" / "results.json",
}
# Order is the comparison the diagnostic reports. Static fine-tunes are controls.
REPS = (
    ("dinov2_frozen", "pooled", "frozen pooled"),
    ("ft_clip_s0", "pooled", "fine-tuned pooled, seed 0"),
    ("ft_clip_s1", "pooled", "fine-tuned pooled, seed 1"),
    ("grid_dinov2_frozen", "grid_frozen", "frozen 4x4"),
    ("grid_ft_clip_s0", "grid_ft", "fine-tuned 4x4, seed 0"),
    ("grid_ft_clip_s1", "grid_ft", "fine-tuned 4x4, seed 1"),
    ("image_position", "grid_ft", "image-position tracker"),
)
CONTROLS = (
    ("ft_static_s0", "pooled", "static fine-tune, pooled"),
    ("grid_ft_static_s0", "grid_ft", "static fine-tune, 4x4"),
)
DECODERS = {
    "dinov2_frozen": ROOT / "features" / "dinov2_frozen.npz",
    "ft_clip_s0": ROOT / "features" / "ft_clip_s0_{fold}.npz",
    "ft_clip_s1": ROOT / "features" / "ft_clip_s1_{fold}.npz",
    "ft_static_s0": ROOT / "features" / "ft_static_s0_{fold}.npz",
    "grid_dinov2_frozen": ROOT / "features" / "grid_dinov2_frozen.npz",
    "grid_ft_clip_s0": ROOT / "features" / "grid_ft_clip_s0_{fold}.npz",
    "grid_ft_clip_s1": ROOT / "features" / "grid_ft_clip_s1_{fold}.npz",
    "grid_ft_static_s0": ROOT / "features" / "grid_ft_static_s0_{fold}.npz",
}
GEOM_LABEL = {
    "blue_milk_carton": "blue carton",
    "mesh_cow": "cow",
    "mesh_cube": "cube",
    "mesh_nut": "nut",
    "white_coconut_milk_carton": "white carton",
}


def load_results() -> dict[str, dict]:
    out = {}
    for key, path in SOURCES.items():
        if not path.is_file():
            sys.exit(f"missing probe results: {path}")
        out[key] = json.loads(path.read_text())
    return out


def index_rows(data: dict) -> dict[tuple, dict]:
    return {(r["representation"], r["interventions"], r["split"], r["target"]): r for r in data["results"]}


def prediction_lines(results_path: Path) -> int:
    pred = results_path.parent / "predictions.jsonl"
    with pred.open() as fh:
        return sum(1 for _ in fh)


def completeness(loaded: dict[str, dict]) -> dict:
    """The fine-tuned 4x4 job is complete when its rows, folds, and prediction file agree."""
    ft = loaded["grid_ft"]
    frozen = loaded["grid_frozen"]
    ft_sets = {(r["n_interventions"], r["interventions"]) for r in ft["results"] if r["representation"] == "grid_ft_clip_s0"}
    fr_sets = {(r["n_interventions"], r["interventions"]) for r in frozen["results"] if r["representation"] == "grid_dinov2_frozen"}
    tracker_sets = {s for _n, s in ft_sets if s != "static"}
    # Three visual encoders include the static subset; the tracker has no static row.
    expected_rows = 3 * len(ft_sets) * 2 * 2 + len(tracker_sets) * 2 * 2
    finite = all(np.isfinite(r["nmae"]) and all(np.isfinite(v["nmae"]) for v in r["per_fold"].values()) for r in ft["results"])
    folds = {g for r in ft["results"] if r["split"] == "logo" for g in r["per_fold"]}
    n_sum = sum(r["n"] for r in ft["results"])
    n_lines = prediction_lines(SOURCES["grid_ft"])
    geoms = set(ft["configs_per_geometry"])
    report = {
        "n_configs": ft["n_configs"],
        "configs_per_geometry": ft["configs_per_geometry"],
        "encoders": ft["encoders"],
        "n_result_rows": len(ft["results"]),
        "expected_rows_for_this_subset_design": expected_rows,
        "subsets_match_frozen_grid": ft_sets == fr_sets,
        "subsets": [s for _, s in sorted(ft_sets)],
        "all_finite": finite,
        "logo_folds": sorted(folds),
        "folds_cover_geometries": folds == geoms,
        "prediction_rows": n_lines,
        "prediction_rows_match_results": n_lines == n_sum,
        "has_position_decoding": set(ft["position_decoding_logo_r2"]) == {"grid_ft_clip_s0", "grid_ft_clip_s1", "grid_ft_static_s0"},
    }
    if not (report["subsets_match_frozen_grid"] and report["all_finite"] and report["folds_cover_geometries"]
            and report["prediction_rows_match_results"] and report["n_result_rows"] == expected_rows
            and report["has_position_decoding"]):
        sys.exit("fine-tuned 4x4 evaluation is incomplete:\n" + json.dumps(report, indent=1))
    return report


def matched_subsets(loaded: dict[str, dict]) -> list[str]:
    rows = loaded["grid_frozen"]["results"]
    return sorted({r["interventions"] for r in rows if r["representation"] == "grid_dinov2_frozen"},
                  key=lambda s: (0 if s == "static" else len(s.split("+")), s))


def mean_nmae(rows_by_key: dict[tuple, dict], rep: str, split: str, target: str, subsets: list[str]) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for subset in subsets:
        if subset == "static":
            continue
        row = rows_by_key[(rep, subset, split, target)]
        buckets[row["n_interventions"]].append(row["nmae"])
    return {n: float(np.mean(v)) for n, v in sorted(buckets.items())}


def per_geometry_nmae(rows_by_key: dict[tuple, dict], rep: str, split: str, target: str,
                      subsets: list[str]) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for subset in subsets:
        if subset == "static":
            continue
        row = rows_by_key[(rep, subset, split, target)]
        for geom, cell in row["per_fold"].items():
            buckets[row["n_interventions"]][geom].append(cell["nmae"])
    return {n: {g: float(np.mean(v)) for g, v in geoms.items()} for n, geoms in sorted(buckets.items())}


def channel_r2(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Same channel averages as ``position_decoding``: dx, dy, log area, then all 15."""
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != 15:
        raise ValueError(f"expected (n, 15) targets, got {y.shape} and {pred.shape}")
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {"r2_dx": float(np.mean(r2[0:10:2])), "r2_dy": float(np.mean(r2[1:10:2])),
            "r2_log_area": float(np.mean(r2[10:15])), "r2_mean": float(np.mean(r2))}


def varying_channel_r2(y: np.ndarray, pred: np.ndarray, min_std: float = 1e-6) -> dict[str, float]:
    """Per-geometry R², skipping channels with no variation inside that geometry.

    A toss can leave horizontal centroid shift exactly constant for a symmetric object.
    Dividing by that zero variance makes R² explode and is not a decoding result.
    """
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != 15:
        raise ValueError(f"expected (n, 15) targets, got {y.shape} and {pred.shape}")
    keep = y.std(0) > min_std
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    r2 = np.full(y.shape[1], np.nan)
    r2[keep] = 1.0 - ss_res[keep] / ss_tot[keep]

    def avg(idx: range) -> float:
        picked = r2[list(idx)]
        picked = picked[np.isfinite(picked)]
        return float(np.mean(picked)) if picked.size else float("nan")

    return {"r2_dx": avg(range(0, 10, 2)), "r2_dy": avg(range(1, 10, 2)),
            "r2_log_area": avg(range(10, 15)), "r2_mean": avg(range(15)),
            "n_channels": int(keep.sum())}


def action_matrix(bank: probe.FeatureBank, cfgs: list[dict], action: str) -> np.ndarray:
    """Frame-difference features for one action. Matches ``visual_clip`` then float64."""
    idx = np.array([[bank.index[p] for p in c["clips"][action]["images"]] for c in cfgs], dtype=np.int64)
    frames = bank.f[idx]
    diff = frames[:, 1:] - frames[:, :1]
    return np.ascontiguousarray(diff.reshape(len(cfgs), -1), dtype=np.float64)


class _Banks:
    def __init__(self, pattern: Path):
        self.pattern = str(pattern)
        self.cache: dict[str, probe.FeatureBank] = {}

    def get(self, fold: str) -> probe.FeatureBank:
        key = "_shared" if "{fold}" not in self.pattern else fold
        if key not in self.cache:
            path = Path(self.pattern.format(fold=fold))
            print(f"  load {path.name}", flush=True)
            self.cache[key] = probe.FeatureBank(path)
        return self.cache[key]


def decode_encoder(name: str, pattern: Path, cfgs: list[dict], stored: dict[str, dict] | None) -> dict:
    banks = _Banks(pattern)
    geoms = np.array([c["geometry"] for c in cfgs])
    targets = {a: np.array([probe.image_position(c["clips"][a]) for c in cfgs]) for a in probe.ACTIONS}
    pred = {a: np.zeros_like(targets[a]) for a in probe.ACTIONS}
    per_geom: dict[str, dict[str, dict]] = {}
    for fold in sorted(set(geoms)):
        bank = banks.get(fold)
        test, train = geoms == fold, geoms != fold
        per_geom[fold] = {}
        for action in probe.ACTIONS:
            x = action_matrix(bank, cfgs, action)
            pred[action][test] = probe.ridge_path_cv(x[train], targets[action][train], geoms[train], x[test])
            per_geom[fold][action] = varying_channel_r2(targets[action][test], pred[action][test])
            per_geom[fold][action]["n"] = int(test.sum())
        print(f"  {name} fold {fold}", flush=True)
    pooled = {a: channel_r2(targets[a], pred[a]) for a in probe.ACTIONS}
    if stored is not None:
        for action, got in pooled.items():
            ref = stored[action]["r2_mean"]
            if abs(got["r2_mean"] - ref) > 1e-4:
                sys.exit(f"{name} {action} pooled R^2 {got['r2_mean']:.6f} != stored {ref:.6f}")
    return {"pooled": pooled, "per_geometry": per_geom}


def _fmt(x: float) -> str:
    return f"{x:.2f}" if np.isfinite(x) else "–"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, rule, *body])


def write_markdown(path: Path, status: dict, scaling: dict, per_geom: dict, decoding: dict, controls: dict) -> None:
    geoms = list(status["configs_per_geometry"])
    lines = [
        "# Spatial-representation diagnostic",
        "",
        "The fine-tuned 4×4 evaluation in `results/phino_action_learning_v2/probes_grid_ft/` is complete: "
        f"{status['n_result_rows']} result rows, {status['prediction_rows']} predictions, all finite, "
        "same nine intervention subsets as the frozen 4×4 run. This file only reads those outputs and "
        "recomputes per-geometry position R² from the saved features. Pooled position R² matches the stored probe files.",
        "",
        "Subsets behind the n=1..4 means (not every combination; n=2 and n=3 are representative):",
        "",
    ]
    for subset in status["subsets"]:
        n = 0 if subset == "static" else len(subset.split("+"))
        lines.append(f"- n={n}: `{subset}`")
    lines += ["", "Leave-one-geometry-out mean nMAE on those subsets (mass / friction). "
              "n=0 is the shared before image and is omitted here.", ""]
    header = ["representation", "1", "2", "3", "4"]
    rows = []
    for _key, _src, label in REPS:
        cells = []
        for n in (1, 2, 3, 4):
            mass = scaling["logo"][_key]["mass"][str(n)]
            fr = scaling["logo"][_key]["friction"][str(n)]
            cells.append(f"{mass:.2f} / {fr:.2f}")
        rows.append([label, *cells])
    lines.append(_md_table(header, rows))
    lines += ["", "Per held-out geometry, leave-one-geometry-out nMAE. Each cell is the mean over the subsets of that size.", ""]
    for target in ("mass", "friction"):
        for n in (1, 2, 3, 4):
            lines.append(f"### {target}, {n} intervention{'s' if n > 1 else ''}")
            lines.append("")
            header = ["geometry", "n"] + [label for _k, _s, label in REPS]
            rows = []
            for g in geoms:
                row = [GEOM_LABEL[g], str(status["configs_per_geometry"][g])]
                for key, _src, _label in REPS:
                    row.append(_fmt(per_geom["logo"][key][target][str(n)][g]))
                rows.append(row)
            lines.append(_md_table(header, rows))
            lines.append("")
    lines += ["## Position decoding", "",
              "Leave-one-geometry-out R² for the 15-d image-position target (centroid shift and log-area change), "
              "one action at a time. The probe is the same unclipped grouped ridge used in the original evaluation. "
              "Pooled R² matches the stored probe files and keeps every channel. Per-geometry R² uses that geometry's "
              "own mean and drops channels with no within-geometry variation, which happens for some toss components.", ""]
    dec_label = {key: label for key, _src, label in (*REPS, *CONTROLS)}
    header = ["representation"] + list(probe.ACTIONS)
    rows = []
    for name in DECODERS:
        if name not in decoding:
            continue
        rows.append([dec_label[name]] + [_fmt(decoding[name]["pooled"][a]["r2_mean"]) for a in probe.ACTIONS])
    lines.append(_md_table(header, rows))
    lines += ["", "Mean per-geometry R² across the four actions. R² uses each held-out geometry's own mean.", ""]
    header = ["geometry"] + [dec_label[name] for name in DECODERS if name in decoding]
    rows = []
    for g in geoms:
        row = [f"{GEOM_LABEL[g]} ({status['configs_per_geometry'][g]})"]
        for name in DECODERS:
            if name not in decoding:
                continue
            vals = [decoding[name]["per_geometry"][g][a]["r2_mean"] for a in probe.ACTIONS]
            row.append(_fmt(float(np.nanmean(vals))))
        rows.append(row)
    lines.append(_md_table(header, rows))
    if decoding:
        dropped = []
        ref = next(iter(decoding))
        for g in geoms:
            for action in probe.ACTIONS:
                kept = decoding[ref]["per_geometry"][g][action]["n_channels"]
                if kept < 15:
                    dropped.append(f"{GEOM_LABEL[g]} `{action}` keeps {kept}/15")
        if dropped:
            lines += ["", "Channels dropped as constant within the held-out geometry: " + "; ".join(dropped) + "."]
    lines += ["", "Static-image fine-tune control, four interventions, leave-one-geometry-out nMAE (mass / friction):", ""]
    for key, _src, label in CONTROLS:
        mass = controls[key]["mass"]
        fr = controls[key]["friction"]
        lines.append(f"- {label}: {mass:.2f} / {fr:.2f}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--skip-decoding", action="store_true",
                    help="write the nMAE tables only; do not recompute position R^2")
    args = ap.parse_args()
    if args.output_dir.exists():
        sys.exit(f"output exists, refusing to overwrite: {args.output_dir}")
    loaded = load_results()
    status = completeness(loaded)
    subsets = matched_subsets(loaded)
    indexed = {name: index_rows(data) for name, data in loaded.items()}
    catalog = [*REPS, *CONTROLS]
    scaling = {"logo": {}, "within": {}}
    per_geom = {"logo": {}, "within": {}}
    for split in scaling:
        for key, src, _label in catalog:
            scaling[split][key] = {}
            per_geom[split][key] = {}
            for target in probe.TARGETS:
                scaling[split][key][target] = {str(n): v for n, v in mean_nmae(indexed[src], key, split, target, subsets).items()}
                per_geom[split][key][target] = {
                    str(n): geoms for n, geoms in per_geometry_nmae(indexed[src], key, split, target, subsets).items()}
    # Tracker rows agree across probe files on the shared subsets.
    for subset in subsets:
        if subset == "static":
            continue
        for target in probe.TARGETS:
            a = indexed["pooled"][("image_position", subset, "logo", target)]["nmae"]
            b = indexed["grid_ft"][("image_position", subset, "logo", target)]["nmae"]
            if abs(a - b) > 1e-9:
                sys.exit(f"tracker nMAE mismatch on {subset} {target}: {a} vs {b}")
    decoding = {}
    if not args.skip_decoding:
        rows, _ = probe.load_dataset(DATA)
        cfgs = [c for c in probe.configs_table(rows).values() if all(a in c["clips"] for a in probe.ACTIONS)]
        cfgs.sort(key=lambda c: (c["geometry"], c["config"]))
        if len(cfgs) != status["n_configs"]:
            sys.exit(f"config count {len(cfgs)} != probe n_configs {status['n_configs']}")
        for name, pattern in DECODERS.items():
            print(f"decode {name}", flush=True)
            stored = None
            for data in loaded.values():
                block = data.get("position_decoding_logo_r2", {}).get(name)
                if block is not None:
                    stored = block
            decoding[name] = decode_encoder(name, pattern, cfgs, stored)
    full = "push_x_v2+push_lo_1.6N+push_hi_3.2N+toss_4.0N"
    controls = {}
    for key, src, _label in CONTROLS:
        controls[key] = {t: indexed[src][(key, full, "logo", t)]["nmae"] for t in probe.TARGETS}
    payload = {
        "status": status,
        "matched_subsets": subsets,
        "scaling_logo_and_within": scaling,
        "per_geometry_nmae": per_geom,
        "position_decoding": decoding,
        "static_control_nmae_logo_n4": controls,
        "note": "n=2 and n=3 average only the representative subsets listed in status.subsets. "
                "n=1 includes every single action and n=4 is the single full set.",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "diagnostic.json").write_text(json.dumps(payload, indent=1))
    write_markdown(args.output_dir / "summary.md", status, scaling, per_geom, decoding, controls)
    print(f"WROTE {args.output_dir}")


if __name__ == "__main__":
    main()
