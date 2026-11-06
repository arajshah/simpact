#!/usr/bin/env python3
"""Compare trajectory-similarity fine-tuning with the two matched baselines.

Reads completed probe ``results.json`` files. It does not train or extract
features. Intervention subsets of a given size share configurations and
overlap in actions, so they are averaged only as a description. Paired
summaries count held-out geometries, not subsets.

    python scripts/phino_action_baseline_report.py \
        --pooled-new results/phino_action_baselines_v1/probes_pooled/results.json \
        --grid-new results/phino_action_baselines_v1/probes_grid/results.json \
        --output-dir docs/results/phino_action_baselines_v1
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import phino_action_probe as probe

FULL = "+".join(probe.ACTIONS)
SUBSET_DEPENDENCE_NOTE = (
    "Intervention subsets share the same configurations and overlap in actions. "
    "Averages over subsets are descriptive. The paired summary's unit is the "
    "held-out geometry. Subset counts are not a sample size, and this report "
    "does not treat them as independent draws."
)
GEOM_LABEL = {
    "blue_milk_carton": "blue carton",
    "mesh_cow": "cow",
    "mesh_cube": "cube",
    "mesh_nut": "nut",
    "white_coconut_milk_carton": "white carton",
}
# Published 4x4 diagnostic. n=2 and n=3 are single representative subsets.
PUBLISHED_GRID_SUBSETS = (
    "push_x_v2",
    "push_lo_1.6N",
    "push_hi_3.2N",
    "toss_4.0N",
    "push_lo_1.6N+push_hi_3.2N",
    "push_hi_3.2N+toss_4.0N",
    "push_lo_1.6N+push_hi_3.2N+toss_4.0N",
    FULL,
)
COMPARISONS = (
    ("trajectory", "instance", "trajectory_minus_instance"),
    ("trajectory", "regression", "trajectory_minus_regression"),
    ("instance", "regression", "instance_minus_regression"),
    ("trajectory", "frozen", "trajectory_minus_frozen"),
)
METHOD_ORDER = ("frozen", "trajectory", "instance", "regression", "tracker")
METHOD_LABEL = {
    "frozen": "frozen DINOv2",
    "trajectory": "trajectory similarity",
    "instance": "generic InfoNCE",
    "regression": "supervised regression",
    "tracker": "image-position tracker",
}


def all_action_subsets() -> list[str]:
    names = []
    for n in range(1, len(probe.ACTIONS) + 1):
        for subset in itertools.combinations(probe.ACTIONS, n):
            names.append("+".join(subset))
    return names


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    indexed = {}
    for row in rows:
        key = (row["representation"], row["interventions"], row["split"], row["target"])
        if key in indexed:
            raise ValueError(f"duplicate probe row {key}")
        indexed[key] = row
    return indexed


def merge_rows(*groups: list[dict]) -> dict[tuple, dict]:
    """Combine probe files. The same row may recur; its nMAE must agree."""
    indexed: dict[tuple, dict] = {}
    for rows in groups:
        for row in rows:
            key = (row["representation"], row["interventions"], row["split"], row["target"])
            if key in indexed:
                if abs(float(indexed[key]["nmae"]) - float(row["nmae"])) > 1e-6:
                    raise ValueError(f"conflicting nMAE for {key}")
                continue
            indexed[key] = row
    return indexed


def per_geometry_subset_mean(rows: list[dict]) -> dict[str, float]:
    """Mean nMAE per geometry across the given subset rows.

    The returned dict has one entry per geometry. ``len(rows)`` is how many
    dependent subsets were averaged and is not a sample size.
    """
    if not rows:
        raise ValueError("no subset rows")
    folds = [set(row["per_fold"]) for row in rows]
    if any(fold != folds[0] for fold in folds):
        raise ValueError("subset rows cover different geometries")
    out = {}
    for geom in sorted(folds[0]):
        out[geom] = float(np.mean([row["per_fold"][geom]["nmae"] for row in rows]))
    return out


def paired_summary(delta_s0: dict[str, float], delta_s1: dict[str, float]) -> dict:
    """Summarize seed-wise per-geometry differences.

    A negative delta means the first method has lower nMAE. Geometries are the
    only unit that is counted.
    """
    if set(delta_s0) != set(delta_s1) or not delta_s0:
        raise ValueError("the two seeds must cover the same non-empty geometry set")
    geoms = sorted(delta_s0)
    averaged = {g: float(0.5 * (delta_s0[g] + delta_s1[g])) for g in geoms}
    values = np.array([averaged[g] for g in geoms], dtype=np.float64)
    seed_gap = np.array([abs(delta_s0[g] - delta_s1[g]) for g in geoms], dtype=np.float64)
    return {
        "n_geometries": len(geoms),
        "mean_seed_averaged_delta": float(values.mean()),
        "min_seed_averaged_delta": float(values.min()),
        "max_seed_averaged_delta": float(values.max()),
        "per_geometry_seed_averaged": averaged,
        "per_geometry_seed0": {g: float(delta_s0[g]) for g in geoms},
        "per_geometry_seed1": {g: float(delta_s1[g]) for g in geoms},
        "n_both_seeds_negative": int(sum(delta_s0[g] < 0 and delta_s1[g] < 0 for g in geoms)),
        "n_both_seeds_positive": int(sum(delta_s0[g] > 0 and delta_s1[g] > 0 for g in geoms)),
        "n_sign_disagreement": int(sum(delta_s0[g] * delta_s1[g] < 0 for g in geoms)),
        "mean_abs_seed_difference": float(seed_gap.mean()),
        "max_abs_seed_difference": float(seed_gap.max()),
    }


def _require(indexed: dict, key: tuple) -> dict:
    if key not in indexed:
        raise KeyError(f"missing probe row {key}")
    row = indexed[key]
    if not np.isfinite(row["nmae"]):
        raise ValueError(f"non-finite nMAE for {key}")
    for cell in row["per_fold"].values():
        if not np.isfinite(cell["nmae"]):
            raise ValueError(f"non-finite per-geometry nMAE for {key}")
    return row


def _assert_same_counts(reference: dict, other: dict, what: str) -> None:
    if reference["n"] != other["n"]:
        raise ValueError(f"{what} evaluates a different number of samples")
    for geom, cell in reference["per_fold"].items():
        if geom not in other["per_fold"] or other["per_fold"][geom]["n"] != cell["n"]:
            raise ValueError(f"{what} fold counts differ at {geom}")


def _method_reps(readout: str) -> dict[str, tuple[str, ...]]:
    prefix = "" if readout == "pooled" else "grid_"
    return {
        "trajectory": (f"{prefix}ft_clip_s0", f"{prefix}ft_clip_s1"),
        "instance": (f"{prefix}ft_instance_s0", f"{prefix}ft_instance_s1"),
        "regression": (f"{prefix}ft_regression_s0", f"{prefix}ft_regression_s1"),
        "frozen": (f"{prefix}dinov2_frozen",),
        "tracker": ("image_position",),
    }


def _deltas(indexed: dict, reps: dict[str, tuple[str, ...]], left: str, right: str,
            subset: str, split: str, target: str, rows_for_left: list[dict] | None = None,
            rows_for_right: list[dict] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Per-seed geometry deltas. A one-seed method (frozen, tracker) is reused for both seeds."""
    def side(method: str, explicit: list[dict] | None) -> list[dict]:
        if explicit is not None:
            return explicit
        return [_require(indexed, (rep, subset, split, target)) for rep in reps[method]]

    left_rows = side(left, rows_for_left)
    right_rows = side(right, rows_for_right)
    if len(left_rows) == 1:
        left_rows = [left_rows[0], left_rows[0]]
    if len(right_rows) == 1:
        right_rows = [right_rows[0], right_rows[0]]
    if len(left_rows) != 2 or len(right_rows) != 2:
        raise ValueError("expected one or two seeds")
    out = []
    for a, b in zip(left_rows, right_rows):
        _assert_same_counts(a, b, f"{left} vs {right}")
        folds_a = per_geometry_subset_mean([a])
        folds_b = per_geometry_subset_mean([b])
        out.append({g: folds_a[g] - folds_b[g] for g in folds_a})
    return out[0], out[1]


def _subset_groups(names: list[str]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for name in names:
        n = name.count("+") + 1
        groups.setdefault(n, []).append(name)
    return dict(sorted(groups.items()))


def _scaling_block(indexed: dict, reps: dict[str, tuple[str, ...]], subsets: list[str], split: str) -> dict:
    groups = _subset_groups(subsets)
    block = {}
    for n, names in groups.items():
        entry: dict = {"n_dependent_subsets": len(names), "subsets": names, "targets": {}}
        geom_by_method_seed: dict[tuple, dict] = {}
        weighted: dict[str, dict] = {}
        for target in probe.TARGETS:
            weighted[target] = {}
            for method, rep_names in reps.items():
                seed_means = []
                seed_geoms = []
                for rep in rep_names:
                    rows = [_require(indexed, (rep, name, split, target)) for name in names]
                    if len(rows) > 1:
                        for row in rows[1:]:
                            _assert_same_counts(rows[0], row, f"{rep} {target}")
                    seed_means.append(float(np.mean([row["nmae"] for row in rows])))
                    seed_geoms.append(per_geometry_subset_mean(rows))
                if len(seed_means) == 1:
                    seed_means = [seed_means[0], seed_means[0]]
                    seed_geoms = [seed_geoms[0], seed_geoms[0]]
                geom_avg = {g: float(0.5 * (seed_geoms[0][g] + seed_geoms[1][g])) for g in seed_geoms[0]}
                geom_by_method_seed[(method, target)] = seed_geoms
                weighted[target][method] = {
                    "seed0": seed_means[0],
                    "seed1": seed_means[1],
                    "seed_mean": float(0.5 * (seed_means[0] + seed_means[1])),
                    "geometry_equal_mean": float(np.mean(list(geom_avg.values()))),
                    "per_geometry_seed_averaged": geom_avg,
                }
            entry["targets"][target] = weighted[target]
        paired = {}
        for target in probe.TARGETS:
            paired[target] = {}
            for left, right, name in COMPARISONS:
                d0 = {g: geom_by_method_seed[(left, target)][0][g] - geom_by_method_seed[(right, target)][0][g]
                      for g in geom_by_method_seed[(left, target)][0]}
                d1 = {g: geom_by_method_seed[(left, target)][1][g] - geom_by_method_seed[(right, target)][1][g]
                      for g in d0}
                paired[target][name] = paired_summary(d0, d1)
        entry["paired_after_within_geometry_subset_average"] = paired
        block[str(n)] = entry
    return block


def _full_detail(indexed: dict, reps: dict[str, tuple[str, ...]], split: str) -> dict:
    detail = {}
    for target in probe.TARGETS:
        per_seed = {}
        weighted = {}
        for method, rep_names in reps.items():
            per_seed[method] = {}
            weighted[method] = {}
            for s, rep in enumerate(rep_names):
                row = _require(indexed, (rep, FULL, split, target))
                per_seed[method][f"seed{s}"] = {g: float(cell["nmae"]) for g, cell in row["per_fold"].items()}
                weighted[method][f"seed{s}"] = float(row["nmae"])
            if len(rep_names) == 1:
                per_seed[method]["seed1"] = per_seed[method]["seed0"]
                weighted[method]["seed1"] = weighted[method]["seed0"]
        ref = _require(indexed, (reps["trajectory"][0], FULL, split, target))
        for method, rep_names in reps.items():
            for rep in rep_names:
                _assert_same_counts(ref, _require(indexed, (rep, FULL, split, target)), method)
        paired = {}
        for left, right, name in COMPARISONS:
            d0, d1 = _deltas(indexed, reps, left, right, FULL, split, target)
            paired[name] = paired_summary(d0, d1)
        detail[target] = {"per_geometry": per_seed, "sample_weighted_nmae": weighted, "paired": paired}
    return detail


def _check_tracker(previous: dict, new_rows: list[dict], subsets: list[str]) -> None:
    old = index_rows(previous["results"])
    new = index_rows(new_rows)
    checked = 0
    for subset in subsets:
        for split in ("logo", "within"):
            for target in probe.TARGETS:
                key = ("image_position", subset, split, target)
                if key not in new:
                    continue
                if key not in old:
                    raise KeyError(f"new tracker row has no previous counterpart: {key}")
                if abs(old[key]["nmae"] - new[key]["nmae"]) > 1e-6:
                    raise ValueError(
                        f"image-position nMAE changed for {key}: {old[key]['nmae']} vs {new[key]['nmae']}"
                    )
                checked += 1
    if checked == 0:
        raise ValueError("new probe file has no image-position rows to check against the previous run")


def assemble(pooled_previous: dict, pooled_new: dict, grid_frozen: dict, grid_previous: dict,
             grid_new: dict) -> dict:
    unknown = set(pooled_new["configs_per_geometry"]) - set(GEOM_LABEL)
    if unknown:
        raise ValueError(f"no display label for {sorted(unknown)}")
    if pooled_previous["n_configs"] != pooled_new["n_configs"]:
        raise ValueError("pooled probe files cover different configurations")
    if grid_previous["n_configs"] != grid_new["n_configs"] or grid_frozen["n_configs"] != grid_new["n_configs"]:
        raise ValueError("4x4 probe files cover different configurations")
    pooled_index = merge_rows(pooled_previous["results"], pooled_new["results"])
    grid_index = merge_rows(grid_frozen["results"], grid_previous["results"], grid_new["results"])
    pooled_subsets = all_action_subsets()
    for name in pooled_subsets:
        _require(pooled_index, ("ft_clip_s0", name, "logo", "mass"))
        _require(pooled_index, ("ft_instance_s0", name, "logo", "mass"))
        _require(pooled_index, ("ft_regression_s0", name, "logo", "mass"))
    for name in PUBLISHED_GRID_SUBSETS:
        _require(grid_index, ("grid_ft_clip_s0", name, "logo", "mass"))
        _require(grid_index, ("grid_ft_instance_s0", name, "logo", "mass"))
        _require(grid_index, ("grid_ft_regression_s0", name, "logo", "mass"))
    _check_tracker(pooled_previous, pooled_new["results"], pooled_subsets)
    _check_tracker(grid_previous, grid_new["results"], list(PUBLISHED_GRID_SUBSETS))
    readouts = {}
    for readout, index, subsets in (
        ("pooled", pooled_index, pooled_subsets),
        ("grid4x4", grid_index, list(PUBLISHED_GRID_SUBSETS)),
    ):
        reps = _method_reps(readout)
        readouts[readout] = {
            "subsets": subsets,
            "logo_full": _full_detail(index, reps, "logo"),
            "within_full": _full_detail(index, reps, "within"),
            "logo_by_count": _scaling_block(index, reps, subsets, "logo"),
            "within_by_count": _scaling_block(index, reps, subsets, "within"),
            "position_decoding_logo_r2": {
                "previous_trajectory_seed0": (grid_previous if readout == "grid4x4" else pooled_previous)
                .get("position_decoding_logo_r2", {}).get(reps["trajectory"][0]),
                "previous_trajectory_seed1": (grid_previous if readout == "grid4x4" else pooled_previous)
                .get("position_decoding_logo_r2", {}).get(reps["trajectory"][1]),
                "instance_seed0": (grid_new if readout == "grid4x4" else pooled_new)
                .get("position_decoding_logo_r2", {}).get(reps["instance"][0]),
                "instance_seed1": (grid_new if readout == "grid4x4" else pooled_new)
                .get("position_decoding_logo_r2", {}).get(reps["instance"][1]),
                "regression_seed0": (grid_new if readout == "grid4x4" else pooled_new)
                .get("position_decoding_logo_r2", {}).get(reps["regression"][0]),
                "regression_seed1": (grid_new if readout == "grid4x4" else pooled_new)
                .get("position_decoding_logo_r2", {}).get(reps["regression"][1]),
            },
        }
    return {
        "note": SUBSET_DEPENDENCE_NOTE,
        "n_configs": pooled_new["n_configs"],
        "configs_per_geometry": pooled_new["configs_per_geometry"],
        "primary_endpoint": (
            "Leave-one-geometry-out normalized MAE at all four interventions. "
            "One intervention set, so no subset average enters the primary paired differences."
        ),
        "grid_count_caveat": (
            "The 4x4 curves use the published nine-subset list. Count 1 is every single "
            "action and count 4 is the full set. Count 2 averages two representative "
            "subsets (push_lo+push_hi and push_hi+toss), and count 3 is the single "
            "subset push_lo+push_hi+toss. Those are not every combination."
        ),
        "fairness": [
            "Generic InfoNCE uses the same clips, shared-frame colour jitter and translation, batch of 24, 400 steps, backbone learning rate 1e-5, head learning rate 1e-3, weight decay 0.05, 40-step warmup, and last four trainable blocks as trajectory-similarity SoftInfoNCE. Its temperature is the same learnable parameter, initialized at 0.07. It does not read a trajectory similarity.",
            "Supervised regression uses those same training choices and the same two augmented views per step. Its target is log mass and log friction, standardized with the mean and population standard deviation of the training geometries only. It sees the probe's label family during training. Trajectory similarity and generic InfoNCE do not. The regression head has the same hidden width as the contrastive head, is not L2-normalized, and is discarded before the probe.",
            "Every learned run trains on one intervention per batch. The probe concatenates one to four interventions afterwards. Regression therefore asks every single intervention, including a friction-blind toss, to predict both properties.",
            "No learning rate, step count, augmentation, or ridge penalty was chosen using a test geometry. The ridge penalty is chosen by grouped cross-validation inside the training geometries only.",
            "Sample-weighted nMAE gives cow, cube, and nut most of the weight. White carton has six configurations. The paired differences give each held-out geometry one vote.",
            "The trajectory-similarity checkpoints from the earlier pilot were not retrained. The image-position tracker was recomputed with the new probe files and checked against that pilot.",
        ],
        "readouts": readouts,
    }


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _pair_line(summary: dict) -> str:
    return (
        f"{summary['mean_seed_averaged_delta']:+.3f} "
        f"(geometry range {summary['min_seed_averaged_delta']:+.3f} to {summary['max_seed_averaged_delta']:+.3f}; "
        f"both seeds lower {summary['n_both_seeds_negative']}/{summary['n_geometries']}; "
        f"sign disagreement {summary['n_sign_disagreement']}/{summary['n_geometries']}; "
        f"mean |seed difference| {summary['mean_abs_seed_difference']:.3f})"
    )


def render_markdown(report: dict) -> str:
    counts = report["configs_per_geometry"]
    geoms = [g for g in GEOM_LABEL if g in counts]
    lines = [
        "# Trajectory-similarity baselines",
        "",
        report["note"],
        "",
        report["primary_endpoint"],
        "",
        report["grid_count_caveat"],
        "",
        "Fairness limitations:",
        "",
    ]
    for item in report["fairness"]:
        lines.append(f"- {item}")
    lines += [
        "",
        f"Configurations: {report['n_configs']}.",
        "",
    ]
    for readout, title in (("pooled", "Pooled CLS‖mean-patch"), ("grid4x4", "4×4 spatial readout")):
        block = report["readouts"][readout]
        lines.append(f"## {title}")
        lines.append("")
        lines.append("Sample-weighted leave-one-geometry-out nMAE, seed-averaged. "
                     "For pooled features, each count averages every subset of that size. "
                     "Those subsets are dependent. On the 4×4 readout, count 2 averages the two "
                     "published representative subsets and count 3 is one representative subset.")
        lines.append("")
        header = ["method", "1", "2", "3", "4"]
        rows = []
        for method in METHOD_ORDER:
            cells = []
            for n in ("1", "2", "3", "4"):
                mass = block["logo_by_count"][n]["targets"]["mass"][method]["seed_mean"]
                friction = block["logo_by_count"][n]["targets"]["friction"][method]["seed_mean"]
                cells.append(f"{mass:.2f} / {friction:.2f}")
            rows.append([METHOD_LABEL[method], *cells])
        lines.append(_md_table(header, rows))
        lines.append("")
        lines.append("Primary endpoint: all four interventions, per held-out geometry. "
                     "Each learned method is seed 0 / seed 1. Frozen and the tracker have one fit.")
        lines.append("")
        for target in probe.TARGETS:
            lines.append(f"### {target}")
            lines.append("")
            header = ["geometry", "n", "frozen", "trajectory", "InfoNCE", "regression", "tracker"]
            rows = []
            detail = block["logo_full"][target]
            for geom in geoms:
                def pair(method: str) -> str:
                    a = detail["per_geometry"][method]["seed0"][geom]
                    b = detail["per_geometry"][method]["seed1"][geom]
                    if method in ("frozen", "tracker"):
                        return _fmt(a)
                    return f"{a:.2f} / {b:.2f}"
                rows.append([
                    GEOM_LABEL[geom], str(counts[geom]),
                    pair("frozen"), pair("trajectory"), pair("instance"), pair("regression"), pair("tracker"),
                ])
            weighted = detail["sample_weighted_nmae"]
            rows.append([
                "sample-weighted", str(report["n_configs"]),
                _fmt(weighted["frozen"]["seed0"]),
                f"{weighted['trajectory']['seed0']:.2f} / {weighted['trajectory']['seed1']:.2f}",
                f"{weighted['instance']['seed0']:.2f} / {weighted['instance']['seed1']:.2f}",
                f"{weighted['regression']['seed0']:.2f} / {weighted['regression']['seed1']:.2f}",
                _fmt(weighted["tracker"]["seed0"]),
            ])
            lines.append(_md_table(header, rows))
            lines.append("")
            lines.append("Paired geometry differences at this endpoint (first method minus second; "
                         "negative means the first has lower error). Each geometry counts once.")
            lines.append("")
            for _left, _right, name in COMPARISONS:
                lines.append(f"- {name}: {_pair_line(detail['paired'][name])}")
            lines.append("")
        lines.append("Within-geometry held-out physics, all four interventions, sample-weighted nMAE "
                     "(seed 0 / seed 1 where two seeds exist):")
        lines.append("")
        header = ["method", "mass", "friction"]
        rows = []
        within = block["within_full"]
        for method in METHOD_ORDER:
            mass = within["mass"]["sample_weighted_nmae"][method]
            friction = within["friction"]["sample_weighted_nmae"][method]
            if method in ("frozen", "tracker"):
                rows.append([METHOD_LABEL[method], _fmt(mass["seed0"]), _fmt(friction["seed0"])])
            else:
                rows.append([
                    METHOD_LABEL[method],
                    f"{mass['seed0']:.2f} / {mass['seed1']:.2f}",
                    f"{friction['seed0']:.2f} / {friction['seed1']:.2f}",
                ])
        lines.append(_md_table(header, rows))
        lines.append("")
    lines.append("## Position decoding")
    lines.append("")
    lines.append("Leave-one-geometry-out mean R² of the 15-d image-position target, stored by the probe. "
                 "This is a diagnostic, not the primary endpoint.")
    lines.append("")
    for readout, title in (("pooled", "Pooled"), ("grid4x4", "4×4")):
        decoding = report["readouts"][readout]["position_decoding_logo_r2"]
        lines.append(f"### {title}")
        lines.append("")
        header = ["method"] + list(probe.ACTIONS)
        rows = []
        for key, label in (
            ("previous_trajectory_seed0", "trajectory seed 0"),
            ("previous_trajectory_seed1", "trajectory seed 1"),
            ("instance_seed0", "InfoNCE seed 0"),
            ("instance_seed1", "InfoNCE seed 1"),
            ("regression_seed0", "regression seed 0"),
            ("regression_seed1", "regression seed 1"),
        ):
            block = decoding.get(key)
            if not block:
                continue
            rows.append([label] + [_fmt(block[action]["r2_mean"]) for action in probe.ACTIONS])
        if rows:
            lines.append(_md_table(header, rows))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_json(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"missing probe results: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pooled-new", type=Path, required=True)
    parser.add_argument("--grid-new", type=Path, required=True)
    parser.add_argument("--pooled-previous", type=Path,
                        default=Path("results/phino_action_learning_v2/probes_all/results.json"))
    parser.add_argument("--grid-frozen", type=Path,
                        default=Path("results/phino_action_learning_v2/probes_grid_frozen/results.json"))
    parser.add_argument("--grid-previous", type=Path,
                        default=Path("results/phino_action_learning_v2/probes_grid_ft/results.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        sys.exit(f"output exists, refusing to overwrite: {args.output_dir}")
    report = assemble(
        load_json(args.pooled_previous), load_json(args.pooled_new),
        load_json(args.grid_frozen), load_json(args.grid_previous), load_json(args.grid_new),
    )
    text = render_markdown(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=1))
    (args.output_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
