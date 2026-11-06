#!/usr/bin/env python3
"""Tables and plots for the action-conditioned PHINO probes.

    python scripts/phino_action_report.py \
        --results results/phino_action_learning_v2/probes_all/results.json \
        --output-dir results/phino_action_learning_v2/report

Normalized MAE (nMAE) = model MAE / MAE of the training-mean predictor on the same
test samples (1.0 = no better than predicting the mean). Scaling curves average
nMAE over every intervention subset of a given size on the SAME configurations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["dinov2_frozen", "ft_clip_s0", "ft_clip_s1", "ft_static_s0", "image_position", "state_physics",
         "state_frames", "state_full"]
LABEL = {"dinov2_frozen": "frozen DINOv2", "ft_clip_s0": "PHINO-ft clips (seed 0)",
         "ft_clip_s1": "PHINO-ft clips (seed 1)", "ft_static_s0": "PHINO-ft static control",
         "image_position": "image-position tracker", "state_physics": "privileged state (physics summary)",
         "state_frames": "privileged state (observed frames)", "state_full": "privileged state (full trajectory)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    res = data["results"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reps = [r for r in ORDER if any(x["representation"] == r for x in res)]

    def get(rep, inter, split, target):
        for x in res:
            if (x["representation"], x["interventions"], x["split"], x["target"]) == (rep, inter, split, target):
                return x
        return None

    # scaling table
    scaling = defaultdict(dict)
    for rep in reps:
        for split in ("logo", "within"):
            for target in ("mass", "friction"):
                by_n = defaultdict(list)
                for x in res:
                    if x["representation"] == rep and x["split"] == split and x["target"] == target:
                        by_n[x["n_interventions"]].append(x["nmae"])
                scaling[rep][f"{split}|{target}"] = {
                    n: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "k": len(v)}
                    for n, v in sorted(by_n.items())}

    # paired fine-tuned vs frozen (per subset x fold)
    paired = {}
    for ft in [r for r in reps if r.startswith("ft_")]:
        for split in ("logo", "within"):
            for target in ("mass", "friction"):
                diffs, folds_better, folds_total = [], 0, 0
                for x in res:
                    if x["representation"] != ft or x["split"] != split or x["target"] != target or x["n_interventions"] == 0:
                        continue
                    fz = get("dinov2_frozen", x["interventions"], split, target)
                    diffs.append(x["nmae"] - fz["nmae"])
                    for g, v in x["per_fold"].items():
                        folds_total += 1
                        folds_better += v["nmae"] < fz["per_fold"][g]["nmae"]
                paired[f"{ft}|{split}|{target}"] = {
                    "mean_delta_nmae_vs_frozen": float(np.mean(diffs)), "median_delta": float(np.median(diffs)),
                    "subsets_improved": int(sum(d < 0 for d in diffs)), "subsets": len(diffs),
                    "fold_cells_improved": folds_better, "fold_cells": folds_total}

    full = "push_x_v2+push_lo_1.6N+push_hi_3.2N+toss_4.0N"
    per_fold_full = {rep: {f"{s}|{t}": (get(rep, full, s, t) or {}).get("per_fold")
                           for s in ("logo", "within") for t in ("mass", "friction")} for rep in reps}
    (args.output_dir / "report.json").write_text(json.dumps(
        {"source": str(args.results), "scaling": scaling, "paired_vs_frozen": paired,
         "per_fold_all_interventions": per_fold_full,
         "position_decoding_logo_r2": data.get("position_decoding_logo_r2"),
         "n_configs": data["n_configs"], "configs_per_geometry": data["configs_per_geometry"]}, indent=1))

    # markdown table: scaling (mean nMAE by number of interventions)
    lines = ["| representation | split | target | n=0 | n=1 | n=2 | n=3 | n=4 |", "|---|---|---|---|---|---|---|---|"]
    for rep in reps:
        for split in ("logo", "within"):
            for target in ("mass", "friction"):
                s = scaling[rep][f"{split}|{target}"]
                cells = [f"{s[n]['mean']:.2f}" if n in s else "–" for n in range(5)]
                lines.append(f"| {LABEL[rep]} | {split} | {target} | " + " | ".join(cells) + " |")
    lines += ["", "| fine-tuned vs frozen | split | target | mean delta nMAE | subsets improved | fold cells improved |",
              "|---|---|---|---|---|---|"]
    for k, v in paired.items():
        ft, split, target = k.split("|")
        lines.append(f"| {LABEL[ft]} | {split} | {target} | {v['mean_delta_nmae_vs_frozen']:+.3f} | "
                     f"{v['subsets_improved']}/{v['subsets']} | {v['fold_cells_improved']}/{v['fold_cells']} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i, target in enumerate(("mass", "friction")):
        for j, split in enumerate(("logo", "within")):
            ax = axes[i, j]
            for rep in reps:
                if rep in ("state_frames", "state_full"):
                    continue
                s = scaling[rep][f"{split}|{target}"]
                ns = sorted(s)
                mean = [s[n]["mean"] for n in ns]
                ax.plot(ns, mean, marker="o", label=LABEL[rep])
                ax.fill_between(ns, [s[n]["min"] for n in ns], [s[n]["max"] for n in ns], alpha=0.08)
            ax.axhline(1.0, color="k", lw=0.6, ls="--")
            ax.set_title(f"{target}, {'leave-one-geometry-out' if split == 'logo' else 'within-geometry held-out physics'}",
                         fontsize=9)
            ax.set_ylim(0, 1.3)
            ax.set_ylabel("normalized MAE (1 = training mean)")
            if i == 1:
                ax.set_xlabel("number of observed interventions (0 = static before image)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Intervention scaling: mean over all subsets of each size (band = min-max); "
                 f"{data['n_configs']} configurations, {len(data['configs_per_geometry'])} geometries", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "intervention_scaling.png", dpi=110)
    plt.close(fig)
    print((args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
