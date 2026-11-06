"""Compare cached PHINO features using the v2 LOGO protocol; never extract or train encoders.

Run from the SIMPACT root. Only the new --out directory is written. Reads compact
results.json, not predictions/logs. Reuses valid cells; evaluates missing ones from
the original feature NPZs. Computes per-geometry position decoding (not saved by
the original evaluator). See the handoff for the CPU-only command.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

import phino_action_probe as probe


def read_summary(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"Incomplete JSON: {path}; ignoring it, leaving it untouched.")
        return {}


def encoder(source, kind, seed):
    for spec in source.get("encoders", []):
        name, path = spec.split("=", 1)
        file = Path(path).name
        if seed is None and "ft_" not in file and "{fold}" not in file:
            if (kind == "grid") == ("grid" in file):
                return name, file
        if seed is not None and file == f"{'grid_' if kind == 'grid' else ''}ft_clip_s{seed}_{{fold}}.npz":
            return name, file
    fallback = ("grid_frozen" if kind == "grid" else "dinov2_frozen") if seed is None else (
        f"{'grid_' if kind == 'grid' else ''}ft_clip_s{seed}_{{fold}}")
    return None, fallback + ".npz"


def fold_targets(configs, fold, subset, rep, bank):
    groups = np.array([c["geometry"] for c in configs])
    test = groups == fold
    x = probe.build_matrix(configs, subset, rep, bank, {})
    y = np.log([[c[t] for t in probe.TARGETS] for c in configs])
    p, _ = probe.fit_predict(x[~test], y[~test], groups[~test], x[test])
    err = np.abs(np.exp(p) - np.exp(y[test])).mean(axis=0)
    base = np.abs(np.exp(y[~test].mean(axis=0)) - np.exp(y[test])).mean(axis=0)
    if not np.all(np.isfinite(err)) or not np.all(base > 0):
        raise ValueError(f"Invalid fold: {fold}")
    return dict(zip(probe.TARGETS, map(float, err/base)))


def position_fold(configs, fold, action, bank):
    groups = np.array([c["geometry"] for c in configs])
    test = groups == fold
    x = probe.build_matrix(configs, (action,), "visual", bank, {})
    y = np.array([probe.image_position(c["clips"][action]) for c in configs])
    pred = probe.ridge_path_cv(x[~test], y[~test], groups[~test], x[test])
    return y[test], pred


def position_scores(y, p, pooled):
    # For the pooled row match the original denominator rule exactly. One fold
    # can have constant image components: report these as undefined, not 0/1.
    sse = ((y-p)**2).sum(axis=0)
    sst = ((y-y.mean(axis=0))**2).sum(axis=0)
    if not np.isfinite(sse).all() or not np.isfinite(sst).all():
        raise ValueError("Non-finite position scores")
    mask = np.ones_like(sst, dtype=bool) if pooled else sst > 1e-12
    r = 1-sse/np.maximum(sst, 1e-12)
    cols = np.arange(len(r))
    result = {}
    for label, selection in (("mean", cols), ("dx", cols[:10:2]), ("dy", cols[1:10:2]), ("log_area", cols[10:])):
        ok = selection[mask[selection]]
        result["r2_"+label] = float(r[ok].mean()) if len(ok) else None
        result["defined_"+label] = len(ok)
    return result


def write_csv(path, rows):
    with path.open("x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/phino_action_learning_v2"))
    ap.add_argument("--data", type=Path, default=probe.DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("--out must be a fresh path")
    manifest = args.data/"samples.jsonl"
    if not manifest.exists():
        ap.error(f"Missing v2 observation index: {manifest}. Run in the Brown checkout.")
    # These visual and tracker representations need only the index, never images or state traces.
    with manifest.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(set(r["sample_id"] for r in rows)) != len(rows):
        raise ValueError("Duplicate sample ids")
    configs = sorted((c for c in probe.configs_table(rows).values()
                      if all(a in c["clips"] for a in probe.ACTIONS)), key=lambda c: (c["geometry"], c["config"]))
    groups = sorted(set(c["geometry"] for c in configs))
    counts = {g: sum(c["geometry"] == g for c in configs) for g in groups}
    if sum(counts.values()) != 292 or len(groups) != 5:
        raise ValueError(f"Unexpected v2 dataset: {counts}")
    pooled, frozen, ft = (read_summary(args.root/d/"results.json") for d in
                          ("probes_all", "probes_grid_frozen", "probes_grid_ft"))
    for source in [pooled, frozen, ft]:
        if source and (source["configs_per_geometry"] != counts
                       or source["n_configs"] != len(configs)
                       or not np.array_equal(source["alphas"], probe.ALPHAS)
                       or source["min_feature_std"] != probe.MIN_FEATURE_STD
                       or source["pred_log_margin"] != probe.PRED_LOG_MARGIN):
            raise ValueError("Cached results use a different cohort or probe protocol")
    # Match the existing frozen-grid design. If a count is missing, use all
    # subsets of that count rather than selecting one based on its score.
    frozen_name, _ = encoder(frozen, "grid", None)
    design = {r["interventions"] for r in frozen.get("results", [])
              if r["split"] == "logo" and r["representation"] == frozen_name}
    subsets = []
    for n in range(1,5):
        choices = list(itertools.combinations(probe.ACTIONS,n))
        subsets.extend([s for s in choices if "+".join(s) in design] or choices)
    for seed in (0,1):
        name, _ = encoder(ft,"grid",seed)
        present = {(r["interventions"],r["target"],g) for r in ft.get("results",[])
                   if r["split"] == "logo" and r["representation"] == name
                   for g,cell in r["per_fold"].items() if g in counts
                   and cell["n"] == counts[g] and np.isfinite(cell["nmae"]) and cell["nmae"] >= 0}
        expected = {("+".join(s),t,g) for s in subsets for t in probe.TARGETS for g in groups}
        print(f"probes_grid_ft seed {seed}: {len(expected & present)}/{len(expected)} "
              "requested LOGO property cells usable",flush=True)
    raw, positions = [], []
    for kind, seed, source in [("pooled",None,pooled),("pooled",0,pooled),("pooled",1,pooled),
                                ("grid",None,frozen),("grid",0,ft),("grid",1,ft),("tracker",None,pooled)]:
        original, file = encoder(source, kind, seed)
        tracker = kind == "tracker"
        if tracker:
            original = "image_position"
        label = kind if tracker else f"{kind}_{'frozen' if seed is None else 'ft_s'+str(seed)}"
        saved = {}
        for r in source.get("results", []):
            if r["representation"] == original and r["split"] == "logo":
                k = (r["interventions"],r["target"])
                if k in saved:
                    raise ValueError(f"Duplicate result {k}")
                saved[k] = r["per_fold"]
        by_action = {a: [] for a in probe.ACTIONS}
        for g in groups:
            bank = None
            if not tracker:
                path = args.root/"features"/file.format(fold=g)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing existing bank {path}; no extraction is allowed")
                bank = probe.FeatureBank(path)
                expected_dim = 12288 if kind == "grid" else 1536
                if bank.f.shape != (len(bank.index), expected_dim) or not np.isfinite(bank.f).all():
                    raise ValueError(f"Invalid feature matrix {path}")
                if seed is not None:
                    with np.load(path,allow_pickle=False) as z:
                        if Path(str(z["checkpoint"].item())).parent.name != f"clip_s{seed}_{g}":
                            raise ValueError(f"Wrong held-out checkpoint: {path}")
            for subset in subsets:
                key = "+".join(subset)
                computed = None
                for target in probe.TARGETS:
                    row = saved.get((key,target),{}).get(g)
                    if row is not None and row["n"] == counts[g] and np.isfinite(row["nmae"]) and row["nmae"] >= 0:
                        score, src = row["nmae"], "cached"
                    else:
                        if computed is None:
                            computed = fold_targets(configs,g,subset,"image_position" if tracker else "visual",bank)
                        score, src = computed[target], "probe"
                    raw.append(dict(representation=label,geometry=g,n=counts[g],target=target,
                                    k=len(subset),subset=key,nmae=score,source=src))
            if bank is not None:
                for a in probe.ACTIONS:
                    y,p = position_fold(configs,g,a,bank)
                    positions.append(dict(representation=label,geometry=g,action=a,
                                          **position_scores(y,p,False)))
                    by_action[a].append((y,p))
            print(f"Finished {label} / {g}",flush=True)
            del bank
        if not tracker:
            for a,pairs in by_action.items():
                y,p = (np.concatenate(x) for x in zip(*pairs))
                positions.append(dict(representation=label,geometry="ALL",action=a,**position_scores(y,p,True)))
    curves = {}
    for r in raw:
        key = tuple(r[k] for k in ("representation","geometry","n","target","k"))
        curves.setdefault(key,[]).append(r["nmae"])
    summary = [dict(representation=l,geometry=g,n=n,target=t,k=k,subsets=len(v),mean_nmae=np.mean(v))
               for (l,g,n,t,k),v in curves.items()]
    args.out.mkdir(parents=True,exist_ok=False)
    write_csv(args.out/"fold_subset_nmae.csv",raw)
    write_csv(args.out/"fold_scaling.csv",summary)
    write_csv(args.out/"position_r2.csv",positions)
    scores = {(r["representation"],r["geometry"],r["target"],r["k"]):r["mean_nmae"] for r in summary}
    paired = []
    for key,v in scores.items():
        label,g,t,k = key
        if "_ft_s" in label:
            ref = label.split("_ft_s")[0]+"_frozen"
            paired.append(dict(representation=label,reference=ref,geometry=g,target=t,k=k,
                               delta_nmae=v-scores[ref,g,t,k]))
    write_csv(args.out/"paired_nmae.csv",paired)
    pos_idx = {(r["representation"],r["geometry"],r["action"]):r for r in positions}
    comparisons = []
    for (label,g,a),r in pos_idx.items():
        if "_ft_s" not in label:
            continue
        ref = label.split("_ft_s")[0]+"_frozen"
        base = pos_idx[ref,g,a]
        comparisons.append(dict(representation=label,reference=ref,geometry=g,action=a,**{
            "delta_"+m:(None if r[m] is None else r[m]-base[m])
            for m in ("r2_mean","r2_dx","r2_dy","r2_log_area")
        }))
    write_csv(args.out/"paired_position.csv",comparisons)
    print(f"Diagnostic tables written to {args.out}")


if __name__ == "__main__":
    main()
