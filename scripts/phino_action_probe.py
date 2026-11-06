#!/usr/bin/env python3
"""Linear probes of mass/friction from action-conditioned observations.

Subcommands (from the SIMPACT root):
    # per-frame DINOv2 features (PHINO's CLS || mean-patch, 1536-d) for every image
    python scripts/phino_action_probe.py extract --name dinov2_frozen
    python scripts/phino_action_probe.py extract --name ft_clip_cow --checkpoint <ckpt.pt>
    # probes for all representations / intervention sets / splits
    python scripts/phino_action_probe.py evaluate --output-dir results/phino_action_learning_v1/probes

Every visual encoder uses the SAME representation and probe:
  intervention a -> concat_k [f(frame_k) - f(before)]  (5 post-onset frames)
  intervention set A -> concat over a in A (fixed action order)
  n = 0 (static control) -> f(before image of the push camera)
  probe: standardized ridge (dual form), alpha by inner grouped CV on the
  training split only; log mass and log friction predicted separately.
Splits: leave-one-geometry-out (primary) and within-geometry leave-one-
physics-cluster-out (finite-difference neighbours grouped with their grid point).
Baselines: training mean, image-position (segmentation centroid/area),
privileged simulator state (observed frames; full trajectory).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ACTIONS = ("push_x_v2", "push_lo_1.6N", "push_hi_3.2N", "toss_4.0N")
FRAME_INDICES = (2, 4, 6, 9, 49)
ALPHAS = np.logspace(-1, 7, 17)
TARGETS = ("mass", "friction")
MIN_FEATURE_STD = 1e-5
GRID = 4                       # spatial readout: 4x4 average-pooled patch tokens per frame
PRED_LOG_MARGIN = 0.5          # predictions clipped to the training log-label range +-0.5 (all models alike)
DEFAULT_DATA = Path("data/phino_simpact_v3obs_v2")
DEFAULT_FEATURES = Path("results/phino_action_learning_v2/features")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = [json.loads(line) for line in (data_dir / "samples.jsonl").read_text().splitlines() if line]
    with np.load(data_dir / "trajectories.npz") as t:
        traj = {k: t[k] for k in t.files}
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample ids")
    return rows, traj


def configs_table(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(geometry, config) -> {mass, friction, cluster, clips: {action: row}}."""
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if not r["object_in_frame"]:
            continue
        key = (r["geometry_short"], r["config"])
        t = table.setdefault(key, {"geometry": r["geometry_short"], "config": r["config"], "mass": r["mass"],
                                   "friction": r["friction"], "cluster": r["cluster"], "clips": {}})
        t["clips"][r["action"]] = r
    return table


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def read_images(data_dir: Path, paths: list[str], workers: int = 12) -> list[np.ndarray]:
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def load(p: str) -> np.ndarray:
        with Image.open(data_dir / p) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(load, paths))


def load_encoder(phino_root: Path, checkpoint: Path | None, device: str):
    import torch
    sys.path.insert(0, str(phino_root.resolve()))
    from train.models.dynaclip import DynaCLIPModel
    model = DynaCLIPModel("dinov2_vitb14")
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.backbone.load_state_dict(state["backbone"], strict=True)
    return model.to(device).eval()


def extract(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    import torchvision.transforms as T

    out = args.features_dir / f"{args.name}.npz"
    if out.exists():
        sys.exit(f"Feature file exists: {out}")
    rows, _ = load_dataset(args.data_dir)
    paths = sorted({p for r in rows for p in r["images"]})
    model = load_encoder(args.phino_root, args.checkpoint, "cuda")
    tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dim = GRID * GRID * model.backbone_dim if args.grid else 1536
    feats = np.empty((len(paths), dim), dtype=np.float16 if args.grid else np.float32)
    arrays = read_images(args.data_dir, paths)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(paths), args.batch_size):
            batch = torch.stack([tf(Image.fromarray(a)) for a in arrays[i:i + args.batch_size]]).cuda()
            if args.grid:
                # spatial readout: patch tokens average-pooled on a GRID x GRID grid
                tok = model.backbone.forward_features(batch)["x_norm_patchtokens"].float()
                side = int(round(tok.shape[1] ** 0.5))
                tok = tok.transpose(1, 2).reshape(len(batch), -1, side, side)
                pooled = torch.nn.functional.adaptive_avg_pool2d(tok, GRID)
                feats[i:i + len(batch)] = pooled.flatten(1).cpu().numpy().astype(np.float16)
            else:
                feats[i:i + len(batch)] = model.extract_features(batch).float().cpu().numpy()
    if not np.isfinite(feats).all():
        sys.exit("non-finite features")
    args.features_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, paths=np.array(paths), features=feats,
                        checkpoint=str(args.checkpoint or "pretrained dinov2_vitb14"))
    print(f"FEATURES={out} n={len(paths)}")


class FeatureBank:
    def __init__(self, path: Path):
        with np.load(path) as d:
            self.index = {p: i for i, p in enumerate(d["paths"].tolist())}
            self.f = d["features"].astype(np.float32)

    def __call__(self, p: str) -> np.ndarray:
        return self.f[self.index[p]]


# ---------------------------------------------------------------------------
# Representations (identical across encoders)
# ---------------------------------------------------------------------------
def visual_clip(bank: FeatureBank, clip: dict[str, Any]) -> np.ndarray:
    f0 = bank(clip["images"][0])
    return np.concatenate([bank(p) - f0 for p in clip["images"][1:]])


def visual_static(bank: FeatureBank, cfg: dict[str, Any]) -> np.ndarray:
    return bank(cfg["clips"]["push_x_v2"]["images"][0])


def image_position(clip: dict[str, Any]) -> np.ndarray:
    c = np.asarray(clip["centroid_px"], float)
    a = np.asarray(clip["mask_area_px"], float)
    return np.concatenate([(c[1:] - c[0]).ravel(), np.log(a[1:] / a[0])])


def state_frames(traj: np.ndarray) -> np.ndarray:
    return traj[list(FRAME_INDICES), :7].ravel()


def state_full(traj: np.ndarray) -> np.ndarray:
    return traj.ravel()


def state_physics(traj: np.ndarray, dt: float = 0.05) -> np.ndarray:
    """Privileged compact response summary (log magnitudes follow the power-law dynamics)."""
    speed = np.linalg.norm(traj[:, 7:9], axis=1)
    q = traj[:, 3:7]
    yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    moving = np.flatnonzero(speed >= 0.01)
    stop = (moving[-1] + 1) * dt if moving.size else 0.0
    return np.array([np.log(np.linalg.norm(traj[-1, :2]) + 1e-4), np.log(speed.max() + 1e-4),
                     np.log(max(traj[:, 2].max(), 0.0) + 1e-4), yaw[-1], stop])


def build_matrix(cfgs: list[dict[str, Any]], actions: tuple[str, ...], rep: str,
                 bank: FeatureBank | None, traj: dict[str, np.ndarray]) -> np.ndarray:
    out = []
    for c in cfgs:
        if not actions:
            out.append(visual_static(bank, c) if rep == "visual" else np.zeros(1))
            continue
        parts = []
        for a in actions:
            clip = c["clips"][a]
            if rep == "visual":
                parts.append(visual_clip(bank, clip))
            elif rep == "image_position":
                parts.append(image_position(clip))
            elif rep == "state_frames":
                parts.append(state_frames(traj[clip["sample_id"]]))
            elif rep == "state_full":
                parts.append(state_full(traj[clip["sample_id"]]))
            elif rep == "state_physics":
                parts.append(state_physics(traj[clip["sample_id"]]))
            else:
                raise KeyError(rep)
        out.append(np.concatenate(parts))
    X = np.asarray(out, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite {rep} features")
    return X


# ---------------------------------------------------------------------------
# Ridge with inner grouped CV (training data only)
# ---------------------------------------------------------------------------
def _standardize(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns that are (numerically) constant in training are dropped, not rescaled:
    dividing simulator noise of ~1e-7 by its own std explodes held-out predictions."""
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.where(sd < MIN_FEATURE_STD, np.inf, sd)
    return (xtr - mu) / sd, (xte - mu) / sd


def ridge_path(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Standardized dual ridge with intercept.

    ytr (n,) -> predictions (len(alphas), n_test); ytr (n, t) -> (len(alphas), n_test, t).
    """
    ztr, zte = _standardize(xtr, xte)
    ym = ytr.mean(0)
    lam, U = np.linalg.eigh(ztr @ ztr.T)
    lam = np.clip(lam, 0, None)
    Uy = U.T @ (ytr - ym)
    kte = zte @ ztr.T
    shape = (-1, 1) if Uy.ndim == 2 else (-1,)
    return np.asarray([ym + kte @ (U @ (Uy / (lam + a).reshape(shape))) for a in alphas])


def fit_predict(xtr, ytr, gtr, xte, alphas=ALPHAS) -> tuple[np.ndarray, np.ndarray]:
    """Per-target alpha by inner grouped CV on the training data; ytr is (n, t)."""
    ytr = ytr.reshape(len(ytr), -1)
    groups = sorted(set(gtr))
    if len(groups) >= 2 and xtr.shape[1] > 1:
        n_inner = min(len(groups), 5)
        folds = [groups[i::n_inner] for i in range(n_inner)]
        err = np.zeros((len(alphas), ytr.shape[1]))
        for held in folds:
            m = np.isin(gtr, held)
            p = ridge_path(xtr[~m], ytr[~m], xtr[m], alphas)
            err += np.abs(p - ytr[m]).sum(axis=1)
        best = np.argmin(err, axis=0)
    else:
        best = np.full(ytr.shape[1], len(alphas) // 2)
    paths = ridge_path(xtr, ytr, xte, alphas[np.unique(best)])
    pred = np.stack([paths[list(np.unique(best)).index(b), :, j] for j, b in enumerate(best)], axis=1)
    pred = np.clip(pred, ytr.min(0) - PRED_LOG_MARGIN, ytr.max(0) + PRED_LOG_MARGIN)
    return pred, alphas[best]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_split(cfgs, X, split: str, fold_encoder: Callable[[str], np.ndarray] | None = None
                   ) -> list[dict[str, Any]]:
    """Out-of-fold predictions for each target. ``fold_encoder(test_geometry)`` returns X for that fold."""
    geoms = np.array([c["geometry"] for c in cfgs])
    clusters = np.array([f"{c['geometry']}|{c['cluster']}" for c in cfgs])
    y = {t: np.log(np.array([c[t] for c in cfgs])) for t in TARGETS}
    rows = []
    if split == "logo":
        folds = [(g, geoms == g, geoms != g, geoms) for g in sorted(set(geoms))]
    else:
        folds = []
        for g in sorted(set(geoms)):
            for cl in sorted(set(clusters[geoms == g])):
                test = clusters == cl
                train = (geoms == g) & ~test
                folds.append((g, test, train, clusters))
    Y = np.column_stack([y[t] for t in TARGETS])
    for g, test, train, grp in folds:
        if test.sum() == 0 or train.sum() < 5:
            continue
        Xf = fold_encoder(g) if fold_encoder else X
        pred, alphas = fit_predict(Xf[train], Y[train], grp[train], Xf[test])
        for j, t in enumerate(TARGETS):
            base = y[t][train].mean()
            for i, idx in enumerate(np.flatnonzero(test)):
                rows.append({"split": split, "test_geometry": g, "target": t, "index": int(idx),
                             "true": float(np.exp(y[t][idx])), "pred": float(np.exp(pred[i, j])),
                             "baseline": float(np.exp(base)), "alpha": float(alphas[j])})
    return rows


def summarize(pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for t in TARGETS:
        r = [p for p in pred_rows if p["target"] == t]
        if not r:
            continue
        err = np.array([abs(p["pred"] - p["true"]) for p in r])
        base = np.array([abs(p["baseline"] - p["true"]) for p in r])
        per_fold = {}
        for g in sorted({p["test_geometry"] for p in r}):
            e = np.array([abs(p["pred"] - p["true"]) for p in r if p["test_geometry"] == g])
            b = np.array([abs(p["baseline"] - p["true"]) for p in r if p["test_geometry"] == g])
            per_fold[g] = {"n": int(len(e)), "mae": float(e.mean()), "nmae": float(e.mean() / b.mean())}
        out[t] = {"n": int(len(err)), "mae": float(err.mean()), "baseline_mae": float(base.mean()),
                  "nmae": float(err.mean() / base.mean()), "per_fold": per_fold}
    return out


def position_decoding(cfgs, bank, action: str) -> dict[str, float]:
    """LOGO R^2 of decoding segmentation centroid shift / log-area ratio from visual features.

    Tests whether the encoder (with the shared representation) preserves the measurable
    image-space response that the 15-d image-position baseline uses directly.
    """
    X_of = (lambda g: build_matrix(cfgs, (action,), "visual", bank[g], {})) if isinstance(bank, dict) \
        else (lambda g, X=build_matrix(cfgs, (action,), "visual", bank, {}): X)
    Y = np.array([image_position(c["clips"][action]) for c in cfgs])
    geoms = np.array([c["geometry"] for c in cfgs])
    pred = np.zeros_like(Y)
    for g in sorted(set(geoms)):
        test, train = geoms == g, geoms != g
        X = X_of(g)
        pred[test] = ridge_path_cv(X[train], Y[train], geoms[train], X[test])
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return {"r2_dx": float(np.mean(r2[0:10:2])), "r2_dy": float(np.mean(r2[1:10:2])),
            "r2_log_area": float(np.mean(r2[10:15])), "r2_mean": float(np.mean(r2))}


def ridge_path_cv(xtr, Ytr, gtr, xte) -> np.ndarray:
    """Unclipped ridge, per-column alpha by the same inner grouped CV; Ytr is (n, t)."""
    groups = sorted(set(gtr))
    folds = [groups[i::min(len(groups), 5)] for i in range(min(len(groups), 5))]
    err = np.zeros((len(ALPHAS), Ytr.shape[1]))
    for held in folds:
        m = np.isin(gtr, held)
        err += np.abs(ridge_path(xtr[~m], Ytr[~m], xtr[m], ALPHAS) - Ytr[m]).sum(axis=1)
    paths = ridge_path(xtr, Ytr, xte, ALPHAS)
    return np.stack([paths[b, :, j] for j, b in enumerate(np.argmin(err, axis=0))], axis=1)


def evaluate(args: argparse.Namespace) -> None:
    rows, traj = load_dataset(args.data_dir)
    table = configs_table(rows)
    common = [c for c in table.values() if all(a in c["clips"] for a in ACTIONS)]
    common.sort(key=lambda c: (c["geometry"], c["config"]))
    out = args.output_dir
    if out.exists():
        sys.exit(f"Output directory exists: {out}")
    banks: dict[str, Any] = {}
    for spec in args.encoder:
        name, _, pattern = spec.partition("=")
        if "{fold}" in pattern:
            banks[name] = {g: FeatureBank(Path(pattern.format(fold=g))) for g in sorted({c["geometry"] for c in common})}
        else:
            banks[name] = FeatureBank(Path(pattern))
    reps = {name: "visual" for name in banks}
    reps.update({r: r for r in args.baseline})
    results, all_preds = [], []
    subsets = [s for n in range(len(ACTIONS) + 1) for s in itertools.combinations(ACTIONS, n)]
    if args.subsets:
        subsets = [s for s in subsets if ("+".join(s) or "static") in args.subsets]
    for rep_name, rep in reps.items():
        for subset in subsets:
            if rep != "visual" and not subset:
                continue
            bank = banks.get(rep_name)
            for split in args.splits:
                if isinstance(bank, dict):
                    cache = {g: build_matrix(common, subset, rep, bank[g], traj) for g in bank}
                    preds = evaluate_split(common, None, split, fold_encoder=lambda g: cache[g])
                else:
                    X = build_matrix(common, subset, rep, bank, traj)
                    preds = evaluate_split(common, X, split)
                s = summarize(preds)
                for t, v in s.items():
                    results.append({"representation": rep_name, "interventions": "+".join(subset) or "static",
                                    "n_interventions": len(subset), "split": split, "target": t,
                                    "n": v["n"], "mae": v["mae"], "baseline_mae": v["baseline_mae"],
                                    "nmae": v["nmae"], "per_fold": v["per_fold"]})
                for p in preds:
                    all_preds.append({"representation": rep_name, "interventions": "+".join(subset) or "static", **p})
            print(f"{rep_name:16s} {('+'.join(subset) or 'static'):45s} "
                  + "  ".join(f"{r['split']}:{r['target'][0]}={r['nmae']:.2f}"
                              for r in results[-2 * len(args.splits):]), flush=True)
    decoding = {} if args.skip_decoding else \
        {name: {a: position_decoding(common, bank, a) for a in ACTIONS} for name, bank in banks.items()}
    for name, d in decoding.items():
        print(f"position decoding {name}: " + "  ".join(f"{a}:{v['r2_mean']:.2f}" for a, v in d.items()))
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps({
        "n_configs": len(common),
        "configs_per_geometry": {g: sum(c["geometry"] == g for c in common) for g in sorted({c["geometry"] for c in common})},
        "encoders": args.encoder, "alphas": ALPHAS.tolist(), "min_feature_std": MIN_FEATURE_STD,
        "pred_log_margin": PRED_LOG_MARGIN, "position_decoding_logo_r2": decoding, "results": results}, indent=1))
    with open(out / "predictions.jsonl", "w") as fh:
        for p in all_preds:
            fh.write(json.dumps(p) + "\n")
    print(f"RESULTS={out / 'results.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--name", required=True)
    e.add_argument("--checkpoint", type=Path, default=None)
    e.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    e.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    e.add_argument("--phino-root", type=Path, default=Path("/nbu/people/ashah264/phino"))
    e.add_argument("--batch-size", type=int, default=256)
    e.add_argument("--grid", action="store_true", help="spatial 4x4 pooled patch-token readout")
    v = sub.add_parser("evaluate")
    v.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--encoder", action="append", default=[],
                   help="name=path.npz, or name=path_{fold}.npz for per-fold (geometry-excluded) encoders")
    v.add_argument("--baseline", action="append", default=None,
                   choices=("image_position", "state_physics", "state_frames", "state_full"))
    v.add_argument("--skip-decoding", action="store_true")
    v.add_argument("--splits", nargs="+", default=["logo", "within"], choices=("logo", "within"))
    v.add_argument("--subsets", nargs="+", default=None,
                   help="restrict to these intervention sets ('static' or a+b joined by '+')")
    args = ap.parse_args()
    if args.cmd == "evaluate" and args.baseline is None:
        args.baseline = ["image_position", "state_physics", "state_frames", "state_full"]
    extract(args) if args.cmd == "extract" else evaluate(args)


if __name__ == "__main__":
    main()
