#!/usr/bin/env python3
"""Contrastive fine-tuning of DINOv2 (PHINO components) on action-conditioned clips.

    CUDA_VISIBLE_DEVICES=2 python scripts/phino_action_contrastive_train.py \
        --test-geometry mesh_cow --mode clip \
        --output-dir results/phino_action_learning_v1/runs/clip_mesh_cow

Reuses PHINO's DynaCLIPModel (DINOv2-B/14 backbone, CLS||mean-patch features,
DynaCLIPProjectionHead), SoftInfoNCELoss, and TrajectoryNormalizer. Changes
required by action-conditioned data, and why:
  * observation = clip of one intervention: head input is concat_k [f(frame_k) - f(before)]
    (the same representation the probes use);
  * every batch holds clips of a SINGLE intervention, and the soft target is the
    DENSE B x B start-relative trajectory similarity (PHINO's normalizer, fit on
    training geometries only, n_actions=1). PHINO's diagonal-only target exists
    because its v0 data lacks cross-pair similarities; here every pair has one.
    Identical observations cannot receive contradictory targets: the target depends
    only on the trajectories that generate the frames.
  * two augmented views per clip (shared colour jitter + translation across the
    clip's frames; no flips, which would reverse the push direction).
--mode static is the negative control: identical targets, but the observation is
only the before image (identical across a geometry's physics configurations).
The test geometry is excluded from all training data and normalizer fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

import phino_action_probe as probe

AUG_SHIFT = 6
AUG_JITTER = 0.10


def preload(data_dir: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    paths = sorted({p for r in rows for p in r["images"]})
    return dict(zip(paths, probe.read_images(data_dir, paths)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=probe.DEFAULT_DATA)
    ap.add_argument("--phino-root", type=Path, default=Path("/nbu/people/ashah264/phino"))
    ap.add_argument("--test-geometry", required=True, help="geometry_short excluded from training")
    ap.add_argument("--mode", choices=("clip", "static"), default="clip")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-clips", type=int, default=24)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--unfreeze-last", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    args = ap.parse_args()
    if args.output_dir.exists():
        sys.exit(f"Output directory exists: {args.output_dir}")

    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.models.dynaclip import DynaCLIPModel, DynaCLIPProjectionHead
    from train.losses.contrastive import SoftInfoNCELoss
    from train.data.mujoco_engine import TrajectoryNormalizer

    class DenseSoftInfoNCE(SoftInfoNCELoss):
        """PHINO SoftInfoNCE with the full B x B similarity matrix as the soft target."""
        def _build_similarity_matrix(self, pair_sim, B, device):
            if pair_sim.shape != (B, B):
                raise ValueError("expected a dense (B, B) similarity matrix")
            return pair_sim

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, traj = probe.load_dataset(args.data_dir)
    rows = [r for r in rows if r["object_in_frame"]]
    geoms = sorted({r["geometry_short"] for r in rows})
    if args.test_geometry not in geoms:
        sys.exit(f"unknown geometry {args.test_geometry}; choose from {geoms}")
    train = [r for r in rows if r["geometry_short"] != args.test_geometry]
    by_action = {a: [r for r in train if r["action"] == a] for a in probe.ACTIONS}
    sims = {}
    norm_stats = {}
    for a, rs in by_action.items():
        T = [traj[r["sample_id"]] for r in rs]
        norm = TrajectoryNormalizer(n_actions=1, timesteps=50).fit(T, rng=np.random.default_rng(args.seed))
        Z = np.stack([norm.transform(t) for t in T])
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).mean(-1)
        S = np.exp(-d2 / norm.sigma)
        if not np.isfinite(S).all():
            sys.exit(f"non-finite similarities for {a}")
        sims[a] = torch.tensor(S, dtype=torch.float32)
        norm_stats[a] = {"n_train_clips": len(rs), "sigma": float(norm.sigma),
                         "median_offdiag_similarity": float(np.median(S[~np.eye(len(S), dtype=bool)]))}
    images = preload(args.data_dir, train)

    device = "cuda"
    model = DynaCLIPModel("dinov2_vitb14", unfreeze_last_n_blocks=args.unfreeze_last)
    n_frames = 1 if args.mode == "static" else len(probe.FRAME_INDICES)
    model.projection_head = DynaCLIPProjectionHead(input_dim=model.feature_dim * n_frames,
                                                   hidden_dim=model.backbone_dim, output_dim=512)
    model.enable_gradient_checkpointing()
    model.to(device).train()
    loss_fn = DenseSoftInfoNCE().to(device)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": list(model.projection_head.parameters()) + list(loss_fn.parameters()), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def view(clip_imgs: torch.Tensor) -> torch.Tensor:
        """clip_imgs (F,3,H,W) uint8 -> augmented, normalized float; same transform for all frames."""
        x = clip_imgs.float() / 255.0
        x = x * (1 + AUG_JITTER * (2 * torch.rand(1, device=device) - 1))
        x = (x - x.mean()) * (1 + AUG_JITTER * (2 * torch.rand(1, device=device) - 1)) + x.mean()
        x = F.pad(x, (AUG_SHIFT,) * 4, mode="replicate")
        i, j = np.random.randint(0, 2 * AUG_SHIFT + 1, size=2)
        x = x[..., i:i + 224, j:j + 224].clamp(0, 1)
        return (x - mean) / std

    def encode(batch_rows, views_per_clip: int) -> list[torch.Tensor]:
        outs = []
        for _ in range(views_per_clip):
            frames = []
            for r in batch_rows:
                sel = r["images"][:1] if args.mode == "static" else r["images"]
                arr = torch.from_numpy(np.stack([images[p] for p in sel])).to(device).permute(0, 3, 1, 2)
                frames.append(view(arr))
            x = torch.cat(frames)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.extract_features(x).float().view(len(batch_rows), -1, model.feature_dim)
                if args.mode == "clip":
                    f = (f[:, 1:] - f[:, :1]).reshape(len(batch_rows), -1)
                else:
                    f = f.reshape(len(batch_rows), -1)
                outs.append(model.projection_head(f).float())
        return outs

    args.output_dir.mkdir(parents=True)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "train_geometries": sorted({r["geometry_short"] for r in train}),
              "normalizers": norm_stats, "n_frames_per_observation": n_frames,
              "backbone_trainable_params": int(sum(p.numel() for p in backbone_params)),
              "loss": "PHINO SoftInfoNCE (temperature 0.07 learnable, similarity temperature 0.1) with dense target"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=1))
    log = open(args.output_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "action", "loss", "temperature", "diag_logit", "lr_backbone", "seconds"])
    actions = [a for a in probe.ACTIONS if len(by_action[a]) >= 4]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = step / args.warmup if step <= args.warmup else \
            0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * frac
        a = actions[(step - 1) % len(actions)]
        idx = np.random.choice(len(by_action[a]), size=min(args.batch_clips, len(by_action[a])), replace=False)
        batch = [by_action[a][i] for i in idx]
        S = sims[a][idx][:, idx].to(device)
        z1, z2 = encode(batch, 2)
        out = loss_fn(F.normalize(z1, dim=-1), F.normalize(z2, dim=-1), S)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(out["loss"]):
            sys.exit(f"non-finite loss at step {step}")
        if step % 10 == 0 or step == 1:
            writer.writerow([step, a, out["loss"].item(), out["temperature"].item(),
                             out["logits_mean"].item(), opt.param_groups[0]["lr"], round(time.time() - t0, 1)])
            log.flush()
        if step % 100 == 0 or step == args.steps:
            torch.save({"backbone": model.backbone.state_dict(), "head": model.projection_head.state_dict(),
                        "step": step, "config": config}, args.output_dir / "checkpoint.pt")
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"time limit reached at step {step}")
            break
    log.close()
    (args.output_dir / "done.json").write_text(json.dumps({"steps": step, "minutes": (time.time() - t0) / 60}))
    print(f"DONE {args.output_dir} steps={step} minutes={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
