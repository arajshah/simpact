#!/usr/bin/env python3
"""Fine-tune DINOv2 on action-conditioned clips with one of three objectives.

    CUDA_VISIBLE_DEVICES=2 python scripts/phino_action_contrastive_train.py \
        --objective instance --test-geometry mesh_cow --mode clip \
        --output-dir results/phino_action_baselines_v1/runs/instance_s0_mesh_cow

The default objective is the existing trajectory-similarity run: PHINO's
DynaCLIPModel (DINOv2-B/14, CLS||mean-patch, last 4 blocks trainable) and dense
SoftInfoNCE. Each batch is one intervention. The soft target is the full
BxB start-relative trajectory similarity, with TrajectoryNormalizer fit on the
training geometries only. Two augmented views share colour jitter and a
translation across the clip's frames. Flips are not used.

``--objective instance`` is standard symmetric InfoNCE on those same two views.
Other clips in the batch are negatives. No trajectory similarity is computed.

``--objective regression`` predicts training-fold standardized log mass and log
friction from the same clip. The head is not the evaluated predictor.

``--mode static`` remains the trajectory-similarity negative control. The test
geometry is excluded from training rows, normalizer fitting, and label
standardization. Hyperparameters are the fixed trajectory-run values.
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

import phino_action_objectives as objectives
import phino_action_probe as probe

AUG_SHIFT = 6
AUG_JITTER = 0.10
LOSS_DESCRIPTION = {
    "trajectory": "PHINO SoftInfoNCE (temperature 0.07 learnable, similarity temperature 0.1) with dense target",
    "instance": "symmetric instance-discrimination InfoNCE; learnable temperature init 0.07 clamped to [0.01, 1]; no trajectory target",
    "regression": "MSE on training-fold standardized log mass and log friction, averaged over two augmented views",
}


def preload(data_dir: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    paths = sorted({p for r in rows for p in r["images"]})
    return dict(zip(paths, probe.read_images(data_dir, paths)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=probe.DEFAULT_DATA)
    ap.add_argument("--phino-root", type=Path, default=Path("/nbu/people/ashah264/phino"))
    ap.add_argument("--test-geometry", required=True, help="geometry_short excluded from training")
    ap.add_argument("--objective", choices=("trajectory", "instance", "regression"), default="trajectory")
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
    if args.objective != "trajectory" and args.mode != "clip":
        sys.exit("instance and regression objectives are defined on action-conditioned clips")

    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(args.phino_root.resolve()))
    from train.models.dynaclip import DynaCLIPModel, DynaCLIPProjectionHead

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, traj = probe.load_dataset(args.data_dir)
    try:
        train = objectives.training_rows(rows, args.test_geometry)
    except ValueError as exc:
        sys.exit(str(exc))
    by_action = {a: [r for r in train if r["action"] == a] for a in probe.ACTIONS}
    sims = {}
    norm_stats = None
    label_stats = None
    if args.objective == "trajectory":
        from train.losses.contrastive import SoftInfoNCELoss
        from train.data.mujoco_engine import TrajectoryNormalizer

        class DenseSoftInfoNCE(SoftInfoNCELoss):
            """PHINO SoftInfoNCE with the full B x B similarity matrix as the soft target."""
            def _build_similarity_matrix(self, pair_sim, B, device):
                if pair_sim.shape != (B, B):
                    raise ValueError("expected a dense (B, B) similarity matrix")
                return pair_sim

        norm_stats = {}
        for action, action_rows in by_action.items():
            traces = [traj[r["sample_id"]] for r in action_rows]
            norm = TrajectoryNormalizer(n_actions=1, timesteps=50).fit(traces, rng=np.random.default_rng(args.seed))
            standardized = np.stack([norm.transform(t) for t in traces])
            distance = ((standardized[:, None, :] - standardized[None, :, :]) ** 2).mean(-1)
            similarity = np.exp(-distance / norm.sigma)
            if not np.isfinite(similarity).all():
                sys.exit(f"non-finite similarities for {action}")
            sims[action] = torch.tensor(similarity, dtype=torch.float32)
            off = ~np.eye(len(similarity), dtype=bool)
            norm_stats[action] = {
                "n_train_clips": len(action_rows), "sigma": float(norm.sigma),
                "median_offdiag_similarity": float(np.median(similarity[off])),
            }
    elif args.objective == "regression":
        try:
            label_stats = objectives.fit_log_label_standardizer(objectives.unique_configurations(train))
        except ValueError as exc:
            sys.exit(str(exc))

    images = preload(args.data_dir, train)
    device = "cuda"
    model = DynaCLIPModel("dinov2_vitb14", unfreeze_last_n_blocks=args.unfreeze_last)
    n_frames = 1 if args.mode == "static" else len(probe.FRAME_INDICES)
    # Build the contrastive head for every objective so the parameter-init RNG
    # matches the trajectory run. Regression then swaps in its own head and
    # restores that RNG state.
    model.projection_head = DynaCLIPProjectionHead(
        input_dim=model.feature_dim * n_frames, hidden_dim=model.backbone_dim, output_dim=512)
    rng_after_contrastive_head = torch.random.get_rng_state()
    if args.objective == "regression":
        model.projection_head = objectives.ClipRegressionHead(
            input_dim=model.feature_dim * n_frames, hidden_dim=model.backbone_dim)
        torch.random.set_rng_state(rng_after_contrastive_head)
    model.enable_gradient_checkpointing()
    model.to(device).train()
    if args.objective == "trajectory":
        loss_fn = DenseSoftInfoNCE().to(device)
    elif args.objective == "instance":
        loss_fn = objectives.InstanceInfoNCE().to(device)
    else:
        loss_fn = None
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.projection_head.parameters())
    if loss_fn is not None:
        head_params += list(loss_fn.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def view(clip_imgs: torch.Tensor) -> torch.Tensor:
        """clip_imgs (F,3,H,W) uint8 -> augmented, normalized float; same transform for all frames."""
        x = clip_imgs.float() / 255.0
        brightness = torch.rand(1, device=device)
        contrast = torch.rand(1, device=device)
        crop_i, crop_j = np.random.randint(0, 2 * AUG_SHIFT + 1, size=2)
        return objectives.shared_view(x, brightness, contrast, int(crop_i), int(crop_j), mean, std,
                                      AUG_SHIFT, AUG_JITTER)

    def encode(batch_rows, views_per_clip: int) -> list[torch.Tensor]:
        outs = []
        for _ in range(views_per_clip):
            frames = []
            for row in batch_rows:
                selected = row["images"][:1] if args.mode == "static" else row["images"]
                array = torch.from_numpy(np.stack([images[p] for p in selected])).to(device).permute(0, 3, 1, 2)
                frames.append(view(array))
            batch = torch.cat(frames)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.extract_features(batch).float().view(len(batch_rows), -1, model.feature_dim)
                if args.mode == "clip":
                    features = (features[:, 1:] - features[:, :1]).reshape(len(batch_rows), -1)
                else:
                    features = features.reshape(len(batch_rows), -1)
                outs.append(model.projection_head(features).float())
        return outs

    args.output_dir.mkdir(parents=True)
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "train_geometries": sorted({r["geometry_short"] for r in train}),
        "test_geometry_in_training": False,
        "normalizers": norm_stats,
        "log_label_standardizer": None if label_stats is None else {
            "mean": label_stats["mean"].tolist(), "std": label_stats["std"].tolist(),
            "fit_on": "unique training configurations; the test geometry is excluded",
        },
        "n_frames_per_observation": n_frames,
        "views_per_clip": 2,
        "backbone_trainable_params": int(sum(p.numel() for p in backbone_params)),
        "loss": LOSS_DESCRIPTION[args.objective],
        "hyperparameters": "fixed to the trajectory-similarity run; not selected on the test geometry",
    }
    if args.test_geometry in config["train_geometries"]:
        sys.exit("test geometry leaked into the recorded training split")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=1))
    log = open(args.output_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "action", "loss", "temperature", "diag_logit", "lr_backbone", "seconds"])
    actions = [a for a in probe.ACTIONS if len(by_action[a]) >= 4]
    if len(actions) != len(probe.ACTIONS):
        sys.exit(f"training split is missing an action with at least 4 clips: {actions}")
    step = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = step / args.warmup if step <= args.warmup else \
            0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        for group, lr in zip(opt.param_groups, base_lrs):
            group["lr"] = lr * frac
        action = actions[(step - 1) % len(actions)]
        chosen = np.random.choice(len(by_action[action]), size=min(args.batch_clips, len(by_action[action])),
                                  replace=False)
        batch = [by_action[action][i] for i in chosen]
        z1, z2 = encode(batch, 2)
        if args.objective == "trajectory":
            target = sims[action][chosen][:, chosen].to(device)
            out = loss_fn(F.normalize(z1, dim=-1), F.normalize(z2, dim=-1), target)
        elif args.objective == "instance":
            out = loss_fn(F.normalize(z1, dim=-1), F.normalize(z2, dim=-1))
        else:
            labels = torch.as_tensor(objectives.standardize_log_labels(batch, label_stats),
                                     dtype=torch.float32, device=device)
            loss = objectives.two_view_regression_loss(z1, z2, labels)
            out = {"loss": loss, "temperature": z1.new_tensor(float("nan")),
                   "logits_mean": 0.5 * ((z1 - labels).abs().mean() + (z2 - labels).abs().mean()).detach()}
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(out["loss"]):
            sys.exit(f"non-finite loss at step {step}")
        if step % 10 == 0 or step == 1:
            temperature = out["temperature"].item()
            writer.writerow([step, action, out["loss"].item(), temperature,
                             out["logits_mean"].item(), opt.param_groups[0]["lr"], round(time.time() - t0, 1)])
            log.flush()
        if step % 100 == 0 or step == args.steps:
            torch.save({"backbone": model.backbone.state_dict(), "head": model.projection_head.state_dict(),
                        "step": step, "config": config}, args.output_dir / "checkpoint.pt")
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"time limit reached at step {step}")
            break
    log.close()
    (args.output_dir / "done.json").write_text(json.dumps(
        {"steps": step, "minutes": (time.time() - t0) / 60, "objective": args.objective}))
    print(f"DONE {args.output_dir} steps={step} minutes={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
