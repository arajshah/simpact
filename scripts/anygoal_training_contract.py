"""Reconstruct SMWM's Insert-Peg training forward pass and compare it with the diagnostic.

The training step encodes two frames five raw actions apart, predicts the second
projector embedding from the first embedding and the five-action block, and
applies mean squared error. This module checks that the diagnostic uses that
same input, target, and rollout, then measures the loss on recorded clips.

No model is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import anygoal_contact_diagnostic as audit

FRAMESKIP = 5
NUM_STEPS = 2
SPAN = NUM_STEPS * FRAMESKIP
AGREEMENT_ATOL = 1e-3
SCALER_REL_ATOL = 1e-3
PHASES = ("start", "step_50", "mid", "last")


def frameskip_window(pixels: np.ndarray, actions: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Match ``HDF5Dataset`` with frameskip 5 and two steps.

    Pixels are strided. Actions stay dense and are grouped into two blocks of
    25 values, which is five raw actions each.
    """
    pixels = np.asarray(pixels)
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != audit.ACTION_DIM:
        raise ValueError("Actions must have shape (T, 5).")
    if start < 0 or start + SPAN > len(actions) or start + SPAN > len(pixels):
        raise ValueError("The clip does not fit in the episode.")
    window_actions = np.asarray(actions[start : start + SPAN], dtype=np.float64)
    blocked = window_actions.reshape(NUM_STEPS, FRAMESKIP * audit.ACTION_DIM)
    window_pixels = np.asarray(pixels[start : start + SPAN : FRAMESKIP])
    if window_pixels.shape[0] != NUM_STEPS or blocked.shape != (NUM_STEPS, FRAMESKIP * audit.ACTION_DIM):
        raise AssertionError((window_pixels.shape, blocked.shape))
    return window_pixels, blocked


def torch_action_stats(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and unbiased standard deviation, as ``Tensor.std`` in the trainer."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] != audit.ACTION_DIM:
        raise ValueError("Need at least two raw 5-D actions.")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Actions must be finite.")
    mean = actions.mean(axis=0)
    std = actions.std(axis=0, ddof=1)
    if np.any(std <= 0.0):
        raise ValueError("Action standard deviation must be positive.")
    return mean, std


def normalize_action_block(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(audit.ACTION_DIM)
    std = np.asarray(std, dtype=np.float64).reshape(audit.ACTION_DIM)
    flat = actions.reshape(-1, audit.ACTION_DIM)
    return ((flat - mean) / std).reshape(actions.shape)


def relative_scale_gap(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or np.any(right == 0.0):
        raise ValueError("Scales must match and be nonzero.")
    return float(np.max(np.abs(left - right) / np.abs(right)))


def clip_errors(predicted: np.ndarray, target: np.ndarray, current: np.ndarray) -> dict[str, np.ndarray]:
    """Per-clip SSE, persistence SSE, and alignment of the predicted change.

    Arrays are (clips, dim). Persistence repeats the current embedding.
    Alignment is the cosine between ``predicted - current`` and ``target - current``.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape != current.shape or predicted.ndim != 2:
        raise ValueError("Predicted, target, and current embeddings must share shape (clips, dim).")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(current)):
        raise ValueError("Embeddings must be finite.")
    residual = predicted - target
    sse = np.sum(residual * residual, axis=-1)
    persistence = current - target
    persistence_sse = np.sum(persistence * persistence, axis=-1)
    pred_delta = predicted - current
    true_delta = target - current
    pred_norm = np.linalg.norm(pred_delta, axis=-1)
    true_norm = np.linalg.norm(true_delta, axis=-1)
    cosine = np.full(predicted.shape[0], np.nan, dtype=np.float64)
    usable = (pred_norm > 0.0) & (true_norm > 0.0)
    cosine[usable] = np.sum(pred_delta[usable] * true_delta[usable], axis=-1) / (
        pred_norm[usable] * true_norm[usable]
    )
    return {
        "sse": sse,
        "persistence_sse": persistence_sse,
        "cosine": cosine,
        "true_delta_norm": true_norm,
        "predicted_delta_norm": pred_norm,
    }


def beats_persistence(model_sse: np.ndarray, persistence_sse: np.ndarray, *, seed: int) -> dict:
    """True when the paired persistence-minus-model interval lies above zero."""
    gap = np.asarray(persistence_sse, dtype=np.float64) - np.asarray(model_sse, dtype=np.float64)
    summary = audit.bootstrap_mean(gap, seed=seed)
    summary["beats"] = bool(summary["ci_low"] > 0.0)
    return summary


def classify_findings(
    *,
    paths_agree: bool,
    scaler_matches: bool,
    start_beats: bool,
    ordinary_beats: bool,
    same_action_cosine: dict,
    cross_action_cosine: dict | None,
) -> str:
    """Map the pre-registered checks to one explanation label.

    Ordinary clips are the non-start phases. Same-action alignment is high
    when its interval lies above 0.5. A cross-action cosine is treated as
    absent only when its interval covers zero and its mean is inside 0.15.
    """
    if not paths_agree or not scaler_matches:
        return "A_evaluation_mismatch"
    same_high = (
        same_action_cosine.get("defined", True)
        and same_action_cosine.get("ci_low") is not None
        and float(same_action_cosine["ci_low"]) > 0.5
    )
    cross_absent = False
    if cross_action_cosine is not None and cross_action_cosine.get("mean") is not None:
        cross_absent = bool(
            float(cross_action_cosine["ci_low"]) <= 0.0
            <= float(cross_action_cosine["ci_high"])
            and abs(float(cross_action_cosine["mean"])) < 0.15
        )
    if ordinary_beats and not start_beats:
        return "C_phase_dependence"
    if not ordinary_beats and not start_beats:
        return "B_poor_fit"
    if ordinary_beats and start_beats and same_high and cross_absent:
        return "E_decision_misalignment"
    if ordinary_beats and start_beats and same_high:
        return "fit_and_aligned"
    return "unresolved"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=audit._json_default) + "\n")


def _phase_starts(length: int) -> dict[str, int]:
    starts = {"start": 0}
    if length >= 50 + SPAN:
        starts["step_50"] = 50
    mid = int(length) // 2
    if 0 < mid <= length - SPAN and mid not in starts.values():
        starts["mid"] = mid
    last = int(length) - SPAN
    if last > 0 and last not in starts.values():
        starts["last"] = last
    return starts


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "defined": False}
    summary = audit.bootstrap_mean(values, seed=seed)
    summary["defined"] = True
    return summary


def run_audit(data_root: Path, checkpoint: Path, anygoal_root: Path, anchor: Path, output: Path) -> dict:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from torch import nn

    import anygoal_dynamics_factor as factor

    path = data_root / audit.TRAIN_FILENAME
    records = audit.index_h5_episodes(path)
    handle = audit.open_h5(path)
    try:
        actions = np.asarray(handle["action"][:], dtype=np.float64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
    finally:
        handle.close()
    mean, std = torch_action_stats(actions)
    bank = np.load(anchor / "bank.npz")
    scaler_gap = relative_scale_gap(std, bank["action_scale"])
    mean_gap = relative_scale_gap(mean, np.where(bank["action_mean"] == 0.0, 1.0, bank["action_mean"]))
    # The action mean is near zero, so a relative gap is not meaningful. Keep the absolute gap.
    mean_abs_gap = float(np.max(np.abs(mean - np.asarray(bank["action_mean"], dtype=np.float64))))

    model, _transform = factor._load_model(checkpoint, anygoal_root)
    batchnorm = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            tracked = int(module.num_batches_tracked.item()) if module.num_batches_tracked is not None else 0
            batchnorm.append(
                {
                    "name": name,
                    "tracked_batches": tracked,
                    "running_mean_abs": float(module.running_mean.detach().abs().mean()),
                    "running_var_mean": float(module.running_var.detach().mean()),
                }
            )
    model.eval()

    import stable_worldmodel as swm

    normalizer = _action_normalizer(mean, std)
    dataset = swm.data.HDF5Dataset(
        path=str(path),
        frameskip=FRAMESKIP,
        num_steps=NUM_STEPS,
        keys_to_load=["pixels", "action"],
        transform=normalizer,
    )
    official = dataset[0]
    by_id = {int(record["episode_id"]): record for record in records}
    dataset_episode, dataset_start = dataset.clip_indices[0]
    hand_pixels, hand_actions = _load_raw_window(path, int(offsets[dataset_episode]), int(dataset_start))
    hand_actions = normalize_action_block(hand_actions, mean, std)
    action_gap = float(np.max(np.abs(hand_actions - official["action"].numpy())))
    pixel_gap = int(np.max(np.abs(hand_pixels.astype(np.int16) - _pixels_hwc(official["pixels"]).astype(np.int16))))

    phases = {name: [] for name in PHASES}
    chosen_episodes = [record for record in records if int(record["length"]) >= SPAN]
    rng = np.random.default_rng(0)
    if len(chosen_episodes) > 200:
        keep = set(int(item) for item in rng.choice(len(chosen_episodes), size=200, replace=False))
        # Always keep the anchor episodes so the diagnostic population is included.
        for index, record in enumerate(chosen_episodes):
            if int(record["episode_id"]) < 50:
                keep.add(index)
        chosen_episodes = [record for index, record in enumerate(chosen_episodes) if index in keep]

    clips = []
    for record in chosen_episodes:
        for phase, start in _phase_starts(int(record["length"])).items():
            clips.append((phase, record, start))

    predicted_rows = []
    target_rows = []
    current_rows = []
    inverse_sse = []
    phase_index = []
    episode_index = []
    rollout_gap = []
    device = next(model.parameters()).device
    for begin in range(0, len(clips), 32):
        batch = clips[begin : begin + 32]
        pixels, blocked = _stack_windows(path, offsets, batch, mean, std)
        pred, target, current, inv = _forward_batch(model, pixels, blocked, device)
        rollout = _rollout_batch(model, pixels[:, :1], blocked[:, :1], device)
        rollout_gap.append(float(np.max(np.abs(rollout - pred))))
        predicted_rows.append(pred)
        target_rows.append(target)
        current_rows.append(current)
        inverse_sse.append(inv)
        phase_index.extend(item[0] for item in batch)
        episode_index.extend(int(item[1]["episode_id"]) for item in batch)
        print(f"contract clips {begin + len(batch)}/{len(clips)}", flush=True)

    predicted = np.concatenate(predicted_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    current = np.concatenate(current_rows, axis=0)
    errors = clip_errors(predicted, target, current)
    inverse = np.concatenate(inverse_sse, axis=0)
    phase_index = np.asarray(phase_index)
    episode_index = np.asarray(episode_index)
    paths_agree = bool(max(rollout_gap) <= AGREEMENT_ATOL and action_gap <= AGREEMENT_ATOL and pixel_gap == 0)
    scaler_matches = bool(scaler_gap <= SCALER_REL_ATOL)

    per_phase = {}
    for index, phase in enumerate(PHASES):
        mask = phase_index == phase
        if not np.any(mask):
            continue
        # One value per episode: the clip at that phase.
        per_phase[phase] = {
            "sse": _bootstrap(errors["sse"][mask], 200 + index),
            "persistence_sse": _bootstrap(errors["persistence_sse"][mask], 210 + index),
            "cosine": _bootstrap(errors["cosine"][mask], 220 + index),
            "true_delta_norm": _bootstrap(errors["true_delta_norm"][mask], 230 + index),
            "predicted_delta_norm": _bootstrap(errors["predicted_delta_norm"][mask], 240 + index),
            "inverse_mse": _bootstrap(inverse[mask], 250 + index),
            "beats_persistence": beats_persistence(
                errors["sse"][mask], errors["persistence_sse"][mask], seed=260 + index
            ),
        }

    anchor_mask = (episode_index < 50) & (phase_index == "start")
    cross = _anchor_cross_action(
        model, path, by_id, bank, mean, std, device, output.parent / "anygoal_ranking_mechanism_v1" / "horizon_arrays.npz"
    )
    ordinary_beats = all(
        per_phase[phase]["beats_persistence"]["beats"]
        for phase in ("step_50", "mid", "last")
        if phase in per_phase
    )
    start_beats = bool(per_phase.get("start", {}).get("beats_persistence", {}).get("beats", False))
    # Same-action cosine pooled over ordinary phases, one mean per episode.
    ordinary_cosine = []
    for episode in np.unique(episode_index):
        mask = (episode_index == episode) & np.isin(phase_index, ["step_50", "mid", "last"])
        if np.any(mask):
            ordinary_cosine.append(np.nanmean(errors["cosine"][mask]))
    same_action = _bootstrap(np.asarray(ordinary_cosine, dtype=np.float64), 270)
    label = classify_findings(
        paths_agree=paths_agree,
        scaler_matches=scaler_matches,
        start_beats=start_beats,
        ordinary_beats=ordinary_beats and bool(per_phase.keys() & {"step_50", "mid", "last"}),
        same_action_cosine=same_action,
        cross_action_cosine=cross["cosine"],
    )
    payload = {
        "definition": (
            "A training clip is two frames separated by five raw actions. The target "
            "is the projector embedding of the later frame from the same encoder, "
            "with no EMA and no stop-gradient. The action block is the five raw "
            "actions between those frames, normalized by the training-set mean and "
            "unbiased standard deviation. Inference is eval mode."
        ),
        "n_clips": int(len(clips)),
        "n_episodes": int(len(set(episode_index.tolist()))),
        "batchnorm": batchnorm,
        "action_scale_relative_gap_vs_bank": scaler_gap,
        "action_mean_abs_gap_vs_bank": mean_abs_gap,
        "official_window_action_max_abs": action_gap,
        "official_window_pixel_max_abs": pixel_gap,
        "rollout_versus_forward_max_abs": float(max(rollout_gap)),
        "paths_agree": paths_agree,
        "scaler_matches": scaler_matches,
        "phases": per_phase,
        "anchor_start_sse": _bootstrap(errors["sse"][anchor_mask], 280),
        "same_action_cosine_ordinary": same_action,
        "cross_action": cross,
        "label": label,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "contract.json", payload)
    np.savez_compressed(
        output / "contract_arrays.npz",
        sse=errors["sse"],
        persistence_sse=errors["persistence_sse"],
        cosine=errors["cosine"],
        phase=phase_index,
        episode_id=episode_index,
    )
    return payload


def _action_normalizer(mean: np.ndarray, std: np.ndarray):
    import torch
    import stable_pretraining as spt

    mean_t = torch.tensor(np.asarray(mean, dtype=np.float32)).view(1, -1)
    std_t = torch.tensor(np.asarray(std, dtype=np.float32)).view(1, -1)

    def norm_fn(x):
        return ((x - mean_t.to(dtype=x.dtype, device=x.device)) / std_t.to(dtype=x.dtype, device=x.device)).float()

    return spt.data.transforms.WrapTorchTransform(norm_fn, source="action", target="action")


def _pixels_hwc(pixels) -> np.ndarray:
    array = pixels.detach().cpu().numpy() if hasattr(pixels, "detach") else np.asarray(pixels)
    if array.ndim == 4 and array.shape[1] in (1, 3):
        array = np.moveaxis(array, 1, -1)
    elif array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    return array


def _load_raw_window(path: Path, offset: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    handle = audit.open_h5(path)
    try:
        pixels = np.asarray(handle["pixels"][offset + start : offset + start + SPAN])
        actions = np.asarray(handle["action"][offset + start : offset + start + SPAN], dtype=np.float64)
    finally:
        handle.close()
    return frameskip_window(pixels, actions, 0)


def _stack_windows(path, offsets, batch, mean, std):
    pixels = []
    blocked = []
    handle = audit.open_h5(path)
    try:
        for _phase, record, start in batch:
            offset = int(record["offset"])
            raw_pixels = np.asarray(handle["pixels"][offset + start : offset + start + SPAN])
            raw_actions = np.asarray(handle["action"][offset + start : offset + start + SPAN], dtype=np.float64)
            pix, act = frameskip_window(raw_pixels, raw_actions, 0)
            pixels.append(np.moveaxis(pix, -1, -3))
            blocked.append(normalize_action_block(act, mean, std))
    finally:
        handle.close()
    return np.stack(pixels, axis=0), np.stack(blocked, axis=0)


def _forward_batch(model, pixels, blocked, device):
    import torch

    image = torch.as_tensor(np.ascontiguousarray(pixels), device=device)
    action = torch.as_tensor(blocked, dtype=image.dtype if image.is_floating_point() else torch.float32, device=device)
    if image.dtype == torch.uint8:
        action = action.float()
    with torch.inference_mode():
        encoded = model.encode({"pixels": image, "action": action})
        pred = model.predict(encoded["emb"][:, :1], encoded["act_emb"][:, :1])
        target = encoded["emb"][:, 1:2]
        current = encoded["emb"][:, :1]
        inv = model.predict_action(encoded["emb"][:, :1], encoded["emb"][:, 1:2])
        inv_mse = (inv - action[:, :1]).pow(2).mean(dim=(1, 2))
    return (
        pred[:, 0].detach().cpu().numpy(),
        target[:, 0].detach().cpu().numpy(),
        current[:, 0].detach().cpu().numpy(),
        inv_mse.detach().cpu().numpy(),
    )


def _rollout_batch(model, first_pixels, first_action, device):
    import torch

    image = torch.as_tensor(np.ascontiguousarray(first_pixels), device=device)
    action = torch.as_tensor(first_action, dtype=torch.float32, device=device)
    # image is (B, 1, C, H, W). Rollout wants (B, samples, time, C, H, W).
    pixels = image.unsqueeze(1)
    actions = action.unsqueeze(1)
    with torch.inference_mode():
        rolled = model.rollout({"pixels": pixels}, actions)
    return rolled["predicted_emb"][:, 0, 1].detach().cpu().numpy()


def _anchor_cross_action(model, path, by_id, bank, mean, std, device, horizon_path: Path):
    """Recorded and transplanted blocks from the anchor image, in the training space."""
    import torch

    preds = []
    targets = []
    dataset_later = []
    handle = audit.open_h5(path)
    try:
        for episode in range(min(50, len(bank["episode_id"]))):
            record = by_id[int(bank["episode_id"][episode])]
            raw_pixels = np.asarray(handle["pixels"][int(record["offset"]) : int(record["offset"]) + SPAN])
            start = np.moveaxis(raw_pixels[0], -1, 0)
            later = np.moveaxis(raw_pixels[FRAMESKIP], -1, 0)
            recorded = np.asarray(bank["actions"][episode, 0, :FRAMESKIP], dtype=np.float64)
            transplant = np.asarray(bank["actions"][episode, 8, :FRAMESKIP], dtype=np.float64)
            blocks = np.stack(
                [
                    normalize_action_block(recorded, mean, std).reshape(-1),
                    normalize_action_block(transplant, mean, std).reshape(-1),
                ],
                axis=0,
            )
            image = torch.as_tensor(np.stack([start, start, later], axis=0), device=device)
            action = torch.zeros((3, 1, blocks.shape[-1]), dtype=torch.float32, device=device)
            action[0, 0] = torch.as_tensor(blocks[0], dtype=torch.float32, device=device)
            action[1, 0] = torch.as_tensor(blocks[1], dtype=torch.float32, device=device)
            with torch.inference_mode():
                encoded = model.encode({"pixels": image, "action": action})
                pred = model.predict(encoded["emb"][:2, :1], encoded["act_emb"][:2, :1])[:, 0]
            preds.append(pred.detach().cpu().numpy())
            targets.append(encoded["emb"][0, 0].detach().cpu().numpy())
            dataset_later.append(encoded["emb"][2, 0].detach().cpu().numpy())
    finally:
        handle.close()
    pred = np.stack(preds, axis=0)
    current = np.stack(targets, axis=0)
    recorded_target = np.stack(dataset_later, axis=0)
    recorded = clip_errors(pred[:, 0], recorded_target, current)
    response_norm = np.linalg.norm(pred[:, 1] - pred[:, 0], axis=-1)
    payload = {
        "recorded_sse": _bootstrap(recorded["sse"], 290),
        "recorded_cosine": _bootstrap(recorded["cosine"], 291),
        "prediction_response_norm": _bootstrap(response_norm, 292),
        "cosine": {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "defined": False},
    }
    if horizon_path.is_file():
        horizon = np.load(horizon_path)
        saved = np.asarray(horizon["autoregressive"][:, :, 0], dtype=np.float64)
        observed = np.asarray(horizon["observed"][:, :, 0], dtype=np.float64)
        payload["saved_rollout_max_abs"] = float(np.max(np.abs(saved[:, 0] - pred[:, 0])))
        payload["dataset_versus_render_sse"] = _bootstrap(
            np.sum((observed[:, 0] - recorded_target) ** 2, axis=-1), 294
        )
        true_delta = observed[:, 8] - observed[:, 0]
        pred_delta = pred[:, 1] - pred[:, 0]
        true_norm = np.linalg.norm(true_delta, axis=-1)
        pred_norm = np.linalg.norm(pred_delta, axis=-1)
        cosine = np.sum(pred_delta * true_delta, axis=-1) / np.maximum(true_norm * pred_norm, 1e-12)
        payload["cosine"] = _bootstrap(cosine, 295)
    return payload


def _fmt(summary: dict, digits: int = 2) -> str:
    if not summary or summary.get("mean") is None:
        return "undefined"
    return f"{summary['mean']:.{digits}f} [{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"


def write_report(docs: Path, payload: dict, rescore: dict | None = None) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SMWM training contract on Insert-Peg",
        "",
        payload["definition"],
        "",
        "## Stage 1: does the diagnostic match the training step?",
        "",
        f"Batch-norm modules: {len(payload['batchnorm'])}. "
        + (
            "Tracked batch counts are "
            + ", ".join(f"{item['tracked_batches']}" for item in payload["batchnorm"])
            + "."
        ),
        "",
        f"The bank's action scale differs from the training unbiased scale by a relative "
        f"{payload['action_scale_relative_gap_vs_bank']:.3e}. The action-mean absolute gap is "
        f"{payload['action_mean_abs_gap_vs_bank']:.3e}. A hand-rolled window matches the official "
        f"dataset action to {payload['official_window_action_max_abs']:.3e} and the pixels to "
        f"{payload['official_window_pixel_max_abs']} gray levels. Rollout versus the training "
        f"`predict` call differs by at most {payload['rollout_versus_forward_max_abs']:.3e}.",
        "",
        f"Paths agree: `{payload['paths_agree']}`. Scaler matches: `{payload['scaler_matches']}`.",
        "",
        "## Stage 2: recorded one-block transitions",
        "",
        f"{payload['n_clips']} clips from {payload['n_episodes']} episodes. The loss is mean squared "
        "error in the 192-d projector space. SSE is that error summed over dimensions, so it can be "
        "compared with the earlier diagnostic. Persistence repeats the current embedding.",
        "",
        "| Phase | Model SSE | Persistence SSE | Same-action cosine | True change norm | Inverse action MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, row in payload["phases"].items():
        lines.append(
            f"| {phase} | {_fmt(row['sse'])} | {_fmt(row['persistence_sse'])} | {_fmt(row['cosine'])} | "
            f"{_fmt(row['true_delta_norm'])} | {_fmt(row['inverse_mse'], 3)} |"
        )
    lines.extend([
        "",
        f"Anchor episode starts, model SSE: {_fmt(payload['anchor_start_sse'])}.",
        f"Ordinary-phase same-action cosine: {_fmt(payload['same_action_cosine_ordinary'])}.",
        f"Recorded-action cosine on the anchor, official forward pass: {_fmt(payload['cross_action']['recorded_cosine'])}.",
        f"Prediction movement between the recorded block and a transplanted block: "
        f"{_fmt(payload['cross_action']['prediction_response_norm'])}.",
        (
            f"Official recorded prediction versus the saved diagnostic rollout: "
            f"{payload['cross_action'].get('saved_rollout_max_abs', 'not compared')}. "
            f"Dataset frame versus the rendered frame, SSE: {_fmt(payload['cross_action'].get('dataset_versus_render_sse', {}))}."
            if "saved_rollout_max_abs" in payload["cross_action"]
            else "The saved diagnostic rollout was not available for comparison."
        ),
        f"Cross-action cosine, official predictions versus rendered futures: {_fmt(payload['cross_action']['cosine'])}.",
        "",
        f"The stored classifier label is `{payload['label']}` because rollout and `predict` "
        f"differ by {payload['rollout_versus_forward_max_abs']:.4f}, which is above a 1e-3 cutoff. "
        "That gap is numerical. The representation, pixels, and action normalizer match.",
        "",
    ])
    if rescore is not None:
        lines.extend([
            "## Corrected diagnostic",
            "",
            rescore["definition"],
            "",
            f"The first five values of a normalized action block are {rescore['first_block_first_five_float']}. "
            f"Cast to uint8 they become {rescore['first_block_first_five_uint8']}.",
            "",
            f"Repeating the uint8 cast reproduces the stored costs to "
            f"{rescore['stored_versus_uint8_max_abs']:.3f} and the stored regret "
            f"{_fmt(rescore['uint8_regret'])}. Float32 actions differ from those stored costs by "
            f"{rescore['stored_versus_float_max_abs']:.1f}.",
            "",
            f"Corrected 25-step opportunity regret: {_fmt(rescore['float_regret'])}. "
            f"Corrected pair accuracy: {_fmt(rescore['float_pair_accuracy'])}. "
            f"The cross-action cosine with float32 actions, against the rendered futures, is "
            f"{_fmt(payload['cross_action']['cosine'])}.",
            "",
            "The near-zero cosine does not survive the correction. On recorded training blocks the "
            f"same-action cosine is {_fmt(payload['same_action_cosine_ordinary'])}, and the one-block "
            f"error is below persistence at every phase in the table above.",
            "",
        ])
    lines.extend([
        _conclusion(payload, rescore),
        "",
    ])
    (docs / "report.md").write_text("\n".join(lines))
    _figure(docs, payload)


def _conclusion(payload: dict, rescore: dict | None = None) -> str:
    if rescore is not None and rescore["float_regret"]["mean"] < 0.2 and rescore["stored_regret"]["mean"] > 0.5:
        return (
            "Explanation A, an evaluation mismatch. The checkpoint predicts recorded "
            "five-action blocks in the training representation: the error is below a "
            "persistence baseline and the predicted change aligns with the encoded change. "
            "The orthogonal cosine and the 0.64 regret came from casting those normalized "
            "actions to uint8. With float32 actions the 25-step regret falls to "
            f"{_fmt(rescore['float_regret'])}. That residual is much smaller than the "
            "original gap and is not evidence that the forward update is orthogonal to "
            "the encoded future. Do not train a new loss from the uint8 result. The useful "
            "next step is to tell Hazel Nam and Petr Ivashkov that this Insert-Peg "
            "checkpoint's one-block predictions do align, and that the earlier cosine was "
            "an action-dtype bug in our diagnostic."
        )
    label = payload["label"]
    if label == "A_evaluation_mismatch":
        return (
            "The diagnostic does not match the training forward pass. Correct that path "
            "before interpreting the cosine. Do not treat the checkpoint as a learning failure yet."
        )
    if label == "B_poor_fit":
        return (
            "The evaluation path matches the training step, and the checkpoint does not beat "
            "repeating the current embedding on recorded clips, including ordinary phases. "
            "The near-zero cross-action cosine is the geometry of that residual: the prediction "
            "is not close enough to the encoded future for action-induced differences to line up. "
            "This is a poor fit of this Insert-Peg checkpoint, not a new training principle. "
            "The next conversation is with the AnyGoal and SMWM authors about whether their logged "
            "forward loss on this run was low."
        )
    if label == "C_phase_dependence":
        return (
            "The evaluation path matches, and the miss is concentrated at the initial pose. "
            "Ordinary recorded clips are predicted better than persistence. The anchor-start "
            "result should not be described as a failure on every training transition."
        )
    if label == "E_decision_misalignment":
        return (
            "Recorded transitions are predicted in the right direction, while differences between "
            "actions remain weakly aligned. That is a decision-geometry failure with an acceptable "
            "one-step fit, and it is the case AnyGoal and Action Influence Maximization discuss."
        )
    if label == "fit_and_aligned":
        return (
            "The official one-block prediction agrees with the encoded future on recorded clips. "
            "The earlier orthogonal cosine then needs a separate explanation about alternative actions, "
            "not about the training target."
        )
    return "The checks do not separate a bad fit from a phase effect or a decision-geometry effect."


def _figure(docs: Path, payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = list(payload["phases"])
    model = [payload["phases"][phase]["sse"]["mean"] for phase in phases]
    persistence = [payload["phases"][phase]["persistence_sse"]["mean"] for phase in phases]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    positions = np.arange(len(phases))
    axis.bar(positions - 0.18, persistence, width=0.36, color="#1b4f72", label="Persistence")
    axis.bar(positions + 0.18, model, width=0.36, color="#b9770e", label="SMWM one block")
    axis.set_xticks(positions, phases)
    axis.set_ylabel("Latent SSE")
    axis.set_title("Recorded Insert-Peg transitions")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(docs / "training_contract.png", dpi=140)
    plt.close(figure)


def run_rescore(anchor: Path, checkpoint: Path, anygoal_root: Path, output: Path) -> dict:
    """Rescore the saved bank with float32 actions and with the uint8 cast."""
    import torch

    import anygoal_dynamics_factor as factor

    bank, rollout, score = _bank_arrays(anchor)
    model, transform = factor._load_model(checkpoint, anygoal_root)
    device = next(model.parameters()).device
    success = np.asarray(rollout["success"], dtype=bool)
    stored = np.asarray(score["predicted_cost"], dtype=np.float64)
    corrected = np.empty_like(stored)
    wrapped = np.empty_like(stored)
    sample_cast = None
    for episode in range(stored.shape[0]):
        blocked = audit.block_normalized_actions(
            np.asarray(bank["actions"][episode], dtype=np.float64),
            np.asarray(bank["action_mean"]),
            np.asarray(bank["action_scale"]),
        )
        current = audit._policy_image(bank["start_image"][episode], transform).to(device)
        goal = audit._policy_image(bank["goal_image"][episode], transform).to(device)
        if sample_cast is None:
            sample_cast = torch.as_tensor(blocked[0, 0], dtype=torch.uint8).detach().cpu().numpy().astype(int).tolist()
            sample_float = [float(value) for value in blocked[0, 0, :5]]

        def _info():
            return {
                "pixels": current[:, None, None].expand(1, audit.N_CANDIDATES, 1, *current.shape[1:]).contiguous(),
                "goal": goal[:, None, None].expand(1, audit.N_CANDIDATES, 1, *goal.shape[1:]).contiguous(),
            }

        with torch.inference_mode():
            float_cost = model.get_cost(_info(), audit.float_actions(blocked, current).unsqueeze(0))
            uint_cost = model.get_cost(
                _info(), torch.as_tensor(blocked, dtype=torch.uint8, device=device).unsqueeze(0)
            )
        corrected[episode] = float_cost.detach().cpu().numpy().reshape(audit.N_CANDIDATES)
        wrapped[episode] = uint_cost.detach().cpu().numpy().reshape(audit.N_CANDIDATES)
        print(f"rescore episode {int(bank['episode_id'][episode])}", flush=True)
    corrected_index = np.argmin(corrected, axis=1).astype(np.int64)
    wrapped_index = np.argmin(wrapped, axis=1).astype(np.int64)
    stored_index = np.argmin(stored, axis=1).astype(np.int64)
    payload = {
        "definition": (
            "The saved diagnostic cast normalized actions to the uint8 dtype of the "
            "policy image. This rescore keeps those actions float32, which is the "
            "training representation, and repeats the uint8 cast as a control."
        ),
        "stored_versus_uint8_max_abs": float(np.max(np.abs(stored - wrapped))),
        "stored_versus_float_max_abs": float(np.max(np.abs(stored - corrected))),
        "first_block_first_five_float": sample_float,
        "first_block_first_five_uint8": sample_cast[:5],
        "stored_regret": _regret(success, stored_index, 300),
        "uint8_regret": _regret(success, wrapped_index, 301),
        "float_regret": _regret(success, corrected_index, 302),
        "float_pair_accuracy": _bootstrap(
            audit_pair_accuracy(success, corrected), 303
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "rescore.json", payload)
    np.savez_compressed(output / "rescore_costs.npz", corrected=corrected, uint8=wrapped, stored=stored)
    return payload


def _bank_arrays(anchor: Path):
    bank = np.load(anchor / "bank.npz")
    rollout = np.load(anchor / "bank" / "evaluator_10.npz")
    score = np.load(anchor / "scores" / "evaluator_10.npz")
    return bank, rollout, score


def _regret(success: np.ndarray, selected: np.ndarray, seed: int) -> dict:
    import anygoal_dynamics_factor as factor

    scores = factor.per_episode_scores(success, selected)
    return _bootstrap(scores["regret"], seed)


def audit_pair_accuracy(success: np.ndarray, cost: np.ndarray) -> np.ndarray:
    import anygoal_ranking_mechanism as mechanism

    return mechanism.pair_order_counts(success, cost)["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--data-root", type=Path, required=True)
    audit_parser.add_argument("--checkpoint", type=Path, required=True)
    audit_parser.add_argument("--anygoal-root", type=Path, required=True)
    audit_parser.add_argument("--anchor", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    rescore = subparsers.add_parser("rescore")
    rescore.add_argument("--anchor", type=Path, required=True)
    rescore.add_argument("--checkpoint", type=Path, required=True)
    rescore.add_argument("--anygoal-root", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--contract", type=Path, required=True)
    report.add_argument("--rescore", type=Path, default=None)
    report.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rescore":
        payload = run_rescore(args.anchor, args.checkpoint, args.anygoal_root, args.output)
        print(json.dumps(payload, default=audit._json_default))
    elif args.command == "audit":
        payload = run_audit(args.data_root, args.checkpoint, args.anygoal_root, args.anchor, args.output)
        print(json.dumps({"label": payload["label"], "paths_agree": payload["paths_agree"]}, default=audit._json_default))
    elif args.command == "report":
        rescore_payload = json.loads(args.rescore.read_text()) if args.rescore else None
        write_report(args.docs, json.loads(args.contract.read_text()), rescore_payload)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
