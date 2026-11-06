"""Matched objectives for the action-conditioned baseline comparison.

The trajectory-similarity run is unchanged: dense PHINO SoftInfoNCE on one
intervention per batch. The two baselines share its clip, augmentation, and
leave-one-geometry-out rule.

* ``instance`` is symmetric InfoNCE. The positive pair is two augmented views
  of the same clip. Every other clip in the batch is a negative. The loss
  does not take a trajectory similarity.
* ``regression`` predicts training-fold standardized log mass and log friction
  from the same augmented clip. The standardizer is fit on the training
  geometries only. The head is discarded before the probe, as the contrastive
  projection head is.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceInfoNCE(nn.Module):
    """Symmetric instance-discrimination InfoNCE with PHINO's temperature.

    Temperature is a learnable log-parameter initialized at 0.07 and clamped
    to [0.01, 1], matching ``SoftInfoNCELoss``. The target is a one-hot match
    between the two views. There is no similarity-temperature and no dynamics
    target.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> dict:
        if z_i.ndim != 2 or z_i.shape != z_j.shape or z_i.shape[0] < 2:
            raise ValueError("expected two views with matching shape (B, D), B >= 2")
        logits = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(z_i.shape[0], device=z_i.device)
        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2j + loss_j2i) / 2
        return {
            "loss": loss,
            "loss_i2j": loss_i2j,
            "loss_j2i": loss_j2i,
            "temperature": self.temperature.detach(),
            "logits_mean": logits.diagonal().mean().detach(),
        }


class ClipRegressionHead(nn.Module):
    """Same MLP width as ``DynaCLIPProjectionHead``, without L2 normalization.

    The output is two unconstrained values, one for log mass and one for log
    friction. Normalization would make those targets inexpressible.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def two_view_regression_loss(z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error, averaged over the two augmented views."""
    if z1.shape != y.shape or z2.shape != y.shape or z1.ndim != 2 or z1.shape[1] != 2:
        raise ValueError("expected matching (B, 2) predictions and standardized log labels")
    return 0.5 * ((z1 - y).pow(2).mean() + (z2 - y).pow(2).mean())


def shared_view(x: torch.Tensor, brightness_u: torch.Tensor, contrast_u: torch.Tensor,
                crop_i: int, crop_j: int, mean: torch.Tensor, std: torch.Tensor,
                shift: int = 6, jitter: float = 0.10) -> torch.Tensor:
    """One colour jitter and one crop, applied to every frame of ``x``.

    ``x`` is float in [0, 1] with shape (F, 3, H, W). ``brightness_u`` and
    ``contrast_u`` are scalar ``torch.rand`` draws. Flips are not applied.
    """
    if x.ndim != 4 or shift < 0:
        raise ValueError("expected a clip (F, 3, H, W) and a non-negative shift")
    height, width = x.shape[-2:]
    x = x * (1 + jitter * (2 * brightness_u - 1))
    x = (x - x.mean()) * (1 + jitter * (2 * contrast_u - 1)) + x.mean()
    x = F.pad(x, (shift,) * 4, mode="replicate")
    x = x[..., crop_i:crop_i + height, crop_j:crop_j + width].clamp(0, 1)
    return (x - mean) / std


def training_rows(rows: list[dict], test_geometry: str) -> list[dict]:
    """Rows whose object is visible and whose geometry is not the held-out one."""
    visible = [r for r in rows if r["object_in_frame"]]
    geoms = {r["geometry_short"] for r in visible}
    if test_geometry not in geoms:
        raise ValueError(f"unknown geometry {test_geometry}; choose from {sorted(geoms)}")
    train = [r for r in visible if r["geometry_short"] != test_geometry]
    if not train:
        raise ValueError("no training rows")
    if any(r["geometry_short"] == test_geometry for r in train):
        raise ValueError("test geometry leaked into training")
    return train


def log_label_matrix(rows: list[dict]) -> np.ndarray:
    mass = np.array([float(r["mass"]) for r in rows], dtype=np.float64)
    friction = np.array([float(r["friction"]) for r in rows], dtype=np.float64)
    if np.any(mass <= 0) or np.any(friction <= 0):
        raise ValueError("mass and friction must be positive")
    labels = np.column_stack([np.log(mass), np.log(friction)])
    if not np.isfinite(labels).all():
        raise ValueError("non-finite log mass or log friction")
    return labels


def unique_configurations(rows: list[dict]) -> list[dict]:
    """One row per geometry and configuration. Labels must agree across actions."""
    chosen: dict[tuple, dict] = {}
    for row in rows:
        key = (row["geometry_short"], row["config"])
        if key in chosen:
            previous = chosen[key]
            if previous["mass"] != row["mass"] or previous["friction"] != row["friction"]:
                raise ValueError(f"conflicting labels for {key}")
            continue
        chosen[key] = row
    if not chosen:
        raise ValueError("no configurations")
    return list(chosen.values())


def fit_log_label_standardizer(rows: list[dict]) -> dict[str, np.ndarray]:
    """Mean and population std of log mass and log friction. Training rows only."""
    labels = log_label_matrix(rows)
    center = labels.mean(axis=0)
    scale = labels.std(axis=0)
    if np.any(scale < 1e-8):
        raise ValueError("training log labels have no variance")
    return {"mean": center, "std": scale}


def standardize_log_labels(rows: list[dict], stats: dict[str, np.ndarray]) -> np.ndarray:
    labels = log_label_matrix(rows)
    return (labels - stats["mean"]) / stats["std"]
