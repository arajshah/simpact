"""CPU checks for the spatial-diagnostic aggregators. No features or probe files."""
from __future__ import annotations

import numpy as np

import phino_spatial_diagnostic as diag


def test_varying_channel_r2_ignores_a_constant_channel():
    y = np.ones((4, 15))
    y[:, 0] = np.arange(4)
    pred = y.copy()
    pred[:, 1] = 100.0
    got = diag.varying_channel_r2(y, pred)
    assert got["n_channels"] == 1 and got["r2_mean"] == 1.0


def test_channel_r2_is_one_for_a_perfect_prediction_and_zero_for_the_mean():
    y = np.arange(30, dtype=float).reshape(2, 15)
    perfect = diag.channel_r2(y, y.copy())
    assert perfect["r2_mean"] == 1.0 and perfect["r2_dx"] == 1.0
    mean = diag.channel_r2(y, np.repeat(y.mean(0, keepdims=True), 2, axis=0))
    assert abs(mean["r2_mean"]) < 1e-9


def test_action_matrix_matches_visual_clip():
    class Bank:
        def __init__(self):
            self.f = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
            self.index = {f"p{i}": i for i in range(6)}

        def __call__(self, p):
            return self.f[self.index[p]]

    bank = Bank()
    images = [f"p{i}" for i in range(6)]
    cfgs = [{"clips": {"push": {"images": images}}}]
    got = diag.action_matrix(bank, cfgs, "push")
    ref = diag.probe.visual_clip(bank, {"images": images}).astype(np.float64)
    assert got.shape == (1, 20) and np.allclose(got[0], ref)


def test_mean_nmae_averages_only_the_requested_subsets():
    rows = {
        ("rep", "a", "logo", "mass"): {"n_interventions": 1, "nmae": 0.2},
        ("rep", "b", "logo", "mass"): {"n_interventions": 1, "nmae": 0.4},
        ("rep", "a+b", "logo", "mass"): {"n_interventions": 2, "nmae": 0.9},
        ("rep", "static", "logo", "mass"): {"n_interventions": 0, "nmae": 1.0},
    }
    got = diag.mean_nmae(rows, "rep", "logo", "mass", ["a", "static"])
    assert got == {1: 0.2}
