"""CPU tests for the action-conditioned probe and dataset helpers."""
from __future__ import annotations

import numpy as np
import pytest

import phino_action_probe as probe
import phino_v3_observation_dataset as obs


def _cfgs(n_geoms=3, per=12, seed=0):
    rng = np.random.default_rng(seed)
    cfgs = []
    for g in range(n_geoms):
        for k in range(per):
            m, f = rng.uniform(0.25, 0.35), rng.uniform(0.2, 0.4)
            cfgs.append({"geometry": f"g{g}", "config": f"c{k}", "cluster": f"cl{k // 3}", "mass": m,
                         "friction": f, "clips": {}})
    return cfgs


def test_ridge_recovers_linear_signal_and_drops_constant_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    X[:, 4] = 3.0 + 1e-9 * rng.normal(size=60)   # numerically constant: must not explode
    y = X[:, 0] - 2 * X[:, 1]
    xte = rng.normal(size=(10, 5))
    xte[:, 4] = 50.0
    pred = probe.ridge_path(X, y, xte, np.array([1e-3]))[0]
    assert np.allclose(pred, xte[:, 0] - 2 * xte[:, 1], atol=1e-2)


def test_fit_predict_clips_to_training_label_range():
    X = np.arange(20, dtype=float)[:, None]
    y = np.linspace(-1.4, -1.0, 20)
    pred, alphas = probe.fit_predict(X, y, np.array(["a", "b"] * 10), np.array([[1e6]]))
    assert pred.shape == (1, 1) and alphas.shape == (1,)
    assert pred[0, 0] <= y.max() + probe.PRED_LOG_MARGIN + 1e-12


def test_splits_never_share_geometry_or_cluster_between_train_and_test(monkeypatch):
    cfgs = _cfgs()
    X = np.arange(len(cfgs), dtype=float)[:, None]       # feature = row index, to recover the split
    calls = []

    def spy(xtr, ytr, gtr, xte, alphas=probe.ALPHAS):
        calls.append((xtr[:, 0].astype(int), xte[:, 0].astype(int), set(gtr)))
        return np.zeros((len(xte), 2)), np.ones(2)
    monkeypatch.setattr(probe, "fit_predict", spy)
    probe.evaluate_split(cfgs, X, "logo")
    assert len(calls) == 3
    for tr, te, groups in calls:
        test_geoms = {cfgs[i]["geometry"] for i in te}
        assert len(test_geoms) == 1 and not test_geoms & {cfgs[i]["geometry"] for i in tr}
        assert groups == {cfgs[i]["geometry"] for i in tr}
    calls.clear()
    probe.evaluate_split(cfgs, X, "within")
    assert len(calls) == 3 * 4
    for tr, te, groups in calls:
        geom = {cfgs[i]["geometry"] for i in te}
        assert geom == {cfgs[i]["geometry"] for i in tr}
        assert not {cfgs[i]["cluster"] for i in te} & {cfgs[i]["cluster"] for i in tr}


def test_representations():
    traj = np.zeros((50, 13))
    traj[:, 3] = 1.0
    traj[:, 0] = np.linspace(0, 0.1, 50)
    traj[:10, 7] = 0.5
    s = probe.state_physics(traj)
    assert np.isclose(s[0], np.log(0.1 + 1e-4)) and np.isclose(s[1], np.log(0.5 + 1e-4))
    assert np.isclose(s[4], 10 * 0.05)
    assert probe.state_frames(traj).shape == (35,)
    clip = {"centroid_px": [[10, 10], [12, 10], [14, 9], [15, 9], [15, 9], [15, 9]],
            "mask_area_px": [100, 100, 110, 100, 100, 100]}
    ip = probe.image_position(clip)
    assert ip.shape == (15,) and ip[0] == 2 and np.isclose(ip[11], np.log(1.1))

    class Bank:
        def __call__(self, p):
            return np.full(1536, float(p[-1]))
    v = probe.visual_clip(Bank(), {"images": ["a0", "a1", "a2", "a3", "a4", "a5"]})
    assert v.shape == (5 * 1536,) and np.allclose(v[:1536], 1.0) and np.allclose(v[-1536:], 5.0)


def test_two_target_fit_matches_single_target_fits():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    Y = np.column_stack([X[:, 0] + 0.1 * rng.normal(size=40), -X[:, 2] + rng.normal(size=40)])
    g = np.array(list("abcd") * 10)
    xte = rng.normal(size=(5, 6))
    both, a2 = probe.fit_predict(X, Y, g, xte)
    for j in range(2):
        single, a1 = probe.fit_predict(X, Y[:, j], g, xte)
        assert np.allclose(single[:, 0], both[:, j]) and a1[0] == a2[j]


def test_summary_normalized_mae():
    rows = [{"target": "mass", "test_geometry": "g", "pred": 0.3, "true": 0.32, "baseline": 0.28},
            {"target": "mass", "test_geometry": "g", "pred": 0.25, "true": 0.25, "baseline": 0.3}]
    s = probe.summarize(rows)["mass"]
    assert np.isclose(s["mae"], 0.01) and np.isclose(s["nmae"], 0.01 / 0.045)


def test_camera_spec_covers_extent():
    spec = obs.camera_spec({"lo": np.array([0.0, -0.01, 0.0]), "hi": np.array([0.4, 0.01, 0.0])}, 0.05, "push", 45.0)
    half_width = spec["distance"] * np.tan(np.radians(22.5))
    assert half_width >= 0.2 + 0.05 and np.isclose(spec["offset"][0], 0.2)
    assert obs.frame_time(49) == pytest.approx(2.5)
