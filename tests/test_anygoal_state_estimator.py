"""Split and state-packing checks for the visual state estimator."""

from __future__ import annotations

import numpy as np
import pytest

import anygoal_state_estimator as estimator


def test_evaluation_episodes_cannot_enter_fit_splits():
    assert estimator.assign_split(0) == "eval"
    assert estimator.assign_split(49) == "eval"
    assert estimator.assign_split(50) == "train"
    assert estimator.assign_split(1299) == "train"
    assert estimator.assign_split(1300) == "val"
    assert estimator.assign_split(1460) == "val"
    with pytest.raises(RuntimeError, match="Evaluation"):
        estimator.assert_fit_episodes(np.array([50, 0]), "train")
    with pytest.raises(RuntimeError, match="other splits"):
        estimator.assert_fit_episodes(np.array([1300]), "train")
    estimator.assert_fit_episodes(np.arange(50, 60), "train")


def test_unit_pairs_and_packed_state_round_trip_angles():
    angle = np.array([0.3, -1.2, 2.4, 0.0, 1.0, -0.4, 0.7, -2.0])
    pairs = np.stack([np.sin(angle), np.cos(angle)], axis=1).reshape(-1)
    normalized = estimator.normalize_unit_pairs(pairs * 3.0)
    np.testing.assert_allclose(normalized, pairs, atol=1e-8)
    state = np.zeros((2, 44))
    state[:, :16] = pairs
    state[0, 35:37] = (np.cos(0.4), np.sin(0.4))
    state[1, 35:37] = -state[0, 35:37]
    targets = estimator.regression_targets(state)
    np.testing.assert_allclose(targets[0, 26:28], targets[1, 26:28], atol=1e-6)
    packed = estimator.pack_state(targets[0], np.zeros(35), np.ones(35))
    np.testing.assert_allclose(estimator.joint_angles(packed), angle, atol=1e-8)
    delta = estimator.wrap_angle(estimator.planar_angles(packed[35:37]) - 0.8)
    assert abs(float(delta)) < 1e-6
    assert packed[24:29].sum() == 0.0


def test_decision_refuses_an_inaccurate_estimator_even_with_low_regret():
    low = {"mean": 0.02, "ci_low": 0.0, "ci_high": 0.08}
    refused = estimator.estimator_decision(0.19, 0.04, low)
    assert refused["decision"] == "INCONCLUSIVE"
    closed = estimator.estimator_decision(0.01, 0.015, low)
    assert closed["decision"] == "NO-GO"
    remaining = estimator.estimator_decision(
        0.01, 0.01, {"mean": 0.40, "ci_low": 0.28, "ci_high": 0.52}
    )
    assert remaining["decision"] == "DIAGNOSIS"
    wide = estimator.estimator_decision(
        0.01, 0.01, {"mean": 0.08, "ci_low": 0.0, "ci_high": 0.22}
    )
    assert wide["decision"] == "INCONCLUSIVE"


def test_frame_choice_includes_the_contact_frame_and_stays_inside_the_episode():
    frames = estimator.choose_frames(1000, 17)
    assert 17 in frames
    assert len(frames) <= estimator.FRAMES_PER_EPISODE
    assert min(frames) >= 0 and max(frames) < 1000
    assert estimator.choose_frames(3, 2) == [0, 1, 2]
