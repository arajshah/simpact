"""Goal vectors store the metric hand and peg pose directly."""

from __future__ import annotations

import numpy as np

import anygoal_goal_pose as goal
import anygoal_state_estimator as estimator


def test_metric_targets_ignore_quaternion_sign_and_pack_back():
    state = np.zeros((2, 44))
    state[:, 29:31] = (0.1, 0.4)
    state[:, 33:35] = (-0.2, 0.15)
    state[0, 31:33] = (np.cos(0.3), np.sin(0.3))
    state[1, 31:33] = -state[0, 31:33]
    targets = goal.metric_targets(state)
    np.testing.assert_allclose(targets[0, 2:4], targets[1, 2:4], atol=1e-6)
    mean = np.zeros(8)
    std = np.ones(8)
    packed = goal.goal_state_from_prediction(targets[0], mean, std)
    np.testing.assert_allclose(packed[29:31], (0.1, 0.4))
    delta = estimator.wrap_angle(estimator.planar_angles(packed[31:33]) - 0.6)
    assert abs(float(delta)) < 1e-6


def test_goal_decision_requires_two_centimeters_before_a_low_regret_counts():
    low = {"mean": 0.08, "ci_low": 0.02, "ci_high": 0.14}
    assert goal.goal_decision(0.05, 0.01, low)["decision"] == "INCONCLUSIVE"
    assert goal.goal_decision(0.01, 0.015, low)["decision"] == "NO-GO"
    remaining = goal.goal_decision(0.01, 0.01, {"mean": 0.35, "ci_low": 0.22, "ci_high": 0.48})
    assert remaining["decision"] == "DIAGNOSIS"
