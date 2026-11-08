"""The proprioceptive baseline may replace only the measured arm."""

from __future__ import annotations

import numpy as np

import anygoal_proprio_baseline as proprio


def test_current_proprio_keeps_the_visual_object_estimate():
    estimated = np.zeros((2, 44))
    measured = np.ones((2, 44))
    estimated[:, 33:40] = 0.3
    estimated[:, 40:44] = -0.2
    mixed = proprio.apply_current_proprio(estimated, measured, zero_object_velocity=False)
    assert np.all(mixed[:, 0:24] == 1.0)
    assert np.all(mixed[:, 29:33] == 1.0)
    assert np.all(mixed[:, 33:40] == 0.3)
    assert np.all(mixed[:, 40:44] == -0.2)
    np.testing.assert_array_equal(estimated[:, 0:24], 0.0)
    zeroed = proprio.apply_current_proprio(estimated, measured, zero_object_velocity=True)
    assert np.all(zeroed[:, 37:40] == 0.0)
    assert np.all(zeroed[:, 33:35] == 0.3)


def test_proprio_decision_closes_only_when_the_interval_is_small():
    closed = proprio.proprio_decision({"mean": 0.08, "ci_low": 0.02, "ci_high": 0.14})
    assert closed["decision"] == "NO-GO"
    remaining = proprio.proprio_decision({"mean": 0.40, "ci_low": 0.28, "ci_high": 0.52})
    assert remaining["decision"] == "DIAGNOSIS"
    wide = proprio.proprio_decision({"mean": 0.12, "ci_low": 0.04, "ci_high": 0.22})
    assert wide["decision"] == "INCONCLUSIVE"
