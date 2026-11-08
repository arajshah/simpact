"""Numerical checks for the AnyGoal contact-action diagnostic.

These tests do not download checkpoints or start MuJoCo. They lock the
pre-registered bank, regret, bootstrap, and decision rules.
"""

from __future__ import annotations

import numpy as np
import pytest

import anygoal_contact_diagnostic as audit


def _episodes(lengths):
    return [
        {"episode_id": episode_id, "length": length, "offset": 0, "store_index": episode_id}
        for episode_id, length in lengths
    ]


def test_episode_order_is_deterministic_and_exclusions_are_recorded():
    records = _episodes([(3, 20), (1, 80), (2, 36), (4, 100)])
    chosen, excluded = audit.select_extract_episodes(records, n_states=2)
    assert [item["episode_id"] for item in chosen] == [1, 2]
    assert excluded == [
        {"episode_id": 3, "length": 20, "reason": "length 20 < 36 for start 10"}
    ]
    with pytest.raises(RuntimeError):
        audit.select_extract_episodes(records, n_states=4)


def test_duplicate_episode_ids_fail():
    records = _episodes([(1, 80), (1, 80)])
    with pytest.raises(ValueError, match="Duplicate"):
        audit.select_extract_episodes(records, n_states=1)


def test_action_bank_is_seeded_clipped_and_taken_from_training_chunks():
    rng = np.random.default_rng(0)
    recorded = np.linspace(-0.2, 0.2, audit.HORIZON_STEPS * audit.ACTION_DIM).reshape(
        audit.HORIZON_STEPS, audit.ACTION_DIM
    )
    train = [
        np.full((40, audit.ACTION_DIM), fill, dtype=np.float64)
        for fill in (0.1, -0.2, 0.3)
    ]
    low = np.full(audit.ACTION_DIM, -0.15)
    high = np.full(audit.ACTION_DIM, 0.15)
    std = np.arange(1, audit.ACTION_DIM + 1, dtype=np.float64)
    bank, meta = audit.build_action_bank(recorded, train, low, high, std, rng)
    assert bank.shape == (16, 25, 5)
    assert np.all(bank[0] == np.clip(recorded, low, high))
    assert [item["kind"] for item in meta[:8]] == ["recorded"] + ["perturbation"] * 7
    assert [item["kind"] for item in meta[8:]] == ["train_chunk"] * 8
    assert np.all(bank >= low) and np.all(bank <= high)
    starts = [(item["train_episode_list_index"], item["start"]) for item in meta[8:]]
    assert starts == sorted(starts)
    for item in meta[8:]:
        source = train[item["train_episode_list_index"]][
            item["start"] : item["start"] + audit.HORIZON_STEPS
        ]
        np.testing.assert_allclose(bank[meta.index(item)], np.clip(source, low, high))
    again, _ = audit.build_action_bank(
        recorded, train, low, high, std, np.random.default_rng(0)
    )
    np.testing.assert_allclose(bank, again)


def test_action_blocks_follow_the_training_scaler_and_frameskip():
    raw = np.arange(16 * 25 * 5, dtype=np.float64).reshape(16, 25, 5)
    mean = np.arange(5, dtype=np.float64)
    scale = np.arange(1, 6, dtype=np.float64)
    blocked = audit.block_normalized_actions(raw, mean, scale)
    assert blocked.shape == (16, 5, 25)
    manual = ((raw - mean) / scale).reshape(16, 5, 5, 5).reshape(16, 5, 25)
    np.testing.assert_allclose(blocked, manual)
    # The first block is the first five raw actions, not a stride-5 subsample.
    np.testing.assert_allclose(blocked[0, 0, :5], (raw[0, 0] - mean) / scale)


def test_goal_criterion_matches_the_released_tolerances():
    goal = np.zeros(44, dtype=np.float64)
    goal[[31, 35, 42]] = 1.0
    current = goal.copy()
    current[29] = 0.03
    current[34] = 0.03
    errors = audit.goal_errors(goal, current)
    assert audit.full_success(errors)
    assert audit.object_success(errors)
    current[33] = 0.05
    errors = audit.goal_errors(goal, current)
    assert not audit.full_success(errors)
    assert not audit.object_success(errors)
    assert audit.full_success(audit.goal_errors(goal, goal))


def test_regret_is_one_only_when_a_successful_candidate_is_missed():
    success = np.array([0, 1, 0, 0] + [0] * 12, dtype=bool)
    assert audit.opportunity_regret(success, 0) == 1
    assert audit.opportunity_regret(success, 1) == 0
    none = np.zeros(16, dtype=bool)
    assert audit.opportunity_regret(none, 3) == 0
    costs = np.array([0.4, 0.2, 0.2, 0.9] + [1.0] * 12)
    selected, ties = audit.select_lowest_cost(costs)
    assert selected == 1
    assert ties == 2
    errors = np.arange(16, dtype=np.float64)
    utility, n_success = audit.select_true_utility(success, errors)
    assert utility == 1
    assert n_success == 1
    assert audit.opportunity_regret(success, utility) == 0


def test_bootstrap_interval_collapses_for_a_constant_and_covers_a_mean():
    constant = audit.bootstrap_mean(np.ones(20))
    assert constant["mean"] == 1.0
    assert constant["ci_low"] == 1.0
    assert constant["ci_high"] == 1.0
    values = np.array([0, 1, 1, 0, 1, 0], dtype=np.float64)
    summary = audit.bootstrap_mean(values, n_resamples=2000, seed=7)
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    with pytest.raises(ValueError):
        audit.bootstrap_mean(np.array([1.0, np.nan]))


def test_state_checkpoint_guard_rejects_a_44d_vector_for_a_33d_model():
    audit.assert_visual_checkpoint_interface(33, include_velocities=False)
    audit.assert_visual_checkpoint_interface(44, include_velocities=True)
    with pytest.raises(RuntimeError, match="44-d"):
        audit.assert_visual_checkpoint_interface(44, include_velocities=False)
    with pytest.raises(RuntimeError, match="state_dim"):
        audit.assert_visual_checkpoint_interface(33, include_velocities=True)


def test_decision_rules_follow_the_memorandum_gates():
    low_regret = {"mean": 0.04, "ci_low": 0.0, "ci_high": 0.10}
    high_regret = {"mean": 0.42, "ci_low": 0.22, "ci_high": 0.60}
    reduction = {"mean": 0.30, "ci_low": 0.12, "ci_high": 0.48}
    no_reduction = {"mean": 0.02, "ci_low": -0.08, "ci_high": 0.12}
    pending = {
        "checks": audit.section_8_checks(
            replay_pass_rate=1.0,
            solvability=0.9,
            regret_p=high_regret,
            regret_reduction=reduction,
        )
    }
    assert audit.interpret_stage_b(pending)["decision"] == "PENDING-STAGE-C"
    scoring = {
        "checks": audit.section_8_checks(
            replay_pass_rate=1.0,
            solvability=0.9,
            regret_p=high_regret,
            regret_reduction=no_reduction,
        )
    }
    assert audit.interpret_stage_b(scoring)["decision"] == "NO-GO"
    assert "goal scoring" in audit.interpret_stage_b(scoring)["reason"]
    replay = {
        "checks": audit.section_8_checks(
            replay_pass_rate=0.5,
            solvability=0.9,
            regret_p=high_regret,
            regret_reduction=reduction,
        )
    }
    assert "restoration" in audit.interpret_stage_b(replay)["reason"]
    weak = {
        "checks": audit.section_8_checks(
            replay_pass_rate=1.0,
            solvability=0.9,
            regret_p=low_regret,
            regret_reduction=no_reduction,
        )
    }
    assert "Hypothesis H fails" in audit.interpret_stage_b(weak)["reason"]


def test_result_text_keeps_dataset_goal_and_native_rates_distinct():
    text = "{'success_rate': 28.0} {'success_rate': 28.0} {'success_rate': 6.0}"
    assert audit._success_rates_from_text(text) == (28.0, 6.0)


def test_grasp_release_index_follows_the_closed_grip_command():
    grip = np.array([0.4] * 5 + [0.1])
    assert audit.grasp_release_index(grip) == 5
    assert audit.grasp_release_index(np.full(4, 0.4)) == 4


def test_numpy_success_string_parses_to_booleans():
    text = "[False  True  True False\n  True]"
    assert audit.parse_bool_vector(text) == [False, True, True, False, True]
    assert audit.parse_bool_vector([False, True]) == [False, True]


def test_spearman_is_one_for_a_monotone_pair_and_nan_for_ties():
    values = np.arange(16, dtype=np.float64)
    assert audit.spearman(values, values * 2.0 + 1.0) == pytest.approx(1.0)
    assert np.isnan(audit.spearman(np.ones(16), values))
