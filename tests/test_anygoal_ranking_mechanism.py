"""Decision rules for the Insert-Peg ranking mechanism.

These tests do not load the checkpoint, the HDF5 file, or MuJoCo.
"""

from __future__ import annotations

import numpy as np
import pytest

import anygoal_ranking_mechanism as mechanism


def test_candidate_kinds_match_the_bank_construction():
    kinds = mechanism.candidate_kinds()
    assert kinds[0] == "recorded"
    assert list(kinds[1:8]) == ["perturbation"] * 7
    assert list(kinds[8:]) == ["transplant"] * 8
    with pytest.raises(ValueError):
        mechanism.candidate_kinds(4)


def test_support_labels_keep_recombined_actions_out_of_the_observed_set():
    assert mechanism.support_label(trajectory_exact=True, sequence_in_training=True) == "observed_continuation"
    assert (
        mechanism.support_label(trajectory_exact=False, sequence_in_training=True)
        == "recombined_expert_action"
    )
    assert mechanism.support_label(trajectory_exact=False, sequence_in_training=False) == "unobserved_action"
    with pytest.raises(ValueError):
        mechanism.support_label(trajectory_exact=True, sequence_in_training=False)


def test_pose_ball_requires_hand_object_and_joints_together():
    distance = {
        "hand_m": np.array([0.01, 0.01]),
        "object_m": np.array([0.01, 0.20]),
        "max_joint_rad": np.array([0.01, 0.01]),
    }
    assert mechanism.in_pose_ball(distance).tolist() == [True, False]


def test_pair_order_counts_ties_as_not_correct():
    success = np.zeros((2, 16), dtype=bool)
    success[0, 0] = True
    success[1, :] = True
    cost = np.ones((2, 16), dtype=np.float64)
    cost[0, 0] = 0.0
    counts = mechanism.pair_order_counts(success, cost)
    assert counts["total"][0] == 15
    assert counts["accuracy"][0] == pytest.approx(1.0)
    assert np.isnan(counts["accuracy"][1])
    tied = cost.copy()
    tied[0, 1] = 0.0
    tied_counts = mechanism.pair_order_counts(success, tied)
    assert tied_counts["ties"][0] == 1
    assert tied_counts["correct"][0] == 14
    assert tied_counts["accuracy"][0] == pytest.approx(14 / 15)


def test_effect_alignment_separates_silence_from_a_wrong_direction():
    observed = np.zeros((1, 3, 2))
    observed[0, 1] = np.array([0.0, 4.0])
    observed[0, 2] = np.array([0.0, 4.0])
    silent = np.zeros_like(observed)
    silent_result = mechanism.effect_alignment(silent, observed, min_observed_norm=1.0)
    assert not np.any(silent_result["usable"])
    wrong = observed.copy()
    wrong[0, 1] = np.array([4.0, 0.0])
    wrong[0, 2] = np.array([4.0, 0.0])
    wrong_result = mechanism.effect_alignment(wrong, observed, min_observed_norm=1.0)
    assert wrong_result["cosine"][0, 1] == pytest.approx(0.0)
    assert wrong_result["parallel"][0, 1] == pytest.approx(0.0)
    assert wrong_result["predicted_norm"][0, 1] == pytest.approx(4.0)
    right = mechanism.effect_alignment(observed, observed, min_observed_norm=1.0)
    assert right["cosine"][0, 1] == pytest.approx(1.0)
    assert right["parallel"][0, 1] == pytest.approx(4.0)
    assert not right["usable"][0, 0]


def test_uniform_regret_is_the_expected_miss_rate():
    success = np.zeros((2, 16), dtype=bool)
    success[0, :4] = True
    regret = mechanism.uniform_regret(success)
    assert regret[0] == pytest.approx(1.0 - 0.25)
    assert regret[1] == pytest.approx(0.0)
    selected = np.array([4, 0])
    gap = mechanism.ranking_gap(success, selected)
    assert gap["regret"][0] == 1.0
    assert gap["regret_minus_uniform"][0] == pytest.approx(1.0 - (1.0 - 0.25))


def test_coverage_gate_fires_on_a_recorded_failure_with_a_small_error_gap():
    recorded_regret = {"defined": True, "mean": 0.76, "ci_low": 0.62, "ci_high": 0.88}
    recorded_sse = {"defined": True, "mean": 50.0, "ci_low": 40.0, "ci_high": 60.0}
    small_gap = {"defined": True, "mean": 1.0, "ci_low": -1.0, "ci_high": 3.0}
    large_gap = {"defined": True, "mean": 40.0, "ci_low": 30.0, "ci_high": 50.0}
    quiet = {"defined": True, "mean": 0.04, "ci_low": 0.0, "ci_high": 0.10}
    assert mechanism.coverage_conclusion(
        recorded_regret=recorded_regret, sse_gap=small_gap, recorded_sse=recorded_sse
    ) == "present_on_recorded_transitions"
    assert mechanism.coverage_conclusion(
        recorded_regret=recorded_regret, sse_gap=large_gap, recorded_sse=recorded_sse
    ) == "present_on_recorded_transitions_with_larger_gap"
    assert mechanism.coverage_conclusion(
        recorded_regret=quiet, sse_gap=large_gap, recorded_sse=recorded_sse
    ) == "concentrated_off_the_recorded_continuation"


def test_accumulation_gate_requires_the_early_ranking_to_fail():
    bad = {"defined": True, "mean": 0.70, "ci_low": 0.55, "ci_high": 0.82}
    fine = {"defined": True, "mean": 0.04, "ci_low": 0.0, "ci_high": 0.08}
    early_sse = {"defined": True, "mean": 40.0}
    late_sse = {"defined": True, "mean": 44.0}
    grown = {"defined": True, "mean": 80.0}
    assert mechanism.accumulation_conclusion(
        early_regret=bad, late_regret=bad, early_sse=early_sse, late_sse=late_sse
    ) == "present_at_first_block"
    assert mechanism.accumulation_conclusion(
        early_regret=fine, late_regret=bad, early_sse=early_sse, late_sse=grown
    ) == "emerges_with_horizon"


def test_goal_cost_is_the_sum_of_squares_against_one_goal():
    predicted = np.ones((2, 16, 4))
    goal = np.zeros((2, 4))
    cost = mechanism.goal_costs(predicted, goal)
    assert cost.shape == (2, 16)
    assert cost[0, 3] == pytest.approx(4.0)


def test_decision_text_stops_when_both_gates_name_the_first_block():
    support = {
        "coverage_conclusion": "present_on_recorded_transitions",
        "effect_cosine_when_outcome_moves": {
            "defined": True, "mean": 0.0, "ci_low": -0.1, "ci_high": 0.1
        },
        "effect_predicted_norm_when_outcome_moves": {
            "defined": True, "mean": 2.0, "ci_low": 1.5, "ci_high": 2.5
        },
    }
    text = mechanism._decision_paragraph(
        support, {"accumulation_conclusion": "present_at_first_block"}
    )
    assert text.startswith("Next action: stop.")
    literature = mechanism._literature_paragraph(support)
    assert "unrelated" in literature
    assert "not near zero" in literature
