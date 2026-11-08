"""Decision rules for the Extract-Peg dynamics factorial.

These tests do not start MuJoCo or load the world-model checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

import anygoal_contact_diagnostic as audit
import anygoal_dynamics_factor as factor
import anygoal_standard_baselines as baselines


def _failure_row(**overrides):
    row = {
        "solvability": 0.98,
        "p_regret_defined": True,
        "p_regret_mean": 0.70,
        "p_regret_ci_low": 0.58,
        "reduction_mean": 0.64,
        "reduction_ci_low": 0.50,
    }
    row.update(overrides)
    return row


def _quiet_row(**overrides):
    row = {
        "solvability": 0.90,
        "p_regret_defined": True,
        "p_regret_mean": 0.04,
        "p_regret_ci_low": 0.00,
        "reduction_mean": 0.02,
        "reduction_ci_low": -0.04,
    }
    row.update(overrides)
    return row


def _primary(**training):
    conditions = {name: _quiet_row() for name in factor.PRIMARY_CONDITIONS}
    conditions["weld_25"] = _failure_row()
    conditions["evaluator_10"] = _quiet_row(**training)
    return conditions


def test_primary_conditions_cross_weld_and_duration_on_valid_models():
    by_name = {factor.CONDITIONS[name].name: factor.CONDITIONS[name] for name in factor.PRIMARY_CONDITIONS}
    assert [(by_name[name].weld, by_name[name].nstep) for name in factor.PRIMARY_CONDITIONS] == [
        (False, 10),
        (False, 25),
        (True, 10),
        (True, 25),
    ]
    assert by_name["evaluator_10"].base == "insert"
    assert by_name["evaluator_10"].use_env_step
    assert by_name["weld_25"].base == "extract"
    with pytest.raises(ValueError, match="no grasp weld"):
        factor.DynamicsCondition("bad", True, 10, "insert", False, False, "x", "bad")


def test_unsolvable_bank_has_zero_regret_and_conditional_mean_is_undefined():
    success = np.zeros((4, 16), dtype=bool)
    success[0, 3] = True
    success[1, 1] = True
    selected = np.array([0, 1, 2, 2])
    scores = factor.per_episode_scores(success, selected)
    assert scores["regret"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert scores["solvable"].tolist() == [True, True, False, False]
    for episode, choice in enumerate(selected):
        assert scores["regret"][episode] == audit.opportunity_regret(success[episode], int(choice))
    undefined = factor.conditional_bootstrap(scores["regret"], np.zeros(4, dtype=bool), seed=0)
    assert undefined["defined"] is False
    assert undefined["mean"] is None
    defined = factor.conditional_bootstrap(scores["regret"], scores["solvable"], seed=0)
    assert defined["n"] == 2
    assert defined["mean"] == pytest.approx(0.5)
    assert factor.regret_is_comparable(0.80)
    assert not factor.regret_is_comparable(0.79)


def test_alternate_goal_must_be_outside_the_reference_tolerance():
    distance = np.linspace(0.0, 0.15, 16)
    within = np.zeros(16, dtype=bool)
    within[:4] = True
    assert factor.alternate_candidate(distance, within) == 15
    assert factor.alternate_candidate(distance, np.ones(16, dtype=bool)) is None
    with pytest.raises(ValueError):
        factor.alternate_candidate(np.full(16, np.nan), within)


def test_latent_sse_matches_the_terminal_cost_reduction():
    left = np.arange(16 * 3, dtype=np.float64).reshape(16, 3)
    goal = np.array([1.0, 0.0, -1.0])
    manual = np.sum((left - goal) ** 2, axis=-1)
    np.testing.assert_allclose(factor.latent_sse(left, goal), manual)


def test_zero_velocity_clears_arm_and_object_rates_only():
    state = np.arange(44, dtype=np.float64)
    cleared = factor.zero_velocity_state(state)
    assert np.all(cleared[16:24] == 0.0)
    assert np.all(cleared[37:40] == 0.0)
    assert cleared[15] == 15
    assert cleared[24] == 24
    assert cleared[36] == 36
    assert cleared[40] == 40
    np.testing.assert_array_equal(state, np.arange(44))


def test_training_integrator_failure_requests_baselines_and_not_a_new_method():
    conditions = _primary()
    conditions["evaluator_10"] = _failure_row(solvability=0.88)
    decision = factor.interpret_dynamics(conditions)
    assert decision["decision"] == "BASELINES"
    assert "not a decision to invent" in decision["reason"]


def test_unsolvable_training_goal_is_not_a_ranking_failure():
    decision = factor.interpret_dynamics(_primary(solvability=0.10, p_regret_defined=False, p_regret_mean=None, p_regret_ci_low=None, reduction_mean=None, reduction_ci_low=None))
    assert decision["decision"] == "NO-GO"
    assert decision["training_bank_can_rank"] is False
    assert "0.80" in decision["reason"]
    assert "do not support" in decision["reason"]


def test_gap_that_appears_only_with_the_weld_is_a_transition_result():
    decision = factor.interpret_dynamics(_primary(solvability=0.92))
    assert decision["decision"] == "NO-GO"
    assert decision["corrected_prediction_failure"] is True
    assert decision["training_prediction_failure"] is False
    assert "training integrator" in decision["reason"]


def test_gap_closure_requires_removing_eighty_percent_of_regret():
    assert baselines.closes_eighty_percent(0.10, 0.64)
    assert not baselines.closes_eighty_percent(0.20, 0.64)
    assert baselines.closes_eighty_percent(0.0, 0.0)
    query = np.array([[0.0, 0.0], [2.0, 2.0]])
    pool = np.array([[2.0, 2.0], [0.1, -0.1], [5.0, 0.0]])
    assert baselines.nearest_neighbor_indices(query, pool).tolist() == [1, 0]


def test_baseline_gate_stops_when_dynamics_or_a_competent_method_explain_the_gap():
    explained = factor.interpret_baselines("NO-GO", oracle_regret=0.0, deployable_closes=False)
    assert explained["decision"] == "NO-GO"
    closed = factor.interpret_baselines("BASELINES", oracle_regret=0.0, deployable_closes=True)
    assert closed["decision"] == "NO-GO"
    assert "80%" in closed["reason"]
    missing = factor.interpret_baselines("BASELINES", oracle_regret=0.0, deployable_closes=None)
    assert missing["decision"] == "NO-GO"
    assert "fair deployable baseline" in missing["reason"]
    residual = factor.interpret_baselines("BASELINES", oracle_regret=0.0, deployable_closes=False)
    assert residual["decision"] == "DIAGNOSIS"
    assert "does not by itself justify" in residual["reason"]
    ceiling_fails = factor.interpret_baselines("BASELINES", oracle_regret=0.40, deployable_closes=False)
    assert ceiling_fails["decision"] == "NO-GO"
