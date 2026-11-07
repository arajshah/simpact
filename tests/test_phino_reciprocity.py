"""Feasibility checks for the planar spring-reciprocity fixtures.

Tolerances are the pre-registered usefulness rule in
``phino_reciprocity_feasibility``: 25 µm, or 2% of the larger displacement,
whichever is larger. They are not adjusted to the observed error.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import phino_reciprocity_feasibility as R


def test_analytical_compliance_is_symmetric_and_finite():
    for spec in R.all_specs():
        compliance = R.privileged_compliance(spec)
        assert compliance.shape[0] == 2 * len(spec.ports)
        assert np.allclose(compliance, compliance.T, atol=1e-12)
        assert np.all(np.isfinite(compliance))
        assert np.all(np.linalg.eigvalsh(compliance) >= -1e-10)


def test_utility_uses_the_preregistered_weights():
    goal = np.array([0.0, 1.0])
    score, parts = R.utility(np.array([0.0, R.ALIGN_TARGET_M]), np.array([R.PROTECT_DEADBAND_M, 0.0]), 0.05, goal)
    assert parts["align"] == pytest.approx(1.0)
    assert parts["protect"] == pytest.approx(0.0)
    assert score == pytest.approx(1.0 - R.W_FORCE * (0.05 / R.FORCE_REFERENCE_N))
    saturated, _ = R.utility(
        np.array([0.0, R.ALIGN_TARGET_M]),
        np.array([R.PROTECT_DEADBAND_M + R.PROTECT_SCALE_M, 0.0]),
        R.FORCE_REFERENCE_N,
        goal,
    )
    assert saturated == pytest.approx(1.0 - R.W_PROTECT - R.W_FORCE)
    wrong_way, _ = R.utility(np.array([0.0, -0.01]), np.zeros(2), R.FORCE_REFERENCE_N, goal)
    assert wrong_way == 0.0
    assert 0.0 <= score <= 1.0


def test_rank_reports_a_tie_separately_from_the_next_action():
    rows = [
        {"port": "a1", "direction": "+y", "magnitude": 0.05, "utility": 0.40},
        {"port": "a2", "direction": "+x", "magnitude": 0.15, "utility": 0.40},
        {"port": "a0", "direction": "+y", "magnitude": 0.05, "utility": 0.22},
    ]
    ranked = R.rank_actions(rows)
    assert ranked["gap_to_next"] == pytest.approx(0.0)
    assert ranked["gap_to_distinct"] == pytest.approx(0.18)
    assert len(ranked["tie"]) == 2


def test_deployable_baseline_does_not_read_hidden_stiffness():
    source = inspect.getsource(R.deployable_compliance)
    for token in ("jnt_stiffness", "xfrc_applied", "qpos", "mujoco"):
        assert token not in source
    visible_match = R.deployable_compliance(R.visible_geometry(R.straight_spec()))
    visible_hidden = R.deployable_compliance(R.visible_geometry(R.straight_spec(stiffness=8.0)))
    assert np.allclose(visible_match, visible_hidden)
    true_soft = R.privileged_compliance(R.straight_spec())
    true_stiff = R.privileged_compliance(R.straight_spec(stiffness=8.0))
    assert np.allclose(visible_match, true_soft)
    assert not np.allclose(true_soft, true_stiff)
    goal = R.goal_direction(R.port_position(R.straight_spec(), next(p for p in R.straight_spec().ports if p.name == "align")))
    public_choice = R.rank_actions(R.predict_action_utilities(visible_match, R.straight_spec(), goal))["best"]
    hidden_choice = R.rank_actions(R.predict_action_utilities(true_stiff, R.straight_spec(stiffness=8.0), goal))["best"]
    assert (public_choice["port"], public_choice["direction"], public_choice["magnitude"]) != (
        hidden_choice["port"],
        hidden_choice["direction"],
        hidden_choice["magnitude"],
    )


def test_equilibrium_is_reproducible_and_at_the_drawn_pose():
    fixture = R.Fixture(R.straight_spec())
    first = fixture.zero_positions()
    fixture._zero_positions = None
    second = fixture.zero_positions()
    for name in first:
        assert np.linalg.norm(first[name] - second[name]) < 1e-12
        drawn = R.port_position(R.straight_spec(), next(p for p in R.straight_spec().ports if p.name == name))
        assert np.linalg.norm(first[name] - drawn) < 1e-9
    assert fixture._zero_info["qvel_max"] < R.SETTLED_QVEL
    assert fixture._zero_info["qpos_max"] < 1e-10


def test_positive_control_reciprocity_and_transverse_response():
    spec = R.straight_spec()
    fixture = R.Fixture(spec)
    catalog = {}
    for port in spec.ports:
        for direction in R.BASIS_DIRECTIONS:
            catalog[(port.name, direction, R.PROBE_FORCE_N)] = fixture.displacement(
                port.name, R.PROBE_FORCE_N * R.DIRECTIONS[direction]
            )
    blocks = R.secant_from_catalog(catalog, spec, R.PROBE_FORCE_N)
    records = R.reciprocity_records(blocks, [port.name for port in spec.ports], R.PROBE_FORCE_N)
    assert records
    assert all(row["agrees"] for row in records)
    signal = catalog[("a7", "+y", R.PROBE_FORCE_N)][0]["align"][1]
    assert signal > 20 * R.DECISION_FLOOR_M
    measured = R.flat_compliance(blocks)
    linear = R.privileged_compliance(spec)
    relative = np.linalg.norm(measured - linear) / np.linalg.norm(linear)
    assert relative < R.RELATIVE_BLOCK_TOL
    repeat_a, _ = fixture.displacement("a7", R.PROBE_FORCE_N * R.DIRECTIONS["+y"])
    repeat_b, _ = fixture.displacement("a7", R.PROBE_FORCE_N * R.DIRECTIONS["+y"])
    assert np.linalg.norm(repeat_a["align"] - repeat_b["align"]) < R.DECISION_FLOOR_M


def test_small_load_transverse_linearity_and_large_load_failure():
    fixture = R.Fixture(R.straight_spec())
    small, _ = fixture.displacement("a7", R.PROBE_FORCE_N * R.DIRECTIONS["+y"])
    doubled, _ = fixture.displacement("a7", 2 * R.PROBE_FORCE_N * R.DIRECTIONS["+y"])
    opposite, _ = fixture.displacement("a7", -R.PROBE_FORCE_N * R.DIRECTIONS["+y"])
    transverse_sign = abs(opposite["align"][1] + small["align"][1])
    transverse_scale = abs(doubled["align"][1] - 2.0 * small["align"][1])
    assert transverse_sign <= R.DECISION_FLOOR_M
    assert transverse_scale <= R.DECISION_FLOOR_M
    heavy_fixture = R.Fixture(R.straight_spec(), settle_time=2.0)
    heavy, info = heavy_fixture.displacement("a7", R.BREAKDOWN_FORCE_N * R.DIRECTIONS["+y"])
    extrapolated = (R.BREAKDOWN_FORCE_N / R.PROBE_FORCE_N) * small["align"]
    heavy_err = float(np.linalg.norm(heavy["align"] - extrapolated))
    assert heavy_err > 0.01
    assert heavy_err > 10.0 * max(transverse_sign, transverse_scale, R.DECISION_FLOOR_M)
    assert info["qvel_max"] < R.SETTLED_QVEL
    assert not info["hit_joint_limit"]


def test_contact_stop_breaks_reciprocity_while_the_small_load_does_not():
    fixture = R.Fixture(R.straight_spec(), contact_port="protect", settle_time=2.0)
    soft, soft_info = fixture.displacement("a7", R.PROBE_FORCE_N * R.DIRECTIONS["+y"], require_settled=False)
    soft_back, soft_back_info = fixture.displacement("protect", R.PROBE_FORCE_N * R.DIRECTIONS["+y"], require_settled=False)
    hard, hard_info = fixture.displacement("a7", R.CONTACT_FORCE_N * R.DIRECTIONS["+y"], require_settled=False)
    hard_back, _ = fixture.displacement("protect", R.CONTACT_FORCE_N * R.DIRECTIONS["+y"], require_settled=False)
    soft_gap = abs(float(soft["protect"][1] - soft_back["a7"][1]))
    hard_gap = abs(float(hard["protect"][1] - hard_back["a7"][1]))
    assert soft_info["ncon"] == 0 and soft_back_info["ncon"] == 0
    assert soft_gap <= R.DECISION_FLOOR_M
    assert hard_info["ncon"] > 0
    assert hard_gap > 10.0 * R.DECISION_FLOOR_M
    assert hard_gap > 10.0 * soft_gap


def test_measured_utility_follows_the_formula_and_prefers_the_distal_push():
    spec = R.straight_spec()
    fixture = R.Fixture(spec)
    goal = R.goal_direction(R.port_position(spec, next(p for p in spec.ports if p.name == "align")))
    rows = []
    for port, direction, magnitude in (("a7", "+y", 0.05), ("a0", "+y", 0.05), ("a7", "+y", 0.15)):
        displacement, _ = fixture.displacement(port, magnitude * R.DIRECTIONS[direction])
        score, parts = R.utility(displacement["align"], displacement["protect"], magnitude, goal)
        rows.append({"port": port, "direction": direction, "magnitude": magnitude, "utility": score, **parts})
        assert 0.0 <= score <= 1.0
        assert np.isfinite(parts["progress_m"])
    ranked = R.rank_actions(rows)
    assert (ranked["best"]["port"], ranked["best"]["direction"], ranked["best"]["magnitude"]) == ("a7", "+y", 0.05)
    assert ranked["gap_to_distinct"] > 0.05
    linear = R.deployable_compliance(R.visible_geometry(spec))
    names = [port.name for port in spec.ports]
    j = names.index("a7")
    align = names.index("align")
    predicted = linear[2 * align : 2 * align + 2, 2 * j : 2 * j + 2] @ (0.05 * R.DIRECTIONS["+y"])
    measured = fixture.displacement("a7", 0.05 * R.DIRECTIONS["+y"])[0]["align"]
    assert np.linalg.norm(predicted - measured) < 0.02 * np.linalg.norm(predicted)


def test_score_probes_leave_action_action_blocks_unidentified():
    straight = R.unidentified_entries(R.straight_spec())
    ell = R.unidentified_entries(R.ell_spec())
    # On the straight beam the protected point is also a3, so pairs that touch a3 are identified.
    assert straight["n_unknown_pairs"] == 21
    assert ell["n_unknown_pairs"] == 28
    assert all("a3" not in pair for pair in straight["not_identified_action_pairs"])
