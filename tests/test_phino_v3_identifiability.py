"""Tests for the v3 simulation primitives and the Stage 3 identifiability pilot."""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

import phino_simpact_grid_v3 as v3
import phino_v3_identifiability_pilot as P
from test_export_phino_dataset import PHINO_ROOT, make_grid


class _V2Action:
    direction = np.array([1.0, 0.0, 0.0])
    force = 8.0
    torque = 0.2


def test_wrench_schedules_are_fixed_and_physics_independent():
    assert list(inspect.signature(v3.wrench).parameters) == ["action", "frame", "originals"]
    lo = v3.ACTION_BY_NAME["push_lo_1.6N"]
    assert np.allclose(v3.wrench(lo, 0, {})[0], [1.6, 0, 0]) and np.allclose(v3.wrench(lo, 3, {})[0], [1.6, 0, 0])
    assert np.allclose(v3.wrench(lo, 4, {})[0], 0) and np.allclose(v3.wrench(lo, 0, {})[1], 0)
    toss = v3.ACTION_BY_NAME["toss_4.0N"]
    assert np.allclose(v3.wrench(toss, 2, {})[0], [0, 0, 4.0])
    hi = v3.ACTION_BY_NAME["push_hi_3.2N"]
    assert np.allclose(v3.wrench(hi, 1, {})[0], [3.2, 0, 0]) and np.allclose(v3.wrench(hi, 2, {})[0], 0)
    ref = v3.ACTION_BY_NAME["push_x_v2"]
    f, tau = v3.wrench(ref, 0, {"push_x": _V2Action()})
    assert np.allclose(f, [0.8, 0, 0]) and np.isclose(tau[2], 0.02)
    assert all(a.magnitude <= 4.5 for a in v3.ACTIONS)


def test_start_relative_recording():
    raw = np.zeros((3, 13))
    half = np.radians(90) / 2
    raw[:, 3:7] = [np.cos(half), 0, 0, np.sin(half)]
    raw[:, 10] = 1.0
    raw[:, :3] = [[1, 2, 3]] * 3
    rel = v3.start_relative(raw, np.array([1, 2, 3.0]), raw[0, 3:7])
    assert np.allclose(rel[:, :3], 0) and np.allclose(rel[:, 3:7], [1, 0, 0, 0])
    assert np.allclose(rel[:, 10:13], [0, 1, 0], atol=1e-12)


def _slide_trace(v0: float, decel: float, dt: float, force_samples: int, n: int = 50) -> np.ndarray:
    tr = np.zeros((n, 13))
    tr[:, 3] = 1.0
    t = (np.arange(n) + 1) * dt
    speed = np.where(np.arange(n) < force_samples, v0,
                     np.maximum(v0 - decel * (t - t[force_samples - 1]), 0))
    tr[:, 7] = speed
    tr[:, 0] = np.cumsum(speed) * dt
    return tr


def test_push_features_recover_deceleration_and_flag_unresolved():
    f = P.action_features(_slide_trace(0.8, 2.0, 0.05, 2), "push_hi", 2, 1, 0.05)
    assert np.isclose(f["deceleration"], 2.0, rtol=1e-6) and f["n_decel"] >= 3
    short = P.action_features(_slide_trace(0.1, 2.0, 0.05, 2), "push_hi", 2, 1, 0.05)
    assert np.isnan(short["deceleration"])
    fine = P.action_features(_slide_trace(0.1, 2.0, 0.01, 10, 250), "push_hi", 2, 5, 0.01)
    assert np.isclose(fine["deceleration"], 2.0, rtol=1e-6)


def test_toss_features_parabolic_apex():
    dt = 0.05
    t = (np.arange(50) + 1) * dt
    tr = np.zeros((50, 13))
    tr[:, 3] = 1.0
    tr[:, 2] = np.maximum(0.2 - 4.9 * (t - 0.523) ** 2, 0)
    f = P.action_features(tr, "toss", 4, 1, dt)
    assert np.isclose(f["apex_height"], 0.2, atol=1e-6) and np.isclose(f["apex_time"], 0.523, atol=1e-6)


def test_visual_and_ideal_precision_models():
    dt, ppm = 0.05, 300.0
    pos = 1.0 / ppm
    assert np.isclose(P.sigma("final_displacement", "visual224", dt, ppm, 0.05, False, 0), np.sqrt(2) * pos)
    assert np.isclose(P.sigma("peak_speed", "visual224", dt, ppm, 0.05, False, 0), np.sqrt(2) * pos / dt)
    d3 = P.sigma("deceleration", "visual224", dt, ppm, 0.05, False, 3)
    d10 = P.sigma("deceleration", "visual224", dt, ppm, 0.05, False, 10)
    assert d10 < d3
    assert np.isinf(P.sigma("final_yaw", "visual224", dt, ppm, 0.05, True, 0))
    assert np.isclose(P.sigma("final_displacement", "ideal", dt, ppm, 0.05, False, 0),
                      P.IDEAL_FACTOR * np.sqrt(2) * pos)
    assert P.sigma("peak_speed", "visual224", 0.01, ppm, 0.05, False, 0) > \
        P.sigma("peak_speed", "visual224", 0.05, ppm, 0.05, False, 0)


def test_fisher_detects_degenerate_and_well_conditioned_designs():
    s = np.ones(2) * 0.01
    good = P.fisher(np.array([[1.0, 0.0], [0.0, 1.0]]), s)
    assert np.isclose(good["kappa"], 1.0) and np.isclose(good["sd_log_mass"], 0.01)
    parallel = P.fisher(np.array([[1.0, 0.5], [2.0, 1.0]]), s)
    assert parallel["kappa"] > 1e12 and np.isinf(parallel["sd_log_friction"])
    assert P.direction_angle({"mass_exponent": -3.23, "friction_exponent": -2.23},
                             {"mass_exponent": -2.28, "friction_exponent": -1.28}) < 6


def test_selection_rule_is_validity_only():
    def entry(fam, mag, per):
        return {"family": fam, "magnitude": mag, "per_geometry": {
            g: {"valid": v, "total": 9, "all_valid_airborne": air} for g, (v, air) in per.items()}}
    ok = {"a": (9, True), "b": (9, True)}
    weak_b = {"a": (9, True), "b": (7, True)}
    table = {"push_x_v2": entry("push_x_v2", 0.8, ok),
             "push_lo_1.6N": entry("push_lo", 1.6, ok), "push_lo_2.0N": entry("push_lo", 2.0, ok),
             "push_hi_3.2N": entry("push_hi", 3.2, ok), "push_hi_4.0N": entry("push_hi", 4.0, weak_b),
             "toss_4.0N": entry("toss", 4.0, ok), "toss_4.5N": entry("toss", 4.5, ok)}
    choice = P.select(table)
    assert choice == {"push_x_v2": "push_x_v2", "push_lo": "push_lo_2.0N",
                      "push_hi": "push_hi_3.2N", "toss": "toss_4.0N"}
    table["toss_4.0N"] = entry("toss", 4.0, {"a": (9, False), "b": (9, True)})
    assert P.select(table)["toss"] == "toss_4.5N"
    table["push_lo_1.6N"] = entry("push_lo", 1.6, weak_b)
    table["push_lo_2.0N"] = entry("push_lo", 2.0, weak_b)
    assert P.select(table)["push_lo"] is None
    assert P.select(table, ["a"])["push_lo"] == "push_lo_2.0N"


def test_config_design_has_grid_offgrid_and_central_differences():
    configs = P.build_configs()
    kinds = [c["kind"] for c in configs]
    assert kinds.count("grid") == 9 and kinds.count("offgrid") == 4 and kinds.count("fd") == 72
    assert len({c["key"] for c in configs}) == len(configs)


@pytest.mark.skipif(PHINO_ROOT is None, reason="PHINO checkout not found")
def test_common_start_and_interventions_on_box_scene(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from pathlib import Path
    import export_phino_dataset as ex
    grid = make_grid(tmp_path, objects=("box_a",))
    geom = ex.load_grid(grid).geometries[0]
    em = ex.load_phino_module(PHINO_ROOT, "train.data.mujoco_engine")
    engine = em.MuJoCoPhysicsEngine()
    originals = {a.name: a for a in em.DIAGNOSTIC_ACTIONS}
    ref = v3.make_sim(Path(geom.isolated_scene), geom.body_name, 0.3, 0.3, engine, mujoco)
    canon = v3.canonical_rest_pose(ref, mujoco)
    starts = []
    for mass, mu in ((0.25, 0.2), (0.35, 0.4)):
        sim = v3.make_sim(Path(geom.isolated_scene), geom.body_name, mass, mu, engine, mujoco)
        st = v3.start_state(sim, canon["q_star"], engine, mujoco)
        assert st["ok"], st
        starts.append(st["q0"][:3])
        toss = v3.rollout(sim, st["q0"], st["v0"], v3.ACTION_BY_NAME["toss_4.0N"], engine, mujoco,
                          originals, 1.0, 60.0)
        assert toss["valid"] and toss["airborne_frames"] >= 2
        assert toss["trace20"].shape == (50, 13) and toss["trace100"].shape == (250, 13)
        zero = v3.rollout(sim, st["q0"], st["v0"], v3.ZERO_ACTION, engine, mujoco, originals, 1.0, 60.0)
        assert np.abs(zero["trace20"][:, :3]).max() < v3.S3_MAX_DISPLACEMENT_M
    assert np.linalg.norm(starts[0] - starts[1]) < 1e-4
