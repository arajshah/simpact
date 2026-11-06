"""Tests for scripts/phino_visual_observability_pilot.py.

Analysis helpers need only numpy. The end-to-end render test needs a PHINO
checkout plus MuJoCo with offscreen GL and reuses the synthetic grid fixture
from test_export_phino_dataset.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytest

import phino_visual_observability_pilot as vis
from test_export_phino_dataset import PHINO_ROOT, make_grid


def _trace(final_xy=(0.0, 0.0), speeds=None) -> np.ndarray:
    trace = np.zeros((50, 13), dtype=np.float32)
    trace[:, 3] = 1.0
    trace[:, :2] = np.linspace(0, 1, 50)[:, None] * np.asarray(final_xy)
    if speeds is not None:
        trace[:, 7] = speeds
    return trace


def test_relative_yaw_and_stop_frame():
    q0 = np.array([1.0, 0, 0, 0])
    half = np.radians(30) / 2
    assert np.isclose(vis.relative_yaw_deg(q0, [np.cos(half), 0, 0, np.sin(half)]), 30.0)
    speeds = np.r_[np.full(10, 0.5), np.zeros(40)]
    assert vis.stop_frame(_trace(speeds=speeds)) == 10
    assert vis.stop_frame(_trace(speeds=np.full(50, 0.5))) is None


def test_response_exponents_recover_power_law_and_direction():
    configs = [vis.Config(m, mu, {"push_x": _trace((m ** -2 * mu ** -1 / 100, 0))})
               for m in (0.25, 0.30, 0.35) for mu in (0.2, 0.3, 0.4)]
    fit = vis.response_exponents(configs, "push_x")
    assert np.isclose(fit["mass_exponent"], -2, atol=1e-3)
    assert np.isclose(fit["friction_exponent"], -1, atol=1e-3)
    assert vis.direction_angle_deg(fit, dict(fit)) == pytest.approx(0.0, abs=1e-6)
    other = {"mass_exponent": 1.0, "friction_exponent": -2.0}
    assert vis.direction_angle_deg(fit, other) == pytest.approx(90.0, abs=1e-6)


def test_pairwise_summary_identifies_closest_pair_and_step_effects():
    configs = [vis.Config(m, mu, {}) for m in (0.3, 0.35) for mu in (0.2, 0.4)]
    feats = {"m0.300_mu0.200": np.array([[0.0, 0]]), "m0.300_mu0.400": np.array([[2.0, 0]]),
             "m0.350_mu0.200": np.array([[1.0, 0]]), "m0.350_mu0.400": np.array([[1.1, 0]])}
    rows = vis.pairwise(configs, feats, vis.final_distance)
    summary = vis.summarize_pairs(rows, (0.5,), configs)
    assert summary["pairs"] == 6
    assert summary["closest_pair"] == ["m0.350_mu0.200", "m0.350_mu0.400"]
    assert summary["closest_kind"] == "friction_only"
    assert summary["below"]["0.5"] == 1
    assert summary["mean_one_mass_step"] == pytest.approx((1.0 + 0.9) / 2)


@pytest.mark.skipif(PHINO_ROOT is None, reason="PHINO checkout not found")
def test_pilot_renders_matched_start_aligned_views(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
    except Exception as exc:
        pytest.skip(f"MuJoCo offscreen rendering unavailable: {exc}")
    grid = make_grid(tmp_path, objects=("box_a", "box_b"))
    for sub, key in (("action_trajectories", "trajectory"), ("fingerprints", "flat_trajectory")):
        for path in (grid / sub).glob("*.npz"):
            data = dict(np.load(path))
            data[key][:, 2] = 0.0  # planar motion: the camera is framed on table-plane motion
            np.savez_compressed(path, **data)
    out = tmp_path / "vis"
    args = argparse.Namespace(grid_dir=grid, phino_root=PHINO_ROOT, output_dir=out,
                              geometry=["box_a"], skip_replay_check=True)
    summary = vis.run(args)
    geom = summary["geometries"]["scene_0__box_a"]
    assert geom["initial_image_max_pixel_diff"]["aligned"] <= 1.0
    assert len(geom["configs"]) == 4
    assert "image_aligned_final_centroid_px" in geom["separability"]
    assert (out / "figures" / "scene_0__box_a_push_x.png").is_file()
    assert (out / "figures" / "settling_confound.png").is_file()
    assert json.loads((out / "summary.json").read_text())["render_action"] == "push_x"
    with pytest.raises(SystemExit, match="already exists"):
        vis.run(args)
