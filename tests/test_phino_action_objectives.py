"""CPU checks for the baseline objectives and the geometry-level comparison."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import phino_action_baseline_report as report
import phino_action_objectives as obj


def _row(geometry, mass, friction, in_frame=True):
    return {"geometry_short": geometry, "mass": mass, "friction": friction,
            "object_in_frame": in_frame, "action": "push_x_v2"}


def test_training_rows_drop_the_test_geometry_and_hidden_objects():
    rows = [_row("cow", 0.3, 0.2), _row("cube", 0.35, 0.4), _row("cow", 0.25, 0.3, in_frame=False)]
    train = obj.training_rows(rows, "cow")
    assert [r["geometry_short"] for r in train] == ["cube"]
    with pytest.raises(ValueError, match="unknown geometry"):
        obj.training_rows(rows, "sphere")


def test_unique_configurations_count_each_object_once():
    first = _row("cow", 0.30, 0.20)
    second = _row("cow", 0.30, 0.20)
    first["config"] = second["config"] = "same"
    second["action"] = "toss_4.0N"
    assert len(obj.unique_configurations([first, second])) == 1
    other = _row("cow", 0.35, 0.20)
    other["config"] = "same"
    with pytest.raises(ValueError, match="conflicting labels"):
        obj.unique_configurations([first, other])


def test_label_standardizer_ignores_the_held_out_geometry():
    train = [_row("cow", 0.30, 0.20), _row("cow", 0.35, 0.40), _row("cube", 0.25, 0.30)]
    held = [_row("nut", 5.0, 0.01)]
    stats = obj.fit_log_label_standardizer(obj.training_rows(train + held, "nut"))
    ref = obj.fit_log_label_standardizer(train)
    assert np.allclose(stats["mean"], ref["mean"]) and np.allclose(stats["std"], ref["std"])
    standardized = obj.standardize_log_labels(train, stats)
    assert standardized.shape == (3, 2)
    assert np.allclose(standardized.mean(0), 0.0, atol=1e-8)
    assert np.allclose(standardized.std(0), 1.0, atol=1e-8)
    with pytest.raises(ValueError, match="positive"):
        obj.fit_log_label_standardizer([_row("cow", 0.0, 0.2), _row("cube", 0.3, 0.3)])


def test_instance_infonce_matches_symmetric_cross_entropy_and_rejects_a_similarity():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(5, 4))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    rolled = np.roll(raw, 1, axis=0)
    temperature = 0.07

    def cross_entropy(logits):
        shift = logits.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(logits - shift).sum(axis=1)) + shift.ravel()
        return float(np.mean(-np.diag(logits) + log_normalizer))

    def numpy_loss(left, right):
        logits = left @ right.T / temperature
        return 0.5 * (cross_entropy(logits) + cross_entropy(logits.T))

    loss_fn = obj.InstanceInfoNCE(temperature=temperature)
    z = torch.tensor(raw, dtype=torch.float32)
    matched = loss_fn(z, z)
    mismatched = loss_fn(z, torch.tensor(rolled, dtype=torch.float32))
    assert matched["loss"].ndim == 0
    matched_value = float(matched["loss"].detach())
    mismatched_value = float(mismatched["loss"].detach())
    assert matched_value == pytest.approx(numpy_loss(raw, raw), abs=1e-5)
    assert mismatched_value == pytest.approx(numpy_loss(raw, rolled), abs=1e-5)
    assert matched_value < mismatched_value
    loss_fn.log_temperature.data.fill_(10.0)
    assert float(loss_fn.temperature.detach()) == pytest.approx(1.0)
    with pytest.raises(TypeError):
        loss_fn(z, z, torch.eye(5))


def test_regression_head_is_not_normalized_and_averages_the_two_views():
    torch.manual_seed(0)
    head = obj.ClipRegressionHead(4, 4)
    with torch.no_grad():
        head.head[-1].weight.zero_()
        head.head[-1].bias.copy_(torch.tensor([3.0, -4.0]))
    prediction = head(torch.randn(2, 4))
    assert torch.allclose(prediction, torch.tensor([[3.0, -4.0], [3.0, -4.0]]))
    z1 = torch.zeros(3, 2)
    z2 = torch.ones(3, 2)
    target = torch.zeros(3, 2)
    loss = obj.two_view_regression_loss(z1, z2, target)
    assert float(loss) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        obj.two_view_regression_loss(z1, z2, torch.zeros(3, 1))


def test_shared_view_shifts_every_frame_by_the_same_crop():
    clip = torch.zeros(2, 3, 224, 224)
    clip[0, :, 20, 30] = 1.0
    clip[1, :, 20, 30] = 1.0
    mean = torch.zeros(1, 3, 1, 1)
    std = torch.ones(1, 3, 1, 1)
    viewed = obj.shared_view(clip, torch.tensor(0.5), torch.tensor(0.5), 4, 6, mean, std, shift=6, jitter=0.10)
    assert viewed.shape == clip.shape
    peaks = []
    for frame in (0, 1):
        plane = viewed[frame, 0].detach().numpy()
        peaks.append(np.unravel_index(int(plane.argmax()), plane.shape))
    # Pad by 6, then crop at (4, 6): original (20, 30) lands at (22, 30).
    assert peaks[0] == peaks[1] == (22, 30)


def test_subset_mean_collapses_dependent_subsets_before_counting_geometries():
    rows = [
        {"per_fold": {"cow": {"nmae": 0.0}, "cube": {"nmae": 1.0}}},
        {"per_fold": {"cow": {"nmae": 0.2}, "cube": {"nmae": 1.2}}},
    ]
    got = report.per_geometry_subset_mean(rows)
    assert got == {"cow": pytest.approx(0.1), "cube": pytest.approx(1.1)}
    assert len(got) == 2
    with pytest.raises(ValueError, match="different geometries"):
        report.per_geometry_subset_mean([rows[0], {"per_fold": {"cow": {"nmae": 0.0}}}])


def test_paired_summary_counts_geometries_and_not_subsets():
    summary = report.paired_summary({"cow": -0.2, "cube": 0.1}, {"cow": -0.1, "cube": -0.05})
    assert summary["n_geometries"] == 2
    assert summary["n_both_seeds_negative"] == 1
    assert summary["n_sign_disagreement"] == 1
    assert "n_subsets" not in summary
    assert "pvalue" not in summary
    assert summary["mean_seed_averaged_delta"] == pytest.approx(0.5 * ((-0.15) + 0.025))


def _probe_row(representation, interventions, nmae, folds):
    return {
        "representation": representation, "interventions": interventions,
        "n_interventions": interventions.count("+") + 1, "split": "logo", "target": "mass",
        "n": 18, "mae": nmae, "baseline_mae": 1.0, "nmae": nmae, "per_fold": folds,
    }


def test_merge_rows_keeps_an_identical_tracker_and_rejects_a_conflict():
    folds = {"mesh_cow": {"n": 10, "nmae": 0.2, "mae": 0.2}, "mesh_cube": {"n": 8, "nmae": 0.3, "mae": 0.3}}
    left = [_probe_row("image_position", "push_x_v2", 0.2, folds)]
    right = [_probe_row("image_position", "push_x_v2", 0.2, folds),
             _probe_row("ft_instance_s0", "push_x_v2", 0.4, folds)]
    merged = report.merge_rows(left, right)
    assert merged[("ft_instance_s0", "push_x_v2", "logo", "mass")]["nmae"] == 0.4
    with pytest.raises(ValueError, match="conflicting"):
        report.merge_rows(left, [_probe_row("image_position", "push_x_v2", 0.9, folds)])


def test_assemble_primary_delta_uses_one_intervention_set():
    subsets = report.all_action_subsets()
    grid_subsets = list(report.PUBLISHED_GRID_SUBSETS)
    geoms = ("mesh_cow", "mesh_cube")
    counts = {"mesh_cow": 10, "mesh_cube": 8}
    level = {
        "ft_clip_s0": 0.60, "ft_clip_s1": 0.50,
        "ft_instance_s0": 0.70, "ft_instance_s1": 0.65,
        "ft_regression_s0": 0.40, "ft_regression_s1": 0.45,
        "dinov2_frozen": 0.80, "image_position": 0.10,
        "grid_ft_clip_s0": 0.60, "grid_ft_clip_s1": 0.50,
        "grid_ft_instance_s0": 0.70, "grid_ft_instance_s1": 0.65,
        "grid_ft_regression_s0": 0.40, "grid_ft_regression_s1": 0.45,
        "grid_dinov2_frozen": 0.80,
    }

    def pack(rep_names, names):
        rows = []
        for rep in rep_names:
            for name in names:
                for split in ("logo", "within"):
                    for target in ("mass", "friction"):
                        folds = {g: {"n": counts[g], "nmae": level[rep], "mae": level[rep]} for g in geoms}
                        rows.append({
                            "representation": rep, "interventions": name,
                            "n_interventions": name.count("+") + 1, "split": split, "target": target,
                            "n": 18, "mae": level[rep], "baseline_mae": 1.0, "nmae": level[rep],
                            "per_fold": folds,
                        })
        return rows

    learned = ["ft_clip_s0", "ft_clip_s1", "ft_instance_s0", "ft_instance_s1",
               "ft_regression_s0", "ft_regression_s1"]
    grid_learned = ["grid_" + name for name in learned]
    previous = {"n_configs": 18, "configs_per_geometry": counts, "results": pack(
        ["ft_clip_s0", "ft_clip_s1", "dinov2_frozen", "image_position"], subsets),
        "position_decoding_logo_r2": {}}
    new = {"n_configs": 18, "configs_per_geometry": counts, "results": pack(
        ["ft_instance_s0", "ft_instance_s1", "ft_regression_s0", "ft_regression_s1", "image_position"], subsets),
        "position_decoding_logo_r2": {}}
    grid_frozen = {"n_configs": 18, "configs_per_geometry": counts,
                   "results": pack(["grid_dinov2_frozen", "image_position"], grid_subsets)}
    grid_prev = {"n_configs": 18, "configs_per_geometry": counts, "results": pack(
        ["grid_ft_clip_s0", "grid_ft_clip_s1", "image_position"], grid_subsets),
        "position_decoding_logo_r2": {}}
    grid_new = {"n_configs": 18, "configs_per_geometry": counts, "results": pack(
        ["grid_ft_instance_s0", "grid_ft_instance_s1", "grid_ft_regression_s0", "grid_ft_regression_s1",
         "image_position"], grid_subsets), "position_decoding_logo_r2": {}}
    assembled = report.assemble(previous, new, grid_frozen, grid_prev, grid_new)
    paired = assembled["readouts"]["pooled"]["logo_full"]["mass"]["paired"]["trajectory_minus_instance"]
    assert paired["n_geometries"] == 2
    assert "n_subsets" not in paired
    # Seed-averaged geometry delta is identical on both geometries in this fixture:
    # seed0 0.60-0.70 = -0.10, seed1 0.50-0.65 = -0.15, mean -0.125.
    assert paired["mean_seed_averaged_delta"] == pytest.approx(-0.125)
    assert paired["n_both_seeds_negative"] == 2
    assert assembled["readouts"]["pooled"]["logo_by_count"]["2"]["n_dependent_subsets"] == 6
    assert assembled["readouts"]["grid4x4"]["logo_by_count"]["2"]["n_dependent_subsets"] == 2
    assert assembled["readouts"]["grid4x4"]["logo_by_count"]["3"]["n_dependent_subsets"] == 1
    text = report.render_markdown(assembled)
    assert "not a sample size" in text
    assert "trajectory_minus_instance" in text
    assert "Generic InfoNCE" in text
