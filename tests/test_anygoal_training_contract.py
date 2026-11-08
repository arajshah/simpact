"""Checks for the SMWM training-step reconstruction.

These tests do not load the checkpoint or the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

import anygoal_contact_diagnostic as audit
import anygoal_training_contract as contract


def test_frameskip_window_strides_pixels_and_keeps_every_action():
    pixels = np.arange(12, dtype=np.uint8).reshape(12, 1, 1, 1)
    actions = np.arange(12 * 5, dtype=np.float64).reshape(12, 5)
    pix, blocked = contract.frameskip_window(pixels, actions, 1)
    assert pix[:, 0, 0, 0].tolist() == [1, 6]
    assert blocked.shape == (2, 25)
    assert blocked[0, :5].tolist() == actions[1].tolist()
    assert blocked[0, 5:10].tolist() == actions[2].tolist()
    assert blocked[1, :5].tolist() == actions[6].tolist()


def test_normalization_matches_the_trainer_standard_deviation():
    actions = np.arange(20, dtype=np.float64).reshape(4, 5)
    mean, std = contract.torch_action_stats(actions)
    assert std[0] == pytest.approx(np.std(actions[:, 0], ddof=1))
    normalized = contract.normalize_action_block(actions[:2], mean, std)
    assert normalized.shape == (2, 5)
    assert normalized[0, 0] == pytest.approx((actions[0, 0] - mean[0]) / std[0])
    population = actions.std(axis=0, ddof=0)
    assert contract.relative_scale_gap(std, population) > 0.0


def test_persistence_and_alignment_on_a_known_change():
    current = np.zeros((2, 4))
    target = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    predicted = target.copy()
    errors = contract.clip_errors(predicted, target, current)
    assert errors["sse"].tolist() == [0.0, 0.0]
    assert errors["persistence_sse"].tolist() == [1.0, 4.0]
    assert errors["cosine"].tolist() == pytest.approx([1.0, 1.0])
    wrong = np.array([[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    orthogonal = contract.clip_errors(wrong, target, current)
    assert orthogonal["cosine"].tolist() == pytest.approx([0.0, 0.0])


def test_normalized_actions_are_not_cast_to_the_image_dtype():
    import torch

    image = torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
    actions = np.array([[0.2, -1.5, 0.0, 3.0, -0.4]], dtype=np.float64)
    tensor = audit.float_actions(actions, image)
    assert tensor.dtype == torch.float32
    assert tensor[0, 1].item() == pytest.approx(-1.5)
    wrapped = torch.as_tensor(actions, dtype=image.dtype)
    assert wrapped.dtype == torch.uint8
    assert int(wrapped[0, 1]) != -1


def test_classifier_rejects_a_pipeline_mismatch_before_the_loss():
    cosine = {"defined": True, "mean": 0.9, "ci_low": 0.8, "ci_high": 1.0}
    assert contract.classify_findings(
        paths_agree=False,
        scaler_matches=True,
        start_beats=True,
        ordinary_beats=True,
        same_action_cosine=cosine,
        cross_action_cosine=None,
    ) == "A_evaluation_mismatch"
    assert contract.classify_findings(
        paths_agree=True,
        scaler_matches=True,
        start_beats=False,
        ordinary_beats=False,
        same_action_cosine={"defined": True, "mean": 0.0, "ci_low": -0.1, "ci_high": 0.1},
        cross_action_cosine={"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1},
    ) == "B_poor_fit"
    assert contract.classify_findings(
        paths_agree=True,
        scaler_matches=True,
        start_beats=False,
        ordinary_beats=True,
        same_action_cosine=cosine,
        cross_action_cosine=None,
    ) == "C_phase_dependence"
    assert contract.classify_findings(
        paths_agree=True,
        scaler_matches=True,
        start_beats=True,
        ordinary_beats=True,
        same_action_cosine=cosine,
        cross_action_cosine={"mean": 0.01, "ci_low": -0.05, "ci_high": 0.05},
    ) == "E_decision_misalignment"
