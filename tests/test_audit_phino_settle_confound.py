"""Tests for the quaternion transforms in scripts/audit_phino_settle_confound.py."""
from __future__ import annotations

import numpy as np

import audit_phino_settle_confound as audit


def _yaw(deg: float) -> np.ndarray:
    half = np.radians(deg) / 2
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _segment(q: np.ndarray, omega_body: np.ndarray, frames: int = 50) -> np.ndarray:
    seg = np.zeros((frames, 13), dtype=np.float32)
    seg[:, 3:7] = q
    seg[:, 10:13] = omega_body
    return seg


def test_quaternion_helpers():
    q = _yaw(90)
    assert np.allclose(audit.qmul(q, audit.qconj(q)), [1, 0, 0, 0])
    assert np.allclose(audit.quat_to_mat(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_start_relative_removes_settle_orientation_and_uses_world_omega():
    q0 = _yaw(90)
    flat = np.concatenate([_segment(q0, np.array([1.0, 0, 0]))] * 4)
    rel = audit.start_relative_fingerprint(flat, q0)
    assert np.allclose(rel[:, 3:7], [1, 0, 0, 0], atol=1e-6)
    assert np.allclose(rel[:, 10:13], [0, 1, 0], atol=1e-6)


def test_settle_swap_is_identity_and_invertible():
    rng = np.random.default_rng(0)
    q0, q1 = _yaw(10), _yaw(55)
    seg = _segment(q0, np.zeros(3))
    seg[:, 3:7] = audit.qmul(np.array([_yaw(a) for a in np.linspace(0, 40, 50)]), q0)
    seg[:, 10:13] = rng.normal(size=(50, 3))
    flat = np.concatenate([seg] * 4)
    assert np.allclose(audit.swap_settle_orientation(flat, q0, q0), flat, atol=1e-6)
    swapped = audit.swap_settle_orientation(flat, q0, q1)
    assert np.allclose(swapped[:50, 3:7][0], audit.qmul(audit.qmul(flat[0, 3:7], audit.qconj(q0)), q1),
                       atol=1e-6)
    assert np.allclose(audit.start_relative_fingerprint(swapped, q1),
                       audit.start_relative_fingerprint(flat, q0), atol=1e-5)
    assert np.allclose(audit.swap_settle_orientation(swapped, q1, q0), flat, atol=1e-5)


def test_rank_corr():
    a = np.arange(10.0)
    assert np.isclose(audit.rank_corr(a, a ** 3), 1.0)
    assert np.isclose(audit.rank_corr(a, -a), -1.0)
