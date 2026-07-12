#!/usr/bin/env python3
"""Property tests for the joint square-root / Householder-QR conditioner."""

import numpy as np
import joint_square_root_conditioner as square_root_module

from joint_square_root_conditioner import (
    condition_correlated_gaussian_qr,
    conditional_measurement_model,
    householder_information_update,
    precision_root_from_covariance,
    whiten_measurement_block,
)
from product_kalman import gaussian_condition_update


def _joint_spd(rng, n, m, floor=0.5):
    a = rng.standard_normal((n + m, n + m))
    sigma = a @ a.T + floor * np.eye(n + m)
    return sigma[:n, :n], sigma[:n, n:], sigma[n:, n:]


def test_scalar_correlated_matches_dense_conditioner():
    mean = np.array([0.2])
    P = np.array([[2.0]])
    R = np.array([[1.0]])
    C = np.array([[0.5]])
    H = np.array([[1.0]])
    y = np.array([1.2])
    root = precision_root_from_covariance(P, jitter=0.0)
    qr = condition_correlated_gaussian_qr(mean, root, y, R, H, C, jitter=0.0)
    dense = gaussian_condition_update(mean, P, y, R, H=H, cross_covariance=C, jitter=0.0)
    assert np.allclose(qr.mean, dense.mean, atol=1e-10)
    assert np.allclose(qr.covariance, dense.covariance, atol=1e-10)
    assert np.allclose(qr.precision_root.T @ qr.precision_root,
                       np.linalg.inv(qr.covariance), atol=1e-9)


def test_random_dense_nonzero_C_matches_gaussian_conditioning():
    rng = np.random.default_rng(7)
    for n, m in ((2, 4), (3, 2), (4, 5)):
        for _ in range(12):
            P, C, R = _joint_spd(rng, n, m)
            H = rng.standard_normal((m, n))
            mean = rng.standard_normal(n)
            y = rng.standard_normal(m)
            root = precision_root_from_covariance(P, jitter=0.0)
            qr = condition_correlated_gaussian_qr(mean, root, y, R, H, C, jitter=0.0)
            dense = gaussian_condition_update(mean, P, y, R, H=H, cross_covariance=C, jitter=0.0)
            assert np.allclose(qr.mean, dense.mean, rtol=2e-8, atol=2e-9)
            assert np.allclose(qr.covariance, dense.covariance, rtol=2e-8, atol=2e-9)


def test_conditional_model_recovers_joint_innovation_covariance():
    rng = np.random.default_rng(11)
    P, C, R = _joint_spd(rng, 3, 4)
    H = rng.standard_normal((4, 3))
    root = precision_root_from_covariance(P, jitter=0.0)
    J, Rc = conditional_measurement_model(root, H, R, C, jitter=0.0)
    recovered = J @ P @ J.T + Rc
    direct = H @ P @ H.T + R + H @ C + (H @ C).T
    assert np.allclose(recovered, direct, atol=1e-9)


def test_independent_blocks_batch_equals_streaming_update():
    rng = np.random.default_rng(19)
    n = 3
    P = np.diag([1.5, 0.8, 2.0])
    root = precision_root_from_covariance(P, jitter=0.0)
    J1 = rng.standard_normal((2, n)); R1 = np.array([[0.8, 0.1], [0.1, 1.1]])
    J2 = rng.standard_normal((3, n)); R2 = np.array([[1.2, 0.1, 0.0], [0.1, 0.9, 0.2], [0.0, 0.2, 1.4]])
    r1 = rng.standard_normal(2); r2 = rng.standard_normal(3)
    A1, b1, _ = whiten_measurement_block(J1, R1, r1, jitter=0.0)
    A2, b2, _ = whiten_measurement_block(J2, R2, r2, jitter=0.0)

    batch = householder_information_update(
        root, np.zeros(n), np.vstack([A1, A2]), np.concatenate([b1, b2]))
    first = householder_information_update(root, np.zeros(n), A1, b1)
    streamed = householder_information_update(
        first.precision_root, first.information_rhs, A2, b2)
    assert np.allclose(batch.precision_root, streamed.precision_root, atol=1e-10)
    assert np.allclose(batch.information_rhs, streamed.information_rhs, atol=1e-10)
    assert np.allclose(batch.solution, streamed.solution, atol=1e-10)


def test_information_qr_rank_check_is_invariant_to_uniform_small_scale():
    rng = np.random.default_rng(21)
    n = 3
    A = rng.standard_normal((5, n))
    z = rng.standard_normal(n)
    b = rng.standard_normal(5)
    reference = householder_information_update(np.eye(n), z, A, b)

    scale = 1e-20
    scaled = householder_information_update(
        scale * np.eye(n), scale * z, scale * A, scale * b
    )

    np.testing.assert_allclose(scaled.solution, reference.solution, atol=2e-12)
    np.testing.assert_allclose(
        scaled.precision_root.T @ scaled.precision_root,
        scale**2 * (reference.precision_root.T @ reference.precision_root),
        rtol=2e-12,
        atol=0.0,
    )


def test_householder_update_is_row_permutation_invariant():
    rng = np.random.default_rng(23)
    n, m = 4, 9
    root = precision_root_from_covariance(np.diag([0.5, 1.0, 2.0, 4.0]), jitter=0.0)
    A = rng.standard_normal((m, n)); b = rng.standard_normal(m)
    ref = householder_information_update(root, np.zeros(n), A, b)
    order = rng.permutation(m)
    got = householder_information_update(root, np.zeros(n), A[order], b[order])
    assert np.allclose(ref.precision_root, got.precision_root, atol=1e-10)
    assert np.allclose(ref.information_rhs, got.information_rhs, atol=1e-10)


def test_materially_invalid_conditional_covariance_raises():
    root = precision_root_from_covariance(np.eye(1), jitter=0.0)
    # R - C.T P^-1 C = 0.1 - 4 < 0: the proposed joint covariance is invalid.
    try:
        conditional_measurement_model(root, np.ones((1, 1)), np.array([[0.1]]), np.array([[2.0]]), jitter=0.0)
    except ValueError as exc:
        assert "conditional observation covariance" in str(exc)
    else:
        raise AssertionError("invalid joint covariance should fail")


def test_composed_conditioner_regularizes_conditional_covariance_once(monkeypatch):
    mean = np.array([0.2])
    P = np.array([[2.0]])
    R = np.array([[1.0]])
    C = np.array([[0.5]])
    H = np.array([[1.0]])
    y = np.array([1.2])
    root = precision_root_from_covariance(P, jitter=0.0)
    original = square_root_module.regularize_covariance
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs.get("name"))
        return original(*args, **kwargs)

    monkeypatch.setattr(square_root_module, "regularize_covariance", tracked)
    condition_correlated_gaussian_qr(mean, root, y, R, H, C, jitter=0.0)
    assert calls == ["conditional observation covariance R - C.T P^-1 C"]
