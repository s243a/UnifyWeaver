#!/usr/bin/env python3
"""Tests for Product-Kalman Gaussian conditioning helpers.

Run: `python3 test_product_kalman.py`.
"""

import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman import (
    fit_error_covariance,
    fit_residual_covariance,
    gaussian_condition_update,
    gaussian_nll,
    regularize_covariance,
    scalar_product_kalman_update,
)


def assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__} should have raised ValueError")


def test_scalar_product_update_matches_closed_form():
    res = scalar_product_kalman_update(
        ell_prior=0.0,
        ell_measurement=10.0,
        prior_var=4.0,
        measurement_var=1.0,
        cross_covariance=0.0,
        jitter=0.0,
    )
    np.testing.assert_allclose(res.gain, [[0.8]], atol=1e-12)
    np.testing.assert_allclose(res.mean, [8.0], atol=1e-12)
    np.testing.assert_allclose(res.covariance, [[0.8]], atol=1e-12)
    np.testing.assert_allclose(res.innovation, [10.0], atol=1e-12)
    np.testing.assert_allclose(res.innovation_covariance, [[5.0]], atol=1e-12)


def test_scalar_cross_covariance_default_warns():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        scalar_product_kalman_update(0.0, 1.0, prior_var=1.0, measurement_var=1.0, jitter=0.0)
    assert any("cross_covariance" in str(w.message) for w in seen)


def test_gaussian_update_arrays_are_read_only():
    res = scalar_product_kalman_update(0.0, 1.0, prior_var=1.0, measurement_var=1.0, cross_covariance=0.0)
    try:
        res.mean[0] = 3.0
    except ValueError:
        pass
    else:
        raise AssertionError("GaussianUpdate arrays should be read-only")


def test_scalar_gain_moves_toward_less_noisy_channel():
    weak_measurement = scalar_product_kalman_update(
        0.0, 10.0, prior_var=1.0, measurement_var=99.0, cross_covariance=0.0, jitter=0.0
    )
    strong_measurement = scalar_product_kalman_update(
        0.0, 10.0, prior_var=99.0, measurement_var=1.0, cross_covariance=0.0, jitter=0.0
    )
    assert weak_measurement.mean[0] < 0.2
    assert strong_measurement.mean[0] > 9.8
    assert weak_measurement.gain[0, 0] < strong_measurement.gain[0, 0]


def test_scalar_limit_cases_approach_theory():
    weak_prior = scalar_product_kalman_update(
        0.0, 7.0, prior_var=1e12, measurement_var=1.0, cross_covariance=0.0, jitter=0.0
    )
    near_exact_measurement = scalar_product_kalman_update(
        0.0, 7.0, prior_var=1.0, measurement_var=1e-12, cross_covariance=0.0, jitter=0.0
    )
    assert weak_prior.gain[0, 0] > 1.0 - 1e-10
    assert abs(weak_prior.mean[0] - 7.0) < 1e-8
    assert near_exact_measurement.gain[0, 0] > 1.0 - 1e-10
    assert abs(near_exact_measurement.mean[0] - 7.0) < 1e-8


def test_vector_update_matches_independent_coordinate_formula():
    res = gaussian_condition_update(
        mean=[0.0, 0.0],
        covariance=[[4.0, 0.0], [0.0, 1.0]],
        observation=[10.0, 10.0],
        observation_covariance=[[1.0, 0.0], [0.0, 3.0]],
        jitter=0.0,
    )
    np.testing.assert_allclose(res.gain, [[0.8, 0.0], [0.0, 0.25]], atol=1e-12)
    np.testing.assert_allclose(res.mean, [8.0, 2.5], atol=1e-12)
    np.testing.assert_allclose(np.diag(res.covariance), [0.8, 0.75], atol=1e-12)


def test_dense_vector_update_with_nonidentity_observation_matrix():
    mean = np.array([1.0, -1.0])
    P = np.array([[2.0, 0.4], [0.4, 1.0]])
    H = np.array([[1.0, 0.5]])
    y = np.array([2.0])
    R = np.array([[0.7]])
    res = gaussian_condition_update(mean, P, y, R, H=H, jitter=0.0)

    S = H @ P @ H.T + R
    cov_xy = P @ H.T
    expected_gain = cov_xy @ np.linalg.inv(S)
    expected_mean = mean + (expected_gain @ (y - H @ mean)).reshape(-1)
    expected_cov = P - expected_gain @ S @ expected_gain.T
    np.testing.assert_allclose(res.gain, expected_gain, atol=1e-12)
    np.testing.assert_allclose(res.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(res.covariance, expected_cov, atol=1e-12)
    assert abs(res.gain[1, 0]) > 0.01


def test_cross_covariance_matches_gaussian_conditioning_formula():
    res = gaussian_condition_update(
        mean=[1.0],
        covariance=[[2.0]],
        observation=[4.0],
        observation_covariance=[[3.0]],
        H=[[1.0]],
        cross_covariance=[[0.5]],
        jitter=0.0,
    )
    innovation_cov = 2.0 + 3.0 + 2.0 * 0.5
    state_innovation_cov = 2.0 + 0.5
    gain = state_innovation_cov / innovation_cov
    np.testing.assert_allclose(res.gain, [[gain]], atol=1e-12)
    assert abs(res.mean[0] - (1.0 + gain * 3.0)) < 1e-12
    assert abs(res.covariance[0, 0] - (2.0 - state_innovation_cov**2 / innovation_cov)) < 1e-12
    np.testing.assert_allclose(res.cross_covariance, [[0.5]], atol=1e-12)


def test_negative_cross_covariance_reduces_innovation_covariance():
    no_cross = gaussian_condition_update([0.0], [[2.0]], [1.0], [[3.0]], cross_covariance=[[0.0]], jitter=0.0)
    neg_cross = gaussian_condition_update([0.0], [[2.0]], [1.0], [[3.0]], cross_covariance=[[-0.4]], jitter=0.0)
    assert neg_cross.innovation_covariance[0, 0] < no_cross.innovation_covariance[0, 0]
    np.testing.assert_allclose(neg_cross.innovation_covariance, [[4.2]], atol=1e-12)


def test_posterior_covariance_stays_symmetric_with_dense_cross_covariance():
    P = np.array([[2.0, 0.3, 0.2], [0.3, 1.5, 0.1], [0.2, 0.1, 1.2]])
    H = np.array([[1.0, 0.2, 0.0], [0.0, 0.3, 1.0]])
    R = np.array([[0.8, 0.1], [0.1, 0.7]])
    C = np.array([[0.05, -0.02], [0.03, 0.01], [-0.01, 0.04]])
    res = gaussian_condition_update([0.1, -0.2, 0.3], P, [0.4, -0.1], R, H=H, cross_covariance=C)
    np.testing.assert_allclose(res.covariance, res.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(res.covariance).min() > 0.0


def test_invalid_cross_covariance_rejected_by_innovation_covariance():
    assert_raises(
        gaussian_condition_update,
        mean=[0.0],
        covariance=[[1.0]],
        observation=[1.0],
        observation_covariance=[[1.0]],
        H=[[1.0]],
        cross_covariance=[[-2.0]],
        jitter=0.0,
    )


def test_residual_covariance_and_shrinkage_targets():
    residuals = np.array([[1.0, 0.0], [-1.0, 2.0], [0.0, -2.0]])
    expected = np.cov(residuals, rowvar=False, bias=False)
    full = fit_residual_covariance(residuals, jitter=0.0)
    diagonal = fit_residual_covariance(residuals, shrinkage=1.0, jitter=0.0)
    scaled_identity = fit_residual_covariance(residuals, shrinkage=1.0, shrinkage_target="scaled_identity", jitter=0.0)
    mle = fit_residual_covariance(residuals, ddof=0, jitter=0.0)
    np.testing.assert_allclose(full, expected, atol=1e-12)
    np.testing.assert_allclose(diagonal, np.diag(np.diag(expected)), atol=1e-12)
    np.testing.assert_allclose(scaled_identity, np.eye(2) * np.trace(expected) / 2.0, atol=1e-12)
    np.testing.assert_allclose(mle, np.cov(residuals, rowvar=False, bias=True), atol=1e-12)
    assert_raises(fit_residual_covariance, residuals, shrinkage=1.0, shrinkage_target="unknown")


def test_error_covariance_uses_observed_minus_predicted_and_checks_inputs():
    predicted = np.array([[1.0], [2.0], [3.0]])
    observed = np.array([[1.5], [1.0], [3.5]])
    expected = np.cov((observed - predicted).reshape(-1), bias=False).reshape(1, 1)
    np.testing.assert_allclose(fit_error_covariance(predicted, observed, jitter=0.0), expected, atol=1e-12)
    assert_raises(fit_error_covariance, predicted, observed.reshape(-1), jitter=0.0)
    assert_raises(fit_error_covariance, [[float("nan")]], [[1.0]])
    assert_raises(fit_error_covariance, [[1.0]], [[float("inf")]])


def test_regularize_covariance_lifts_tiny_semidefinite_floor_only():
    lifted = regularize_covariance([[1.0, 0.0], [0.0, 0.0]], jitter=1e-6)
    assert np.linalg.eigvalsh(lifted).min() >= 1e-6 - 1e-12
    assert_raises(regularize_covariance, [[1.0, 0.0], [0.0, -0.1]], jitter=1e-9)
    assert_raises(regularize_covariance, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], jitter=0.0)


def test_gaussian_nll_matches_unit_normal_constant_and_accepts_column_vectors():
    assert abs(gaussian_nll([[0.0]], [[0.0]], [[1.0]], jitter=0.0) - 0.5 * math.log(2.0 * math.pi)) < 1e-12
    assert abs(gaussian_nll([0.0], [0.0], [[1.0]], jitter=0.0, include_constant=False)) < 1e-12
    assert_raises(gaussian_nll, [0.0, 1.0], [0.0], [[1.0]], jitter=0.0)


def test_shape_errors_and_scalar_variance_guards_fail_fast():
    assert_raises(gaussian_condition_update, [0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], [0.0], [[1.0]])
    assert_raises(fit_residual_covariance, [[1.0, 2.0]], jitter=0.0)
    assert_raises(fit_residual_covariance, [[1.0], [float("nan")]], jitter=0.0)
    assert_raises(fit_residual_covariance, [[1.0], [2.0]], shrinkage=1.5, jitter=0.0)
    assert_raises(fit_residual_covariance, [[1.0], [2.0]], ddof=-1, jitter=0.0)
    assert_raises(scalar_product_kalman_update, 0.0, 1.0, prior_var=0.0, measurement_var=1.0)
    assert_raises(scalar_product_kalman_update, 0.0, 1.0, prior_var=float("nan"), measurement_var=1.0)
    assert_raises(scalar_product_kalman_update, 0.0, 1.0, prior_var=1.0, measurement_var=-1.0)
    assert_raises(scalar_product_kalman_update, 0.0, 1.0, prior_var=1.0, measurement_var=float("inf"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman tests passed")
