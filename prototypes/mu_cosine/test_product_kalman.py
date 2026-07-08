#!/usr/bin/env python3
"""Tests for Product-Kalman Gaussian conditioning helpers.

Run: `python3 test_product_kalman.py`.
"""

import math
import sys
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
    res = scalar_product_kalman_update(ell_prior=0.0, ell_measurement=10.0, prior_var=4.0, measurement_var=1.0,
                                       jitter=0.0)
    assert np.allclose(res.gain, [[0.8]])
    assert abs(res.mean[0] - 8.0) < 1e-12
    assert abs(res.covariance[0, 0] - 0.8) < 1e-12
    assert abs(res.innovation[0] - 10.0) < 1e-12
    assert abs(res.innovation_covariance[0, 0] - 5.0) < 1e-12


def test_scalar_gain_moves_toward_less_noisy_channel():
    weak_measurement = scalar_product_kalman_update(0.0, 10.0, prior_var=1.0, measurement_var=99.0, jitter=0.0)
    strong_measurement = scalar_product_kalman_update(0.0, 10.0, prior_var=99.0, measurement_var=1.0, jitter=0.0)
    assert weak_measurement.mean[0] < 0.2
    assert strong_measurement.mean[0] > 9.8
    assert weak_measurement.gain[0, 0] < strong_measurement.gain[0, 0]


def test_vector_update_matches_independent_coordinate_formula():
    res = gaussian_condition_update(
        mean=[0.0, 0.0],
        covariance=[[4.0, 0.0], [0.0, 1.0]],
        observation=[10.0, 10.0],
        observation_covariance=[[1.0, 0.0], [0.0, 3.0]],
        jitter=0.0,
    )
    assert np.allclose(res.gain, [[0.8, 0.0], [0.0, 0.25]])
    assert np.allclose(res.mean, [8.0, 2.5])
    assert np.allclose(np.diag(res.covariance), [0.8, 0.75])


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
    assert np.allclose(res.gain, [[gain]])
    assert abs(res.mean[0] - (1.0 + gain * 3.0)) < 1e-12
    assert abs(res.covariance[0, 0] - (2.0 - state_innovation_cov**2 / innovation_cov)) < 1e-12
    assert np.allclose(res.cross_covariance, [[0.5]])


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


def test_residual_covariance_and_diagonal_shrinkage():
    residuals = np.array([[1.0, 0.0], [-1.0, 2.0], [0.0, -2.0]])
    expected = np.cov(residuals, rowvar=False, bias=False)
    full = fit_residual_covariance(residuals, jitter=0.0)
    diagonal = fit_residual_covariance(residuals, shrinkage=1.0, jitter=0.0)
    assert np.allclose(full, expected)
    assert np.allclose(diagonal, np.diag(np.diag(expected)))


def test_error_covariance_uses_observed_minus_predicted():
    predicted = np.array([[1.0], [2.0], [3.0]])
    observed = np.array([[1.5], [1.0], [3.5]])
    expected = np.cov((observed - predicted).reshape(-1), bias=False).reshape(1, 1)
    assert np.allclose(fit_error_covariance(predicted, observed, jitter=0.0), expected)
    assert_raises(fit_error_covariance, predicted, observed.reshape(-1), jitter=0.0)


def test_regularize_covariance_lifts_tiny_semidefinite_floor_only():
    lifted = regularize_covariance([[1.0, 0.0], [0.0, 0.0]], jitter=1e-6)
    assert np.linalg.eigvalsh(lifted).min() >= 1e-6 - 1e-12
    assert_raises(regularize_covariance, [[1.0, 0.0], [0.0, -0.1]], jitter=1e-9)


def test_gaussian_nll_matches_unit_normal_constant():
    assert abs(gaussian_nll([0.0], [0.0], [[1.0]], jitter=0.0) - 0.5 * math.log(2.0 * math.pi)) < 1e-12
    assert abs(gaussian_nll([0.0], [0.0], [[1.0]], jitter=0.0, include_constant=False)) < 1e-12
    assert_raises(gaussian_nll, [0.0, 1.0], [0.0], [[1.0]], jitter=0.0)


def test_shape_errors_fail_fast():
    assert_raises(gaussian_condition_update, [0.0, 1.0], [[1.0]], [0.0], [[1.0]])
    assert_raises(fit_residual_covariance, [[1.0, 2.0]], jitter=0.0)
    assert_raises(fit_residual_covariance, [[1.0], [float("nan")]], jitter=0.0)
    assert_raises(fit_residual_covariance, [[1.0], [2.0]], shrinkage=1.5, jitter=0.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman tests passed")
