#!/usr/bin/env python3
"""Tests for Product-Kalman calibration helpers.

Run: `python3 test_product_kalman_calibration.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman import gaussian_condition_update, gaussian_nll
from product_kalman_calibration import (
    ProductKalmanCalibration,
    apply_product_kalman_calibration,
    assert_disjoint_ids,
    fit_product_kalman_calibration,
)


def assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__} should have raised ValueError")


def toy_calibration_inputs():
    target = np.array([
        [0.0, 0.0],
        [1.0, 0.5],
        [2.0, -0.5],
        [3.0, 1.0],
        [4.0, 0.0],
    ])
    prior_error = np.array([
        [0.2, -0.1],
        [-0.3, 0.2],
        [0.1, 0.0],
        [0.4, -0.2],
        [-0.4, 0.1],
    ])
    measurement_error = np.array([
        [0.1],
        [-0.2],
        [0.3],
        [-0.1],
        [0.0],
    ])
    H = np.array([[1.0, 0.5]])
    prior = target - prior_error
    measurement = target @ H.T + measurement_error
    return prior, measurement, target, H, prior_error, measurement_error


def test_fit_calibration_splits_joint_covariance_into_blocks():
    prior, measurement, target, H, prior_error, measurement_error = toy_calibration_inputs()
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H, jitter=0.0)
    joint = np.concatenate([prior_error, measurement_error], axis=1)
    expected = np.cov(joint, rowvar=False, bias=False)
    np.testing.assert_allclose(cal.state_covariance, expected[:2, :2], atol=1e-12)
    np.testing.assert_allclose(cal.observation_covariance, expected[2:, 2:], atol=1e-12)
    np.testing.assert_allclose(cal.cross_covariance, expected[:2, 2:], atol=1e-12)
    np.testing.assert_allclose(cal.H, H, atol=1e-12)
    assert cal.n_samples == len(target)
    assert cal.ddof == 1


def test_apply_calibration_matches_core_update_rowwise():
    prior, measurement, target, H, _, _ = toy_calibration_inputs()
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H, jitter=0.0)
    out = apply_product_kalman_calibration(cal, prior[:2], measurement[:2], jitter=0.0)
    direct = gaussian_condition_update(
        prior[0],
        cal.state_covariance,
        measurement[0],
        cal.observation_covariance,
        H=cal.H,
        cross_covariance=cal.cross_covariance,
        jitter=0.0,
    )
    np.testing.assert_allclose(out.mean[0], direct.mean, atol=1e-12)
    np.testing.assert_allclose(out.covariance, direct.covariance, atol=1e-12)
    np.testing.assert_allclose(out.gain, direct.gain, atol=1e-12)
    np.testing.assert_allclose(out.innovation[0], direct.innovation, atol=1e-12)


def test_calibration_and_batch_results_are_read_only():
    prior, measurement, target, H, _, _ = toy_calibration_inputs()
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H)
    out = apply_product_kalman_calibration(cal, prior[:1], measurement[:1])
    for arr in (cal.state_covariance, cal.observation_covariance, cal.cross_covariance, cal.H, out.mean, out.gain):
        try:
            arr.flat[0] = 99.0
        except ValueError:
            pass
        else:
            raise AssertionError("calibration/update arrays should be read-only")


def test_assert_disjoint_ids_flags_leakage():
    assert_disjoint_ids(["a", "b"], ["c", "d"])
    assert_raises(assert_disjoint_ids, ["a", "b"], ["b", "c"])


def test_shape_and_type_errors_fail_fast():
    prior, measurement, target, H, _, _ = toy_calibration_inputs()
    assert_raises(fit_product_kalman_calibration, prior[:, :1], measurement, target, H=H)
    assert_raises(fit_product_kalman_calibration, prior, measurement[:-1], target, H=H)
    assert_raises(fit_product_kalman_calibration, prior, measurement, target, H=None)
    assert_raises(fit_product_kalman_calibration, prior, measurement, target, H=np.eye(2))
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H)
    assert_raises(apply_product_kalman_calibration, "not-calibration", prior, measurement)
    assert_raises(apply_product_kalman_calibration, cal, prior[:, :1], measurement)
    assert_raises(apply_product_kalman_calibration, cal, prior, np.hstack([measurement, measurement]))


def test_synthetic_calibration_update_improves_eval_nll_over_prior():
    rng = np.random.default_rng(11)
    n_cal = 5000
    n_eval = 2000
    n = n_cal + n_eval
    target = rng.normal(size=(n, 1))
    error_cov = np.array([[0.9, 0.25], [0.25, 0.35]])
    errors = rng.multivariate_normal([0.0, 0.0], error_cov, size=n)
    prior_error = errors[:, :1]
    measurement_error = errors[:, 1:]
    prior = target - prior_error
    measurement = target + measurement_error

    cal = fit_product_kalman_calibration(prior[:n_cal], measurement[:n_cal], target[:n_cal], jitter=1e-9)
    out = apply_product_kalman_calibration(cal, prior[n_cal:], measurement[n_cal:])

    prior_nll = np.mean([gaussian_nll(target[i], prior[i], cal.state_covariance) for i in range(n_cal, n)])
    post_nll = np.mean([gaussian_nll(target[n_cal + i], out.mean[i], out.covariance) for i in range(n_eval)])
    assert abs(cal.cross_covariance[0, 0] - error_cov[0, 1]) < 0.04
    assert post_nll < prior_nll - 0.25


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman calibration tests passed")
