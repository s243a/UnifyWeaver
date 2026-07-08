#!/usr/bin/env python3
"""Tests for Product-Kalman calibration helpers.

Run: `python3 test_product_kalman_calibration.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman import fit_residual_covariance, gaussian_condition_update, gaussian_nll
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
    expected = fit_residual_covariance(joint, jitter=0.0)
    np.testing.assert_allclose(cal.joint_covariance, expected, atol=1e-12)
    np.testing.assert_allclose(cal.state_covariance, expected[:2, :2], atol=1e-12)
    np.testing.assert_allclose(cal.observation_covariance, expected[2:, 2:], atol=1e-12)
    np.testing.assert_allclose(cal.cross_covariance, expected[:2, 2:], atol=1e-12)
    np.testing.assert_allclose(cal.H, H, atol=1e-12)
    assert cal.n_samples == len(target)
    assert cal.ddof == 1


def test_jitter_is_applied_once_to_joint_covariance_before_slicing():
    prior, measurement, target, H, prior_error, measurement_error = toy_calibration_inputs()
    jitter = 0.05
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H, jitter=jitter)
    joint = np.concatenate([prior_error, measurement_error], axis=1)
    expected = fit_residual_covariance(joint, jitter=jitter)
    np.testing.assert_allclose(cal.joint_covariance, expected, atol=1e-12)


def test_apply_calibration_matches_core_update_for_every_row():
    prior, measurement, target, H, _, _ = toy_calibration_inputs()
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H, jitter=0.0)
    out = apply_product_kalman_calibration(cal, prior[:3], measurement[:3], jitter=0.0)
    for i in range(3):
        direct = gaussian_condition_update(
            prior[i],
            cal.state_covariance,
            measurement[i],
            cal.observation_covariance,
            H=cal.H,
            cross_covariance=cal.cross_covariance,
            jitter=0.0,
        )
        np.testing.assert_allclose(out.mean[i], direct.mean, atol=1e-12)
        np.testing.assert_allclose(out.covariance, direct.covariance, atol=1e-12)
        np.testing.assert_allclose(out.gain, direct.gain, atol=1e-12)
        np.testing.assert_allclose(out.innovation[i], direct.innovation, atol=1e-12)


def test_identity_H_success_path_for_scalar_row_matrices():
    target = np.array([[0.0], [1.0], [2.0], [3.0]])
    prior = target - np.array([[0.2], [-0.1], [0.1], [-0.2]])
    measurement = target + np.array([[0.05], [-0.05], [0.02], [-0.02]])
    cal = fit_product_kalman_calibration(prior, measurement, target, H=None, jitter=0.0)
    np.testing.assert_allclose(cal.H, [[1.0]], atol=1e-12)
    out = apply_product_kalman_calibration(cal, prior[:2], measurement[:2], jitter=0.0)
    assert out.mean.shape == (2, 1)


def test_calibration_and_batch_results_are_read_only():
    prior, measurement, target, H, _, _ = toy_calibration_inputs()
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H)
    out = apply_product_kalman_calibration(cal, prior[:2], measurement[:2])
    arrays = (
        cal.state_covariance,
        cal.observation_covariance,
        cal.cross_covariance,
        cal.H,
        cal.joint_covariance,
        out.mean,
        out.covariance,
        out.gain,
        out.innovation,
        out.innovation_covariance,
    )
    for arr in arrays:
        assert not arr.flags.writeable


def test_constructor_invariants_fail_fast():
    assert_raises(
        ProductKalmanCalibration,
        np.eye(2),
        np.eye(1),
        np.zeros((1, 1)),
        np.zeros((1, 2)),
        5,
        1,
        0.0,
        "diagonal",
    )
    assert_raises(
        ProductKalmanCalibration,
        np.eye(2),
        np.eye(1),
        np.zeros((2, 1)),
        np.zeros((2, 2)),
        5,
        1,
        0.0,
        "diagonal",
    )
    assert_raises(
        ProductKalmanCalibration,
        [[1.0, 2.0]],
        np.eye(1),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        5,
        1,
        0.0,
        "diagonal",
    )
    assert_raises(
        ProductKalmanCalibration,
        np.eye(2),
        np.eye(1),
        np.zeros((2, 1)),
        np.zeros((1, 2)),
        3,
        1,
        0.0,
        "diagonal",
    )


def test_assert_disjoint_ids_flags_leakage_and_duplicates():
    assert_disjoint_ids(["a", "b"], ["c", "d"])
    assert_raises(assert_disjoint_ids, ["a", "b"], ["b", "c"])
    assert_raises(assert_disjoint_ids, ["a", "a"], ["b", "c"])
    assert_raises(assert_disjoint_ids, ["a", "b"], ["c", "c"])


def test_shape_and_type_errors_fail_fast():
    prior, measurement, target, H, _, _ = toy_calibration_inputs()
    assert_raises(fit_product_kalman_calibration, prior[:, 0], measurement, target, H=H)
    assert_raises(fit_product_kalman_calibration, prior, measurement[:, 0], target, H=H)
    assert_raises(fit_product_kalman_calibration, prior, measurement, target[:, 0], H=H)
    assert_raises(fit_product_kalman_calibration, prior[:, :1], measurement, target, H=H)
    assert_raises(fit_product_kalman_calibration, prior, measurement[:-1], target, H=H)
    assert_raises(fit_product_kalman_calibration, prior, measurement, target, H=None)
    assert_raises(fit_product_kalman_calibration, prior, measurement, target, H=np.eye(2))
    assert_raises(fit_product_kalman_calibration, prior[:3], measurement[:3], target[:3], H=H)
    cal = fit_product_kalman_calibration(prior, measurement, target, H=H)
    assert_raises(apply_product_kalman_calibration, "not-calibration", prior, measurement)
    assert_raises(apply_product_kalman_calibration, cal, prior[0], measurement[:1])
    assert_raises(apply_product_kalman_calibration, cal, prior[:, :1], measurement)
    assert_raises(apply_product_kalman_calibration, cal, prior, np.hstack([measurement, measurement]))


def test_shrinkage_path_handles_near_singular_joint_errors():
    target = np.array([[i, 2.0 * i] for i in range(6)], dtype=float)
    prior_error = np.array([[0.1 * i, 0.2 * i] for i in range(6)], dtype=float)
    measurement_error = np.array([[0.05 * i] for i in range(6)], dtype=float)
    H = np.array([[1.0, -0.25]])
    prior = target - prior_error
    measurement = target @ H.T + measurement_error
    cal = fit_product_kalman_calibration(
        prior,
        measurement,
        target,
        H=H,
        shrinkage=0.5,
        shrinkage_target="scaled_identity",
        jitter=1e-8,
    )
    out = apply_product_kalman_calibration(cal, prior[:2], measurement[:2])
    assert np.linalg.eigvalsh(cal.joint_covariance).min() > 0.0
    assert np.linalg.eigvalsh(out.innovation_covariance).min() > 0.0


def test_core_update_contract_used_by_batch_template():
    res = gaussian_condition_update([0.0], [[1.0]], [0.5], [[1.0]], cross_covariance=[[0.0]])
    for name in ("mean", "covariance", "gain", "innovation", "innovation_covariance"):
        assert hasattr(res, name)


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
    # The synthetic measurement channel is much cleaner than the prior channel; require a large enough NLL margin
    # to catch regressions that accidentally drop the measurement or its cross-covariance.
    assert post_nll < prior_nll - 0.25


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman calibration tests passed")
