#!/usr/bin/env python3
"""Tests for Product-Kalman holdout evaluation helpers.

Run: `python3 test_product_kalman_evaluation.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman_evaluation import (
    evaluate_product_kalman_holdout,
    score_gaussian_predictions,
)


def assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (KeyError, ValueError):
        return
    raise AssertionError(f"{fn.__name__} should have raised")


def synthetic_identity_split(n_cal=8000, n_eval=4000, seed=23):
    rng = np.random.default_rng(seed)
    n = n_cal + n_eval
    target = rng.normal(size=(n, 1))
    error_covariance = np.array([[1.0, 0.4], [0.4, 0.4]])
    errors = rng.multivariate_normal([0.0, 0.0], error_covariance, size=n)
    prior = target - errors[:, :1]
    measurement = target + errors[:, 1:]
    return (
        prior[:n_cal],
        measurement[:n_cal],
        target[:n_cal],
        prior[n_cal:],
        measurement[n_cal:],
        target[n_cal:],
    )


def test_correlated_product_kalman_beats_zero_cross_control_on_heldout_nll():
    cal_prior, cal_measure, cal_target, eval_prior, eval_measure, eval_target = synthetic_identity_split()
    result = evaluate_product_kalman_holdout(
        cal_prior,
        cal_measure,
        cal_target,
        eval_prior,
        eval_measure,
        eval_target,
        calibration_ids=[f"cal-{i}" for i in range(len(cal_target))],
        evaluation_ids=[f"eval-{i}" for i in range(len(eval_target))],
        jitter=1e-9,
    )
    names = [score.name for score in result.scores]
    assert names == ["prior", "measurement", "independent_kalman", "product_kalman"]
    assert abs(result.calibration.cross_covariance[0, 0] - 0.4) < 0.035
    assert result.nll_improvement("prior", "product_kalman") > 0.85
    assert result.nll_improvement("independent_kalman", "product_kalman") > 0.12
    assert result.score("product_kalman").mse < result.score("independent_kalman").mse
    assert result.correlated_update.mean.flags.writeable is False
    assert result.independent_update.mean.flags.writeable is False


def test_split_ids_are_checked_before_scoring():
    cal_prior, cal_measure, cal_target, eval_prior, eval_measure, eval_target = synthetic_identity_split(
        n_cal=10,
        n_eval=6,
    )
    assert_raises(
        evaluate_product_kalman_holdout,
        cal_prior,
        cal_measure,
        cal_target,
        eval_prior,
        eval_measure,
        eval_target,
        calibration_ids=["shared"] + [f"cal-{i}" for i in range(9)],
        evaluation_ids=["shared"] + [f"eval-{i}" for i in range(5)],
    )
    assert_raises(
        evaluate_product_kalman_holdout,
        cal_prior,
        cal_measure,
        cal_target,
        eval_prior,
        eval_measure,
        eval_target,
        calibration_ids=[f"cal-{i}" for i in range(9)],
        evaluation_ids=[f"eval-{i}" for i in range(6)],
    )
    assert_raises(
        evaluate_product_kalman_holdout,
        cal_prior,
        cal_measure,
        cal_target,
        eval_prior,
        eval_measure,
        eval_target,
        calibration_ids=["dup", "dup"] + [f"cal-{i}" for i in range(8)],
        evaluation_ids=[f"eval-{i}" for i in range(6)],
    )


def test_nonidentity_observation_omits_measurement_baseline():
    rng = np.random.default_rng(5)
    n_cal = 200
    n_eval = 80
    n = n_cal + n_eval
    target = rng.normal(size=(n, 2))
    H = np.array([[1.0, -0.4]])
    prior_error = rng.multivariate_normal([0.0, 0.0], [[0.7, 0.1], [0.1, 0.5]], size=n)
    measurement_error = rng.normal(scale=0.25, size=(n, 1))
    prior = target - prior_error
    measurement = target @ H.T + measurement_error
    result = evaluate_product_kalman_holdout(
        prior[:n_cal],
        measurement[:n_cal],
        target[:n_cal],
        prior[n_cal:],
        measurement[n_cal:],
        target[n_cal:],
        H=H,
        jitter=1e-8,
    )
    names = [score.name for score in result.scores]
    assert names == ["prior", "independent_kalman", "product_kalman"]
    assert result.correlated_update.mean.shape == (n_eval, 2)
    assert_raises(result.score, "measurement")


def test_score_gaussian_predictions_validates_shapes_and_reports_mse():
    target = np.array([[1.0], [2.0]])
    mean = np.array([[1.5], [1.5]])
    score = score_gaussian_predictions("toy", target, mean, [[0.25]], jitter=1e-9)
    assert score.name == "toy"
    assert score.n == 2
    assert abs(score.mse - 0.25) < 1e-12
    assert score.covariance_trace == 0.25
    assert_raises(score_gaussian_predictions, "bad", target[:, 0], mean, [[1.0]])
    assert_raises(score_gaussian_predictions, "bad", target, mean, [[1.0, 0.0]])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman evaluation tests passed")
