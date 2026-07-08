#!/usr/bin/env python3
"""Tests for Product-Kalman holdout evaluation helpers.

Run: `python3 test_product_kalman_evaluation.py`.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman_evaluation import (
    evaluate_product_kalman_holdout,
    evaluation_artifact_arrays,
    evaluation_to_json_dict,
    main,
    run_product_kalman_holdout_npz,
    score_gaussian_predictions,
    write_evaluation_npz,
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


def write_identity_npz(path, n_cal=8000, n_eval=4000):
    cal_prior, cal_measure, cal_target, eval_prior, eval_measure, eval_target = synthetic_identity_split(
        n_cal=n_cal,
        n_eval=n_eval,
    )
    np.savez(
        path,
        calibration_prior_mean=cal_prior,
        calibration_measurement=cal_measure,
        calibration_target_state=cal_target,
        evaluation_prior_mean=eval_prior,
        evaluation_measurement=eval_measure,
        evaluation_target_state=eval_target,
        calibration_ids=np.array([f"cal-{i}" for i in range(n_cal)]),
        evaluation_ids=np.array([f"eval-{i}" for i in range(n_eval)]),
    )


def test_npz_runner_and_json_cli_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "holdout.npz"
        output_path = Path(tmp) / "scores.json"
        artifact_path = Path(tmp) / "artifacts.npz"
        write_identity_npz(input_path)
        result = run_product_kalman_holdout_npz(input_path)
        data = evaluation_to_json_dict(result)
        assert data["score_order"] == ["prior", "measurement", "independent_kalman", "product_kalman"]
        assert data["calibration"]["state_dim"] == 1
        assert data["nll_improvement_vs_prior"]["product_kalman"] > 0.85
        assert data["nll_improvement_vs_independent_kalman"]["product_kalman"] > 0.12

        rc = main([
            str(input_path),
            "--output-json",
            str(output_path),
            "--output-npz",
            str(artifact_path),
            "--indent",
            "0",
        ])
        assert rc == 0
        from_cli = json.loads(output_path.read_text())
        assert from_cli["score_order"] == data["score_order"]
        assert abs(
            from_cli["scores"]["product_kalman"]["mean_nll"]
            - data["scores"]["product_kalman"]["mean_nll"]
        ) < 1e-12

        with np.load(artifact_path, allow_pickle=False) as artifact:
            assert int(artifact["schema_version"]) == 1
            assert artifact["score_names"].tolist() == data["score_order"]
            assert artifact["product_kalman_mean"].shape == result.correlated_update.mean.shape
            np.testing.assert_allclose(artifact["product_kalman_mean"], result.correlated_update.mean)
            np.testing.assert_allclose(artifact["independent_kalman_mean"], result.independent_update.mean)
            np.testing.assert_allclose(artifact["calibration_cross_covariance"], result.calibration.cross_covariance)
            np.testing.assert_allclose(artifact["independent_cross_covariance"], 0.0)


def test_evaluation_artifact_arrays_are_npz_ready():
    cal_prior, cal_measure, cal_target, eval_prior, eval_measure, eval_target = synthetic_identity_split(
        n_cal=80,
        n_eval=20,
    )
    result = evaluate_product_kalman_holdout(
        cal_prior,
        cal_measure,
        cal_target,
        eval_prior,
        eval_measure,
        eval_target,
        jitter=1e-8,
    )
    arrays = evaluation_artifact_arrays(result)
    assert arrays["score_names"].tolist() == ["prior", "measurement", "independent_kalman", "product_kalman"]
    assert arrays["score_mean_nll"].shape == (4,)
    assert arrays["product_kalman_innovation"].shape == eval_target.shape
    assert arrays["product_kalman_covariance"].shape == (1, 1)
    assert arrays["independent_kalman_gain"].shape == (1, 1)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = Path(tmp) / "artifact.npz"
        write_evaluation_npz(artifact_path, result)
        with np.load(artifact_path, allow_pickle=False) as artifact:
            assert set(arrays).issubset(set(artifact.files))
            np.testing.assert_allclose(artifact["score_mean_nll"], arrays["score_mean_nll"])


def test_npz_runner_validates_required_keys():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "missing.npz"
        np.savez(input_path, calibration_prior_mean=np.zeros((4, 1)))
        assert_raises(run_product_kalman_holdout_npz, input_path)


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
