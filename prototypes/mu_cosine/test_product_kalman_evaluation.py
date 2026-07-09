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
    GroupResidualCovariances,
    bootstrap_nll_improvements_from_evaluation_npz,
    evaluate_product_kalman_holdout,
    evaluation_artifact_arrays,
    evaluation_to_json_dict,
    fit_group_residual_covariances,
    main,
    paired_bootstrap_nll_improvement,
    row_covariances_from_groups,
    run_product_kalman_holdout_npz,
    score_gaussian_prediction_vectors_rowwise,
    score_gaussian_prediction_vectors,
    score_gaussian_predictions_rowwise,
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
    assert 0.85 < result.score("product_kalman").mahalanobis_per_dim < 1.15
    product_score = result.score("product_kalman")
    assert product_score.squared_mahalanobis_q50 <= product_score.squared_mahalanobis_q90
    assert product_score.squared_mahalanobis_q90 <= product_score.squared_mahalanobis_q95
    product_vectors = result.score_vector("product_kalman")
    assert product_vectors.nll.shape == (len(eval_target),)
    assert abs(float(product_vectors.nll.mean()) - product_score.mean_nll) < 1e-12
    assert product_vectors.nll.flags.writeable is False
    assert result.correlated_update.mean.flags.writeable is False
    assert result.independent_update.mean.flags.writeable is False
    boot = paired_bootstrap_nll_improvement(
        result,
        "independent_kalman",
        "product_kalman",
        n_boot=200,
        seed=11,
        confidence=0.90,
    )
    observed_gain = result.nll_improvement("independent_kalman", "product_kalman")
    assert abs(boot["observed_mean_gain"] - observed_gain) < 1e-12
    assert boot["ci_low"] < boot["observed_mean_gain"] < boot["ci_high"]
    assert boot["method"] == "paired_row_resample"
    assert boot["confidence"] == 0.90


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
        data = evaluation_to_json_dict(
            result,
            bootstrap_nll=200,
            bootstrap_seed=3,
            bootstrap_confidence=0.90,
        )
        assert data["score_order"] == ["prior", "measurement", "independent_kalman", "product_kalman"]
        assert data["calibration"]["state_dim"] == 1
        assert "mean_squared_mahalanobis" in data["scores"]["product_kalman"]
        assert "squared_mahalanobis_q95" in data["scores"]["product_kalman"]
        assert 0.85 < data["scores"]["product_kalman"]["mahalanobis_per_dim"] < 1.15
        assert data["nll_improvement_vs_prior"]["product_kalman"] > 0.85
        assert data["nll_improvement_vs_independent_kalman"]["product_kalman"] > 0.12
        boot = data["nll_improvement_bootstrap_vs_independent_kalman"]["product_kalman"]
        assert boot["n_boot"] == 200
        assert boot["seed"] == 3
        assert boot["confidence"] == 0.90
        assert abs(
            boot["observed_mean_gain"]
            - data["nll_improvement_vs_independent_kalman"]["product_kalman"]
        ) < 1e-12

        rc = main([
            str(input_path),
            "--output-json",
            str(output_path),
            "--output-npz",
            str(artifact_path),
            "--bootstrap-nll",
            "200",
            "--bootstrap-seed",
            "3",
            "--bootstrap-confidence",
            "0.90",
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
        assert from_cli["nll_improvement_bootstrap_vs_independent_kalman"]["product_kalman"] == boot

        artifact_boot = bootstrap_nll_improvements_from_evaluation_npz(
            artifact_path,
            n_boot=200,
            seed=3,
            confidence=0.90,
        )
        assert artifact_boot["nll_improvement_bootstrap_vs_independent_kalman"]["product_kalman"] == boot

        with np.load(artifact_path, allow_pickle=False) as artifact:
            assert int(artifact["schema_version"]) == 1
            assert artifact["score_names"].tolist() == data["score_order"]
            assert artifact["score_mahalanobis_per_dim"].shape == (4,)
            assert artifact["score_squared_mahalanobis_q95"].shape == (4,)
            assert artifact["score_row_nll"].shape == (4, 4000)
            assert artifact["score_row_squared_error"].shape == (4, 4000)
            assert artifact["score_row_squared_mahalanobis"].shape == (4, 4000)
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
    assert arrays["score_mean_squared_mahalanobis"].shape == (4,)
    assert arrays["score_squared_mahalanobis_q50"].shape == (4,)
    assert arrays["score_squared_mahalanobis_q90"].shape == (4,)
    assert arrays["score_squared_mahalanobis_q95"].shape == (4,)
    assert arrays["score_row_nll"].shape == (4, 20)
    assert arrays["score_row_squared_error"].shape == (4, 20)
    assert arrays["score_row_squared_mahalanobis"].shape == (4, 20)
    np.testing.assert_allclose(arrays["score_row_nll"].mean(axis=1), arrays["score_mean_nll"])
    assert np.isfinite(arrays["score_mahalanobis_per_dim"]).all()
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
        assert_raises(bootstrap_nll_improvements_from_evaluation_npz, input_path, n_boot=10)


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


def test_group_residual_covariances_feed_rowwise_scores():
    rng = np.random.default_rng(17)
    n_cal_per_group = 240
    n_eval_per_group = 120
    cal_groups = np.array(["tight"] * n_cal_per_group + ["wide"] * n_cal_per_group)
    eval_groups = np.array(["tight"] * n_eval_per_group + ["wide"] * n_eval_per_group)
    cal_pred = np.zeros((2 * n_cal_per_group, 1))
    eval_pred = np.zeros((2 * n_eval_per_group, 1))
    cal_obs = np.concatenate([
        rng.normal(scale=0.20, size=n_cal_per_group),
        rng.normal(scale=1.10, size=n_cal_per_group),
    ])[:, None]
    eval_obs = np.concatenate([
        rng.normal(scale=0.20, size=n_eval_per_group),
        rng.normal(scale=1.10, size=n_eval_per_group),
    ])[:, None]

    fitted = fit_group_residual_covariances(cal_pred, cal_obs, cal_groups, min_group_rows=20, jitter=1e-8)
    assert fitted.group_counts == {"tight": n_cal_per_group, "wide": n_cal_per_group}
    assert fitted.covariance_by_group["tight"][0, 0] < fitted.fallback_covariance[0, 0]
    assert fitted.covariance_by_group["wide"][0, 0] > fitted.fallback_covariance[0, 0]
    row_covariances = fitted.row_covariances(eval_groups)
    assert row_covariances.shape == (2 * n_eval_per_group, 1, 1)
    assert row_covariances.flags.writeable is False

    global_score = score_gaussian_predictions("global", eval_obs, eval_pred, fitted.fallback_covariance, jitter=1e-8)
    grouped_score = score_gaussian_predictions_rowwise("grouped", eval_obs, eval_pred, row_covariances, jitter=1e-8)
    assert grouped_score.mean_nll < global_score.mean_nll
    assert grouped_score.covariance_trace == np.mean(np.trace(row_covariances, axis1=1, axis2=2))


def test_group_residual_covariance_fallbacks_and_validation():
    pred = np.zeros((8, 1))
    obs = np.array([[0.0], [0.2], [-0.1], [0.1], [1.0], [-1.1], [0.9], [0.05]])
    groups = np.array(["common", "common", "common", "common", "wide", "wide", "wide", "rare"])
    fitted = fit_group_residual_covariances(pred, obs, groups, min_group_rows=3, jitter=1e-8)
    assert fitted.group_counts == {"common": 4, "wide": 3, "rare": 1}
    np.testing.assert_allclose(fitted.covariance_by_group["rare"], fitted.fallback_covariance)
    rows = row_covariances_from_groups(
        ["common", "rare", "unseen"],
        fitted.covariance_by_group,
        fallback_covariance=fitted.fallback_covariance,
    )
    assert rows.shape == (3, 1, 1)
    np.testing.assert_allclose(rows[2], fitted.fallback_covariance)
    assert_raises(row_covariances_from_groups, ["unseen"], fitted.covariance_by_group)
    assert_raises(fit_group_residual_covariances, pred, obs, ["x"] * 7)
    assert_raises(fit_group_residual_covariances, pred, obs, ["x"] * 8, min_group_rows=1)
    assert_raises(
        GroupResidualCovariances,
        covariance_by_group={"a": [[1.0]]},
        fallback_covariance=[[1.0]],
        group_counts={"a": 1, "extra": 1},
        min_group_rows=2,
    )


def test_score_gaussian_predictions_validates_shapes_and_reports_mse():
    target = np.array([[1.0], [2.0]])
    mean = np.array([[1.5], [1.5]])
    vectors = score_gaussian_prediction_vectors("toy", target, mean, [[0.25]], jitter=1e-9)
    assert vectors.name == "toy"
    assert np.allclose(vectors.squared_error, [0.25, 0.25])
    assert np.allclose(vectors.squared_mahalanobis, [1.0, 1.0])
    assert vectors.nll.flags.writeable is False
    score = score_gaussian_predictions("toy", target, mean, [[0.25]], jitter=1e-9)
    assert score.name == "toy"
    assert score.n == 2
    assert abs(score.mse - 0.25) < 1e-12
    assert score.covariance_trace == 0.25
    assert abs(score.mean_squared_mahalanobis - 1.0) < 1e-12
    assert abs(score.mahalanobis_per_dim - 1.0) < 1e-12
    assert abs(score.squared_mahalanobis_q50 - 1.0) < 1e-12
    assert abs(score.squared_mahalanobis_q90 - 1.0) < 1e-12
    assert abs(score.squared_mahalanobis_q95 - 1.0) < 1e-12
    assert_raises(score_gaussian_predictions, "bad", target[:, 0], mean, [[1.0]])
    assert_raises(score_gaussian_predictions, "bad", target, mean, [[1.0, 0.0]])


def test_rowwise_gaussian_scoring_supports_variable_covariances():
    target = np.array([[1.0], [2.0]])
    mean = np.array([[1.5], [1.5]])
    shared_covariances = np.array([[[0.25]], [[0.25]]])
    shared_vectors = score_gaussian_prediction_vectors("shared", target, mean, [[0.25]], jitter=1e-9)
    rowwise_shared = score_gaussian_prediction_vectors_rowwise(
        "shared",
        target,
        mean,
        shared_covariances,
        jitter=1e-9,
    )
    np.testing.assert_allclose(rowwise_shared.nll, shared_vectors.nll)
    np.testing.assert_allclose(rowwise_shared.squared_mahalanobis, shared_vectors.squared_mahalanobis)

    row_covariances = np.array([[[0.25]], [[1.0]]])
    row_vectors = score_gaussian_prediction_vectors_rowwise("rowwise", target, mean, row_covariances, jitter=1e-9)
    assert row_vectors.name == "rowwise"
    assert np.allclose(row_vectors.squared_error, [0.25, 0.25])
    assert np.allclose(row_vectors.squared_mahalanobis, [1.0, 0.25])
    assert row_vectors.nll.flags.writeable is False
    row_score = score_gaussian_predictions_rowwise("rowwise", target, mean, row_covariances, jitter=1e-9)
    assert row_score.n == 2
    assert abs(row_score.mse - 0.25) < 1e-12
    assert abs(row_score.covariance_trace - 0.625) < 1e-12
    assert abs(row_score.mean_squared_mahalanobis - 0.625) < 1e-12
    assert abs(row_score.mahalanobis_per_dim - 0.625) < 1e-12
    assert_raises(score_gaussian_predictions_rowwise, "bad", target, mean, row_covariances[:1])
    assert_raises(score_gaussian_predictions_rowwise, "bad", target, mean, row_covariances[:, :, 0])
    bad_covariances = row_covariances.copy()
    bad_covariances[1, 0, 0] = np.nan
    assert_raises(score_gaussian_predictions_rowwise, "bad", target, mean, bad_covariances)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman evaluation tests passed")
