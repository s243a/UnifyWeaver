#!/usr/bin/env python3
"""Holdout evaluation helpers for Product-Kalman calibration prototypes.

This module is the bridge between the covariance-fitting helpers and later real
corpus experiments. It fits Product-Kalman covariance blocks on a calibration
split, applies them to a separate evaluation split, and reports comparable
Gaussian NLL/MSE scores for the prior, a zero-cross-covariance update, and the
correlated Product-Kalman update.

The helpers are deliberately numpy-only and require caller-supplied split rows.
They do not sample data, train a model, or decide that a Product-Kalman variant
has won; they make the held-out comparison auditable.
"""

from dataclasses import dataclass

import numpy as np

try:
    from .product_kalman import gaussian_nll
    from .product_kalman_calibration import (
        ProductKalmanCalibration,
        apply_product_kalman_calibration,
        assert_disjoint_ids,
        fit_product_kalman_calibration,
    )
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman import gaussian_nll
    from product_kalman_calibration import (
        ProductKalmanCalibration,
        apply_product_kalman_calibration,
        assert_disjoint_ids,
        fit_product_kalman_calibration,
    )


__all__ = [
    "GaussianScore",
    "ProductKalmanHoldoutEvaluation",
    "evaluate_product_kalman_holdout",
    "score_gaussian_predictions",
]


def _as_2d(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D row matrix; use shape (n, 1) for scalar channels")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _as_covariance(name, value, dim):
    cov = np.asarray(value, dtype=float)
    if cov.shape != (dim, dim):
        raise ValueError(f"{name} shape {cov.shape} must be ({dim}, {dim})")
    if not np.isfinite(cov).all():
        raise ValueError(f"{name} must be finite")
    return 0.5 * (cov + cov.T)


@dataclass(frozen=True)
class GaussianScore:
    """Scalar summary for one Gaussian prediction family on one held-out split."""

    name: str
    mean_nll: float
    mse: float
    n: int
    covariance_trace: float

    def __post_init__(self):
        name = str(self.name)
        n = int(self.n)
        mean_nll = float(self.mean_nll)
        mse = float(self.mse)
        covariance_trace = float(self.covariance_trace)
        if not name:
            raise ValueError("score name must be nonempty")
        if n <= 0:
            raise ValueError("score n must be positive")
        for field, value in (("mean_nll", mean_nll), ("mse", mse), ("covariance_trace", covariance_trace)):
            if not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "mean_nll", mean_nll)
        object.__setattr__(self, "mse", mse)
        object.__setattr__(self, "covariance_trace", covariance_trace)


@dataclass(frozen=True)
class ProductKalmanHoldoutEvaluation:
    """One calibration/evaluation split comparison.

    `calibration` contains the fitted correlated covariance blocks. `independent`
    is the same fit with `cross_covariance=0`, used as the explicit no-correlation
    control. The two batch updates are retained so callers can inspect row-level
    posterior means after reading the aggregate scores.
    """

    calibration: ProductKalmanCalibration
    independent_calibration: ProductKalmanCalibration
    correlated_update: object
    independent_update: object
    scores: tuple

    def __post_init__(self):
        if not isinstance(self.calibration, ProductKalmanCalibration):
            raise ValueError("calibration must be a ProductKalmanCalibration")
        if not isinstance(self.independent_calibration, ProductKalmanCalibration):
            raise ValueError("independent_calibration must be a ProductKalmanCalibration")
        scores = tuple(self.scores)
        names = [s.name for s in scores]
        if len(names) != len(set(names)):
            raise ValueError("score names must be unique")
        object.__setattr__(self, "scores", scores)

    def score(self, name):
        """Return the named `GaussianScore`."""
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(name)

    def nll_improvement(self, baseline, candidate):
        """Positive means `candidate` has lower mean NLL than `baseline`."""
        return self.score(baseline).mean_nll - self.score(candidate).mean_nll


def score_gaussian_predictions(name, target_state, mean, covariance, jitter=1e-9):
    """Score row-wise Gaussian predictions against held-out target states.

    `target_state` and `mean` must be `(n, d)` row matrices. `covariance` is one
    shared `(d, d)` covariance for the prediction family, matching the current
    Product-Kalman calibration/update contract.
    """
    target = _as_2d("target_state", target_state)
    pred = _as_2d("mean", mean)
    if pred.shape != target.shape:
        raise ValueError(f"mean shape {pred.shape} must match target_state shape {target.shape}")
    cov = _as_covariance("covariance", covariance, target.shape[1])
    nll = [gaussian_nll(target[i], pred[i], cov, jitter=jitter) for i in range(target.shape[0])]
    residual = target - pred
    mse = float(np.mean(np.sum(residual * residual, axis=1)))
    return GaussianScore(
        name=name,
        mean_nll=float(np.mean(nll)),
        mse=mse,
        n=int(target.shape[0]),
        covariance_trace=float(np.trace(cov)),
    )


def _check_split_ids(calibration_ids, evaluation_ids, n_calibration, n_evaluation):
    if calibration_ids is None and evaluation_ids is None:
        return
    if calibration_ids is None or evaluation_ids is None:
        raise ValueError("calibration_ids and evaluation_ids must be provided together")
    calibration_ids = list(calibration_ids)
    evaluation_ids = list(evaluation_ids)
    if len(calibration_ids) != n_calibration:
        raise ValueError("calibration_ids length must match calibration rows")
    if len(evaluation_ids) != n_evaluation:
        raise ValueError("evaluation_ids length must match evaluation rows")
    assert_disjoint_ids(calibration_ids, evaluation_ids)


def _zero_cross_calibration(calibration):
    return ProductKalmanCalibration(
        state_covariance=calibration.state_covariance,
        observation_covariance=calibration.observation_covariance,
        cross_covariance=np.zeros_like(calibration.cross_covariance),
        H=calibration.H,
        n_samples=calibration.n_samples,
        ddof=calibration.ddof,
        shrinkage=calibration.shrinkage,
        shrinkage_target=calibration.shrinkage_target,
    )


def _has_identity_observation(calibration):
    if calibration.state_dim != calibration.observation_dim:
        return False
    return bool(np.allclose(calibration.H, np.eye(calibration.state_dim), atol=1e-12, rtol=1e-12))


def evaluate_product_kalman_holdout(
    calibration_prior_mean,
    calibration_measurement,
    calibration_target_state,
    evaluation_prior_mean,
    evaluation_measurement,
    evaluation_target_state,
    H=None,
    calibration_ids=None,
    evaluation_ids=None,
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
):
    """Fit calibration blocks on one split and score updates on a held-out split.

    The returned scores always include:
    - `prior`: evaluation prior mean with fitted prior-error covariance;
    - `independent_kalman`: same fitted `P`/`R`, but `C=0`;
    - `product_kalman`: fitted correlated Product-Kalman update.

    If `H` is identity and observation/state dimensions match, a `measurement`
    baseline is also scored. Non-identity observations are not directly target-
    coordinate predictions, so the measurement baseline is omitted rather than
    silently applying an inverse map.
    """
    cal_prior = _as_2d("calibration_prior_mean", calibration_prior_mean)
    cal_measure = _as_2d("calibration_measurement", calibration_measurement)
    cal_target = _as_2d("calibration_target_state", calibration_target_state)
    eval_prior = _as_2d("evaluation_prior_mean", evaluation_prior_mean)
    eval_measure = _as_2d("evaluation_measurement", evaluation_measurement)
    eval_target = _as_2d("evaluation_target_state", evaluation_target_state)
    _check_split_ids(calibration_ids, evaluation_ids, cal_target.shape[0], eval_target.shape[0])

    calibration = fit_product_kalman_calibration(
        cal_prior,
        cal_measure,
        cal_target,
        H=H,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )
    independent = _zero_cross_calibration(calibration)
    correlated_update = apply_product_kalman_calibration(calibration, eval_prior, eval_measure, jitter=jitter)
    independent_update = apply_product_kalman_calibration(independent, eval_prior, eval_measure, jitter=jitter)

    scores = [
        score_gaussian_predictions("prior", eval_target, eval_prior, calibration.state_covariance, jitter=jitter),
        score_gaussian_predictions(
            "independent_kalman",
            eval_target,
            independent_update.mean,
            independent_update.covariance,
            jitter=jitter,
        ),
        score_gaussian_predictions(
            "product_kalman",
            eval_target,
            correlated_update.mean,
            correlated_update.covariance,
            jitter=jitter,
        ),
    ]
    if _has_identity_observation(calibration):
        scores.insert(
            1,
            score_gaussian_predictions(
                "measurement",
                eval_target,
                eval_measure,
                calibration.observation_covariance,
                jitter=jitter,
            ),
        )

    return ProductKalmanHoldoutEvaluation(
        calibration=calibration,
        independent_calibration=independent,
        correlated_update=correlated_update,
        independent_update=independent_update,
        scores=tuple(scores),
    )
