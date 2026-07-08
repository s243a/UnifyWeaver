#!/usr/bin/env python3
"""Gaussian conditioning core for Product-Kalman / correlated-PoE prototypes.

The functions here operate in an already chosen evidence coordinate, such as
`log(mu_lower)`, `logit(mu_direct)`, or another finite product-space link. They do
not construct product proxies themselves; use `product_space.py` for that layer.

This module deliberately exposes cross-covariance between the prior state error and
measurement noise. When prior and measurement channels share evidence, treating that
cross term as zero is an assumption, not a default fact.

Prototype note: this module prioritizes correctness and auditability over speed. A
full update regularizes several covariance matrices with eigendecompositions; a
performance-sensitive path should use a Cholesky-specialized implementation.
"""

from dataclasses import dataclass
import warnings

import numpy as np


NUMERIC_EIGEN_FLOOR = 1e-12
SEMIDEFINITE_TOL_MULTIPLIER = 100.0

__all__ = [
    "GaussianUpdate",
    "NUMERIC_EIGEN_FLOOR",
    "SEMIDEFINITE_TOL_MULTIPLIER",
    "fit_error_covariance",
    "fit_residual_covariance",
    "gaussian_condition_update",
    "gaussian_nll",
    "regularize_covariance",
    "scalar_product_kalman_update",
]


def _readonly_array(value):
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class GaussianUpdate:
    """Result of one Gaussian conditioning / Kalman-style update.

    Array fields are copied and marked read-only in `__post_init__`; `frozen=True`
    alone would not prevent in-place mutation of numpy arrays.
    """

    mean: np.ndarray
    covariance: np.ndarray
    gain: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    cross_covariance: np.ndarray

    def __post_init__(self):
        for name in ("mean", "covariance", "gain", "innovation", "innovation_covariance", "cross_covariance"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))


def _as_vector(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, column vector, row vector, or scalar")
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _as_matrix(name, value, rows=None, cols=None):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix or scalar")
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {rows}")
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"{name} has {arr.shape[1]} columns, expected {cols}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _square_covariance(name, value, dim):
    cov = _as_matrix(name, value, dim, dim)
    return 0.5 * (cov + cov.T)


def regularize_covariance(covariance, jitter=1e-9, name="covariance"):
    """Return a symmetric positive-definite covariance with diagonal jitter.

    Small negative eigenvalues from floating-point symmetry error are lifted. A
    materially negative eigenvalue raises instead of silently accepting an invalid
    covariance model. Even when `jitter=0`, exact semidefinite covariances are
    lifted to `NUMERIC_EIGEN_FLOOR` so downstream solves see positive-definite
    matrices rather than singular ones.
    """
    if jitter < 0:
        raise ValueError("jitter must be nonnegative")
    cov = _as_matrix(name, covariance)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square")
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    min_eig = float(eig[0])
    floor = max(float(jitter), NUMERIC_EIGEN_FLOOR)
    if min_eig < -SEMIDEFINITE_TOL_MULTIPLIER * floor:
        raise ValueError(f"{name} is not positive semidefinite; min eigenvalue={min_eig:g}")
    lift = max(floor - min_eig, 0.0)
    if lift:
        cov = cov + np.eye(cov.shape[0]) * lift
    return cov


def _shrinkage_target(cov, target):
    if target == "diagonal":
        return np.diag(np.diag(cov))
    if target == "scaled_identity":
        return np.eye(cov.shape[0]) * float(np.trace(cov) / cov.shape[0])
    raise ValueError("shrinkage_target must be 'diagonal' or 'scaled_identity'")


def fit_residual_covariance(residuals, shrinkage=0.0, jitter=1e-9, ddof=1, shrinkage_target="diagonal"):
    """Estimate residual covariance from rows of calibration residuals.

    `ddof=1` gives the unbiased sample covariance; use `ddof=0` for an MLE
    covariance estimate. `shrinkage=1` keeps only the selected target covariance:
    sample `diagonal` by default, or `scaled_identity` for a Ledoit-Wolf-style
    isotropic target. Use residuals from an appropriate calibration split, not the
    data used for the claimed evaluation.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a nonnegative integer")
    r = np.asarray(residuals, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if r.ndim != 2:
        raise ValueError("residuals must be a 1-D or 2-D array")
    if r.shape[0] <= ddof:
        raise ValueError("residual rows must exceed ddof")
    if not np.isfinite(r).all():
        raise ValueError("residuals must be finite")
    centered = r - r.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / float(r.shape[0] - ddof)
    if shrinkage:
        target = _shrinkage_target(cov, shrinkage_target)
        cov = (1.0 - shrinkage) * cov + shrinkage * target
    return regularize_covariance(cov, jitter=jitter, name="residual covariance")


def fit_error_covariance(predicted, observed, shrinkage=0.0, jitter=1e-9, ddof=1, shrinkage_target="diagonal"):
    """Estimate covariance of `observed - predicted` calibration errors."""
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"predicted shape {pred.shape} must match observed shape {obs.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predicted must be finite")
    if not np.isfinite(obs).all():
        raise ValueError("observed must be finite")
    return fit_residual_covariance(
        obs - pred,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )


def gaussian_condition_update(mean, covariance, observation, observation_covariance, H=None, cross_covariance=None,
                              jitter=1e-9):
    """Condition a Gaussian state on a linear noisy observation.

    State prior:
        x ~ N(mean, covariance)
    Observation:
        y = H x + v

    `observation_covariance` is Cov(v). `cross_covariance` is Cov(x - mean, v),
    with shape `(state_dim, obs_dim)`. If omitted, it is treated as zero; callers
    should pass it explicitly when prior and measurement channels share evidence.

    The returned `innovation_covariance` is the regularized S used for the gain
    computation; it may include a small diagonal lift for numerical positive
    definiteness. The posterior covariance is the Gaussian-conditioning covariance
    `P - Cov(x,y) S^-1 Cov(y,x)`, symmetrized/regularized after the update.
    """
    m = _as_vector("mean", mean)
    y = _as_vector("observation", observation)
    n = m.size
    k = y.size
    P = regularize_covariance(_square_covariance("covariance", covariance, n), jitter=jitter, name="covariance")
    R = regularize_covariance(
        _square_covariance("observation_covariance", observation_covariance, k),
        jitter=jitter,
        name="observation covariance",
    )
    if H is None:
        if k != n:
            raise ValueError("H is required when observation dimension differs from state dimension")
        Hm = np.eye(n)
    else:
        Hm = _as_matrix("H", H, k, n)
    C = np.zeros((n, k)) if cross_covariance is None else _as_matrix("cross_covariance", cross_covariance, n, k)

    innovation = y - Hm @ m
    innovation_cov = Hm @ P @ Hm.T + Hm @ C + C.T @ Hm.T + R
    innovation_cov = regularize_covariance(innovation_cov, jitter=jitter, name="innovation covariance")
    state_innovation_cov = P @ Hm.T + C
    chol = np.linalg.cholesky(innovation_cov)
    gain = np.linalg.solve(chol.T, np.linalg.solve(chol, state_innovation_cov.T)).T
    posterior_mean = m + gain @ innovation
    posterior_cov = P - gain @ innovation_cov @ gain.T
    posterior_cov = regularize_covariance(posterior_cov, jitter=jitter, name="posterior covariance")

    return GaussianUpdate(
        mean=posterior_mean,
        covariance=posterior_cov,
        gain=gain,
        innovation=innovation,
        innovation_covariance=innovation_cov,
        cross_covariance=C,
    )


def scalar_product_kalman_update(ell_prior, ell_measurement, prior_var, measurement_var, cross_covariance=None,
                                 jitter=1e-9):
    """Scalar Product-Kalman update in one log/link evidence coordinate.

    `cross_covariance=None` warns and assumes zero. Pass `0.0` explicitly only
    when independence or negligible prior-measurement covariance is an intentional
    modeling assumption.
    """
    if not np.isfinite(prior_var) or prior_var <= 0:
        raise ValueError(f"prior_var must be finite and positive, got {prior_var}")
    if not np.isfinite(measurement_var) or measurement_var <= 0:
        raise ValueError(f"measurement_var must be finite and positive, got {measurement_var}")
    if cross_covariance is None:
        warnings.warn(
            "assuming zero prior-measurement cross_covariance; pass 0.0 explicitly if this is intentional",
            RuntimeWarning,
            stacklevel=2,
        )
        cross_covariance = 0.0
    return gaussian_condition_update(
        [ell_prior],
        [[prior_var]],
        [ell_measurement],
        [[measurement_var]],
        H=[[1.0]],
        cross_covariance=[[cross_covariance]],
        jitter=jitter,
    )


def gaussian_nll(observed, mean, covariance, jitter=1e-9, include_constant=True):
    """Negative log likelihood of one Gaussian observation.

    Argument order is `observed, mean, covariance`, matching residual notation
    `observed - mean` used elsewhere in this prototype.
    """
    y = _as_vector("observed", observed)
    m = _as_vector("mean", mean)
    if y.shape != m.shape:
        raise ValueError(f"observed shape {y.shape} must match mean shape {m.shape}")
    cov = regularize_covariance(_square_covariance("covariance", covariance, y.size), jitter=jitter, name="covariance")
    residual = y - m
    solved = np.linalg.solve(cov, residual)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:  # Defensive guard after covariance regularization.
        raise ValueError("covariance must have positive determinant")
    constant = y.size * np.log(2.0 * np.pi) if include_constant else 0.0
    return float(0.5 * (residual @ solved + logdet + constant))
