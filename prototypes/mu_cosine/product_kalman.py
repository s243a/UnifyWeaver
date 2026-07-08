#!/usr/bin/env python3
"""Gaussian conditioning core for Product-Kalman / correlated-PoE prototypes.

The functions here operate in an already chosen evidence coordinate, such as
`log(mu_lower)`, `logit(mu_direct)`, or another finite product-space link. They do
not construct product proxies themselves; use `product_space.py` for that layer.

This module deliberately exposes cross-covariance between the prior state error and
measurement noise. When prior and measurement channels share evidence, treating that
cross term as zero is an assumption, not a default fact.
"""

from dataclasses import dataclass

import numpy as np


__all__ = [
    "GaussianUpdate",
    "fit_error_covariance",
    "fit_residual_covariance",
    "gaussian_condition_update",
    "gaussian_nll",
    "regularize_covariance",
    "scalar_product_kalman_update",
]


@dataclass(frozen=True)
class GaussianUpdate:
    """Result of one Gaussian conditioning / Kalman-style update."""

    mean: np.ndarray
    covariance: np.ndarray
    gain: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    cross_covariance: np.ndarray


def _as_vector(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector or scalar")
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
    covariance model.
    """
    if jitter < 0:
        raise ValueError("jitter must be nonnegative")
    cov = _as_matrix(name, covariance)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square")
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    min_eig = float(eig[0])
    floor = max(float(jitter), 1e-12)
    if min_eig < -100.0 * floor:
        raise ValueError(f"{name} is not positive semidefinite; min eigenvalue={min_eig:g}")
    lift = max(float(jitter) - min_eig, 0.0)
    if lift:
        cov = cov + np.eye(cov.shape[0]) * lift
    return cov


def fit_residual_covariance(residuals, shrinkage=0.0, jitter=1e-9):
    """Estimate residual covariance from rows of calibration residuals.

    `shrinkage=1` keeps only the diagonal; `shrinkage=0` keeps the empirical full
    covariance. Use residuals from an appropriate calibration split, not the data
    used for the claimed evaluation.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    r = np.asarray(residuals, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if r.ndim != 2:
        raise ValueError("residuals must be a 1-D or 2-D array")
    if r.shape[0] < 2:
        raise ValueError("at least two residual rows are required")
    if not np.isfinite(r).all():
        raise ValueError("residuals must be finite")
    centered = r - r.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / float(r.shape[0] - 1)
    if shrinkage:
        diag = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
    return regularize_covariance(cov, jitter=jitter, name="residual covariance")


def fit_error_covariance(predicted, observed, shrinkage=0.0, jitter=1e-9):
    """Estimate covariance of `observed - predicted` calibration errors."""
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"predicted shape {pred.shape} must match observed shape {obs.shape}")
    return fit_residual_covariance(obs - pred, shrinkage=shrinkage, jitter=jitter)


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
    Hm = np.eye(n) if H is None else _as_matrix("H", H, k, n)
    C = np.zeros((n, k)) if cross_covariance is None else _as_matrix("cross_covariance", cross_covariance, n, k)

    innovation = y - Hm @ m
    innovation_cov = Hm @ P @ Hm.T + Hm @ C + C.T @ Hm.T + R
    innovation_cov = regularize_covariance(innovation_cov, jitter=jitter, name="innovation covariance")
    state_innovation_cov = P @ Hm.T + C
    gain = np.linalg.solve(innovation_cov.T, state_innovation_cov.T).T
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


def scalar_product_kalman_update(ell_prior, ell_measurement, prior_var, measurement_var, cross_covariance=0.0,
                                 jitter=1e-9):
    """Scalar Product-Kalman update in one log/link evidence coordinate."""
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
    """Negative log likelihood of one Gaussian observation."""
    y = _as_vector("observed", observed)
    m = _as_vector("mean", mean)
    if y.shape != m.shape:
        raise ValueError(f"observed shape {y.shape} must match mean shape {m.shape}")
    cov = regularize_covariance(_square_covariance("covariance", covariance, y.size), jitter=jitter, name="covariance")
    residual = y - m
    solved = np.linalg.solve(cov, residual)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance must have positive determinant")
    constant = y.size * np.log(2.0 * np.pi) if include_constant else 0.0
    return float(0.5 * (residual @ solved + logdet + constant))
