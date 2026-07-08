#!/usr/bin/env python3
"""Calibration helpers for Product-Kalman / correlated-PoE prototypes.

This layer estimates the covariance blocks used by `product_kalman.py` from a
calibration split with known target states in the chosen evidence coordinate:

    prior_error       = target_state - prior_mean
    measurement_error = measurement - H @ target_state

The fitted blocks are:

    P = Cov(prior_error)
    R = Cov(measurement_error)
    C = Cov(prior_error, measurement_error)

Use calibration data that is node-disjoint from training data and from the final
evaluation split. These helpers do not decide the split; they make leakage harder
to miss once the split IDs are known.
"""

from dataclasses import dataclass

import numpy as np

try:
    from .product_kalman import fit_residual_covariance, gaussian_condition_update, regularize_covariance
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman import fit_residual_covariance, gaussian_condition_update, regularize_covariance


__all__ = [
    "ProductKalmanBatchUpdate",
    "ProductKalmanCalibration",
    "assert_disjoint_ids",
    "apply_product_kalman_calibration",
    "fit_product_kalman_calibration",
]


def _readonly_array(value):
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ProductKalmanCalibration:
    """Covariance blocks fitted on a calibration split."""

    state_covariance: np.ndarray
    observation_covariance: np.ndarray
    cross_covariance: np.ndarray
    H: np.ndarray
    n_samples: int
    ddof: int
    shrinkage: float
    shrinkage_target: str

    def __post_init__(self):
        for name in ("state_covariance", "observation_covariance", "cross_covariance", "H"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))

    @property
    def state_dim(self):
        return int(self.state_covariance.shape[0])

    @property
    def observation_dim(self):
        return int(self.observation_covariance.shape[0])


@dataclass(frozen=True)
class ProductKalmanBatchUpdate:
    """Batch application of one Product-Kalman calibration."""

    mean: np.ndarray
    covariance: np.ndarray
    gain: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray

    def __post_init__(self):
        for name in ("mean", "covariance", "gain", "innovation", "innovation_covariance"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))


def _as_2d(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1-D row or 2-D array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _observation_matrix(H, state_dim, observation_dim):
    if H is None:
        if state_dim != observation_dim:
            raise ValueError("H is required when target_state and measurement dimensions differ")
        return np.eye(state_dim)
    mat = np.asarray(H, dtype=float)
    if mat.shape != (observation_dim, state_dim):
        raise ValueError(f"H shape {mat.shape} must be ({observation_dim}, {state_dim})")
    if not np.isfinite(mat).all():
        raise ValueError("H must be finite")
    return mat


def assert_disjoint_ids(left_ids, right_ids, left_name="calibration", right_name="evaluation"):
    """Raise if two split-ID collections overlap."""
    left = set(left_ids)
    right = set(right_ids)
    overlap = sorted(left & right, key=lambda x: repr(x))
    if overlap:
        preview = ", ".join(repr(x) for x in overlap[:5])
        suffix = "" if len(overlap) <= 5 else f", ... (+{len(overlap) - 5} more)"
        raise ValueError(f"{left_name} and {right_name} IDs overlap: {preview}{suffix}")


def fit_product_kalman_calibration(prior_mean, measurement, target_state, H=None, shrinkage=0.0, jitter=1e-9,
                                   ddof=1, shrinkage_target="diagonal"):
    """Fit Product-Kalman covariance blocks from calibration residuals.

    `prior_mean` and `target_state` are rows in the state/evidence coordinate.
    `measurement` rows are observations of `H @ target_state` in the same linked
    coordinate system used by the update. `C` is estimated jointly with `P` and
    `R`, so correlated prior/measurement errors are retained instead of silently
    dropped.
    """
    prior = _as_2d("prior_mean", prior_mean)
    target = _as_2d("target_state", target_state)
    obs = _as_2d("measurement", measurement)
    if prior.shape != target.shape:
        raise ValueError(f"prior_mean shape {prior.shape} must match target_state shape {target.shape}")
    if obs.shape[0] != target.shape[0]:
        raise ValueError("measurement rows must match target_state rows")
    Hm = _observation_matrix(H, target.shape[1], obs.shape[1])

    prior_error = target - prior
    measurement_error = obs - target @ Hm.T
    joint_error = np.concatenate([prior_error, measurement_error], axis=1)
    joint_cov = fit_residual_covariance(
        joint_error,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )
    state_dim = target.shape[1]
    obs_dim = obs.shape[1]
    P = joint_cov[:state_dim, :state_dim]
    R = joint_cov[state_dim:, state_dim:]
    C = joint_cov[:state_dim, state_dim:]
    return ProductKalmanCalibration(
        state_covariance=regularize_covariance(P, jitter=jitter, name="state covariance"),
        observation_covariance=regularize_covariance(R, jitter=jitter, name="observation covariance"),
        cross_covariance=C,
        H=Hm,
        n_samples=int(target.shape[0]),
        ddof=int(ddof),
        shrinkage=float(shrinkage),
        shrinkage_target=str(shrinkage_target),
    )


def apply_product_kalman_calibration(calibration, prior_mean, measurement, jitter=1e-9):
    """Apply one fitted calibration to a batch of prior/measurement rows."""
    if not isinstance(calibration, ProductKalmanCalibration):
        raise ValueError("calibration must be a ProductKalmanCalibration")
    prior = _as_2d("prior_mean", prior_mean)
    obs = _as_2d("measurement", measurement)
    if prior.shape[1] != calibration.state_dim:
        raise ValueError(f"prior_mean has {prior.shape[1]} columns, expected {calibration.state_dim}")
    if obs.shape != (prior.shape[0], calibration.observation_dim):
        raise ValueError(f"measurement shape {obs.shape} must be ({prior.shape[0]}, {calibration.observation_dim})")

    template = gaussian_condition_update(
        np.zeros(calibration.state_dim),
        calibration.state_covariance,
        np.zeros(calibration.observation_dim),
        calibration.observation_covariance,
        H=calibration.H,
        cross_covariance=calibration.cross_covariance,
        jitter=jitter,
    )
    innovation = obs - prior @ calibration.H.T
    mean = prior + innovation @ template.gain.T
    return ProductKalmanBatchUpdate(
        mean=mean,
        covariance=template.covariance,
        gain=template.gain,
        innovation=innovation,
        innovation_covariance=template.innovation_covariance,
    )
