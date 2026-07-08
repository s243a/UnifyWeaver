#!/usr/bin/env python3
"""Calibration helpers for Product-Kalman / correlated-PoE prototypes.

This layer estimates the covariance blocks used by `product_kalman.py` from a
calibration split with known target states in the chosen evidence coordinate:

    prior_error       = target_state - prior_mean
    measurement_error = measurement - H @ target_state

The fitted blocks are one regularized joint covariance sliced as:

    [[P, C],
     [C.T, R]]

Use calibration data that is node-disjoint from training data and from the final
evaluation split. Inputs are row matrices: use shape `(n, 1)` for scalar states
or measurements. These helpers do not decide the split; they make leakage harder
to miss once the split IDs are known.
"""

from dataclasses import dataclass

import numpy as np

try:
    from .product_kalman import fit_residual_covariance, gaussian_condition_update
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman import fit_residual_covariance, gaussian_condition_update


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


def _check_finite(name, arr):
    """Validate a pre-converted ndarray is finite."""
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")


def _covariance_atol(arr):
    return 1e-10 * max(1.0, float(np.linalg.norm(arr, ord=np.inf)))


def _check_square_covariance(name, arr):
    """Validate a pre-converted ndarray has covariance-matrix shape and symmetry."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    _check_finite(name, arr)
    if not np.allclose(arr, arr.T, atol=_covariance_atol(arr), rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")


def _check_positive_semidefinite(name, arr):
    sym = 0.5 * (arr + arr.T)
    floor = -1e-9 * max(1.0, float(np.linalg.norm(sym, ord=np.inf)))
    min_eig = float(np.linalg.eigvalsh(sym).min())
    if min_eig < floor:
        raise ValueError(f"{name} must be positive semidefinite; min eigenvalue {min_eig:.3g}")


@dataclass(frozen=True)
class ProductKalmanCalibration:
    """Covariance blocks fitted on a calibration split.

    `__post_init__` validates the public constructor and then replaces array
    fields with read-only copies; `frozen=True` alone would not protect numpy
    arrays from in-place mutation.
    """

    state_covariance: np.ndarray
    observation_covariance: np.ndarray
    cross_covariance: np.ndarray
    H: np.ndarray
    n_samples: int
    ddof: int
    shrinkage: float
    shrinkage_target: str

    def __post_init__(self):
        state_cov = np.asarray(self.state_covariance, dtype=float)
        obs_cov = np.asarray(self.observation_covariance, dtype=float)
        cross_cov = np.asarray(self.cross_covariance, dtype=float)
        H = np.asarray(self.H, dtype=float)

        _check_square_covariance("state_covariance", state_cov)
        _check_square_covariance("observation_covariance", obs_cov)
        _check_positive_semidefinite("state_covariance", state_cov)
        _check_positive_semidefinite("observation_covariance", obs_cov)
        state_dim = state_cov.shape[0]
        obs_dim = obs_cov.shape[0]
        if cross_cov.shape != (state_dim, obs_dim):
            raise ValueError(f"cross_covariance shape {cross_cov.shape} must be ({state_dim}, {obs_dim})")
        if H.shape != (obs_dim, state_dim):
            raise ValueError(f"H shape {H.shape} must be ({obs_dim}, {state_dim})")
        _check_finite("cross_covariance", cross_cov)
        _check_finite("H", H)
        joint_cov = np.block([[state_cov, cross_cov], [cross_cov.T, obs_cov]])
        _check_positive_semidefinite("joint_covariance", joint_cov)

        n_samples = int(self.n_samples)
        ddof = int(self.ddof)
        shrinkage = float(self.shrinkage)
        shrinkage_target = str(self.shrinkage_target)
        # This is a conservative policy guard on recorded fit metadata: jitter may make
        # smaller fits invertible, but then rank is coming mostly from regularization.
        if n_samples <= state_dim + obs_dim:
            raise ValueError("n_samples must exceed state_dim + observation_dim for a full-rank calibration fit")
        if ddof < 0 or n_samples <= ddof:
            raise ValueError("n_samples must exceed nonnegative ddof")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if shrinkage_target not in ("diagonal", "scaled_identity"):
            raise ValueError("shrinkage_target must be 'diagonal' or 'scaled_identity'")

        for name, value in (
            ("state_covariance", state_cov),
            ("observation_covariance", obs_cov),
            ("cross_covariance", cross_cov),
            ("H", H),
        ):
            object.__setattr__(self, name, _readonly_array(value))
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "ddof", ddof)
        object.__setattr__(self, "shrinkage", shrinkage)
        object.__setattr__(self, "shrinkage_target", shrinkage_target)

    @property
    def state_dim(self):
        return int(self.state_covariance.shape[0])

    @property
    def observation_dim(self):
        return int(self.observation_covariance.shape[0])

    @property
    def joint_covariance(self):
        """Reconstruct the regularized joint covariance block matrix."""
        return _readonly_array(np.block([
            [self.state_covariance, self.cross_covariance],
            [self.cross_covariance.T, self.observation_covariance],
        ]))


@dataclass(frozen=True)
class ProductKalmanBatchUpdate:
    """Batch application of one Product-Kalman calibration.

    The posterior covariance, gain, and innovation covariance are shared across
    rows because the fitted linear-Gaussian calibration is shared; only `mean`
    and `innovation` vary by row.
    """

    mean: np.ndarray
    covariance: np.ndarray
    gain: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        gain = np.asarray(self.gain, dtype=float)
        innovation = np.asarray(self.innovation, dtype=float)
        innovation_covariance = np.asarray(self.innovation_covariance, dtype=float)
        if mean.ndim != 2 or innovation.ndim != 2:
            raise ValueError("mean and innovation must be 2-D row matrices")
        _check_square_covariance("covariance", covariance)
        _check_square_covariance("innovation_covariance", innovation_covariance)
        if gain.shape != (mean.shape[1], innovation.shape[1]):
            raise ValueError(f"gain shape {gain.shape} must be ({mean.shape[1]}, {innovation.shape[1]})")
        if covariance.shape != (mean.shape[1], mean.shape[1]):
            raise ValueError("covariance dimension must match mean columns")
        if innovation_covariance.shape != (innovation.shape[1], innovation.shape[1]):
            raise ValueError("innovation_covariance dimension must match innovation columns")
        if mean.shape[0] != innovation.shape[0]:
            raise ValueError("mean and innovation must have the same row count")
        for name, value in (
            ("mean", mean),
            ("covariance", covariance),
            ("gain", gain),
            ("innovation", innovation),
            ("innovation_covariance", innovation_covariance),
        ):
            _check_finite(name, value)
            object.__setattr__(self, name, _readonly_array(value))


def _as_2d(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D row matrix; use shape (n, 1) for scalar channels")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    _check_finite(name, arr)
    return arr


def _observation_matrix(H, state_dim, observation_dim):
    if H is None:
        if state_dim != observation_dim:
            raise ValueError("H is required when target_state and measurement dimensions differ")
        return np.eye(state_dim)
    mat = np.asarray(H, dtype=float)
    if mat.shape != (observation_dim, state_dim):
        raise ValueError(f"H shape {mat.shape} must be ({observation_dim}, {state_dim})")
    _check_finite("H", mat)
    return mat


def _duplicates(values):
    seen = set()
    dup = []
    for value in values:
        try:
            already_seen = value in seen
            if already_seen and value not in dup:
                dup.append(value)
            seen.add(value)
        except TypeError as exc:
            raise ValueError("split IDs must be hashable") from exc
    return dup


def assert_disjoint_ids(left_ids, right_ids, left_name="calibration", right_name="evaluation"):
    """Raise if hashable split-ID collections overlap or contain duplicate IDs."""
    left_values = list(left_ids)
    right_values = list(right_ids)
    for name, values in ((left_name, left_values), (right_name, right_values)):
        dup = _duplicates(values)
        if dup:
            preview = ", ".join(repr(x) for x in dup[:5])
            suffix = "" if len(dup) <= 5 else f", ... (+{len(dup) - 5} more)"
            raise ValueError(f"{name} IDs contain duplicates: {preview}{suffix}")
    left = set(left_values)
    right = set(right_values)
    overlap = sorted(left & right, key=lambda x: repr(x))
    if overlap:
        preview = ", ".join(repr(x) for x in overlap[:5])
        suffix = "" if len(overlap) <= 5 else f", ... (+{len(overlap) - 5} more)"
        raise ValueError(f"{left_name} and {right_name} IDs overlap: {preview}{suffix}")


def fit_product_kalman_calibration(prior_mean, measurement, target_state, H=None, shrinkage=0.0, jitter=1e-9,
                                   ddof=1, shrinkage_target="diagonal"):
    """Fit Product-Kalman covariance blocks from calibration residuals.

    `prior_mean` and `target_state` are row matrices in the state/evidence
    coordinate. `measurement` rows are observations of `H @ target_state` in the
    same linked coordinate system used by the update. `H` is shape-validated, but
    its scientific meaning is caller-supplied. `C` is estimated jointly with `P`
    and `R`, so correlated prior/measurement errors are retained instead of
    silently dropped.
    """
    prior = _as_2d("prior_mean", prior_mean)
    target = _as_2d("target_state", target_state)
    obs = _as_2d("measurement", measurement)
    if prior.shape != target.shape:
        raise ValueError(f"prior_mean shape {prior.shape} must match target_state shape {target.shape}")
    if obs.shape[0] != target.shape[0]:
        raise ValueError("measurement rows must match target_state rows")
    Hm = _observation_matrix(H, target.shape[1], obs.shape[1])
    state_dim = target.shape[1]
    obs_dim = obs.shape[1]
    total_dim = state_dim + obs_dim
    # Policy guard: require enough rows that jitter stabilizes the empirical covariance
    # rather than supplying all of its missing rank.
    if target.shape[0] <= total_dim:
        raise ValueError("calibration rows must exceed state_dim + observation_dim")

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
    if joint_cov.shape != (total_dim, total_dim):
        raise ValueError(f"joint covariance shape {joint_cov.shape} must be ({total_dim}, {total_dim})")
    P = joint_cov[:state_dim, :state_dim]
    R = joint_cov[state_dim:total_dim, state_dim:total_dim]
    C = joint_cov[:state_dim, state_dim:total_dim]
    return ProductKalmanCalibration(
        state_covariance=P,
        observation_covariance=R,
        cross_covariance=C,
        H=Hm,
        n_samples=int(target.shape[0]),
        ddof=int(ddof),
        shrinkage=float(shrinkage),
        shrinkage_target=str(shrinkage_target),
    )


def apply_product_kalman_calibration(calibration, prior_mean, measurement, jitter=1e-9):
    """Apply one fitted calibration to a batch of prior/measurement rows.

    This uses one zero-mean template update because the Kalman gain, posterior
    covariance, and innovation covariance are independent of row means in the
    linear-Gaussian model. The batch mean is equivalent to applying
    `gaussian_condition_update` independently to each row with the same fitted
    covariance blocks.
    """
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
