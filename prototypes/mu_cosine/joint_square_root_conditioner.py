#!/usr/bin/env python3
"""Joint square-root / Householder-QR Gaussian conditioner.

This module is the square-root-information implementation of the same static
correlated Gaussian update used by ``product_kalman.gaussian_condition_update``.
It maintains a factor ``U`` of the precision matrix,

    P^-1 = U.T @ U,

and incorporates a measurement block by triangularising the augmented least-
squares array with Householder reflections.  It is an implementation change,
not a new statistical fusion rule.

For prior error ``e = truth - prior`` and measurement noise
``v = observation - H @ truth`` with ``Cov(e, v) = C``, first condition the
measurement noise on ``e``:

    Rc = R - C.T @ P^-1 @ C
    J  = H + C.T @ P^-1
    innovation | e ~ N(J @ e, Rc)

Whitening ``Rc`` produces rows ``A`` and ``b``.  Stacking those below the
previous information root and applying Householder QR gives the posterior
information root and right-hand side:

    [ U_prior   z_prior ]       [ U_post   z_post ]
    [ A         b       ]  ->   [ 0        residual ]

Householder transformations are reflections.  Givens transformations are the
closely related plane rotations often used for streaming one row at a time.
"""

from dataclasses import dataclass

import numpy as np

from product_kalman import regularize_covariance


__all__ = [
    "InformationRootUpdate",
    "JointSquareRootUpdate",
    "condition_correlated_gaussian_qr",
    "conditional_measurement_model",
    "householder_information_update",
    "precision_root_from_covariance",
    "whiten_measurement_block",
]


def _readonly(value):
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


def _vector(name, value, length=None):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if length is not None and len(arr) != length:
        raise ValueError(f"{name} has length {len(arr)}, expected {length}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _matrix(name, value, rows=None, cols=None):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {rows}")
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"{name} has {arr.shape[1]} columns, expected {cols}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class InformationRootUpdate:
    """Result of one Householder information-root update.

    ``information_rhs`` is the square-root RHS ``z``.  The canonical
    information vector is ``eta = precision_root.T @ z``.
    """

    precision_root: np.ndarray
    information_rhs: np.ndarray
    solution: np.ndarray
    trailing_residual: np.ndarray
    residual_sum_squares: float

    def __post_init__(self):
        for name in ("precision_root", "information_rhs", "solution", "trailing_residual"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class JointSquareRootUpdate:
    """Correlated Gaussian conditioning result in square-root-information form."""

    mean: np.ndarray
    covariance: np.ndarray
    precision_root: np.ndarray
    information_rhs: np.ndarray
    innovation: np.ndarray
    effective_observation_matrix: np.ndarray
    conditional_observation_covariance: np.ndarray
    whitened_observation_matrix: np.ndarray
    whitened_innovation: np.ndarray
    residual_sum_squares: float

    def __post_init__(self):
        for name in (
            "mean",
            "covariance",
            "precision_root",
            "information_rhs",
            "innovation",
            "effective_observation_matrix",
            "conditional_observation_covariance",
            "whitened_observation_matrix",
            "whitened_innovation",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def _normalise_root_signs(root, rhs):
    """Choose the conventional non-negative diagonal without changing R.T R."""
    root = np.array(root, dtype=float, copy=True)
    rhs = np.array(rhs, dtype=float, copy=True)
    signs = np.where(np.diag(root) < 0.0, -1.0, 1.0)
    root *= signs[:, None]
    rhs *= signs
    return root, rhs


def _scaled_norm(value):
    """Euclidean norm without overflow/underflow from squaring raw entries."""
    value = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0.0:
        return 0.0
    return scale * float(np.linalg.norm(value / scale))


def householder_information_update(prior_precision_root, prior_information_rhs,
                                   measurement_matrix, measurement_rhs):
    """Triangularise a prior information root plus whitened measurement rows.

    ``prior_precision_root`` need only satisfy ``Lambda = U.T @ U``; the
    first call may pass a non-triangular inverse covariance factor.  The
    returned root is upper triangular.  Reusing that returned root and RHS in
    a later call is the streaming block update.
    """
    U = _matrix("prior_precision_root", prior_precision_root)
    if U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError("prior_precision_root must be nonempty and square")
    n = U.shape[0]
    z = _vector("prior_information_rhs", prior_information_rhs, n)
    A = _matrix("measurement_matrix", measurement_matrix, cols=n)
    b = _vector("measurement_rhs", measurement_rhs, A.shape[0])

    work = np.vstack([
        np.column_stack([U, z]),
        np.column_stack([A, b]),
    ]).astype(float, copy=False)
    pre_array = work[:, :n]
    entry_scale = float(np.max(np.abs(pre_array), initial=0.0))
    if entry_scale == 0.0:
        raise np.linalg.LinAlgError("information pre-array is rank deficient")
    if entry_scale < np.finfo(float).tiny:
        raise np.linalg.LinAlgError(
            "information pre-array scale is subnormal; rescale before QR"
        )
    relative_frobenius = float(np.linalg.norm(pre_array / entry_scale))
    relative_tol = (
        np.finfo(float).eps * max(pre_array.shape) * relative_frobenius
    )

    # Apply each reflector directly to the remaining coefficient/RHS array;
    # Q is never formed.  This is the standard numerically stable QR update.
    for col in range(n):
        x = work[col:, col].copy()
        norm_x = _scaled_norm(x)
        if not np.isfinite(norm_x) or norm_x / entry_scale <= relative_tol:
            raise np.linalg.LinAlgError("information pre-array is rank deficient")
        first_sign = 1.0 if x[0] >= 0.0 else -1.0
        alpha = -first_sign * norm_x
        x[0] -= alpha
        norm_v = _scaled_norm(x)
        if not np.isfinite(norm_v) or norm_v / entry_scale <= relative_tol:
            raise np.linalg.LinAlgError("cannot construct a stable Householder reflector")
        v = x / norm_v
        tail = work[col:, col:]
        tail -= 2.0 * np.outer(v, v @ tail)
        work[col:, col:] = tail
        work[col + 1:, col] = 0.0

    root = np.triu(work[:n, :n])
    rhs = work[:n, n]
    root, rhs = _normalise_root_signs(root, rhs)
    diagonal = np.abs(np.diag(root))
    if np.any(~np.isfinite(diagonal)) or np.any(
        diagonal / entry_scale <= relative_tol
    ):
        raise np.linalg.LinAlgError("posterior information root is singular")
    solution = np.linalg.solve(root, rhs)
    residual = work[n:, n]
    return InformationRootUpdate(
        precision_root=root,
        information_rhs=rhs,
        solution=solution,
        trailing_residual=residual,
        residual_sum_squares=float(residual @ residual),
    )


def precision_root_from_covariance(covariance, jitter=1e-9):
    """Return U with ``covariance^-1 = U.T @ U`` without forming the inverse."""
    cov = regularize_covariance(covariance, jitter=jitter, name="prior covariance")
    chol = np.linalg.cholesky(cov)
    inverse_factor = np.linalg.solve(chol, np.eye(chol.shape[0]))
    empty = np.zeros((0, chol.shape[0]))
    return np.array(householder_information_update(
        inverse_factor,
        np.zeros(chol.shape[0]),
        empty,
        np.zeros(0),
    ).precision_root)


def conditional_measurement_model(prior_precision_root, H, observation_covariance,
                                  cross_covariance, jitter=1e-9):
    """Decorrelate measurement noise from the prior error in information form.

    Returns ``(J, Rc)`` such that innovation ``r | e ~ N(J e, Rc)``.
    Block-sequential updates are exact only when ``Rc`` (not raw ``R``) is
    block diagonal for the proposed measurement partition.
    """
    U = _matrix("prior_precision_root", prior_precision_root)
    if U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError("prior_precision_root must be nonempty and square")
    n = U.shape[0]
    Hm = _matrix("H", H, cols=n)
    m = Hm.shape[0]
    R = _matrix("observation_covariance", observation_covariance, m, m)
    C = _matrix("cross_covariance", cross_covariance, n, m)

    UC = U @ C
    effective_H = Hm + UC.T @ U
    conditional_R = R - UC.T @ UC
    conditional_R = regularize_covariance(
        conditional_R,
        jitter=jitter,
        name="conditional observation covariance R - C.T P^-1 C",
    )
    return effective_H, conditional_R


def whiten_measurement_block(measurement_matrix, measurement_covariance, measurement_rhs,
                             jitter=1e-9, *, covariance_is_regularized=False):
    """Whiten one independent likelihood block with a Cholesky solve.

    Set ``covariance_is_regularized`` only when the covariance came directly
    from :func:`conditional_measurement_model`; this avoids applying the
    jitter/PSD policy twice in the composed conditioner.
    """
    J = _matrix("measurement_matrix", measurement_matrix)
    m, _ = J.shape
    Rc = _matrix("measurement_covariance", measurement_covariance, m, m)
    if not covariance_is_regularized:
        Rc = regularize_covariance(
            Rc,
            jitter=jitter,
            name="measurement covariance",
        )
    r = _vector("measurement_rhs", measurement_rhs, m)
    chol = np.linalg.cholesky(Rc)
    return np.linalg.solve(chol, J), np.linalg.solve(chol, r), Rc


def condition_correlated_gaussian_qr(mean, prior_precision_root, observation,
                                     observation_covariance, H, cross_covariance,
                                     jitter=1e-9):
    """Condition a static Gaussian using the joint square-root/QR algorithm.

    ``cross_covariance`` uses the same sign convention as
    ``product_kalman.gaussian_condition_update``: ``Cov(truth-mean, v)`` for
    ``observation = H @ truth + v``.
    """
    x = _vector("mean", mean)
    U = _matrix("prior_precision_root", prior_precision_root, len(x), len(x))
    y = _vector("observation", observation)
    Hm = _matrix("H", H, len(y), len(x))
    innovation = y - Hm @ x

    effective_H, conditional_R = conditional_measurement_model(
        U,
        Hm,
        observation_covariance,
        cross_covariance,
        jitter=jitter,
    )
    A, b, conditional_R = whiten_measurement_block(
        effective_H,
        conditional_R,
        innovation,
        jitter=jitter,
        covariance_is_regularized=True,
    )
    info = householder_information_update(U, np.zeros(len(x)), A, b)
    posterior_mean = x + np.asarray(info.solution)
    root_inv = np.linalg.solve(np.asarray(info.precision_root), np.eye(len(x)))
    posterior_cov = root_inv @ root_inv.T
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)

    return JointSquareRootUpdate(
        mean=posterior_mean,
        covariance=posterior_cov,
        precision_root=info.precision_root,
        information_rhs=info.information_rhs,
        innovation=innovation,
        effective_observation_matrix=effective_H,
        conditional_observation_covariance=conditional_R,
        whitened_observation_matrix=A,
        whitened_innovation=b,
        residual_sum_squares=info.residual_sum_squares,
    )
