#!/usr/bin/env python3
"""Batched PyTorch backend for the joint square-root/QR conditioner.

The statistical model is identical to ``joint_square_root_conditioner.py``.
This module changes the linear algebra implementation: it stores the compact
Householder representation returned by :func:`torch.geqrf` and applies
``Q.T`` to one or many right-hand sides with :func:`torch.ormqr`.  It never
forms ``Q`` merely to obtain ``R`` or transform an observation.

For a fixed ``(U, H, R, C)`` design, use
:class:`CompiledCorrelatedConditionerTorch`.  One factorization can then
condition a whole observation batch as columns of one RHS matrix.  For truly
sequential, conditionally independent measurement blocks, use
:class:`SquareRootInformationStateTorch`; each update threads ``U_post`` (and
the information RHS) into the next QR factorization.

For a fixed design that only needs posterior means/covariance, the matched
throughput baseline is :class:`CompiledDenseGainConditionerTorch`: it caches
the correlated gain and is expected to be faster.  Its gain cannot be reused
when a later block changes the posterior, which is the root-threading QR case.

All functions require real floating PyTorch tensors and preserve their dtype
and device.  The implementation is inference-oriented.  In particular,
``geqrf``/``ormqr`` backend and autograd coverage is narrower than
``torch.linalg.qr`` on some PyTorch/device combinations.  One-time covariance
and coefficient preparation validates finiteness; repeated RHS hot paths avoid
device-synchronising finiteness checks, so NaN/Inf observations propagate and
must be gated by the caller when that is not acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


__all__ = [
    "CompiledCorrelatedConditionerTorch",
    "CompiledDenseGainConditionerTorch",
    "CompiledInformationQRTorch",
    "CovarianceLoadingDiagnosticsTorch",
    "DenseGaussianUpdateTorch",
    "InformationRootUpdateTorch",
    "JointSquareRootUpdateTorch",
    "NoiseWhitenerTorch",
    "SquareRootInformationStateTorch",
    "compile_information_update_torch",
    "condition_correlated_gaussian_qr_torch",
    "conditional_measurement_model_torch",
    "conditional_measurement_whitener_torch",
    "householder_information_update_torch",
    "precision_root_from_covariance_torch",
    "prepare_noise_whitener_torch",
    "regularize_covariance_torch",
]


_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype float32 or float64")
    # Deliberately avoid a per-call ``isfinite(...).item()``: that would force
    # a device-to-host synchronisation in the hot batched-conditioning path.
    # LAPACK/cuSOLVER will propagate non-finite inputs to the result.
    return value


def _same_backend(reference: torch.Tensor, **values: torch.Tensor) -> None:
    for name, value in values.items():
        if value.device != reference.device:
            raise ValueError(
                f"{name} is on {value.device}, expected {reference.device}"
            )
        if value.dtype != reference.dtype:
            raise ValueError(
                f"{name} has dtype {value.dtype}, expected {reference.dtype}"
            )


def _matrix(
    name: str,
    value: torch.Tensor,
    *,
    rows: int | None = None,
    cols: int | None = None,
) -> torch.Tensor:
    value = _tensor(name, value)
    if value.ndim < 2:
        raise ValueError(f"{name} must be a matrix or a batch of matrices")
    if rows is not None and value.shape[-2] != rows:
        raise ValueError(f"{name} has {value.shape[-2]} rows, expected {rows}")
    if cols is not None and value.shape[-1] != cols:
        raise ValueError(f"{name} has {value.shape[-1]} columns, expected {cols}")
    return value


def _vector(name: str, value: torch.Tensor, *, length: int) -> torch.Tensor:
    value = _tensor(name, value)
    if value.ndim < 1 or value.shape[-1] != length:
        raise ValueError(f"{name} must end in a vector of length {length}")
    return value


def _broadcast_matrices(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    batch_shape = torch.broadcast_shapes(*(v.shape[:-2] for v in values))
    return tuple(v.expand(batch_shape + v.shape[-2:]) for v in values)


def _normalised_root_from_packed(
    packed: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_root = torch.triu(packed[..., :n, :n])
    diagonal = torch.diagonal(raw_root, dim1=-2, dim2=-1)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return raw_root * signs.unsqueeze(-1), signs


def _pre_array_scale(
    pre_array: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve the prior Frobenius-scale rank sensitivity without its unit
    # clamp. Return ``(a, ||G/a||_F)`` instead of materialising
    # ``a * ||G/a||_F``: the latter can overflow even when every entry and QR
    # diagonal is representable. The pair remains homogeneous under a
    # *global* scalar rescaling when used by ``_check_root_rank``.
    entry_scale = torch.amax(torch.abs(pre_array), dim=(-2, -1))
    nonfinite = ~torch.isfinite(entry_scale)
    if bool(torch.any(nonfinite).item()):
        raise ValueError(
            "information pre-array must be finite"
            + _batch_failure_detail(nonfinite)
        )
    zero = entry_scale == 0
    if bool(torch.any(zero).item()):
        raise torch.linalg.LinAlgError(
            "information pre-array is rank deficient (zero scale)"
            + _batch_failure_detail(zero)
        )
    subnormal = entry_scale < torch.finfo(pre_array.dtype).tiny
    if bool(torch.any(subnormal).item()):
        raise torch.linalg.LinAlgError(
            "information pre-array scale is subnormal; rescale before QR"
            + _batch_failure_detail(subnormal)
        )
    normalised = pre_array / entry_scale[..., None, None]
    relative_frobenius = torch.linalg.vector_norm(normalised, dim=(-2, -1))
    return entry_scale, relative_frobenius


def _check_root_rank(
    pre_array: torch.Tensor,
    root: torch.Tensor,
    scale: tuple[torch.Tensor, torch.Tensor],
) -> None:
    entry_scale, relative_frobenius = scale
    relative_tolerance = (
        torch.finfo(pre_array.dtype).eps
        * max(pre_array.shape[-2:])
        * relative_frobenius
    )
    smallest = torch.amin(torch.abs(torch.diagonal(root, dim1=-2, dim2=-1)), dim=-1)
    failed = (~torch.isfinite(smallest)) | (
        smallest / entry_scale <= relative_tolerance
    )
    if bool(torch.any(failed).item()):
        if failed.ndim:
            indices = torch.nonzero(failed, as_tuple=False).detach().cpu().tolist()
            detail = f" at batch indices {indices}"
        else:
            detail = " for the unbatched design"
        raise torch.linalg.LinAlgError(f"information pre-array is rank deficient{detail}")


@dataclass(frozen=True)
class InformationRootUpdateTorch:
    """One information-root update, possibly batched or with multiple RHSs.

    ``information_rhs`` is the square-root RHS ``z``.  The canonical
    information vector is ``eta = precision_root.mT @ z``.
    """

    precision_root: torch.Tensor
    information_rhs: torch.Tensor
    solution: torch.Tensor
    trailing_residual: torch.Tensor
    residual_sum_squares: torch.Tensor


@dataclass(frozen=True)
class JointSquareRootUpdateTorch:
    """Result of correlated Gaussian conditioning in square-root form."""

    mean: torch.Tensor
    covariance: torch.Tensor
    precision_root: torch.Tensor
    information_rhs: torch.Tensor
    innovation: torch.Tensor
    effective_observation_matrix: torch.Tensor
    conditional_observation_covariance: torch.Tensor
    conditional_loading_diagnostics: CovarianceLoadingDiagnosticsTorch
    whitened_observation_matrix: torch.Tensor
    whitened_innovation: torch.Tensor
    residual_sum_squares: torch.Tensor


@dataclass(frozen=True)
class DenseGaussianUpdateTorch:
    """Fixed-design dense correlated-gain update, possibly batched over rows."""

    mean: torch.Tensor
    covariance: torch.Tensor
    gain: torch.Tensor
    innovation: torch.Tensor
    innovation_covariance: torch.Tensor
    conditional_loading_diagnostics: CovarianceLoadingDiagnosticsTorch


@dataclass(frozen=True)
class CovarianceLoadingDiagnosticsTorch:
    """Observable contract for scale-relative covariance diagonal loading.

    Tensor fields have the covariance batch shape.  ``minimum_eigenvalue`` is
    measured before loading; ``target_minimum_eigenvalue`` and
    ``diagonal_loading`` state exactly what was added.  A materially indefinite
    input is rejected instead of appearing here as a large silent repair.
    """

    matrix_scale: torch.Tensor
    minimum_eigenvalue: torch.Tensor
    target_minimum_eigenvalue: torch.Tensor
    diagonal_loading: torch.Tensor
    relative_diagonal_loading: torch.Tensor
    relative_symmetry_error: torch.Tensor
    was_loaded: torch.Tensor
    relative_eigenvalue_floor: float
    negative_eigenvalue_tolerance: float
    maximum_relative_loading: float
    source: str


@dataclass(frozen=True)
class NoiseWhitenerTorch:
    """Implicit Cholesky inverse factor for one or a batch of noise models.

    If ``covariance = L L.T``, this object stores ``L`` and applies the
    whitening factor ``L^-1`` exclusively with triangular solves.  It never
    materialises an explicit matrix inverse.  Use :meth:`apply_vectors` for
    ``(..., dimension)`` samples and :meth:`apply_columns` for
    ``(..., dimension, nrhs)`` right-hand-side matrices.
    """

    covariance: torch.Tensor
    cholesky_factor: torch.Tensor
    diagnostics: CovarianceLoadingDiagnosticsTorch

    @property
    def dimension(self) -> int:
        return self.cholesky_factor.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.cholesky_factor.shape[:-2]

    def apply_columns(self, value: torch.Tensor) -> torch.Tensor:
        """Apply ``L^-1`` to ``(..., dimension, nrhs)`` with a solve."""
        value = _matrix("value", value, rows=self.dimension)
        _same_backend(self.cholesky_factor, value=value)
        batch_shape = torch.broadcast_shapes(self.batch_shape, value.shape[:-2])
        factor = self.cholesky_factor.expand(
            batch_shape + self.cholesky_factor.shape[-2:]
        ).contiguous()
        value = value.expand(batch_shape + value.shape[-2:])
        return torch.linalg.solve_triangular(
            factor, value.contiguous(), upper=False
        )

    def apply_vectors(self, value: torch.Tensor) -> torch.Tensor:
        """Apply ``L^-1`` to a vector or an arbitrary vector batch."""
        value = _vector("value", value, length=self.dimension)
        _same_backend(self.cholesky_factor, value=value)
        vector_batch = value.shape[:-1]
        if len(self.batch_shape) == 0:
            count = math.prod(vector_batch) if vector_batch else 1
            columns = value.reshape(count, self.dimension).transpose(0, 1)
            whitened = self.apply_columns(columns)
            return whitened.transpose(0, 1).reshape(
                vector_batch + (self.dimension,)
            )
        batch_shape = torch.broadcast_shapes(self.batch_shape, vector_batch)
        value = value.expand(batch_shape + (self.dimension,))
        return self.apply_columns(value.unsqueeze(-1)).squeeze(-1)

    def materialize_row_root(self) -> torch.Tensor:
        """Materialise ``L^-1`` with a solve for inspection or interop.

        The result ``U = L^-1`` is a lower-triangular *row root* satisfying
        ``covariance^-1 = U.T @ U``; it is not the canonical upper Cholesky
        factor of the precision.  QR-triangularise it if an upper root is
        required.  Hot paths should use :meth:`apply_vectors` or
        :meth:`apply_columns`; both avoid allocating this dense factor.  This
        method still uses a triangular solve and never calls a matrix-inverse
        routine.
        """
        identity = torch.eye(
            self.dimension,
            dtype=self.cholesky_factor.dtype,
            device=self.cholesky_factor.device,
        ).expand(self.batch_shape + (self.dimension, self.dimension))
        return torch.linalg.solve_triangular(
            self.cholesky_factor, identity, upper=False
        )


@dataclass(frozen=True)
class CompiledInformationQRTorch:
    """Compact Householder factorization of ``stack([U_prior, A])``.

    ``packed`` and ``tau`` are the LAPACK/cuSOLVER-style reflector storage from
    ``geqrf``.  ``apply_columns`` is the most general API: its final dimension
    is the number of right-hand sides.  ``apply_vectors`` treats leading
    dimensions as independent observations; for an unbatched design it folds
    all of them into RHS columns so one ``ormqr`` call handles the batch.
    """

    packed: torch.Tensor
    tau: torch.Tensor
    precision_root: torch.Tensor
    row_signs: torch.Tensor
    state_dim: int
    measurement_dim: int

    @property
    def batch_shape(self) -> torch.Size:
        return self.packed.shape[:-2]

    def apply_columns(
        self,
        prior_information_rhs: torch.Tensor,
        measurement_rhs: torch.Tensor,
    ) -> InformationRootUpdateTorch:
        """Apply the compiled reflectors to ``(..., rows, nrhs)`` tensors."""
        n, m = self.state_dim, self.measurement_dim
        z = _matrix("prior_information_rhs", prior_information_rhs, rows=n)
        b = _matrix("measurement_rhs", measurement_rhs, rows=m)
        _same_backend(self.packed, prior_information_rhs=z, measurement_rhs=b)
        if z.shape[-1] != b.shape[-1]:
            raise ValueError("prior and measurement RHS counts must match")

        batch_shape = torch.broadcast_shapes(
            self.batch_shape, z.shape[:-2], b.shape[:-2]
        )
        packed = self.packed.expand(batch_shape + self.packed.shape[-2:]).contiguous()
        tau = self.tau.expand(batch_shape + self.tau.shape[-1:]).contiguous()
        z = z.expand(batch_shape + z.shape[-2:])
        b = b.expand(batch_shape + b.shape[-2:])
        rhs = torch.cat([z, b], dim=-2).contiguous()

        # ormqr applies the compact reflectors directly; Q is never materialised.
        transformed = torch.ormqr(packed, tau, rhs, left=True, transpose=True)
        signs = self.row_signs.expand(batch_shape + self.row_signs.shape[-1:])
        root = self.precision_root.expand(
            batch_shape + self.precision_root.shape[-2:]
        )
        information_rhs = transformed[..., :n, :] * signs.unsqueeze(-1)
        solution = torch.linalg.solve_triangular(
            root, information_rhs, upper=True
        )
        trailing = transformed[..., n:, :]
        return InformationRootUpdateTorch(
            precision_root=root,
            information_rhs=information_rhs,
            solution=solution,
            trailing_residual=trailing,
            residual_sum_squares=torch.sum(trailing.square(), dim=-2),
        )

    def apply_vectors(
        self,
        prior_information_rhs: torch.Tensor,
        measurement_rhs: torch.Tensor,
    ) -> InformationRootUpdateTorch:
        """Apply to vector batches ``(..., n)`` and ``(..., m)``.

        For a single fixed design, arbitrary observation batch dimensions are
        flattened into RHS columns.  That is the throughput-oriented GPU path.
        A batched design instead applies one factorization per broadcast batch
        element.
        """
        n, m = self.state_dim, self.measurement_dim
        z = _vector("prior_information_rhs", prior_information_rhs, length=n)
        b = _vector("measurement_rhs", measurement_rhs, length=m)
        _same_backend(self.packed, prior_information_rhs=z, measurement_rhs=b)
        vector_batch = torch.broadcast_shapes(z.shape[:-1], b.shape[:-1])

        if len(self.batch_shape) == 0:
            z = z.expand(vector_batch + (n,))
            b = b.expand(vector_batch + (m,))
            count = math.prod(vector_batch) if vector_batch else 1
            columns = self.apply_columns(
                z.reshape(count, n).transpose(0, 1),
                b.reshape(count, m).transpose(0, 1),
            )
            root = self.precision_root.expand(vector_batch + (n, n))
            return InformationRootUpdateTorch(
                precision_root=root,
                information_rhs=columns.information_rhs.transpose(0, 1).reshape(
                    vector_batch + (n,)
                ),
                solution=columns.solution.transpose(0, 1).reshape(vector_batch + (n,)),
                trailing_residual=columns.trailing_residual.transpose(0, 1).reshape(
                    vector_batch + (m,)
                ),
                residual_sum_squares=columns.residual_sum_squares.reshape(vector_batch),
            )

        batch_shape = torch.broadcast_shapes(self.batch_shape, vector_batch)
        z = z.expand(batch_shape + (n,))
        b = b.expand(batch_shape + (m,))
        columns = self.apply_columns(z.unsqueeze(-1), b.unsqueeze(-1))
        return InformationRootUpdateTorch(
            precision_root=columns.precision_root,
            information_rhs=columns.information_rhs.squeeze(-1),
            solution=columns.solution.squeeze(-1),
            trailing_residual=columns.trailing_residual.squeeze(-1),
            residual_sum_squares=columns.residual_sum_squares.squeeze(-1),
        )


def compile_information_update_torch(
    prior_precision_root: torch.Tensor,
    measurement_matrix: torch.Tensor,
) -> CompiledInformationQRTorch:
    """Factor ``stack([U_prior, A])`` into compact Householder storage once."""
    root = _matrix("prior_precision_root", prior_precision_root)
    if root.shape[-2] != root.shape[-1] or root.shape[-1] == 0:
        raise ValueError("prior_precision_root must be nonempty and square")
    n = root.shape[-1]
    matrix = _matrix("measurement_matrix", measurement_matrix, cols=n)
    _same_backend(root, measurement_matrix=matrix)
    root, matrix = _broadcast_matrices(root, matrix)
    pre_array = torch.cat([root, matrix], dim=-2).contiguous()
    scale = _pre_array_scale(pre_array)
    packed, tau = torch.geqrf(pre_array)
    posterior_root, signs = _normalised_root_from_packed(packed, n)
    _check_root_rank(pre_array, posterior_root, scale)
    return CompiledInformationQRTorch(
        packed=packed,
        tau=tau,
        precision_root=posterior_root,
        row_signs=signs,
        state_dim=n,
        measurement_dim=matrix.shape[-2],
    )


def householder_information_update_torch(
    prior_precision_root: torch.Tensor,
    prior_information_rhs: torch.Tensor,
    measurement_matrix: torch.Tensor,
    measurement_rhs: torch.Tensor,
) -> InformationRootUpdateTorch:
    """Factor and apply one possibly batched whitened information update."""
    compiled = compile_information_update_torch(
        prior_precision_root, measurement_matrix
    )
    return compiled.apply_vectors(prior_information_rhs, measurement_rhs)


def _batch_failure_detail(failed: torch.Tensor) -> str:
    if failed.ndim == 0:
        return " for the unbatched input"
    indices = torch.nonzero(failed, as_tuple=False).detach().cpu().tolist()
    return f" at batch indices {indices}"


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def prepare_noise_whitener_torch(
    covariance: torch.Tensor | None = None,
    *,
    correlation: torch.Tensor | None = None,
    stddev: torch.Tensor | None = None,
    relative_eigenvalue_floor: float | None = None,
    absolute_eigenvalue_floor: float = 0.0,
    negative_eigenvalue_tolerance: float | None = None,
    maximum_relative_loading: float = 1e-3,
    symmetry_tolerance: float | None = None,
    correlation_tolerance: float | None = None,
    name: str = "noise covariance",
) -> NoiseWhitenerTorch:
    """Validate, minimally load, and Cholesky-factor a noise model.

    Supply exactly one of:

    - ``covariance``; or
    - ``correlation`` plus strictly positive ``stddev``, which constructs
      ``diag(stddev) @ correlation @ diag(stddev)``.

    Loading is relative to the covariance spectral scale.  A nearly singular
    positive-semidefinite matrix is lifted to a documented eigenvalue floor;
    a materially indefinite matrix or a repair exceeding
    ``maximum_relative_loading`` is rejected.  All loading is returned in
    :class:`CovarianceLoadingDiagnosticsTorch`, including one value per batch
    element.  This preparation step intentionally synchronises to make invalid
    input observable; repeated hot-path whitening does not.

    The returned object represents ``L^-1`` implicitly and applies it only via
    :func:`torch.linalg.solve_triangular`.

    By default the relative eigenvalue floor is
    ``max(sqrt(machine_epsilon), 8 * dimension * machine_epsilon)``.  The
    ``sqrt(eps)`` term intentionally caps the loaded covariance condition
    number near ``1/sqrt(eps)`` (about ``2.9e3`` in float32 and ``6.7e7`` in
    float64).  The float32 load can be statistically material; it is exposed
    in diagnostics and may be lowered explicitly when a caller has validated
    a tighter numerical error budget.  The default negative-eigenvalue
    tolerance is capped at half of the remaining loading budget, so any
    round-off-negative input accepted by the hard indefiniteness gate remains
    repairable within ``maximum_relative_loading``.  An explicitly supplied
    tolerance may intentionally override that coupling and still be rejected
    by the separate loading-budget gate.
    """
    covariance_mode = covariance is not None
    correlation_mode = correlation is not None or stddev is not None
    if covariance_mode == correlation_mode:
        raise ValueError(
            "supply either covariance or correlation plus stddev, but not both"
        )

    source: str
    if covariance_mode:
        if correlation is not None or stddev is not None:
            raise ValueError("stddev/correlation cannot accompany covariance")
        raw = _matrix(name, covariance)
        if raw.shape[-2] != raw.shape[-1] or raw.shape[-1] == 0:
            raise ValueError(f"{name} must be nonempty and square")
        raw = raw.clone()
        source = "covariance"
    else:
        if correlation is None or stddev is None:
            raise ValueError("correlation and stddev must be supplied together")
        corr = _matrix("correlation", correlation)
        if corr.shape[-2] != corr.shape[-1] or corr.shape[-1] == 0:
            raise ValueError("correlation must be nonempty and square")
        standard_deviation = _vector(
            "stddev", stddev, length=corr.shape[-1]
        )
        _same_backend(corr, stddev=standard_deviation)
        batch_shape = torch.broadcast_shapes(
            corr.shape[:-2], standard_deviation.shape[:-1]
        )
        corr = corr.expand(batch_shape + corr.shape[-2:]).clone()
        standard_deviation = standard_deviation.expand(
            batch_shape + standard_deviation.shape[-1:]
        ).clone()
        if not bool(torch.isfinite(corr).all().item()):
            raise ValueError("correlation must be finite")
        if not bool(torch.isfinite(standard_deviation).all().item()):
            raise ValueError("stddev must be finite")
        nonpositive = torch.any(standard_deviation <= 0, dim=-1)
        if bool(torch.any(nonpositive).item()):
            raise ValueError(
                "stddev must be strictly positive"
                + _batch_failure_detail(nonpositive)
            )
        n = corr.shape[-1]
        eps = torch.finfo(corr.dtype).eps
        corr_tol = (
            64.0 * n * eps
            if correlation_tolerance is None
            else _nonnegative_finite("correlation_tolerance", correlation_tolerance)
        )
        diagonal_error = torch.amax(
            torch.abs(torch.diagonal(corr, dim1=-2, dim2=-1) - 1.0), dim=-1
        )
        bad_diagonal = diagonal_error > corr_tol
        if bool(torch.any(bad_diagonal).item()):
            worst = float(torch.amax(diagonal_error).detach().cpu())
            raise ValueError(
                f"correlation diagonal must be one within tolerance {corr_tol:g}; "
                f"maximum error={worst:g}"
                + _batch_failure_detail(bad_diagonal)
            )
        maximum_entry = torch.amax(torch.abs(corr), dim=(-2, -1))
        bad_entry = maximum_entry > 1.0 + corr_tol
        if bool(torch.any(bad_entry).item()):
            worst = float(torch.amax(maximum_entry).detach().cpu())
            raise ValueError(
                f"correlation entries must satisfy |rho| <= 1 within tolerance "
                f"{corr_tol:g}; maximum |rho|={worst:g}"
                + _batch_failure_detail(bad_entry)
            )
        raw = (
            standard_deviation.unsqueeze(-1)
            * corr
            * standard_deviation.unsqueeze(-2)
        )
        name = "noise covariance from correlation"
        source = "correlation+stddev"

    if not bool(torch.isfinite(raw).all().item()):
        raise ValueError(f"{name} must be finite")
    n = raw.shape[-1]
    eps = torch.finfo(raw.dtype).eps
    relative_floor = (
        max(math.sqrt(eps), 8.0 * n * eps)
        if relative_eigenvalue_floor is None
        else _nonnegative_finite(
            "relative_eigenvalue_floor", relative_eigenvalue_floor
        )
    )
    absolute_floor = _nonnegative_finite(
        "absolute_eigenvalue_floor", absolute_eigenvalue_floor
    )
    maximum_loading = _nonnegative_finite(
        "maximum_relative_loading", maximum_relative_loading
    )
    if negative_eigenvalue_tolerance is None:
        repair_budget = max(maximum_loading - relative_floor, 0.0)
        negative_tolerance = min(64.0 * n * eps, 0.5 * repair_budget)
    else:
        negative_tolerance = _nonnegative_finite(
            "negative_eigenvalue_tolerance", negative_eigenvalue_tolerance
        )
    symmetry_limit = (
        64.0 * n * eps
        if symmetry_tolerance is None
        else _nonnegative_finite("symmetry_tolerance", symmetry_tolerance)
    )

    entry_scale = torch.amax(torch.abs(raw), dim=(-2, -1))
    symmetry_error = torch.amax(
        torch.abs(raw - raw.transpose(-2, -1)), dim=(-2, -1)
    )
    relative_symmetry_error = symmetry_error / torch.clamp(
        entry_scale, min=torch.finfo(raw.dtype).tiny
    )
    nonsymmetric = relative_symmetry_error > symmetry_limit
    if bool(torch.any(nonsymmetric).item()):
        worst = float(torch.amax(relative_symmetry_error).detach().cpu())
        raise ValueError(
            f"{name} is not symmetric within relative tolerance "
            f"{symmetry_limit:g}; maximum relative error={worst:g}"
            + _batch_failure_detail(nonsymmetric)
        )

    symmetric = 0.5 * (raw + raw.transpose(-2, -1))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = eigenvalues[..., 0]
    matrix_scale = torch.amax(torch.abs(eigenvalues), dim=-1)
    zero_scale = matrix_scale == 0
    if absolute_floor == 0.0 and bool(torch.any(zero_scale).item()):
        raise ValueError(
            f"{name} has zero spectral scale; supply positive variances, or "
            "provide absolute_eigenvalue_floor > 0 together with "
            "maximum_relative_loading >= 1 to authorize a full-scale repair"
            + _batch_failure_detail(zero_scale)
        )

    materially_indefinite = minimum < -negative_tolerance * matrix_scale
    if bool(torch.any(materially_indefinite).item()):
        relative_negative = -minimum / torch.clamp(
            matrix_scale, min=torch.finfo(raw.dtype).tiny
        )
        worst = float(torch.amax(relative_negative).detach().cpu())
        raise ValueError(
            f"{name} is genuinely indefinite: worst relative negative "
            f"eigenvalue={worst:g} exceeds tolerance {negative_tolerance:g}"
            + _batch_failure_detail(materially_indefinite)
        )

    absolute_floor_tensor = torch.as_tensor(
        absolute_floor, dtype=raw.dtype, device=raw.device
    )
    target_minimum = torch.maximum(
        relative_floor * matrix_scale, absolute_floor_tensor
    )
    loading = torch.clamp(target_minimum - minimum, min=0.0)
    loading_reference = torch.where(
        matrix_scale > 0, matrix_scale, target_minimum
    )
    relative_loading = loading / torch.clamp(
        loading_reference, min=torch.finfo(raw.dtype).tiny
    )
    excessive_loading = relative_loading > maximum_loading
    if bool(torch.any(excessive_loading).item()):
        worst = float(torch.amax(relative_loading).detach().cpu())
        raise ValueError(
            f"{name} requires relative diagonal loading {worst:g}, exceeding "
            f"maximum_relative_loading={maximum_loading:g}"
            + _batch_failure_detail(excessive_loading)
        )

    identity = torch.eye(n, dtype=raw.dtype, device=raw.device)
    loaded_covariance = symmetric + loading[..., None, None] * identity
    cholesky, info = torch.linalg.cholesky_ex(loaded_covariance, check_errors=False)
    failed_cholesky = info != 0
    if bool(torch.any(failed_cholesky).item()):
        raise torch.linalg.LinAlgError(
            f"{name} remained non-positive-definite after declared loading; "
            "increase relative_eigenvalue_floor"
            + _batch_failure_detail(failed_cholesky)
        )

    diagnostics = CovarianceLoadingDiagnosticsTorch(
        matrix_scale=matrix_scale,
        minimum_eigenvalue=minimum,
        target_minimum_eigenvalue=target_minimum,
        diagonal_loading=loading,
        relative_diagonal_loading=relative_loading,
        relative_symmetry_error=relative_symmetry_error,
        was_loaded=loading > 0,
        relative_eigenvalue_floor=relative_floor,
        negative_eigenvalue_tolerance=negative_tolerance,
        maximum_relative_loading=maximum_loading,
        source=source,
    )
    return NoiseWhitenerTorch(
        covariance=loaded_covariance,
        cholesky_factor=cholesky,
        diagnostics=diagnostics,
    )


def regularize_covariance_torch(
    covariance: torch.Tensor,
    *,
    jitter: float = 1e-9,
    name: str = "covariance",
) -> torch.Tensor:
    """Compatibility helper using an absolute covariance floor.

    New square-root and noise-conditioning paths use
    :func:`prepare_noise_whitener_torch`, whose loading is scale-relative and
    observable.  Keep this helper only for callers that explicitly depend on
    the historical absolute-unit policy.
    """
    if jitter < 0:
        raise ValueError("jitter must be nonnegative")
    covariance = _matrix(name, covariance)
    if covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError(f"{name} must be square")
    covariance = 0.5 * (covariance + covariance.transpose(-2, -1))
    eigenvalues = torch.linalg.eigvalsh(covariance)
    minimum = eigenvalues[..., 0]
    floor = max(float(jitter), torch.finfo(covariance.dtype).eps)
    if bool(torch.any(minimum < -100.0 * floor).item()):
        worst = float(torch.amin(minimum).detach().cpu())
        raise ValueError(f"{name} is not positive semidefinite; min eigenvalue={worst:g}")
    lift = torch.clamp(floor - minimum, min=0.0)
    identity = torch.eye(
        covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
    )
    return covariance + lift[..., None, None] * identity


def precision_root_from_covariance_torch(
    covariance: torch.Tensor,
    *,
    jitter: float = 0.0,
    relative_eigenvalue_floor: float | None = None,
    negative_eigenvalue_tolerance: float | None = None,
    maximum_relative_loading: float = 1e-3,
) -> torch.Tensor:
    """Return upper ``U`` with ``P^-1 = U.T @ U`` without forming ``P^-1``.

    ``jitter`` is an optional absolute floor in covariance units; it defaults
    to zero so the scale-relative policy remains valid under unit changes.
    """
    whitener = prepare_noise_whitener_torch(
        covariance,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
        absolute_eigenvalue_floor=jitter,
        negative_eigenvalue_tolerance=negative_eigenvalue_tolerance,
        maximum_relative_loading=maximum_relative_loading,
        name="prior covariance",
    )
    inverse_factor = whitener.materialize_row_root()
    n = inverse_factor.shape[-1]
    scale = _pre_array_scale(inverse_factor)
    packed, _ = torch.geqrf(inverse_factor.contiguous())
    root, _ = _normalised_root_from_packed(packed, n)
    _check_root_rank(inverse_factor, root, scale)
    return root


def conditional_measurement_model_torch(
    prior_precision_root: torch.Tensor,
    H: torch.Tensor,
    observation_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the loaded decorrelated ``(J, Rc)`` likelihood.

    For loading diagnostics and the reusable implicit inverse factor, call
    :func:`conditional_measurement_whitener_torch` directly.
    """
    effective_H, whitener = conditional_measurement_whitener_torch(
        prior_precision_root,
        H,
        observation_covariance,
        cross_covariance,
        jitter=jitter,
    )
    return effective_H, whitener.covariance


def conditional_measurement_whitener_torch(
    prior_precision_root: torch.Tensor,
    H: torch.Tensor,
    observation_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
    *,
    jitter: float = 0.0,
    relative_eigenvalue_floor: float | None = None,
    negative_eigenvalue_tolerance: float | None = None,
    maximum_relative_loading: float = 1e-3,
) -> tuple[torch.Tensor, NoiseWhitenerTorch]:
    """Decorrelate a measurement block and prepare its implicit whitener.

    ``jitter`` is an absolute eigenvalue floor in the measurement's units;
    the scale-relative floor and repair diagnostics come from
    :func:`prepare_noise_whitener_torch`.  A bad Schur complement is rejected,
    not silently converted into a plausible covariance.
    """
    if jitter < 0:
        raise ValueError("jitter must be nonnegative")
    root = _matrix("prior_precision_root", prior_precision_root)
    if root.shape[-2] != root.shape[-1] or root.shape[-1] == 0:
        raise ValueError("prior_precision_root must be nonempty and square")
    n = root.shape[-1]
    H = _matrix("H", H, cols=n)
    m = H.shape[-2]
    observation_covariance = _matrix(
        "observation_covariance", observation_covariance, rows=m, cols=m
    )
    cross_covariance = _matrix(
        "cross_covariance", cross_covariance, rows=n, cols=m
    )
    _same_backend(
        root,
        H=H,
        observation_covariance=observation_covariance,
        cross_covariance=cross_covariance,
    )
    root, H, observation_covariance, cross_covariance = _broadcast_matrices(
        root, H, observation_covariance, cross_covariance
    )
    root_cross = root @ cross_covariance
    effective_H = H + root_cross.transpose(-2, -1) @ root
    raw_conditional_R = (
        observation_covariance
        - root_cross.transpose(-2, -1) @ root_cross
    )
    whitener = prepare_noise_whitener_torch(
        raw_conditional_R,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
        absolute_eigenvalue_floor=jitter,
        negative_eigenvalue_tolerance=negative_eigenvalue_tolerance,
        maximum_relative_loading=maximum_relative_loading,
        name="conditional observation covariance R - C.T P^-1 C",
    )
    return effective_H, whitener


def _posterior_covariance(root: torch.Tensor) -> torch.Tensor:
    n = root.shape[-1]
    identity = torch.eye(n, dtype=root.dtype, device=root.device).expand(
        root.shape[:-2] + (n, n)
    )
    inverse_root = torch.linalg.solve_triangular(root, identity, upper=True)
    covariance = inverse_root @ inverse_root.transpose(-2, -1)
    return 0.5 * (covariance + covariance.transpose(-2, -1))


def _covariance_from_precision_factor(factor: torch.Tensor) -> torch.Tensor:
    """Recover covariance from any square ``U`` satisfying ``Lambda=U.T U``."""
    n = factor.shape[-1]
    identity = torch.eye(n, dtype=factor.dtype, device=factor.device).expand(
        factor.shape[:-2] + (n, n)
    )
    inverse_factor = torch.linalg.solve(factor, identity)
    covariance = inverse_factor @ inverse_factor.transpose(-2, -1)
    return 0.5 * (covariance + covariance.transpose(-2, -1))


def condition_correlated_gaussian_qr_torch(
    mean: torch.Tensor,
    prior_precision_root: torch.Tensor,
    observation: torch.Tensor,
    observation_covariance: torch.Tensor,
    H: torch.Tensor,
    cross_covariance: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> JointSquareRootUpdateTorch:
    """Condition a Gaussian with batched tensors and compact Householder QR.

    This functional entry point validates and factorizes the design on every
    call, including an eigendecomposition and device synchronization in the
    covariance guard. Repeated fixed-design throughput should use
    :class:`CompiledCorrelatedConditionerTorch`.
    """
    root = _matrix("prior_precision_root", prior_precision_root)
    if root.shape[-2] != root.shape[-1] or root.shape[-1] == 0:
        raise ValueError("prior_precision_root must be nonempty and square")
    n = root.shape[-1]
    mean = _vector("mean", mean, length=n)
    H = _matrix("H", H, cols=n)
    m = H.shape[-2]
    observation = _vector("observation", observation, length=m)
    _same_backend(root, mean=mean, H=H, observation=observation)

    effective_H, conditional_whitener = conditional_measurement_whitener_torch(
        root,
        H,
        observation_covariance,
        cross_covariance,
        jitter=jitter,
    )
    conditional_R = conditional_whitener.covariance
    design_batch = torch.broadcast_shapes(
        root.shape[:-2], H.shape[:-2], effective_H.shape[:-2]
    )
    result_batch = torch.broadcast_shapes(
        design_batch, mean.shape[:-1], observation.shape[:-1]
    )
    root = root.expand(design_batch + (n, n))
    H_design = H.expand(design_batch + (m, n))
    effective_H = effective_H.expand(design_batch + (m, n))
    conditional_R = conditional_R.expand(design_batch + (m, m))
    whitened_H = conditional_whitener.apply_columns(effective_H)
    compiled = compile_information_update_torch(root, whitened_H)

    mean = mean.expand(result_batch + (n,))
    observation = observation.expand(result_batch + (m,))
    H_result = H_design.expand(result_batch + (m, n))
    innovation = observation - (H_result @ mean.unsqueeze(-1)).squeeze(-1)
    whitened_innovation = conditional_whitener.apply_vectors(innovation)
    information = compiled.apply_vectors(
        torch.zeros_like(mean), whitened_innovation
    )
    return JointSquareRootUpdateTorch(
        mean=mean + information.solution,
        covariance=_posterior_covariance(information.precision_root),
        precision_root=information.precision_root,
        information_rhs=information.information_rhs,
        innovation=innovation,
        effective_observation_matrix=effective_H.expand(result_batch + (m, n)),
        conditional_observation_covariance=conditional_R.expand(
            result_batch + (m, m)
        ),
        conditional_loading_diagnostics=conditional_whitener.diagnostics,
        whitened_observation_matrix=whitened_H.expand(result_batch + (m, n)),
        whitened_innovation=whitened_innovation,
        residual_sum_squares=information.residual_sum_squares,
    )


@dataclass(frozen=True)
class CompiledCorrelatedConditionerTorch:
    """Compile one fixed correlated measurement design and reuse it per row."""

    prior_precision_root: torch.Tensor
    H: torch.Tensor
    effective_observation_matrix: torch.Tensor
    conditional_observation_covariance: torch.Tensor
    conditional_loading_diagnostics: CovarianceLoadingDiagnosticsTorch
    conditional_cholesky: torch.Tensor
    whitened_observation_matrix: torch.Tensor
    information_qr: CompiledInformationQRTorch
    posterior_covariance: torch.Tensor

    @classmethod
    def compile(
        cls,
        prior_precision_root: torch.Tensor,
        H: torch.Tensor,
        observation_covariance: torch.Tensor,
        cross_covariance: torch.Tensor,
        *,
        jitter: float = 0.0,
    ) -> "CompiledCorrelatedConditionerTorch":
        """Compile an unbatched fixed design.

        Unbatched design is intentional: it lets all sample dimensions become
        RHS columns of one ``ormqr`` call.  Use the functional API for a batch
        of distinct designs.
        """
        for name, value in (
            ("prior_precision_root", prior_precision_root),
            ("H", H),
            ("observation_covariance", observation_covariance),
            ("cross_covariance", cross_covariance),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"{name} must be one unbatched matrix")
        # Compiled objects own a snapshot of every design input. Otherwise an
        # in-place caller mutation could combine a new innovation with stale
        # QR reflectors (or race an asynchronous device operation).
        prior_precision_root = prior_precision_root.clone()
        H = H.clone()
        observation_covariance = observation_covariance.clone()
        cross_covariance = cross_covariance.clone()
        effective_H, conditional_whitener = conditional_measurement_whitener_torch(
            prior_precision_root,
            H,
            observation_covariance,
            cross_covariance,
            jitter=jitter,
        )
        conditional_R = conditional_whitener.covariance
        chol = conditional_whitener.cholesky_factor
        whitened_H = conditional_whitener.apply_columns(effective_H)
        information_qr = compile_information_update_torch(
            prior_precision_root, whitened_H
        )
        return cls(
            prior_precision_root=prior_precision_root,
            H=H,
            effective_observation_matrix=effective_H,
            conditional_observation_covariance=conditional_R,
            conditional_loading_diagnostics=conditional_whitener.diagnostics,
            conditional_cholesky=chol,
            whitened_observation_matrix=whitened_H,
            information_qr=information_qr,
            posterior_covariance=_posterior_covariance(
                information_qr.precision_root
            ),
        )

    def condition(
        self, mean: torch.Tensor, observation: torch.Tensor
    ) -> JointSquareRootUpdateTorch:
        """Condition one row or an arbitrary batch using the compiled design."""
        n = self.prior_precision_root.shape[-1]
        m = self.H.shape[-2]
        mean = _vector("mean", mean, length=n)
        observation = _vector("observation", observation, length=m)
        _same_backend(
            self.prior_precision_root, mean=mean, observation=observation
        )
        batch_shape = torch.broadcast_shapes(mean.shape[:-1], observation.shape[:-1])
        mean = mean.expand(batch_shape + (n,))
        observation = observation.expand(batch_shape + (m,))
        innovation = observation - torch.einsum("mn,...n->...m", self.H, mean)
        count = math.prod(batch_shape) if batch_shape else 1
        innovation_columns = innovation.reshape(count, m).transpose(0, 1)
        whitened_innovation = torch.linalg.solve_triangular(
            self.conditional_cholesky,
            innovation_columns.contiguous(),
            upper=False,
        ).transpose(0, 1).reshape(batch_shape + (m,))
        information = self.information_qr.apply_vectors(
            torch.zeros_like(mean), whitened_innovation
        )
        return JointSquareRootUpdateTorch(
            mean=mean + information.solution,
            covariance=self.posterior_covariance.expand(batch_shape + (n, n)),
            precision_root=information.precision_root,
            information_rhs=information.information_rhs,
            innovation=innovation,
            effective_observation_matrix=self.effective_observation_matrix.expand(
                batch_shape + (m, n)
            ),
            conditional_observation_covariance=(
                self.conditional_observation_covariance.expand(
                    batch_shape + (m, m)
                )
            ),
            conditional_loading_diagnostics=self.conditional_loading_diagnostics,
            whitened_observation_matrix=self.whitened_observation_matrix.expand(
                batch_shape + (m, n)
            ),
            whitened_innovation=whitened_innovation,
            residual_sum_squares=information.residual_sum_squares,
        )


@dataclass(frozen=True)
class CompiledDenseGainConditionerTorch:
    """Compile the dense correlated Gaussian gain for one fixed design.

    This is the matched static-throughput baseline for the compiled QR path.
    It uses the same conditional ``(J, Rc)`` model (and therefore the same
    cross-covariance validation/regularisation), then caches

    ``K = Cov(e, r) @ Cov(r)^-1``.

    Per row, conditioning is only ``r = y - H mean`` and ``mean + K r``.
    Unlike square-root QR, this class does not carry an updated precision root
    into a later measurement block; recompile it whenever the design changes.
    """

    H: torch.Tensor
    gain: torch.Tensor
    posterior_covariance: torch.Tensor
    innovation_covariance: torch.Tensor
    conditional_loading_diagnostics: CovarianceLoadingDiagnosticsTorch

    @classmethod
    def compile(
        cls,
        prior_precision_root: torch.Tensor,
        H: torch.Tensor,
        observation_covariance: torch.Tensor,
        cross_covariance: torch.Tensor,
        *,
        jitter: float = 0.0,
    ) -> "CompiledDenseGainConditionerTorch":
        """Compile an unbatched fixed design into a dense correlated gain."""
        for name, value in (
            ("prior_precision_root", prior_precision_root),
            ("H", H),
            ("observation_covariance", observation_covariance),
            ("cross_covariance", cross_covariance),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"{name} must be one unbatched matrix")
        root = _matrix("prior_precision_root", prior_precision_root).clone()
        if root.shape[-2] != root.shape[-1] or root.shape[-1] == 0:
            raise ValueError("prior_precision_root must be nonempty and square")
        n = root.shape[-1]
        H = _matrix("H", H, cols=n).clone()
        observation_covariance = observation_covariance.clone()
        cross_covariance = cross_covariance.clone()
        _same_backend(root, H=H)

        # Use the same Schur-complement model as QR.  This both validates the
        # proposed joint covariance and makes jitter behavior directly
        # comparable between the two static implementations.
        effective_H, conditional_whitener = conditional_measurement_whitener_torch(
            root,
            H,
            observation_covariance,
            cross_covariance,
            jitter=jitter,
        )
        conditional_R = conditional_whitener.covariance
        # The first precision factor need not already be triangular: left
        # orthogonal rotations preserve U.T U.  Use a general solve here;
        # posterior QR roots use the cheaper triangular helper above.
        prior_covariance = _covariance_from_precision_factor(root)
        state_innovation_cross = prior_covariance @ effective_H.mT
        innovation_covariance = (
            effective_H @ state_innovation_cross + conditional_R
        )
        innovation_covariance = 0.5 * (
            innovation_covariance + innovation_covariance.mT
        )
        innovation_chol = torch.linalg.cholesky(innovation_covariance)
        gain = torch.cholesky_solve(
            state_innovation_cross.mT, innovation_chol
        ).mT
        # Joseph form avoids the catastrophic subtraction in
        # ``P - K S K.T`` when an observation is much more precise than the
        # prior.  Both summands are PSD and preserve the covariance's units.
        identity = torch.eye(n, dtype=root.dtype, device=root.device)
        residual_map = identity - gain @ effective_H
        posterior_covariance = (
            residual_map @ prior_covariance @ residual_map.mT
            + gain @ conditional_R @ gain.mT
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.mT
        )
        return cls(
            H=H,
            gain=gain,
            posterior_covariance=posterior_covariance,
            innovation_covariance=innovation_covariance,
            conditional_loading_diagnostics=conditional_whitener.diagnostics,
        )

    def condition(
        self, mean: torch.Tensor, observation: torch.Tensor
    ) -> DenseGaussianUpdateTorch:
        """Condition one row or an arbitrary row batch with the cached gain."""
        n = self.H.shape[-1]
        m = self.H.shape[-2]
        mean = _vector("mean", mean, length=n)
        observation = _vector("observation", observation, length=m)
        _same_backend(self.H, mean=mean, observation=observation)
        batch_shape = torch.broadcast_shapes(mean.shape[:-1], observation.shape[:-1])
        mean = mean.expand(batch_shape + (n,))
        observation = observation.expand(batch_shape + (m,))
        innovation = observation - torch.einsum("mn,...n->...m", self.H, mean)
        posterior_mean = mean + torch.einsum(
            "nm,...m->...n", self.gain, innovation
        )
        return DenseGaussianUpdateTorch(
            mean=posterior_mean,
            covariance=self.posterior_covariance.expand(batch_shape + (n, n)),
            gain=self.gain.expand(batch_shape + (n, m)),
            innovation=innovation,
            innovation_covariance=self.innovation_covariance.expand(
                batch_shape + (m, m)
            ),
            conditional_loading_diagnostics=self.conditional_loading_diagnostics,
        )


@dataclass(frozen=True)
class SquareRootInformationStateTorch:
    """Sequential information state in one fixed coordinate system.

    ``information_rhs`` is the square-root RHS ``z``, not the canonical
    information vector ``eta``; ``eta = U.T @ z``. Each block threads
    ``U_post`` and ``z``, so every measurement RHS must use
    the same state origin.  To recenter at the current posterior mean, instead
    reset ``z`` to zero and shift each later RHS by ``-A @ current_solution``;
    do not both carry ``z`` and recenter the likelihood.
    """

    precision_root: torch.Tensor
    information_rhs: torch.Tensor

    def update(
        self, measurement_matrix: torch.Tensor, measurement_rhs: torch.Tensor
    ) -> tuple["SquareRootInformationStateTorch", InformationRootUpdateTorch]:
        """Absorb one independent whitened block in the state's coordinates."""
        result = householder_information_update_torch(
            self.precision_root,
            self.information_rhs,
            measurement_matrix,
            measurement_rhs,
        )
        state = SquareRootInformationStateTorch(
            precision_root=result.precision_root,
            information_rhs=result.information_rhs,
        )
        return state, result

    def update_noise_block(
        self,
        measurement_matrix: torch.Tensor,
        measurement_rhs: torch.Tensor,
        whitener: NoiseWhitenerTorch,
    ) -> tuple["SquareRootInformationStateTorch", InformationRootUpdateTorch]:
        """Whiten one noise block consistently, then absorb it with QR.

        ``measurement_matrix`` is the unwhitened conditional design ``J`` and
        ``measurement_rhs`` is its matching innovation ``r``.  Both are acted
        on from the left by the same implicit ``L^-1``.  This convenience
        avoids materialising an inverse root or accidentally applying the
        whitening factor with the wrong orientation.  This hot path does not
        recheck ``measurement_matrix`` or ``measurement_rhs`` for finiteness;
        callers that cannot tolerate NaN/Inf propagation must gate them first.
        """
        if not isinstance(whitener, NoiseWhitenerTorch):
            raise TypeError("whitener must be a NoiseWhitenerTorch")
        whitened_matrix = whitener.apply_columns(measurement_matrix)
        whitened_rhs = whitener.apply_vectors(measurement_rhs)
        return self.update(whitened_matrix, whitened_rhs)
