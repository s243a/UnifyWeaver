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
``torch.linalg.qr`` on some PyTorch/device combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


__all__ = [
    "CompiledCorrelatedConditionerTorch",
    "CompiledDenseGainConditionerTorch",
    "CompiledInformationQRTorch",
    "DenseGaussianUpdateTorch",
    "InformationRootUpdateTorch",
    "JointSquareRootUpdateTorch",
    "SquareRootInformationStateTorch",
    "compile_information_update_torch",
    "condition_correlated_gaussian_qr_torch",
    "conditional_measurement_model_torch",
    "householder_information_update_torch",
    "precision_root_from_covariance_torch",
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


def _check_root_rank(pre_array: torch.Tensor, root: torch.Tensor) -> None:
    scale = torch.linalg.matrix_norm(pre_array, ord="fro", dim=(-2, -1))
    scale = torch.maximum(scale, torch.ones_like(scale))
    tolerance = torch.finfo(pre_array.dtype).eps * max(pre_array.shape[-2:]) * scale
    smallest = torch.amin(torch.abs(torch.diagonal(root, dim1=-2, dim2=-1)), dim=-1)
    failed = smallest <= tolerance
    if bool(torch.any(failed).item()):
        if failed.ndim:
            indices = torch.nonzero(failed, as_tuple=False).detach().cpu().tolist()
            detail = f" at batch indices {indices}"
        else:
            detail = " for the unbatched design"
        raise torch.linalg.LinAlgError(f"information pre-array is rank deficient{detail}")


@dataclass(frozen=True)
class InformationRootUpdateTorch:
    """One information-root update, possibly batched or with multiple RHSs."""

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
    packed, tau = torch.geqrf(pre_array)
    posterior_root, signs = _normalised_root_from_packed(packed, n)
    _check_root_rank(pre_array, posterior_root)
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


def regularize_covariance_torch(
    covariance: torch.Tensor,
    *,
    jitter: float = 1e-9,
    name: str = "covariance",
) -> torch.Tensor:
    """Symmetrise and minimally lift a PSD covariance, in batch.

    The effective floor is at least machine epsilon.  Thus float64 matches the
    NumPy prototype's default ``1e-9`` floor, while float32 automatically uses
    a representable stability floor.
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
    covariance: torch.Tensor, *, jitter: float = 1e-9
) -> torch.Tensor:
    """Return upper ``U`` with ``P^-1 = U.T @ U`` without forming ``P^-1``."""
    covariance = regularize_covariance_torch(
        covariance, jitter=jitter, name="prior covariance"
    )
    n = covariance.shape[-1]
    chol = torch.linalg.cholesky(covariance)
    identity = torch.eye(n, dtype=chol.dtype, device=chol.device).expand(
        chol.shape[:-2] + (n, n)
    )
    inverse_factor = torch.linalg.solve_triangular(chol, identity, upper=False)
    packed, _ = torch.geqrf(inverse_factor.contiguous())
    root, _ = _normalised_root_from_packed(packed, n)
    _check_root_rank(inverse_factor, root)
    return root


def conditional_measurement_model_torch(
    prior_precision_root: torch.Tensor,
    H: torch.Tensor,
    observation_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
    *,
    jitter: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the decorrelated ``(J, Rc)`` likelihood in information form."""
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
    conditional_R = (
        observation_covariance
        - root_cross.transpose(-2, -1) @ root_cross
    )
    conditional_R = regularize_covariance_torch(
        conditional_R,
        jitter=jitter,
        name="conditional observation covariance R - C.T P^-1 C",
    )
    return effective_H, conditional_R


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
    jitter: float = 1e-9,
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

    effective_H, conditional_R = conditional_measurement_model_torch(
        root,
        H,
        observation_covariance,
        cross_covariance,
        jitter=jitter,
    )
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
    chol = torch.linalg.cholesky(conditional_R)
    whitened_H = torch.linalg.solve_triangular(
        chol, effective_H, upper=False
    )
    compiled = compile_information_update_torch(root, whitened_H)

    mean = mean.expand(result_batch + (n,))
    observation = observation.expand(result_batch + (m,))
    H_result = H_design.expand(result_batch + (m, n))
    innovation = observation - (H_result @ mean.unsqueeze(-1)).squeeze(-1)
    chol_result = chol.expand(result_batch + (m, m))
    whitened_innovation = torch.linalg.solve_triangular(
        chol_result, innovation.unsqueeze(-1), upper=False
    ).squeeze(-1)
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
        jitter: float = 1e-9,
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
        # Compiled objects own a snapshot of coefficient inputs. Otherwise an
        # in-place caller mutation of H would combine a new innovation with
        # stale QR reflectors.
        prior_precision_root = prior_precision_root.clone()
        H = H.clone()
        effective_H, conditional_R = conditional_measurement_model_torch(
            prior_precision_root,
            H,
            observation_covariance,
            cross_covariance,
            jitter=jitter,
        )
        chol = torch.linalg.cholesky(conditional_R)
        whitened_H = torch.linalg.solve_triangular(
            chol, effective_H, upper=False
        )
        information_qr = compile_information_update_torch(
            prior_precision_root, whitened_H
        )
        return cls(
            prior_precision_root=prior_precision_root,
            H=H,
            effective_observation_matrix=effective_H,
            conditional_observation_covariance=conditional_R,
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

    @classmethod
    def compile(
        cls,
        prior_precision_root: torch.Tensor,
        H: torch.Tensor,
        observation_covariance: torch.Tensor,
        cross_covariance: torch.Tensor,
        *,
        jitter: float = 1e-9,
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
        _same_backend(root, H=H)

        # Use the same Schur-complement model as QR.  This both validates the
        # proposed joint covariance and makes jitter behavior directly
        # comparable between the two static implementations.
        effective_H, conditional_R = conditional_measurement_model_torch(
            root,
            H,
            observation_covariance,
            cross_covariance,
            jitter=jitter,
        )
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
        posterior_covariance = (
            prior_covariance - gain @ state_innovation_cross.mT
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.mT
        )
        # The dense subtraction can lose the last positive bits when a very
        # informative observation makes P_post tiny.  Keep this baseline a
        # valid Gaussian rather than returning a singular/negative covariance;
        # the square-root path is positive definite by construction.
        posterior_covariance = regularize_covariance_torch(
            posterior_covariance,
            jitter=jitter,
            name="dense posterior covariance",
        )
        return cls(
            H=H,
            gain=gain,
            posterior_covariance=posterior_covariance,
            innovation_covariance=innovation_covariance,
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
        )


@dataclass(frozen=True)
class SquareRootInformationStateTorch:
    """Sequential information state in one fixed coordinate system.

    Each block threads ``U_post`` and ``z``, so every measurement RHS must use
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
