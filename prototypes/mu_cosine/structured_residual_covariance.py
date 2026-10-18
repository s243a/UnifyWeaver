#!/usr/bin/env python3
"""PSD structured cross-item covariance for conditional judge residuals.

The fitted family is an additive linear model of coregionalisation (LMC),

    Rc = I_item kron B0 + K_sem kron B_sem + K_graph kron B_graph,

with every channel component parameterized as ``B = L L.T``.  The module keeps statistical nugget covariance
inside ``B0`` separate from the existing conditioner's observable numerical loading.

All arrays use item-major order: the channels for item 0 are contiguous, followed by the channels for item 1.
Fitting is deterministic CPU float64 pairwise composite likelihood.  Held evaluation uses the full materialized
covariance; pair terms are an optimization device, not independent evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from joint_square_root_conditioner_torch import (
    SquareRootInformationStateTorch,
    precision_root_from_covariance_torch,
    prepare_noise_whitener_torch,
)


LOG2PI = math.log(2.0 * math.pi)


def _matrix(name, value, *, rows=None, cols=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} has {array.shape[0]} rows, expected {rows}")
    if cols is not None and array.shape[1] != cols:
        raise ValueError(f"{name} has {array.shape[1]} columns, expected {cols}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric(name, value, *, dimension=None):
    array = _matrix(name, value)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if dimension is not None and array.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    scale = max(float(np.max(np.abs(array), initial=0.0)), 1.0)
    if not np.allclose(array, array.T, rtol=0.0, atol=128 * np.finfo(float).eps * scale):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (array + array.T)


def _positive_finite(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _psd_floor(covariance, relative_floor=1e-10):
    """Return a symmetric SPD covariance and the explicit absolute loading used."""
    covariance = _symmetric("covariance", covariance)
    relative_floor = _positive_finite("relative_floor", relative_floor)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), np.finfo(float).tiny)
    target = relative_floor * scale
    loading = max(target - float(eigenvalues[0]), 0.0)
    return covariance + np.eye(len(covariance)) * loading, loading


def conditional_residuals(
    prior_errors,
    measurement_errors,
    prior_covariance,
    cross_covariance,
    observation_matrix,
):
    """Return conditional design ``J`` and residuals ``q`` under the project sign convention.

    ``e = truth-prior``, ``v = measurement-H truth`` and ``C=Cov(e,v)`` imply
    ``J=H+C.T P^-1`` and ``q=v-C.T P^-1 e``.  Rows of ``prior_errors`` and
    ``measurement_errors`` are aligned campaign items.
    """
    e = _matrix("prior_errors", prior_errors)
    v = _matrix("measurement_errors", measurement_errors, rows=len(e))
    P = _symmetric("prior_covariance", prior_covariance, dimension=e.shape[1])
    C = _matrix("cross_covariance", cross_covariance, rows=e.shape[1], cols=v.shape[1])
    H = _matrix("observation_matrix", observation_matrix, rows=v.shape[1], cols=e.shape[1])
    solved = np.linalg.solve(P, C)
    return H + C.T @ np.linalg.solve(P, np.eye(P.shape[0])), v - e @ solved


def median_rbf_bandwidth(features):
    """Train-only median nonzero Euclidean distance for an RBF kernel."""
    features = _matrix("features", features)
    if len(features) < 2:
        raise ValueError("at least two feature rows are required")
    norm = np.sum(features * features, axis=1)
    squared = np.maximum(norm[:, None] + norm[None, :] - 2.0 * features @ features.T, 0.0)
    upper = squared[np.triu_indices(len(features), 1)]
    positive = upper[upper > np.finfo(float).eps]
    if not len(positive):
        raise ValueError("RBF bandwidth is undefined when all feature rows are identical")
    return float(np.sqrt(np.median(positive)))


def rbf_kernel(features_a, features_b=None, *, length_scale):
    """PSD Euclidean RBF Gram/cross-kernel; bandwidth must be fit without held outcomes."""
    left = _matrix("features_a", features_a)
    right = left if features_b is None else _matrix(
        "features_b", features_b, cols=left.shape[1]
    )
    length_scale = _positive_finite("length_scale", length_scale)
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    kernel = np.exp(-0.5 * squared / (length_scale * length_scale))
    if features_b is None:
        kernel = 0.5 * (kernel + kernel.T)
        np.fill_diagonal(kernel, 1.0)
    return kernel


@dataclass(frozen=True)
class RegionalMeanModel:
    """Train-only kernel-ridge mean with exact intercept-aware LOO residuals."""

    kernel_name: str
    ridge: float
    intercept: np.ndarray
    alpha: np.ndarray
    loo_residuals: np.ndarray
    loo_mse: float

    @property
    def global_mean(self):
        """Compatibility name: the unpenalized fitted intercept vector."""
        return self.intercept

    def predict(self, cross_kernel):
        cross = _matrix("cross_kernel", cross_kernel, cols=len(self.alpha))
        return self.intercept + cross @ self.alpha

    def to_dict(self):
        return {
            "kernel_name": self.kernel_name,
            "ridge": self.ridge,
            "intercept": self.intercept.tolist(),
            "loo_mse": self.loo_mse,
        }


def _kernel_ridge_loo(residuals, kernel, ridge):
    residuals = _matrix("residuals", residuals)
    kernel = _symmetric("kernel", kernel, dimension=len(residuals))
    ridge = _positive_finite("ridge", ridge)
    A = kernel + np.eye(len(kernel)) * ridge
    inverse = np.linalg.solve(A, np.eye(len(A)))
    ones = np.ones(len(A))
    inverse_ones = inverse @ ones
    denominator = float(ones @ inverse_ones)
    if denominator <= 0.0:
        raise np.linalg.LinAlgError("kernel-ridge intercept system is not positive definite")
    intercept = (ones @ inverse @ residuals) / denominator
    centered = residuals - intercept
    projection = inverse - np.outer(inverse_ones, inverse_ones) / denominator
    alpha = projection @ residuals
    diagonal = np.diag(projection)
    if np.any(diagonal <= 0.0):
        raise np.linalg.LinAlgError("kernel-ridge LOO leverage is invalid")
    # K alpha + intercept = y - ridge*alpha; exact linear-smoother LOO residual is alpha/diag(P).
    loo_residuals = alpha / diagonal[:, None]
    loo_prediction = residuals - loo_residuals
    mse = float(np.mean((residuals - loo_prediction) ** 2))
    return RegionalMeanModel("", ridge, intercept, alpha, loo_residuals, mse)


def select_kernel_ridge_mean(residuals, kernels, *, ridge_grid):
    """Select a regional mean only by exact train LOO MSE."""
    residuals = _matrix("residuals", residuals)
    if not kernels:
        raise ValueError("kernels must not be empty")
    candidates = []
    for kernel_name in sorted(kernels):
        kernel = _symmetric(kernel_name, kernels[kernel_name], dimension=len(residuals))
        for ridge in ridge_grid:
            fitted = _kernel_ridge_loo(residuals, kernel, ridge)
            candidates.append(RegionalMeanModel(
                kernel_name, fitted.ridge, fitted.intercept, fitted.alpha,
                fitted.loo_residuals, fitted.loo_mse,
            ))
    if not candidates:
        raise ValueError("ridge_grid must contain at least one positive value")
    return min(candidates, key=lambda value: (value.loo_mse, value.kernel_name, value.ridge))


@dataclass(frozen=True)
class StructuredResidualCovarianceModel:
    """Train-fitted additive LMC component matrices in original channel units."""

    kind: str
    independent_covariance: np.ndarray
    semantic_covariance: np.ndarray
    graph_covariance: np.ndarray
    train_objective: float
    initial_objective: float
    steps: int
    pair_count: int
    statistical_floor: float
    semantic_weight: float | None = None
    graph_weight: float | None = None
    channel_rms_scale: np.ndarray | None = None
    block_reference_objective: float | None = None
    initial_model_reference_objective: float | None = None

    @property
    def channel_count(self):
        return self.independent_covariance.shape[0]

    def materialize(self, semantic_kernel, graph_kernel):
        semantic = _symmetric("semantic_kernel", semantic_kernel)
        graph = _symmetric("graph_kernel", graph_kernel, dimension=len(semantic))
        return (
            np.kron(np.eye(len(semantic)), self.independent_covariance)
            + np.kron(semantic, self.semantic_covariance)
            + np.kron(graph, self.graph_covariance)
        )

    def to_dict(self):
        if self.kind == "block":
            objective_type = "full_joint_gaussian_nll"
            objective_units = "original_channel_units"
            floor_units = "original_channel_variance"
            restored_floor = np.full(self.channel_count, self.statistical_floor)
        else:
            objective_type = "pairwise_composite_gaussian_nll"
            objective_units = "train_rms_standardized_channels"
            floor_units = "train_rms_standardized_variance"
            if self.channel_rms_scale is None:
                raise ValueError("structured model is missing its channel RMS scale")
            restored_floor = self.statistical_floor * self.channel_rms_scale ** 2
        return {
            "kind": self.kind,
            "independent_covariance": self.independent_covariance.tolist(),
            "semantic_covariance": self.semantic_covariance.tolist(),
            "graph_covariance": self.graph_covariance.tolist(),
            "train_objective_per_scalar": self.train_objective,
            "initial_objective_per_scalar": self.initial_objective,
            "block_reference_objective_per_scalar": self.block_reference_objective,
            "initial_model_reference_objective_per_scalar": self.initial_model_reference_objective,
            "train_objective_type": objective_type,
            "train_objective_units": objective_units,
            "steps": self.steps,
            "pair_count": self.pair_count,
            "statistical_floor": self.statistical_floor,
            "statistical_floor_units": floor_units,
            "statistical_floor_original_channel_diagonal": restored_floor.tolist(),
            "channel_rms_scale": self.channel_rms_scale.tolist()
            if self.channel_rms_scale is not None else None,
            "semantic_weight": self.semantic_weight,
            "graph_weight": self.graph_weight,
        }


def fit_block_model(residuals, *, shrinkage=0.05, relative_floor=1e-8):
    """Analytic Gaussian MLE for independent item blocks."""
    residuals = _matrix("residuals", residuals)
    if len(residuals) < 2:
        raise ValueError("at least two residual rows are required")
    shrinkage = float(shrinkage)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0,1]")
    covariance = residuals.T @ residuals / len(residuals)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    covariance, loading = _psd_floor(covariance, relative_floor)
    zeros = np.zeros_like(covariance)
    model = StructuredResidualCovarianceModel(
        "block", covariance, zeros, zeros, float("nan"), float("nan"), 0, 0, loading,
    )
    covariance_train = model.materialize(np.eye(len(residuals)), np.eye(len(residuals)))
    objective = gaussian_joint_nll(residuals, covariance_train).per_scalar
    return StructuredResidualCovarianceModel(
        "block", covariance, zeros, zeros, objective, objective, 0, 0, loading,
    )


def _factor_initial(covariance, fraction, floor):
    target = covariance * fraction + np.eye(len(covariance)) * floor
    return np.linalg.cholesky(target)


def _pair_indices(count, maximum, seed):
    left, right = np.triu_indices(count, 1)
    if not len(left):
        raise ValueError("pairwise fitting requires at least two residual rows")
    if maximum is not None and maximum < len(left):
        if maximum < 1:
            raise ValueError("max_pairs must be positive")
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(left), size=maximum, replace=False))
        left, right = left[selected], right[selected]
    return left, right


def _component_from_factor(factor, floor=0.0):
    value = torch.tril(factor)
    covariance = value @ value.mT
    if floor:
        covariance = covariance + torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        ) * floor
    return covariance


def _pairwise_objective(residuals, left, right, K_sem, K_graph, components):
    B0, B_sem, B_graph = components
    diagonal = B0 + B_sem + B_graph
    cross = (
        K_sem[left, right, None, None] * B_sem
        + K_graph[left, right, None, None] * B_graph
    )
    plus_covariance = diagonal + cross
    minus_covariance = diagonal - cross
    plus_factor, plus_info = torch.linalg.cholesky_ex(plus_covariance)
    minus_factor, minus_info = torch.linalg.cholesky_ex(minus_covariance)
    if torch.any(plus_info) or torch.any(minus_info):
        raise torch.linalg.LinAlgError("pairwise LMC covariance is not positive definite")
    scale = math.sqrt(0.5)
    plus = (residuals[left] + residuals[right]) * scale
    minus = (residuals[left] - residuals[right]) * scale

    def term(value, factor):
        solved = torch.cholesky_solve(value.unsqueeze(-1), factor).squeeze(-1)
        quadratic = torch.sum(value * solved, dim=-1)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)), dim=-1)
        return quadratic + logdet

    judge_count = residuals.shape[1]
    nll = 0.5 * (term(plus, plus_factor) + term(minus, minus_factor) + 2 * judge_count * LOG2PI)
    return torch.mean(nll) / (2.0 * judge_count)


def fit_lmc_model(
    residuals,
    semantic_kernel,
    graph_kernel,
    *,
    kind,
    steps=150,
    learning_rate=0.03,
    max_pairs=4096,
    seed=0,
    relative_floor=1e-6,
    initial_model=None,
):
    """Fit separable or additive-LMC components by deterministic pairwise likelihood."""
    residuals = _matrix("residuals", residuals)
    semantic = _symmetric("semantic_kernel", semantic_kernel, dimension=len(residuals))
    graph = _symmetric("graph_kernel", graph_kernel, dimension=len(residuals))
    if kind not in {"separable", "dense_lmc"}:
        raise ValueError("kind must be 'separable' or 'dense_lmc'")
    if steps < 1:
        raise ValueError("steps must be positive")
    learning_rate = _positive_finite("learning_rate", learning_rate)
    diagonal_tolerance = 1e-8
    if not np.allclose(np.diag(semantic), 1.0, atol=diagonal_tolerance) or not np.allclose(
        np.diag(graph), 1.0, atol=diagonal_tolerance
    ):
        raise ValueError("LMC item kernels must have unit diagonal")

    rms = np.sqrt(np.mean(residuals * residuals, axis=0))
    rms = np.where(rms > 1e-8, rms, 1.0)
    standardized = residuals / rms
    marginal = standardized.T @ standardized / len(standardized)
    marginal, _ = _psd_floor(marginal, 1e-8)
    judge_count = residuals.shape[1]
    statistical_floor = relative_floor * float(np.trace(marginal) / judge_count)
    left_np, right_np = _pair_indices(len(residuals), max_pairs, seed)

    dtype = torch.float64
    z = torch.tensor(standardized, dtype=dtype)
    K_sem = torch.tensor(semantic, dtype=dtype)
    K_graph = torch.tensor(graph, dtype=dtype)
    left = torch.tensor(left_np, dtype=torch.long)
    right = torch.tensor(right_np, dtype=torch.long)
    initial_reference_components = None
    if kind == "separable":
        parameters = [
            torch.nn.Parameter(torch.tensor(_factor_initial(marginal, 0.8, 0.0), dtype=dtype)),
            torch.nn.Parameter(torch.tensor(_factor_initial(marginal, 0.2, 0.0), dtype=dtype)),
            torch.nn.Parameter(torch.zeros(2, dtype=dtype)),
        ]
    elif initial_model is None:
        parameters = [
            torch.nn.Parameter(torch.tensor(_factor_initial(marginal, 0.8, 0.0), dtype=dtype)),
            torch.nn.Parameter(torch.tensor(_factor_initial(marginal, 0.1, 0.0), dtype=dtype)),
            torch.nn.Parameter(torch.tensor(_factor_initial(marginal, 0.1, 0.0), dtype=dtype)),
        ]
    else:
        if not isinstance(initial_model, StructuredResidualCovarianceModel):
            raise TypeError("initial_model must be a StructuredResidualCovarianceModel")
        inverse_unit = np.diag(1.0 / rms)

        initial_reference_components = tuple(
            torch.tensor(inverse_unit @ value @ inverse_unit, dtype=dtype)
            for value in (
                initial_model.independent_covariance,
                initial_model.semantic_covariance,
                initial_model.graph_covariance,
            )
        )

        def standardized_factor(value, *, subtract_floor=False, seed_if_near_zero=False):
            scaled = inverse_unit @ value @ inverse_unit
            if subtract_floor:
                scaled = scaled - np.eye(judge_count) * statistical_floor
            if seed_if_near_zero:
                marginal_scale = max(float(np.max(np.abs(marginal))), np.finfo(float).tiny)
                if float(np.max(np.abs(scaled))) <= 1e-8 * marginal_scale:
                    # A zero LMC factor has zero gradient under B=L L.T.  Preserve the inherited
                    # model as an exact saved candidate below, but start Adam from material mass.
                    scaled = scaled + 0.05 * marginal
            scaled, _ = _psd_floor(scaled, 1e-10)
            return np.linalg.cholesky(scaled)

        parameters = [
            torch.nn.Parameter(torch.tensor(
                standardized_factor(initial_model.independent_covariance, subtract_floor=True), dtype=dtype
            )),
            torch.nn.Parameter(torch.tensor(
                standardized_factor(initial_model.semantic_covariance, seed_if_near_zero=True),
                dtype=dtype,
            )),
            torch.nn.Parameter(torch.tensor(
                standardized_factor(initial_model.graph_covariance, seed_if_near_zero=True),
                dtype=dtype,
            )),
        ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def components():
        B0 = _component_from_factor(parameters[0], statistical_floor)
        if kind == "separable":
            shared = _component_from_factor(parameters[1])
            weights = torch.softmax(parameters[2], dim=0)
            return B0, weights[0] * shared, weights[1] * shared
        return B0, _component_from_factor(parameters[1]), _component_from_factor(parameters[2])

    def current_weights():
        if kind != "separable":
            return None
        return torch.softmax(parameters[2], dim=0).detach().clone()

    def clone_components(values):
        return tuple(value.detach().clone() for value in values)

    # The structured families contain the independent-item model.  Keep that exact submodel as a
    # deterministic fallback so a weak Adam iterate cannot manufacture evidence against structure.
    block_components = (
        torch.tensor(marginal, dtype=dtype) + torch.eye(judge_count, dtype=dtype) * statistical_floor,
        torch.zeros((judge_count, judge_count), dtype=dtype),
        torch.zeros((judge_count, judge_count), dtype=dtype),
    )
    with torch.no_grad():
        initial_components = components()
        initial = float(_pairwise_objective(
            z, left, right, K_sem, K_graph, initial_components
        ))
        block_reference = float(_pairwise_objective(
            z, left, right, K_sem, K_graph, block_components
        ))
        initial_model_reference = (
            float(_pairwise_objective(
                z, left, right, K_sem, K_graph, initial_reference_components
            ))
            if initial_reference_components is not None else None
        )
    if initial < block_reference:
        best_loss = initial
        best_components = clone_components(initial_components)
        best_weights = current_weights()
    else:
        best_loss = block_reference
        best_components = clone_components(block_components)
        best_weights = torch.tensor([0.5, 0.5], dtype=dtype) if kind == "separable" else None
    if initial_model_reference is not None and initial_model_reference < best_loss:
        best_loss = initial_model_reference
        best_components = clone_components(initial_reference_components)
        best_weights = None
    for _ in range(steps):
        optimizer.zero_grad()
        candidate_components = components()
        loss = _pairwise_objective(z, left, right, K_sem, K_graph, candidate_components)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite pairwise LMC objective")
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_components = clone_components(candidate_components)
            best_weights = current_weights()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()
    with torch.no_grad():
        last_components = components()
        last_loss = float(_pairwise_objective(
            z, left, right, K_sem, K_graph, last_components
        ))
    if last_loss < best_loss:
        best_loss = last_loss
        best_components = clone_components(last_components)
        best_weights = current_weights()
    B0_t, B_sem_t, B_graph_t = best_components
    final = best_loss
    weights = best_weights.cpu().numpy() if best_weights is not None else None
    unit = np.diag(rms)

    def restore(value):
        array = value.detach().cpu().numpy()
        return unit @ array @ unit

    return StructuredResidualCovarianceModel(
        kind,
        restore(B0_t),
        restore(B_sem_t),
        restore(B_graph_t),
        final,
        initial,
        int(steps),
        int(len(left)),
        float(statistical_floor),
        float(weights[0]) if weights is not None else None,
        float(weights[1]) if weights is not None else None,
        rms.copy(),
        block_reference,
        initial_model_reference,
    )


@dataclass(frozen=True)
class GaussianJointNLL:
    total: float
    per_scalar: float


def gaussian_joint_nll(residuals, covariance):
    """Full joint Gaussian NLL for one held residual field."""
    residuals = _matrix("residuals", residuals)
    covariance = _symmetric(
        "covariance", covariance, dimension=residuals.shape[0] * residuals.shape[1]
    )
    factor = np.linalg.cholesky(covariance)
    flat = residuals.reshape(-1)
    whitened = np.linalg.solve(factor, flat)
    logdet = 2.0 * np.sum(np.log(np.diag(factor)))
    total = 0.5 * (float(whitened @ whitened) + float(logdet) + len(flat) * LOG2PI)
    return GaussianJointNLL(total, total / len(flat))


@dataclass(frozen=True)
class OffBlockDiagnostics:
    relative_off_item_frobenius_mass: float
    maximum_whitened_off_block_spectral_norm: float
    maximum_coupling_item_pair: tuple[int, int] | None
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float

    def to_dict(self):
        return {
            "relative_off_item_frobenius_mass": self.relative_off_item_frobenius_mass,
            "maximum_whitened_off_block_spectral_norm": self.maximum_whitened_off_block_spectral_norm,
            "maximum_coupling_item_pair": list(self.maximum_coupling_item_pair)
            if self.maximum_coupling_item_pair is not None else None,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "condition_number": self.condition_number,
        }


def off_block_diagnostics(covariance, block_size):
    """Materialized covariance diagnostics required by the streaming independence contract."""
    covariance = _symmetric("covariance", covariance)
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if len(covariance) % block_size:
        raise ValueError("covariance dimension must be divisible by block_size")
    count = len(covariance) // block_size
    block_diagonal = np.zeros_like(covariance)
    factors = []
    for i in range(count):
        section = slice(i * block_size, (i + 1) * block_size)
        block = covariance[section, section]
        block_diagonal[section, section] = block
        factors.append(np.linalg.cholesky(block))
    off = covariance - block_diagonal
    total_norm = np.linalg.norm(covariance, ord="fro")
    relative = float(np.linalg.norm(off, ord="fro") / total_norm) if total_norm else 0.0
    maximum, pair = 0.0, None
    for i in range(count):
        rows = slice(i * block_size, (i + 1) * block_size)
        for j in range(i + 1, count):
            cols = slice(j * block_size, (j + 1) * block_size)
            left_whitened = np.linalg.solve(factors[i], covariance[rows, cols])
            whitened = np.linalg.solve(factors[j], left_whitened.T).T
            value = float(np.linalg.norm(whitened, ord=2))
            if value > maximum:
                maximum, pair = value, (i, j)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return OffBlockDiagnostics(
        relative,
        maximum,
        pair,
        float(eigenvalues[0]),
        float(eigenvalues[-1]),
        float(eigenvalues[-1] / eigenvalues[0]),
    )


def _loading_to_dict(diagnostics):
    fields = (
        "matrix_scale",
        "minimum_eigenvalue",
        "target_minimum_eigenvalue",
        "diagonal_loading",
        "relative_diagonal_loading",
        "was_loaded",
        "relative_eigenvalue_floor",
        "negative_eigenvalue_tolerance",
        "maximum_relative_loading",
        "source",
    )
    out = {}
    for field in fields:
        value = getattr(diagnostics, field)
        if torch.is_tensor(value):
            value = value.detach().cpu()
            value = value.item() if value.ndim == 0 else value.tolist()
        out[field] = value
    return out


@dataclass(frozen=True)
class ConditionedItemBatch:
    state_mean: np.ndarray
    marginal_covariances: np.ndarray
    full_covariance: np.ndarray
    loading_diagnostics: dict
    prior_loading_diagnostics: dict


def condition_item_batch(
    prior_covariance,
    conditional_design,
    innovation,
    conditional_covariance,
    *,
    maximum_relative_loading=1e-3,
):
    """Joint prior-centered information-QR update for a coupled held item batch."""
    prior = _symmetric("prior_covariance", prior_covariance)
    design = _matrix("conditional_design", conditional_design, cols=len(prior))
    innovation = _matrix("innovation", innovation, cols=design.shape[0])
    item_count = len(innovation)
    covariance = _symmetric(
        "conditional_covariance",
        conditional_covariance,
        dimension=item_count * design.shape[0],
    )
    dtype = torch.float64
    P0 = torch.tensor(prior, dtype=dtype)
    J0 = torch.tensor(design, dtype=dtype)
    residual = torch.tensor(innovation.reshape(-1), dtype=dtype)
    Rc = torch.tensor(covariance, dtype=dtype)
    prior_whitener = prepare_noise_whitener_torch(
        P0,
        maximum_relative_loading=maximum_relative_loading,
        name="prior covariance",
    )
    root0 = precision_root_from_covariance_torch(
        prior_whitener.covariance,
        maximum_relative_loading=maximum_relative_loading,
    )
    identity_items = torch.eye(item_count, dtype=dtype)
    root = torch.kron(identity_items, root0)
    J = torch.kron(identity_items, J0)
    whitener = prepare_noise_whitener_torch(
        Rc,
        maximum_relative_loading=maximum_relative_loading,
        name="structured conditional residual covariance",
    )
    state = SquareRootInformationStateTorch(
        root, torch.zeros(item_count * len(prior), dtype=dtype)
    )
    _, update = state.update_noise_block(J, residual, whitener)
    posterior_root = update.precision_root
    identity_state = torch.eye(posterior_root.shape[-1], dtype=dtype)
    inverse_root = torch.linalg.solve_triangular(posterior_root, identity_state, upper=True)
    posterior_covariance = inverse_root @ inverse_root.mT
    state_mean = update.solution.reshape(item_count, len(prior))
    marginals = torch.stack([
        posterior_covariance[
            i * len(prior):(i + 1) * len(prior),
            i * len(prior):(i + 1) * len(prior),
        ]
        for i in range(item_count)
    ])
    return ConditionedItemBatch(
        state_mean.detach().cpu().numpy(),
        marginals.detach().cpu().numpy(),
        posterior_covariance.detach().cpu().numpy(),
        _loading_to_dict(whitener.diagnostics),
        _loading_to_dict(prior_whitener.diagnostics),
    )
