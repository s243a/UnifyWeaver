#!/usr/bin/env python3
"""PSD covariance-misspecification and posterior-sensitivity primitives.

The primary path preserves the independent block model's within-item covariance while varying only the
cross-item correlation learned by a structured model.  Arrays use item-major order throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


LOG2PI = math.log(2.0 * math.pi)


def _matrix(name, value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
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


def _unit_interval(name, value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return value


def _positive(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def shrink_channel_covariance(covariance, beta):
    """Convexly shrink channel interactions toward their diagonal while preserving PSD."""
    covariance = _symmetric("covariance", covariance)
    beta = _unit_interval("beta", beta)
    value = (1.0 - beta) * covariance + beta * np.diag(np.diag(covariance))
    return 0.5 * (value + value.T)


def materialize_channel_shrunk(model, semantic_kernel, graph_kernel, beta):
    """Materialize an LMC after PSD shrinkage of structured channel interactions."""
    semantic = _symmetric("semantic_kernel", semantic_kernel)
    graph = _symmetric("graph_kernel", graph_kernel, dimension=len(semantic))
    beta = _unit_interval("beta", beta)
    B_sem = shrink_channel_covariance(model.semantic_covariance, beta)
    B_graph = shrink_channel_covariance(model.graph_covariance, beta)
    return (
        np.kron(np.eye(len(semantic)), model.independent_covariance)
        + np.kron(semantic, B_sem)
        + np.kron(graph, B_graph)
    )


def direct_covariance_blend(block_covariance, structured_covariance, alpha):
    """Secondary raw-covariance convex blend; this does not preserve item marginals."""
    block = _symmetric("block_covariance", block_covariance)
    structured = _symmetric(
        "structured_covariance", structured_covariance, dimension=len(block)
    )
    alpha = _unit_interval("alpha", alpha)
    return (1.0 - alpha) * block + alpha * structured


def expected_rbf_kernel_gaussian(features, *, length_scale, input_std=0.0, normalize=True):
    """Expected RBF between isotropic-Gaussian uncertain feature vectors.

    Row ``i`` represents ``X_i ~ Normal(features[i], input_std[i]^2 I)``.  The raw result is
    ``E[k(X_i,X_j)]`` for independent draws.  Diagonal normalization converts it to a correlation kernel.
    With a common standard deviation, that normalized kernel is exactly an RBF with
    ``ell_effective^2 = ell^2 + 2*input_std^2``.
    """
    features = _matrix("features", features)
    length_scale = _positive("length_scale", length_scale)
    std = np.asarray(input_std, dtype=float)
    if std.ndim == 0:
        std = np.full(len(features), float(std))
    if std.shape != (len(features),) or not np.isfinite(std).all() or np.any(std < 0.0):
        raise ValueError("input_std must be nonnegative scalar or one value per feature row")
    squared_norm = np.sum(features * features, axis=1)
    squared_distance = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * features @ features.T,
        0.0,
    )
    variance = std[:, None] ** 2 + std[None, :] ** 2
    denominator = length_scale * length_scale + variance
    dimension = features.shape[1]
    amplitude = (length_scale * length_scale / denominator) ** (0.5 * dimension)
    kernel = amplitude * np.exp(-0.5 * squared_distance / denominator)
    kernel = 0.5 * (kernel + kernel.T)
    if normalize:
        diagonal = np.sqrt(np.diag(kernel))
        kernel = kernel / diagonal[:, None] / diagonal[None, :]
        kernel = 0.5 * (kernel + kernel.T)
        np.fill_diagonal(kernel, 1.0)
    return kernel


def centered_linear_kernel(features_a, features_b=None, *, center=0.5, normalize=False):
    """PSD linear Gram after centering bounded features around a neutral value.

    For scalar ``mu`` values this is ``K[i,j] = (mu[i]-center) * (mu[j]-center)``.
    Unlike an RBF, this kernel distinguishes values on the same versus opposite sides of the midpoint;
    subtracting a common center inside an RBF distance would instead cancel exactly.

    Optional normalization produces a correlation-like Gram only when every centered row has nonzero norm.
    The raw Gram is the useful form when distance from the neutral midpoint should affect amplitude.
    """
    left = np.asarray(features_a, dtype=float)
    if left.ndim == 1:
        left = left[:, None]
    if left.ndim != 2 or not np.isfinite(left).all():
        raise ValueError("features_a must be a finite vector or matrix")
    right = left if features_b is None else np.asarray(features_b, dtype=float)
    if right.ndim == 1:
        right = right[:, None]
    if right.ndim != 2 or not np.isfinite(right).all() or right.shape[1] != left.shape[1]:
        raise ValueError("features_b must be a finite vector or matrix with matching columns")
    center = np.asarray(center, dtype=float)
    if not np.isfinite(center).all():
        raise ValueError("center must be finite")
    if center.ndim > 1 or (center.ndim == 1 and center.shape != (left.shape[1],)):
        raise ValueError("center must be scalar or one value per feature column")
    centered_left = left - center
    centered_right = right - center
    kernel = centered_left @ centered_right.T
    if features_b is None:
        kernel = 0.5 * (kernel + kernel.T)
    if normalize:
        left_norms = np.linalg.norm(centered_left, axis=1)
        right_norms = np.linalg.norm(centered_right, axis=1)
        if np.any(left_norms == 0.0) or np.any(right_norms == 0.0):
            raise ValueError("cannot normalize a centered feature row with zero norm")
        kernel = kernel / left_norms[:, None] / right_norms[None, :]
        if features_b is None:
            kernel = 0.5 * (kernel + kernel.T)
            np.fill_diagonal(kernel, 1.0)
    return kernel


def gaussian_input_std_ratio_for_length_multiplier(multiplier):
    """Map a broadened normalized-RBF multiplier to common Gaussian input std / base length."""
    multiplier = _positive("multiplier", multiplier)
    if multiplier < 1.0:
        return None
    return math.sqrt((multiplier * multiplier - 1.0) / 2.0)


@dataclass(frozen=True)
class CorrelationPath:
    """Eigen representation of a marginal-matched block-to-structured correlation path."""

    block_item_covariance: np.ndarray
    structured_item_covariance: np.ndarray
    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    block_size: int

    @property
    def item_count(self):
        return len(self.correlation) // self.block_size

    def mixture_eigenvalues(self, alpha):
        alpha = _unit_interval("alpha", alpha)
        values = 1.0 + alpha * (self.eigenvalues - 1.0)
        if np.min(values) <= 0.0:
            raise np.linalg.LinAlgError("correlation path is not positive definite")
        return values

    def covariance(self, alpha):
        values = self.mixture_eigenvalues(alpha)
        normalized = (self.eigenvectors * values) @ self.eigenvectors.T
        factor = np.kron(
            np.eye(self.item_count), np.linalg.cholesky(self.block_item_covariance)
        )
        covariance = factor @ normalized @ factor.T
        return 0.5 * (covariance + covariance.T)

    def nll_per_scalar(self, residuals, alpha):
        residuals = _matrix("residuals", residuals)
        if residuals.shape != (self.item_count, self.block_size):
            raise ValueError("residuals do not match correlation-path geometry")
        factor = np.linalg.cholesky(self.block_item_covariance)
        whitened = np.linalg.solve(factor, residuals.T).T.reshape(-1)
        coordinates = self.eigenvectors.T @ whitened
        values = self.mixture_eigenvalues(alpha)
        logdet_block = 2.0 * np.sum(np.log(np.diag(factor)))
        total = 0.5 * (
            float(np.sum(coordinates * coordinates / values))
            + self.item_count * float(logdet_block)
            + float(np.sum(np.log(values)))
            + len(coordinates) * LOG2PI
        )
        return total / len(coordinates)


def build_correlation_path(block_item_covariance, structured_covariance, block_size):
    """Match structured correlations to fixed block marginals and eigendecompose the path."""
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    block = _symmetric(
        "block_item_covariance", block_item_covariance, dimension=block_size
    )
    np.linalg.cholesky(block)
    structured = _symmetric("structured_covariance", structured_covariance)
    if len(structured) % block_size:
        raise ValueError("structured covariance dimension must be divisible by block_size")
    item_count = len(structured) // block_size
    diagonal_blocks = np.stack([
        structured[
            item * block_size:(item + 1) * block_size,
            item * block_size:(item + 1) * block_size,
        ]
        for item in range(item_count)
    ])
    structured_item = diagonal_blocks.mean(axis=0)
    scale = max(float(np.max(np.abs(structured_item))), 1.0)
    if not np.allclose(diagonal_blocks, structured_item, rtol=0.0, atol=1e-9 * scale):
        raise ValueError("structured covariance must have a constant within-item block")
    factor = np.linalg.cholesky(structured_item)
    inverse_factor = np.linalg.solve(factor, np.eye(block_size))
    whitener = np.kron(np.eye(item_count), inverse_factor)
    correlation = whitener @ structured @ whitener.T
    correlation = 0.5 * (correlation + correlation.T)
    for item in range(item_count):
        section = slice(item * block_size, (item + 1) * block_size)
        correlation[section, section] = np.eye(block_size)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues[0] <= 0.0:
        raise np.linalg.LinAlgError("normalized structured correlation is not positive definite")
    return CorrelationPath(
        block, structured_item, correlation, eigenvalues, eigenvectors, block_size
    )


@dataclass(frozen=True)
class PosteriorSensitivity:
    state_mean: np.ndarray
    state_covariance: np.ndarray
    directional_derivative: np.ndarray
    mean_item_derivative_norm: float
    covariance_direction_relative_norm: float
    relative_state_sensitivity: float

    def to_dict(self):
        return {
            "mean_item_derivative_norm": self.mean_item_derivative_norm,
            "covariance_direction_relative_norm": self.covariance_direction_relative_norm,
            "relative_state_sensitivity": self.relative_state_sensitivity,
        }


def _dense_posterior_system(prior_covariance, design, innovation, covariance):
    prior = _symmetric("prior_covariance", prior_covariance)
    design = _matrix("design", design)
    innovation = _matrix("innovation", innovation)
    if design.shape[1] != len(prior) or innovation.shape[1] != design.shape[0]:
        raise ValueError("design/innovation dimensions do not match the prior")
    item_count = len(innovation)
    noise = _symmetric(
        "covariance", covariance, dimension=item_count * design.shape[0]
    )
    prior_batch = np.kron(np.eye(item_count), prior)
    design_batch = np.kron(np.eye(item_count), design)
    residual = innovation.reshape(-1)
    weighted_design = np.linalg.solve(noise, design_batch)
    weighted_residual = np.linalg.solve(noise, residual)
    precision = np.linalg.solve(prior_batch, np.eye(len(prior_batch)))
    system = precision + design_batch.T @ weighted_design
    rhs = design_batch.T @ weighted_residual
    state = np.linalg.solve(system, rhs)
    posterior_covariance = np.linalg.solve(system, np.eye(len(system)))
    return state, posterior_covariance, system, design_batch, residual, noise


def posterior_mean_dense(prior_covariance, design, innovation, covariance):
    """Dense reference posterior for a prior-centered batch."""
    state, posterior, *_ = _dense_posterior_system(
        prior_covariance, design, innovation, covariance
    )
    return state.reshape(len(innovation), -1), posterior


def directional_posterior_sensitivity(
    prior_covariance,
    design,
    innovation,
    covariance,
    covariance_direction,
):
    """Analytic first derivative of posterior mean along a symmetric covariance direction."""
    state, posterior, system, design_batch, residual, noise = _dense_posterior_system(
        prior_covariance, design, innovation, covariance
    )
    direction = _symmetric(
        "covariance_direction", covariance_direction, dimension=len(noise)
    )
    model_residual = residual - design_batch @ state
    right = np.linalg.solve(noise, model_residual)
    twice_weighted = np.linalg.solve(noise, direction @ right)
    derivative = -np.linalg.solve(system, design_batch.T @ twice_weighted)
    item_count = len(innovation)
    state_size = len(prior_covariance)
    derivative_rows = derivative.reshape(item_count, state_size)
    covariance_relative = float(
        np.linalg.norm(direction, ord="fro") / np.linalg.norm(noise, ord="fro")
    )
    state_scale = max(
        float(np.linalg.norm(state)),
        math.sqrt(max(float(np.trace(posterior)), np.finfo(float).tiny)),
    )
    relative = float(np.linalg.norm(derivative) / state_scale)
    if covariance_relative > 0.0:
        relative /= covariance_relative
    return PosteriorSensitivity(
        state.reshape(item_count, state_size),
        posterior,
        derivative_rows,
        float(np.mean(np.linalg.norm(derivative_rows, axis=1))),
        covariance_relative,
        relative,
    )


def select_nested_candidate(records, *, tolerance=1e-3, minimum_positive_folds=2):
    """Conservatively select a covariance candidate from repeated inner node partitions."""
    if not records:
        raise ValueError("records must not be empty")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative and finite")
    if (
        not isinstance(minimum_positive_folds, (int, np.integer))
        or isinstance(minimum_positive_folds, (bool, np.bool_))
        or minimum_positive_folds < 1
    ):
        raise ValueError("minimum_positive_folds must be a positive integer")
    grouped = {}
    block_by_fold = {}
    for row in records:
        fold = int(row["fold"])
        alpha = _unit_interval("alpha", row["alpha"])
        nll = float(row["nll_per_scalar"])
        if not np.isfinite(nll):
            raise ValueError("candidate NLL must be finite")
        if alpha == 0.0:
            if fold in block_by_fold and not math.isclose(
                block_by_fold[fold], nll, rel_tol=0.0, abs_tol=64 * np.finfo(float).eps
            ):
                raise ValueError("equivalent alpha=0 records disagree within a fold")
            block_by_fold[fold] = nll
            continue
        key = (
            alpha,
            _positive("semantic_multiplier", row["semantic_multiplier"]),
            _positive("graph_multiplier", row["graph_multiplier"]),
            _unit_interval("beta", row["beta"]),
        )
        if fold in grouped.setdefault(key, {}):
            raise ValueError("candidate is duplicated within a fold")
        grouped[key][fold] = nll
    folds = sorted(block_by_fold)
    if len(folds) < minimum_positive_folds:
        raise ValueError("insufficient block folds for conservative selection")
    summaries = []
    for key, values in grouped.items():
        if sorted(values) != folds:
            raise ValueError("every nonzero candidate must be scored on every fold")
        gains = np.asarray([block_by_fold[fold] - values[fold] for fold in folds])
        summaries.append({
            "alpha": key[0],
            "semantic_multiplier": key[1],
            "graph_multiplier": key[2],
            "beta": key[3],
            "macro_nll_per_scalar": float(np.mean([values[fold] for fold in folds])),
            "macro_gain_vs_block": float(gains.mean()),
            "positive_folds": int(np.sum(gains > 0.0)),
            "fold_gains": gains.tolist(),
        })
    eligible = [
        row for row in summaries
        if row["macro_gain_vs_block"] > 0.0
        and row["positive_folds"] >= minimum_positive_folds
    ]
    block = {
        "alpha": 0.0,
        "semantic_multiplier": 1.0,
        "graph_multiplier": 1.0,
        "beta": 1.0,
        "macro_nll_per_scalar": float(np.mean(list(block_by_fold.values()))),
        "macro_gain_vs_block": 0.0,
        "positive_folds": 0,
        "fold_gains": [0.0 for _ in folds],
    }
    if not eligible:
        return block, [block] + summaries
    best = min(row["macro_nll_per_scalar"] for row in eligible)
    near = [row for row in eligible if row["macro_nll_per_scalar"] <= best + tolerance]
    selected = min(
        near,
        key=lambda row: (
            row["alpha"],
            -row["beta"],
            abs(math.log(row["semantic_multiplier"]))
            + abs(math.log(row["graph_multiplier"])),
            row["semantic_multiplier"],
            row["graph_multiplier"],
        ),
    )
    return selected, [block] + summaries


def maximum_eligible_macro_gain(candidate_summaries, *, minimum_positive_folds=2):
    """Family-wise statistic for the same eligibility rule used by nested selection."""
    if not candidate_summaries:
        raise ValueError("candidate_summaries must not be empty")
    if (
        not isinstance(minimum_positive_folds, (int, np.integer))
        or isinstance(minimum_positive_folds, (bool, np.bool_))
        or minimum_positive_folds < 1
    ):
        raise ValueError("minimum_positive_folds must be a positive integer")
    eligible_gains = []
    for row in candidate_summaries:
        alpha = _unit_interval("alpha", row["alpha"])
        gain = float(row["macro_gain_vs_block"])
        positive_folds = int(row["positive_folds"])
        if not np.isfinite(gain):
            raise ValueError("candidate macro gain must be finite")
        if alpha > 0.0 and gain > 0.0 and positive_folds >= minimum_positive_folds:
            eligible_gains.append(gain)
    return float(max(eligible_gains, default=0.0))


def finite_null_maximum_threshold(null_maximum_gains, *, confidence=0.95):
    """Conservative finite-simulation upper quantile for a family-wise null statistic."""
    values = np.asarray(null_maximum_gains, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("null_maximum_gains must be a non-empty finite vector")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if np.any(values < 0.0):
        raise ValueError("null family-wise maximum gains must be nonnegative")
    rank = min(int(math.ceil(confidence * (len(values) + 1))), len(values))
    threshold = float(np.sort(values)[rank - 1])
    return threshold, rank


def select_nested_candidate_null_calibrated(
    records,
    null_maximum_gains,
    *,
    confidence=0.95,
    tolerance=1e-3,
    minimum_positive_folds=2,
):
    """Versioned v2 selector: v1 tie-break gated by a family-wise null maximum.

    ``null_maximum_gains`` must come from repeating the complete candidate search under the intended null.
    This helper deliberately accepts precomputed maxima so callers can use a full-procedure simulator rather
    than silently substituting a conditional fixed-path null.
    """
    v1_selected, summaries = select_nested_candidate(
        records,
        tolerance=tolerance,
        minimum_positive_folds=minimum_positive_folds,
    )
    observed = maximum_eligible_macro_gain(
        summaries, minimum_positive_folds=minimum_positive_folds
    )
    threshold, rank = finite_null_maximum_threshold(
        null_maximum_gains, confidence=confidence
    )
    comparison_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(observed), abs(threshold)
    )
    reject = bool(observed > threshold + comparison_tolerance)
    selected = dict(v1_selected if reject else summaries[0])
    selected["selection_version"] = "v2_familywise_null_calibrated"
    selected["v1_selected_alpha"] = float(v1_selected["alpha"])
    selected["familywise_null_rejected"] = reject
    calibration = {
        "selection_version": "v2_familywise_null_calibrated",
        "confidence": float(confidence),
        "null_draws": int(len(null_maximum_gains)),
        "order_statistic_rank_one_based": int(rank),
        "familywise_macro_gain_threshold": threshold,
        "observed_maximum_eligible_macro_gain": observed,
        "strict_comparison_absolute_tolerance": comparison_tolerance,
        "strictly_exceeds_threshold": reject,
        "v1_selected_candidate": dict(v1_selected),
    }
    return selected, summaries, calibration


def gaussian_kl(mean_p, covariance_p, mean_q, covariance_q):
    """KL[N_p || N_q] for one finite-dimensional Gaussian pair."""
    mean_p = np.asarray(mean_p, dtype=float)
    mean_q = np.asarray(mean_q, dtype=float)
    if mean_p.ndim != 1 or mean_q.shape != mean_p.shape:
        raise ValueError("means must be aligned vectors")
    P = _symmetric("covariance_p", covariance_p, dimension=len(mean_p))
    Q = _symmetric("covariance_q", covariance_q, dimension=len(mean_p))
    delta = mean_q - mean_p
    factor_p = np.linalg.cholesky(P)
    factor_q = np.linalg.cholesky(Q)
    logdet_p = 2.0 * np.sum(np.log(np.diag(factor_p)))
    logdet_q = 2.0 * np.sum(np.log(np.diag(factor_q)))
    return 0.5 * (
        float(np.trace(np.linalg.solve(Q, P)))
        + float(delta @ np.linalg.solve(Q, delta))
        - len(delta)
        + float(logdet_q - logdet_p)
    )


def mean_marginal_symmetric_kl(mean_a, covariance_a, mean_b, covariance_b):
    """Mean symmetric KL across aligned per-item marginal Gaussians."""
    mean_a = _matrix("mean_a", mean_a)
    mean_b = _matrix("mean_b", mean_b)
    covariance_a = np.asarray(covariance_a, dtype=float)
    covariance_b = np.asarray(covariance_b, dtype=float)
    if mean_b.shape != mean_a.shape:
        raise ValueError("means must be aligned")
    expected = (len(mean_a), mean_a.shape[1], mean_a.shape[1])
    if covariance_a.shape != expected or covariance_b.shape != expected:
        raise ValueError("marginal covariance arrays have the wrong shape")
    values = []
    for ma, Pa, mb, Pb in zip(mean_a, covariance_a, mean_b, covariance_b):
        values.append(0.5 * (
            gaussian_kl(ma, Pa, mb, Pb) + gaussian_kl(mb, Pb, ma, Pa)
        ))
    return float(np.mean(values))


def whitened_covariance_error(reference, candidate):
    """Spectral norm of a covariance perturbation in reference-whitened coordinates."""
    reference = _symmetric("reference", reference)
    candidate = _symmetric("candidate", candidate, dimension=len(reference))
    factor = np.linalg.cholesky(reference)
    left = np.linalg.solve(factor, candidate - reference)
    whitened = np.linalg.solve(factor, left.T).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues = np.linalg.eigvalsh(whitened)
    return float(np.max(np.abs(eigenvalues), initial=0.0))
