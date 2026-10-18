#!/usr/bin/env python3
"""Legacy v1 synthetic controls for nested covariance-sensitivity selection.

The controls use deterministic item geometry and known PSD correlation endpoints.  Outcome-blind kernel
systems and correlation eigensystems are cached, while every replicate refits the regional KRR coefficients,
selects its mean model, estimates the block marginal, and runs the frozen three-partition conservative
selector.  No real-data LMC optimization is used here; that separation is deliberate.

Important: the first v1 output's post-KRR "known truth" recovery denominator is invalid because the scored
residual is ``e_H-W e_T``, not ``e_H``.  The corrected, versioned runner separates end-to-end harm from an
oracle-mean/known-B power mechanism; see ``run_covariance_sensitivity_synthetic_v2.py``.  This file remains to
reproduce the v1 selection failure, not as a source of recovery claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covariance_sensitivity import select_nested_candidate
from structured_residual_covariance import gaussian_joint_nll, median_rbf_bandwidth, rbf_kernel


CHANNELS = 4
STATE_DIMENSION = 2
ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
MULTIPLIERS = (0.5, 1.0, 2.0)
BETAS = (0.0, 0.5, 1.0)
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SCENARIO_ORDER = (
    "block_null",
    "regional_mean_only",
    "wrong_item_geometry",
    "in_family_coupling_0.04",
    "in_family_coupling_0.10",
    "in_family_coupling_0.20",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("summary input must be a non-empty finite vector")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _ratio(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return float(numerator) / denominator


@dataclass(frozen=True)
class Scenario:
    name: str
    candidate_coupling: float
    true_coupling: float
    regional_mean: bool = False
    wrong_geometry: bool = False


SCENARIOS = (
    Scenario("block_null", 0.20, 0.0),
    Scenario("regional_mean_only", 0.20, 0.0, regional_mean=True),
    Scenario("wrong_item_geometry", 0.20, 0.20, wrong_geometry=True),
    # The common 0.20 endpoint makes 0.04 and 0.10 exactly alpha=0.20 and alpha=0.50,
    # respectively, so those scenarios contain both under- and over-correlated candidates.
    Scenario("in_family_coupling_0.04", 0.20, 0.04),
    Scenario("in_family_coupling_0.10", 0.20, 0.10),
    Scenario("in_family_coupling_0.20", 0.20, 0.20),
)


@dataclass(frozen=True)
class CachedCorrelation:
    """Outcome-blind eigensystem for one fixed marginal-preserving endpoint."""

    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def mixture_eigenvalues(self, alpha):
        alpha = float(alpha)
        values = 1.0 + alpha * (self.eigenvalues - 1.0)
        if values[0] <= 0.0:
            raise np.linalg.LinAlgError("synthetic correlation path is not positive definite")
        return values

    def mixture(self, alpha):
        alpha = float(alpha)
        return (1.0 - alpha) * np.eye(len(self.correlation)) + alpha * self.correlation


class NLLScorer:
    """One block whitening reused across the complete endpoint/alpha grid."""

    def __init__(self, residuals, block_covariance):
        self.residuals = np.asarray(residuals, dtype=float)
        self.block_covariance = np.asarray(block_covariance, dtype=float)
        factor = np.linalg.cholesky(self.block_covariance)
        self.whitened = np.linalg.solve(factor, self.residuals.T).T.reshape(-1)
        self.block_logdet = 2.0 * np.sum(np.log(np.diag(factor)))
        self.constant = len(self.whitened) * math.log(2.0 * math.pi)

    @property
    def block_nll(self):
        total = 0.5 * (
            float(self.whitened @ self.whitened)
            + len(self.residuals) * float(self.block_logdet)
            + self.constant
        )
        return total / len(self.whitened)

    def score_path(self, endpoint):
        squared = (endpoint.eigenvectors.T @ self.whitened) ** 2
        output = {}
        for alpha in ALPHAS[1:]:
            values = endpoint.mixture_eigenvalues(alpha)
            total = 0.5 * (
                float(np.sum(squared / values))
                + len(self.residuals) * float(self.block_logdet)
                + float(np.sum(np.log(values)))
                + self.constant
            )
            output[alpha] = total / len(self.whitened)
        return output


@dataclass(frozen=True)
class CachedKRRCandidate:
    kernel_name: str
    ridge: float
    intercept_weights: np.ndarray
    projection: np.ndarray
    loo_diagonal: np.ndarray
    cross_kernel: np.ndarray


@dataclass(frozen=True)
class RegionalFit:
    kernel_name: str
    ridge: float
    intercept: np.ndarray
    alpha: np.ndarray
    loo_residuals: np.ndarray
    loo_mse: float
    prediction: np.ndarray


class KRRCache:
    """Exact intercept-aware KRR refits with only outcome-blind matrix systems cached."""

    def __init__(self, train_kernels, cross_kernels, ridge_grid=RIDGES):
        if set(train_kernels) != set(cross_kernels):
            raise ValueError("train and cross KRR kernels must have identical names")
        candidates = []
        for name in sorted(train_kernels):
            kernel = np.asarray(train_kernels[name], dtype=float)
            cross = np.asarray(cross_kernels[name], dtype=float)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError("KRR train kernels must be square")
            if cross.ndim != 2 or cross.shape[1] != len(kernel):
                raise ValueError("KRR cross kernels must align with train kernels")
            for ridge in ridge_grid:
                system = kernel + float(ridge) * np.eye(len(kernel))
                inverse = np.linalg.solve(system, np.eye(len(system)))
                ones = np.ones(len(kernel))
                inverse_ones = inverse @ ones
                denominator = float(ones @ inverse_ones)
                projection = inverse - np.outer(inverse_ones, inverse_ones) / denominator
                diagonal = np.diag(projection)
                if denominator <= 0.0 or np.any(diagonal <= 0.0):
                    raise np.linalg.LinAlgError("invalid cached KRR system")
                candidates.append(CachedKRRCandidate(
                    name,
                    float(ridge),
                    (ones @ inverse) / denominator,
                    projection,
                    diagonal,
                    cross,
                ))
        if not candidates:
            raise ValueError("at least one KRR candidate is required")
        self.candidates = tuple(candidates)

    def fit_predict(self, residuals):
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim != 2 or len(residuals) != len(self.candidates[0].projection):
            raise ValueError("KRR residuals do not match cached train geometry")
        fitted = []
        for candidate in self.candidates:
            intercept = candidate.intercept_weights @ residuals
            alpha = candidate.projection @ residuals
            loo_residuals = alpha / candidate.loo_diagonal[:, None]
            fitted.append(RegionalFit(
                candidate.kernel_name,
                candidate.ridge,
                intercept,
                alpha,
                loo_residuals,
                float(np.mean(loo_residuals * loo_residuals)),
                intercept + candidate.cross_kernel @ alpha,
            ))
        return min(fitted, key=lambda value: (value.loo_mse, value.kernel_name, value.ridge))


class SyntheticGeometry:
    """Fixed features, partitions, KRR systems, and covariance endpoint eigensystems."""

    def __init__(self, item_count, outer_held_count, inner_held_count):
        if item_count < 24:
            raise ValueError("at least 24 synthetic items are required")
        if not 4 <= outer_held_count <= item_count - 12:
            raise ValueError("outer held count leaves too few fit or held items")
        outer_train_count = item_count - outer_held_count
        if not 4 <= inner_held_count <= outer_train_count - 8:
            raise ValueError("inner held count leaves too few fit or held items")
        rng = np.random.default_rng(24051)
        latent = rng.normal(size=(item_count, 3))
        self.semantic_features = np.column_stack((
            latent[:, 0],
            0.55 * latent[:, 1] + 0.20 * np.sin(latent[:, 0]),
        ))
        self.graph_features = np.column_stack((
            0.65 * latent[:, 0] - 0.35 * latent[:, 1] + 0.30 * latent[:, 2],
            np.sin(latent[:, 1]) + 0.25 * latent[:, 2],
        ))
        self.semantic_length = median_rbf_bandwidth(self.semantic_features)
        self.graph_length = median_rbf_bandwidth(self.graph_features)
        permutation = np.random.default_rng(907).permutation(item_count)
        self.outer_held = np.sort(permutation[:outer_held_count])
        self.outer_train = np.sort(permutation[outer_held_count:])
        self.inner_partitions = []
        for fold in range(3):
            inner_order = np.random.default_rng(10000 + fold).permutation(self.outer_train)
            held = np.sort(inner_order[:inner_held_count])
            fit = np.sort(inner_order[inner_held_count:])
            self.inner_partitions.append((fit, held))
        self.channel_shape = self._channel_shape()
        self.block_covariance = np.array([
            [1.00, 0.22, -0.08, 0.12],
            [0.22, 0.82, 0.14, -0.05],
            [-0.08, 0.14, 0.68, 0.09],
            [0.12, -0.05, 0.09, 0.74],
        ])
        np.linalg.cholesky(self.block_covariance)
        self.prior_covariance = np.array([[0.80, 0.18], [0.18, 0.65]])
        self.design = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.75, 0.20],
            [-0.15, 0.85],
        ])
        self.mean = self._regional_mean()
        self._item_kernels = {}
        for semantic_multiplier in MULTIPLIERS:
            for graph_multiplier in MULTIPLIERS:
                self._item_kernels[(float(semantic_multiplier), float(graph_multiplier))] = (
                    0.5 * rbf_kernel(
                        self.semantic_features,
                        length_scale=self.semantic_length * semantic_multiplier,
                    )
                    + 0.5 * rbf_kernel(
                        self.graph_features,
                        length_scale=self.graph_length * graph_multiplier,
                    )
                )
        self.canonical_item_kernel = self._item_kernels[(1.0, 1.0)]
        # Congruence by a fixed permutation preserves the complete spectrum and off-diagonal energy while
        # assigning the same correlations to the wrong item pairs.  A separate narrow RBF would make the
        # misspecified control artificially easier merely by weakening its covariance mass.
        base_permutation = np.random.default_rng(33517).permutation(item_count)
        for shift in range(item_count):
            candidate_permutation = np.roll(base_permutation, shift)
            if np.all(candidate_permutation != np.arange(item_count)):
                self.wrong_permutation = candidate_permutation
                break
        else:  # pragma: no cover - a cyclic shift always supplies a derangement for item_count > 1.
            raise RuntimeError("failed to construct the frozen wrong-geometry derangement")
        self.wrong_item_kernel = self.canonical_item_kernel[
            np.ix_(self.wrong_permutation, self.wrong_permutation)
        ]
        self.canonical_maximum = self._maximum_off_diagonal(self.canonical_item_kernel)
        self._krr = {}
        for name, (fit, evaluate) in self.partition_map.items():
            self._krr[name] = self._make_krr_cache(fit, evaluate)
        self._endpoint_cache = {}
        self._true_cache = {}
        self._true_factor_cache = {}
        self._posterior_system_cache = {}

    @property
    def item_count(self):
        return len(self.semantic_features)

    @property
    def partition_map(self):
        partitions = {"outer": (self.outer_train, self.outer_held)}
        partitions.update({f"inner_{fold}": value for fold, value in enumerate(self.inner_partitions)})
        return partitions

    @staticmethod
    def _channel_shape():
        vector = np.array([0.62, -0.47, 0.51, -0.36])
        vector /= np.linalg.norm(vector)
        return np.outer(vector, vector)

    def _regional_mean(self):
        scalar = (
            np.sin(0.9 * self.semantic_features[:, 0])
            + 0.45 * np.cos(1.1 * self.semantic_features[:, 1])
        )
        scalar = (scalar - scalar.mean()) / scalar.std()
        return scalar[:, None] * np.array([[0.75, -0.50, 0.38, -0.28]])

    @staticmethod
    def _maximum_off_diagonal(kernel):
        mask = ~np.eye(len(kernel), dtype=bool)
        return float(np.max(np.abs(kernel[mask])))

    def _make_krr_cache(self, fit, evaluate):
        train = {
            "semantic": rbf_kernel(
                self.semantic_features[fit], length_scale=self.semantic_length
            ),
            "graph": rbf_kernel(
                self.graph_features[fit], length_scale=self.graph_length
            ),
        }
        train["equal_mixture"] = 0.5 * (train["semantic"] + train["graph"])
        cross = {
            "semantic": rbf_kernel(
                self.semantic_features[evaluate],
                self.semantic_features[fit],
                length_scale=self.semantic_length,
            ),
            "graph": rbf_kernel(
                self.graph_features[evaluate],
                self.graph_features[fit],
                length_scale=self.graph_length,
            ),
        }
        cross["equal_mixture"] = 0.5 * (cross["semantic"] + cross["graph"])
        return KRRCache(train, cross)

    def candidate_endpoint(self, scenario, partition_name, semantic_multiplier, graph_multiplier, beta):
        key = (
            scenario.candidate_coupling,
            partition_name,
            float(semantic_multiplier),
            float(graph_multiplier),
            float(beta),
        )
        if key not in self._endpoint_cache:
            _, evaluate = self.partition_map[partition_name]
            item = self._item_kernels[(float(semantic_multiplier), float(graph_multiplier))][
                np.ix_(evaluate, evaluate)
            ]
            channel = (1.0 - beta) * self.channel_shape + beta * np.diag(
                np.diag(self.channel_shape)
            )
            gamma = scenario.candidate_coupling / self.canonical_maximum
            correlation = np.eye(CHANNELS * len(evaluate)) + gamma * np.kron(
                item - np.eye(len(item)), channel
            )
            correlation = 0.5 * (correlation + correlation.T)
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            if eigenvalues[0] <= 0.0:
                raise np.linalg.LinAlgError("candidate synthetic endpoint is not SPD")
            self._endpoint_cache[key] = CachedCorrelation(
                correlation, eigenvalues, eigenvectors
            )
        return self._endpoint_cache[key]

    def true_correlation(self, scenario, indices):
        key = (scenario.name, tuple(map(int, indices)))
        if key not in self._true_cache:
            if scenario.true_coupling == 0.0:
                correlation = np.eye(CHANNELS * len(indices))
            else:
                item_full = (
                    self.wrong_item_kernel if scenario.wrong_geometry
                    else self.canonical_item_kernel
                )
                item = item_full[np.ix_(indices, indices)]
                maximum = (
                    self._maximum_off_diagonal(item_full)
                    if scenario.wrong_geometry else self.canonical_maximum
                )
                gamma = scenario.true_coupling / maximum
                correlation = np.eye(CHANNELS * len(indices)) + gamma * np.kron(
                    item - np.eye(len(item)), self.channel_shape
                )
                correlation = 0.5 * (correlation + correlation.T)
            np.linalg.cholesky(correlation)
            self._true_cache[key] = correlation
        return self._true_cache[key]

    def covariance_from_correlation(self, correlation, block_covariance):
        item_count = len(correlation) // CHANNELS
        factor = np.kron(np.eye(item_count), np.linalg.cholesky(block_covariance))
        covariance = factor @ correlation @ factor.T
        return 0.5 * (covariance + covariance.T)

    def true_covariance(self, scenario, indices):
        return self.covariance_from_correlation(
            self.true_correlation(scenario, indices), self.block_covariance
        )

    def draw(self, scenario, seed):
        rng = np.random.default_rng(seed)
        if scenario.name not in self._true_factor_cache:
            correlation = self.true_correlation(scenario, np.arange(self.item_count))
            covariance = self.covariance_from_correlation(correlation, self.block_covariance)
            self._true_factor_cache[scenario.name] = np.linalg.cholesky(covariance)
        factor = self._true_factor_cache[scenario.name]
        noise = (factor @ rng.standard_normal(len(factor))).reshape(
            self.item_count, CHANNELS
        )
        state = rng.multivariate_normal(
            np.zeros(STATE_DIMENSION), self.prior_covariance, size=self.item_count
        )
        regional_mean = self.mean if scenario.regional_mean else np.zeros_like(self.mean)
        conditional_residual = regional_mean + noise
        innovation = state @ self.design.T + conditional_residual
        return conditional_residual, innovation, state

    def posterior_system(self, item_count):
        if item_count not in self._posterior_system_cache:
            design = np.kron(np.eye(item_count), self.design)
            prior_precision = np.kron(
                np.eye(item_count), np.linalg.solve(self.prior_covariance, np.eye(STATE_DIMENSION))
            )
            self._posterior_system_cache[item_count] = (design, prior_precision)
        return self._posterior_system_cache[item_count]


def _candidate_covariance(geometry, block_covariance, endpoint, alpha):
    return geometry.covariance_from_correlation(endpoint.mixture(alpha), block_covariance)


def _fit_block_covariance(residuals, shrinkage):
    """Analytic block marginal, matching ``fit_block_model`` without its unused train NLL."""
    residuals = np.asarray(residuals, dtype=float)
    block = residuals.T @ residuals / len(residuals)
    block = (1.0 - shrinkage) * block + shrinkage * np.diag(np.diag(block))
    eigenvalues = np.linalg.eigvalsh(block)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    loading = max(1e-8 * scale - float(eigenvalues[0]), 0.0)
    return block + loading * np.eye(residuals.shape[1])


def _fit_partition(geometry, partition_name, conditional_residual, shrinkage):
    fit, evaluate = geometry.partition_map[partition_name]
    regional = geometry._krr[partition_name].fit_predict(conditional_residual[fit])
    block = _fit_block_covariance(regional.loo_residuals, shrinkage)
    centered = conditional_residual[evaluate] - regional.prediction
    return fit, evaluate, regional, block, centered


def _inner_selection(geometry, scenario, conditional_residual, shrinkage):
    records = []
    regional_records = []
    for fold in range(3):
        partition_name = f"inner_{fold}"
        fit, evaluate, regional, block, centered = _fit_partition(
            geometry, partition_name, conditional_residual, shrinkage
        )
        scorer = NLLScorer(centered, block)
        block_nll = scorer.block_nll
        records.append({
            "fold": fold,
            "alpha": 0.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 1.0,
            "nll_per_scalar": float(block_nll),
        })
        regional_records.append({
            "fold": fold,
            "fit_items": len(fit),
            "held_items": len(evaluate),
            "kernel_name": regional.kernel_name,
            "ridge": regional.ridge,
            "loo_mse": regional.loo_mse,
        })
        for semantic_multiplier in MULTIPLIERS:
            for graph_multiplier in MULTIPLIERS:
                for beta in BETAS:
                    endpoint = geometry.candidate_endpoint(
                        scenario,
                        partition_name,
                        semantic_multiplier,
                        graph_multiplier,
                        beta,
                    )
                    path_scores = scorer.score_path(endpoint)
                    for alpha in ALPHAS[1:]:
                        records.append({
                            "fold": fold,
                            "alpha": float(alpha),
                            "semantic_multiplier": float(semantic_multiplier),
                            "graph_multiplier": float(graph_multiplier),
                            "beta": float(beta),
                            "nll_per_scalar": path_scores[alpha],
                        })
    selected, summaries = select_nested_candidate(records)
    return selected, summaries, regional_records


def _posterior_mse(geometry, innovation, state, evaluate, regional_prediction, covariance):
    centered_innovation = innovation[evaluate] - regional_prediction
    design, prior_precision = geometry.posterior_system(len(evaluate))
    residual = centered_innovation.reshape(-1)
    weighted = np.linalg.solve(covariance, np.column_stack((design, residual)))
    system = prior_precision + design.T @ weighted[:, :-1]
    rhs = design.T @ weighted[:, -1]
    posterior_mean = np.linalg.solve(system, rhs).reshape(len(evaluate), STATE_DIMENSION)
    return float(np.mean((posterior_mean - state[evaluate]) ** 2))


def run_replicate(geometry, scenario, replicate, seed, shrinkage):
    conditional_residual, innovation, state = geometry.draw(scenario, seed + replicate)
    selection, candidate_summaries, inner_regional = _inner_selection(
        geometry, scenario, conditional_residual, shrinkage
    )
    _, evaluate, regional, block, centered = _fit_partition(
        geometry, "outer", conditional_residual, shrinkage
    )
    dimension = CHANNELS * len(evaluate)
    block_covariance = np.kron(np.eye(len(evaluate)), block)
    scorer = NLLScorer(centered, block)
    block_nll = scorer.block_nll
    selected_endpoint = geometry.candidate_endpoint(
        scenario,
        "outer",
        selection["semantic_multiplier"],
        selection["graph_multiplier"],
        selection["beta"],
    )
    selected_covariance = _candidate_covariance(
        geometry, block, selected_endpoint, selection["alpha"]
    )
    selected_scores = scorer.score_path(selected_endpoint)
    selected_nll = (
        block_nll if selection["alpha"] == 0.0 else selected_scores[selection["alpha"]]
    )

    grid = [{
        "alpha": 0.0,
        "semantic_multiplier": 1.0,
        "graph_multiplier": 1.0,
        "beta": 1.0,
        "nll_per_scalar": float(block_nll),
    }]
    for semantic_multiplier in MULTIPLIERS:
        for graph_multiplier in MULTIPLIERS:
            for beta in BETAS:
                endpoint = geometry.candidate_endpoint(
                    scenario, "outer", semantic_multiplier, graph_multiplier, beta
                )
                path_scores = scorer.score_path(endpoint)
                for alpha in ALPHAS[1:]:
                    grid.append({
                        "alpha": float(alpha),
                        "semantic_multiplier": float(semantic_multiplier),
                        "graph_multiplier": float(graph_multiplier),
                        "beta": float(beta),
                        "nll_per_scalar": path_scores[alpha],
                    })
    oracle = min(grid, key=lambda row: row["nll_per_scalar"])
    oracle_endpoint = geometry.candidate_endpoint(
        scenario,
        "outer",
        oracle["semantic_multiplier"],
        oracle["graph_multiplier"],
        oracle["beta"],
    )
    oracle_covariance = _candidate_covariance(
        geometry, block, oracle_endpoint, oracle["alpha"]
    )
    true_covariance = geometry.true_covariance(scenario, evaluate)
    true_nll = gaussian_joint_nll(centered, true_covariance).per_scalar
    matched_covariance = geometry.covariance_from_correlation(
        geometry.true_correlation(scenario, evaluate), block
    )
    matched_nll = gaussian_joint_nll(centered, matched_covariance).per_scalar

    posterior_mse = {
        "block": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, block_covariance
        ),
        "selected": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, selected_covariance
        ),
        "outer_nll_oracle": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, oracle_covariance
        ),
        "known_true_covariance": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, true_covariance
        ),
        "matched_true_correlation_refit_marginal": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, matched_covariance
        ),
    }
    return {
        "replicate": int(replicate),
        "seed": int(seed + replicate),
        "selection": selection,
        "eligible_nonzero_candidates": int(sum(
            row["alpha"] > 0.0 and row["macro_gain_vs_block"] > 0.0
            and row["positive_folds"] >= 2
            for row in candidate_summaries
        )),
        "inner_regional_means": inner_regional,
        "outer_regional_mean": {
            "kernel_name": regional.kernel_name,
            "ridge": regional.ridge,
            "loo_mse": regional.loo_mse,
        },
        "outer_items": int(len(evaluate)),
        "measurement_dimension": int(dimension),
        "residual_nll_per_scalar": {
            "block": float(block_nll),
            "selected": float(selected_nll),
            "outer_grid_oracle": float(oracle["nll_per_scalar"]),
            "known_true_covariance": float(true_nll),
            "matched_true_correlation_refit_marginal": float(matched_nll),
        },
        "outer_grid_oracle": oracle,
        "posterior_state_mse": posterior_mse,
    }


def aggregate_scenario(scenario, records):
    selected_alpha = np.asarray([row["selection"]["alpha"] for row in records])
    block_nll = np.asarray([
        row["residual_nll_per_scalar"]["block"] for row in records
    ])
    selected_nll = np.asarray([
        row["residual_nll_per_scalar"]["selected"] for row in records
    ])
    oracle_nll = np.asarray([
        row["residual_nll_per_scalar"]["outer_grid_oracle"] for row in records
    ])
    true_nll = np.asarray([
        row["residual_nll_per_scalar"]["known_true_covariance"] for row in records
    ])
    matched_nll = np.asarray([
        row["residual_nll_per_scalar"]["matched_true_correlation_refit_marginal"]
        for row in records
    ])
    block_mse = np.asarray([row["posterior_state_mse"]["block"] for row in records])
    selected_mse = np.asarray([row["posterior_state_mse"]["selected"] for row in records])
    oracle_mse = np.asarray([
        row["posterior_state_mse"]["outer_nll_oracle"] for row in records
    ])
    true_mse = np.asarray([
        row["posterior_state_mse"]["known_true_covariance"] for row in records
    ])
    matched_mse = np.asarray([
        row["posterior_state_mse"]["matched_true_correlation_refit_marginal"]
        for row in records
    ])
    selected_nll_gain = block_nll - selected_nll
    true_nll_gain = block_nll - true_nll
    matched_nll_gain = block_nll - matched_nll
    selected_mse_gain = block_mse - selected_mse
    true_mse_gain = block_mse - true_mse
    matched_mse_gain = block_mse - matched_mse
    block_rate = float(np.mean(selected_alpha == 0.0))
    nonzero_rate = float(np.mean(selected_alpha > 0.0))
    residual_recovery = _ratio(np.mean(selected_nll_gain), np.mean(true_nll_gain))
    posterior_recovery = _ratio(np.mean(selected_mse_gain), np.mean(true_mse_gain))
    criterion = {
        "type": "measured_power_only",
        "pass": None,
        "requirements": [],
    }
    if scenario.name in {"block_null", "regional_mean_only", "wrong_item_geometry"}:
        harm = float(np.mean(selected_nll - block_nll))
        criterion = {
            "type": "null_or_misspecified_geometry_control",
            "requirements": [
                "mean held harm <= 0.001 NLL/scalar",
                "block selected in at least 80% of replicates",
            ],
            "mean_held_harm_nll_per_scalar": harm,
            "block_selection_rate": block_rate,
            "harm_requirement_pass": bool(harm <= 0.001),
            "selection_requirement_pass": bool(block_rate >= 0.80),
            "pass": bool(harm <= 0.001 and block_rate >= 0.80),
        }
    elif scenario.true_coupling >= 0.10:
        criterion = {
            "type": "in_family_power_control",
            "requirements": [
                "nonzero alpha selected in at least 80% of replicates",
                "recover at least 50% of known-true-covariance residual NLL gain",
                "recover at least 50% of known-true-covariance posterior-risk gain",
            ],
            "nonzero_alpha_selection_rate": nonzero_rate,
            "residual_nll_recovery_fraction": residual_recovery,
            "posterior_risk_recovery_fraction": posterior_recovery,
            "selection_requirement_pass": bool(nonzero_rate >= 0.80),
            "residual_recovery_requirement_pass": bool(
                residual_recovery is not None and residual_recovery >= 0.50
            ),
            "posterior_recovery_requirement_pass": bool(
                posterior_recovery is not None and posterior_recovery >= 0.50
            ),
        }
        criterion["pass"] = bool(
            criterion["selection_requirement_pass"]
            and criterion["residual_recovery_requirement_pass"]
            and criterion["posterior_recovery_requirement_pass"]
        )
    return {
        "scenario": scenario.name,
        "replicates": len(records),
        "true_maximum_whitened_off_block_coupling": scenario.true_coupling,
        "candidate_endpoint_maximum_whitened_off_block_coupling": scenario.candidate_coupling,
        "selection": {
            "block_rate": block_rate,
            "nonzero_alpha_rate": nonzero_rate,
            "alpha_counts": dict(Counter(map(str, selected_alpha.tolist()))),
            "semantic_multiplier_counts": dict(Counter(
                str(row["selection"]["semantic_multiplier"]) for row in records
            )),
            "graph_multiplier_counts": dict(Counter(
                str(row["selection"]["graph_multiplier"]) for row in records
            )),
            "beta_counts": dict(Counter(
                str(row["selection"]["beta"]) for row in records
            )),
        },
        "residual_nll_per_scalar": {
            "block": _summary(block_nll),
            "selected": _summary(selected_nll),
            "outer_grid_oracle": _summary(oracle_nll),
            "known_true_covariance": _summary(true_nll),
            "matched_true_correlation_refit_marginal": _summary(matched_nll),
            "selected_gain_vs_block": _summary(selected_nll_gain),
            "outer_grid_oracle_gain_vs_block": _summary(block_nll - oracle_nll),
            "known_true_covariance_gain_vs_block": _summary(true_nll_gain),
            "matched_true_correlation_gain_vs_block": _summary(matched_nll_gain),
            "selected_recovery_fraction_of_known_truth_mean_gain": residual_recovery,
        },
        "posterior_state_mse": {
            "block": _summary(block_mse),
            "selected": _summary(selected_mse),
            "outer_nll_grid_oracle": _summary(oracle_mse),
            "known_true_covariance": _summary(true_mse),
            "matched_true_correlation_refit_marginal": _summary(matched_mse),
            "selected_gain_vs_block": _summary(selected_mse_gain),
            "known_true_covariance_gain_vs_block": _summary(true_mse_gain),
            "matched_true_correlation_gain_vs_block": _summary(matched_mse_gain),
            "selected_recovery_fraction_of_known_truth_mean_gain": posterior_recovery,
        },
        "preregistered_criterion": criterion,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--items", type=int, default=96)
    parser.add_argument("--outer-held-items", type=int, default=32)
    parser.add_argument("--inner-held-items", type=int, default=22)
    parser.add_argument("--seed", type=int, default=881000)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIO_ORDER, default=list(SCENARIO_ORDER)
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--out", default="/tmp/covariance_sensitivity_synthetic.json")
    return parser


def _validate_args(args):
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if not 0.0 <= args.shrinkage <= 1.0:
        raise ValueError("--shrinkage must be in [0,1]")


def _write_payload(path, payload):
    path = os.path.abspath(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main():
    args = build_arg_parser().parse_args()
    _validate_args(args)
    started = time.perf_counter()
    geometry = SyntheticGeometry(
        args.items, args.outer_held_items, args.inner_held_items
    )
    scenario_by_name = {scenario.name: scenario for scenario in SCENARIOS}
    aggregates = {}
    replicate_records = {}
    for name in args.scenarios:
        scenario = scenario_by_name[name]
        scenario_index = SCENARIO_ORDER.index(name)
        records = [
            run_replicate(
                geometry,
                scenario,
                replicate,
                args.seed + 100000 * scenario_index,
                args.shrinkage,
            )
            for replicate in range(args.replicates)
        ]
        aggregates[name] = aggregate_scenario(scenario, records)
        if not args.summary_only:
            replicate_records[name] = records
    required = [
        aggregates[name]["preregistered_criterion"]["pass"]
        for name in args.scenarios
        if aggregates[name]["preregistered_criterion"]["pass"] is not None
    ]
    invalidated_gate_evaluable = set(args.scenarios) == set(SCENARIO_ORDER)
    invalidated_gate_pass = bool(
        invalidated_gate_evaluable and required and all(required)
    )
    payload = {
        "status": (
            "INVALIDATED/AMENDED v1 diagnostic; corrected derangement no longer reproduces the original "
            "weak wrong-geometry run; do not interpret known-truth recovery fractions"
        ),
        "blocking_erratum": (
            "after fitted/outcome-selected KRR, centered residual is e_H-W e_T and does not have R_HH; "
            "use corrected v2 oracle-mean/known-B mechanism for recovery"
        ),
        "protocol": (
            "100-replicate default synthetic selector controls with known PSD covariance endpoints; "
            "regional mean and block marginal refit per partition; no LMC optimization"
        ),
        "configuration": {
            "replicates": args.replicates,
            "seed": args.seed,
            "requested_scenarios": list(args.scenarios),
            "items": args.items,
            "outer_train_items": len(geometry.outer_train),
            "outer_held_items": len(geometry.outer_held),
            "inner_fit_items": len(geometry.inner_partitions[0][0]),
            "inner_held_items": len(geometry.inner_partitions[0][1]),
            "channels": CHANNELS,
            "state_dimension": STATE_DIMENSION,
            "alphas": list(ALPHAS),
            "length_multipliers": list(MULTIPLIERS),
            "channel_shrinkage_betas": list(BETAS),
            "regional_mean_ridges": list(RIDGES),
            "block_covariance_shrinkage": args.shrinkage,
            "semantic_base_length": geometry.semantic_length,
            "graph_base_length": geometry.graph_length,
            "wrong_geometry": {
                "construction": (
                    "fixed derangement congruence P K_canonical P.T; first cyclic shift of seed "
                    "permutation having no fixed points"
                ),
                "permutation_seed": 33517,
                "spectrum_and_off_diagonal_energy_preserved": True,
            },
            "outcome_blind_cache": (
                "fixed partitions, KRR linear systems, kernels, and correlation eigensystems only"
            ),
        },
        "scenario_aggregates": aggregates,
        "invalidated_original_criteria_gate_evaluable": invalidated_gate_evaluable,
        "invalidated_original_criteria_all_pass": invalidated_gate_pass,
        "replicate_records": None if args.summary_only else replicate_records,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_payload(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "blocking_erratum": payload["blocking_erratum"],
        "output": os.path.abspath(args.out),
        "scenario_aggregates": aggregates,
        "invalidated_original_criteria_gate_evaluable": invalidated_gate_evaluable,
        "invalidated_original_criteria_all_pass": invalidated_gate_pass,
        "wall_seconds": payload["wall_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
