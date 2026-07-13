"""Repeat-aware synthetic power primitives for graph-local judge covariance.

This is a sizing mechanism, not a real-data analysis.  Every synthetic
replicate regenerates its component partition, cross-fits five outer folds,
selects mean regularization and covariance capacity inside outer training, and
scores two distinct held endpoints: residual Gaussian NLL and the NLL of a
small linear-Gaussian posterior over latent D/S states.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


ROWS_PER_COMPONENT = 3
CHANNELS = 4  # operating D/S, Luna D/S
STATE_CHANNELS = 2
OUTER_FOLDS = 5
INNER_FOLDS = 3
DEFAULT_GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_RHOS = (0.0, 0.025, 0.05, 0.10, 0.20)
DEFAULT_MEAN_RIDGES = (0.0, 0.01, 0.10, 1.0, 10.0)
PRIMARY_ENDPOINTS = ("residual_nll", "posterior_state_nll")
MAX_PROMPT_ROWS = 10
SYNTHETIC_CORPORA = ("synthetic_corpus_1", "synthetic_corpus_2")


def derive_seed(base: int, *parts) -> int:
    payload = json.dumps((int(base), *parts), separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def _symmetric(value):
    value = np.asarray(value, dtype=float)
    return 0.5 * (value + value.T)


def _spd_floor(value, relative_floor=1e-8):
    value = _symmetric(value)
    eigenvalues = np.linalg.eigvalsh(value)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    loading = max(relative_floor * scale - float(eigenvalues[0]), 0.0)
    return value + loading * np.eye(len(value)), float(loading)


def _normalize_rows(features):
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("feature rows must be nonzero")
    return features / norms[:, None]


def explicit_item_kernels():
    """Frozen explicit PSD cumulative-walk and Nomic proxy Gram blocks."""
    cumulative_features = _normalize_rows(np.array([
        [1.00, 0.00, 0.00],
        [0.85, math.sqrt(1.0 - 0.85**2), 0.00],
        [0.10, 0.00, math.sqrt(1.0 - 0.10**2)],
    ]))
    nomic_features = _normalize_rows(np.array([
        [1.00, 0.00, 0.00],
        [0.45, math.sqrt(1.0 - 0.45**2), 0.00],
        [0.65, -0.15, math.sqrt(1.0 - 0.65**2 - 0.15**2)],
    ]))
    kernels = {
        "cumulative": _symmetric(cumulative_features @ cumulative_features.T),
        "nomic": _symmetric(nomic_features @ nomic_features.T),
    }
    kernels["graph"] = kernels["cumulative"]  # compatibility alias; never a separate vote
    kernels["deranged_cumulative"] = deranged_item_kernel(kernels["cumulative"])
    for name, kernel in kernels.items():
        if not np.allclose(np.diag(kernel), 1.0, atol=1e-12):
            raise AssertionError(f"{name} kernel is not unit diagonal")
        if np.linalg.eigvalsh(kernel)[0] < -1e-12:
            raise AssertionError(f"{name} kernel is not PSD")
    return kernels


def deranged_item_kernel(kernel):
    """Equal-energy/spectrum control that exchanges adjacent and distant roles."""
    kernel = np.asarray(kernel, dtype=float)
    if kernel.shape != (ROWS_PER_COMPONENT, ROWS_PER_COMPONENT):
        raise ValueError("item kernel must be 3x3")
    permutation = np.array([0, 2, 1])
    return _symmetric(kernel[np.ix_(permutation, permutation)])


def gamma_item_kernel(gamma, kernels=None):
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1]")
    kernels = explicit_item_kernels() if kernels is None else kernels
    return _symmetric(
        gamma * np.asarray(kernels["cumulative"])
        + (1.0 - gamma) * np.asarray(kernels["nomic"])
    )


def maximum_off_diagonal(kernel):
    kernel = np.asarray(kernel, dtype=float)
    if kernel.shape != (ROWS_PER_COMPONENT, ROWS_PER_COMPONENT):
        raise ValueError("item kernel must be 3x3")
    return float(np.max(np.abs(kernel[~np.eye(len(kernel), dtype=bool)])))


def rho_matched_correlation(kernel, rho):
    kernel = _symmetric(np.asarray(kernel, dtype=float))
    rho = float(rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0,1)")
    scale = maximum_off_diagonal(kernel)
    if scale <= 0.0:
        if rho == 0.0:
            return np.eye(len(kernel))
        raise ValueError("a zero off-diagonal kernel cannot carry nonzero rho")
    alpha = rho / scale
    if alpha >= 0.95:
        raise ValueError("rho requires path amplitude at or above the frozen 0.95 limit")
    correlation = _symmetric((1.0 - alpha) * np.eye(len(kernel)) + alpha * kernel)
    if np.linalg.eigvalsh(correlation)[0] <= 0.0:
        raise np.linalg.LinAlgError("rho-matched item correlation is not SPD")
    return correlation


PERSISTENT_CHANNEL_COVARIANCE = np.array([
    [0.70, 0.15, 0.18, 0.05],
    [0.15, 0.62, -0.06, 0.14],
    [0.18, -0.06, 0.58, 0.11],
    [0.05, 0.14, 0.11, 0.66],
])
CALL_CHANNEL_COVARIANCE = np.array([
    [0.30, 0.08, 0.00, 0.00],
    [0.08, 0.25, 0.00, 0.00],
    [0.00, 0.00, 0.20, -0.04],
    [0.00, 0.00, -0.04, 0.28],
])
REQUEST_CHANNEL_COVARIANCE = np.array([
    [0.075, 0.018, 0.000, 0.000],
    [0.018, 0.060, 0.000, 0.000],
    [0.000, 0.000, 0.055, -0.010],
    [0.000, 0.000, -0.010, 0.070],
])
WAVE_CHANNEL_COVARIANCE = 0.025 * np.array([
    [1.0, 0.2, 0.0, 0.0],
    [0.2, 0.8, 0.0, 0.0],
    [0.0, 0.0, 0.7, -0.1],
    [0.0, 0.0, -0.1, 0.9],
])
PRIOR_STATE_COVARIANCE = np.array([[0.85, 0.12], [0.12, 0.70]])
MEASUREMENT_DESIGN = np.array([
    [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]
])


@dataclass(frozen=True)
class RepeatedScenario:
    name: str
    truth_gamma: float
    truth_rho: float
    mean_strength: float = 0.65
    deranged_truth: bool = False


def _rho_name(value):
    return f"{float(value):.2f}"


SCENARIOS = tuple([
    RepeatedScenario("block_null", 1.0, 0.0),
    RepeatedScenario("mean_only", 1.0, 0.0, mean_strength=1.0),
] + [
    RepeatedScenario(f"{family}_rho_{_rho_name(rho)}", gamma, rho, deranged_truth=deranged)
    for family, gamma, deranged in (
        ("cumulative", 1.0, False),
        ("nomic", 0.0, False),
        ("mixture", 0.5, False),
        ("deranged", 1.0, True),
    )
    for rho in (0.04, 0.10, 0.20)
])
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


@dataclass(frozen=True)
class CampaignGeometry:
    component_count: int
    mean_design: np.ndarray
    true_mean: np.ndarray
    kernels: Mapping[str, np.ndarray]
    prompt_blocks: Tuple[np.ndarray, ...]
    prompt_block_index: np.ndarray


def _validate_prompt_blocks(component_count, prompt_blocks, max_prompt_rows=MAX_PROMPT_ROWS):
    blocks = tuple(np.sort(np.asarray(block, dtype=int)) for block in prompt_blocks)
    if not blocks or any(not len(block) or len(block) > max_prompt_rows for block in blocks):
        raise ValueError("prompt blocks must contain one to ten components")
    flattened = np.concatenate(blocks)
    if sorted(flattened.tolist()) != list(range(component_count)):
        raise ValueError("prompt blocks must partition every component exactly once")
    index = np.empty(component_count, dtype=int)
    for block_index, block in enumerate(blocks):
        index[block] = block_index
    return blocks, index


def _simple_prompt_blocks(component_count, seed, max_prompt_rows=MAX_PROMPT_ROWS):
    block_count = max(OUTER_FOLDS, int(math.ceil(component_count / max_prompt_rows)))
    order = np.random.default_rng(seed).permutation(component_count)
    return tuple(np.sort(chunk) for chunk in np.array_split(order, block_count))


def build_campaign_geometry(
    component_count,
    seed=24051,
    *,
    prompt_blocks=None,
    max_prompt_rows=MAX_PROMPT_ROWS,
):
    if component_count < 12:
        raise ValueError("at least 12 endpoint components are required")
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((component_count, 2))
    design = np.zeros((component_count, ROWS_PER_COMPONENT, 6), dtype=float)
    for role in range(ROWS_PER_COMPONENT):
        design[:, role, 0] = 1.0
        design[:, role, 1:3] = latent
        design[:, role, 3] = float(role == 1)
        design[:, role, 4] = float(role == 2)
        design[:, role, 5] = latent[:, 0] * float(role == 1) - latent[:, 1] * float(role == 2)
    coefficients = np.array([
        [0.12, -0.08, 0.06, -0.04],
        [0.18, -0.13, 0.09, -0.07],
        [-0.11, 0.16, -0.05, 0.12],
        [0.20, -0.06, 0.13, -0.10],
        [-0.14, 0.09, -0.12, 0.08],
        [0.08, 0.05, -0.07, 0.04],
    ])
    true_mean = design @ coefficients
    nonlinear = 0.10 * np.sin(1.25 * latent[:, 0])[:, None, None]
    true_mean += nonlinear * np.array([1.0, 0.65, -0.35])[None, :, None] * np.array(
        [0.70, -0.45, 0.35, -0.25]
    )[None, None, :]
    if prompt_blocks is None:
        prompt_blocks = _simple_prompt_blocks(
            component_count, derive_seed(seed, "prompt-blocks"), max_prompt_rows
        )
    prompt_blocks, prompt_block_index = _validate_prompt_blocks(
        component_count, prompt_blocks, max_prompt_rows
    )
    return CampaignGeometry(
        int(component_count),
        design,
        true_mean,
        explicit_item_kernels(),
        prompt_blocks,
        prompt_block_index,
    )


def scenario_item_kernel(geometry, scenario):
    kernel = gamma_item_kernel(scenario.truth_gamma, geometry.kernels)
    return deranged_item_kernel(kernel) if scenario.deranged_truth else kernel


def draw_repeated_field(
    geometry,
    scenario,
    repeats,
    seed,
    *,
    missing_rate=0.02,
    include_wave_effect=True,
):
    """Draw ``[component,row,repeat,channel]``; missing calls are whole-response NaNs."""
    if repeats < 3:
        raise ValueError("the repeated-judge design requires at least three calls")
    if not 0.0 <= missing_rate < 0.25:
        raise ValueError("missing_rate must lie in [0,.25)")
    rng = np.random.default_rng(seed)
    item = rho_matched_correlation(scenario_item_kernel(geometry, scenario), scenario.truth_rho)
    persistent_factor = np.linalg.cholesky(np.kron(item, PERSISTENT_CHANNEL_COVARIANCE))
    persistent = (
        rng.standard_normal((geometry.component_count, ROWS_PER_COMPONENT * CHANNELS))
        @ persistent_factor.T
    ).reshape(geometry.component_count, ROWS_PER_COMPONENT, CHANNELS)
    call_factor = np.linalg.cholesky(CALL_CHANNEL_COVARIANCE)
    call = rng.standard_normal(
        (geometry.component_count, ROWS_PER_COMPONENT, repeats, CHANNELS)
    ) @ call_factor.T
    # A request contains one role from each component in one stable prompt
    # block.  Separate judge-family requests make this covariance block
    # diagonal across the two D/S families, as frozen above.
    request = rng.standard_normal((
        len(geometry.prompt_blocks), ROWS_PER_COMPONENT, repeats, CHANNELS
    )) @ np.linalg.cholesky(REQUEST_CHANNEL_COVARIANCE).T
    wave = np.zeros((repeats, CHANNELS))
    if include_wave_effect:
        wave = rng.standard_normal((repeats, CHANNELS)) @ np.linalg.cholesky(
            WAVE_CHANNEL_COVARIANCE
        ).T
        wave -= wave.mean(axis=0, keepdims=True)
    field = (
        scenario.mean_strength * geometry.true_mean[:, :, None, :]
        + persistent[:, :, None, :]
        + call
        + request[geometry.prompt_block_index]
        + wave[None, None, :, :]
    )
    if missing_rate:
        # Failures occur at the request level: all rows in a prompt block lose
        # the same judge-family response.  Each request contains no repeated
        # component and every block/role/family retains at least two waves.
        observed_request = rng.random((
            len(geometry.prompt_blocks), ROWS_PER_COMPONENT, repeats, 2
        )) >= missing_rate
        for block in range(len(geometry.prompt_blocks)):
            for row in range(ROWS_PER_COMPONENT):
                for family in range(2):
                    if np.sum(observed_request[block, row, :, family]) < 2:
                        observed_request[block, row, :2, family] = True
        field = field.copy()
        observed = observed_request[geometry.prompt_block_index]
        for family, start in enumerate((0, 2)):
            field[..., start:start + 2] = np.where(
                observed[..., family, None], field[..., start:start + 2], np.nan
            )
    return field


@dataclass(frozen=True)
class OuterFold:
    train: np.ndarray
    held: np.ndarray
    inner: Tuple[Tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class ComponentSplits:
    outer: Tuple[OuterFold, ...]
    seed: int
    outer_label: np.ndarray
    inner_label: np.ndarray
    prompt_blocks: Tuple[np.ndarray, ...]
    prompt_block_index: np.ndarray

    @property
    def outer_train(self):  # compatibility: first fold only
        return self.outer[0].train

    @property
    def outer_held(self):
        return self.outer[0].held

    @property
    def inner(self):
        return self.outer[0].inner


def _balanced_inner_rotations(outer_sizes):
    """Choose deterministic cell rotations that balance every leave-one-row margin."""
    best = None
    for rotations in itertools.product(range(INNER_FOLDS), repeat=OUTER_FOLDS):
        table = np.zeros((OUTER_FOLDS, INNER_FOLDS), dtype=int)
        for outer, (size, rotation) in enumerate(zip(outer_sizes, rotations)):
            quotient, remainder = divmod(int(size), INNER_FOLDS)
            table[outer] = quotient
            for offset in range(remainder):
                table[outer, (rotation + offset) % INNER_FOLDS] += 1
        leave_one_imbalances = [
            int(np.ptp(np.sum(np.delete(table, outer, axis=0), axis=0)))
            for outer in range(OUTER_FOLDS)
        ]
        score = (max(leave_one_imbalances), int(np.ptp(table.sum(axis=0))), rotations)
        if best is None or score < best[0]:
            best = (score, table)
    return best[1]


def component_splits(
    component_count,
    *,
    seed=51001,
    outer_folds=OUTER_FOLDS,
    inner_folds=INNER_FOLDS,
    max_prompt_rows=MAX_PROMPT_ROWS,
):
    if outer_folds != OUTER_FOLDS:
        raise ValueError("the frozen procedure requires exactly five outer folds")
    if inner_folds != INNER_FOLDS:
        raise ValueError("the frozen procedure requires exactly three inner folds")
    if component_count < outer_folds * 2:
        raise ValueError("too few components for five outer folds")
    if not 1 <= max_prompt_rows <= MAX_PROMPT_ROWS:
        raise ValueError("prompt capacity must lie in [1,10]")
    rng = np.random.default_rng(seed)
    order = rng.permutation(component_count)
    held_chunks = np.array_split(order, outer_folds)
    outer_label = np.empty(component_count, dtype=int)
    inner_label = np.empty(component_count, dtype=int)
    for outer_index, chunk in enumerate(held_chunks):
        outer_label[chunk] = outer_index
    inner_counts = _balanced_inner_rotations([len(chunk) for chunk in held_chunks])
    for outer_index, chunk in enumerate(held_chunks):
        shuffled = np.random.default_rng(
            derive_seed(seed, "global-inner", outer_index)
        ).permutation(chunk)
        start = 0
        for inner_index, count in enumerate(inner_counts[outer_index]):
            inner_label[shuffled[start:start + count]] = inner_index
            start += count

    prompt_blocks = []
    for outer_index in range(outer_folds):
        for inner_index in range(inner_folds):
            cell = np.flatnonzero(
                (outer_label == outer_index) & (inner_label == inner_index)
            )
            cell = np.random.default_rng(
                derive_seed(seed, "prompt-cell", outer_index, inner_index)
            ).permutation(cell)
            if not len(cell):
                continue
            block_count = max(1, int(math.ceil(len(cell) / max_prompt_rows)))
            prompt_blocks.extend(np.sort(part) for part in np.array_split(cell, block_count))
    prompt_blocks, prompt_block_index = _validate_prompt_blocks(
        component_count, prompt_blocks, max_prompt_rows
    )

    all_components = set(range(component_count))
    outer = []
    for outer_index in range(outer_folds):
        held = np.flatnonzero(outer_label == outer_index)
        train = np.asarray(sorted(all_components - set(map(int, held))), dtype=int)
        inner = []
        train_set = set(map(int, train))
        for inner_index in range(inner_folds):
            inner_held = np.flatnonzero(
                (outer_label != outer_index) & (inner_label == inner_index)
            )
            inner_fit = np.asarray(sorted(train_set - set(map(int, inner_held))), dtype=int)
            inner.append((inner_fit, inner_held))
        outer.append(OuterFold(train, held, tuple(inner)))
    if sorted(np.concatenate([fold.held for fold in outer]).tolist()) != list(range(component_count)):
        raise AssertionError("five held folds must cover every component exactly once")
    # A stable prompt request can never cross an outer or global-inner label.
    for block in prompt_blocks:
        if len(set(outer_label[block])) != 1 or len(set(inner_label[block])) != 1:
            raise AssertionError("a prompt block crossed an analysis split signature")
    return ComponentSplits(
        tuple(outer),
        int(seed),
        outer_label,
        inner_label,
        prompt_blocks,
        prompt_block_index,
    )


def _fit_linear_mean(geometry, repeat_means, fit, evaluate, ridge):
    fit = np.asarray(fit, dtype=int)
    evaluate = np.asarray(evaluate, dtype=int)
    X = geometry.mean_design[fit].reshape(-1, geometry.mean_design.shape[-1])
    y = repeat_means[fit].reshape(-1, CHANNELS)
    penalty = np.eye(X.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    return (
        coefficients,
        geometry.mean_design[fit] @ coefficients,
        geometry.mean_design[evaluate] @ coefficients,
    )


def _select_mean_ridge(geometry, means, fit, ridges):
    ridges = tuple(sorted(set(map(float, ridges))))
    if not ridges or any(value < 0.0 for value in ridges):
        raise ValueError("mean ridge grid must be nonempty and nonnegative")
    fit = np.asarray(fit, dtype=int)
    fit_blocks = np.unique(geometry.prompt_block_index[fit])
    chunks = np.array_split(fit_blocks, min(3, len(fit_blocks)))
    scores = []
    fit_set = set(map(int, fit))
    for ridge in ridges:
        fold_scores = []
        for held_blocks in chunks:
            held = fit[np.isin(geometry.prompt_block_index[fit], held_blocks)]
            train = np.asarray(sorted(fit_set - set(map(int, held))), dtype=int)
            if not len(train) or not len(held):
                continue
            _coef, _train_pred, held_pred = _fit_linear_mean(
                geometry, means, train, held, ridge
            )
            fold_scores.append(float(np.mean((means[held] - held_pred) ** 2)))
        scores.append((float(np.mean(fold_scores)), ridge))
    return min(scores)[1]


def _estimate_wave_effects(field, fit):
    fit_values = field[np.asarray(fit, dtype=int)]
    wave_means = np.nanmean(fit_values, axis=(0, 1))
    grand = np.nanmean(wave_means, axis=0, keepdims=True)
    return wave_means - grand


def _fit_total_repeat_covariance(field):
    row_means = np.nanmean(field, axis=2, keepdims=True)
    centered = field - row_means
    structured = np.zeros((CHANNELS, CHANNELS), dtype=float)
    for start in (0, 2):
        estimates = []
        for component in range(field.shape[0]):
            for row in range(field.shape[1]):
                observed = np.isfinite(centered[component, row, :, start])
                values = centered[component, row, observed, start:start + 2]
                if len(values) >= 2:
                    estimates.append(values.T @ values / (len(values) - 1.0))
        if not estimates:
            raise ValueError("no row retains two calls for repeat-covariance estimation")
        block = _symmetric(np.mean(estimates, axis=0))
        structured[start:start + 2, start:start + 2] = 0.95 * block + 0.05 * np.diag(
            np.diag(block)
        )
    return _spd_floor(structured)[0]


def _fit_request_covariance(field, geometry, fit):
    """Method-of-moments covariance for the shared prompt-request effect.

    The cross-component product within one stable request eliminates the
    row-specific call noise.  Complete-wave component pairs are used so the
    usual ``1 - 1/R`` centering correction is exact.  The estimator is refit
    from training prompt blocks only and receives the frozen 5% diagonal
    shrinkage plus the reported numerical SPD floor.
    """
    fit = np.asarray(fit, dtype=int)
    structured = np.zeros((CHANNELS, CHANNELS), dtype=float)
    for family, start in enumerate((0, 2)):
        estimates = []
        for block_id in np.unique(geometry.prompt_block_index[fit]):
            components = fit[geometry.prompt_block_index[fit] == block_id]
            if len(components) < 2:
                continue
            for row in range(ROWS_PER_COMPONENT):
                values = field[components, row, :, start:start + 2]
                complete = np.all(np.isfinite(values), axis=(1, 2))
                values = values[complete]
                if len(values) < 2:
                    continue
                values = values - values.mean(axis=1, keepdims=True)
                repeats = values.shape[1]
                centering = 1.0 - 1.0 / repeats
                for repeat in range(repeats):
                    wave_values = values[:, repeat, :]
                    total = np.sum(wave_values, axis=0)
                    cross_sum = (
                        np.outer(total, total)
                        - np.einsum("ni,nj->ij", wave_values, wave_values)
                    ) / (len(wave_values) * (len(wave_values) - 1.0))
                    estimates.append(cross_sum / centering)
        if not estimates:
            raise ValueError(
                "request covariance is not identified from two complete components "
                f"in any training prompt block for family {family}"
            )
        block = _symmetric(np.mean(estimates, axis=0))
        block = 0.95 * block + 0.05 * np.diag(np.diag(block))
        structured[start:start + 2, start:start + 2] = block
    return _spd_floor(structured)


@dataclass(frozen=True)
class NuisanceFit:
    coefficients: np.ndarray
    selected_mean_ridge: float
    wave_effects: np.ndarray
    call_covariance: np.ndarray
    request_covariance: np.ndarray
    repeat_covariance: np.ndarray
    persistent_covariance: np.ndarray
    call_loading: float
    request_loading: float
    persistent_loading: float
    evaluate_prediction: np.ndarray
    evaluate_centered_means: np.ndarray
    evaluate_repeat_counts: np.ndarray
    evaluate_observed_calls: np.ndarray


def fit_repeat_nuisance(
    field,
    geometry,
    fit_components,
    evaluate_components,
    *,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
):
    field = np.asarray(field, dtype=float)
    if field.ndim != 4 or field.shape[:2] != (geometry.component_count, ROWS_PER_COMPONENT) or field.shape[3] != CHANNELS:
        raise ValueError("field must have shape [G,3,R,4]")
    if field.shape[2] < 3:
        raise ValueError("at least three repeats are required")
    fit = np.asarray(fit_components, dtype=int)
    evaluate = np.asarray(evaluate_components, dtype=int)
    wave_effects = _estimate_wave_effects(field, fit)
    adjusted = field - wave_effects[None, None, :, :]
    observed_calls = np.stack([
        np.isfinite(adjusted[..., start]) for start in (0, 2)
    ], axis=-1)
    counts = np.sum(observed_calls, axis=2)
    if np.any(counts < 2):
        raise ValueError("every row must retain at least two calls")
    means = np.nanmean(adjusted, axis=2)
    selected_ridge = _select_mean_ridge(geometry, means, fit, mean_ridges)
    coefficients, train_prediction, evaluate_prediction = _fit_linear_mean(
        geometry, means, fit, evaluate, selected_ridge
    )
    train_residual = means[fit] - train_prediction
    total_repeat_covariance = _fit_total_repeat_covariance(adjusted[fit])
    request_covariance, request_loading = _fit_request_covariance(
        adjusted, geometry, fit
    )
    call_raw = _symmetric(total_repeat_covariance - request_covariance)
    call_raw = (1.0 - shrinkage) * call_raw + shrinkage * np.diag(np.diag(call_raw))
    call_covariance, call_loading = _spd_floor(call_raw)
    repeat_covariance = call_covariance + request_covariance
    flat = train_residual.reshape(-1, CHANNELS)
    mean_covariance = flat.T @ flat / len(flat)
    sampling_covariance = np.zeros((CHANNELS, CHANNELS), dtype=float)
    for family, start in enumerate((0, 2)):
        sl = slice(start, start + 2)
        average_inverse_repeats = float(np.mean(1.0 / counts[fit, :, family]))
        sampling_covariance[sl, sl] = (
            repeat_covariance[sl, sl] * average_inverse_repeats
        )
    persistent = _symmetric(mean_covariance - sampling_covariance)
    persistent = (1.0 - shrinkage) * persistent + shrinkage * np.diag(np.diag(persistent))
    persistent_covariance, persistent_loading = _spd_floor(persistent)
    return NuisanceFit(
        coefficients,
        float(selected_ridge),
        wave_effects,
        call_covariance,
        request_covariance,
        repeat_covariance,
        persistent_covariance,
        call_loading,
        request_loading,
        persistent_loading,
        evaluate_prediction,
        means[evaluate] - evaluate_prediction,
        counts[evaluate],
        observed_calls[evaluate],
    )


def candidate_covariance(nuisance, item_kernel, rho, repeat_counts=None):
    item = rho_matched_correlation(item_kernel, rho)
    counts = nuisance.evaluate_repeat_counts if repeat_counts is None else np.asarray(repeat_counts, dtype=float)
    if counts.ndim == 2 and counts.shape == (ROWS_PER_COMPONENT, 2):
        counts = counts[None, :, :]
    if counts.ndim != 3 or counts.shape[1:] != (ROWS_PER_COMPONENT, 2) or np.any(counts < 1):
        raise ValueError("repeat counts must have shape [N,3,2] and be positive")
    base = np.kron(item, nuisance.persistent_covariance)
    output = np.repeat(base[None, :, :], len(counts), axis=0)
    for row in range(ROWS_PER_COMPONENT):
        for family, channel_start in enumerate((0, 2)):
            start = row * CHANNELS + channel_start
            sl = slice(start, start + 2)
            family_sl = slice(channel_start, channel_start + 2)
            output[:, sl, sl] += (
                nuisance.repeat_covariance[None, family_sl, family_sl]
                / counts[:, row, family, None, None]
            )
    output = 0.5 * (output + np.swapaxes(output, -1, -2))
    # The batched Cholesky is both a fail-closed PSD check and substantially
    # cheaper than a Python loop at the preregistered G=800 ceiling.
    np.linalg.cholesky(output)
    return output


def component_gaussian_nll(residuals, covariance):
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 3 or residuals.shape[1:] != (ROWS_PER_COMPONENT, CHANNELS):
        raise ValueError("residuals must have shape [G,3,4]")
    flat = residuals.reshape(len(residuals), -1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 2:
        covariance = np.repeat(covariance[None, :, :], len(flat), axis=0)
    if covariance.shape != (len(flat), flat.shape[1], flat.shape[1]):
        raise ValueError("covariance must be [12,12] or [G,12,12]")
    factor = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(factor, flat[..., None])[..., 0]
    logdet = 2.0 * np.sum(
        np.log(np.diagonal(factor, axis1=-2, axis2=-1)), axis=1
    )
    return 0.5 * (
        np.sum(whitened * whitened, axis=1)
        + logdet
        + flat.shape[1] * math.log(2.0 * math.pi)
    ) / flat.shape[1]


def prompt_block_candidate_covariance(
    nuisance,
    item_kernel,
    rho,
    positions,
):
    """Joint repeat-mean covariance for one stable prompt block.

    The diagonal component blocks contain persistent item covariance plus the
    marginal row-call and shared-request sampling variance.  Off-component
    blocks contain the fitted request covariance multiplied by the overlap of
    the recorded repeat schedules.  Thus a shared prompt request changes the
    Gaussian score itself; clustering is not its only acknowledgement.
    """
    positions = np.asarray(positions, dtype=int)
    if positions.ndim != 1 or not len(positions):
        raise ValueError("positions must identify at least one component")
    marginal = candidate_covariance(
        nuisance,
        item_kernel,
        rho,
        nuisance.evaluate_repeat_counts[positions],
    )
    component_dimension = ROWS_PER_COMPONENT * CHANNELS
    output = np.zeros((
        len(positions) * component_dimension,
        len(positions) * component_dimension,
    ))
    for local, covariance in enumerate(marginal):
        sl = slice(local * component_dimension, (local + 1) * component_dimension)
        output[sl, sl] = covariance
    observed = nuisance.evaluate_observed_calls[positions]
    counts = nuisance.evaluate_repeat_counts[positions]
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            for row in range(ROWS_PER_COMPONENT):
                for family, channel_start in enumerate((0, 2)):
                    overlap = float(np.sum(
                        observed[left, row, :, family]
                        & observed[right, row, :, family]
                    ))
                    coefficient = overlap / (
                        counts[left, row, family] * counts[right, row, family]
                    )
                    left_start = left * component_dimension + row * CHANNELS + channel_start
                    right_start = right * component_dimension + row * CHANNELS + channel_start
                    left_sl = slice(left_start, left_start + 2)
                    right_sl = slice(right_start, right_start + 2)
                    family_sl = slice(channel_start, channel_start + 2)
                    cross = nuisance.request_covariance[family_sl, family_sl] * coefficient
                    output[left_sl, right_sl] = cross
                    output[right_sl, left_sl] = cross.T
    output = _symmetric(output)
    np.linalg.cholesky(output)
    return output


def _prompt_block_positions(prompt_block_ids):
    prompt_block_ids = np.asarray(prompt_block_ids, dtype=int)
    if prompt_block_ids.ndim != 1:
        raise ValueError("prompt block IDs must be a vector")
    return tuple(np.flatnonzero(prompt_block_ids == block) for block in np.unique(prompt_block_ids))


def prompt_block_gaussian_nll(
    residuals,
    nuisance,
    item_kernel,
    rho,
    prompt_block_ids,
):
    """Joint composite residual NLL, returned on an equal-component scale."""
    residuals = np.asarray(residuals, dtype=float)
    output = np.empty(len(residuals), dtype=float)
    for positions in _prompt_block_positions(prompt_block_ids):
        covariance = prompt_block_candidate_covariance(
            nuisance, item_kernel, rho, positions
        )
        vector = residuals[positions].reshape(-1)
        factor = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factor, vector)
        logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
        score = 0.5 * (
            whitened @ whitened
            + logdet
            + len(vector) * math.log(2.0 * math.pi)
        ) / len(vector)
        output[positions] = score
    return output


def draw_latent_states(component_count, seed):
    factor = np.linalg.cholesky(PRIOR_STATE_COVARIANCE)
    return np.random.default_rng(seed).standard_normal(
        (component_count, ROWS_PER_COMPONENT, STATE_CHANNELS)
    ) @ factor.T


def posterior_state_nll(residuals, covariance, states):
    """Score true latent states under a candidate linear-Gaussian posterior."""
    residuals = np.asarray(residuals, dtype=float)
    states = np.asarray(states, dtype=float)
    if states.shape != (len(residuals), ROWS_PER_COMPONENT, STATE_CHANNELS):
        raise ValueError("states must have shape [G,3,2]")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 2:
        covariance = np.repeat(covariance[None], len(residuals), axis=0)
    design = np.kron(np.eye(ROWS_PER_COMPONENT), MEASUREMENT_DESIGN)
    prior = np.kron(np.eye(ROWS_PER_COMPONENT), PRIOR_STATE_COVARIANCE)
    prior_precision = np.linalg.solve(prior, np.eye(len(prior)))
    output = []
    for error, truth, noise_covariance in zip(residuals, states, covariance):
        truth = truth.reshape(-1)
        observed = design @ truth + error.reshape(-1)
        solved_design = np.linalg.solve(noise_covariance, design)
        precision = prior_precision + design.T @ solved_design
        posterior_covariance = np.linalg.solve(precision, np.eye(len(precision)))
        posterior_mean = posterior_covariance @ (
            design.T @ np.linalg.solve(noise_covariance, observed)
        )
        delta = truth - posterior_mean
        sign, logdet = np.linalg.slogdet(posterior_covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("posterior covariance is not positive definite")
        output.append(0.5 * (
            delta @ precision @ delta + logdet + len(delta) * math.log(2.0 * math.pi)
        ) / len(delta))
    return np.asarray(output)


def prompt_block_posterior_state_nll(
    residuals,
    nuisance,
    item_kernel,
    rho,
    states,
    prompt_block_ids,
):
    """Joint latent-state posterior NLL under the prompt schedule covariance."""
    residuals = np.asarray(residuals, dtype=float)
    states = np.asarray(states, dtype=float)
    if states.shape != (len(residuals), ROWS_PER_COMPONENT, STATE_CHANNELS):
        raise ValueError("states must have shape [G,3,2]")
    component_design = np.kron(np.eye(ROWS_PER_COMPONENT), MEASUREMENT_DESIGN)
    component_prior = np.kron(np.eye(ROWS_PER_COMPONENT), PRIOR_STATE_COVARIANCE)
    output = np.empty(len(residuals), dtype=float)
    for positions in _prompt_block_positions(prompt_block_ids):
        noise_covariance = prompt_block_candidate_covariance(
            nuisance, item_kernel, rho, positions
        )
        design = np.kron(np.eye(len(positions)), component_design)
        prior = np.kron(np.eye(len(positions)), component_prior)
        prior_precision = np.linalg.solve(prior, np.eye(len(prior)))
        truth = states[positions].reshape(-1)
        observed = design @ truth + residuals[positions].reshape(-1)
        solved_design = np.linalg.solve(noise_covariance, design)
        precision = prior_precision + design.T @ solved_design
        posterior_covariance = np.linalg.solve(precision, np.eye(len(precision)))
        posterior_mean = posterior_covariance @ (
            design.T @ np.linalg.solve(noise_covariance, observed)
        )
        delta = truth - posterior_mean
        sign, logdet = np.linalg.slogdet(posterior_covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("posterior covariance is not positive definite")
        score = 0.5 * (
            delta @ precision @ delta
            + logdet
            + len(delta) * math.log(2.0 * math.pi)
        ) / len(delta)
        output[positions] = score
    return output


@dataclass(frozen=True)
class Candidate:
    gamma: float
    rho: float

    @property
    def is_block(self):
        return self.rho == 0.0


BLOCK_CANDIDATE = Candidate(0.5, 0.0)


@dataclass(frozen=True)
class CandidateSummary:
    candidate: Candidate
    macro_gain: float
    positive_folds: int
    fold_gains: Tuple[float, ...]


@dataclass(frozen=True)
class InnerSearch:
    selected: Candidate
    summaries: Tuple[CandidateSummary, ...]
    maximum_eligible_gain: float


def candidate_grid(gammas=DEFAULT_GAMMAS, rhos=DEFAULT_RHOS):
    gammas = tuple(sorted(set(map(float, gammas))))
    rhos = tuple(sorted(set(map(float, rhos))))
    if 0.0 not in rhos:
        raise ValueError("rho grid must contain block rho=0")
    output = [BLOCK_CANDIDATE]
    for gamma in gammas:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma values must lie in [0,1]")
        for rho in rhos:
            if rho > 0.0:
                # Declare path eligibility before any outcome is scored.
                rho_matched_correlation(gamma_item_kernel(gamma), rho)
                output.append(Candidate(gamma, rho))
    return tuple(output)


def inner_candidate_search(
    field,
    geometry,
    outer_fold,
    *,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
):
    candidates = candidate_grid(gammas, rhos)
    gains: Dict[Candidate, list] = {candidate: [] for candidate in candidates}
    for fit, held in outer_fold.inner:
        nuisance = fit_repeat_nuisance(
            field, geometry, fit, held, mean_ridges=mean_ridges, shrinkage=shrinkage
        )
        held_prompt_blocks = geometry.prompt_block_index[held]
        block_nll = prompt_block_gaussian_nll(
            nuisance.evaluate_centered_means,
            nuisance,
            gamma_item_kernel(0.5, geometry.kernels),
            0.0,
            held_prompt_blocks,
        )
        for candidate in candidates:
            if candidate.is_block:
                candidate_nll = block_nll
            else:
                candidate_nll = prompt_block_gaussian_nll(
                    nuisance.evaluate_centered_means,
                    nuisance,
                    gamma_item_kernel(candidate.gamma, geometry.kernels),
                    candidate.rho,
                    held_prompt_blocks,
                )
            gains[candidate].append(float(np.mean(block_nll - candidate_nll)))
    summaries = tuple(CandidateSummary(
        candidate,
        float(np.mean(values)),
        int(np.sum(np.asarray(values) > 0.0)),
        tuple(values),
    ) for candidate, values in gains.items())
    eligible = [
        row for row in summaries
        if not row.candidate.is_block and row.macro_gain > 0.0 and row.positive_folds >= 2
    ]
    if not eligible:
        return InnerSearch(BLOCK_CANDIDATE, summaries, 0.0)
    best = min(eligible, key=lambda row: (
        -row.macro_gain, row.candidate.rho, abs(row.candidate.gamma - 0.5), row.candidate.gamma
    ))
    return InnerSearch(best.candidate, summaries, float(max(row.macro_gain for row in eligible)))


def finite_null_maximum_threshold(maxima, confidence=0.95):
    values = np.asarray(maxima, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("null maxima must be a nonempty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    rank = min(int(math.ceil(confidence * (len(values) + 1))), len(values))
    return float(np.sort(values)[rank - 1]), rank


def select_strictly_calibrated(search, threshold):
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    rejected = bool(not search.selected.is_block and search.maximum_eligible_gain > threshold)
    return (search.selected if rejected else BLOCK_CANDIDATE), rejected


def _replicate_geometry_and_splits(component_count, seed, max_prompt_rows=MAX_PROMPT_ROWS):
    splits = component_splits(
        component_count,
        seed=derive_seed(seed, "splits"),
        max_prompt_rows=max_prompt_rows,
    )
    geometry = build_campaign_geometry(
        component_count,
        derive_seed(seed, "geometry"),
        prompt_blocks=splits.prompt_blocks,
        max_prompt_rows=max_prompt_rows,
    )
    return geometry, splits


def calibrate_synthetic_selector_null(
    component_count,
    *,
    repeats,
    draws,
    seed,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
    confidence=0.95,
    missing_rate=0.02,
    max_prompt_rows=MAX_PROMPT_ROWS,
):
    if draws < 1:
        raise ValueError("draws must be positive")
    maxima = []
    for draw in range(draws):
        replicate_seed = derive_seed(seed, "null", draw)
        joint_search_maxima = []
        for corpus in SYNTHETIC_CORPORA:
            corpus_seed = derive_seed(replicate_seed, corpus)
            geometry, splits = _replicate_geometry_and_splits(
                component_count, corpus_seed, max_prompt_rows
            )
            field = draw_repeated_field(
                geometry,
                SCENARIO_BY_NAME["block_null"],
                repeats,
                derive_seed(corpus_seed, "field"),
                missing_rate=missing_rate,
            )
            searches = [inner_candidate_search(
                field,
                geometry,
                fold,
                gammas=gammas,
                rhos=rhos,
                mean_ridges=mean_ridges,
                shrinkage=shrinkage,
            ) for fold in splits.outer]
            joint_search_maxima.extend(
                search.maximum_eligible_gain for search in searches
            )
        maxima.append(max(joint_search_maxima))
    values = np.asarray(maxima)
    threshold, rank = finite_null_maximum_threshold(values, confidence)
    return values, threshold, rank


def prompt_block_multiplier_simultaneous_lower_bounds(
    values,
    prompt_block_ids,
    *,
    confidence=0.95,
    draws=999,
    seed=0,
):
    """One-sided max-t bounds with stable prompt blocks as inference clusters.

    The point estimate remains the preregistered equal-component mean.  Wild
    multiplier perturbations sum centered component influences inside each
    request-connected prompt block before signs are drawn, so batching cannot
    masquerade as independent component information.
    """
    lower, critical, cluster_counts = multicorpus_prompt_block_lower_bounds(
        (values,),
        (prompt_block_ids,),
        confidence=confidence,
        draws=draws,
        seed=seed,
    )
    return lower[0], critical, cluster_counts[0]


def multicorpus_prompt_block_lower_bounds(
    corpus_values,
    corpus_prompt_block_ids,
    *,
    confidence=0.95,
    draws=999,
    seed=0,
):
    """Simultaneous lower bounds across endpoints and required corpora."""
    corpus_values = tuple(np.asarray(value, dtype=float) for value in corpus_values)
    corpus_prompt_block_ids = tuple(
        np.asarray(value, dtype=int) for value in corpus_prompt_block_ids
    )
    if not corpus_values or len(corpus_values) != len(corpus_prompt_block_ids):
        raise ValueError("corpus values and block-ID collections must align")
    endpoint_count = corpus_values[0].shape[1] if corpus_values[0].ndim == 2 else 0
    for values, prompt_block_ids in zip(corpus_values, corpus_prompt_block_ids):
        if (
            values.ndim != 2
            or values.shape[0] < 2
            or values.shape[1] != endpoint_count
            or endpoint_count < 2
            or not np.isfinite(values).all()
        ):
            raise ValueError("values must be finite [components,endpoints] matrices")
        if prompt_block_ids.shape != (len(values),):
            raise ValueError("prompt block IDs must have one entry per component")
    if draws < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid multiplier configuration")
    point = np.concatenate([values.mean(axis=0) for values in corpus_values])
    cluster_rows = []
    cluster_counts = []
    total_endpoints = len(corpus_values) * endpoint_count
    for corpus_index, (values, prompt_block_ids) in enumerate(zip(
        corpus_values, corpus_prompt_block_ids
    )):
        blocks = np.unique(prompt_block_ids)
        if len(blocks) < 2:
            raise ValueError("at least two independent prompt blocks are required")
        cluster_counts.append(int(len(blocks)))
        centered = values - values.mean(axis=0)
        correction = math.sqrt(len(blocks) / (len(blocks) - 1.0))
        target = slice(corpus_index * endpoint_count, (corpus_index + 1) * endpoint_count)
        for block in blocks:
            row = np.zeros(total_endpoints, dtype=float)
            row[target] = (
                np.sum(centered[prompt_block_ids == block], axis=0)
                / len(values)
                * correction
            )
            cluster_rows.append(row)
    cluster_influence = np.asarray(cluster_rows)
    standard_error = np.sqrt(np.sum(cluster_influence * cluster_influence, axis=0))
    safe = np.where(standard_error > np.finfo(float).eps, standard_error, 1.0)
    signs = np.random.default_rng(seed).choice(
        (-1.0, 1.0), size=(draws, len(cluster_influence))
    )
    deviations = signs @ cluster_influence
    max_stat = np.max(-deviations / safe[None, :], axis=1)
    critical = float(np.quantile(max_stat, confidence))
    lower = point - critical * standard_error
    lower = np.where(standard_error > np.finfo(float).eps, lower, point)
    return lower.reshape(len(corpus_values), endpoint_count), critical, tuple(cluster_counts)


@dataclass(frozen=True)
class PowerReplicate:
    scenario: str
    selected: Tuple[Tuple[Candidate, ...], ...]
    corpus_selector_rejected: Tuple[bool, ...]
    familywise_rejected: bool
    maximum_inner_gain: float
    endpoint_component_gains: np.ndarray
    endpoint_mean_gains: np.ndarray
    endpoint_lower_bounds: np.ndarray
    multiplier_critical_value: float
    inference_prompt_blocks: Tuple[int, ...]
    topology_component_advantage: np.ndarray | None
    topology_truth_beats_derangement: bool | None
    promoted: bool
    call_loading: float
    persistent_loading: float
    request_loading: float


@dataclass(frozen=True)
class _CorpusPowerReplicate:
    selected: Tuple[Candidate, ...]
    selector_rejected: bool
    maximum_inner_gain: float
    endpoint_component_gains: np.ndarray
    prompt_block_ids: np.ndarray
    topology_component_advantage: np.ndarray | None
    call_loading: float
    persistent_loading: float
    request_loading: float


def _run_corpus_power_replicate(
    component_count,
    scenario,
    *,
    repeats,
    seed,
    null_threshold,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
    missing_rate=0.02,
    max_prompt_rows=MAX_PROMPT_ROWS,
):
    geometry, splits = _replicate_geometry_and_splits(
        component_count, seed, max_prompt_rows
    )
    field = draw_repeated_field(
        geometry, scenario, repeats, derive_seed(seed, "field"), missing_rate=missing_rate
    )
    states = draw_latent_states(component_count, derive_seed(seed, "states"))
    endpoint_gains = np.empty((component_count, len(PRIMARY_ENDPOINTS)), dtype=float)
    topology_advantage = np.empty(component_count, dtype=float) if (
        scenario.truth_rho > 0.0 and not scenario.deranged_truth
    ) else None
    selected_by_fold = []
    rejected_by_fold = []
    maxima = []
    call_loadings = []
    persistent_loadings = []
    request_loadings = []
    for fold in splits.outer:
        search = inner_candidate_search(
            field,
            geometry,
            fold,
            gammas=gammas,
            rhos=rhos,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
        )
        selected, rejected = select_strictly_calibrated(search, null_threshold)
        selected_by_fold.append(selected)
        rejected_by_fold.append(rejected)
        maxima.append(search.maximum_eligible_gain)
        nuisance = fit_repeat_nuisance(
            field,
            geometry,
            fold.train,
            fold.held,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
        )
        call_loadings.append(nuisance.call_loading)
        persistent_loadings.append(nuisance.persistent_loading)
        request_loadings.append(nuisance.request_loading)
        held_prompt_blocks = geometry.prompt_block_index[fold.held]
        block_kernel = gamma_item_kernel(0.5, geometry.kernels)
        selected_kernel = (
            block_kernel if selected.is_block
            else gamma_item_kernel(selected.gamma, geometry.kernels)
        )
        block_residual = prompt_block_gaussian_nll(
            nuisance.evaluate_centered_means,
            nuisance,
            block_kernel,
            0.0,
            held_prompt_blocks,
        )
        selected_residual = prompt_block_gaussian_nll(
            nuisance.evaluate_centered_means,
            nuisance,
            selected_kernel,
            selected.rho,
            held_prompt_blocks,
        )
        block_posterior = prompt_block_posterior_state_nll(
            nuisance.evaluate_centered_means,
            nuisance,
            block_kernel,
            0.0,
            states[fold.held],
            held_prompt_blocks,
        )
        selected_posterior = prompt_block_posterior_state_nll(
            nuisance.evaluate_centered_means,
            nuisance,
            selected_kernel,
            selected.rho,
            states[fold.held],
            held_prompt_blocks,
        )
        endpoint_gains[fold.held, 0] = block_residual - selected_residual
        endpoint_gains[fold.held, 1] = block_posterior - selected_posterior
        if topology_advantage is not None:
            truth_kernel = scenario_item_kernel(geometry, scenario)
            topology_advantage[fold.held] = (
                prompt_block_gaussian_nll(
                    nuisance.evaluate_centered_means,
                    nuisance,
                    deranged_item_kernel(truth_kernel),
                    scenario.truth_rho,
                    held_prompt_blocks,
                )
                - prompt_block_gaussian_nll(
                    nuisance.evaluate_centered_means,
                    nuisance,
                    truth_kernel,
                    scenario.truth_rho,
                    held_prompt_blocks,
                )
            )
    return _CorpusPowerReplicate(
        tuple(selected_by_fold),
        bool(any(rejected_by_fold)),
        float(max(maxima)),
        endpoint_gains,
        geometry.prompt_block_index,
        topology_advantage,
        float(max(call_loadings)),
        float(max(persistent_loadings)),
        float(max(request_loadings)),
    )


def run_power_replicate(
    component_count,
    scenario,
    *,
    repeats,
    seed,
    null_threshold,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
    confidence=0.95,
    multiplier_draws=999,
    missing_rate=0.02,
    max_prompt_rows=MAX_PROMPT_ROWS,
):
    """Run one joint event spanning both required synthetic corpora."""
    corpus_records = tuple(
        _run_corpus_power_replicate(
            component_count,
            scenario,
            repeats=repeats,
            seed=derive_seed(seed, corpus),
            null_threshold=null_threshold,
            gammas=gammas,
            rhos=rhos,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
            missing_rate=missing_rate,
            max_prompt_rows=max_prompt_rows,
        )
        for corpus in SYNTHETIC_CORPORA
    )
    endpoint_gains = np.stack([
        record.endpoint_component_gains for record in corpus_records
    ])
    lower, critical, inference_prompt_blocks = multicorpus_prompt_block_lower_bounds(
        tuple(record.endpoint_component_gains for record in corpus_records),
        tuple(record.prompt_block_ids for record in corpus_records),
        confidence=confidence,
        draws=multiplier_draws,
        seed=derive_seed(seed, "joint-multiplier"),
    )
    corpus_selector_rejected = tuple(
        record.selector_rejected for record in corpus_records
    )
    familywise_rejected = bool(all(corpus_selector_rejected))
    if scenario.truth_rho > 0.0 and not scenario.deranged_truth:
        topology_advantage = np.stack([
            record.topology_component_advantage for record in corpus_records
        ])
        topology_beats = bool(np.all(np.mean(topology_advantage, axis=1) > 0.0))
    else:
        topology_advantage = None
        topology_beats = None
    promoted = bool(familywise_rejected and np.all(lower > 0.0))
    return PowerReplicate(
        scenario.name,
        tuple(record.selected for record in corpus_records),
        corpus_selector_rejected,
        familywise_rejected,
        float(max(record.maximum_inner_gain for record in corpus_records)),
        endpoint_gains,
        endpoint_gains.mean(axis=1),
        lower,
        critical,
        inference_prompt_blocks,
        topology_advantage,
        topology_beats,
        promoted,
        float(max(record.call_loading for record in corpus_records)),
        float(max(record.persistent_loading for record in corpus_records)),
        float(max(record.request_loading for record in corpus_records)),
    )


def summary(values: Iterable[float]):
    values = np.asarray(tuple(values), dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("summary needs a nonempty finite vector")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def aggregate_power_records(records):
    if not records:
        raise ValueError("at least one power record is required")
    if len({row.scenario for row in records}) != 1:
        raise ValueError("power records must share one scenario")
    selected = [
        candidate
        for row in records
        for corpus_selected in row.selected
        for candidate in corpus_selected
    ]
    topology = [
        row.topology_truth_beats_derangement for row in records
        if row.topology_truth_beats_derangement is not None
    ]
    return {
        "scenario": records[0].scenario,
        "replicates": len(records),
        "both_corpus_selector_rejections": int(sum(row.familywise_rejected for row in records)),
        "both_corpus_selector_rejection_rate": float(np.mean([row.familywise_rejected for row in records])),
        "per_corpus_selector_rejection_rate": {
            corpus: float(np.mean([
                row.corpus_selector_rejected[index] for row in records
            ]))
            for index, corpus in enumerate(SYNTHETIC_CORPORA)
        },
        "joint_synthetic_primary_events": int(sum(row.promoted for row in records)),
        "joint_synthetic_primary_event_rate": float(np.mean([row.promoted for row in records])),
        "endpoint_mean_gain_per_scalar": {
            corpus: {
                endpoint: summary(
                    row.endpoint_mean_gains[corpus_index, endpoint_index]
                    for row in records
                )
                for endpoint_index, endpoint in enumerate(PRIMARY_ENDPOINTS)
            }
            for corpus_index, corpus in enumerate(SYNTHETIC_CORPORA)
        },
        "endpoint_simultaneous_lower_bound": {
            corpus: {
                endpoint: summary(
                    row.endpoint_lower_bounds[corpus_index, endpoint_index]
                    for row in records
                )
                for endpoint_index, endpoint in enumerate(PRIMARY_ENDPOINTS)
            }
            for corpus_index, corpus in enumerate(SYNTHETIC_CORPORA)
        },
        "multiplier_max_t_critical_value": summary(
            row.multiplier_critical_value for row in records
        ),
        "inference_prompt_blocks": {
            corpus: {
                "minimum": int(min(
                    row.inference_prompt_blocks[index] for row in records
                )),
                "maximum": int(max(
                    row.inference_prompt_blocks[index] for row in records
                )),
            }
            for index, corpus in enumerate(SYNTHETIC_CORPORA)
        },
        "topology_truth_beats_derangement_rate": (
            float(np.mean(topology)) if topology else None
        ),
        "selected_gamma_counts_across_outer_folds": {
            str(value): int(sum(candidate.gamma == value for candidate in selected))
            for value in sorted({candidate.gamma for candidate in selected})
        },
        "selected_rho_counts_across_outer_folds": {
            str(value): int(sum(candidate.rho == value for candidate in selected))
            for value in sorted({candidate.rho for candidate in selected})
        },
        "maximum_call_loading": float(max(row.call_loading for row in records)),
        "maximum_persistent_loading": float(max(row.persistent_loading for row in records)),
        "maximum_request_loading": float(max(row.request_loading for row in records)),
    }
