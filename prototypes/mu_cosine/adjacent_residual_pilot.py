#!/usr/bin/env python3
"""Component-safe primitives for the adjacent-row conditional-residual pilot.

The functions in this module are outcome-agnostic except for whitening and the
final cross-product statistic.  They deliberately return conditional stability
summaries rather than population confidence intervals; see
``DESIGN_adjacent_residual_pilot.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


VECH = tuple((i, j) for i in range(4) for j in range(i, 4))
TAG_ORDER = {
    "campaign_sib": 6.0,
    "campaign_cous": 7.0,
    "campaign_rand": 8.0,
}


def _stable_key(value):
    return type(value).__qualname__, repr(value)


def _finite_matrix(name, value, *, rows=None, cols=None):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{name} has {value.shape[0]} rows, expected {rows}")
    if cols is not None and value.shape[1] != cols:
        raise ValueError(f"{name} has {value.shape[1]} columns, expected {cols}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def endpoint_component_ids(pairs):
    """Return canonical row-component ids under exact endpoint sharing."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must not be empty")
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            if _stable_key(left) <= _stable_key(right):
                parent[right] = left
            else:
                parent[left] = right

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("every pair must have two endpoints")
        union(pair[0], pair[1])
    roots = [find(pair[0]) for pair in pairs]
    canonical = {
        root: i for i, root in enumerate(sorted(set(roots), key=_stable_key))
    }
    return np.asarray([canonical[root] for root in roots], dtype=int)


def _tag_value(tag):
    tag = str(tag)
    match = re.fullmatch(r"campaign_h([1-9][0-9]*)", tag)
    if match:
        return float(match.group(1))
    return TAG_ORDER.get(tag, 100.0)


def _degree_bin(value):
    value = max(0, int(value))
    return int(math.floor(math.log2(value + 1)))


def _direct_edge(neighbors, left, right):
    return right in neighbors.get(left, ()) or left in neighbors.get(right, ())


def positive_row_pairs(pairs, neighbors):
    """Enumerate same-descendant row pairs whose roots share a direct edge."""
    pairs = list(pairs)
    by_left = {}
    for row, (left, _right) in enumerate(pairs):
        by_left.setdefault(left, []).append(row)
    output = []
    for rows in by_left.values():
        for offset, first in enumerate(rows):
            for second in rows[offset + 1:]:
                if _direct_edge(neighbors, pairs[first][1], pairs[second][1]):
                    output.append((first, second))
    return tuple(output)


@dataclass(frozen=True)
class ComponentFoldAssignment:
    folds: tuple[np.ndarray, ...]
    component_ids: np.ndarray
    component_count: int
    largest_component: int
    fold_diagnostics: tuple[dict, ...]
    seed: int


def component_balanced_folds(pairs, tags, positive_pairs, *, n_folds=5, seed=20):
    """Assign whole endpoint components to deterministic outcome-blind folds."""
    pairs = list(pairs)
    tags = np.asarray(tags)
    if len(tags) != len(pairs):
        raise ValueError("tags must align with pairs")
    if not 2 <= n_folds <= len(pairs):
        raise ValueError("n_folds must be between two and the row count")
    component = endpoint_component_ids(pairs)
    components = sorted(set(component.tolist()))
    if len(components) < n_folds:
        raise ValueError("fewer endpoint components than folds")
    tag_names = sorted({str(value) for value in tags})
    tag_index = {name: i for i, name in enumerate(tag_names)}
    positive_count = np.zeros(len(components), dtype=float)
    for first, second in positive_pairs:
        if component[first] != component[second]:
            raise ValueError("a positive row pair crosses endpoint components")
        positive_count[component[first]] += 1.0
    features = np.zeros((len(components), 2 + len(tag_names)), dtype=float)
    rows_by_component = []
    for value in components:
        rows = np.flatnonzero(component == value)
        rows_by_component.append(rows)
        features[value, 0] = len(rows)
        features[value, 1] = positive_count[value]
        for row in rows:
            features[value, 2 + tag_index[str(tags[row])]] += 1.0
    target = features.sum(axis=0) / n_folds
    scale = np.where(target > 0.0, target, 1.0)
    rng = np.random.default_rng(seed)
    tie = rng.random(len(components))
    order = sorted(
        components,
        key=lambda value: (
            -features[value, 1],
            -features[value, 0],
            -float(np.max(features[value, 2:], initial=0.0)),
            tie[value],
            value,
        ),
    )
    totals = np.zeros((n_folds, features.shape[1]), dtype=float)
    assigned = [[] for _ in range(n_folds)]
    fold_tie = rng.permutation(n_folds)
    fold_rank = np.empty(n_folds, dtype=int)
    fold_rank[fold_tie] = np.arange(n_folds)
    for value in order:
        candidates = []
        for fold in range(n_folds):
            prospective = totals.copy()
            prospective[fold] += features[value]
            score = float(np.sum(((prospective - target) / scale) ** 2))
            candidates.append((score, len(assigned[fold]), fold_rank[fold], fold))
        fold = min(candidates)[-1]
        assigned[fold].append(value)
        totals[fold] += features[value]
    folds, diagnostics = [], []
    for fold, values in enumerate(assigned):
        rows = np.sort(np.concatenate([rows_by_component[value] for value in values]))
        folds.append(rows)
        diagnostics.append({
            "fold": fold,
            "rows": int(len(rows)),
            "components": int(len(values)),
            "positive_pairs": int(totals[fold, 1]),
            "tag_counts": {
                name: int(totals[fold, 2 + at]) for at, name in enumerate(tag_names)
            },
        })
    if sorted(np.concatenate(folds).tolist()) != list(range(len(pairs))):
        raise AssertionError("component folds must partition every row exactly once")
    for left in range(n_folds):
        left_nodes = {node for row in folds[left] for node in pairs[row]}
        for right in range(left + 1, n_folds):
            right_nodes = {node for row in folds[right] for node in pairs[row]}
            if left_nodes & right_nodes:
                raise AssertionError("endpoint components leaked across folds")
    sizes = np.bincount(component)
    return ComponentFoldAssignment(
        tuple(folds),
        component,
        len(components),
        int(np.max(sizes)),
        tuple(diagnostics),
        int(seed),
    )


@dataclass(frozen=True)
class AnchorContrast:
    positive: tuple[int, int]
    controls: tuple[tuple[int, int], ...]
    component: int
    positive_tag_pair: tuple[str, str]
    mean_tag_distance: float
    mean_degree_bin_difference: float
    mean_semantic_distance: float


def anchor_matched_contrasts(
    pairs,
    tags,
    neighbors,
    degrees,
    semantic,
    *,
    maximum_controls=3,
):
    """Match each adjacent-root pair to deterministic anchor-sharing controls."""
    pairs = list(pairs)
    tags = np.asarray(tags)
    semantic = _finite_matrix("semantic", semantic, rows=len(pairs))
    if len(tags) != len(pairs):
        raise ValueError("tags must align with pairs")
    if maximum_controls < 1:
        raise ValueError("maximum_controls must be positive")
    component = endpoint_component_ids(pairs)
    by_left = {}
    for row, (left, _right) in enumerate(pairs):
        by_left.setdefault(left, []).append(row)
    records, excluded = [], []
    for first, second in positive_row_pairs(pairs, neighbors):
        candidates = []
        for anchor, partner in ((first, second), (second, first)):
            partner_root = pairs[partner][1]
            for candidate in by_left[pairs[first][0]]:
                if candidate in (first, second):
                    continue
                candidate_root = pairs[candidate][1]
                if _direct_edge(neighbors, pairs[anchor][1], candidate_root):
                    continue
                tag_distance = abs(_tag_value(tags[partner]) - _tag_value(tags[candidate]))
                degree_distance = abs(
                    _degree_bin(degrees.get(partner_root, 0))
                    - _degree_bin(degrees.get(candidate_root, 0))
                )
                semantic_distance = float(
                    1.0 - np.clip(semantic[partner] @ semantic[candidate], -1.0, 1.0)
                )
                candidates.append((
                    tag_distance,
                    degree_distance,
                    semantic_distance,
                    anchor,
                    candidate,
                ))
        candidates.sort()
        chosen = candidates[:maximum_controls]
        if not chosen:
            excluded.append((first, second))
            continue
        records.append(AnchorContrast(
            (first, second),
            tuple((value[3], value[4]) for value in chosen),
            int(component[first]),
            tuple(sorted((str(tags[first]), str(tags[second])))),
            float(np.mean([value[0] for value in chosen])),
            float(np.mean([value[1] for value in chosen])),
            float(np.mean([value[2] for value in chosen])),
        ))
    return tuple(records), tuple(excluded)


def principal_whiten_rows(residuals, covariance):
    """Whiten row residuals with the unique symmetric inverse square root."""
    residuals = _finite_matrix("residuals", residuals)
    covariance = _finite_matrix(
        "covariance", covariance, rows=residuals.shape[1], cols=residuals.shape[1]
    )
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0.0):
        raise np.linalg.LinAlgError("covariance must be positive definite")
    inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    return residuals @ inverse_root


def _symmetric_cross_product(rows, pair):
    first, second = pair
    value = np.outer(rows[first], rows[second])
    return 0.5 * (value + value.T)


@dataclass(frozen=True)
class ComponentContrastEstimate:
    component_ids: np.ndarray
    contrast_matrices: np.ndarray
    positive_matrices: np.ndarray
    control_matrices: np.ndarray
    contrast: np.ndarray
    positive: np.ndarray
    control: np.ndarray
    primary_trace: float
    edge_weighted_contrast: np.ndarray
    record_count: int


def component_contrast_estimate(whitened_rows, records):
    """Compute anchor-matched matrices, then average endpoint components equally."""
    rows = _finite_matrix("whitened_rows", whitened_rows, cols=4)
    records = tuple(records)
    if not records:
        raise ValueError("records must not be empty")
    by_component = {}
    edge_contrasts = []
    for record in records:
        positive = _symmetric_cross_product(rows, record.positive)
        controls = np.mean([
            _symmetric_cross_product(rows, pair) for pair in record.controls
        ], axis=0)
        contrast = positive - controls
        edge_contrasts.append(contrast)
        by_component.setdefault(record.component, []).append((contrast, positive, controls))
    component_ids = np.asarray(sorted(by_component), dtype=int)
    contrast, positive, control = [], [], []
    for value in component_ids:
        rows_for_component = by_component[int(value)]
        contrast.append(np.mean([row[0] for row in rows_for_component], axis=0))
        positive.append(np.mean([row[1] for row in rows_for_component], axis=0))
        control.append(np.mean([row[2] for row in rows_for_component], axis=0))
    contrast = np.asarray(contrast)
    positive = np.asarray(positive)
    control = np.asarray(control)
    point = np.mean(contrast, axis=0)
    return ComponentContrastEstimate(
        component_ids,
        contrast,
        positive,
        control,
        point,
        np.mean(positive, axis=0),
        np.mean(control, axis=0),
        float(np.trace(point) / 4.0),
        np.mean(edge_contrasts, axis=0),
        len(records),
    )


def _statistic_vector(matrices):
    matrices = np.asarray(matrices, dtype=float)
    trace = np.trace(matrices, axis1=-2, axis2=-1) / 4.0
    cells = np.stack([matrices[..., i, j] for i, j in VECH], axis=-1)
    return np.concatenate([trace[..., None], cells], axis=-1)


@dataclass(frozen=True)
class MultiplierStabilityBand:
    names: tuple[str, ...]
    estimate: np.ndarray
    standard_error: np.ndarray
    pointwise_low: np.ndarray
    pointwise_high: np.ndarray
    simultaneous_low: np.ndarray
    simultaneous_high: np.ndarray
    simultaneous_critical_value: float
    spectral_error_radius: float
    spectral_norm: float
    spectral_norm_lower: float
    spectral_norm_upper: float
    leave_one_component_out_min: float
    leave_one_component_out_max: float
    leave_one_component_out_positive_fraction: float
    components: int
    effective_components: float
    maximum_component_weight: float
    gate_evaluable: bool
    draws: int
    confidence: float


def component_multiplier_stability(component_matrices, *, draws=9999, seed=0, confidence=0.95):
    """Conditional Rademacher multiplier bands for equal component estimates."""
    matrices = np.asarray(component_matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("component_matrices must have shape [G,4,4]")
    if len(matrices) < 2 or draws < 1:
        raise ValueError("at least two components and one draw are required")
    if not 0.0 < confidence < 1.0 or not np.isfinite(matrices).all():
        raise ValueError("confidence must be in (0,1) and matrices must be finite")
    matrices = 0.5 * (matrices + matrices.transpose(0, 2, 1))
    values = _statistic_vector(matrices)
    point = np.mean(values, axis=0)
    centered = values - point
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
    rng = np.random.default_rng(seed)
    multipliers = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    deviations = multipliers @ centered / len(values)
    valid = standard_error > np.finfo(float).eps
    if np.any(valid):
        maximum_t = np.max(np.abs(deviations[:, valid] / standard_error[valid]), axis=1)
        critical = float(np.quantile(maximum_t, confidence))
    else:
        critical = 0.0
    alpha = (1.0 - confidence) / 2.0
    low_deviation, high_deviation = np.quantile(deviations, [alpha, 1.0 - alpha], axis=0)
    centered_matrices = matrices - np.mean(matrices, axis=0)
    matrix_deviation = np.einsum("bg,gij->bij", multipliers, centered_matrices) / len(matrices)
    spectral_deviation = np.linalg.eigvalsh(matrix_deviation)
    spectral_error = np.max(np.abs(spectral_deviation), axis=1)
    radius = float(np.quantile(spectral_error, confidence))
    point_matrix = np.mean(matrices, axis=0)
    point_spectral = float(np.max(np.abs(np.linalg.eigvalsh(point_matrix))))
    leave_one_out = []
    for held in range(len(matrices)):
        reduced = np.mean(np.delete(matrices, held, axis=0), axis=0)
        leave_one_out.append(float(np.trace(reduced) / 4.0))
    component_weight = 1.0 / len(matrices)
    names = ("trace_over_4",) + tuple(f"channel_{i}_{j}" for i, j in VECH)
    return MultiplierStabilityBand(
        names,
        point,
        standard_error,
        point + low_deviation,
        point + high_deviation,
        point - critical * standard_error,
        point + critical * standard_error,
        critical,
        radius,
        point_spectral,
        max(0.0, point_spectral - radius),
        point_spectral + radius,
        float(np.min(leave_one_out)),
        float(np.max(leave_one_out)),
        float(np.mean(np.asarray(leave_one_out) > 0.0)),
        len(matrices),
        float(len(matrices)),
        component_weight,
        bool(len(matrices) >= 30 and component_weight <= 0.10),
        int(draws),
        float(confidence),
    )


def adjacency_feature_kernel(pairs_a, pairs_b, neighbors):
    """PSD-compatible same-descendant closed-root-neighborhood Gram block."""
    pairs_a, pairs_b = list(pairs_a), list(pairs_b)
    closed = {}
    for _left, root in pairs_a + pairs_b:
        closed.setdefault(root, frozenset((root, *neighbors.get(root, ()))))
    output = np.zeros((len(pairs_a), len(pairs_b)), dtype=float)
    for i, (left_a, root_a) in enumerate(pairs_a):
        set_a = closed[root_a]
        for j, (left_b, root_b) in enumerate(pairs_b):
            if left_a != left_b:
                continue
            set_b = closed[root_b]
            output[i, j] = len(set_a & set_b) / math.sqrt(len(set_a) * len(set_b))
    return output


def within_descendant_derangement(kernel, pairs, *, seed):
    """Derange non-singleton descendant blocks; singleton rows remain fixed."""
    kernel = _finite_matrix("kernel", kernel)
    pairs = list(pairs)
    if kernel.shape != (len(pairs), len(pairs)):
        raise ValueError("kernel must align with pairs")
    if not np.allclose(kernel, kernel.T, atol=1e-10):
        raise ValueError("kernel must be symmetric")
    by_left = {}
    for row, (left, _right) in enumerate(pairs):
        by_left.setdefault(left, []).append(row)
    permutation = np.arange(len(pairs))
    rng = np.random.default_rng(seed)
    for left in sorted(by_left, key=_stable_key):
        rows = np.asarray(by_left[left], dtype=int)
        if len(rows) < 2:
            continue
        order = rows[rng.permutation(len(rows))]
        shift = int(rng.integers(1, len(rows)))
        permutation[order] = np.roll(order, shift)
    return kernel[np.ix_(permutation, permutation)], permutation


def marginal_preserving_adjacency_covariance(kernel, block_covariance, alpha):
    """Return ``C_alpha kron B`` with exact within-row block ``B``."""
    kernel = _finite_matrix("kernel", kernel)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if not np.isfinite(kernel).all() or not np.allclose(kernel, kernel.T, atol=1e-10):
        raise ValueError("kernel must be finite and symmetric")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not np.allclose(np.diag(kernel), 1.0, atol=1e-10):
        raise ValueError("kernel must have unit diagonal")
    block = _finite_matrix("block_covariance", block_covariance)
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError("block_covariance must be square")
    if not np.isfinite(block).all() or not np.allclose(block, block.T, atol=1e-10):
        raise ValueError("block_covariance must be finite and symmetric")
    correlation = (1.0 - alpha) * np.eye(len(kernel)) + alpha * kernel
    if np.min(np.linalg.eigvalsh(correlation)) < -1e-10:
        raise ValueError("kernel path is not positive semidefinite")
    if np.min(np.linalg.eigvalsh(block)) <= 0.0:
        raise ValueError("block_covariance must be positive definite")
    return np.kron(correlation, block)
