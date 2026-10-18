#!/usr/bin/env python3
"""Tests for the component-safe adjacent residual pilot primitives."""
import numpy as np
import pytest

from adjacent_residual_pilot import (
    AnchorContrast,
    adjacency_feature_kernel,
    anchor_matched_contrasts,
    component_balanced_folds,
    component_contrast_estimate,
    component_multiplier_stability,
    endpoint_component_ids,
    marginal_preserving_adjacency_covariance,
    positive_row_pairs,
    principal_whiten_rows,
    within_descendant_derangement,
)
from run_adjacent_residual_synthetic import simulate_component_contrasts


def test_endpoint_components_and_balanced_folds_are_disjoint_and_deterministic():
    pairs, tags = [], []
    neighbors = {}
    for component in range(9):
        left = f"left-{component}"
        first, second = f"root-{component}-a", f"root-{component}-b"
        pairs.extend([(left, first), (left, second)])
        tags.extend(["campaign_h1", "campaign_h2"])
        neighbors[first] = {second}
        neighbors[second] = {first}
    positives = positive_row_pairs(pairs, neighbors)
    first = component_balanced_folds(pairs, tags, positives, n_folds=3, seed=20)
    second = component_balanced_folds(pairs, tags, positives, n_folds=3, seed=20)

    assert first.component_count == 9
    assert first.largest_component == 2
    assert [fold.tolist() for fold in first.folds] == [fold.tolist() for fold in second.folds]
    assert sorted(np.concatenate(first.folds).tolist()) == list(range(18))
    assert [row["positive_pairs"] for row in first.fold_diagnostics] == [3, 3, 3]
    for left in range(3):
        left_nodes = {node for row in first.folds[left] for node in pairs[row]}
        for right in range(left + 1, 3):
            right_nodes = {node for row in first.folds[right] for node in pairs[row]}
            assert left_nodes.isdisjoint(right_nodes)


def test_anchor_controls_share_descendant_and_one_positive_row():
    pairs = [("x", "r1"), ("x", "r2"), ("x", "r3"), ("x", "r4")]
    tags = ["campaign_h1", "campaign_h2", "campaign_h3", "campaign_h4"]
    neighbors = {"r1": {"r2"}, "r2": {"r1"}, "r3": set(), "r4": set()}
    degrees = {root: len(values) for root, values in neighbors.items()}
    semantic = np.eye(4)
    records, excluded = anchor_matched_contrasts(
        pairs, tags, neighbors, degrees, semantic, maximum_controls=3
    )

    assert not excluded
    assert len(records) == 1
    record = records[0]
    assert record.positive == (0, 1)
    assert len(record.controls) == 3
    assert all(set(pair) & {0, 1} for pair in record.controls)
    assert all(pairs[a][0] == pairs[b][0] == "x" for a, b in record.controls)
    assert all(not (pairs[b][1] in neighbors.get(pairs[a][1], set())) for a, b in record.controls)


def test_principal_whitening_uses_symmetric_inverse_root():
    covariance = np.array([
        [2.0, 0.3, 0.0, 0.0],
        [0.3, 1.0, 0.1, 0.0],
        [0.0, 0.1, 1.5, 0.2],
        [0.0, 0.0, 0.2, 0.8],
    ])
    values, vectors = np.linalg.eigh(covariance)
    symmetric_root = (vectors * np.sqrt(values)) @ vectors.T
    whitened = principal_whiten_rows(symmetric_root, covariance)
    np.testing.assert_allclose(whitened, np.eye(4), atol=1e-12)


def test_component_macro_contrast_and_multiplier_stability():
    rows = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    records = (
        AnchorContrast((0, 1), ((0, 2), (1, 3)), 10, ("h1", "h2"), 1, 0, 0),
        AnchorContrast((4, 5), ((4, 6), (5, 7)), 20, ("h1", "h2"), 1, 0, 0),
    )
    estimate = component_contrast_estimate(rows, records)

    assert estimate.record_count == 2
    assert estimate.component_ids.tolist() == [10, 20]
    assert estimate.primary_trace > 0.0
    band = component_multiplier_stability(
        np.repeat(estimate.contrast_matrices, 20, axis=0), draws=999, seed=8
    )
    assert band.components == 40
    assert band.gate_evaluable
    assert band.estimate[0] == pytest.approx(estimate.primary_trace)
    assert band.leave_one_component_out_positive_fraction == 1.0


def test_adjacency_gram_and_covariance_path_are_psd_and_marginal_matched():
    pairs = [("x", "a"), ("x", "b"), ("x", "c"), ("y", "a")]
    neighbors = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    kernel = adjacency_feature_kernel(pairs, pairs, neighbors)

    np.testing.assert_allclose(np.diag(kernel), 1.0)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-12
    assert kernel[0, 3] == 0.0  # descendant role keeps otherwise identical roots separate
    deranged, permutation = within_descendant_derangement(kernel, pairs, seed=71)
    np.testing.assert_allclose(np.linalg.eigvalsh(deranged), np.linalg.eigvalsh(kernel))
    np.testing.assert_allclose(np.diag(deranged), 1.0)
    assert permutation[0] != 0 and permutation[3] == 3
    block = np.array([[2.0, 0.2], [0.2, 1.0]])
    covariance = marginal_preserving_adjacency_covariance(kernel, block, 0.35)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    for row in range(len(pairs)):
        np.testing.assert_allclose(
            covariance[row * 2:(row + 1) * 2, row * 2:(row + 1) * 2], block
        )


def test_component_and_covariance_input_guards():
    assert endpoint_component_ids([("a", "b"), ("b", "c"), ("x", "y")]).tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match="unit diagonal"):
        marginal_preserving_adjacency_covariance(np.eye(2) * 2.0, np.eye(2), 0.5)
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        principal_whiten_rows(np.zeros((2, 4)), np.zeros((4, 4)))


def test_synthetic_anchor_contrast_recovers_planted_trace():
    matrices = simulate_component_contrasts(2000, 0.20, seed=901)
    assert np.trace(np.mean(matrices, axis=0)) / 4.0 == pytest.approx(0.20, abs=0.025)
    with pytest.raises(ValueError, match="components"):
        simulate_component_contrasts(1, 0.20, seed=1)
