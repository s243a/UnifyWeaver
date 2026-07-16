#!/usr/bin/env python3
"""Regression tests for PSD-safe graph and embedding item geometries."""
import numpy as np
import pytest

from graph_geometry import (
    closed_neighborhood_kernel,
    convex_kernel_mixture,
    cumulative_walk_feature_kernel,
    cumulative_walk_feature_map,
    descendant_gated_item_kernel,
    embedding_item_kernel,
    heat_kernel_reference,
    kernel_diagnostics,
    median_pairwise_distance,
    normalize_psd_kernel,
    resolvent_kernel_reference,
    role_aware_pair_features,
    schur_kernel_product,
    symmetric_normalized_laplacian,
    walk_feature_kernel,
    walk_feature_map,
)


def _neighbors(edges, extra=()):
    output = {node: set() for node in extra}
    for left, right in edges:
        output.setdefault(left, set()).add(right)
        output.setdefault(right, set()).add(left)
    return {node: frozenset(values) for node, values in output.items()}


def _assert_correlation_kernel(value):
    assert np.all(np.isfinite(value))
    assert np.allclose(value, value.T, atol=1e-12)
    assert np.allclose(np.diag(value), 1.0, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(value)) >= -1e-10


@pytest.mark.parametrize(
    "neighbors,nodes",
    [
        (_neighbors((("a", "b"), ("b", "c"))), ("a", "b", "c")),
        (_neighbors((("o", "a"), ("o", "b"), ("o", "c"))), ("o", "a", "b", "c")),
        (_neighbors((("a", "b"),), extra=("z",)), ("a", "b", "z")),
        (_neighbors((), extra=("z",)), ("z",)),
    ],
)
def test_reference_graph_kernels_are_psd_on_basic_topologies(neighbors, nodes):
    for constructor, argument in (
        (heat_kernel_reference, {"diffusion_time": 0.7}),
        (resolvent_kernel_reference, {"scale": 0.7}),
    ):
        returned, kernel = constructor(nodes, neighbors, **argument)
        assert returned == nodes
        _assert_correlation_kernel(kernel)


def test_two_node_heat_and_resolvent_match_analytic_spectrum():
    nodes = ("a", "b")
    neighbors = _neighbors((("a", "b"),))
    _, laplacian = symmetric_normalized_laplacian(nodes, neighbors)
    assert np.allclose(laplacian, [[1.0, -1.0], [-1.0, 1.0]])

    time = 0.6
    _, heat = heat_kernel_reference(nodes, neighbors, diffusion_time=time)
    raw_heat = 0.5 * np.array([
        [1.0 + np.exp(-2.0 * time), 1.0 - np.exp(-2.0 * time)],
        [1.0 - np.exp(-2.0 * time), 1.0 + np.exp(-2.0 * time)],
    ])
    assert np.allclose(heat, normalize_psd_kernel(raw_heat))

    scale = 0.4
    _, resolvent = resolvent_kernel_reference(nodes, neighbors, scale=scale)
    raw_resolvent = np.linalg.inv(np.eye(2) + scale * laplacian)
    assert np.allclose(resolvent, normalize_psd_kernel(raw_resolvent))


def test_reference_scales_approach_identity_and_disconnected_blocks_stay_zero():
    nodes = ("a", "b", "x", "y")
    neighbors = _neighbors((("a", "b"), ("x", "y")))
    _, heat = heat_kernel_reference(nodes, neighbors, diffusion_time=1e-8)
    _, resolvent = resolvent_kernel_reference(nodes, neighbors, scale=1e-8)
    assert np.allclose(heat, np.eye(4), atol=3e-8)
    assert np.allclose(resolvent, np.eye(4), atol=3e-8)
    assert np.allclose(heat[:2, 2:], 0.0, atol=1e-12)
    assert np.allclose(resolvent[:2, 2:], 0.0, atol=1e-12)


def test_walk_feature_kernel_is_explicit_psd_and_order_equivariant():
    neighbors = _neighbors((("a", "b"), ("b", "c"), ("c", "d")))
    nodes = ("a", "b", "c", "d")
    features, basis = walk_feature_map(nodes, neighbors, (1.0, 0.5, 0.25))
    returned, kernel, returned_basis = walk_feature_kernel(
        nodes, neighbors, (1.0, 0.5, 0.25)
    )
    assert returned == nodes and returned_basis == basis
    assert np.allclose(kernel, normalize_psd_kernel(features @ features.T))
    _assert_correlation_kernel(kernel)

    reverse = tuple(reversed(nodes))
    _, reversed_kernel, _ = walk_feature_kernel(reverse, neighbors, (1.0, 0.5, 0.25))
    assert np.allclose(reversed_kernel, kernel[::-1, ::-1])


def test_closed_neighborhood_kernel_is_the_binary_feature_gram():
    neighbors = _neighbors((("a", "b"), ("b", "c")))
    nodes, kernel, basis = closed_neighborhood_kernel(("a", "b", "c"), neighbors)
    assert nodes == ("a", "b", "c")
    assert set(basis) == {"a", "b", "c"}
    _assert_correlation_kernel(kernel)
    assert kernel[0, 2] == pytest.approx(0.5)
    assert kernel[0, 1] == pytest.approx(2.0 / np.sqrt(6.0))


def test_cumulative_walk_features_include_cross_hop_adjacent_overlap():
    neighbors = _neighbors((("a", "b"), ("b", "c")))
    nodes = ("a", "b", "c")
    features, basis = cumulative_walk_feature_map(nodes, neighbors, (1.0, 1.0))
    returned, kernel, returned_basis = cumulative_walk_feature_kernel(
        nodes, neighbors, (1.0, 1.0)
    )
    assert returned == nodes and returned_basis == basis
    assert np.allclose(kernel, normalize_psd_kernel((features @ features.T).toarray()))
    _assert_correlation_kernel(kernel)
    assert kernel[0, 1] > 0.0
    assert kernel[0, 1] > kernel[0, 2]


def test_zero_hop_walk_features_are_identity_and_invalid_weights_fail():
    neighbors = _neighbors((("a", "b"),))
    _, kernel, _ = walk_feature_kernel(("a", "b"), neighbors, (1.0,))
    assert np.array_equal(kernel, np.eye(2))
    with pytest.raises(ValueError, match="nonnegative"):
        walk_feature_map(("a",), neighbors, (1.0, -0.1))
    with pytest.raises(ValueError, match="at least one positive"):
        walk_feature_map(("a",), neighbors, (0.0, 0.0))


def test_descendant_gate_lifts_root_kernel_and_preserves_psd():
    root_nodes = ("r1", "r2", "r3")
    neighbors = _neighbors((("r1", "r2"), ("r2", "r3")))
    _, root_kernel, _ = walk_feature_kernel(root_nodes, neighbors, (1.0, 1.0))
    pairs = (("x", "r1"), ("x", "r2"), ("y", "r2"), ("y", "r3"))
    item = descendant_gated_item_kernel(pairs, root_nodes, root_kernel)
    _assert_correlation_kernel(item)
    assert np.all(item[:2, 2:] == 0.0)
    assert item[0, 1] == pytest.approx(root_kernel[0, 1])


def test_role_aware_embedding_rbf_and_bandwidth_are_psd():
    embeddings = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "r1": np.array([0.8, 0.2, 0.0]),
        "r2": np.array([0.7, 0.3, 0.0]),
        "r3": np.array([0.0, 0.2, 0.8]),
    }
    pairs = (("x", "r1"), ("x", "r2"), ("y", "r2"), ("y", "r3"))
    features = role_aware_pair_features(pairs, embeddings)
    bandwidth = median_pairwise_distance(features)
    kernel = embedding_item_kernel(pairs, embeddings, length_scale=bandwidth)
    _assert_correlation_kernel(kernel)
    assert np.all(kernel[:2, 2:] == 0.0)


def test_convex_and_schur_combinations_preserve_psd_and_reject_bad_weights():
    first = np.eye(3)
    second = np.full((3, 3), 0.25)
    np.fill_diagonal(second, 1.0)
    mixed = convex_kernel_mixture((first, second), (0.25, 0.75))
    product = schur_kernel_product((mixed, second))
    _assert_correlation_kernel(mixed)
    _assert_correlation_kernel(product)
    with pytest.raises(ValueError, match="nonnegative"):
        convex_kernel_mixture((first, second), (1.0, -0.1))


def test_indefinite_input_is_rejected_not_repaired():
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="not positive semidefinite"):
        normalize_psd_kernel(indefinite)


def test_kernel_diagnostics_are_stable_and_content_addressed():
    value = np.array([[1.0, 0.2], [0.2, 1.0]])
    first = kernel_diagnostics(value)
    second = kernel_diagnostics(value.copy())
    assert first == second
    assert first.minimum_eigenvalue == pytest.approx(0.8)
    assert first.maximum_eigenvalue == pytest.approx(1.2)
    assert first.rank == 2
    assert first.positive_spectrum_condition_number == pytest.approx(1.5)
    assert len(first.sha256) == 64

    singular = kernel_diagnostics(np.ones((2, 2)))
    assert singular.rank == 1
    assert singular.positive_spectrum_condition_number == pytest.approx(1.0)
