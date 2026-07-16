"""Tests for the topology-only repeated-judge source-dependence bridge."""

from dataclasses import replace
import math

import numpy as np
import pytest

from repeated_judge_source_dependence import (
    RegionExposure,
    SourceDependenceError,
    audit_source_dependence,
    build_region_exposure,
    certified_ess_lower_bound,
    component_correlation_matrix,
    compute_per_hop_outside_landing_mass,
    exact_mean_ess,
    exposure_aware_greedy_allocation,
    full_region_capacities,
)
from graph_geometry import cumulative_walk_feature_map
from repeated_judge_source_regions import build_source_region_partition


def graph_from_edges(nodes, edges, *, reverse=False):
    parents = {node: set() for node in nodes}
    children = {node: set() for node in nodes}
    for child, parent in edges:
        parents[child].add(parent)
        children[parent].add(child)
    if reverse:
        parents = dict(reversed(list(parents.items())))
        children = dict(reversed(list(children.items())))
    return parents, children


def line_graph(size, *, reverse=False):
    nodes = [f"n{index:03d}" for index in range(size)]
    return graph_from_edges(
        nodes,
        [(nodes[index + 1], nodes[index]) for index in range(size - 1)],
        reverse=reverse,
    )


def manual_exposure(matrix):
    matrix = np.asarray(matrix, dtype=float)
    region_ids = tuple(f"r{index}" for index in range(len(matrix)))
    return RegionExposure(
        region_ids,
        matrix,
        (1.0,),
        tuple((0.0,) for _ in region_ids),
        (0.0,),
    )


def test_exposure_is_psd_unit_diagonal_and_input_order_invariant():
    first_graph = line_graph(18)
    second_graph = line_graph(18, reverse=True)
    first_partition = build_source_region_partition(*first_graph, 3, halo_hops=0)
    second_partition = build_source_region_partition(*second_graph, 3, halo_hops=0)
    first = build_region_exposure(*first_graph, first_partition)
    second = build_region_exposure(*second_graph, second_partition)
    assert first.region_ids == second.region_ids
    assert np.array_equal(first.matrix, second.matrix)
    assert np.allclose(first.matrix, first.matrix.T, atol=1e-14)
    assert np.allclose(np.diag(first.matrix), 1.0, atol=1e-14)
    assert np.min(first.matrix) >= 0.0
    assert np.linalg.eigvalsh(first.matrix)[0] >= -1e-12
    assert first.matrix.flags.writeable is False


def test_per_hop_outside_landing_handles_boundaries_and_isolated_self_mass():
    edge_graph = graph_from_edges(["a", "b"], [("b", "a")])
    edge_partition = build_source_region_partition(*edge_graph, 2, halo_hops=0)
    edge = compute_per_hop_outside_landing_mass(
        *edge_graph, edge_partition, walk_weights=(1.0, 1.0)
    )
    assert edge["mean_by_hop"] == [0.0, 1.0]

    isolated_graph = graph_from_edges(["a", "b"], [])
    isolated_partition = build_source_region_partition(
        *isolated_graph, 2, halo_hops=0
    )
    isolated = compute_per_hop_outside_landing_mass(
        *isolated_graph, isolated_partition, walk_weights=(1.0, 1.0, 1.0)
    )
    assert isolated["mean_by_hop"] == [0.0, 0.0, 0.0]


def test_region_aggregation_matches_canonical_cumulative_walk_features():
    graph = line_graph(18)
    partition = build_source_region_partition(*graph, 3, halo_hops=0)
    exposure = build_region_exposure(*graph, partition)
    nodes = tuple(sorted(graph[0]))
    adjacency = {
        node: set(graph[0][node]) | set(graph[1][node]) for node in nodes
    }
    node_features, _basis = cumulative_walk_feature_map(
        nodes, adjacency, exposure.walk_weights
    )
    node_index = {node: index for index, node in enumerate(nodes)}
    regional = np.vstack([
        np.asarray(
            node_features[
                [node_index[node] for node in partition.region_nodes[region]]
            ].mean(axis=0)
        ).ravel()
        for region in exposure.region_ids
    ])
    regional /= np.linalg.norm(regional, axis=1, keepdims=True)
    direct = regional @ regional.T
    assert np.allclose(exposure.matrix, direct, atol=2e-15, rtol=0.0)


def test_full_region_caps_and_exposure_aware_greedy_allocation():
    graph = line_graph(80)
    partition = build_source_region_partition(*graph, 10, halo_hops=0)
    capacities = full_region_capacities(partition, 20)
    assert set(capacities.values()) == {2}
    assert sum(capacities.values()) == 20

    exposure = manual_exposure([
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    allocation = exposure_aware_greedy_allocation(
        exposure, {"r0": 2, "r1": 2, "r2": 2}, 3
    )
    assert allocation["assignment_region_ids"] == ["r0", "r2", "r1"]
    assert allocation["counts_by_region"] == {"r0": 1, "r1": 1, "r2": 1}
    assert allocation["quadratic_exposure"] == pytest.approx(4.8)


def test_exact_mean_ess_matches_materialized_component_correlation():
    exposure = manual_exposure(np.eye(3))
    allocation = {"r0": 2, "r1": 1, "r2": 0}
    result = exact_mean_ess(exposure, allocation, 0.2)
    correlation = component_correlation_matrix(exposure, allocation, 0.2)
    direct = float(np.ones(3) @ correlation @ np.ones(3))
    assert result["quadratic_exposure"] == 5.0
    assert result["one_C_one"] == pytest.approx(direct)
    assert result["effective_components"] == pytest.approx(9.0 / direct)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.linalg.eigvalsh(correlation)[0] >= -1e-12


def test_certified_bound_covers_every_small_cap_feasible_allocation():
    exposure = manual_exposure([
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])
    capacities = {"r0": 2, "r1": 2, "r2": 2}
    bound = certified_ess_lower_bound(exposure, capacities, 3, 0.2)
    for left in range(3):
        for middle in range(3):
            right = 3 - left - middle
            if 0 <= right <= 2:
                allocation = {"r0": left, "r1": middle, "r2": right}
                exact = exact_mean_ess(exposure, allocation, 0.2)
                assert exact["quadratic_exposure"] <= (
                    bound["quadratic_exposure_upper_bound"] + 1e-12
                )
                assert exact["effective_components"] >= (
                    bound["effective_components_lower_bound"] - 1e-12
                )


def test_row_sum_bound_covers_fully_coherent_exposure():
    exposure = manual_exposure(np.ones((3, 3)))
    capacities = {"r0": 2, "r1": 2, "r2": 2}
    bound = certified_ess_lower_bound(exposure, capacities, 3, 0.2)
    exact = exact_mean_ess(exposure, {"r0": 1, "r1": 1, "r2": 1}, 0.2)
    assert bound["lambda_max_upper"] >= 3.0
    assert "row sum" in bound["lambda_max_upper_method"]
    assert bound["quadratic_exposure_upper_bound"] >= 9.0
    assert bound["effective_components_lower_bound"] <= exact[
        "effective_components"
    ]


def test_audit_covers_registered_grid_and_keeps_authorization_false():
    audit = audit_source_dependence(
        *line_graph(80),
        20,
        (20,),
        rho_grid=(0.0, 0.2),
    )
    row = audit["registered_size_results"]["20"]
    assert row["gates"]["all_topology_gates_pass"] is True
    assert audit["gates"]["all_registered_sizes_pass"] is True
    assert row["exact_mean_ess_by_rho"]["0.0"]["effective_components"] == 20
    assert row["certified_mean_ess_lower_bound_by_rho"]["0.2"][
        "effective_components_lower_bound"
    ] <= row["exact_mean_ess_by_rho"]["0.2"]["effective_components"]
    assert all(value is False for value in audit["authorization"].values())


def test_malformed_inputs_fail_closed():
    graph = line_graph(12)
    partition = build_source_region_partition(*graph, 3, halo_hops=0)
    with pytest.raises(SourceDependenceError, match="SourceRegionPartition"):
        build_region_exposure(*graph, object())
    with pytest.raises(SourceDependenceError, match="walk weights"):
        build_region_exposure(*graph, partition, walk_weights=(1.0, -1.0))
    broken = replace(partition, assignment={})
    with pytest.raises(SourceDependenceError, match="assignment disagrees"):
        build_region_exposure(*graph, broken)

    exposure = manual_exposure(np.eye(2))
    with pytest.raises(SourceDependenceError, match="every canonical region"):
        exposure_aware_greedy_allocation(exposure, {"r0": 2}, 2)
    with pytest.raises(SourceDependenceError, match="cannot allocate"):
        exposure_aware_greedy_allocation(
            exposure, {"r0": 0, "r1": 1}, 2
        )
    with pytest.raises(SourceDependenceError, match=r"\[0,1\]"):
        exact_mean_ess(exposure, {"r0": 1, "r1": 1}, 1.1)
    with pytest.raises(SourceDependenceError, match="unique positive"):
        audit_source_dependence(*graph, 3, (20, 20))
    with pytest.raises(SourceDependenceError, match="unique values"):
        audit_source_dependence(*graph, 3, (20,), rho_grid=(0.1, 0.1))
