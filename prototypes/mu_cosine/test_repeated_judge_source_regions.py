#!/usr/bin/env python3
"""Tests for deterministic topology-only repeated-judge source regions."""

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from repeated_judge_source_regions import (
    SourceRegionError,
    audit_source_region_partition,
    build_source_region_partition,
)


def graph_from_edges(nodes, edges, *, reverse=False):
    parents = {node: set() for node in nodes}
    children = {node: set() for node in nodes}
    for child, parent in edges:
        parents[child].add(parent)
        children[parent].add(child)
    if reverse:
        parents = dict(reversed([(key, set(reversed(sorted(values)))) for key, values in parents.items()]))
        children = dict(reversed([(key, set(reversed(sorted(values)))) for key, values in children.items()]))
    return parents, children


def line_graph(size, *, reverse=False):
    nodes = [f"n{index:03d}" for index in range(size)]
    return graph_from_edges(
        nodes,
        [(nodes[index + 1], nodes[index]) for index in range(size - 1)],
        reverse=reverse,
    )


def adjacency(parents, children):
    output = {node: set() for node in set(parents) | set(children)}
    for node, values in parents.items():
        for neighbor in values:
            output.setdefault(node, set()).add(neighbor)
            output.setdefault(neighbor, set()).add(node)
    for node, values in children.items():
        for neighbor in values:
            output.setdefault(node, set()).add(neighbor)
            output.setdefault(neighbor, set()).add(node)
    return output


def connected(nodes, graph):
    nodes = set(nodes)
    seen = set()
    queue = deque((next(iter(nodes)),))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(graph[node] & nodes - seen)
    return seen == nodes


def test_hub_graph_meets_exact_region_count_and_connectivity():
    leaves = [f"leaf-{index}" for index in range(8)]
    graph = graph_from_edges(["center", *leaves], [(leaf, "center") for leaf in leaves])
    partition = build_source_region_partition(*graph, 4, halo_hops=0)
    assert len(partition.region_nodes) == 4
    assert set(partition.assignment) == {"center", *leaves}
    undirected = adjacency(*graph)
    assert all(connected(nodes, undirected) for nodes in partition.region_nodes.values())


def test_partition_is_input_order_invariant_and_content_addressed():
    first = build_source_region_partition(*line_graph(32), 8, halo_hops=3)
    second = build_source_region_partition(*line_graph(32, reverse=True), 8, halo_hops=3)
    assert first.assignment == second.assignment
    assert first.weak_component_id == second.weak_component_id
    assert first.assignment_record == second.assignment_record
    assert first.core_assignment_record == second.core_assignment_record


def test_regions_never_cross_true_weak_components():
    nodes = [f"a{index}" for index in range(5)] + [f"b{index}" for index in range(7)]
    edges = [
        (f"a{index + 1}", f"a{index}") for index in range(4)
    ] + [
        (f"b{index + 1}", f"b{index}") for index in range(6)
    ]
    partition = build_source_region_partition(*graph_from_edges(nodes, edges), 5, halo_hops=0)
    assert len(partition.region_nodes) == 5
    for region_nodes in partition.region_nodes.values():
        assert len({partition.weak_component_id[node] for node in region_nodes}) == 1


def test_three_hop_core_is_exact_ball_containment_on_a_line():
    graph = line_graph(24)
    partition = build_source_region_partition(*graph, 2, halo_hops=3)
    undirected = adjacency(*graph)
    boundary = {node for edge in partition.cut_edges for node in edge}
    distances = {node: 0 for node in boundary}
    queue = deque(boundary)
    while queue:
        node = queue.popleft()
        for neighbor in undirected[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    assert partition.halo_nodes == {
        node for node, distance in distances.items() if distance < 3
    }
    assert all(distances[node] >= 3 for nodes in partition.region_core_nodes.values() for node in nodes)
    for region, nodes in partition.region_core_nodes.items():
        for node in nodes:
            local = {node: 0}
            queue = deque((node,))
            while queue:
                current = queue.popleft()
                if local[current] == 3:
                    continue
                for neighbor in undirected[current]:
                    if neighbor not in local:
                        local[neighbor] = local[current] + 1
                        queue.append(neighbor)
            assert all(partition.assignment[other] == region for other in local)


def test_u4_uses_core_nodes_and_can_pass_a_toy_grid():
    audit = audit_source_region_partition(
        *line_graph(40),
        10,
        (10,),
        halo_hops=0,
        min_core_fraction=1.0,
        min_effective_regions=10,
    )
    assert audit["actual_region_count"] == 10
    assert audit["four_endpoint_same_core_capacity_grid"]["10"][
        "optimistic_capacity_upper_bound"
    ] == 10
    assert audit["gates"]["all_four_endpoint_capacity_bounds_pass"] is True
    assert audit["gates"]["cumulative_walk_cross_core_support_certified_disjoint"] is False
    assert audit["gates"]["all_topology_gates_pass"] is False


@pytest.mark.parametrize(
    "target,halo,message",
    [
        (0, 3, "positive"),
        (True, 3, "integer"),
        (2, -1, "nonnegative"),
    ],
)
def test_invalid_partition_parameters_fail_closed(target, halo, message):
    with pytest.raises(SourceRegionError, match=message):
        build_source_region_partition(*line_graph(8), target, halo_hops=halo)


def test_boolean_core_fraction_fails_closed():
    with pytest.raises(SourceRegionError, match="not boolean"):
        audit_source_region_partition(
            *line_graph(8),
            2,
            (2,),
            min_core_fraction=True,
            min_effective_regions=1,
        )


def test_target_below_true_weak_component_count_fails_closed():
    graph = graph_from_edges(["a", "b"], [])
    with pytest.raises(SourceRegionError, match="between component count"):
        build_source_region_partition(*graph, 1, halo_hops=0)


def test_target_above_node_count_fails_closed():
    with pytest.raises(SourceRegionError, match="between component count and node count"):
        build_source_region_partition(*line_graph(4), 5, halo_hops=0)


def test_non_string_node_rejected_by_portable_identity_contract():
    with pytest.raises(SourceRegionError, match="canonical strings"):
        build_source_region_partition({1: set()}, {1: set()}, 1, halo_hops=0)


def test_dense_boundary_can_empty_every_three_hop_core_without_fallback():
    nodes = [f"n{index}" for index in range(8)]
    edges = [
        (nodes[right], nodes[left])
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    ]
    graph = graph_from_edges(nodes, edges)
    partition = build_source_region_partition(*graph, 2, halo_hops=3)
    assert partition.halo_nodes == set(nodes)
    assert all(not core for core in partition.region_core_nodes.values())
    audit = audit_source_region_partition(
        *graph,
        2,
        (2,),
        halo_hops=3,
        min_core_fraction=0.5,
        min_effective_regions=1,
    )
    assert audit["gates"]["all_topology_gates_pass"] is False
    assert audit["core_node_count"] == 0


def test_assignment_records_are_pythonhashseed_invariant():
    script = """
import json
from repeated_judge_source_regions import build_source_region_partition
nodes = {f'n{i:02d}' for i in range(24)}
parents = {node: set() for node in nodes}
children = {node: set() for node in nodes}
ordered = sorted(nodes)
for child, parent in zip(ordered[1:], ordered[:-1]):
    parents[child].add(parent)
    children[parent].add(child)
p = build_source_region_partition(parents, children, 6, halo_hops=3)
print(json.dumps([p.assignment_record, p.core_assignment_record], sort_keys=True))
"""
    environment = dict(os.environ)
    module_dir = str(Path(__file__).resolve().parent)
    environment["PYTHONPATH"] = module_dir + os.pathsep + environment.get("PYTHONPATH", "")
    outputs = []
    for seed in ("1", "8675309"):
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=environment,
                text=True,
            ).strip()
        )
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])) == 2
