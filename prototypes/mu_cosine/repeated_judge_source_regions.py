"""Deterministic topology-only source regions for repeated-judge sampling.

The regions in this module are engineering units for concentration limits,
fold containment, and dependence sensitivities.  They are never called graph
connected components and never assert statistical independence.

For each true weak component, recursive cuts of a graph-distance-rooted
spanning tree create exactly the requested number of connected regions.  A
node is in its region's radius-``halo_hops`` core exactly when its complete
undirected ball of that radius stays in the region.  Different cores therefore
have disjoint radius-``halo_hops`` graph-feature support by construction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math

from repeated_judge_campaign import FROZEN_WALK_WEIGHTS
from repeated_judge_candidate_capacity import (
    CandidateCapacityError,
    ENDPOINTS_PER_COMPONENT,
    SOURCE_COMPONENT_CAP_FRACTION as _TEN_PERCENT_CAP_FRACTION,
    _canonical_graph,
    optimistic_capacity_bound,
)


def _walk_support_radius(weights):
    if not weights or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in weights
    ):
        raise RuntimeError("frozen walk weights must be finite nonnegative numbers")
    nonzero = [index for index, value in enumerate(weights) if float(value) > 0.0]
    if not nonzero:
        raise RuntimeError("frozen walk weights must contain positive support")
    return max(nonzero)


DEFAULT_REGION_COUNT_GRID = (64, 96, 128)
CUMULATIVE_WALK_SUPPORT_RADIUS = _walk_support_radius(FROZEN_WALK_WEIGHTS)
DEFAULT_HALO_HOPS = CUMULATIVE_WALK_SUPPORT_RADIUS
DEFAULT_MIN_CORE_FRACTION = 0.50
DEFAULT_MIN_EFFECTIVE_REGIONS = 20
SOURCE_REGION_CAP_FRACTION = _TEN_PERCENT_CAP_FRACTION


class SourceRegionError(ValueError):
    """Raised when a source-region input or invariant fails closed."""


def _canonical_source_graph(parents, children):
    try:
        adjacency = _canonical_graph(parents, children)
    except CandidateCapacityError as exc:
        raise SourceRegionError(str(exc)) from exc
    if any(not isinstance(node, str) or not node for node in adjacency):
        raise SourceRegionError(
            "source-region graph nodes must be non-empty canonical strings"
        )
    return adjacency


def _node_key(node):
    return node


def _node_token(node):
    return node


def _content_record(value):
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}


def _validate_partition_parameters(
    target_region_count,
    halo_hops,
    min_core_fraction,
    min_effective_regions,
):
    for name, value in (
        ("target_region_count", target_region_count),
        ("halo_hops", halo_hops),
        ("min_effective_regions", min_effective_regions),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SourceRegionError(f"{name} must be an integer")
    if target_region_count < 1:
        raise SourceRegionError("target_region_count must be positive")
    if halo_hops < 0:
        raise SourceRegionError("halo_hops must be nonnegative")
    if min_effective_regions < 1:
        raise SourceRegionError("min_effective_regions must be positive")
    if isinstance(min_core_fraction, bool):
        raise SourceRegionError("min_core_fraction must be numeric, not boolean")
    try:
        min_core_fraction = float(min_core_fraction)
    except (TypeError, ValueError) as exc:
        raise SourceRegionError("min_core_fraction must be numeric") from exc
    if not math.isfinite(min_core_fraction) or not 0.0 < min_core_fraction <= 1.0:
        raise SourceRegionError("min_core_fraction must lie in (0,1]")
    return min_core_fraction


def _connected_components(adjacency, nodes=None):
    remaining = set(adjacency if nodes is None else nodes)
    components = []
    while remaining:
        start = min(remaining, key=_node_key)
        remaining.remove(start)
        queue = deque((start,))
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            discovered = adjacency[node] & remaining
            remaining.difference_update(discovered)
            queue.extend(sorted(discovered, key=_node_key))
        components.append(tuple(sorted(component, key=_node_key)))
    return tuple(sorted(components, key=lambda values: _node_key(values[0])))


def _allocate_counts(sizes, total):
    """Allocate integer region counts, at least one and at most size."""
    sizes = tuple(sizes)
    if not sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in sizes
    ):
        raise SourceRegionError("sizes must contain positive integers")
    if isinstance(total, bool) or not isinstance(total, int):
        raise SourceRegionError("total region count must be an integer")
    if total < len(sizes) or total > sum(sizes):
        raise SourceRegionError(
            "total region count must lie between component count and node count"
        )
    allocated = [1] * len(sizes)
    while sum(allocated) < total:
        eligible = [index for index in range(len(sizes)) if allocated[index] < sizes[index]]
        if not eligible:
            raise AssertionError("region allocation exhausted before reaching its total")

        def better(left, right):
            # Compare sizes[left] / allocated[left] without floating point.
            lhs = sizes[left] * allocated[right]
            rhs = sizes[right] * allocated[left]
            if lhs != rhs:
                return lhs > rhs
            return left < right

        winner = eligible[0]
        for index in eligible[1:]:
            if better(index, winner):
                winner = index
        allocated[winner] += 1
    return tuple(allocated)


def _bfs_distances(adjacency, allowed, start):
    allowed = set(allowed)
    if start not in allowed:
        raise SourceRegionError("BFS start must belong to its allowed node set")
    distances = {start: 0}
    queue = deque((start,))
    while queue:
        node = queue.popleft()
        distance = distances[node] + 1
        for neighbor in sorted(adjacency[node] & allowed, key=_node_key):
            if neighbor not in distances:
                distances[neighbor] = distance
                queue.append(neighbor)
    return distances


def _farthest(distances):
    if not distances:
        raise SourceRegionError("cannot select a landmark from empty distances")
    greatest = max(distances.values())
    return min(
        (node for node, distance in distances.items() if distance == greatest),
        key=_node_key,
    )


def _bfs_tree(adjacency, allowed, start):
    """Return deterministic BFS distances, parents, and discovery order."""
    allowed = frozenset(allowed)
    distances = {start: 0}
    parents = {start: None}
    order = [start]
    queue = deque((start,))
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node] & allowed, key=_node_key):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            parents[neighbor] = node
            order.append(neighbor)
            queue.append(neighbor)
    if len(distances) != len(allowed):
        raise SourceRegionError("metric-tree input must be connected")
    return distances, parents, tuple(order)


def _metric_tree_cut(adjacency, nodes, parts, stats):
    """Cut one metric-rooted tree edge and allocate final parts to both sides.

    Removing a spanning-tree edge leaves two connected tree subgraphs.  The
    integer part allocation is chosen jointly with the edge to minimize the
    exact proportional discrepancy ``|subtree_size*parts-node_count*q|``.
    This avoids the hub-graph fragmentation caused by thresholding a
    distance/Voronoi ordering.
    """
    nodes = tuple(sorted(nodes, key=_node_key))
    if not 1 < parts <= len(nodes):
        raise SourceRegionError("metric-tree cut requires 2..node_count parts")
    first = nodes[0]
    landmark = _farthest(_bfs_distances(adjacency, nodes, first))
    _distances, parents, order = _bfs_tree(adjacency, nodes, landmark)
    subtree_sizes = {node: 1 for node in nodes}
    for node in reversed(order[1:]):
        subtree_sizes[parents[node]] += subtree_sizes[node]

    node_count = len(nodes)
    best = None
    best_discrepancy = None
    tied = 0
    for child in order[1:]:
        subtree_size = subtree_sizes[child]
        other_size = node_count - subtree_size
        minimum_parts = max(1, parts - other_size)
        maximum_parts = min(parts - 1, subtree_size)
        floor_ideal = (subtree_size * parts) // node_count
        choices = {
            minimum_parts,
            maximum_parts,
            max(minimum_parts, min(maximum_parts, floor_ideal)),
            max(minimum_parts, min(maximum_parts, floor_ideal + 1)),
        }
        for subtree_parts in sorted(choices):
            # The nearest feasible floor/ceiling values minimize this exact,
            # integer-only proportional discrepancy for the current edge.
            discrepancy = abs(subtree_size * parts - node_count * subtree_parts)
            candidate = (discrepancy, _node_key(child), subtree_parts, child)
            if best_discrepancy is None or discrepancy < best_discrepancy:
                best = candidate
                best_discrepancy = discrepancy
                tied = 1
            elif discrepancy == best_discrepancy:
                tied += 1
                if candidate < best:
                    best = candidate
    if best is None:
        raise AssertionError("connected metric tree had no feasible edge allocation")
    stats["metric_tree_cut_count"] += 1
    stats["tie_broken_cut_count"] += int(tied > 1)

    child = best[3]
    subtree_parts = best[2]
    tree_children = {node: [] for node in nodes}
    for node in order[1:]:
        tree_children[parents[node]].append(node)
    subtree = set()
    queue = [child]
    while queue:
        node = queue.pop()
        subtree.add(node)
        queue.extend(tree_children[node])
    other = set(nodes) - subtree
    if not subtree or not other:
        raise AssertionError("metric-tree cut produced an empty side")
    return (
        frozenset(subtree),
        subtree_parts,
        frozenset(other),
        parts - subtree_parts,
    )


def _partition_connected(adjacency, nodes, parts, stats):
    """Partition nodes; emitted pieces are connected and deterministic."""
    nodes = frozenset(nodes)
    if not nodes:
        return ()
    components = _connected_components(adjacency, nodes)
    if len(components) != 1:
        raise SourceRegionError("connected partition input unexpectedly fragmented")
    if parts <= 1:
        return (components[0],)
    if parts > len(nodes):
        raise SourceRegionError("cannot create more connected regions than nodes")
    left, left_parts, right, right_parts = _metric_tree_cut(
        adjacency, nodes, parts, stats
    )
    return (
        *_partition_connected(adjacency, left, left_parts, stats),
        *_partition_connected(adjacency, right, right_parts, stats),
    )


def _stable_group_id(prefix, nodes):
    record = _content_record([_node_token(node) for node in sorted(nodes, key=_node_key)])
    return f"{prefix}-{record['sha256'][:16]}"


def _core_and_halo_nodes(adjacency, assignment, support_radius):
    """Return cross-region edges and the exact complement of safe-radius cores."""
    cut_edges = []
    boundary = set()
    for node in sorted(adjacency, key=_node_key):
        for neighbor in adjacency[node]:
            if _node_key(node) >= _node_key(neighbor):
                continue
            if assignment[node] != assignment[neighbor]:
                cut_edges.append((node, neighbor))
                boundary.update((node, neighbor))
    distances = {node: 0 for node in boundary}
    queue = deque(sorted(boundary, key=_node_key))
    while queue:
        node = queue.popleft()
        # A node whose nearest cut-edge endpoint is at distance d has a node
        # from another region at distance at most d + 1.  Its radius-r ball is
        # therefore safe exactly when d >= r.  Thus the halo cutoff is r - 1.
        if distances[node] >= max(support_radius - 1, 0):
            continue
        for neighbor in sorted(adjacency[node], key=_node_key):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    halo = (
        frozenset()
        if support_radius == 0
        else frozenset(
            node
            for node, distance in distances.items()
            if distance < support_radius
        )
    )
    return tuple(sorted(cut_edges, key=lambda edge: (_node_key(edge[0]), _node_key(edge[1])))), halo


def _region_capacity(core_sizes, components_per_corpus, endpoints_per_component):
    positive = tuple(size for size in core_sizes if size > 0)
    if positive:
        raw = optimistic_capacity_bound(
            positive,
            components_per_corpus,
            cap_fraction=SOURCE_REGION_CAP_FRACTION,
            endpoints_per_component=endpoints_per_component,
        )
        cap = raw["source_component_cap"]
        contributions = [
            min(size // endpoints_per_component, cap)
            for size in sorted(positive, reverse=True)
        ]
        total = sum(contributions)
        squared = sum(value * value for value in contributions)
        return {
            "components_per_corpus": raw["components_per_corpus"],
            "source_region_cap": raw["source_component_cap"],
            "endpoints_per_campaign_component": raw["endpoints_per_component"],
            "optimistic_capacity_upper_bound": raw["optimistic_capacity_upper_bound"],
            "necessary_capacity_gate_passes": raw["necessary_capacity_gate_passes"],
            "nonzero_source_regions": raw["nonzero_source_components"],
            "largest_source_contributions": raw["largest_source_contributions"],
            "capacity_weight_ess": (total * total / squared if squared else 0.0),
        }
    cap = max(1, int(math.floor(components_per_corpus * SOURCE_REGION_CAP_FRACTION + 1e-12)))
    return {
        "components_per_corpus": components_per_corpus,
        "source_region_cap": cap,
        "endpoints_per_campaign_component": endpoints_per_component,
        "optimistic_capacity_upper_bound": 0,
        "necessary_capacity_gate_passes": False,
        "nonzero_source_regions": 0,
        "largest_source_contributions": [],
        "capacity_weight_ess": 0.0,
    }


@dataclass(frozen=True)
class SourceRegionPartition:
    target_region_count: int
    assignment: dict
    weak_component_id: dict
    region_nodes: dict
    region_core_nodes: dict
    halo_nodes: frozenset
    cut_edges: tuple
    assignment_record: dict
    core_assignment_record: dict
    metric_tree_cut_count: int
    tie_broken_cut_count: int


def build_source_region_partition(
    parents,
    children,
    target_region_count,
    *,
    halo_hops=DEFAULT_HALO_HOPS,
):
    """Build one deterministic partition without outcomes or embeddings."""
    min_core_fraction = _validate_partition_parameters(
        target_region_count,
        halo_hops,
        DEFAULT_MIN_CORE_FRACTION,
        DEFAULT_MIN_EFFECTIVE_REGIONS,
    )
    del min_core_fraction  # validation shared with audit; not a construction input
    adjacency = _canonical_source_graph(parents, children)
    weak_components = _connected_components(adjacency)
    allocation = _allocate_counts(tuple(map(len, weak_components)), target_region_count)
    pieces = []
    stats = {"metric_tree_cut_count": 0, "tie_broken_cut_count": 0}
    weak_by_node = {}
    for component, count in zip(weak_components, allocation):
        weak_id = _stable_group_id("weak", component)
        for node in component:
            weak_by_node[node] = weak_id
        pieces.extend(_partition_connected(adjacency, component, count, stats))
    pieces = sorted(pieces, key=lambda values: _node_key(values[0]))
    if len(pieces) != target_region_count:
        raise AssertionError("source-region partition did not meet its exact target count")
    region_nodes = {}
    assignment = {}
    for piece in pieces:
        region_id = _stable_group_id("region", piece)
        if region_id in region_nodes:
            raise AssertionError("source-region content hash collision")
        region_nodes[region_id] = frozenset(piece)
        for node in piece:
            if node in assignment:
                raise AssertionError("node assigned to more than one source region")
            assignment[node] = region_id
    if set(assignment) != set(adjacency):
        raise AssertionError("source regions did not cover every graph node exactly once")
    cut_edges, halo = _core_and_halo_nodes(adjacency, assignment, halo_hops)
    region_core_nodes = {
        region: frozenset(nodes - halo)
        for region, nodes in region_nodes.items()
    }
    assignment_rows = [
        [
            _node_token(node),
            assignment[node],
            weak_by_node[node],
            node in halo,
        ]
        for node in sorted(adjacency, key=_node_key)
    ]
    core_assignment_rows = [row[:3] for row in assignment_rows if not row[3]]
    return SourceRegionPartition(
        target_region_count=int(target_region_count),
        assignment=assignment,
        weak_component_id=weak_by_node,
        region_nodes=region_nodes,
        region_core_nodes=region_core_nodes,
        halo_nodes=halo,
        cut_edges=cut_edges,
        assignment_record=_content_record(assignment_rows),
        core_assignment_record=_content_record(core_assignment_rows),
        metric_tree_cut_count=stats["metric_tree_cut_count"],
        tie_broken_cut_count=stats["tie_broken_cut_count"],
    )


def audit_source_region_partition(
    parents,
    children,
    target_region_count,
    registered_sizes,
    *,
    halo_hops=DEFAULT_HALO_HOPS,
    min_core_fraction=DEFAULT_MIN_CORE_FRACTION,
    min_effective_regions=DEFAULT_MIN_EFFECTIVE_REGIONS,
):
    """Return a path-free structural audit for one requested region count."""
    min_core_fraction = _validate_partition_parameters(
        target_region_count,
        halo_hops,
        min_core_fraction,
        min_effective_regions,
    )
    registered_sizes = tuple(registered_sizes)
    if not registered_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in registered_sizes
    ):
        raise SourceRegionError("registered_sizes must contain positive integers")
    if len(set(registered_sizes)) != len(registered_sizes):
        raise SourceRegionError("registered_sizes must be unique")
    adjacency = _canonical_source_graph(parents, children)
    partition = build_source_region_partition(
        parents,
        children,
        target_region_count,
        halo_hops=halo_hops,
    )
    node_count = len(adjacency)
    edge_count = sum(map(len, adjacency.values())) // 2
    region_sizes = tuple(sorted(map(len, partition.region_nodes.values()), reverse=True))
    core_sizes = tuple(sorted(map(len, partition.region_core_nodes.values()), reverse=True))
    core_count = sum(core_sizes)
    effective_regions = sum(size >= ENDPOINTS_PER_COMPONENT for size in core_sizes)
    u1 = {
        str(size): _region_capacity(core_sizes, size, 1)
        for size in registered_sizes
    }
    u4 = {
        str(size): _region_capacity(core_sizes, size, ENDPOINTS_PER_COMPONENT)
        for size in registered_sizes
    }
    weak_ids = set(partition.weak_component_id.values())
    region_weak_ids = {
        region: {partition.weak_component_id[node] for node in nodes}
        for region, nodes in partition.region_nodes.items()
    }
    if any(len(values) != 1 for values in region_weak_ids.values()):
        raise AssertionError("a source region crossed a true weak component")
    induced_connected = all(
        len(_connected_components(adjacency, nodes)) == 1
        for nodes in partition.region_nodes.values()
    )
    if not induced_connected:
        raise AssertionError("a source region was not induced-connected")
    core_and_halo_cover = (
        set().union(*partition.region_core_nodes.values(), partition.halo_nodes)
        == set(adjacency)
    )
    core_halo_disjoint = all(
        not (nodes & partition.halo_nodes)
        for nodes in partition.region_core_nodes.values()
    )
    if not core_and_halo_cover or not core_halo_disjoint:
        raise AssertionError("source-region core/halo partition was inconsistent")
    capacity_passes = all(
        value["necessary_capacity_gate_passes"] for value in u4.values()
    )
    core_fraction = core_count / node_count
    gates = {
        "all_four_endpoint_capacity_bounds_pass": capacity_passes,
        "core_fraction_passes": core_fraction >= min_core_fraction,
        "effective_region_count_passes": effective_regions >= min_effective_regions,
        "all_regions_within_one_weak_component": True,
        "all_regions_induced_connected": induced_connected,
        "exact_requested_region_count": len(region_sizes) == target_region_count,
        "core_halo_complete_and_disjoint": core_and_halo_cover and core_halo_disjoint,
        "cumulative_walk_cross_core_support_certified_disjoint": (
            halo_hops >= CUMULATIVE_WALK_SUPPORT_RADIUS
        ),
    }
    gates["all_topology_gates_pass"] = all(gates.values())
    return {
        "target_region_count": target_region_count,
        "actual_region_count": len(region_sizes),
        "weak_component_count": len(weak_ids),
        "node_count": node_count,
        "edge_count": edge_count,
        "cut_edge_count": len(partition.cut_edges),
        "cut_edge_fraction": len(partition.cut_edges) / max(edge_count, 1),
        "graph_feature_support_radius_hops": halo_hops,
        "halo_cut_endpoint_distance_rule": "distance < support radius",
        "halo_node_count": len(partition.halo_nodes),
        "core_node_count": core_count,
        "core_node_fraction": core_fraction,
        "core_definition": "B_radius(node) is wholly inside its assigned region",
        "certified_minimum_cross_region_core_distance": 2 * halo_hops + 1,
        "cumulative_walk_support_radius": CUMULATIVE_WALK_SUPPORT_RADIUS,
        "effective_core_region_count_min_four_nodes": effective_regions,
        "region_sizes_descending": list(region_sizes),
        "core_sizes_descending": list(core_sizes),
        "assignment_record": partition.assignment_record,
        "core_assignment_record": partition.core_assignment_record,
        "metric_tree_cut_count": partition.metric_tree_cut_count,
        "tie_broken_cut_count": partition.tie_broken_cut_count,
        "ambiguous_assignment_count": 0,
        "one_endpoint_capacity_grid": u1,
        "four_endpoint_same_core_capacity_grid": u4,
        "gates": gates,
    }
