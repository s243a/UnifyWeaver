"""Outcome-blind structural capacity bounds for the repeated-judge campaign.

This module deliberately stops before candidate enumeration, embedding, or
scoring.  Endpoint-disjoint selection and a per-connected-component source cap
imply the deliberately loose necessary bound

    U_1(G) = sum_s min(|V_s|, max(1, floor(f_source * G))).

It charges only one distinct endpoint to the declared source component.  This
remains valid even if the preregistered ``disconnected`` negative comparator is
interpreted as living outside that component.  If all four distinct endpoints
(``descendant``, ``anchor``, ``adjacent``, and ``distant``) have finite paths in
the same canonical graph, the sharper diagnostic is

    U_4(G) = sum_s min(floor(|V_s| / 4), max(1, floor(f_source * G))).

Every later requirement (hop cells, degree quartiles, graph/Nomic agreement,
historical exclusions, and actual packability) can only reduce capacity.  Thus
``U_1(G) < G`` is a conclusive, outcome-free reason to stop before expensive
candidate or Nomic work.
"""

from __future__ import annotations

from collections import deque
import math


ENDPOINTS_PER_COMPONENT = 4
GUARANTEED_SOURCE_ENDPOINTS_PER_COMPONENT = 1
SOURCE_COMPONENT_CAP_FRACTION = 0.10


class CandidateCapacityError(ValueError):
    """Raised when a structural capacity input violates its contract."""


def _canonical_graph(parents, children):
    """Return a finite undirected adjacency map from directed graph maps."""
    if not isinstance(parents, dict) or not isinstance(children, dict):
        raise CandidateCapacityError("parents and children must be dictionaries")
    adjacency = {}
    canonical = {"parents": {}, "children": {}}

    def add_node(node):
        try:
            hash(node)
        except TypeError as exc:
            raise CandidateCapacityError("graph nodes must be hashable") from exc
        if isinstance(node, str) and not node:
            raise CandidateCapacityError("graph node names must be non-empty")
        adjacency.setdefault(node, set())

    for mapping_name, mapping in (("parents", parents), ("children", children)):
        for node, neighbors in mapping.items():
            add_node(node)
            if isinstance(neighbors, (str, bytes)):
                raise CandidateCapacityError(
                    f"{mapping_name}[{node!r}] must be an iterable of nodes, not text"
                )
            try:
                values = tuple(neighbors)
            except TypeError as exc:
                raise CandidateCapacityError(
                    f"{mapping_name}[{node!r}] must be an iterable of nodes"
                ) from exc
            canonical_values = set()
            for neighbor in values:
                add_node(neighbor)
                if neighbor == node:
                    continue
                canonical_values.add(neighbor)
                adjacency[node].add(neighbor)
                adjacency[neighbor].add(node)
            canonical[mapping_name][node] = canonical_values
    if not adjacency:
        raise CandidateCapacityError("graph must contain at least one node")
    for node in adjacency:
        canonical["parents"].setdefault(node, set())
        canonical["children"].setdefault(node, set())
    for child, values in canonical["parents"].items():
        for parent in values:
            if child not in canonical["children"][parent]:
                raise CandidateCapacityError(
                    f"parents/children maps disagree on edge {child!r}->{parent!r}"
                )
    for parent, values in canonical["children"].items():
        for child in values:
            if parent not in canonical["parents"][child]:
                raise CandidateCapacityError(
                    f"children/parents maps disagree on edge {child!r}->{parent!r}"
                )
    return adjacency


def connected_component_sizes(parents, children):
    """Return descending weak-component sizes, independent of input ordering."""
    adjacency = _canonical_graph(parents, children)
    unseen = set(adjacency)
    sizes = []
    while unseen:
        start = min(unseen, key=lambda value: (type(value).__name__, repr(value)))
        unseen.remove(start)
        queue = deque((start,))
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            discovered = adjacency[node] & unseen
            unseen.difference_update(discovered)
            queue.extend(discovered)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def edge_count(parents, children):
    """Count unique non-self undirected edges in the canonical graph."""
    adjacency = _canonical_graph(parents, children)
    return sum(len(values) for values in adjacency.values()) // 2


def source_component_cap(components_per_corpus, cap_fraction=SOURCE_COMPONENT_CAP_FRACTION):
    if isinstance(components_per_corpus, bool) or not isinstance(components_per_corpus, int):
        raise CandidateCapacityError("components_per_corpus must be an integer")
    if components_per_corpus < 1:
        raise CandidateCapacityError("components_per_corpus must be positive")
    cap_fraction = float(cap_fraction)
    if not math.isfinite(cap_fraction) or not 0.0 < cap_fraction <= 1.0:
        raise CandidateCapacityError("cap_fraction must lie in (0,1]")
    return max(1, int(math.floor(components_per_corpus * cap_fraction + 1e-12)))


def optimistic_capacity_bound(
    component_sizes,
    components_per_corpus,
    *,
    cap_fraction=SOURCE_COMPONENT_CAP_FRACTION,
    endpoints_per_component=GUARANTEED_SOURCE_ENDPOINTS_PER_COMPONENT,
):
    """Necessary pre-history capacity gate under endpoint and source caps."""
    if isinstance(endpoints_per_component, bool) or not isinstance(endpoints_per_component, int):
        raise CandidateCapacityError("endpoints_per_component must be an integer")
    if endpoints_per_component < 1:
        raise CandidateCapacityError("endpoints_per_component must be positive")
    sizes = tuple(component_sizes)
    if not sizes:
        raise CandidateCapacityError("component_sizes must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in sizes):
        raise CandidateCapacityError("component_sizes must contain positive integers")
    cap = source_component_cap(components_per_corpus, cap_fraction)
    contributions = tuple(
        min(size // endpoints_per_component, cap)
        for size in sorted(sizes, reverse=True)
    )
    upper = sum(contributions)
    return {
        "components_per_corpus": components_per_corpus,
        "source_component_cap": cap,
        "endpoints_per_component": endpoints_per_component,
        "optimistic_capacity_upper_bound": upper,
        "necessary_capacity_gate_passes": upper >= components_per_corpus,
        "nonzero_source_components": sum(value > 0 for value in contributions),
        "largest_source_contributions": list(sorted(contributions, reverse=True)[:10]),
    }


def audit_graph_capacity(
    parents,
    children,
    registered_sizes,
    *,
    cap_fraction=SOURCE_COMPONENT_CAP_FRACTION,
):
    """Summarize one frozen graph without using outcomes or embeddings."""
    sizes = connected_component_sizes(parents, children)
    conservative_grid = {
        str(value): optimistic_capacity_bound(
            sizes,
            value,
            cap_fraction=cap_fraction,
            endpoints_per_component=GUARANTEED_SOURCE_ENDPOINTS_PER_COMPONENT,
        )
        for value in registered_sizes
    }
    same_source_four_endpoint_grid = {
        str(value): optimistic_capacity_bound(
            sizes,
            value,
            cap_fraction=cap_fraction,
            endpoints_per_component=ENDPOINTS_PER_COMPONENT,
        )
        for value in registered_sizes
    }
    return {
        "node_count": sum(sizes),
        "edge_count": edge_count(parents, children),
        "connected_component_count": len(sizes),
        "connected_component_sizes_descending": list(sizes),
        "largest_connected_component_fraction": sizes[0] / sum(sizes),
        "registered_grid": conservative_grid,
        "same_source_four_endpoint_sensitivity_grid": same_source_four_endpoint_grid,
        "all_registered_sizes_pass_necessary_gate": all(
            value["necessary_capacity_gate_passes"]
            for value in conservative_grid.values()
        ),
    }
