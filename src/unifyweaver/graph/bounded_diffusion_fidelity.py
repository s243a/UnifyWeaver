"""Outcome-blind bounded-domain selectors and diffusion fidelity diagnostics.

This module compares several deterministic ways to retain a small domain from
one large graph.  It deliberately keeps domain selection separate from the
exact Dirichlet operator in :mod:`unifyweaver.graph.local_diffusion`:

* every selector returns one union-of-anchors :class:`LocalDiffusionDomain`;
* every omitted incident edge remains represented by the builder's full
  Dirichlet cut shunt ``beta``;
* candidate and larger reference models use the same caller-frozen intrinsic
  leakage ``alpha``; and
* fidelity is evaluated only on an explicit common protected node set.

The API contains no labels, model outcomes, winner selection, or entropy
weighting. Experimental closure is opt-in and graph-derived: caller-supplied
Schur/Dirichlet-to-Neumann terms replace an audited share of ``beta`` through
a per-port mass ledger; they are never added on top of full ``beta``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import time
from typing import Mapping

import numpy as np

from .leaky_diffusion import _stable_radial_factor
from .local_diffusion import (
    LocalDiffusionDomain,
    LocalGroundedSemanticDiffusion,
    _assemble_local_components,
    _build_local_from_components,
    _canonical_unique,
    _positive_integer,
    _read_incident_neighbors,
    _stable_key,
    build_local_grounded_semantic_diffusion,
    select_hop_local_domain,
)


__all__ = [
    "BoundedDomainSelection",
    "BoundedFidelityResult",
    "ExperimentalBoundaryClosure",
    "ExperimentalBoundaryClosureConfig",
    "ExteriorComponent",
    "ExteriorComponentDiscovery",
    "ExteriorTraversalLimitError",
    "ProtectedSetCoverageError",
    "build_experimental_boundary_closure",
    "discover_exterior_components",
    "ensure_matched_budget",
    "evaluate_bounded_domain_fidelity",
    "select_hop_budget_domain",
    "select_semantic_resistance_domain",
    "select_topology_skeleton_domain",
    "select_union_hop_reference",
]


_NO_CLOSURE_POLICY = "none_full_dirichlet_beta"
_EXPERIMENTAL_CLOSURE_POLICY = "experimental_graph_derived_schur_closure"


class ProtectedSetCoverageError(ValueError):
    """A frozen protected set is not jointly scoreable by both domains."""

    def __init__(self, missing_candidate, missing_reference):
        self.missing_candidate = tuple(sorted(missing_candidate, key=_stable_key))
        self.missing_reference = tuple(sorted(missing_reference, key=_stable_key))
        super().__init__(
            "protected-set coverage failure: candidate missing "
            f"{self.missing_candidate!r}; reference missing "
            f"{self.missing_reference!r}"
        )


def _finite_nonnegative(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_finite(name, value):
    result = _finite_nonnegative(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _unit_interval(name, value):
    result = _finite_nonnegative(name, value)
    if result >= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return result


def _canonical_parameters(parameters):
    output = []
    for name, value in sorted(dict(parameters).items()):
        if not isinstance(name, str) or not name:
            raise ValueError("selector parameter names must be non-empty strings")
        if isinstance(value, (bool, int, str)) or value is None:
            canonical = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("selector parameters must be finite")
            canonical = value
        else:
            canonical = repr(value)
        output.append((name, canonical))
    return tuple(output)


def _stable_node_token(node):
    module, qualname, representation = _stable_key(node)
    return [module, qualname, representation]



def _canonical_undirected_edge(left, right):
    if left == right:
        raise ValueError("incident adjacency must not contain self loops")
    if _stable_key(right) < _stable_key(left):
        left, right = right, left
    return left, right

def _fingerprint(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _NeighborSnapshot:
    def __init__(self, provider):
        self._provider = provider
        self._cache = {}
        self.calls = 0
        self.maximum_touched_degree = 0

    def get(self, node):
        if node not in self._cache:
            values = _read_incident_neighbors(self._provider, node)
            self._cache[node] = values
            self.calls += 1
            self.maximum_touched_degree = max(
                self.maximum_touched_degree,
                len(values),
            )
        return self._cache[node]

class ExteriorTraversalLimitError(RuntimeError):
    """Exterior traversal hit its frozen cap and therefore failed closed."""

    def __init__(
        self,
        *,
        component_start,
        maximum_component_nodes,
        visited_nodes,
        blocked_neighbor,
    ):
        self.component_start = component_start
        self.maximum_component_nodes = maximum_component_nodes
        self.visited_nodes = tuple(sorted(visited_nodes, key=_stable_key))
        self.blocked_neighbor = blocked_neighbor
        super().__init__(
            "exterior component traversal exceeded maximum_component_nodes; "
            "the unresolved exterior must remain grounded or be retried with "
            "a preregistered larger cap"
        )


@dataclass(frozen=True)
class ExteriorComponent:
    """One complete allowed exterior component and all of its boundary edges."""

    nodes: tuple
    ports: tuple
    cut_edges: tuple
    outside_bath_edges: tuple
    internal_edges: tuple
    internal_edge_count: int
    component_fingerprint: str

    def __post_init__(self):
        nodes = tuple(sorted(self.nodes, key=_stable_key))
        ports = tuple(sorted(self.ports, key=_stable_key))
        edge_key = lambda edge: (_stable_key(edge[0]), _stable_key(edge[1]))
        cut_edges = tuple(sorted(self.cut_edges, key=edge_key))
        bath_edges = tuple(sorted(self.outside_bath_edges, key=edge_key))
        internal_edges = tuple(sorted(self.internal_edges, key=edge_key))
        if not nodes or not ports or not cut_edges:
            raise ValueError("an exterior component needs nodes, ports, and cut edges")
        if len(set(nodes)) != len(nodes) or len(set(ports)) != len(ports):
            raise ValueError("exterior component nodes and ports must be unique")
        for name, edges in (
            ("cut", cut_edges),
            ("bath", bath_edges),
            ("internal", internal_edges),
        ):
            if len(set(edges)) != len(edges):
                raise ValueError(f"exterior component {name} edges must be unique")
        node_set = set(nodes)
        port_set = set(ports)
        for port, omitted in cut_edges:
            if port not in port_set or omitted not in node_set:
                raise ValueError("component cut edge does not align with its ports")
        for interior, outside in bath_edges:
            if interior not in node_set or outside in node_set or outside in port_set:
                raise ValueError("outside bath edge does not leave the component")
        for left, right in internal_edges:
            if left not in node_set or right not in node_set:
                raise ValueError("internal edge endpoint is outside the component")
            if (left, right) != _canonical_undirected_edge(left, right):
                raise ValueError("internal edges must have canonical orientation")
        internal = int(self.internal_edge_count)
        if internal != self.internal_edge_count or internal < 0:
            raise ValueError("internal_edge_count must be a nonnegative integer")
        if internal != len(internal_edges):
            raise ValueError("internal_edge_count must equal stored internal edges")
        payload = {
            "nodes": [_stable_node_token(node) for node in nodes],
            "ports": [_stable_node_token(node) for node in ports],
            "cut_edges": [
                [_stable_node_token(port), _stable_node_token(omitted)]
                for port, omitted in cut_edges
            ],
            "outside_bath_edges": [
                [_stable_node_token(interior), _stable_node_token(outside)]
                for interior, outside in bath_edges
            ],
            "internal_edges": [
                [_stable_node_token(left), _stable_node_token(right)]
                for left, right in internal_edges
            ],
            "internal_edge_count": internal,
        }
        fingerprint = str(self.component_fingerprint)
        if fingerprint != _fingerprint(payload):
            raise ValueError("component_fingerprint does not match component topology")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "cut_edges", cut_edges)
        object.__setattr__(self, "outside_bath_edges", bath_edges)
        object.__setattr__(self, "internal_edges", internal_edges)
        object.__setattr__(self, "internal_edge_count", internal)
        object.__setattr__(self, "component_fingerprint", fingerprint)

    def provenance_dict(self):
        return {
            "nodes": [_stable_node_token(node) for node in self.nodes],
            "ports": [_stable_node_token(node) for node in self.ports],
            "cut_edges": [
                [_stable_node_token(port), _stable_node_token(omitted)]
                for port, omitted in self.cut_edges
            ],
            "outside_bath_edges": [
                [_stable_node_token(interior), _stable_node_token(outside)]
                for interior, outside in self.outside_bath_edges
            ],
            "internal_edges": [
                [_stable_node_token(left), _stable_node_token(right)]
                for left, right in self.internal_edges
            ],
            "internal_edge_count": self.internal_edge_count,
            "component_fingerprint": self.component_fingerprint,
        }


@dataclass(frozen=True)
class ExteriorComponentDiscovery:
    """Auditable partition of the allowed reference exterior touching a domain."""

    retained_nodes: tuple
    allowed_exterior_nodes: tuple | None
    cut_edges: tuple
    outside_allowed_cut_edges: tuple
    components: tuple
    provider_calls: int
    maximum_touched_degree: int
    maximum_component_nodes: int | None
    discovery_fingerprint: str

    def __post_init__(self):
        retained = tuple(self.retained_nodes)
        allowed = self.allowed_exterior_nodes
        if allowed is not None:
            allowed = tuple(sorted(allowed, key=_stable_key))
        components = tuple(self.components)
        if len(set(retained)) != len(retained):
            raise ValueError("retained_nodes must be unique")
        if allowed is not None and len(set(allowed)) != len(allowed):
            raise ValueError("allowed_exterior_nodes must be unique")
        if allowed is not None and set(retained).intersection(allowed):
            raise ValueError("allowed exterior must exclude retained nodes")
        if any(not isinstance(value, ExteriorComponent) for value in components):
            raise TypeError("components must contain ExteriorComponent values")
        edge_key = lambda edge: (_stable_key(edge[0]), _stable_key(edge[1]))
        cut_edges = tuple(sorted(self.cut_edges, key=edge_key))
        outside_cut = tuple(sorted(self.outside_allowed_cut_edges, key=edge_key))
        if len(set(cut_edges)) != len(cut_edges):
            raise ValueError("discovery cut edges must be unique")
        if len(set(outside_cut)) != len(outside_cut):
            raise ValueError("outside-allowed cut edges must be unique")
        component_edges = {
            edge for component in components for edge in component.cut_edges
        }
        if component_edges.union(outside_cut) != set(cut_edges):
            raise ValueError("represented and outside cut edges must partition the cut")
        if component_edges.intersection(outside_cut):
            raise ValueError("a cut edge cannot be represented and outside")
        observed_nodes = set()
        for component in components:
            if observed_nodes.intersection(component.nodes):
                raise ValueError("omitted nodes cannot occur in two components")
            observed_nodes.update(component.nodes)
        if allowed is not None and not observed_nodes.issubset(allowed):
            raise ValueError("component escaped allowed_exterior_nodes")
        calls = int(self.provider_calls)
        degree = int(self.maximum_touched_degree)
        if calls != self.provider_calls or degree != self.maximum_touched_degree:
            raise ValueError("discovery resource counts must be integers")
        if calls < 0 or degree < 0:
            raise ValueError("discovery resource counts must be nonnegative")
        cap = self.maximum_component_nodes
        if cap is not None:
            cap = _positive_integer("maximum_component_nodes", cap)
        fingerprint = str(self.discovery_fingerprint)
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("discovery_fingerprint must be lowercase SHA-256")
        object.__setattr__(self, "allowed_exterior_nodes", allowed)
        object.__setattr__(self, "cut_edges", cut_edges)
        object.__setattr__(self, "outside_allowed_cut_edges", outside_cut)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "provider_calls", calls)
        object.__setattr__(self, "maximum_touched_degree", degree)
        object.__setattr__(self, "maximum_component_nodes", cap)
        object.__setattr__(self, "discovery_fingerprint", fingerprint)

    @property
    def component_count(self):
        return len(self.components)

    @property
    def cut_edge_count(self):
        return len(self.cut_edges)

    @property
    def represented_cut_edge_count(self):
        return self.cut_edge_count - len(self.outside_allowed_cut_edges)

    def provenance_dict(self):
        return {
            "retained_nodes": [
                _stable_node_token(node) for node in self.retained_nodes
            ],
            "allowed_exterior_nodes": (
                None
                if self.allowed_exterior_nodes is None
                else [
                    _stable_node_token(node)
                    for node in self.allowed_exterior_nodes
                ]
            ),
            "cut_edges": [
                [_stable_node_token(port), _stable_node_token(omitted)]
                for port, omitted in self.cut_edges
            ],
            "outside_allowed_cut_edges": [
                [_stable_node_token(port), _stable_node_token(omitted)]
                for port, omitted in self.outside_allowed_cut_edges
            ],
            "components": [
                component.provenance_dict() for component in self.components
            ],
            "provider_calls": self.provider_calls,
            "maximum_touched_degree": self.maximum_touched_degree,
            "maximum_component_nodes": self.maximum_component_nodes,
            "discovery_fingerprint": self.discovery_fingerprint,
        }


def discover_exterior_components(
    domain,
    incident_neighbors,
    *,
    allowed_exterior_nodes=None,
    maximum_component_nodes=None,
):
    """Partition omitted topology inside an explicit bounded reference exterior.

    ``allowed_exterior_nodes`` is normally ``R minus D``. Edges from ``D`` to
    nodes outside that set remain residual candidate grounding. Edges from an
    allowed exterior component to nodes outside ``D union R`` become recorded
    bath shunts in that component's precision. With ``None``, traversal is
    unrestricted and is intended only for exact small/full-graph tests.

    Global memoization terminates and deduplicates traversal. A collision only
    means path convergence in the same component; it never creates a cut. A
    frozen cap hit raises :class:`ExteriorTraversalLimitError` and returns no
    partial physical component.
    """

    if not isinstance(domain, LocalDiffusionDomain):
        raise TypeError("domain must be a LocalDiffusionDomain")
    cap = maximum_component_nodes
    if cap is not None:
        cap = _positive_integer("maximum_component_nodes", cap)
    retained = set(domain.nodes)
    allowed = None
    canonical_allowed = None
    if allowed_exterior_nodes is not None:
        try:
            values = tuple(allowed_exterior_nodes)
        except TypeError as exc:
            raise ValueError("allowed_exterior_nodes must be iterable") from exc
        if len(set(values)) != len(values):
            raise ValueError("allowed_exterior_nodes must not contain duplicates")
        if retained.intersection(values):
            raise ValueError("allowed exterior must exclude retained nodes")
        canonical_allowed = tuple(sorted(values, key=_stable_key))
        allowed = set(canonical_allowed)
    snapshot = _NeighborSnapshot(incident_neighbors)
    domain_neighbors = domain.neighbor_mapping
    cut_edges = set()
    outside_cut_edges = set()
    represented_roots = set()
    for port in domain.nodes:
        incident = snapshot.get(port)
        if incident != domain_neighbors[port]:
            raise ValueError(
                "incident_neighbors disagrees with the frozen local domain"
            )
        for neighbor in incident:
            if neighbor in retained:
                continue
            cut_edges.add((port, neighbor))
            if allowed is not None and neighbor not in allowed:
                outside_cut_edges.add((port, neighbor))
            else:
                represented_roots.add(neighbor)

    roots = tuple(sorted(represented_roots, key=_stable_key))
    assigned = {}
    components = []
    for root in roots:
        if root in assigned:
            continue
        queue = deque((root,))
        component_nodes = {root}
        component_ports = set()
        component_cut_edges = set()
        outside_bath_edges = set()
        internal_edges = set()
        while queue:
            node = queue.popleft()
            for neighbor in snapshot.get(node):
                if neighbor in retained:
                    if node not in snapshot.get(neighbor):
                        raise ValueError("incident adjacency must be reciprocal")
                    component_ports.add(neighbor)
                    component_cut_edges.add((neighbor, node))
                    continue
                if allowed is not None and neighbor not in allowed:
                    if node not in snapshot.get(neighbor):
                        raise ValueError("incident adjacency must be reciprocal")
                    outside_bath_edges.add((node, neighbor))
                    continue
                if node not in snapshot.get(neighbor):
                    raise ValueError("incident adjacency must be reciprocal")
                if neighbor in assigned:
                    raise ValueError(
                        "completed exterior components cannot share an edge"
                    )
                internal_edges.add(
                    _canonical_undirected_edge(node, neighbor)
                )
                if neighbor in component_nodes:
                    continue
                if cap is not None and len(component_nodes) >= cap:
                    raise ExteriorTraversalLimitError(
                        component_start=root,
                        maximum_component_nodes=cap,
                        visited_nodes=component_nodes,
                        blocked_neighbor=neighbor,
                    )
                component_nodes.add(neighbor)
                queue.append(neighbor)
        canonical_nodes = tuple(sorted(component_nodes, key=_stable_key))
        canonical_ports = tuple(sorted(component_ports, key=_stable_key))
        edge_key = lambda edge: (_stable_key(edge[0]), _stable_key(edge[1]))
        canonical_cut_edges = tuple(sorted(component_cut_edges, key=edge_key))
        canonical_bath_edges = tuple(sorted(outside_bath_edges, key=edge_key))
        canonical_internal_edges = tuple(sorted(internal_edges, key=edge_key))
        payload = {
            "nodes": [_stable_node_token(node) for node in canonical_nodes],
            "ports": [_stable_node_token(node) for node in canonical_ports],
            "cut_edges": [
                [_stable_node_token(port), _stable_node_token(omitted)]
                for port, omitted in canonical_cut_edges
            ],
            "outside_bath_edges": [
                [_stable_node_token(interior), _stable_node_token(outside)]
                for interior, outside in canonical_bath_edges
            ],
            "internal_edges": [
                [_stable_node_token(left), _stable_node_token(right)]
                for left, right in canonical_internal_edges
            ],
            "internal_edge_count": len(internal_edges),
        }
        component = ExteriorComponent(
            nodes=canonical_nodes,
            ports=canonical_ports,
            cut_edges=canonical_cut_edges,
            outside_bath_edges=canonical_bath_edges,
            internal_edges=canonical_internal_edges,
            internal_edge_count=len(internal_edges),
            component_fingerprint=_fingerprint(payload),
        )
        component_index = len(components)
        for node in component_nodes:
            assigned[node] = component_index
        components.append(component)

    edge_key = lambda edge: (_stable_key(edge[0]), _stable_key(edge[1]))
    canonical_cut_edges = tuple(sorted(cut_edges, key=edge_key))
    canonical_outside_cut = tuple(sorted(outside_cut_edges, key=edge_key))
    payload = {
        "retained_nodes": [_stable_node_token(node) for node in domain.nodes],
        "allowed_exterior_nodes": (
            None
            if canonical_allowed is None
            else [_stable_node_token(node) for node in canonical_allowed]
        ),
        "cut_edges": [
            [_stable_node_token(port), _stable_node_token(omitted)]
            for port, omitted in canonical_cut_edges
        ],
        "outside_allowed_cut_edges": [
            [_stable_node_token(port), _stable_node_token(omitted)]
            for port, omitted in canonical_outside_cut
        ],
        "components": [
            component.component_fingerprint for component in components
        ],
        "maximum_component_nodes": cap,
    }
    return ExteriorComponentDiscovery(
        retained_nodes=domain.nodes,
        allowed_exterior_nodes=canonical_allowed,
        cut_edges=canonical_cut_edges,
        outside_allowed_cut_edges=canonical_outside_cut,
        components=tuple(components),
        provider_calls=snapshot.calls,
        maximum_touched_degree=snapshot.maximum_touched_degree,
        maximum_component_nodes=cap,
        discovery_fingerprint=_fingerprint(payload),
    )


def _read_provider_values(provider, node, *, kind):
    try:
        if callable(provider):
            values = provider(node)
        elif isinstance(provider, Mapping):
            values = provider[node]
        else:
            raise TypeError(f"{kind} provider must be callable or a mapping")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unable to read {kind} for {node!r}") from exc
    if values is None:
        values = ()
    try:
        return tuple(sorted(set(values), key=_stable_key))
    except TypeError as exc:
        raise ValueError(f"{kind} identifiers must be hashable") from exc


def _induced_hop_distances(anchors, selected, snapshot):
    selected = set(selected)
    distances = {anchor: 0 for anchor in anchors}
    queue = deque(anchors)
    while queue:
        node = queue.popleft()
        for neighbor in snapshot.get(node):
            if neighbor in selected and neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    missing = selected.difference(distances)
    if missing:
        labels = ", ".join(repr(node) for node in sorted(missing, key=_stable_key))
        raise ValueError(
            "selected domain is not connected to the anchor union: " + labels
        )
    return distances


def _domain_from_selected(
    anchors,
    selected,
    snapshot,
    *,
    maximum_nodes,
    selection_metric,
    truncated_tie_count=0,
):
    selected = set(selected)
    if len(selected) > maximum_nodes:
        raise ValueError("selected domain exceeds maximum_nodes")
    for node in selected:
        snapshot.get(node)
    distances = _induced_hop_distances(anchors, selected, snapshot)
    nodes = tuple(
        sorted(
            selected,
            key=lambda node: (distances[node], _stable_key(node)),
        )
    )
    return LocalDiffusionDomain(
        nodes=nodes,
        anchors=anchors,
        hop_distance=np.asarray([distances[node] for node in nodes], dtype=np.int64),
        neighbors=tuple(snapshot.get(node) for node in nodes),
        maximum_nodes=maximum_nodes,
        complete_distance_shell=False,
        truncated_tie_count=truncated_tie_count,
        selection_metric=selection_metric,
    )


@dataclass(frozen=True)
class BoundedDomainSelection:
    """One deterministic selector result with auditable resource provenance."""

    domain: LocalDiffusionDomain
    strategy: str
    requested_nodes: int
    selection_distance: np.ndarray
    selector_parameters: tuple
    provider_calls: int
    maximum_touched_degree: int
    selection_seconds: float
    selection_fingerprint: str
    structural_nodes: int = 0
    shared_parent_nodes: int = 0
    closure_policy: str = _NO_CLOSURE_POLICY

    def __post_init__(self):
        if not isinstance(self.domain, LocalDiffusionDomain):
            raise TypeError("domain must be a LocalDiffusionDomain")
        strategy = str(self.strategy)
        if not strategy:
            raise ValueError("strategy must be non-empty")
        requested = _positive_integer("requested_nodes", self.requested_nodes)
        if requested != self.domain.maximum_nodes:
            raise ValueError("requested_nodes must equal domain.maximum_nodes")
        if len(self.domain.nodes) > requested:
            raise ValueError("realized domain exceeds requested_nodes")
        distance = np.asarray(self.selection_distance, dtype=float)
        if distance.shape != (len(self.domain.nodes),):
            raise ValueError("selection_distance must align with domain.nodes")
        if not np.isfinite(distance).all() or np.any(distance < 0.0):
            raise ValueError("selection_distance must be finite and nonnegative")
        index = {node: row for row, node in enumerate(self.domain.nodes)}
        if any(distance[index[anchor]] != 0.0 for anchor in self.domain.anchors):
            raise ValueError("every anchor must have zero selection distance")
        parameters = _canonical_parameters(self.selector_parameters)
        calls = int(self.provider_calls)
        maximum_degree = int(self.maximum_touched_degree)
        structural = int(self.structural_nodes)
        shared = int(self.shared_parent_nodes)
        if min(calls, maximum_degree, structural, shared) < 0:
            raise ValueError("selection counts must be nonnegative")
        seconds = _finite_nonnegative("selection_seconds", self.selection_seconds)
        fingerprint = str(self.selection_fingerprint)
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("selection_fingerprint must be lowercase SHA-256")
        if self.closure_policy != _NO_CLOSURE_POLICY:
            raise ValueError(
                "bounded fidelity currently permits only full Dirichlet beta "
                "with no boundary closure"
            )
        immutable_distance = np.array(distance, copy=True)
        immutable_distance.setflags(write=False)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "requested_nodes", requested)
        object.__setattr__(self, "selection_distance", immutable_distance)
        object.__setattr__(self, "selector_parameters", parameters)
        object.__setattr__(self, "provider_calls", calls)
        object.__setattr__(self, "maximum_touched_degree", maximum_degree)
        object.__setattr__(self, "selection_seconds", seconds)
        object.__setattr__(self, "structural_nodes", structural)
        object.__setattr__(self, "shared_parent_nodes", shared)

    @property
    def realized_nodes(self):
        return len(self.domain.nodes)

    def provenance_dict(self):
        return {
            "strategy": self.strategy,
            "anchors": list(self.domain.anchors),
            "requested_nodes": self.requested_nodes,
            "realized_nodes": self.realized_nodes,
            "provider_calls": self.provider_calls,
            "maximum_touched_degree": self.maximum_touched_degree,
            "selection_seconds": self.selection_seconds,
            "selector_parameters": dict(self.selector_parameters),
            "selection_fingerprint": self.selection_fingerprint,
            "structural_nodes": self.structural_nodes,
            "shared_parent_nodes": self.shared_parent_nodes,
            "closure_policy": self.closure_policy,
        }


def _make_selection(
    domain,
    *,
    strategy,
    requested_nodes,
    distances,
    parameters,
    snapshot,
    elapsed,
    structural_nodes=0,
    shared_parent_nodes=0,
):
    parameters = _canonical_parameters(parameters)
    aligned = np.asarray([distances[node] for node in domain.nodes], dtype=float)
    payload = {
        "strategy": strategy,
        "anchors": [_stable_node_token(node) for node in domain.anchors],
        "nodes": [_stable_node_token(node) for node in domain.nodes],
        "requested_nodes": requested_nodes,
        "selection_distance_hex": [float(value).hex() for value in aligned],
        "parameters": parameters,
        "closure_policy": _NO_CLOSURE_POLICY,
    }
    return BoundedDomainSelection(
        domain=domain,
        strategy=strategy,
        requested_nodes=requested_nodes,
        selection_distance=aligned,
        selector_parameters=parameters,
        provider_calls=snapshot.calls,
        maximum_touched_degree=snapshot.maximum_touched_degree,
        selection_seconds=elapsed,
        selection_fingerprint=_fingerprint(payload),
        structural_nodes=structural_nodes,
        shared_parent_nodes=shared_parent_nodes,
    )


def select_hop_budget_domain(anchors, incident_neighbors, *, maximum_nodes):
    """Select the deterministic hard-K multi-source hop-ball baseline."""

    anchors = _canonical_unique(anchors, name="anchors")
    maximum_nodes = _positive_integer("maximum_nodes", maximum_nodes)
    snapshot = _NeighborSnapshot(incident_neighbors)
    started = time.perf_counter()
    domain = select_hop_local_domain(
        anchors,
        snapshot.get,
        maximum_nodes=maximum_nodes,
        complete_distance_shell=False,
    )
    elapsed = time.perf_counter() - started
    distances = domain.distance_by_node
    return _make_selection(
        domain,
        strategy="hop_ball",
        requested_nodes=maximum_nodes,
        distances=distances,
        parameters={"complete_distance_shell": False},
        snapshot=snapshot,
        elapsed=elapsed,
    )


def select_topology_skeleton_domain(
    anchors,
    incident_neighbors,
    parents,
    *,
    maximum_nodes,
    ancestor_depth=2,
):
    """Retain anchor ancestry/common parents, then deterministically fill to K.

    ``parents`` is an outcome-blind directed-topology provider.  Every declared
    parent edge must also exist in the complete undirected incident adjacency.
    Required ancestry is never silently truncated: if it exceeds ``K`` this
    selector fails closed.
    """

    anchors = _canonical_unique(anchors, name="anchors")
    maximum_nodes = _positive_integer("maximum_nodes", maximum_nodes)
    if isinstance(ancestor_depth, bool):
        raise ValueError("ancestor_depth must be a nonnegative integer")
    try:
        ancestor_depth = int(ancestor_depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("ancestor_depth must be a nonnegative integer") from exc
    if ancestor_depth < 0:
        raise ValueError("ancestor_depth must be a nonnegative integer")

    snapshot = _NeighborSnapshot(incident_neighbors)
    started = time.perf_counter()
    selected = set(anchors)
    layer = tuple(anchors)
    parent_children = {}
    parent_relation = {}

    def reaches_parent(start, target):
        pending = [start]
        visited = set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            pending.extend(parent_relation.get(node, ()))
        return False

    for _ in range(ancestor_depth):
        next_layer = set()
        for child in layer:
            child_parents = _read_provider_values(parents, child, kind="parents")
            incident = set(snapshot.get(child))
            for parent in child_parents:
                if parent == child:
                    raise ValueError("parent providers must not contain self loops")
                if parent not in incident:
                    raise ValueError(
                        f"declared parent edge {child!r}->{parent!r} is absent "
                        "from incident adjacency"
                    )
                if reaches_parent(parent, child):
                    raise ValueError("cyclic parent data are not permitted")
                parent_relation.setdefault(child, set()).add(parent)
                parent_children.setdefault(parent, set()).add(child)
                if parent not in selected:
                    next_layer.add(parent)
                selected.add(parent)
        layer = tuple(sorted(next_layer, key=_stable_key))
        if not layer:
            break

    structural_count = len(selected)
    if structural_count > maximum_nodes:
        raise ValueError(
            "required ancestor/common-parent skeleton exceeds maximum_nodes"
        )
    distances = _induced_hop_distances(anchors, selected, snapshot)
    heap = []
    best = {}

    def offer_from(node):
        candidate_distance = distances[node] + 1
        for neighbor in snapshot.get(node):
            if neighbor in selected:
                continue
            old = best.get(neighbor)
            if old is None or candidate_distance < old:
                best[neighbor] = candidate_distance
                heapq.heappush(
                    heap,
                    (candidate_distance, _stable_key(neighbor), neighbor),
                )

    for node in sorted(selected, key=lambda item: (distances[item], _stable_key(item))):
        offer_from(node)
    while heap and len(selected) < maximum_nodes:
        distance, _, node = heapq.heappop(heap)
        if node in selected or best.get(node) != distance:
            continue
        selected.add(node)
        distances[node] = distance
        offer_from(node)

    domain = _domain_from_selected(
        anchors,
        selected,
        snapshot,
        maximum_nodes=maximum_nodes,
        selection_metric="topology_skeleton_then_hop_fill",
    )
    # Report true induced hop distances after the structural seed and fill.
    distances = domain.distance_by_node
    shared_count = sum(
        len(children) >= 2 for children in parent_children.values()
    )
    elapsed = time.perf_counter() - started
    return _make_selection(
        domain,
        strategy="topology_skeleton",
        requested_nodes=maximum_nodes,
        distances=distances,
        parameters={"ancestor_depth": ancestor_depth},
        snapshot=snapshot,
        elapsed=elapsed,
        structural_nodes=structural_count,
        shared_parent_nodes=shared_count,
    )


class _EmbeddingSnapshot:
    def __init__(self, embeddings):
        if not isinstance(embeddings, Mapping):
            raise TypeError("node_embeddings must be a mapping")
        self._embeddings = embeddings
        self._cache = {}
        self._width = None

    def get(self, node):
        if node not in self._cache:
            try:
                row = np.asarray(self._embeddings[node], dtype=float)
            except KeyError as exc:
                raise ValueError(f"node absent from embeddings: {node!r}") from exc
            if row.ndim != 1 or not len(row) or not np.isfinite(row).all():
                raise ValueError("every touched embedding must be one finite vector")
            if self._width is None:
                self._width = len(row)
            if len(row) != self._width:
                raise ValueError("every touched embedding must have the same width")
            self._cache[node] = np.array(row, copy=True)
        return self._cache[node]


def select_semantic_resistance_domain(
    anchors,
    incident_neighbors,
    node_embeddings,
    *,
    maximum_nodes,
    length_scale,
    conductance_floor=0.0,
):
    """Select a truncated Dijkstra prefix using resistance on existing edges.

    Semantics changes edge cost but never creates an edge.  Zero-conductance
    edges (possible only with a zero floor) are not traversable in the weighted
    selector.  The returned physical model still uses the exact same semantic
    conductance formula and full Dirichlet cut aggregation.
    """

    anchors = _canonical_unique(anchors, name="anchors")
    maximum_nodes = _positive_integer("maximum_nodes", maximum_nodes)
    if len(anchors) > maximum_nodes:
        raise ValueError("maximum_nodes must be at least the number of anchors")
    length_scale = _positive_finite("length_scale", length_scale)
    floor = _unit_interval("conductance_floor", conductance_floor)
    snapshot = _NeighborSnapshot(incident_neighbors)
    embeddings = _EmbeddingSnapshot(node_embeddings)
    started = time.perf_counter()
    distances = {anchor: 0.0 for anchor in anchors}
    heap = [(0.0, _stable_key(anchor), anchor) for anchor in anchors]
    heapq.heapify(heap)
    selected = set()

    while heap and len(selected) < maximum_nodes:
        distance, _, node = heapq.heappop(heap)
        if node in selected or distance != distances.get(node):
            continue
        selected.add(node)
        left = embeddings.get(node)
        for neighbor in snapshot.get(node):
            right = embeddings.get(neighbor)
            radial = _stable_radial_factor(left, right, length_scale)
            conductance = floor + (1.0 - floor) * radial
            if conductance <= 0.0:
                continue
            candidate = distance + 1.0 / conductance
            if not math.isfinite(candidate):
                raise ValueError("semantic resistance path length is not finite")
            old = distances.get(neighbor)
            if old is None or candidate < old:
                distances[neighbor] = candidate
                heapq.heappush(
                    heap,
                    (candidate, _stable_key(neighbor), neighbor),
                )

    domain = _domain_from_selected(
        anchors,
        selected,
        snapshot,
        maximum_nodes=maximum_nodes,
        selection_metric="semantic_resistance",
    )
    elapsed = time.perf_counter() - started
    return _make_selection(
        domain,
        strategy="semantic_resistance",
        requested_nodes=maximum_nodes,
        distances=distances,
        parameters={
            "length_scale": length_scale,
            "conductance_floor": floor,
            "edge_support": "topology_only",
        },
        snapshot=snapshot,
        elapsed=elapsed,
    )


def ensure_matched_budget(selections):
    """Validate that a family comparison uses the same anchors and requested K."""

    selections = tuple(selections)
    if len(selections) < 2:
        raise ValueError("matched-budget validation requires at least two selections")
    if any(not isinstance(value, BoundedDomainSelection) for value in selections):
        raise TypeError("every selection must be a BoundedDomainSelection")
    anchors = selections[0].domain.anchors
    budget = selections[0].requested_nodes
    for selection in selections[1:]:
        if selection.domain.anchors != anchors:
            raise ValueError("matched-budget selections must use the same anchors")
        if selection.requested_nodes != budget:
            raise ValueError("matched-budget selections must request the same K")
    return budget


def select_union_hop_reference(
    selections,
    incident_neighbors,
    *,
    maximum_nodes,
):
    """Preserve every candidate node, then hop-expand their common reference.

    The expansion is multi-source from the frozen union of all candidate
    domains. This makes a cross-family reference explicit: no candidate is
    compared against a reference that omits nodes selected by another family.
    If the union exceeds ``maximum_nodes``, fail rather than truncate it.
    """

    selections = tuple(selections)
    if not selections:
        raise ValueError("reference construction requires at least one selection")
    if any(not isinstance(value, BoundedDomainSelection) for value in selections):
        raise TypeError("every selection must be a BoundedDomainSelection")
    maximum_nodes = _positive_integer("maximum_nodes", maximum_nodes)
    anchors = selections[0].domain.anchors
    for selection in selections[1:]:
        if selection.domain.anchors != anchors:
            raise ValueError("reference selections must use the same anchors")

    candidate_union = {
        node for selection in selections for node in selection.domain.nodes
    }
    if len(candidate_union) > maximum_nodes:
        raise ValueError("candidate-domain union exceeds reference maximum_nodes")

    snapshot = _NeighborSnapshot(incident_neighbors)
    started = time.perf_counter()
    selected = set(candidate_union)
    for node in sorted(selected, key=_stable_key):
        snapshot.get(node)
    frontier = deque(sorted(selected, key=_stable_key))
    while frontier and len(selected) < maximum_nodes:
        shell_size = len(frontier)
        shell_candidates = set()
        for _ in range(shell_size):
            node = frontier.popleft()
            for neighbor in snapshot.get(node):
                if neighbor not in selected:
                    shell_candidates.add(neighbor)
        if not shell_candidates:
            break
        for node in sorted(shell_candidates, key=_stable_key):
            if len(selected) >= maximum_nodes:
                break
            selected.add(node)
            frontier.append(node)

    domain = _domain_from_selected(
        anchors,
        selected,
        snapshot,
        maximum_nodes=maximum_nodes,
        selection_metric="candidate_union_then_hop_fill",
    )
    elapsed = time.perf_counter() - started
    union_payload = {
        "candidate_selection_fingerprints": sorted(
            selection.selection_fingerprint for selection in selections
        ),
        "candidate_union_nodes": [
            _stable_node_token(node)
            for node in sorted(candidate_union, key=_stable_key)
        ],
    }
    return _make_selection(
        domain,
        strategy="candidate_union_hop_reference",
        requested_nodes=maximum_nodes,
        distances=domain.distance_by_node,
        parameters={
            "candidate_count": len(selections),
            "candidate_union_nodes": len(candidate_union),
            "candidate_union_fingerprint": _fingerprint(union_payload),
        },
        snapshot=snapshot,
        elapsed=elapsed,
    )


@dataclass(frozen=True)
class ExperimentalBoundaryClosureConfig:
    """Frozen controls for graph-derived sparse boundary closure."""

    maximum_edges: int
    closure_mass_fraction: float
    ordinary_branch_conductance: float
    bridge_conductance_cap: float
    pair_conductance_source: str
    ledger_mode: str
    semantic_similarity_minimum: float | None = None

    def __post_init__(self):
        maximum_edges = _positive_integer("maximum_edges", self.maximum_edges)
        fraction = _finite_nonnegative(
            "closure_mass_fraction", self.closure_mass_fraction
        )
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("closure_mass_fraction must be in (0, 1]")
        ordinary = _positive_finite(
            "ordinary_branch_conductance",
            self.ordinary_branch_conductance,
        )
        cap = _positive_finite(
            "bridge_conductance_cap", self.bridge_conductance_cap
        )
        if cap >= ordinary:
            raise ValueError(
                "bridge_conductance_cap must be strictly below the recorded "
                "ordinary branch conductance"
            )
        source = str(self.pair_conductance_source)
        if source not in (
            "two_terminal_series",
            "sparse_dtn",
            "exact_component_schur",
        ):
            raise ValueError(
                "pair_conductance_source must be two_terminal_series, "
                "sparse_dtn, or exact_component_schur"
            )
        if source == "exact_component_schur" and fraction != 1.0:
            raise ValueError(
                "exact_component_schur requires closure_mass_fraction == 1.0; "
                "approximation rho does not apply to an exact reduction"
            )
        ledger_mode = str(self.ledger_mode)
        if ledger_mode not in ("equal_return_transfer", "explicit_self_return"):
            raise ValueError(
                "ledger_mode must be equal_return_transfer or explicit_self_return"
            )
        if (
            source == "exact_component_schur"
            and ledger_mode != "explicit_self_return"
        ):
            raise ValueError(
                "exact_component_schur requires explicit_self_return ledger mode"
            )
        threshold = self.semantic_similarity_minimum
        if threshold is not None:
            threshold = _finite_nonnegative(
                "semantic_similarity_minimum", threshold
            )
            if threshold > 1.0:
                raise ValueError("semantic_similarity_minimum must be in [0, 1]")
        object.__setattr__(self, "maximum_edges", maximum_edges)
        object.__setattr__(self, "closure_mass_fraction", fraction)
        object.__setattr__(self, "ordinary_branch_conductance", ordinary)
        object.__setattr__(self, "bridge_conductance_cap", cap)
        object.__setattr__(self, "pair_conductance_source", source)
        object.__setattr__(self, "ledger_mode", ledger_mode)
        object.__setattr__(self, "semantic_similarity_minimum", threshold)




@dataclass(frozen=True)
class ExperimentalBoundaryClosure:
    """A local model plus its exact Schur-shaped per-port mass ledger."""

    model: LocalGroundedSemanticDiffusion
    edges: tuple
    original_cut_conductance: np.ndarray
    transfer_degree: np.ndarray
    self_return_conductance: np.ndarray
    residual_ground_conductance: np.ndarray
    supplied_pair_count: int
    filtered_pair_count: int
    config: ExperimentalBoundaryClosureConfig
    ledger_fingerprint: str
    closure_policy: str = _EXPERIMENTAL_CLOSURE_POLICY

    def __post_init__(self):
        if not isinstance(self.model, LocalGroundedSemanticDiffusion):
            raise TypeError("model must be a LocalGroundedSemanticDiffusion")
        if not isinstance(self.config, ExperimentalBoundaryClosureConfig):
            raise TypeError("config must be ExperimentalBoundaryClosureConfig")
        size = len(self.model.nodes)
        original = np.asarray(self.original_cut_conductance, dtype=float)
        transfer = np.asarray(self.transfer_degree, dtype=float)
        self_return = np.asarray(self.self_return_conductance, dtype=float)
        residual = np.asarray(self.residual_ground_conductance, dtype=float)
        vectors = (original, transfer, self_return, residual)
        if any(value.shape != (size,) for value in vectors):
            raise ValueError("closure ledger vectors must align with model nodes")
        if any(
            not np.isfinite(value).all() or np.any(value < 0.0)
            for value in vectors
        ):
            raise ValueError("closure ledger vectors must be finite and nonnegative")
        scale = max(float(np.max(original, initial=0.0)), 1.0)
        tolerance = 64.0 * np.finfo(float).eps * scale
        if not np.allclose(
            residual + self_return + transfer,
            original,
            rtol=1e-12,
            atol=tolerance,
        ):
            raise ValueError(
                "closure ledger must split beta into residual, self-return, "
                "and transfer mass"
            )
        if (
            self.config.ledger_mode == "equal_return_transfer"
            and not np.allclose(
                self_return, transfer, rtol=1e-12, atol=tolerance
            )
        ):
            raise ValueError(
                "equal_return_transfer requires self-return equal to transfer degree"
            )
        if not np.allclose(
            self.model.cut_conductance,
            residual,
            rtol=1e-12,
            atol=tolerance,
        ):
            raise ValueError("model cut conductance must equal residual ground beta")
        index = {node: row for row, node in enumerate(self.model.nodes)}
        observed_transfer = np.zeros(size)
        observed_pairs = set()
        for edge in self.edges:
            if len(edge) != 3:
                raise ValueError("closure edges must be (left, right, conductance)")
            left, right, conductance = edge
            if left == right:
                raise ValueError("closure edges must not be self edges")
            if left not in index or right not in index:
                raise ValueError("closure edge endpoints must be retained")
            pair = frozenset((left, right))
            if pair in observed_pairs:
                raise ValueError("closure edges must not repeat an undirected pair")
            observed_pairs.add(pair)
            value = _positive_finite("closure edge conductance", conductance)
            if (
                self.config.pair_conductance_source
                != "exact_component_schur"
                and value > self.config.bridge_conductance_cap
            ):
                raise ValueError("closure edge exceeds bridge_conductance_cap")
            left_row = index[left]
            right_row = index[right]
            if original[left_row] <= 0.0 or original[right_row] <= 0.0:
                raise ValueError("closure endpoints must both be boundary nodes")
            if (
                right in self.model.domain.neighbor_mapping[left]
                and self.config.pair_conductance_source
                != "exact_component_schur"
            ):
                raise ValueError("approximate closure must join graph non-neighbours")
            observed_transfer[left_row] += value
            observed_transfer[right_row] += value
        if not np.allclose(
            observed_transfer,
            transfer,
            rtol=1e-12,
            atol=tolerance,
        ):
            raise ValueError("transfer_degree does not equal closure edge degree")
        if (
            self.config.ledger_mode == "equal_return_transfer"
            and np.any(2.0 * transfer > original + tolerance)
        ):
            raise ValueError(
                "closure transfer oversubscribes factor-two beta capacity"
            )
        supplied_count = int(self.supplied_pair_count)
        filtered_count = int(self.filtered_pair_count)
        if (
            supplied_count != self.supplied_pair_count
            or filtered_count != self.filtered_pair_count
            or supplied_count < 0
            or filtered_count < 0
            or filtered_count > supplied_count
        ):
            raise ValueError("closure pair provenance counts are invalid")
        if len(self.edges) > supplied_count - filtered_count:
            raise ValueError("realized closure edges exceed unfiltered supplied pairs")
        if len(self.edges) > self.config.maximum_edges:
            raise ValueError("closure edge count exceeds maximum_edges")
        if self.closure_policy != _EXPERIMENTAL_CLOSURE_POLICY:
            raise ValueError("closure_policy must remain explicitly experimental")
        fingerprint = str(self.ledger_fingerprint)
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("ledger_fingerprint must be lowercase SHA-256")
        for name, value in (
            ("original_cut_conductance", original),
            ("transfer_degree", transfer),
            ("self_return_conductance", self_return),
            ("residual_ground_conductance", residual),
        ):
            immutable = np.array(value, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        object.__setattr__(self, "edges", tuple(self.edges))

    @property
    def pair_source(self):
        return self.config.pair_conductance_source

    @property
    def realized_pair_count(self):
        return len(self.edges)

    @property
    def total_transfer_mass(self):
        return float(np.sum(self.transfer_degree))

    @property
    def total_self_return_mass(self):
        return float(np.sum(self.self_return_conductance))

def build_experimental_boundary_closure(
    domain,
    *,
    intrinsic_leakage_conductance,
    pair_conductances,
    config,
    self_return_conductance=None,
    node_embeddings=None,
    semantic_filter_embeddings=None,
    semantic_filter_length_scale=None,
    semantic_length_scale=None,
    conductance_floor=0.0,
    minimum_reciprocal_condition=None,
):
    """Build opt-in graph-derived closure with a Schur-shaped mass ledger.

    Pair strengths must come from graph physics: either one true two-terminal
    series/effective-resistance reduction or an audited sparse approximation to
    a multi-terminal exterior Dirichlet-to-Neumann map. Semantic similarity can
    filter supplied pairs, but never determines kappa.  The explicit ledger
    accepts the graph-derived diagonal self-return separately from the
    off-diagonal transfer conductance.  ``equal_return_transfer`` is only the
    factor-two convenience form for the symmetric zero-leakage series case.
    """

    if not isinstance(config, ExperimentalBoundaryClosureConfig):
        raise TypeError("config must be ExperimentalBoundaryClosureConfig")
    try:
        supplied_edges = tuple(pair_conductances)
    except TypeError as exc:
        raise ValueError(
            "pair_conductances must contain (left, right, kappa) triples"
        ) from exc
    if (
        config.pair_conductance_source == "two_terminal_series"
        and len(supplied_edges) != 1
    ):
        raise ValueError(
            "two_terminal_series permits exactly one pair; multi-terminal "
            "pairwise 1/R composition requires an audited sparse_dtn source"
        )
    conductance, base_laplacian, intrinsic, beta = _assemble_local_components(
        domain,
        intrinsic_leakage_conductance=intrinsic_leakage_conductance,
        node_embeddings=node_embeddings,
        length_scale=semantic_length_scale,
        conductance_floor=conductance_floor,
    )
    if (
        config.semantic_similarity_minimum is not None
        and (
            semantic_filter_embeddings is None
            or semantic_filter_length_scale is None
        )
    ):
        raise ValueError(
            "semantic closure filtering requires separate filter embeddings "
            "and filter length_scale"
        )
    filter_embeddings = (
        None
        if config.semantic_similarity_minimum is None
        else _EmbeddingSnapshot(semantic_filter_embeddings)
    )
    index = {node: row for row, node in enumerate(domain.nodes)}
    neighbor_mapping = domain.neighbor_mapping
    candidates = []
    observed_pairs = set()
    for edge in supplied_edges:
        if len(edge) != 3:
            raise ValueError(
                "pair_conductances must contain (left, right, kappa) triples"
            )
        left, right, supplied = edge
        if left == right:
            raise ValueError("closure pair conductances must not contain self edges")
        if left not in index or right not in index:
            raise ValueError("closure pair endpoints must be retained")
        pair = frozenset((left, right))
        if pair in observed_pairs:
            raise ValueError("closure pair conductances must not repeat a pair")
        observed_pairs.add(pair)
        left_row = index[left]
        right_row = index[right]
        if beta[left_row] <= 0.0 or beta[right_row] <= 0.0:
            raise ValueError("closure pair endpoints must both be boundary nodes")
        if (
            right in neighbor_mapping[left]
            and config.pair_conductance_source != "exact_component_schur"
        ):
            raise ValueError(
                "approximate closure pairs must join graph non-neighbours"
            )
        supplied = _positive_finite("graph-derived pair conductance", supplied)
        if filter_embeddings is not None:
            similarity = _stable_radial_factor(
                filter_embeddings.get(left),
                filter_embeddings.get(right),
                semantic_filter_length_scale,
            )
            if similarity < config.semantic_similarity_minimum:
                continue
        candidates.append(
            (
                -supplied,
                _stable_key(left),
                _stable_key(right),
                left,
                right,
                left_row,
                right_row,
                supplied,
            )
        )
    candidates.sort()
    explicit_ledger = config.ledger_mode == "explicit_self_return"
    if explicit_ledger and config.semantic_similarity_minimum is not None:
        raise ValueError(
            "explicit sparse-DtN ledgers must be filtered before construction "
            "and recomputed as one coherent approximation"
        )
    if explicit_ledger and len(candidates) > config.maximum_edges:
        raise ValueError(
            "explicit sparse-DtN pair set exceeds maximum_edges; truncate and "
            "recompute its self-return ledger upstream"
        )
    if not explicit_ledger and self_return_conductance is not None:
        raise ValueError(
            "self_return_conductance requires explicit_self_return ledger mode"
        )
    remaining_transfer = 0.5 * config.closure_mass_fraction * beta
    transfer = np.zeros(len(domain.nodes), dtype=float)
    edges = []
    closure_conductance = np.array(conductance, copy=True)
    for (
        _,
        __,
        ___,
        left,
        right,
        left_row,
        right_row,
        supplied,
    ) in candidates:
        if len(edges) >= config.maximum_edges:
            break
        if explicit_ledger:
            if (
                config.pair_conductance_source != "exact_component_schur"
                and supplied > config.bridge_conductance_cap
            ):
                raise ValueError(
                    "explicit graph-derived pair conductance exceeds "
                    "bridge_conductance_cap"
                )
            value = supplied
        else:
            proposed = min(supplied, config.bridge_conductance_cap)
            value = min(
                proposed,
                remaining_transfer[left_row],
                remaining_transfer[right_row],
            )
        if value <= 0.0:
            continue
        if config.pair_conductance_source == "exact_component_schur":
            closure_conductance[left_row, right_row] += value
            closure_conductance[right_row, left_row] += value
        else:
            closure_conductance[left_row, right_row] = value
            closure_conductance[right_row, left_row] = value
        if not explicit_ledger:
            remaining_transfer[left_row] -= value
            remaining_transfer[right_row] -= value
        transfer[left_row] += value
        transfer[right_row] += value
        edges.append((left, right, float(value)))
    if explicit_ledger:
        if not isinstance(self_return_conductance, Mapping):
            raise ValueError(
                "explicit_self_return requires a retained-node mapping"
            )
        unknown = set(self_return_conductance).difference(domain.nodes)
        if unknown:
            raise ValueError("self-return mapping contains unretained nodes")
        self_return = np.asarray(
            [
                _finite_nonnegative(
                    "self-return conductance",
                    self_return_conductance.get(node, 0.0),
                )
                for node in domain.nodes
            ],
            dtype=float,
        )
    else:
        self_return = np.array(transfer, copy=True)
    allocated = self_return + transfer
    tolerance = 64.0 * np.finfo(float).eps * max(
        float(np.max(beta, initial=0.0)),
        1.0,
    )
    allocation_limit = (
        beta
        if config.pair_conductance_source == "exact_component_schur"
        else config.closure_mass_fraction * beta
    )
    if np.any(allocated > allocation_limit + tolerance):
        raise ValueError(
            "closure self-return plus transfer oversubscribes the frozen beta "
            "mass fraction"
        )
    residual = beta - allocated
    if np.any(residual < -tolerance):
        raise ValueError("closure ledger oversubscribes original beta")
    residual = np.maximum(residual, 0.0)
    closure_laplacian = (
        np.diag(np.sum(closure_conductance, axis=1)) - closure_conductance
    )
    model = _build_local_from_components(
        domain,
        closure_conductance,
        closure_laplacian,
        intrinsic,
        residual,
        semantic_length_scale=(
            None if node_embeddings is None else float(semantic_length_scale)
        ),
        conductance_floor=float(conductance_floor),
        bath_temperature=0.0,
        minimum_reciprocal_condition=minimum_reciprocal_condition,
    )
    baseline_precision = base_laplacian + np.diag(intrinsic + beta)
    schur_update = np.diag(self_return)
    for left, right, value in edges:
        left_row = index[left]
        right_row = index[right]
        schur_update[left_row, right_row] += value
        schur_update[right_row, left_row] += value
    scale = max(
        float(np.max(np.abs(baseline_precision), initial=0.0)),
        1.0,
    )
    if not np.allclose(
        model.precision,
        baseline_precision - schur_update,
        rtol=1e-12,
        atol=64.0 * np.finfo(float).eps * scale,
    ):
        raise ValueError("closure precision does not match the Schur-shaped update")
    minimum_update_eigenvalue = float(np.min(np.linalg.eigvalsh(schur_update)))
    if minimum_update_eigenvalue < -64.0 * np.finfo(float).eps * scale:
        raise ValueError("closure Schur-shaped update must be positive semidefinite")
    payload = {
        "nodes": [_stable_node_token(node) for node in domain.nodes],
        "supplied_pairs": [
            [_stable_node_token(left), _stable_node_token(right), float(value).hex()]
            for left, right, value in supplied_edges
        ],
        "realized_edges": [
            [_stable_node_token(left), _stable_node_token(right), value.hex()]
            for left, right, value in edges
        ],
        "beta": [float(value).hex() for value in beta],
        "transfer": [float(value).hex() for value in transfer],
        "self_return": [float(value).hex() for value in self_return],
        "residual": [float(value).hex() for value in residual],
        "config": asdict(config),
    }
    return ExperimentalBoundaryClosure(
        model=model,
        edges=tuple(edges),
        original_cut_conductance=beta,
        transfer_degree=transfer,
        self_return_conductance=self_return,
        residual_ground_conductance=residual,
        supplied_pair_count=len(supplied_edges),
        filtered_pair_count=len(supplied_edges) - len(candidates),
        config=config,
        ledger_fingerprint=_fingerprint(payload),
    )



def _leakage_for_nodes(value, nodes, all_required_nodes):
    if isinstance(value, Mapping):
        missing = set(all_required_nodes).difference(value)
        if missing:
            labels = ", ".join(repr(node) for node in sorted(missing, key=_stable_key))
            raise ValueError("intrinsic alpha mapping misses selected nodes: " + labels)
        output = {
            node: _finite_nonnegative("intrinsic alpha", value[node])
            for node in nodes
        }
    else:
        scalar = _positive_finite("intrinsic_leakage_conductance", value)
        output = scalar
    return output


def _rank_order(nodes, values):
    return tuple(
        sorted(
            nodes,
            key=lambda node: (-float(values[node]), _stable_key(node)),
        )
    )


def _kendall_total_order(left, right):
    if set(left) != set(right):
        raise ValueError("rankings must contain the same nodes")
    if len(left) < 2:
        return 1.0
    right_position = {node: position for position, node in enumerate(right)}
    concordant = 0
    discordant = 0
    for i, left_node in enumerate(left[:-1]):
        left_position = right_position[left_node]
        for right_node in left[i + 1 :]:
            if left_position < right_position[right_node]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


@dataclass(frozen=True)
class BoundedFidelityResult:
    """Outcome-blind fidelity summary; no field chooses or promotes a selector."""

    candidate_strategy: str
    reference_strategy: str
    candidate_selection_fingerprint: str
    reference_selection_fingerprint: str
    alpha_fingerprint: str
    protected_nodes: tuple
    source_nodes: tuple
    per_anchor_raw_relative_l2_error: tuple
    per_anchor_maximum_h_absolute_error: tuple
    per_anchor_rank_inversion_fraction: tuple
    per_anchor_top_k_overlap: tuple
    per_anchor_source_diagonal_relative_error: tuple
    raw_relative_l2_error_90th_percentile: float
    maximum_h_absolute_error_90th_percentile: float
    rank_inversion_fraction_90th_percentile: float
    top_k_overlap_10th_percentile: float
    candidate_nodes: int
    reference_nodes: int
    protected_nodes_count: int
    protected_candidate_fraction: float
    protected_reference_fraction: float
    maximum_raw_absolute_error: float
    maximum_raw_relative_error: float
    raw_relative_frobenius_error: float
    maximum_h_absolute_error: float
    h_root_mean_square_error: float
    mean_kendall_rank_agreement: float
    minimum_kendall_rank_agreement: float
    mean_top_k_overlap: float
    minimum_top_k_overlap: float
    rank_top_k: int
    rank_excludes_source: bool
    candidate_boundary_harmonic_max: float
    reference_boundary_harmonic_max: float
    candidate_cut_current_fraction_max: float
    reference_cut_current_fraction_max: float
    candidate_reciprocal_condition: float
    reference_reciprocal_condition: float
    candidate_selection_seconds: float
    reference_selection_seconds: float
    candidate_build_seconds: float
    reference_build_seconds: float
    solve_seconds: float
    effective_resistance_evaluated: bool
    maximum_effective_resistance_absolute_error: float | None
    maximum_effective_resistance_relative_error: float | None
    per_anchor_effective_resistance_relative_error: tuple | None
    effective_resistance_relative_error_90th_percentile: float | None
    closure_edges: int
    closure_mass_fraction: float
    closure_ledger_fingerprint: str | None
    closure_pair_source: str | None
    closure_approximation_limits_apply: bool
    closure_supplied_pairs: int
    closure_filtered_pairs: int
    closure_realized_pairs: int
    closure_total_transfer_mass: float
    closure_total_self_return_mass: float
    closure_policy: str = _NO_CLOSURE_POLICY

    def __post_init__(self):
        if not self.protected_nodes or not self.source_nodes:
            raise ValueError("protected_nodes and source_nodes must be non-empty")
        if self.protected_nodes_count != len(self.protected_nodes):
            raise ValueError("protected_nodes_count must match protected_nodes")
        per_anchor_vectors = (
            self.per_anchor_raw_relative_l2_error,
            self.per_anchor_maximum_h_absolute_error,
            self.per_anchor_rank_inversion_fraction,
            self.per_anchor_top_k_overlap,
            self.per_anchor_source_diagonal_relative_error,
        )
        if any(len(values) != len(self.source_nodes) for values in per_anchor_vectors):
            raise ValueError("per-anchor metric vectors must align with source_nodes")
        if any(
            not math.isfinite(float(value)) or value < 0.0
            for values in per_anchor_vectors
            for value in values
        ):
            raise ValueError("per-anchor metric vectors must be finite and nonnegative")
        if any(value > 1.0 for value in self.per_anchor_top_k_overlap):
            raise ValueError("top-k overlap must be in [0, 1]")
        if any(value > 1.0 for value in self.per_anchor_rank_inversion_fraction):
            raise ValueError("rank-inversion fraction must be in [0, 1]")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.rank_excludes_source is not True:
            raise ValueError("rank_excludes_source must remain true")
        if not isinstance(
            self.closure_approximation_limits_apply, (bool, np.bool_)
        ):
            raise ValueError(
                "closure_approximation_limits_apply must be boolean"
            )
        if (
            self.closure_pair_source == "exact_component_schur"
            and self.closure_approximation_limits_apply
        ):
            raise ValueError(
                "exact component Schur reductions cannot apply approximate limits"
            )
        if not isinstance(self.effective_resistance_evaluated, (bool, np.bool_)):
            raise ValueError("effective_resistance_evaluated must be boolean")
        resistance_values = (
            self.maximum_effective_resistance_absolute_error,
            self.maximum_effective_resistance_relative_error,
        )
        resistance_anchor_values = (
            self.per_anchor_effective_resistance_relative_error
        )
        resistance_summary = (
            self.effective_resistance_relative_error_90th_percentile
        )
        if bool(self.effective_resistance_evaluated) != all(
            value is not None for value in resistance_values
        ):
            raise ValueError(
                "effective-resistance errors must be present exactly when evaluated"
            )
        if bool(self.effective_resistance_evaluated) != (
            resistance_anchor_values is not None
            and resistance_summary is not None
        ):
            raise ValueError(
                "per-anchor resistance endpoints must be present exactly when evaluated"
            )
        if resistance_anchor_values is not None:
            if len(resistance_anchor_values) != len(self.source_nodes):
                raise ValueError("per-anchor resistance must align with source_nodes")
            if any(
                not math.isfinite(float(value)) or value < 0.0
                for value in resistance_anchor_values
            ):
                raise ValueError(
                    "per-anchor resistance errors must be finite and nonnegative"
                )
        if any(value is not None and value < 0.0 for value in resistance_values):
            raise ValueError("effective-resistance errors must be nonnegative")
        closure_counts = (
            self.closure_edges,
            self.closure_supplied_pairs,
            self.closure_filtered_pairs,
            self.closure_realized_pairs,
        )
        if any(int(value) != value or value < 0 for value in closure_counts):
            raise ValueError("closure counts must be nonnegative integers")
        if self.closure_realized_pairs != self.closure_edges:
            raise ValueError("closure_edges must equal closure_realized_pairs")
        if self.closure_filtered_pairs > self.closure_supplied_pairs:
            raise ValueError("filtered closure pairs cannot exceed supplied pairs")
        if (
            self.closure_realized_pairs
            > self.closure_supplied_pairs - self.closure_filtered_pairs
        ):
            raise ValueError("realized closure pairs exceed unfiltered supplied pairs")
        closure_masses = (
            self.closure_total_transfer_mass,
            self.closure_total_self_return_mass,
        )
        if any(value < 0.0 for value in closure_masses):
            raise ValueError("closure ledger masses must be nonnegative")
        if not 0.0 <= self.closure_mass_fraction <= 1.0:
            raise ValueError("closure_mass_fraction must be in [0, 1]")
        if self.closure_policy == _NO_CLOSURE_POLICY:
            if (
                any(closure_counts)
                or any(closure_masses)
                or self.closure_mass_fraction
                or self.closure_ledger_fingerprint is not None
                or self.closure_pair_source is not None
            ):
                raise ValueError("no-closure result must not carry a closure ledger")
        elif self.closure_policy == _EXPERIMENTAL_CLOSURE_POLICY:
            if self.closure_ledger_fingerprint is None:
                raise ValueError("experimental closure must carry a ledger fingerprint")
            if self.closure_pair_source not in (
                "two_terminal_series",
                "sparse_dtn",
                "exact_component_schur",
            ):
                raise ValueError("experimental closure must record its pair source")
        else:
            raise ValueError("unknown closure_policy")


    def as_dict(self):
        return asdict(self)


def evaluate_bounded_domain_fidelity(
    candidate,
    reference,
    *,
    protected_nodes,
    intrinsic_leakage_conductance,
    node_embeddings=None,
    length_scale=None,
    conductance_floor=0.0,
    rank_top_k=None,
    minimum_reciprocal_condition=None,
    boundary_closure_config=None,
    boundary_closure_pair_conductances=None,
    boundary_closure_self_return=None,
    boundary_closure_filter_embeddings=None,
    boundary_closure_filter_length_scale=None,
    include_effective_resistance=False,
):
    """Compare one bounded domain with a larger exact-Dirichlet reference.

    ``intrinsic_leakage_conductance`` is one frozen scalar or a global mapping
    used by both domains.  It is never recalibrated inside this function.
    ``protected_nodes`` must be an explicit non-empty subset of both domains;
    missing nodes are rejected rather than silently placed last in rankings.
    """

    if not isinstance(candidate, BoundedDomainSelection) or not isinstance(
        reference, BoundedDomainSelection
    ):
        raise TypeError("candidate and reference must be BoundedDomainSelection values")
    if candidate.domain.anchors != reference.domain.anchors:
        raise ValueError("candidate and reference must use the same anchor union")
    if reference.realized_nodes <= candidate.realized_nodes:
        raise ValueError("reference domain must contain more nodes than candidate")
    protected = _canonical_unique(protected_nodes, name="protected_nodes")
    candidate_set = set(candidate.domain.nodes)
    reference_set = set(reference.domain.nodes)
    if not candidate_set.issubset(reference_set):
        missing = tuple(sorted(candidate_set.difference(reference_set), key=_stable_key))
        raise ValueError(
            "candidate domain must be a subset of the reference domain; "
            f"reference misses {missing!r}"
        )
    candidate_neighbors = candidate.domain.neighbor_mapping
    reference_neighbors = reference.domain.neighbor_mapping
    changed_adjacency = tuple(
        node
        for node in candidate.domain.nodes
        if candidate_neighbors[node] != reference_neighbors[node]
    )
    if changed_adjacency:
        raise ValueError(
            "candidate and reference must use identical complete incident "
            f"adjacency on shared nodes; changed {changed_adjacency!r}"
        )
    missing_candidate = set(protected).difference(candidate_set)
    missing_reference = set(protected).difference(reference_set)
    if missing_candidate or missing_reference:
        raise ProtectedSetCoverageError(missing_candidate, missing_reference)
    sources = candidate.domain.anchors
    if not set(sources).issubset(protected):
        raise ValueError("protected_nodes must include every source anchor")
    if not set(sources).issubset(reference_set):
        raise ValueError("every candidate anchor must be retained by the reference")
    minimum_rankable = min(
        len(protected) - int(source in protected) for source in sources
    )
    if minimum_rankable <= 0:
        raise ValueError(
            "protected set must include a non-source node for every source ranking"
        )
    if rank_top_k is None:
        rank_top_k = min(8, minimum_rankable)
    rank_top_k = _positive_integer("rank_top_k", rank_top_k)
    if rank_top_k > minimum_rankable:
        raise ValueError(
            "rank_top_k cannot exceed the smallest source-excluded protected set"
        )

    required_nodes = candidate_set.union(reference_set)
    candidate_alpha = _leakage_for_nodes(
        intrinsic_leakage_conductance,
        candidate.domain.nodes,
        required_nodes,
    )
    reference_alpha = _leakage_for_nodes(
        intrinsic_leakage_conductance,
        reference.domain.nodes,
        required_nodes,
    )
    alpha_payload = []
    for node in sorted(required_nodes, key=_stable_key):
        value = (
            intrinsic_leakage_conductance[node]
            if isinstance(intrinsic_leakage_conductance, Mapping)
            else intrinsic_leakage_conductance
        )
        alpha_payload.append([_stable_node_token(node), float(value).hex()])
    alpha_fingerprint = _fingerprint(alpha_payload)

    build_arguments = {
        "node_embeddings": node_embeddings,
        "length_scale": length_scale,
        "conductance_floor": conductance_floor,
        "minimum_reciprocal_condition": minimum_reciprocal_condition,
    }
    build_arguments = {
        name: value
        for name, value in build_arguments.items()
        if value is not None
    }
    if not isinstance(include_effective_resistance, (bool, np.bool_)):
        raise ValueError("include_effective_resistance must be boolean")
    started = time.perf_counter()
    closure = None
    if boundary_closure_config is None:
        if (
            boundary_closure_pair_conductances is not None
            or boundary_closure_self_return is not None
            or boundary_closure_filter_embeddings is not None
            or boundary_closure_filter_length_scale is not None
        ):
            raise ValueError(
                "closure pair ledger inputs require boundary_closure_config"
            )
        candidate_model = build_local_grounded_semantic_diffusion(
            candidate.domain,
            intrinsic_leakage_conductance=candidate_alpha,
            **build_arguments,
        )
    else:
        closure = build_experimental_boundary_closure(
            candidate.domain,
            intrinsic_leakage_conductance=candidate_alpha,
            pair_conductances=boundary_closure_pair_conductances,
            self_return_conductance=boundary_closure_self_return,
            node_embeddings=node_embeddings,
            semantic_filter_embeddings=(
                boundary_closure_filter_embeddings
            ),
            semantic_filter_length_scale=(
                boundary_closure_filter_length_scale
            ),
            semantic_length_scale=length_scale,
            conductance_floor=conductance_floor,
            config=boundary_closure_config,
            minimum_reciprocal_condition=minimum_reciprocal_condition,
        )
        candidate_model = closure.model
    candidate_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference_model = build_local_grounded_semantic_diffusion(
        reference.domain,
        intrinsic_leakage_conductance=reference_alpha,
        **build_arguments,
    )
    reference_build_seconds = time.perf_counter() - started

    candidate_index = {node: row for row, node in enumerate(candidate_model.nodes)}
    reference_index = {node: row for row, node in enumerate(reference_model.nodes)}
    candidate_protected = np.asarray([candidate_index[node] for node in protected])
    reference_protected = np.asarray([reference_index[node] for node in protected])
    started = time.perf_counter()
    candidate_source = np.zeros((candidate.realized_nodes, len(sources)))
    reference_source = np.zeros((reference.realized_nodes, len(sources)))
    for column, node in enumerate(sources):
        candidate_source[candidate_index[node], column] = 1.0
        reference_source[reference_index[node], column] = 1.0
    candidate_full = candidate_model.equilibrium_response(candidate_source)
    reference_full = reference_model.equilibrium_response(reference_source)
    solve_seconds = time.perf_counter() - started
    candidate_raw = candidate_full[candidate_protected, :]
    reference_raw = reference_full[reference_protected, :]
    difference = candidate_raw - reference_raw
    maximum_absolute = float(np.max(np.abs(difference), initial=0.0))
    denominator = np.maximum(np.abs(reference_raw), np.finfo(float).tiny)
    maximum_relative = float(
        np.max(np.abs(difference) / denominator, initial=0.0)
    )
    reference_norm = float(np.linalg.norm(reference_raw))
    relative_frobenius = float(np.linalg.norm(difference)) / max(
        reference_norm,
        np.finfo(float).tiny,
    )
    per_anchor_raw_relative_l2 = np.linalg.norm(difference, axis=0) / np.maximum(
        np.linalg.norm(reference_raw, axis=0), np.finfo(float).tiny
    )

    candidate_diagonal = np.asarray(
        [
            candidate_full[candidate_index[node], column]
            for column, node in enumerate(sources)
        ]
    )
    reference_diagonal = np.asarray(
        [
            reference_full[reference_index[node], column]
            for column, node in enumerate(sources)
        ]
    )
    if np.any(candidate_diagonal <= 0.0) or np.any(reference_diagonal <= 0.0):
        raise np.linalg.LinAlgError("anchor Green diagonals must be positive")
    candidate_h = candidate_raw / candidate_diagonal[None, :]
    reference_h = reference_raw / reference_diagonal[None, :]
    h_difference = candidate_h - reference_h
    maximum_h = float(np.max(np.abs(h_difference), initial=0.0))
    h_rmse = float(np.sqrt(np.mean(np.square(h_difference))))
    per_anchor_maximum_h = np.max(np.abs(h_difference), axis=0)
    per_anchor_diagonal_relative = (
        np.abs(candidate_diagonal - reference_diagonal)
        / reference_diagonal
    )

    kendall = []
    overlaps = []
    inversions = []
    for column, source_node in enumerate(sources):
        rank_nodes = tuple(node for node in protected if node != source_node)
        candidate_values = {
            node: candidate_h[row, column] for row, node in enumerate(protected)
        }
        reference_values = {
            node: reference_h[row, column] for row, node in enumerate(protected)
        }
        candidate_order = _rank_order(rank_nodes, candidate_values)
        reference_order = _rank_order(rank_nodes, reference_values)
        kendall.append(_kendall_total_order(candidate_order, reference_order))
        inversions.append(0.5 * (1.0 - kendall[-1]))
        overlaps.append(
            len(
                set(candidate_order[:rank_top_k]).intersection(
                    reference_order[:rank_top_k]
                )
            )
            / rank_top_k
        )

    resistance_absolute = None
    resistance_relative = None
    per_anchor_resistance_relative = None
    resistance_relative_p90 = None
    if include_effective_resistance:
        pairs = [
            (source_column, source_node, target_node)
            for source_column, source_node in enumerate(sources)
            for target_node in protected
            if target_node != source_node
        ]
        candidate_rhs = np.zeros((candidate.realized_nodes, len(pairs)))
        reference_rhs = np.zeros((reference.realized_nodes, len(pairs)))
        for column, (_, source_node, target_node) in enumerate(pairs):
            candidate_rhs[candidate_index[source_node], column] = 1.0
            candidate_rhs[candidate_index[target_node], column] = -1.0
            reference_rhs[reference_index[source_node], column] = 1.0
            reference_rhs[reference_index[target_node], column] = -1.0
        resistance_started = time.perf_counter()
        candidate_voltage = candidate_model.equilibrium_response(candidate_rhs)
        reference_voltage = reference_model.equilibrium_response(reference_rhs)
        solve_seconds += time.perf_counter() - resistance_started
        candidate_resistance = np.sum(candidate_rhs * candidate_voltage, axis=0)
        reference_resistance = np.sum(reference_rhs * reference_voltage, axis=0)
        if (
            not np.isfinite(candidate_resistance).all()
            or not np.isfinite(reference_resistance).all()
            or np.any(candidate_resistance <= 0.0)
            or np.any(reference_resistance <= 0.0)
        ):
            raise np.linalg.LinAlgError(
                "selected grounded effective resistances must be positive"
            )
        resistance_difference = np.abs(
            candidate_resistance - reference_resistance
        )
        relative_values = resistance_difference / reference_resistance
        resistance_absolute = float(
            np.max(resistance_difference, initial=0.0)
        )
        resistance_relative = float(np.max(relative_values, initial=0.0))
        per_anchor_resistance_relative = tuple(
            float(
                np.max(
                    [
                        relative_values[column]
                        for column, (owner, _, __) in enumerate(pairs)
                        if owner == source_column
                    ],
                    initial=0.0,
                )
            )
            for source_column in range(len(sources))
        )
        resistance_relative_p90 = float(
            np.quantile(per_anchor_resistance_relative, 0.9)
        )
    candidate_harmonic = candidate_model.boundary_harmonic_measure()[
        candidate_protected
    ]
    reference_harmonic = reference_model.boundary_harmonic_measure()[
        reference_protected
    ]
    candidate_cut = []
    reference_cut = []
    for node in sources:
        candidate_cut.append(
            candidate_model.cut_current_fraction(
                np.eye(1, candidate.realized_nodes, candidate_index[node]).reshape(-1)
            )
        )
        reference_cut.append(
            reference_model.cut_current_fraction(
                np.eye(1, reference.realized_nodes, reference_index[node]).reshape(-1)
            )
        )

    return BoundedFidelityResult(
        candidate_strategy=candidate.strategy,
        reference_strategy=reference.strategy,
        candidate_selection_fingerprint=candidate.selection_fingerprint,
        reference_selection_fingerprint=reference.selection_fingerprint,
        alpha_fingerprint=alpha_fingerprint,
        protected_nodes=protected,
        source_nodes=sources,
        per_anchor_raw_relative_l2_error=tuple(
            float(value) for value in per_anchor_raw_relative_l2
        ),
        per_anchor_maximum_h_absolute_error=tuple(
            float(value) for value in per_anchor_maximum_h
        ),
        per_anchor_rank_inversion_fraction=tuple(
            float(value) for value in inversions
        ),
        per_anchor_top_k_overlap=tuple(float(value) for value in overlaps),
        per_anchor_source_diagonal_relative_error=tuple(
            float(value) for value in per_anchor_diagonal_relative
        ),
        raw_relative_l2_error_90th_percentile=float(
            np.quantile(per_anchor_raw_relative_l2, 0.9)
        ),
        maximum_h_absolute_error_90th_percentile=float(
            np.quantile(per_anchor_maximum_h, 0.9)
        ),
        rank_inversion_fraction_90th_percentile=float(
            np.quantile(inversions, 0.9)
        ),
        top_k_overlap_10th_percentile=float(np.quantile(overlaps, 0.1)),
        candidate_nodes=candidate.realized_nodes,
        reference_nodes=reference.realized_nodes,
        protected_nodes_count=len(protected),
        protected_candidate_fraction=len(protected) / candidate.realized_nodes,
        protected_reference_fraction=len(protected) / reference.realized_nodes,
        maximum_raw_absolute_error=maximum_absolute,
        maximum_raw_relative_error=maximum_relative,
        raw_relative_frobenius_error=relative_frobenius,
        maximum_h_absolute_error=maximum_h,
        h_root_mean_square_error=h_rmse,
        mean_kendall_rank_agreement=float(np.mean(kendall)),
        minimum_kendall_rank_agreement=float(np.min(kendall)),
        mean_top_k_overlap=float(np.mean(overlaps)),
        minimum_top_k_overlap=float(np.min(overlaps)),
        rank_top_k=rank_top_k,
        rank_excludes_source=True,
        candidate_boundary_harmonic_max=float(
            np.max(candidate_harmonic, initial=0.0)
        ),
        reference_boundary_harmonic_max=float(
            np.max(reference_harmonic, initial=0.0)
        ),
        candidate_cut_current_fraction_max=float(np.max(candidate_cut)),
        reference_cut_current_fraction_max=float(np.max(reference_cut)),
        candidate_reciprocal_condition=(
            candidate_model.model.reciprocal_condition_number
        ),
        reference_reciprocal_condition=(
            reference_model.model.reciprocal_condition_number
        ),
        candidate_selection_seconds=candidate.selection_seconds,
        reference_selection_seconds=reference.selection_seconds,
        candidate_build_seconds=candidate_build_seconds,
        reference_build_seconds=reference_build_seconds,
        solve_seconds=solve_seconds,
        effective_resistance_evaluated=bool(include_effective_resistance),
        maximum_effective_resistance_absolute_error=resistance_absolute,
        maximum_effective_resistance_relative_error=resistance_relative,
        per_anchor_effective_resistance_relative_error=(
            per_anchor_resistance_relative
        ),
        effective_resistance_relative_error_90th_percentile=(
            resistance_relative_p90
        ),
        closure_edges=0 if closure is None else len(closure.edges),
        closure_mass_fraction=(
            0.0 if closure is None else closure.config.closure_mass_fraction
        ),
        closure_ledger_fingerprint=(
            None if closure is None else closure.ledger_fingerprint
        ),
        closure_pair_source=(None if closure is None else closure.pair_source),
        closure_approximation_limits_apply=(
            closure is not None
            and closure.pair_source != "exact_component_schur"
        ),
        closure_supplied_pairs=(
            0 if closure is None else closure.supplied_pair_count
        ),
        closure_filtered_pairs=(
            0 if closure is None else closure.filtered_pair_count
        ),
        closure_realized_pairs=(
            0 if closure is None else closure.realized_pair_count
        ),
        closure_total_transfer_mass=(
            0.0 if closure is None else closure.total_transfer_mass
        ),
        closure_total_self_return_mass=(
            0.0 if closure is None else closure.total_self_return_mass
        ),
        closure_policy=(
            _NO_CLOSURE_POLICY
            if closure is None
            else _EXPERIMENTAL_CLOSURE_POLICY
        ),
    )
