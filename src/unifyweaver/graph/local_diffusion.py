"""Query-local Dirichlet domains for grounded semantic diffusion.

The global graph may be arbitrarily large.  This module retains a bounded,
topology-connected node set, clamps the omitted exterior to a common bath, and
reuses the dense grounded-diffusion implementation as a small-domain
correctness backend.

For retained nodes S, every omitted edge contributes its conductance to a
boundary shunt,

    beta_i = sum_{j outside S} c_ij,

so the local precision is the exact Dirichlet principal block

    J_S = L(W_SS) + diag(alpha + beta).

The inverse of this operator is a conditional/bath-clamped Green kernel.  It
is not the marginal principal block of the full Green kernel, which would
require a Schur complement.
"""

from __future__ import annotations

from collections import deque

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .leaky_diffusion import (
    GroundedSemanticDiffusion,
    _DEFAULT_MINIMUM_RECIPROCAL_CONDITION,
    _embedding_matrix,
    _positive_finite,
    _positive_unit_interval,
    _stable_radial_factor,
    _unit_interval,
    _build_grounded_semantic_diffusion_from_components,
    combinatorial_laplacian,
    semantic_conductance_matrix,
)


__all__ = [
    "AnchorLeakageCalibrationResult",
    "AnchorScreeningProvenance",
    "LeakageCalibrationMinimalityCertificate",
    "LeakageCalibrationResult",
    "LocalDiffusionDomain",
    "LocalGroundedSemanticDiffusion",
    "NestedDomainDiagnostics",
    "PerAnchorLeakageCalibrationResult",
    "build_local_grounded_semantic_diffusion",
    "calibrate_uniform_leakage",
    "calibrate_uniform_leakage_per_anchor",
    "compare_nested_domains",
    "select_hop_local_domain",
]


def _stable_key(value):
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _positive_integer(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_scalar(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(value, *, dtype=float):
    output = np.array(value, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


def _canonical_unique(values, *, name):
    raw = tuple(values)
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    try:
        result = tuple(sorted(set(raw), key=_stable_key))
    except TypeError as exc:
        raise ValueError(f"{name} must contain hashable nodes") from exc
    return result


def _read_incident_neighbors(provider, node):
    try:
        if callable(provider):
            values = provider(node)
        elif isinstance(provider, Mapping):
            # An explicit empty value denotes a genuine isolate.  A missing
            # key is incomplete adjacency and must not silently delete edges.
            values = provider[node]
        else:
            raise TypeError(
                "incident-neighbor provider must be callable or a mapping"
            )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unable to read incident neighbors for {node!r}") from exc
    if values is None:
        values = ()
    try:
        result = tuple(
            sorted({value for value in values if value != node}, key=_stable_key)
        )
    except TypeError as exc:
        raise ValueError("incident neighbor identifiers must be hashable") from exc
    return result


@dataclass(frozen=True)
class LocalDiffusionDomain:
    """A deterministic hop-local domain with complete incident adjacency."""

    nodes: tuple
    anchors: tuple
    hop_distance: np.ndarray
    neighbors: tuple
    maximum_nodes: int
    complete_distance_shell: bool
    truncated_tie_count: int
    selection_metric: str = "hop_distance"

    def __post_init__(self):
        nodes = tuple(self.nodes)
        anchors = tuple(self.anchors)
        if not nodes:
            raise ValueError("nodes must be non-empty and unique")
        if not anchors:
            raise ValueError("anchors must be non-empty and unique")
        try:
            node_index = {node: row for row, node in enumerate(nodes)}
            anchor_set = set(anchors)
        except TypeError as exc:
            raise ValueError("nodes and anchors must be hashable") from exc
        if len(node_index) != len(nodes):
            raise ValueError("nodes must be non-empty and unique")
        if len(anchor_set) != len(anchors):
            raise ValueError("anchors must be non-empty and unique")
        if not anchor_set.issubset(node_index):
            raise ValueError("every anchor must be retained")
        maximum_nodes = _positive_integer("maximum_nodes", self.maximum_nodes)
        if len(anchors) > maximum_nodes:
            raise ValueError("maximum_nodes must retain every anchor")
        complete = bool(self.complete_distance_shell)
        if len(nodes) > maximum_nodes and not complete:
            raise ValueError("a hard-K domain cannot exceed maximum_nodes")
        try:
            truncated = int(self.truncated_tie_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "truncated_tie_count must be a nonnegative integer"
            ) from exc
        if truncated != self.truncated_tie_count or truncated < 0:
            raise ValueError("truncated_tie_count must be a nonnegative integer")
        if complete and truncated:
            raise ValueError("a complete distance shell cannot report truncated ties")

        raw_distance = np.asarray(self.hop_distance)
        if raw_distance.shape != (len(nodes),):
            raise ValueError("hop_distance must align with nodes")
        if not np.issubdtype(raw_distance.dtype, np.integer):
            if not np.isfinite(raw_distance).all() or not np.equal(
                raw_distance, np.floor(raw_distance)
            ).all():
                raise ValueError("hop_distance must contain nonnegative integers")
        distance = np.asarray(raw_distance, dtype=np.int64)
        if np.any(distance < 0):
            raise ValueError("hop_distance must contain nonnegative integers")
        if any(distance[node_index[anchor]] != 0 for anchor in anchors):
            raise ValueError("anchors must have hop distance zero")
        if np.any(distance[:-1] > distance[1:]):
            raise ValueError("nodes must be ordered by nondecreasing hop distance")
        if (
            len(nodes) > maximum_nodes
            and distance[maximum_nodes - 1] != distance[-1]
        ):
            raise ValueError("nodes beyond maximum_nodes must complete one shell")

        raw_neighbors = tuple(tuple(values) for values in self.neighbors)
        if len(raw_neighbors) != len(nodes):
            raise ValueError("neighbors must align with nodes")
        canonical_neighbors = []
        neighbor_sets = {}
        for node, values in zip(nodes, raw_neighbors):
            try:
                value_set = set(values)
            except TypeError as exc:
                raise ValueError(
                    "incident neighbor identifiers must be hashable"
                ) from exc
            if node in value_set:
                raise ValueError("incident neighbor lists must omit self loops")
            if len(value_set) != len(values):
                raise ValueError("incident neighbor lists must be unique")
            canonical_neighbors.append(tuple(sorted(value_set, key=_stable_key)))
            neighbor_sets[node] = value_set
        mapping = dict(zip(nodes, canonical_neighbors))
        for row, (node, values) in enumerate(mapping.items()):
            for neighbor in values:
                if neighbor in node_index:
                    neighbor_row = node_index[neighbor]
                    if abs(int(distance[row]) - int(distance[neighbor_row])) > 1:
                        raise ValueError("retained edges cannot skip a hop shell")
                    if node not in neighbor_sets[neighbor]:
                        raise ValueError(
                            "incident neighbors must provide reciprocal "
                            "undirected adjacency"
                        )
            if node not in anchor_set:
                parent_distance = distance[row] - 1
                if not any(
                    neighbor in node_index
                    and distance[node_index[neighbor]] == parent_distance
                    for neighbor in values
                ):
                    raise ValueError(
                        "every retained non-anchor must have a predecessor "
                        "one hop nearer"
                    )

        metric = str(self.selection_metric)
        if not metric:
            raise ValueError("selection_metric must be non-empty")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "hop_distance", _readonly(distance, dtype=np.int64))
        object.__setattr__(self, "neighbors", tuple(canonical_neighbors))
        object.__setattr__(self, "maximum_nodes", maximum_nodes)
        object.__setattr__(self, "complete_distance_shell", complete)
        object.__setattr__(self, "truncated_tie_count", truncated)
        object.__setattr__(self, "selection_metric", metric)

    @property
    def cutoff_distance(self):
        return int(np.max(self.hop_distance))

    @property
    def frontier_nodes(self):
        cutoff = self.cutoff_distance
        return tuple(
            node
            for node, distance in zip(self.nodes, self.hop_distance)
            if distance == cutoff
        )

    @property
    def distance_by_node(self):
        return {
            node: int(distance)
            for node, distance in zip(self.nodes, self.hop_distance)
        }

    @property
    def neighbor_mapping(self):
        return dict(zip(self.nodes, self.neighbors))


def select_hop_local_domain(
    anchors,
    incident_neighbors,
    *,
    maximum_nodes,
    complete_distance_shell=False,
):
    """Select a deterministic multi-source BFS domain without scanning all nodes.

    The neighbor provider must return the complete undirected incident
    adjacency for a requested node.  It may be a mapping, callable, CSR/LMDB
    adapter, or any object exposing get(node, default).

    With complete_distance_shell false, the result contains at most K nodes
    and breaks a tied final shell by a stable node key.  With it true, the
    entire first shell that reaches K is retained, so realized size may exceed
    K.  Either construction is connected to at least one anchor.
    """

    anchors = _canonical_unique(anchors, name="anchors")
    maximum_nodes = _positive_integer("maximum_nodes", maximum_nodes)
    if len(anchors) > maximum_nodes:
        raise ValueError("maximum_nodes must be at least the number of anchors")

    cache = {}

    def neighbors_for(node):
        if node not in cache:
            cache[node] = _read_incident_neighbors(incident_neighbors, node)
        return cache[node]

    selected = list(anchors)
    distance_by_node = {node: 0 for node in anchors}
    seen = set(anchors)
    layer = list(anchors)
    truncated = 0

    while layer and len(selected) < maximum_nodes:
        candidates = set()
        for node in layer:
            for neighbor in neighbors_for(node):
                if neighbor not in seen:
                    candidates.add(neighbor)
        ordered = tuple(sorted(candidates, key=_stable_key))
        if not ordered:
            break
        next_distance = distance_by_node[layer[0]] + 1
        remaining = maximum_nodes - len(selected)
        if len(ordered) > remaining:
            if complete_distance_shell:
                chosen = ordered
            else:
                chosen = ordered[:remaining]
                truncated = len(ordered) - remaining
            selected.extend(chosen)
            for node in chosen:
                distance_by_node[node] = next_distance
            seen.update(chosen)
            layer = list(chosen)
            break
        selected.extend(ordered)
        for node in ordered:
            distance_by_node[node] = next_distance
        seen.update(ordered)
        layer = list(ordered)

    for node in selected:
        neighbors_for(node)
    distances = np.asarray(
        [distance_by_node[node] for node in selected],
        dtype=np.int64,
    )
    return LocalDiffusionDomain(
        nodes=tuple(selected),
        anchors=anchors,
        hop_distance=distances,
        neighbors=tuple(cache[node] for node in selected),
        maximum_nodes=maximum_nodes,
        complete_distance_shell=(truncated == 0),
        truncated_tie_count=truncated,
    )


def _nonnegative_node_vector(name, nodes, value):
    if isinstance(value, Mapping):
        unknown = set(value).difference(nodes)
        if unknown:
            labels = ", ".join(sorted(repr(node) for node in unknown))
            raise ValueError(f"{name} contains unknown nodes: {labels}")
        result = np.asarray([value.get(node, 0.0) for node in nodes], dtype=float)
    else:
        result = np.asarray(value, dtype=float)
        if result.ndim == 0:
            result = np.full(len(nodes), float(result), dtype=float)
    if result.shape != (len(nodes),):
        raise ValueError(f"{name} must be scalar, node mapping, or aligned vector")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _embedding_row(node_embeddings, node, width):
    try:
        row = np.asarray(node_embeddings[node], dtype=float)
    except KeyError as exc:
        raise ValueError(
            f"exterior cut neighbor absent from embeddings: {exc.args[0]!r}"
        ) from exc
    if row.shape != (width,) or not np.isfinite(row).all():
        raise ValueError(
            "every retained and one-hop exterior embedding must have one finite width"
        )
    return row


def _assemble_local_components(
    domain,
    *,
    intrinsic_leakage_conductance,
    node_embeddings,
    length_scale,
    conductance_floor,
):
    if not isinstance(domain, LocalDiffusionDomain):
        raise TypeError("domain must be a LocalDiffusionDomain")
    nodes = domain.nodes
    intrinsic = _nonnegative_node_vector(
        "intrinsic_leakage_conductance",
        nodes,
        intrinsic_leakage_conductance,
    )
    cut = np.zeros(len(nodes), dtype=float)
    retained = set(nodes)
    exterior_embeddings = {}

    if node_embeddings is None:
        embeddings = None
        embedding_snapshot = None
        floor = 0.0
        scale = None
    else:
        scale = _positive_finite("length_scale", length_scale)
        floor = _unit_interval("conductance_floor", conductance_floor)
        embeddings = _embedding_matrix(nodes, node_embeddings)
        embedding_snapshot = {
            node: embeddings[row].copy()
            for row, node in enumerate(nodes)
        }
        width = embeddings.shape[1]


    _, conductance = semantic_conductance_matrix(
        nodes,
        domain.neighbor_mapping,
        embedding_snapshot,
        length_scale=length_scale,
        conductance_floor=conductance_floor,
    )
    laplacian = combinatorial_laplacian(conductance)
    for row, (node, incident) in enumerate(zip(nodes, domain.neighbors)):
        for neighbor in incident:
            if neighbor in retained:
                continue
            value = 1.0
            if embeddings is not None:
                exterior = exterior_embeddings.get(neighbor)
                if exterior is None:
                    exterior = _embedding_row(
                        node_embeddings, neighbor, width
                    ).copy()
                    exterior_embeddings[neighbor] = exterior
                radial = _stable_radial_factor(embeddings[row], exterior, scale)
                value = floor + (1.0 - floor) * radial
            cut[row] += value
        if not math.isfinite(float(cut[row])):
            raise ValueError(f"cut conductance is not finite at node {node!r}")

    return conductance, laplacian, intrinsic, cut


@dataclass(frozen=True)
class LocalGroundedSemanticDiffusion:
    """A local operator whose intrinsic and cut shunts share one common bath."""

    domain: LocalDiffusionDomain
    model: GroundedSemanticDiffusion
    intrinsic_leakage_conductance: np.ndarray
    cut_conductance: np.ndarray
    bath_temperature: float = 0.0

    def __post_init__(self):
        if not isinstance(self.domain, LocalDiffusionDomain):
            raise TypeError("domain must be a LocalDiffusionDomain")
        if not isinstance(self.model, GroundedSemanticDiffusion):
            raise TypeError("model must be a GroundedSemanticDiffusion")
        if self.model.nodes != self.domain.nodes:
            raise ValueError("model and local domain node order must match")
        intrinsic = np.asarray(self.intrinsic_leakage_conductance, dtype=float)
        cut = np.asarray(self.cut_conductance, dtype=float)
        expected_shape = (len(self.domain.nodes),)
        if intrinsic.shape != expected_shape or cut.shape != expected_shape:
            raise ValueError("intrinsic and cut conductance must align with nodes")
        if (
            not np.isfinite(intrinsic).all()
            or not np.isfinite(cut).all()
            or np.any(intrinsic < 0.0)
            or np.any(cut < 0.0)
        ):
            raise ValueError("intrinsic and cut conductance must be nonnegative")
        expected_leakage = intrinsic + cut
        leakage_scale = max(
            float(np.max(np.abs(expected_leakage), initial=0.0)),
            float(np.max(np.abs(self.model.leakage_conductance), initial=0.0)),
            np.finfo(float).tiny,
        )
        if not np.allclose(
            self.model.leakage_conductance,
            expected_leakage,
            rtol=1e-12,
            atol=64.0 * np.finfo(float).eps * leakage_scale,
        ):
            raise ValueError("model leakage must equal intrinsic plus cut conductance")
        object.__setattr__(
            self,
            "intrinsic_leakage_conductance",
            _readonly(intrinsic),
        )
        object.__setattr__(self, "cut_conductance", _readonly(cut))
        object.__setattr__(
            self,
            "bath_temperature",
            _finite_scalar("bath_temperature", self.bath_temperature),
        )

    @property
    def nodes(self):
        return self.domain.nodes

    @property
    def precision(self):
        return self.model.precision

    def equilibrium_response(self, source, *, absolute=False):
        """Return bath-relative deviation, or the absolute common-bath state.

        ``absolute=True`` assumes that both intrinsic leakage ``alpha`` and the
        Dirichlet cut ``beta`` terminate at ``bath_temperature``.  It does not
        model an exterior-only bath with intrinsic leakage held at another
        reference potential.
        """

        response = self.model.equilibrium_response(source)
        if absolute:
            with np.errstate(over="ignore", invalid="ignore"):
                response = response + self.bath_temperature
            if not np.isfinite(response).all():
                raise np.linalg.LinAlgError(
                    "absolute common-bath response is not finite"
                )
        return response

    def boundary_harmonic_measure(self):
        """Return A^-1 beta, the influence of an artificial unit bath boundary."""

        if not np.any(self.cut_conductance):
            return np.zeros(len(self.nodes), dtype=float)
        value = self.model.equilibrium_response(self.cut_conductance)
        scale = max(float(np.max(np.abs(value), initial=0.0)), 1.0)
        if np.min(value) < -1e-12 * scale or np.max(value) > 1.0 + 1e-10:
            raise np.linalg.LinAlgError("boundary harmonic measure left [0, 1]")
        return np.clip(value, 0.0, 1.0)

    def cut_current_fraction(self, source):
        """Fraction of a nonnegative maintained source drained by the cut."""

        source = np.asarray(source, dtype=float)
        if source.shape != (len(self.nodes),):
            raise ValueError("source must be one node-aligned vector")
        if not np.isfinite(source).all() or np.any(source < 0.0):
            raise ValueError("source must be finite and nonnegative")
        total = float(np.sum(source))
        if total <= 0.0:
            raise ValueError("source must inject positive total current")
        response = self.model.equilibrium_response(source)
        cut_current = float(self.cut_conductance @ response)
        leakage_current = float(self.intrinsic_leakage_conductance @ response)
        balance = cut_current + leakage_current
        scale = max(total, abs(cut_current), abs(leakage_current), 1.0)
        if not math.isfinite(balance) or abs(balance - total) > 1e-10 * scale:
            raise np.linalg.LinAlgError(
                "cut and intrinsic leakage currents do not balance the source"
            )
        fraction = cut_current / total
        if not math.isfinite(fraction) or fraction < -1e-12 or fraction > 1.0 + 1e-10:
            raise np.linalg.LinAlgError("cut current fraction left [0, 1]")
        return min(max(fraction, 0.0), 1.0)

    def frontier_attenuation(self, source_nodes=None, shell_nodes=None):
        """Maximum raw Green voltage ratio from sources to a fixed shell."""

        sources = _selected_nodes(
            self.nodes,
            self.domain.anchors if source_nodes is None else source_nodes,
            name="source_nodes",
        )
        shell = _selected_nodes(
            self.nodes,
            self.domain.frontier_nodes if shell_nodes is None else shell_nodes,
            name="shell_nodes",
        )
        index = {node: row for row, node in enumerate(self.nodes)}
        maximum = None
        for source_node in sources:
            source_row = index[source_node]
            rhs = np.zeros(len(self.nodes))
            rhs[source_row] = 1.0
            response = self.model.equilibrium_response(rhs)
            scale = max(float(np.max(np.abs(response), initial=0.0)), 1.0)
            if np.min(response) < -1e-11 * scale or response[source_row] <= 0.0:
                raise np.linalg.LinAlgError("Green response is not nonnegative")
            for shell_node in shell:
                if shell_node == source_node:
                    continue
                ratio = max(float(response[index[shell_node]]), 0.0) / float(
                    response[source_row]
                )
                maximum = ratio if maximum is None else max(maximum, ratio)
        if maximum is None:
            raise ValueError("attenuation shell must include a non-source node")
        return float(maximum)


def _selected_nodes(nodes, values, *, name):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of retained nodes") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    unknown = set(result).difference(nodes)
    if unknown:
        labels = ", ".join(sorted(repr(node) for node in unknown))
        raise ValueError(f"{name} contains unretained nodes: {labels}")
    return tuple(sorted(result, key=_stable_key))


def _build_local_from_components(
    domain,
    conductance,
    laplacian,
    intrinsic,
    cut,
    *,
    semantic_length_scale,
    conductance_floor,
    bath_temperature,
    minimum_reciprocal_condition,
):
    """Finish one local model without rereading graph or embedding providers."""

    arguments = {}
    if minimum_reciprocal_condition is not None:
        arguments["minimum_reciprocal_condition"] = minimum_reciprocal_condition
    model = _build_grounded_semantic_diffusion_from_components(
        domain.nodes,
        conductance,
        laplacian,
        leakage_conductance=intrinsic + cut,
        semantic_length_scale=semantic_length_scale,
        conductance_floor=float(conductance_floor),
        **arguments,
    )
    return LocalGroundedSemanticDiffusion(
        domain=domain,
        model=model,
        intrinsic_leakage_conductance=intrinsic,
        cut_conductance=cut,
        bath_temperature=bath_temperature,
    )


def build_local_grounded_semantic_diffusion(
    domain,
    *,
    intrinsic_leakage_conductance=0.0,
    node_embeddings=None,
    length_scale=None,
    conductance_floor=0.0,
    bath_temperature=0.0,
    minimum_reciprocal_condition=None,
):
    """Build the exact dense Dirichlet principal block on one local domain."""

    conductance, laplacian, intrinsic, cut = _assemble_local_components(
        domain,
        intrinsic_leakage_conductance=intrinsic_leakage_conductance,
        node_embeddings=node_embeddings,
        length_scale=length_scale,
        conductance_floor=conductance_floor,
    )
    return _build_local_from_components(
        domain,
        conductance,
        laplacian,
        intrinsic,
        cut,
        semantic_length_scale=(
            None if node_embeddings is None else float(length_scale)
        ),
        conductance_floor=float(conductance_floor),
        bath_temperature=bath_temperature,
        minimum_reciprocal_condition=minimum_reciprocal_condition,
    )


@dataclass(frozen=True)
class AnchorScreeningProvenance:
    """One anchor's discrete raw-response screening-radius provenance."""

    anchor: object
    shell_attenuation: float
    attenuation_threshold: float
    radius_lower: float
    radius_upper: float | None
    right_censored: bool
    maximum_observed_radius: float
    distance_metric: str

    def __post_init__(self):
        try:
            hash(self.anchor)
        except TypeError as exc:
            raise ValueError("anchor must be hashable") from exc
        attenuation = _finite_scalar(
            "shell_attenuation", self.shell_attenuation
        )
        if attenuation < 0.0 or attenuation > 1.0 + 1e-8:
            raise ValueError("shell_attenuation must be in [0, 1]")
        threshold = _positive_unit_interval(
            "attenuation_threshold", self.attenuation_threshold
        )
        lower = _finite_scalar("radius_lower", self.radius_lower)
        maximum = _finite_scalar(
            "maximum_observed_radius", self.maximum_observed_radius
        )
        if lower < 0.0 or maximum < 1.0 or lower > maximum:
            raise ValueError("screening radii must be nonnegative and ordered")
        if not isinstance(self.right_censored, (bool, np.bool_)):
            raise ValueError("right_censored must be boolean")
        censored = bool(self.right_censored)
        if censored:
            if self.radius_upper is not None or lower != maximum:
                raise ValueError(
                    "a right-censored radius must end at the observed maximum"
                )
            upper = None
        else:
            upper = _finite_scalar("radius_upper", self.radius_upper)
            if upper <= lower or upper > maximum:
                raise ValueError("screening-radius bracket must be ordered")
        metric = str(self.distance_metric)
        if not metric:
            raise ValueError("distance_metric must be non-empty")
        object.__setattr__(self, "shell_attenuation", attenuation)
        object.__setattr__(self, "attenuation_threshold", threshold)
        object.__setattr__(self, "radius_lower", lower)
        object.__setattr__(self, "radius_upper", upper)
        object.__setattr__(self, "right_censored", censored)
        object.__setattr__(self, "maximum_observed_radius", maximum)
        object.__setattr__(self, "distance_metric", metric)


@dataclass(frozen=True)
class LeakageCalibrationResult:
    model: LocalGroundedSemanticDiffusion
    leakage_conductance: float
    leakage_resistance: float
    target_attenuation: float
    achieved_attenuation: float
    source_nodes: tuple
    shell_nodes: tuple
    anchor_screening: tuple
    iterations: int
    numerical_minimum_leakage: float

    def __post_init__(self):
        if not isinstance(self.model, LocalGroundedSemanticDiffusion):
            raise TypeError("model must be a LocalGroundedSemanticDiffusion")
        sources = _selected_nodes(
            self.model.nodes,
            self.source_nodes,
            name="source_nodes",
        )
        shell = _selected_nodes(
            self.model.nodes,
            self.shell_nodes,
            name="shell_nodes",
        )
        leakage = _finite_scalar("leakage_conductance", self.leakage_conductance)
        if leakage < 0.0:
            raise ValueError("leakage_conductance must be nonnegative")
        try:
            resistance = float(self.leakage_resistance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "leakage_resistance must be reciprocal conductance"
            ) from exc
        expected = math.inf if leakage == 0.0 else 1.0 / leakage
        if leakage == 0.0 and resistance != math.inf:
            raise ValueError("leakage_resistance must be reciprocal conductance")
        if leakage > 0.0 and (
            not math.isfinite(resistance)
            or not math.isclose(resistance, expected, rel_tol=1e-12)
        ):
            raise ValueError("leakage_resistance must be reciprocal conductance")
        target = _positive_unit_interval(
            "target_attenuation", self.target_attenuation
        )
        screening = tuple(self.anchor_screening)
        if not screening or not all(
            isinstance(item, AnchorScreeningProvenance)
            for item in screening
        ):
            raise ValueError(
                "anchor_screening must contain per-anchor provenance"
            )
        if tuple(item.anchor for item in screening) != sources:
            raise ValueError(
                "anchor_screening must align exactly with source_nodes"
            )
        for item in screening:
            if not math.isclose(
                item.attenuation_threshold,
                target,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "per-anchor attenuation thresholds must match the target"
                )
        achieved = _finite_scalar("achieved_attenuation", self.achieved_attenuation)
        if achieved < 0.0 or achieved > target * (1.0 + 1e-8):
            raise ValueError("calibrated attenuation does not meet its target")
        per_anchor_maximum = max(
            item.shell_attenuation for item in screening
        )
        achieved_scale = max(
            abs(achieved),
            abs(per_anchor_maximum),
            np.finfo(float).tiny,
        )
        if abs(achieved - per_anchor_maximum) > (
            64.0 * np.finfo(float).eps * achieved_scale
        ):
            raise ValueError(
                "achieved attenuation must equal the per-anchor maximum"
            )
        if (
            isinstance(self.iterations, (bool, np.bool_))
            or int(self.iterations) != self.iterations
            or self.iterations < 0
        ):
            raise ValueError("iterations must be a nonnegative integer")
        iterations = int(self.iterations)
        numeric = _finite_scalar(
            "numerical_minimum_leakage", self.numerical_minimum_leakage
        )
        numeric_scale = max(
            abs(leakage),
            abs(numeric),
            np.finfo(float).tiny,
        )
        if (
            numeric < 0.0
            or leakage + 64.0 * np.finfo(float).eps * numeric_scale < numeric
        ):
            raise ValueError("selected leakage is below the numerical minimum")
        object.__setattr__(self, "source_nodes", sources)
        object.__setattr__(self, "shell_nodes", shell)
        object.__setattr__(self, "anchor_screening", screening)
        object.__setattr__(self, "leakage_conductance", leakage)
        object.__setattr__(self, "leakage_resistance", resistance)
        object.__setattr__(self, "target_attenuation", target)
        object.__setattr__(self, "achieved_attenuation", achieved)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "numerical_minimum_leakage", numeric)


@dataclass(frozen=True)
class LeakageCalibrationMinimalityCertificate:
    """Final attenuation bracket certifying one anchor's leakage minimum.

    Except when the conditioning-imposed initial lower endpoint already meets
    the attenuation target, the lower endpoint is known to miss the target and
    the upper endpoint is the selected leakage that meets it.  The remaining
    relative bracket width therefore states the numerical resolution of the
    minimality claim. ``bracket_seed_radius`` records only the radius used by
    the chain-model initializer; it is not the scientific shell definition.
    """

    lower_added_leakage_conductance: float
    upper_added_leakage_conductance: float
    attenuation_at_lower: float
    attenuation_at_upper: float
    target_attenuation: float
    relative_tolerance: float
    initial_lower_passed: bool
    bracket_seed_radius: int

    def __post_init__(self):
        lower = _finite_scalar(
            "lower_added_leakage_conductance",
            self.lower_added_leakage_conductance,
        )
        upper = _finite_scalar(
            "upper_added_leakage_conductance",
            self.upper_added_leakage_conductance,
        )
        lower_attenuation = _finite_scalar(
            "attenuation_at_lower",
            self.attenuation_at_lower,
        )
        upper_attenuation = _finite_scalar(
            "attenuation_at_upper",
            self.attenuation_at_upper,
        )
        target = _positive_unit_interval(
            "target_attenuation",
            self.target_attenuation,
        )
        tolerance = _positive_finite(
            "relative_tolerance",
            self.relative_tolerance,
        )
        if lower < 0.0 or upper < 0.0:
            raise ValueError("leakage bracket endpoints must be nonnegative")
        if lower > upper:
            raise ValueError("leakage bracket endpoints must be ordered")
        if (
            lower_attenuation < 0.0
            or lower_attenuation > 1.0 + 1e-8
            or upper_attenuation < 0.0
            or upper_attenuation > 1.0 + 1e-8
        ):
            raise ValueError("bracket attenuations must be in [0, 1]")
        attenuation_scale = max(
            abs(lower_attenuation),
            abs(upper_attenuation),
            np.finfo(float).tiny,
        )
        if upper_attenuation > lower_attenuation + (
            64.0 * np.finfo(float).eps * attenuation_scale
        ):
            raise ValueError(
                "attenuation must not increase across the leakage bracket"
            )
        if upper_attenuation > target:
            raise ValueError("upper leakage endpoint must meet the target")
        if not isinstance(self.initial_lower_passed, (bool, np.bool_)):
            raise ValueError("initial_lower_passed must be boolean")
        initial_passed = bool(self.initial_lower_passed)
        if initial_passed:
            if lower != upper or lower_attenuation != upper_attenuation:
                raise ValueError(
                    "an initial-lower pass must return a collapsed bracket"
                )
            if lower_attenuation > target:
                raise ValueError(
                    "an initial-lower pass must meet the attenuation target"
                )
        elif lower_attenuation <= target:
            raise ValueError(
                "the lower endpoint must miss the target unless the initial "
                "lower endpoint passed"
            )
        scale = max(abs(upper), np.finfo(float).tiny)
        relative_width = (upper - lower) / scale
        if not math.isfinite(relative_width) or relative_width > tolerance:
            raise ValueError(
                "leakage bracket exceeds its stated relative tolerance"
            )
        seed_radius = _positive_integer(
            "bracket_seed_radius",
            self.bracket_seed_radius,
        )
        object.__setattr__(self, "lower_added_leakage_conductance", lower)
        object.__setattr__(self, "upper_added_leakage_conductance", upper)
        object.__setattr__(self, "attenuation_at_lower", lower_attenuation)
        object.__setattr__(self, "attenuation_at_upper", upper_attenuation)
        object.__setattr__(self, "target_attenuation", target)
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "initial_lower_passed", initial_passed)
        object.__setattr__(self, "bracket_seed_radius", seed_radius)

    @property
    def relative_bracket_width(self):
        scale = max(
            abs(self.upper_added_leakage_conductance),
            np.finfo(float).tiny,
        )
        return (
            self.upper_added_leakage_conductance
            - self.lower_added_leakage_conductance
        ) / scale


@dataclass(frozen=True)
class AnchorLeakageCalibrationResult:
    """Independent leakage requirement and screening provenance for one anchor.

    ``added_leakage_conductance`` is the scalar shunt added uniformly on top of
    the caller-provided intrinsic leakage vector. Both screening records use
    this anchor's own shell. The first is evaluated at the anchor's independently
    selected leakage; the second is evaluated at the conservative shared study
    leakage (the maximum independent requirement across anchors).
    """

    anchor: object
    shell_nodes: tuple
    added_leakage_conductance: float
    achieved_attenuation: float
    iterations: int
    numerical_minimum_added_leakage: float
    screening_at_selected_leakage: AnchorScreeningProvenance
    screening_at_study_leakage: AnchorScreeningProvenance
    minimality_certificate: LeakageCalibrationMinimalityCertificate

    def __post_init__(self):
        try:
            hash(self.anchor)
        except TypeError as exc:
            raise ValueError("anchor must be hashable") from exc
        shell = _canonical_unique(self.shell_nodes, name="shell_nodes")
        added = _finite_scalar(
            "added_leakage_conductance",
            self.added_leakage_conductance,
        )
        numerical = _finite_scalar(
            "numerical_minimum_added_leakage",
            self.numerical_minimum_added_leakage,
        )
        if added < 0.0 or numerical < 0.0:
            raise ValueError("added leakage values must be nonnegative")
        scale = max(abs(added), abs(numerical), np.finfo(float).tiny)
        if added + 64.0 * np.finfo(float).eps * scale < numerical:
            raise ValueError("selected leakage is below the numerical minimum")
        achieved = _finite_scalar(
            "achieved_attenuation",
            self.achieved_attenuation,
        )
        if achieved < 0.0 or achieved > 1.0 + 1e-8:
            raise ValueError("achieved_attenuation must be in [0, 1]")
        if (
            isinstance(self.iterations, (bool, np.bool_))
            or int(self.iterations) != self.iterations
            or self.iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")
        iterations = int(self.iterations)
        selected = self.screening_at_selected_leakage
        study = self.screening_at_study_leakage
        if not isinstance(selected, AnchorScreeningProvenance) or not isinstance(
            study,
            AnchorScreeningProvenance,
        ):
            raise TypeError("screening provenance must use AnchorScreeningProvenance")
        if selected.anchor != self.anchor or study.anchor != self.anchor:
            raise ValueError("screening provenance anchor must match the result")
        certificate = self.minimality_certificate
        if not isinstance(
            certificate,
            LeakageCalibrationMinimalityCertificate,
        ):
            raise TypeError(
                "minimality_certificate must use "
                "LeakageCalibrationMinimalityCertificate"
            )
        if certificate.upper_added_leakage_conductance != added:
            raise ValueError(
                "selected leakage must equal the certificate upper endpoint"
            )
        if certificate.lower_added_leakage_conductance < numerical:
            raise ValueError(
                "minimality bracket cannot fall below the numerical minimum"
            )
        if certificate.initial_lower_passed and (
            certificate.lower_added_leakage_conductance != numerical
        ):
            raise ValueError(
                "an initial-lower pass must occur at the numerical minimum"
            )
        if not math.isclose(
            selected.attenuation_threshold,
            study.attenuation_threshold,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("selected and study screening thresholds must match")
        observed_scale = max(
            abs(achieved),
            abs(selected.shell_attenuation),
            np.finfo(float).tiny,
        )
        if abs(achieved - selected.shell_attenuation) > (
            64.0 * np.finfo(float).eps * observed_scale
        ):
            raise ValueError(
                "achieved attenuation must equal selected-leakage screening"
            )
        if not math.isclose(
            certificate.target_attenuation,
            selected.attenuation_threshold,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "minimality-certificate target must match screening"
            )
        certificate_scale = max(
            abs(achieved),
            abs(certificate.attenuation_at_upper),
            np.finfo(float).tiny,
        )
        if abs(achieved - certificate.attenuation_at_upper) > (
            128.0 * np.finfo(float).eps * certificate_scale
        ):
            raise ValueError(
                "selected attenuation must equal the certificate upper "
                "attenuation"
            )
        if study.shell_attenuation > selected.shell_attenuation + 1e-8:
            raise ValueError(
                "shared study leakage cannot weaken shell attenuation"
            )
        object.__setattr__(self, "shell_nodes", shell)
        object.__setattr__(self, "added_leakage_conductance", added)
        object.__setattr__(self, "achieved_attenuation", achieved)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "numerical_minimum_added_leakage", numerical)
        object.__setattr__(self, "minimality_certificate", certificate)


@dataclass(frozen=True)
class PerAnchorLeakageCalibrationResult:
    """One-factor study model plus independent per-anchor leakage requirements.

    The returned ``model`` uses ``base_intrinsic_leakage_conductance`` plus
    ``study_added_leakage_conductance``. The study addition is the maximum of
    the independently calibrated anchor additions, so one shared precision can
    safely serve the whole batch without pretending that the intrinsic leakage
    was learned or replaced.
    """

    model: LocalGroundedSemanticDiffusion
    base_intrinsic_leakage_conductance: np.ndarray
    study_added_leakage_conductance: float
    target_attenuation: float
    numerical_minimum_added_leakage: float
    per_anchor: tuple
    total_evaluations: int
    eigendecomposition_count: int = 1

    def __post_init__(self):
        if not isinstance(self.model, LocalGroundedSemanticDiffusion):
            raise TypeError("model must be a LocalGroundedSemanticDiffusion")
        base = np.asarray(self.base_intrinsic_leakage_conductance, dtype=float)
        if base.shape != (len(self.model.nodes),):
            raise ValueError("base intrinsic leakage must align with model nodes")
        if not np.isfinite(base).all() or np.any(base < 0.0):
            raise ValueError("base intrinsic leakage must be finite and nonnegative")
        target = _positive_unit_interval(
            "target_attenuation",
            self.target_attenuation,
        )
        study = _finite_scalar(
            "study_added_leakage_conductance",
            self.study_added_leakage_conductance,
        )
        numerical = _finite_scalar(
            "numerical_minimum_added_leakage",
            self.numerical_minimum_added_leakage,
        )
        if study < 0.0 or numerical < 0.0:
            raise ValueError("added leakage values must be nonnegative")
        records = tuple(self.per_anchor)
        if not records or not all(
            isinstance(record, AnchorLeakageCalibrationResult)
            for record in records
        ):
            raise ValueError("per_anchor must contain anchor calibration results")
        anchors = tuple(record.anchor for record in records)
        if anchors != tuple(sorted(set(anchors), key=_stable_key)):
            raise ValueError("per-anchor results must use canonical unique anchors")
        if any(
            not math.isclose(
                record.screening_at_selected_leakage.attenuation_threshold,
                target,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            for record in records
        ):
            raise ValueError("all per-anchor thresholds must match the batch target")
        required = max(record.added_leakage_conductance for record in records)
        required_scale = max(abs(required), abs(study), np.finfo(float).tiny)
        if abs(required - study) > 64.0 * np.finfo(float).eps * required_scale:
            raise ValueError("study leakage must equal the maximum anchor requirement")
        expected_intrinsic = base + study
        intrinsic_scale = max(
            float(np.max(np.abs(expected_intrinsic), initial=0.0)),
            np.finfo(float).tiny,
        )
        if not np.allclose(
            self.model.intrinsic_leakage_conductance,
            expected_intrinsic,
            rtol=1e-12,
            atol=64.0 * np.finfo(float).eps * intrinsic_scale,
        ):
            raise ValueError(
                "study model intrinsic leakage must equal base plus added leakage"
            )
        if (
            isinstance(self.total_evaluations, (bool, np.bool_))
            or int(self.total_evaluations) != self.total_evaluations
            or self.total_evaluations <= 0
        ):
            raise ValueError("total_evaluations must be a positive integer")
        total = int(self.total_evaluations)
        if total != sum(record.iterations for record in records):
            raise ValueError("total evaluations must equal per-anchor iterations")
        if self.eigendecomposition_count != 1:
            raise ValueError("per-anchor calibration must use one eigendecomposition")
        object.__setattr__(self, "base_intrinsic_leakage_conductance", _readonly(base))
        object.__setattr__(self, "study_added_leakage_conductance", study)
        object.__setattr__(self, "target_attenuation", target)
        object.__setattr__(self, "numerical_minimum_added_leakage", numerical)
        object.__setattr__(self, "per_anchor", records)
        object.__setattr__(self, "total_evaluations", total)
        object.__setattr__(self, "eigendecomposition_count", 1)

    @property
    def by_anchor(self):
        return {record.anchor: record for record in self.per_anchor}

    @property
    def screening_provenance_semantics(self):
        return "per_anchor_selected_and_shared_study_added_leakage"


def _positive_conductance_hop_distances(conductance, source_row):
    """Return source-relative hops inside one realized conductance component."""

    conductance = np.asarray(conductance, dtype=float)
    if conductance.ndim != 2 or conductance.shape[0] != conductance.shape[1]:
        raise ValueError("conductance must be square")
    size = conductance.shape[0]
    if not 0 <= source_row < size:
        raise ValueError("source_row must index conductance")
    distances = np.full(size, -1, dtype=np.int64)
    distances[source_row] = 0
    pending = deque((source_row,))
    while pending:
        row = pending.popleft()
        for neighbor in np.flatnonzero(conductance[row] > 0.0):
            neighbor = int(neighbor)
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = distances[row] + 1
            pending.append(neighbor)
    return distances


def _tail_envelope_crossing(distances, attenuation, threshold):
    """Bracket the first threshold crossing of a monotone radial tail envelope."""

    distances = np.asarray(distances, dtype=float)
    attenuation = np.asarray(attenuation, dtype=float)
    if distances.shape != attenuation.shape or distances.ndim != 1:
        raise ValueError("distances and attenuation must be aligned vectors")
    if (
        not np.isfinite(distances).all()
        or not np.isfinite(attenuation).all()
        or np.any(distances < 0.0)
        or np.any(attenuation < 0.0)
        or np.any(attenuation > 1.0 + 1e-8)
    ):
        raise ValueError("screening profile must be finite and in [0, 1]")
    threshold = _positive_unit_interval("attenuation_threshold", threshold)
    positive_radii = np.unique(distances[distances > 0.0])
    if not len(positive_radii):
        raise ValueError("screening profile needs a reachable non-source node")
    maximum = float(positive_radii[-1])
    previous = 0.0
    for radius in positive_radii:
        tail = float(np.max(attenuation[distances >= radius]))
        if tail <= threshold:
            return previous, float(radius), False, maximum
        previous = float(radius)
    return maximum, None, True, maximum


def _anchor_screening_provenance(
    local,
    source_nodes,
    shell_rows,
    source_distances,
    *,
    threshold,
):
    """Summarize shell tightness and robust per-anchor screening radii."""

    index = {node: row for row, node in enumerate(local.nodes)}
    records = []
    for source_node, distances in zip(source_nodes, source_distances):
        source_row = index[source_node]
        source = np.zeros(len(local.nodes), dtype=float)
        source[source_row] = 1.0
        response = local.model.equilibrium_response(source)
        scale = max(float(np.max(np.abs(response), initial=0.0)), 1.0)
        if (
            not np.isfinite(response).all()
            or response[source_row] <= 0.0
            or np.min(response) < -1e-10 * scale
        ):
            raise np.linalg.LinAlgError(
                "per-anchor Green response is not nonnegative"
            )
        attenuation = np.maximum(response, 0.0) / float(response[source_row])
        if float(np.max(attenuation)) > 1.0 + 1e-8:
            raise np.linalg.LinAlgError(
                "per-anchor Green attenuation left [0, 1]"
            )
        attenuation = np.clip(attenuation, 0.0, 1.0)
        reachable_shell = tuple(
            row
            for row in shell_rows
            if row != source_row and distances[row] > 0
        )
        if not reachable_shell:
            raise ValueError(
                "calibration shell must include a reachable non-source "
                f"node for anchor {source_node!r}"
            )
        shell_attenuation = float(
            np.max(attenuation[list(reachable_shell)])
        )
        reachable = distances >= 0
        lower, upper, censored, maximum = _tail_envelope_crossing(
            distances[reachable],
            attenuation[reachable],
            threshold,
        )
        records.append(
            AnchorScreeningProvenance(
                anchor=source_node,
                shell_attenuation=shell_attenuation,
                attenuation_threshold=threshold,
                radius_lower=lower,
                radius_upper=upper,
                right_censored=censored,
                maximum_observed_radius=maximum,
                distance_metric="realized_positive_conductance_hops",
            )
        )
    return tuple(records)


def _attenuation_from_eigendecomposition(
    eigenvalues,
    eigenvectors,
    *,
    leakage,
    source_rows,
    shell_rows,
):
    denominators = eigenvalues + leakage
    if np.any(denominators <= 0.0) or not np.isfinite(denominators).all():
        return math.inf
    maximum = None
    for source_row in source_rows:
        response = eigenvectors @ (eigenvectors[source_row, :] / denominators)
        source_value = float(response[source_row])
        if source_value <= 0.0 or not np.isfinite(response).all():
            return math.inf
        scale = max(float(np.max(np.abs(response), initial=0.0)), 1.0)
        if np.min(response) < -1e-10 * scale:
            raise np.linalg.LinAlgError("spectral Green response is negative")
        for shell_row in shell_rows:
            if shell_row == source_row:
                continue
            ratio = max(float(response[shell_row]), 0.0) / source_value
            maximum = ratio if maximum is None else max(maximum, ratio)
    if maximum is None:
        raise ValueError("attenuation shell must include a non-source node")
    return float(maximum)


def calibrate_uniform_leakage(
    domain,
    *,
    anchors=None,
    shell_nodes,
    target_attenuation=math.exp(-1.0),
    intrinsic_leakage_conductance=0.0,
    node_embeddings=None,
    length_scale=None,
    conductance_floor=0.0,
    bath_temperature=0.0,
    minimum_reciprocal_condition=None,
    maximum_leakage_conductance=None,
    relative_tolerance=1e-8,
    maximum_iterations=80,
):
    """Choose the smallest added uniform leakage meeting a shell attenuation.
    The calibration shell is required explicitly so callers must name an
    intended interior e-fold shell instead of silently defaulting to the hard
    boundary.  Interiority is a caller-owned design choice, not inferred here.


    Calibration uses G(i,s)/G(s,s), the killed-walk hitting probability.  It
    deliberately does not use correlation-normalized Green entries, whose
    dependence on leakage need not be monotone.
    """

    target = _positive_unit_interval("target_attenuation", target_attenuation)
    tolerance = _positive_finite("relative_tolerance", relative_tolerance)
    maximum_iterations = _positive_integer("maximum_iterations", maximum_iterations)
    source_nodes = _selected_nodes(
        domain.nodes,
        domain.anchors if anchors is None else anchors,
        name="anchors",
    )
    shell = _selected_nodes(
        domain.nodes,
        shell_nodes,
        name="shell_nodes",
    )
    index = {node: row for row, node in enumerate(domain.nodes)}
    source_rows = tuple(index[node] for node in source_nodes)
    shell_rows = tuple(index[node] for node in shell)

    conductance, laplacian, intrinsic, cut = _assemble_local_components(
        domain,
        intrinsic_leakage_conductance=intrinsic_leakage_conductance,
        node_embeddings=node_embeddings,
        length_scale=length_scale,
        conductance_floor=conductance_floor,
    )
    source_distances = tuple(
        _positive_conductance_hop_distances(conductance, source_row)
        for source_row in source_rows
    )
    for source_node, source_row, distances in zip(
        source_nodes,
        source_rows,
        source_distances,
    ):
        if not any(
            row != source_row and distances[row] > 0
            for row in shell_rows
        ):
            raise ValueError(
                "calibration shell must include a reachable non-source "
                f"node for anchor {source_node!r}"
            )
    base_precision = laplacian + np.diag(intrinsic + cut)
    eigenvalues, eigenvectors = np.linalg.eigh(base_precision)
    spectral_scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(eigenvalues[0]) < -1e-12 * spectral_scale:
        raise np.linalg.LinAlgError("local base precision is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])

    required_rcond = (
        _DEFAULT_MINIMUM_RECIPROCAL_CONDITION
        if minimum_reciprocal_condition is None
        else _positive_unit_interval(
            "minimum_reciprocal_condition",
            minimum_reciprocal_condition,
        )
    )
    if required_rcond == 1.0 and maximum > minimum:
        raise ValueError(
            "finite uniform leakage cannot make a nonscalar spectrum exact"
        )
    if required_rcond == 1.0:
        numerical_minimum = 0.0
    else:
        numerical_minimum = max(
            0.0,
            (required_rcond * maximum - minimum) / (1.0 - required_rcond),
        )
        if numerical_minimum > 0.0:
            numerical_minimum += (
                64.0 * np.finfo(float).eps * max(maximum, np.finfo(float).tiny)
            )
    representable = np.nextafter(1.0 / np.finfo(float).max, math.inf)
    numerical_minimum = max(numerical_minimum, representable - minimum, 0.0)
    if numerical_minimum > 0.0:
        numerical_minimum = float(np.nextafter(numerical_minimum, math.inf))

    if maximum_leakage_conductance is None:
        maximum_allowed = math.inf
    else:
        maximum_allowed = _finite_scalar(
            "maximum_leakage_conductance",
            maximum_leakage_conductance,
        )
        if maximum_allowed < 0.0:
            raise ValueError("maximum_leakage_conductance must be nonnegative")
    if numerical_minimum > maximum_allowed:
        raise np.linalg.LinAlgError(
            "numerical conditioning requires leakage above the allowed maximum"
        )

    evaluations = 0

    def evaluate(leakage):
        nonlocal evaluations
        evaluations += 1
        return _attenuation_from_eigendecomposition(
            eigenvalues,
            eigenvectors,
            leakage=leakage,
            source_rows=source_rows,
            shell_rows=shell_rows,
        )

    lower = numerical_minimum
    lower_value = evaluate(lower)
    if lower_value <= target:
        selected = lower
        achieved = lower_value
    else:
        positive_edges = conductance[conductance > 0.0]
        typical_conductance = (
            float(np.median(positive_edges)) if len(positive_edges) else 1.0
        )
        radius = max(domain.cutoff_distance, 1)
        gamma = math.log(1.0 / target) / radius
        try:
            chain_seed = 2.0 * typical_conductance * (math.cosh(gamma) - 1.0)
        except OverflowError:
            chain_seed = math.inf
        upper = max(
            float(np.nextafter(lower, math.inf)) * 2.0,
            chain_seed,
            np.finfo(float).tiny,
        )
        upper = min(upper, maximum_allowed)
        upper_value = evaluate(upper)
        while upper_value > target and evaluations < maximum_iterations:
            if upper >= maximum_allowed or not math.isfinite(upper):
                break
            candidate = min(upper * 2.0, maximum_allowed)
            if candidate <= upper:
                break
            upper = candidate
            upper_value = evaluate(upper)
        if upper_value > target:
            if evaluations >= maximum_iterations:
                raise np.linalg.LinAlgError(
                    "leakage calibration exhausted evaluations before "
                    "bracketing"
                )
            raise np.linalg.LinAlgError(
                "no allowed uniform leakage meets the attenuation target"
            )

        while evaluations < maximum_iterations:
            scale = max(abs(upper), np.finfo(float).tiny)
            if upper - lower <= tolerance * scale:
                break
            if lower > 0.0 and upper / lower > 4.0:
                middle = math.sqrt(lower * upper)
            else:
                middle = lower + 0.5 * (upper - lower)
            middle_value = evaluate(middle)
            if middle_value <= target:
                upper = middle
                upper_value = middle_value
            else:
                lower = middle
        selected = upper
        scale = max(abs(upper), np.finfo(float).tiny)
        if upper - lower > tolerance * scale:
            raise np.linalg.LinAlgError(
                "leakage calibration did not converge within maximum_iterations"
            )
        achieved = upper_value

    final_intrinsic = intrinsic + selected
    local = _build_local_from_components(
        domain,
        conductance,
        laplacian,
        final_intrinsic,
        cut,
        semantic_length_scale=(
            None if node_embeddings is None else float(length_scale)
        ),
        conductance_floor=float(conductance_floor),
        bath_temperature=bath_temperature,
        minimum_reciprocal_condition=required_rcond,
    )
    anchor_screening = _anchor_screening_provenance(
        local,
        source_nodes,
        shell_rows,
        source_distances,
        threshold=target,
    )
    observed = max(
        item.shell_attenuation for item in anchor_screening
    )
    direct_observed = local.frontier_attenuation(source_nodes, shell)
    observed_scale = max(abs(observed), abs(direct_observed), 1.0)
    if abs(observed - direct_observed) > (
        64.0 * np.finfo(float).eps * observed_scale
    ):
        raise np.linalg.LinAlgError(
            "per-anchor screening provenance disagrees with the final model"
        )
    if observed > target * (1.0 + max(tolerance, 1e-10)):
        raise np.linalg.LinAlgError("final model missed the attenuation target")
    return LeakageCalibrationResult(
        model=local,
        leakage_conductance=selected,
        leakage_resistance=(math.inf if selected == 0.0 else 1.0 / selected),
        target_attenuation=target,
        achieved_attenuation=observed,
        source_nodes=source_nodes,
        shell_nodes=shell,
        anchor_screening=anchor_screening,
        iterations=evaluations,
        numerical_minimum_leakage=numerical_minimum,
    )


def _attenuation_from_selected_spectral_rows(
    eigenvalues,
    eigenvectors,
    *,
    leakage,
    source_row,
    shell_rows,
):
    """Evaluate one shell without materializing a full Green response."""

    denominators = eigenvalues + leakage
    if np.any(denominators <= 0.0) or not np.isfinite(denominators).all():
        return math.inf
    source_eigenvector = eigenvectors[source_row, :]
    weighted_source = source_eigenvector / denominators
    source_value = float(source_eigenvector @ weighted_source)
    shell_values = eigenvectors[np.asarray(shell_rows, dtype=np.int64), :] @ (
        weighted_source
    )
    if source_value <= 0.0 or not math.isfinite(source_value):
        return math.inf
    if not np.isfinite(shell_values).all():
        return math.inf
    scale = max(
        abs(source_value),
        float(np.max(np.abs(shell_values), initial=0.0)),
        1.0,
    )
    if float(np.min(shell_values, initial=0.0)) < -1e-10 * scale:
        raise np.linalg.LinAlgError("spectral Green shell response is negative")
    return float(np.max(np.maximum(shell_values, 0.0)) / source_value)


def _full_spectral_response(
    eigenvalues,
    eigenvectors,
    *,
    leakage,
    source_row,
):
    denominators = eigenvalues + leakage
    if np.any(denominators <= 0.0) or not np.isfinite(denominators).all():
        raise np.linalg.LinAlgError("spectral Green response is singular")
    response = eigenvectors @ (eigenvectors[source_row, :] / denominators)
    scale = max(float(np.max(np.abs(response), initial=0.0)), 1.0)
    if (
        not np.isfinite(response).all()
        or response[source_row] <= 0.0
        or float(np.min(response, initial=0.0)) < -1e-10 * scale
    ):
        raise np.linalg.LinAlgError("spectral Green response is not nonnegative")
    return response


def _screening_from_response(
    anchor,
    source_row,
    shell_rows,
    distances,
    response,
    *,
    threshold,
):
    source_value = float(response[source_row])
    attenuation = np.maximum(response, 0.0) / source_value
    if float(np.max(attenuation)) > 1.0 + 1e-8:
        raise np.linalg.LinAlgError("per-anchor Green attenuation left [0, 1]")
    attenuation = np.clip(attenuation, 0.0, 1.0)
    shell_attenuation = float(np.max(attenuation[list(shell_rows)]))
    reachable = distances >= 0
    lower, upper, censored, maximum = _tail_envelope_crossing(
        distances[reachable],
        attenuation[reachable],
        threshold,
    )
    return AnchorScreeningProvenance(
        anchor=anchor,
        shell_attenuation=shell_attenuation,
        attenuation_threshold=threshold,
        radius_lower=lower,
        radius_upper=upper,
        right_censored=censored,
        maximum_observed_radius=maximum,
        distance_metric="realized_positive_conductance_hops",
    )


def _minimum_uniform_leakage_for_spectrum(
    minimum,
    maximum,
    *,
    required_rcond,
):
    if required_rcond == 1.0 and maximum > minimum:
        raise ValueError(
            "finite uniform leakage cannot make a nonscalar spectrum exact"
        )
    if required_rcond == 1.0:
        numerical_minimum = 0.0
    else:
        numerical_minimum = max(
            0.0,
            (required_rcond * maximum - minimum) / (1.0 - required_rcond),
        )
        if numerical_minimum > 0.0:
            numerical_minimum += (
                64.0 * np.finfo(float).eps * max(maximum, np.finfo(float).tiny)
            )
    representable = np.nextafter(1.0 / np.finfo(float).max, math.inf)
    numerical_minimum = max(numerical_minimum, representable - minimum, 0.0)
    if numerical_minimum > 0.0:
        numerical_minimum = float(np.nextafter(numerical_minimum, math.inf))
    return float(numerical_minimum)


def _calibrate_one_anchor_from_spectrum(
    eigenvalues,
    eigenvectors,
    *,
    source_row,
    shell_rows,
    target,
    numerical_minimum,
    maximum_allowed,
    typical_conductance,
    bracket_seed_radius,
    tolerance,
    maximum_iterations,
):
    evaluations = 0

    def evaluate(leakage):
        nonlocal evaluations
        evaluations += 1
        return _attenuation_from_selected_spectral_rows(
            eigenvalues,
            eigenvectors,
            leakage=leakage,
            source_row=source_row,
            shell_rows=shell_rows,
        )

    lower = numerical_minimum
    lower_value = evaluate(lower)
    if lower_value <= target:
        certificate = LeakageCalibrationMinimalityCertificate(
            lower_added_leakage_conductance=lower,
            upper_added_leakage_conductance=lower,
            attenuation_at_lower=lower_value,
            attenuation_at_upper=lower_value,
            target_attenuation=target,
            relative_tolerance=tolerance,
            initial_lower_passed=True,
            bracket_seed_radius=bracket_seed_radius,
        )
        return lower, lower_value, evaluations, certificate

    gamma = math.log(1.0 / target) / bracket_seed_radius
    try:
        chain_seed = 2.0 * typical_conductance * (math.cosh(gamma) - 1.0)
    except OverflowError:
        chain_seed = math.inf
    upper = max(
        float(np.nextafter(lower, math.inf)) * 2.0,
        chain_seed,
        np.finfo(float).tiny,
    )
    upper = min(upper, maximum_allowed)
    upper_value = evaluate(upper)
    while upper_value > target and evaluations < maximum_iterations:
        if upper >= maximum_allowed or not math.isfinite(upper):
            break
        candidate = min(upper * 2.0, maximum_allowed)
        if candidate <= upper:
            break
        upper = candidate
        upper_value = evaluate(upper)
    if upper_value > target:
        if evaluations >= maximum_iterations:
            raise np.linalg.LinAlgError(
                "leakage calibration exhausted evaluations before bracketing"
            )
        raise np.linalg.LinAlgError(
            "no allowed uniform leakage meets the attenuation target"
        )

    while evaluations < maximum_iterations:
        scale = max(abs(upper), np.finfo(float).tiny)
        if upper - lower <= tolerance * scale:
            break
        if lower > 0.0 and upper / lower > 4.0:
            middle = math.sqrt(lower * upper)
        else:
            middle = lower + 0.5 * (upper - lower)
        middle_value = evaluate(middle)
        if middle_value <= target:
            upper = middle
            upper_value = middle_value
        else:
            lower = middle
            lower_value = middle_value
    scale = max(abs(upper), np.finfo(float).tiny)
    if upper - lower > tolerance * scale:
        raise np.linalg.LinAlgError(
            "leakage calibration did not converge within maximum_iterations"
        )
    certificate = LeakageCalibrationMinimalityCertificate(
        lower_added_leakage_conductance=lower,
        upper_added_leakage_conductance=upper,
        attenuation_at_lower=lower_value,
        attenuation_at_upper=upper_value,
        target_attenuation=target,
        relative_tolerance=tolerance,
        initial_lower_passed=False,
        bracket_seed_radius=bracket_seed_radius,
    )
    return upper, upper_value, evaluations, certificate


def calibrate_uniform_leakage_per_anchor(
    domain,
    *,
    anchors=None,
    shell_nodes_by_anchor,
    target_attenuation=math.exp(-1.0),
    intrinsic_leakage_conductance=0.0,
    node_embeddings=None,
    length_scale=None,
    conductance_floor=0.0,
    bath_temperature=0.0,
    minimum_reciprocal_condition=None,
    maximum_leakage_conductance=None,
    relative_tolerance=1e-8,
    maximum_iterations=80,
    bracket_seed_radius=None,
):
    """Calibrate independent anchor shells with one shared eigendecomposition.

    Each anchor is calibrated only against its own explicitly supplied shell.
    The base precision includes the caller's existing intrinsic leakage and the
    exact Dirichlet cut. The returned per-anchor values are *added* uniform
    leakage requirements; they do not replace or relabel intrinsic leakage.

    Bisection evaluates only the source and shell rows of the spectral Green
    response. Full response vectors are formed once per anchor at its selected
    leakage and at the maximum shared study leakage solely to record robust
    realized screening-radius provenance.

    ``bracket_seed_radius`` affects only the chain-model starting value used to
    bracket the monotone solve. It does not alter the supplied scientific shell
    or target. By default it preserves the historical behavior of using the
    selected domain's cutoff distance (with a minimum of one).
    """

    if not isinstance(domain, LocalDiffusionDomain):
        raise TypeError("domain must be a LocalDiffusionDomain")
    if not isinstance(shell_nodes_by_anchor, Mapping):
        raise TypeError("shell_nodes_by_anchor must be a mapping")
    target = _positive_unit_interval("target_attenuation", target_attenuation)
    tolerance = _positive_finite("relative_tolerance", relative_tolerance)
    maximum_iterations = _positive_integer(
        "maximum_iterations",
        maximum_iterations,
    )
    if bracket_seed_radius is None:
        seed_radius = max(domain.cutoff_distance, 1)
    else:
        seed_radius = _positive_integer(
            "bracket_seed_radius",
            bracket_seed_radius,
        )
    source_nodes = _selected_nodes(
        domain.nodes,
        domain.anchors if anchors is None else anchors,
        name="anchors",
    )
    try:
        supplied_anchors = set(shell_nodes_by_anchor)
    except TypeError as exc:
        raise ValueError("shell mapping anchors must be hashable") from exc
    expected_anchors = set(source_nodes)
    if supplied_anchors != expected_anchors:
        missing = expected_anchors.difference(supplied_anchors)
        extra = supplied_anchors.difference(expected_anchors)
        details = []
        if missing:
            details.append(
                "missing " + ", ".join(sorted((repr(node) for node in missing)))
            )
        if extra:
            details.append(
                "extra " + ", ".join(sorted((repr(node) for node in extra)))
            )
        raise ValueError(
            "shell_nodes_by_anchor keys must exactly match anchors ("
            + "; ".join(details)
            + ")"
        )

    index = {node: row for row, node in enumerate(domain.nodes)}
    source_rows = {node: index[node] for node in source_nodes}
    shells = {}
    shell_rows = {}
    for anchor in source_nodes:
        shell = _selected_nodes(
            domain.nodes,
            shell_nodes_by_anchor[anchor],
            name=f"shell_nodes_by_anchor[{anchor!r}]",
        )
        shells[anchor] = shell
        shell_rows[anchor] = tuple(index[node] for node in shell)

    conductance, laplacian, intrinsic, cut = _assemble_local_components(
        domain,
        intrinsic_leakage_conductance=intrinsic_leakage_conductance,
        node_embeddings=node_embeddings,
        length_scale=length_scale,
        conductance_floor=conductance_floor,
    )
    distances = {
        anchor: _positive_conductance_hop_distances(
            conductance,
            source_rows[anchor],
        )
        for anchor in source_nodes
    }
    for anchor in source_nodes:
        source_row = source_rows[anchor]
        invalid = tuple(
            domain.nodes[row]
            for row in shell_rows[anchor]
            if row == source_row or distances[anchor][row] <= 0
        )
        if invalid:
            labels = ", ".join(repr(node) for node in invalid)
            raise ValueError(
                "each anchor shell must contain only reachable non-source "
                f"nodes; anchor {anchor!r} has invalid shell nodes: {labels}"
            )

    base_precision = laplacian + np.diag(intrinsic + cut)
    eigenvalues, eigenvectors = np.linalg.eigh(base_precision)
    spectral_scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(eigenvalues[0]) < -1e-12 * spectral_scale:
        raise np.linalg.LinAlgError(
            "local base precision is not positive semidefinite"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    spectral_minimum = float(eigenvalues[0])
    spectral_maximum = float(eigenvalues[-1])
    required_rcond = (
        _DEFAULT_MINIMUM_RECIPROCAL_CONDITION
        if minimum_reciprocal_condition is None
        else _positive_unit_interval(
            "minimum_reciprocal_condition",
            minimum_reciprocal_condition,
        )
    )
    numerical_minimum = _minimum_uniform_leakage_for_spectrum(
        spectral_minimum,
        spectral_maximum,
        required_rcond=required_rcond,
    )

    if maximum_leakage_conductance is None:
        maximum_allowed = math.inf
    else:
        maximum_allowed = _finite_scalar(
            "maximum_leakage_conductance",
            maximum_leakage_conductance,
        )
        if maximum_allowed < 0.0:
            raise ValueError("maximum_leakage_conductance must be nonnegative")
    if numerical_minimum > maximum_allowed:
        raise np.linalg.LinAlgError(
            "numerical conditioning requires leakage above the allowed maximum"
        )

    positive_edges = conductance[conductance > 0.0]
    typical_conductance = (
        float(np.median(positive_edges)) if len(positive_edges) else 1.0
    )
    selected = {}
    selected_screening = {}
    minimality_certificates = {}
    total_evaluations = 0
    for anchor in source_nodes:
        added, _, evaluations, certificate = _calibrate_one_anchor_from_spectrum(
            eigenvalues,
            eigenvectors,
            source_row=source_rows[anchor],
            shell_rows=shell_rows[anchor],
            target=target,
            numerical_minimum=numerical_minimum,
            maximum_allowed=maximum_allowed,
            typical_conductance=typical_conductance,
            bracket_seed_radius=seed_radius,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        response = _full_spectral_response(
            eigenvalues,
            eigenvectors,
            leakage=added,
            source_row=source_rows[anchor],
        )
        provenance = _screening_from_response(
            anchor,
            source_rows[anchor],
            shell_rows[anchor],
            distances[anchor],
            response,
            threshold=target,
        )
        if provenance.shell_attenuation > target * (
            1.0 + max(tolerance, 1e-10)
        ):
            raise np.linalg.LinAlgError(
                f"final model missed the attenuation target for {anchor!r}"
            )
        selected[anchor] = (added, evaluations)
        selected_screening[anchor] = provenance
        minimality_certificates[anchor] = certificate
        total_evaluations += evaluations

    study_added = max(value[0] for value in selected.values())
    study_screening = {}
    for anchor in source_nodes:
        response = _full_spectral_response(
            eigenvalues,
            eigenvectors,
            leakage=study_added,
            source_row=source_rows[anchor],
        )
        provenance = _screening_from_response(
            anchor,
            source_rows[anchor],
            shell_rows[anchor],
            distances[anchor],
            response,
            threshold=target,
        )
        if provenance.shell_attenuation > target * (
            1.0 + max(tolerance, 1e-10)
        ):
            raise np.linalg.LinAlgError(
                f"shared study leakage missed the target for {anchor!r}"
            )
        study_screening[anchor] = provenance

    study_model = _build_local_from_components(
        domain,
        conductance,
        laplacian,
        intrinsic + study_added,
        cut,
        semantic_length_scale=(
            None if node_embeddings is None else float(length_scale)
        ),
        conductance_floor=float(conductance_floor),
        bath_temperature=bath_temperature,
        minimum_reciprocal_condition=required_rcond,
    )
    records = tuple(
        AnchorLeakageCalibrationResult(
            anchor=anchor,
            shell_nodes=shells[anchor],
            added_leakage_conductance=selected[anchor][0],
            achieved_attenuation=(
                selected_screening[anchor].shell_attenuation
            ),
            iterations=selected[anchor][1],
            numerical_minimum_added_leakage=numerical_minimum,
            screening_at_selected_leakage=selected_screening[anchor],
            screening_at_study_leakage=study_screening[anchor],
            minimality_certificate=minimality_certificates[anchor],
        )
        for anchor in source_nodes
    )
    return PerAnchorLeakageCalibrationResult(
        model=study_model,
        base_intrinsic_leakage_conductance=intrinsic,
        study_added_leakage_conductance=study_added,
        target_attenuation=target,
        numerical_minimum_added_leakage=numerical_minimum,
        per_anchor=records,
        total_evaluations=total_evaluations,
        eigendecomposition_count=1,
    )


@dataclass(frozen=True)
class NestedDomainDiagnostics:
    source_nodes: tuple
    protected_nodes: tuple
    maximum_protected_absolute_error: float
    maximum_protected_relative_error: float
    minimum_monotone_increment: float
    monotonic: bool
    inner_boundary_harmonic_max: float
    outer_boundary_harmonic_max: float

    def __post_init__(self):
        sources = _canonical_unique(self.source_nodes, name="source_nodes")
        protected = _canonical_unique(
            self.protected_nodes,
            name="protected_nodes",
        )
        maximum_absolute = _finite_scalar(
            "maximum_protected_absolute_error",
            self.maximum_protected_absolute_error,
        )
        maximum_relative = _finite_scalar(
            "maximum_protected_relative_error",
            self.maximum_protected_relative_error,
        )
        if maximum_absolute < 0.0 or maximum_relative < 0.0:
            raise ValueError("protected-domain errors must be nonnegative")
        minimum_increment = _finite_scalar(
            "minimum_monotone_increment",
            self.minimum_monotone_increment,
        )
        if not isinstance(self.monotonic, (bool, np.bool_)):
            raise ValueError("monotonic must be boolean")
        monotonic = bool(self.monotonic)
        inner_harmonic = _finite_scalar(
            "inner_boundary_harmonic_max",
            self.inner_boundary_harmonic_max,
        )
        outer_harmonic = _finite_scalar(
            "outer_boundary_harmonic_max",
            self.outer_boundary_harmonic_max,
        )
        if (
            not 0.0 <= inner_harmonic <= 1.0
            or not 0.0 <= outer_harmonic <= 1.0
        ):
            raise ValueError("boundary harmonic maxima must be in [0, 1]")
        object.__setattr__(self, "source_nodes", sources)
        object.__setattr__(self, "protected_nodes", protected)
        object.__setattr__(
            self,
            "maximum_protected_absolute_error",
            maximum_absolute,
        )
        object.__setattr__(
            self,
            "maximum_protected_relative_error",
            maximum_relative,
        )
        object.__setattr__(
            self,
            "minimum_monotone_increment",
            minimum_increment,
        )
        object.__setattr__(self, "monotonic", monotonic)
        object.__setattr__(
            self,
            "inner_boundary_harmonic_max",
            inner_harmonic,
        )
        object.__setattr__(
            self,
            "outer_boundary_harmonic_max",
            outer_harmonic,
        )


def compare_nested_domains(
    inner,
    outer,
    *,
    source_nodes=None,
    protected_nodes=None,
):
    """Compare raw responses on a common core of two exact nested domains."""

    if not isinstance(inner, LocalGroundedSemanticDiffusion) or not isinstance(
        outer, LocalGroundedSemanticDiffusion
    ):
        raise TypeError("inner and outer must be local grounded models")
    if not set(inner.nodes).issubset(outer.nodes):
        raise ValueError("inner nodes must be a subset of outer nodes")
    sources = _selected_nodes(
        inner.nodes,
        inner.domain.anchors if source_nodes is None else source_nodes,
        name="source_nodes",
    )
    protected = _selected_nodes(
        inner.nodes,
        inner.nodes if protected_nodes is None else protected_nodes,
        name="protected_nodes",
    )
    if not math.isclose(
        inner.bath_temperature,
        outer.bath_temperature,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("nested models must use the same bath temperature")
    if (
        inner.model.semantic_length_scale != outer.model.semantic_length_scale
        or inner.model.conductance_floor != outer.model.conductance_floor
    ):
        raise ValueError("nested models must use the same semantic conductance")
    inner_index = {node: row for row, node in enumerate(inner.nodes)}
    outer_index = {node: row for row, node in enumerate(outer.nodes)}
    for node in inner.nodes:
        left = inner_index[node]
        right = outer_index[node]
        inner_leakage = float(inner.intrinsic_leakage_conductance[left])
        outer_leakage = float(outer.intrinsic_leakage_conductance[right])
        leakage_scale = max(
            abs(inner_leakage),
            abs(outer_leakage),
            np.finfo(float).tiny,
        )
        if not math.isclose(
            inner_leakage,
            outer_leakage,
            rel_tol=1e-12,
            abs_tol=64.0 * np.finfo(float).eps * leakage_scale,
        ):
            raise ValueError("nested models must use the same intrinsic leakage")
        if set(inner.domain.neighbors[left]) != set(outer.domain.neighbors[right]):
            raise ValueError("nested models must use the same incident graph")

    common_outer = np.asarray(
        [outer_index[node] for node in inner.nodes],
        dtype=int,
    )
    outer_common_precision = outer.precision[
        np.ix_(common_outer, common_outer)
    ]
    precision_scale = max(
        float(np.max(np.abs(inner.precision), initial=0.0)),
        float(np.max(np.abs(outer_common_precision), initial=0.0)),
        np.finfo(float).tiny,
    )
    if not np.allclose(
        inner.precision,
        outer_common_precision,
        rtol=1e-12,
        atol=64.0 * np.finfo(float).eps * precision_scale,
    ):
        raise ValueError("nested models must use the same realized precision")
    protected_inner = np.asarray([inner_index[node] for node in protected])
    protected_outer = np.asarray([outer_index[node] for node in protected])
    maximum_absolute = 0.0
    maximum_relative = 0.0
    minimum_increment = math.inf
    response_scale = np.finfo(float).tiny
    for source_node in sources:
        inner_rhs = np.zeros(len(inner.nodes))
        outer_rhs = np.zeros(len(outer.nodes))
        inner_rhs[inner_index[source_node]] = 1.0
        outer_rhs[outer_index[source_node]] = 1.0
        inner_response = inner.model.equilibrium_response(inner_rhs)[protected_inner]
        outer_response = outer.model.equilibrium_response(outer_rhs)[protected_outer]
        difference = outer_response - inner_response
        maximum_absolute = max(
            maximum_absolute,
            float(np.max(np.abs(difference), initial=0.0)),
        )
        denominator = np.maximum(np.abs(outer_response), np.finfo(float).tiny)
        maximum_relative = max(
            maximum_relative,
            float(np.max(np.abs(difference) / denominator, initial=0.0)),
        )
        minimum_increment = min(
            minimum_increment,
            float(np.min(difference, initial=math.inf)),
        )
        response_scale = max(
            response_scale,
            float(np.max(np.abs(outer_response), initial=0.0)),
        )

    inner_harmonic = inner.boundary_harmonic_measure()[protected_inner]
    outer_harmonic = outer.boundary_harmonic_measure()[protected_outer]
    tolerance = 1e-10 * response_scale
    return NestedDomainDiagnostics(
        source_nodes=sources,
        protected_nodes=protected,
        maximum_protected_absolute_error=maximum_absolute,
        maximum_protected_relative_error=maximum_relative,
        minimum_monotone_increment=minimum_increment,
        monotonic=(minimum_increment >= -tolerance),
        inner_boundary_harmonic_max=float(
            np.max(inner_harmonic, initial=0.0)
        ),
        outer_boundary_harmonic_max=float(
            np.max(outer_harmonic, initial=0.0)
        ),
    )
