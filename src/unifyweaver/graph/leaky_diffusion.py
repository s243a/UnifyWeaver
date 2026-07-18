"""Grounded graph diffusion with semantic edge resistance.

This module is a dense, float64 reference for a general graph primitive.  It
keeps topology and semantics in distinct roles:

* graph edges define which direct electrical/thermal paths exist;
* embedding distance modulates the conductance of those existing edges;
* shunt conductance to a common ground makes the Laplacian precision positive
  definite and gives long-range decay a physical interpretation.

For conductance matrix ``W`` and combinatorial Laplacian ``L = D - W``, the
grounded precision and equilibrium Green kernel are

    J = L + diag(alpha),        G = J^-1.

``alpha_i`` is the leakage conductance from node ``i`` to ground.  If
``J = U.T @ U``, then ``U`` is already an inverse-covariance square root for
``G``; no explicit inverse is required for equilibrium solves or whitening.

The implementation deliberately does not add semantic k-nearest-neighbour
edges.  That is a different graph model, not merely a different resistance.
It also keeps physical leakage separate from any future floating-point jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


__all__ = [
    "GroundedSemanticDiffusion",
    "build_grounded_semantic_diffusion",
    "combinatorial_laplacian",
    "semantic_conductance_matrix",
]


_SYMMETRY_TOLERANCE = 1e-12
_FLOAT64_EPSILON = np.finfo(float).eps
_FLOAT64_MAXIMUM = np.finfo(float).max
_DEFAULT_MINIMUM_RECIPROCAL_CONDITION = math.sqrt(_FLOAT64_EPSILON)


def _symmetric_part(value):
    """Return a symmetry cleanup without overflowing on representable values."""

    value = np.asarray(value, dtype=float)
    output = 0.5 * value + 0.5 * value.T
    if not np.isfinite(output).all():
        raise np.linalg.LinAlgError("symmetric matrix result is not finite")
    return output


def _normalize_psd_kernel(value):
    """Normalize a symmetric positive-definite matrix to unit diagonal."""

    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not len(value):
        raise ValueError("kernel must be a non-empty square matrix")
    if not np.isfinite(value).all():
        raise ValueError("kernel must be finite")
    if not np.allclose(value, value.T, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise ValueError("kernel must be symmetric")
    value = _symmetric_part(value)
    diagonal = np.diag(value)
    if np.any(diagonal <= 0.0):
        raise ValueError("a normalizable kernel must have positive diagonal")
    scale = np.sqrt(diagonal)
    output = value / scale[:, None] / scale[None, :]
    output = _symmetric_part(output)
    np.fill_diagonal(output, 1.0)
    minimum = float(np.min(np.linalg.eigvalsh(output)))
    if minimum < -1e-10:
        raise ValueError(f"kernel is not positive semidefinite: {minimum:.3e}")
    return output


def _readonly(value):
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output


def _nodes(value):
    nodes = tuple(value)
    if not nodes:
        raise ValueError("nodes must be non-empty")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    return nodes


def _positive_finite(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _nonnegative_finite(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return value


def _unit_interval(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1)") from exc
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return value


def _positive_unit_interval(name, value):
    value = _positive_finite(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return value


def _embedding_matrix(nodes, node_embeddings):
    rows = []
    width = None
    for node in nodes:
        try:
            row = np.asarray(node_embeddings[node], dtype=float)
        except KeyError as exc:
            raise ValueError(f"node absent from embeddings: {exc.args[0]!r}") from exc
        if row.ndim != 1 or not len(row):
            raise ValueError("every node embedding must be a non-empty vector")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("every node embedding must have the same width")
        if not np.isfinite(row).all():
            raise ValueError("node embeddings must be finite")
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _stable_radial_factor(left, right, length_scale):
    """Evaluate an RBF factor without overflow in differences or squares."""

    cutoff = math.sqrt(-2.0 * math.log(np.nextafter(0.0, 1.0)))
    log_cutoff = math.log(cutoff)
    radius = 0.0
    for left_value, right_value in zip(left, right):
        left_value = float(left_value)
        right_value = float(right_value)
        if left_value == right_value:
            continue
        if math.copysign(1.0, left_value) == math.copysign(1.0, right_value):
            difference = abs(left_value - right_value)
            log_coordinate = math.log(difference) - math.log(length_scale)
        else:
            scale = max(abs(left_value), abs(right_value))
            normalized = abs(left_value / scale - right_value / scale)
            log_coordinate = (
                math.log(normalized)
                + math.log(scale)
                - math.log(length_scale)
            )
        if log_coordinate >= log_cutoff:
            return 0.0
        coordinate = math.exp(log_coordinate)
        radius = math.hypot(radius, coordinate)
        if radius >= cutoff:
            return 0.0
    return math.exp(-0.5 * radius * radius)


def semantic_conductance_matrix(
    nodes,
    neighbors,
    node_embeddings=None,
    *,
    length_scale=None,
    conductance_floor=0.0,
):
    """Return an undirected conductance matrix on one fixed node universe.

    Each retained topological edge has unit base conductance.  With embeddings,

    ``w_ij = floor + (1-floor) exp(-||z_i-z_j||^2 / (2 ell^2))``.

    The floor applies only to existing edges; it never creates semantic
    shortcuts between graph non-neighbours.  Neighbor references outside the
    fixed ``nodes`` universe are ignored, matching an induced-subgraph view.
    Directional neighbor declarations are combined by undirected union.
    With a zero floor, a sufficiently distant retained edge may underflow to
    zero conductance; use a positive floor when topology must be preserved.
    """

    nodes = _nodes(nodes)
    floor = _unit_interval("conductance_floor", conductance_floor)
    if node_embeddings is None:
        if length_scale is not None:
            raise ValueError("length_scale requires node_embeddings")
        if floor != 0.0:
            raise ValueError("conductance_floor requires node_embeddings")
        embeddings = None
    else:
        length_scale = _positive_finite("length_scale", length_scale)
        embeddings = _embedding_matrix(nodes, node_embeddings)

    index = {node: row for row, node in enumerate(nodes)}
    edges = set()
    for left in nodes:
        i = index[left]
        for right in neighbors.get(left, ()):
            j = index.get(right)
            if j is not None and i != j:
                edges.add((min(i, j), max(i, j)))

    conductance = np.zeros((len(nodes), len(nodes)), dtype=float)
    for i, j in edges:
        value = 1.0
        if embeddings is not None:
            radial = _stable_radial_factor(
                embeddings[i], embeddings[j], length_scale
            )
            value = floor + (1.0 - floor) * radial
        conductance[i, j] = value
        conductance[j, i] = value
    if not np.isfinite(conductance).all():
        raise ValueError("semantic conductance calculation was not finite")
    return nodes, conductance


def combinatorial_laplacian(conductance):
    """Return ``diag(W 1) - W`` after validating resistor conductances."""

    value = np.asarray(conductance, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not len(value):
        raise ValueError("conductance must be a non-empty square matrix")
    if not np.isfinite(value).all():
        raise ValueError("conductance must be finite")
    if np.any(value < 0.0):
        raise ValueError("conductance must be nonnegative")
    if not np.allclose(value, value.T, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise ValueError("conductance must be symmetric")
    if not np.allclose(np.diag(value), 0.0, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise ValueError("conductance diagonal must be zero")
    value = _symmetric_part(value)
    laplacian = np.diag(np.sum(value, axis=1)) - value
    laplacian = _symmetric_part(laplacian)
    scale = max(float(np.max(np.diag(laplacian), initial=0.0)), 1.0)
    if float(np.min(np.linalg.eigvalsh(laplacian))) < -1e-12 * scale:
        raise ValueError("conductance produced a non-PSD graph Laplacian")
    return laplacian


def _leakage_vector(nodes, leakage_conductance):
    if isinstance(leakage_conductance, Mapping):
        unknown = set(leakage_conductance).difference(nodes)
        if unknown:
            labels = ", ".join(sorted(repr(node) for node in unknown))
            raise ValueError(f"leakage mapping contains unknown nodes: {labels}")
        values = np.asarray(
            [leakage_conductance.get(node, 0.0) for node in nodes], dtype=float
        )
    else:
        values = np.asarray(leakage_conductance, dtype=float)
        if values.ndim == 0:
            values = np.full(len(nodes), float(values), dtype=float)
    if values.shape != (len(nodes),):
        raise ValueError("leakage_conductance must be scalar, node mapping, or node vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("leakage conductance must be finite and nonnegative")
    if not np.any(values > 0.0):
        raise ValueError("at least one node must have positive leakage conductance")
    return values


def _rhs(name, value, size):
    output = np.asarray(value, dtype=float)
    if output.ndim not in (1, 2) or output.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},) or ({size}, k)")
    if not np.isfinite(output).all():
        raise ValueError(f"{name} must be finite")
    return output


def _positive_weight_components(conductance):
    """Return connected components of the realized positive-weight graph."""

    unseen = set(range(len(conductance)))
    components = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            row = stack.pop()
            adjacent = set(np.flatnonzero(conductance[row] > 0.0)).intersection(
                unseen
            )
            unseen.difference_update(adjacent)
            stack.extend(adjacent)
            component.extend(adjacent)
        components.append(np.asarray(sorted(component), dtype=int))
    return tuple(components)


@dataclass(frozen=True)
class GroundedSemanticDiffusion:
    """A fixed grounded electrical/thermal graph and its precision root."""

    nodes: tuple
    conductance: np.ndarray
    laplacian: np.ndarray
    leakage_conductance: np.ndarray
    precision: np.ndarray
    precision_root: np.ndarray
    minimum_precision_eigenvalue: float
    maximum_precision_eigenvalue: float
    condition_number: float
    reciprocal_condition_number: float
    minimum_reciprocal_condition: float
    semantic_length_scale: float | None
    conductance_floor: float

    def __post_init__(self):
        nodes = _nodes(self.nodes)
        object.__setattr__(self, "nodes", nodes)
        size = len(nodes)
        arrays = {
            "conductance": np.asarray(self.conductance, dtype=float),
            "laplacian": np.asarray(self.laplacian, dtype=float),
            "leakage_conductance": np.asarray(
                self.leakage_conductance, dtype=float
            ),
            "precision": np.asarray(self.precision, dtype=float),
            "precision_root": np.asarray(self.precision_root, dtype=float),
        }
        for name in ("conductance", "laplacian", "precision", "precision_root"):
            if arrays[name].shape != (size, size):
                raise ValueError(f"{name} must have shape ({size}, {size})")
        if arrays["leakage_conductance"].shape != (size,):
            raise ValueError(f"leakage_conductance must have shape ({size},)")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("model arrays must be finite")
        if np.any(arrays["conductance"] < 0.0):
            raise ValueError("conductance must be nonnegative")
        if np.any(arrays["leakage_conductance"] < 0.0):
            raise ValueError("leakage conductance must be nonnegative")
        for name in ("conductance", "laplacian", "precision"):
            if not np.allclose(
                arrays[name],
                arrays[name].T,
                atol=_SYMMETRY_TOLERANCE,
                rtol=0.0,
            ):
                raise ValueError(f"{name} must be symmetric")
        expected_laplacian = (
            np.diag(np.sum(arrays["conductance"], axis=1))
            - arrays["conductance"]
        )
        if not np.allclose(
            arrays["laplacian"], expected_laplacian, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("laplacian does not match conductance")
        expected_precision = arrays["laplacian"] + np.diag(
            arrays["leakage_conductance"]
        )
        if not np.allclose(
            arrays["precision"], expected_precision, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("precision does not match laplacian plus leakage")
        reconstructed = arrays["precision_root"].T @ arrays["precision_root"]
        if not np.isfinite(reconstructed).all() or not np.allclose(
            reconstructed, arrays["precision"], rtol=1e-11, atol=1e-12
        ):
            raise ValueError("precision_root does not factor precision")
        eigenvalues = np.linalg.eigvalsh(arrays["precision"])
        observed_minimum = _positive_finite(
            "minimum_precision_eigenvalue", self.minimum_precision_eigenvalue
        )
        observed_maximum = _positive_finite(
            "maximum_precision_eigenvalue", self.maximum_precision_eigenvalue
        )
        observed_condition = _positive_finite(
            "condition_number", self.condition_number
        )
        observed_reciprocal = _positive_unit_interval(
            "reciprocal_condition_number", self.reciprocal_condition_number
        )
        required_reciprocal = _positive_unit_interval(
            "minimum_reciprocal_condition", self.minimum_reciprocal_condition
        )
        expected_minimum = float(eigenvalues[0])
        expected_maximum = float(eigenvalues[-1])
        if not math.isfinite(expected_minimum) or expected_minimum <= 0.0:
            raise ValueError("precision must be positive definite")
        if not math.isfinite(expected_maximum):
            raise ValueError("precision spectrum must be finite")
        expected_condition = expected_maximum / expected_minimum
        expected_reciprocal = expected_minimum / expected_maximum
        diagnostics = (
            (observed_minimum, expected_minimum),
            (observed_maximum, expected_maximum),
            (observed_condition, expected_condition),
            (observed_reciprocal, expected_reciprocal),
        )
        if any(
            not math.isclose(left, right, rel_tol=1e-10, abs_tol=0.0)
            for left, right in diagnostics
        ):
            raise ValueError("spectral diagnostics do not match precision")
        if observed_reciprocal < required_reciprocal:
            raise ValueError("reciprocal condition violates the recorded contract")
        semantic_length_scale = self.semantic_length_scale
        if semantic_length_scale is not None:
            semantic_length_scale = _positive_finite(
                "semantic_length_scale", semantic_length_scale
            )
        object.__setattr__(self, "semantic_length_scale", semantic_length_scale)
        object.__setattr__(
            self,
            "conductance_floor",
            _unit_interval("conductance_floor", self.conductance_floor),
        )
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly(value))

    def equilibrium_response(self, source):
        """Solve ``J u = source`` through the stored triangular root."""

        source = _rhs("source", source, len(self.nodes))
        intermediate = np.linalg.solve(self.precision_root.T, source)
        response = np.linalg.solve(self.precision_root, intermediate)
        if not np.isfinite(response).all():
            raise np.linalg.LinAlgError("equilibrium response is not finite")
        return response

    def green_kernel(self, *, normalize=False):
        """Materialize ``J^-1`` for reference, optionally as a correlation."""

        value = self.equilibrium_response(np.eye(len(self.nodes)))
        value = _symmetric_part(value)
        return _normalize_psd_kernel(value) if normalize else value

    def correlation_precision_root(self):
        """Return ``U_c`` with ``C^-1 = U_c.T U_c`` for normalized ``G``.

        If ``D = diag(G)``, then ``C = D^-1/2 G D^-1/2`` and
        ``C^-1 = D^1/2 J D^1/2``.  Consequently the desired root is obtained
        by scaling columns of the already available grounded precision root.
        """

        diagonal = np.diag(self.green_kernel())
        if np.any(diagonal <= 0.0):
            raise np.linalg.LinAlgError("Green kernel must have positive diagonal")
        root = np.asarray(self.precision_root) * np.sqrt(diagonal)[None, :]
        if not np.isfinite(root).all():
            raise np.linalg.LinAlgError("correlation precision root is not finite")
        return root

    def heat_kernel(self, time, *, normalize=False):
        """Return the grounded impulse propagator ``exp(-time J)``."""

        time = _nonnegative_finite("time", time)
        value = np.zeros_like(self.precision)
        for component in _positive_weight_components(self.conductance):
            block = self.precision[np.ix_(component, component)]
            eigenvalues, eigenvectors = np.linalg.eigh(block)
            if normalize:
                eigenvalues = np.maximum(eigenvalues - eigenvalues[0], 0.0)
            with np.errstate(over="ignore", under="ignore"):
                factors = np.exp(-time * eigenvalues)
            heat = (eigenvectors * factors) @ eigenvectors.T
            value[np.ix_(component, component)] = _symmetric_part(heat)
        return _normalize_psd_kernel(value) if normalize else value

    def impulse_response(self, source, *, time):
        """Evolve an impulse with unit heat capacity (or unit capacitance)."""

        source = _rhs("source", source, len(self.nodes))
        return self.heat_kernel(time) @ source

    def step_response(self, source, *, time):
        """Response after a constant source is switched on at time zero."""

        source = _rhs("source", source, len(self.nodes))
        time = _nonnegative_finite("time", time)
        eigenvalues, eigenvectors = np.linalg.eigh(self.precision)
        with np.errstate(over="ignore", under="ignore"):
            scaled = time * eigenvalues
        factors = np.empty_like(eigenvalues)
        finite = np.isfinite(scaled)
        finite_scaled = scaled[finite]
        phi = np.ones_like(finite_scaled)
        nonzero = finite_scaled != 0.0
        phi[nonzero] = (
            -np.expm1(-finite_scaled[nonzero]) / finite_scaled[nonzero]
        )
        factors[finite] = time * phi
        factors[~finite] = 1.0 / eigenvalues[~finite]
        projected = eigenvectors.T @ source
        if projected.ndim == 1:
            response = eigenvectors @ (factors * projected)
        else:
            response = eigenvectors @ (factors[:, None] * projected)
        if not np.isfinite(response).all():
            raise np.linalg.LinAlgError("step response is not finite")
        return response

    def resistance_distance(self, *, squared=False):
        """Pairwise grounded effective-resistance distance induced by ``G``."""

        inverse_transpose = np.linalg.solve(
            self.precision_root.T, np.eye(len(self.nodes))
        )
        distance_squared = np.zeros_like(self.precision)
        for left in range(len(self.nodes)):
            for right in range(left):
                difference = inverse_transpose[:, left] - inverse_transpose[:, right]
                norm = float(np.linalg.norm(difference))
                value = norm * norm
                if not math.isfinite(value):
                    raise np.linalg.LinAlgError(
                        "effective resistance is not representable in float64"
                    )
                distance_squared[left, right] = value
                distance_squared[right, left] = value
        return distance_squared if squared else np.sqrt(distance_squared)


def build_grounded_semantic_diffusion(
    nodes,
    neighbors,
    *,
    leakage_conductance,
    node_embeddings=None,
    length_scale=None,
    conductance_floor=0.0,
    minimum_reciprocal_condition=_DEFAULT_MINIMUM_RECIPROCAL_CONDITION,
):
    """Build a grounded diffusion model without a covariance inversion.

    Leakage may be uniform, a node-aligned vector, or a sparse node mapping.
    Zero leakage at some nodes is allowed only when the resulting grounded
    precision is positive definite (every connected component must reach
    ground).  Failure is explicit; no numerical diagonal loading is hidden.
    The default float64 contract also requires reciprocal spectral condition
    at least sqrt(machine epsilon); a caller-supplied weaker threshold is
    explicit and recorded on the returned model.
    """

    nodes, conductance = semantic_conductance_matrix(
        nodes,
        neighbors,
        node_embeddings,
        length_scale=length_scale,
        conductance_floor=conductance_floor,
    )
    laplacian = combinatorial_laplacian(conductance)
    leakage = _leakage_vector(nodes, leakage_conductance)
    minimum_reciprocal_condition = _positive_unit_interval(
        "minimum_reciprocal_condition", minimum_reciprocal_condition
    )
    precision = _symmetric_part(laplacian + np.diag(leakage))
    components = _positive_weight_components(conductance)
    for component in components:
        if not np.any(leakage[component] > 0.0):
            raise np.linalg.LinAlgError(
                "grounded precision is not positive definite; every graph component "
                "must have a path to positive leakage"
            )
        block_scale = float(
            np.max(np.abs(precision[np.ix_(component, component)]))
        )
        if block_scale < 1.0 / _FLOAT64_MAXIMUM:
            raise np.linalg.LinAlgError(
                "grounded precision has an unrepresentable equilibrium scale"
            )
    eigenvalues = np.linalg.eigvalsh(precision)
    minimum = float(np.min(eigenvalues))
    maximum = float(np.max(eigenvalues))
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum <= 0.0
    ):
        raise np.linalg.LinAlgError(
            "grounded precision has no positive numerical spectral floor"
        )
    if minimum < 1.0 / _FLOAT64_MAXIMUM:
        raise np.linalg.LinAlgError(
            "grounded precision has an unrepresentable equilibrium scale"
        )
    reciprocal_condition = minimum / maximum
    if reciprocal_condition < minimum_reciprocal_condition:
        raise np.linalg.LinAlgError(
            "grounded precision is too ill-conditioned for the requested "
            "float64 contract: reciprocal condition "
            f"{reciprocal_condition:.3e} < {minimum_reciprocal_condition:.3e}"
        )
    try:
        lower_root = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "grounded precision is not positive definite; every graph component "
            "must have a path to positive leakage"
        ) from exc
    return GroundedSemanticDiffusion(
        nodes=nodes,
        conductance=conductance,
        laplacian=laplacian,
        leakage_conductance=leakage,
        precision=precision,
        precision_root=lower_root.T,
        minimum_precision_eigenvalue=minimum,
        maximum_precision_eigenvalue=maximum,
        condition_number=maximum / minimum,
        reciprocal_condition_number=reciprocal_condition,
        minimum_reciprocal_condition=minimum_reciprocal_condition,
        semantic_length_scale=(
            None if node_embeddings is None else float(length_scale)
        ),
        conductance_floor=float(conductance_floor),
    )
