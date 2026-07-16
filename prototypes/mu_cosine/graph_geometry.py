"""PSD-safe item geometries for conditional residual covariance experiments.

The public constructors return float64, symmetric, unit-diagonal Gram matrices.
Exact spectral kernels are small-graph references.  ``walk_feature_kernel`` is
the scalable family: its explicit feature map makes positive semidefiniteness
structural rather than a post-hoc eigenvalue repair.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy import sparse


PSD_TOLERANCE = 1e-10


def _stable_key(value):
    return type(value).__name__, repr(value)


def _positive_finite(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _kernel(name, value, *, unit_diagonal=True):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.allclose(value, value.T, atol=PSD_TOLERANCE, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    value = 0.5 * (value + value.T)
    if unit_diagonal and not np.allclose(
        np.diag(value), 1.0, atol=PSD_TOLERANCE, rtol=0.0
    ):
        raise ValueError(f"{name} must have unit diagonal")
    minimum = float(np.min(np.linalg.eigvalsh(value))) if len(value) else 0.0
    if minimum < -PSD_TOLERANCE:
        raise ValueError(f"{name} is not positive semidefinite: min eigenvalue {minimum:.3e}")
    return value


def normalize_psd_kernel(value):
    """Normalize a PSD matrix to unit diagonal without repairing indefiniteness."""
    value = _kernel("value", value, unit_diagonal=False)
    diagonal = np.diag(value)
    if np.any(diagonal <= 0.0):
        raise ValueError("a normalizable kernel must have a strictly positive diagonal")
    scale = np.sqrt(diagonal)
    normalized = value / scale[:, None] / scale[None, :]
    normalized = 0.5 * (normalized + normalized.T)
    np.fill_diagonal(normalized, 1.0)
    return _kernel("normalized kernel", normalized)


def _fixed_node_graph(nodes, neighbors):
    nodes = tuple(nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    index = {node: row for row, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    for left in nodes:
        i = index[left]
        for right in neighbors.get(left, ()):
            j = index.get(right)
            if j is not None and i != j:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    return nodes, adjacency


def symmetric_normalized_laplacian(nodes, neighbors):
    """Return the normalized Laplacian on one fixed outcome-blind node universe."""
    nodes, adjacency = _fixed_node_graph(nodes, neighbors)
    degree = adjacency.sum(axis=1)
    inverse_root = np.zeros_like(degree)
    active = degree > 0.0
    inverse_root[active] = 1.0 / np.sqrt(degree[active])
    normalized_adjacency = inverse_root[:, None] * adjacency * inverse_root[None, :]
    laplacian = -normalized_adjacency
    laplacian[np.diag_indices_from(laplacian)] = active.astype(float)
    laplacian = 0.5 * (laplacian + laplacian.T)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    if np.min(eigenvalues) < -PSD_TOLERANCE or np.max(eigenvalues) > 2.0 + PSD_TOLERANCE:
        raise ValueError("normalized Laplacian spectrum must lie in [0,2]")
    return nodes, laplacian


def heat_kernel_reference(nodes, neighbors, *, diffusion_time):
    """Exact ``exp(-t L)`` reference on a fixed, small graph domain.

    Do not rebuild the node domain per measurement batch: that would make the
    geometry depend on batching.  Large graphs should use walk features.
    """
    diffusion_time = _positive_finite("diffusion_time", diffusion_time)
    nodes, laplacian = symmetric_normalized_laplacian(nodes, neighbors)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    value = (eigenvectors * np.exp(-diffusion_time * eigenvalues)) @ eigenvectors.T
    return nodes, normalize_psd_kernel(value)


def resolvent_kernel_reference(nodes, neighbors, *, scale):
    """Exact ``(I + scale L)^-1`` reference on a fixed, small graph domain."""
    scale = _positive_finite("scale", scale)
    nodes, laplacian = symmetric_normalized_laplacian(nodes, neighbors)
    value = np.linalg.solve(np.eye(len(nodes)) + scale * laplacian, np.eye(len(nodes)))
    return nodes, normalize_psd_kernel(value)


def _walk_basis(query_nodes, neighbors, maximum_hop):
    query_nodes = tuple(query_nodes)
    if len(set(query_nodes)) != len(query_nodes):
        raise ValueError("query_nodes must be unique")
    if not query_nodes:
        raise ValueError("query_nodes must be non-empty")
    reached = set(query_nodes)
    frontier = set(query_nodes)
    for _ in range(maximum_hop):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in neighbors.get(node, ())
            if neighbor not in reached
        }
        reached.update(frontier)
    return tuple(sorted(reached, key=_stable_key))


def _walk_distributions(query_nodes, neighbors, hop_weights):
    weights = np.asarray(hop_weights, dtype=float)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all():
        raise ValueError("hop_weights must be a non-empty finite vector")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("hop_weights must be nonnegative with at least one positive entry")
    query_nodes = tuple(query_nodes)
    basis = _walk_basis(query_nodes, neighbors, len(weights) - 1)
    index = {node: row for row, node in enumerate(basis)}
    row_indices, column_indices, values = [], [], []
    for node in basis:
        i = index[node]
        adjacent = tuple(neighbors.get(node, ()))
        if not adjacent:
            row_indices.append(i)
            column_indices.append(i)
            values.append(1.0)
            continue
        retained = [index[value] for value in adjacent if value in index]
        row_indices.extend([i] * len(retained))
        column_indices.extend(retained)
        values.extend([1.0 / len(adjacent)] * len(retained))
    transition = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(basis), len(basis)),
        dtype=float,
    )
    distribution = sparse.csr_matrix(
        (
            np.ones(len(query_nodes), dtype=float),
            (np.arange(len(query_nodes)), [index[node] for node in query_nodes]),
        ),
        shape=(len(query_nodes), len(basis)),
    )
    distributions = []
    for hop in range(len(weights)):
        distributions.append(distribution)
        if hop + 1 < len(weights):
            distribution = distribution @ transition
    return query_nodes, basis, weights, distributions


def walk_feature_map(query_nodes, neighbors, hop_weights):
    """Explicit finite-hop random-walk features with nonnegative hop weights.

    This materializes the feature matrix and is intended for inspection/tests.
    ``walk_feature_kernel`` accumulates sparse feature Grams directly at scale.
    """
    query_nodes, basis, weights, distributions = _walk_distributions(
        query_nodes, neighbors, hop_weights
    )
    blocks = [
        np.sqrt(weight) * distribution.toarray()
        for weight, distribution in zip(weights, distributions)
        if weight > 0.0
    ]
    features = np.concatenate(blocks, axis=1)
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0.0):
        raise ValueError("walk features must be finite and nonzero")
    return features, basis


def walk_feature_kernel(query_nodes, neighbors, hop_weights):
    query_nodes, basis, weights, distributions = _walk_distributions(
        query_nodes, neighbors, hop_weights
    )
    gram = np.zeros((len(query_nodes), len(query_nodes)), dtype=float)
    for weight, distribution in zip(weights, distributions):
        if weight > 0.0:
            gram += weight * (distribution @ distribution.T).toarray()
    return query_nodes, normalize_psd_kernel(gram), basis


def cumulative_walk_feature_map(query_nodes, neighbors, hop_weights):
    """Cross-hop diffusion features ``sum_h sqrt(w_h) p_h``."""
    query_nodes, basis, weights, distributions = _walk_distributions(
        query_nodes, neighbors, hop_weights
    )
    features = sparse.csr_matrix(distributions[0].shape, dtype=float)
    for weight, distribution in zip(weights, distributions):
        if weight > 0.0:
            features = features + np.sqrt(weight) * distribution
    return features, basis


def cumulative_walk_feature_kernel(query_nodes, neighbors, hop_weights):
    """PSD cumulative-walk Gram; unlike separate hop blocks, includes cross-hop overlap."""
    features, basis = cumulative_walk_feature_map(query_nodes, neighbors, hop_weights)
    gram = (features @ features.T).toarray()
    return tuple(query_nodes), normalize_psd_kernel(gram), basis


def closed_neighborhood_kernel(query_nodes, neighbors):
    """Binary closed-neighborhood cosine Gram used by the merged pilot."""
    query_nodes = tuple(query_nodes)
    if len(set(query_nodes)) != len(query_nodes) or not query_nodes:
        raise ValueError("query_nodes must be non-empty and unique")
    basis = tuple(sorted(
        set(query_nodes).union(*(
            set(neighbors.get(node, ())) for node in query_nodes
        )),
        key=_stable_key,
    ))
    index = {node: column for column, node in enumerate(basis)}
    features = np.zeros((len(query_nodes), len(basis)), dtype=float)
    for row, node in enumerate(query_nodes):
        for member in (node, *neighbors.get(node, ())):
            features[row, index[member]] = 1.0
    return query_nodes, normalize_psd_kernel(features @ features.T), basis


def descendant_gated_item_kernel(pairs, root_nodes, root_kernel):
    """Lift a root kernel to pair items and gate it by equal descendant/left role."""
    pairs = tuple(pairs)
    root_nodes = tuple(root_nodes)
    root_kernel = _kernel("root_kernel", root_kernel)
    if root_kernel.shape != (len(root_nodes), len(root_nodes)):
        raise ValueError("root_kernel must align with root_nodes")
    index = {node: row for row, node in enumerate(root_nodes)}
    try:
        roots = np.asarray([index[root] for _left, root in pairs], dtype=int)
    except KeyError as exc:
        raise ValueError(f"pair root absent from root_nodes: {exc.args[0]!r}") from exc
    lifted = root_kernel[np.ix_(roots, roots)]
    left = [pair[0] for pair in pairs]
    gate = np.equal.outer(left, left).astype(float)
    return _kernel("descendant-gated item kernel", lifted * gate)


def role_aware_pair_features(pairs, node_embeddings):
    """Normalized ``concat(left_embedding, root_embedding)`` item features."""
    pairs = tuple(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty")
    rows = []
    dimension = None
    for left, root in pairs:
        try:
            left_vector = np.asarray(node_embeddings[left], dtype=float)
            root_vector = np.asarray(node_embeddings[root], dtype=float)
        except KeyError as exc:
            raise ValueError(f"pair node absent from embeddings: {exc.args[0]!r}") from exc
        if left_vector.ndim != 1 or root_vector.shape != left_vector.shape:
            raise ValueError("every node embedding must be a same-width vector")
        dimension = len(left_vector) if dimension is None else dimension
        if len(left_vector) != dimension:
            raise ValueError("every node embedding must have the same width")
        rows.append(np.concatenate((left_vector, root_vector)))
    features = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms <= 0.0):
        raise ValueError("pair features must be finite and nonzero")
    return features / norms


def median_pairwise_distance(features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite matrix with at least two rows")
    squared = np.maximum(
        np.sum(features * features, axis=1)[:, None]
        + np.sum(features * features, axis=1)[None, :]
        - 2.0 * features @ features.T,
        0.0,
    )
    values = np.sqrt(squared[np.triu_indices(len(features), 1)])
    positive = values[values > np.finfo(float).eps]
    if not len(positive):
        raise ValueError("median distance is undefined for identical features")
    return float(np.median(positive))


def embedding_item_kernel(pairs, node_embeddings, *, length_scale, gate_descendant=True):
    """Role-aware Euclidean RBF, optionally Schur-gated by equal descendants."""
    length_scale = _positive_finite("length_scale", length_scale)
    pairs = tuple(pairs)
    features = role_aware_pair_features(pairs, node_embeddings)
    norm = np.sum(features * features, axis=1)
    squared = np.maximum(norm[:, None] + norm[None, :] - 2.0 * features @ features.T, 0.0)
    value = np.exp(-0.5 * squared / (length_scale * length_scale))
    if gate_descendant:
        left = [pair[0] for pair in pairs]
        value *= np.equal.outer(left, left)
    np.fill_diagonal(value, 1.0)
    return _kernel("embedding item kernel", value)


def convex_kernel_mixture(kernels, weights):
    """Nonnegative normalized sum of unit-diagonal PSD kernels."""
    kernels = tuple(_kernel(f"kernels[{i}]", value) for i, value in enumerate(kernels))
    weights = np.asarray(weights, dtype=float)
    if not kernels or weights.shape != (len(kernels),) or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite vector aligned with non-empty kernels")
    if np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("mixture weights must be nonnegative with positive sum")
    if any(value.shape != kernels[0].shape for value in kernels):
        raise ValueError("all kernels must have the same shape")
    weights = weights / np.sum(weights)
    value = sum(weight * kernel for weight, kernel in zip(weights, kernels))
    return _kernel("convex kernel mixture", value)


def schur_kernel_product(kernels):
    """Unit-diagonal Schur product of aligned PSD kernels."""
    kernels = tuple(_kernel(f"kernels[{i}]", value) for i, value in enumerate(kernels))
    if not kernels or any(value.shape != kernels[0].shape for value in kernels):
        raise ValueError("aligned non-empty kernels are required")
    value = np.ones_like(kernels[0])
    for kernel in kernels:
        value *= kernel
    return _kernel("Schur kernel product", value)


@dataclass(frozen=True)
class KernelDiagnostics:
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    rank: int
    positive_spectrum_condition_number: float
    effective_rank: float
    off_diagonal_rms: float
    sha256: str


def kernel_diagnostics(value):
    value = _kernel("value", value)
    eigenvalues = np.linalg.eigvalsh(value)
    positive = eigenvalues[eigenvalues > np.finfo(float).eps]
    condition = float(np.max(positive) / np.min(positive)) if len(positive) else math.inf
    total = float(np.sum(eigenvalues))
    effective_rank = float(total * total / np.sum(eigenvalues * eigenvalues)) if total else 0.0
    off = value - np.eye(len(value))
    return KernelDiagnostics(
        float(np.min(eigenvalues)),
        float(np.max(eigenvalues)),
        int(len(positive)),
        condition,
        effective_rank,
        float(np.sqrt(np.mean(off * off))),
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    )
