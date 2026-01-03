# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Force-directed layout algorithm with NumPy acceleration.

"""
Force-directed layout using spring-electric model.

This implements the Fruchterman-Reingold algorithm with NumPy
for O(n^2) force calculations, providing significant speedup
for graphs with more than ~50 nodes.
"""

from typing import Dict, List, Tuple, Optional, Any
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def force_directed(
    graph: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute force-directed layout for a mind map graph.

    Args:
        graph: Dictionary with 'nodes' and 'edges' keys.
               nodes: List of dicts with 'id' and optional 'label', 'type'
               edges: List of dicts with 'source' and 'target' keys
        options: Optional parameters:
            - iterations: Number of iterations (default: 100)
            - spring_k: Spring constant (default: 0.1)
            - repulsion: Repulsion strength (default: 1000)
            - damping: Velocity damping (default: 0.85)
            - width: Layout width (default: 800)
            - height: Layout height (default: 600)

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    options = options or {}
    iterations = options.get('iterations', 100)
    spring_k = options.get('spring_k', 0.1)
    repulsion = options.get('repulsion', 1000)
    damping = options.get('damping', 0.85)
    width = options.get('width', 800)
    height = options.get('height', 600)

    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    if not nodes:
        return {}

    # Use NumPy implementation if available and graph is large enough
    if HAS_NUMPY and len(nodes) > 20:
        return numpy_force(graph, options)

    # Pure Python implementation for small graphs
    return _pure_python_force(
        nodes, edges, iterations, spring_k, repulsion, damping, width, height
    )


def _pure_python_force(
    nodes: List[Dict],
    edges: List[Dict],
    iterations: int,
    spring_k: float,
    repulsion: float,
    damping: float,
    width: float,
    height: float
) -> Dict[str, Tuple[float, float]]:
    """Pure Python force-directed layout."""
    n = len(nodes)
    node_ids = [node['id'] for node in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Initialize positions in a grid
    cols = math.ceil(math.sqrt(n))
    positions = []
    for i, _ in enumerate(nodes):
        row = i // cols
        col = i % cols
        x = (col + 0.5) * width / cols
        y = (row + 0.5) * height / max(1, (n // cols + 1))
        positions.append([x, y])

    # Build adjacency for spring forces
    adjacency = {i: set() for i in range(n)}
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in id_to_idx and tgt in id_to_idx:
            i, j = id_to_idx[src], id_to_idx[tgt]
            adjacency[i].add(j)
            adjacency[j].add(i)

    # Iterate
    for _ in range(iterations):
        forces = [[0.0, 0.0] for _ in range(n)]

        # Repulsion between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = max(1.0, math.sqrt(dx * dx + dy * dy))
                force = repulsion / (dist * dist)
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy

        # Attraction along edges
        for i in range(n):
            for j in adjacency[i]:
                if j > i:
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    force = spring_k * dist
                    if dist > 0:
                        fx = (dx / dist) * force
                        fy = (dy / dist) * force
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy

        # Apply forces with damping
        for i in range(n):
            positions[i][0] += forces[i][0] * damping
            positions[i][1] += forces[i][1] * damping

            # Keep within bounds
            positions[i][0] = max(50, min(width - 50, positions[i][0]))
            positions[i][1] = max(50, min(height - 50, positions[i][1]))

    return {node_ids[i]: (positions[i][0], positions[i][1]) for i in range(n)}


def numpy_force(
    graph: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    NumPy-accelerated force-directed layout.

    Uses vectorized operations for O(n^2) force calculations,
    providing ~10-50x speedup for large graphs.
    """
    if not HAS_NUMPY:
        return force_directed(graph, options)

    options = options or {}
    iterations = options.get('iterations', 100)
    spring_k = options.get('spring_k', 0.1)
    repulsion = options.get('repulsion', 1000)
    damping = options.get('damping', 0.85)
    width = options.get('width', 800)
    height = options.get('height', 600)

    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    if not nodes:
        return {}

    n = len(nodes)
    node_ids = [node['id'] for node in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Initialize positions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    positions = np.zeros((n, 2))
    for i in range(n):
        positions[i, 0] = ((i % cols) + 0.5) * width / cols
        positions[i, 1] = ((i // cols) + 0.5) * height / rows

    # Build edge matrix
    edge_matrix = np.zeros((n, n), dtype=bool)
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in id_to_idx and tgt in id_to_idx:
            i, j = id_to_idx[src], id_to_idx[tgt]
            edge_matrix[i, j] = True
            edge_matrix[j, i] = True

    # Iterate with vectorized operations
    for _ in range(iterations):
        # Compute pairwise differences
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
        dist = np.maximum(dist, 1.0)  # Avoid division by zero

        # Repulsion forces (all pairs)
        repulsion_force = repulsion / (dist ** 2)
        repulsion_force = np.where(dist > 0, repulsion_force, 0)
        np.fill_diagonal(repulsion_force, 0)

        # Normalize direction
        direction = diff / dist[:, :, np.newaxis]
        direction = np.nan_to_num(direction)

        # Sum repulsion forces
        forces = np.sum(direction * repulsion_force[:, :, np.newaxis], axis=1)

        # Attraction forces (connected nodes only)
        attraction_force = spring_k * dist * edge_matrix
        forces -= np.sum(direction * attraction_force[:, :, np.newaxis], axis=1)

        # Apply forces with damping
        positions += forces * damping

        # Keep within bounds
        positions[:, 0] = np.clip(positions[:, 0], 50, width - 50)
        positions[:, 1] = np.clip(positions[:, 1], 50, height - 50)

    return {node_ids[i]: (float(positions[i, 0]), float(positions[i, 1]))
            for i in range(n)}


# Convenience function matching Prolog binding signature
def compute(nodes, edges, **options):
    """Compute layout - matches Prolog binding interface."""
    graph = {'nodes': nodes, 'edges': edges}
    return force_directed(graph, options)
