# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Edge crossing minimization optimizer.

"""
Edge crossing minimization for mind map layouts.

Implements heuristics to reduce the number of edge crossings
by adjusting node positions while preserving layout structure.
"""

from typing import Dict, List, Tuple, Optional, Any
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def crossing_minimization(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Dict[str, str]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Minimize edge crossings by adjusting node positions.

    Args:
        positions: Dictionary mapping node IDs to (x, y) positions.
        edges: List of edge dictionaries with 'source' and 'target' keys.
        options: Optional parameters:
            - iterations: Maximum iterations (default: 50)
            - temperature: Initial temperature for simulated annealing (default: 10.0)
            - cooling_rate: Temperature decay rate (default: 0.95)
            - max_displacement: Maximum node movement per iteration (default: 20)

    Returns:
        Dictionary mapping node IDs to adjusted (x, y) positions.
    """
    options = options or {}
    iterations = options.get('iterations', 50)
    temperature = options.get('temperature', 10.0)
    cooling_rate = options.get('cooling_rate', 0.95)
    max_displacement = options.get('max_displacement', 20)

    if not positions or not edges:
        return positions.copy() if positions else {}

    # Convert to working format
    node_ids = list(positions.keys())
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build edge list with indices
    edge_list = []
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in id_to_idx and tgt in id_to_idx:
            edge_list.append((id_to_idx[src], id_to_idx[tgt]))

    if not edge_list:
        return positions.copy()

    # Copy positions
    pos = [[positions[nid][0], positions[nid][1]] for nid in node_ids]

    # Count initial crossings
    initial_crossings = _count_crossings(pos, edge_list)

    if initial_crossings == 0:
        return positions.copy()

    # Simulated annealing to minimize crossings
    best_pos = [p[:] for p in pos]
    best_crossings = initial_crossings
    current_crossings = initial_crossings

    for iteration in range(iterations):
        # Try moving each node slightly
        for i in range(n):
            # Generate random displacement
            angle = math.random() * 2 * math.pi if hasattr(math, 'random') else (iteration * i) % (2 * math.pi)
            displacement = temperature * max_displacement / (iterations + 1)
            dx = displacement * math.cos(angle)
            dy = displacement * math.sin(angle)

            # Try the move
            old_x, old_y = pos[i]
            pos[i][0] += dx
            pos[i][1] += dy

            new_crossings = _count_crossings(pos, edge_list)

            # Accept or reject
            if new_crossings < current_crossings:
                current_crossings = new_crossings
                if new_crossings < best_crossings:
                    best_crossings = new_crossings
                    best_pos = [p[:] for p in pos]
            else:
                # Reject move
                pos[i][0] = old_x
                pos[i][1] = old_y

        # Cool down
        temperature *= cooling_rate

        # Early termination if no crossings
        if best_crossings == 0:
            break

    return {node_ids[i]: (best_pos[i][0], best_pos[i][1]) for i in range(n)}


def _count_crossings(pos: List[List[float]], edges: List[Tuple[int, int]]) -> int:
    """Count the number of edge crossings."""
    crossings = 0
    m = len(edges)

    for i in range(m):
        e1 = edges[i]
        p1 = pos[e1[0]]
        p2 = pos[e1[1]]

        for j in range(i + 1, m):
            e2 = edges[j]
            # Skip if edges share a vertex
            if e1[0] in e2 or e1[1] in e2:
                continue

            p3 = pos[e2[0]]
            p4 = pos[e2[1]]

            if _segments_intersect(p1, p2, p3, p4):
                crossings += 1

    return crossings


def _segments_intersect(
    p1: List[float], p2: List[float],
    p3: List[float], p4: List[float]
) -> bool:
    """Check if line segments p1-p2 and p3-p4 intersect."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def detect_crossings(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Dict[str, str]]
) -> List[Tuple[int, int]]:
    """
    Detect all edge crossings in the layout.

    Returns:
        List of (edge_index_1, edge_index_2) tuples for crossing edges.
    """
    node_ids = list(positions.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    pos = [[positions[nid][0], positions[nid][1]] for nid in node_ids]

    edge_list = []
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in id_to_idx and tgt in id_to_idx:
            edge_list.append((id_to_idx[src], id_to_idx[tgt]))

    crossings = []
    m = len(edge_list)

    for i in range(m):
        e1 = edge_list[i]
        p1 = pos[e1[0]]
        p2 = pos[e1[1]]

        for j in range(i + 1, m):
            e2 = edge_list[j]
            if e1[0] in e2 or e1[1] in e2:
                continue

            p3 = pos[e2[0]]
            p4 = pos[e2[1]]

            if _segments_intersect(p1, p2, p3, p4):
                crossings.append((i, j))

    return crossings


# Alias for Prolog binding interface
minimize_crossings = crossing_minimization


def compute(positions, edges, **options):
    """Compute optimization - matches Prolog binding interface."""
    return crossing_minimization(positions, edges, options)
