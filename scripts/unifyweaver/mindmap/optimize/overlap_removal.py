# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Overlap removal optimizer with NumPy acceleration.

"""
Overlap removal algorithm for mind map layouts.

Iteratively pushes apart overlapping nodes while preserving
the general structure of the layout.
"""

from typing import Dict, List, Tuple, Optional, Any
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def overlap_removal(
    positions: Dict[str, Tuple[float, float]],
    node_data: Optional[Dict[str, Dict[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Remove overlaps between nodes by pushing them apart.

    Args:
        positions: Dictionary mapping node IDs to (x, y) positions.
        node_data: Optional dictionary mapping node IDs to properties
                   including 'width' and 'height'.
        options: Optional parameters:
            - min_distance: Minimum distance between nodes (default: 20)
            - iterations: Maximum iterations (default: 100)
            - damping: Movement damping factor (default: 0.5)
            - default_width: Default node width (default: 80)
            - default_height: Default node height (default: 40)

    Returns:
        Dictionary mapping node IDs to adjusted (x, y) positions.
    """
    options = options or {}
    min_distance = options.get('min_distance', 20)
    iterations = options.get('iterations', 100)
    damping = options.get('damping', 0.5)
    default_width = options.get('default_width', 80)
    default_height = options.get('default_height', 40)

    node_data = node_data or {}

    if not positions:
        return {}

    # Use NumPy for large graphs
    if HAS_NUMPY and len(positions) > 20:
        return _numpy_overlap_removal(
            positions, node_data, min_distance, iterations, damping,
            default_width, default_height
        )

    return _pure_python_overlap_removal(
        positions, node_data, min_distance, iterations, damping,
        default_width, default_height
    )


def _pure_python_overlap_removal(
    positions: Dict[str, Tuple[float, float]],
    node_data: Dict[str, Dict[str, Any]],
    min_distance: float,
    iterations: int,
    damping: float,
    default_width: float,
    default_height: float
) -> Dict[str, Tuple[float, float]]:
    """Pure Python overlap removal."""
    node_ids = list(positions.keys())
    n = len(node_ids)

    # Get node dimensions
    widths = []
    heights = []
    for nid in node_ids:
        data = node_data.get(nid, {})
        widths.append(data.get('width', default_width))
        heights.append(data.get('height', default_height))

    # Copy positions
    pos = {nid: list(positions[nid]) for nid in node_ids}

    for _ in range(iterations):
        moved = False

        for i in range(n):
            nid_i = node_ids[i]
            xi, yi = pos[nid_i]
            wi, hi = widths[i], heights[i]

            for j in range(i + 1, n):
                nid_j = node_ids[j]
                xj, yj = pos[nid_j]
                wj, hj = widths[j], heights[j]

                # Check for overlap
                half_w = (wi + wj) / 2 + min_distance
                half_h = (hi + hj) / 2 + min_distance

                dx = xi - xj
                dy = yi - yj

                overlap_x = half_w - abs(dx)
                overlap_y = half_h - abs(dy)

                if overlap_x > 0 and overlap_y > 0:
                    moved = True

                    # Push apart along the axis with less overlap
                    if overlap_x < overlap_y:
                        push = overlap_x * damping / 2
                        if dx >= 0:
                            pos[nid_i][0] += push
                            pos[nid_j][0] -= push
                        else:
                            pos[nid_i][0] -= push
                            pos[nid_j][0] += push
                    else:
                        push = overlap_y * damping / 2
                        if dy >= 0:
                            pos[nid_i][1] += push
                            pos[nid_j][1] -= push
                        else:
                            pos[nid_i][1] -= push
                            pos[nid_j][1] += push

        if not moved:
            break

    return {nid: (pos[nid][0], pos[nid][1]) for nid in node_ids}


def _numpy_overlap_removal(
    positions: Dict[str, Tuple[float, float]],
    node_data: Dict[str, Dict[str, Any]],
    min_distance: float,
    iterations: int,
    damping: float,
    default_width: float,
    default_height: float
) -> Dict[str, Tuple[float, float]]:
    """NumPy-accelerated overlap removal."""
    node_ids = list(positions.keys())
    n = len(node_ids)

    # Build position array
    pos = np.array([[positions[nid][0], positions[nid][1]] for nid in node_ids])

    # Build dimension arrays
    widths = np.array([node_data.get(nid, {}).get('width', default_width)
                       for nid in node_ids])
    heights = np.array([node_data.get(nid, {}).get('height', default_height)
                        for nid in node_ids])

    for _ in range(iterations):
        # Compute pairwise differences
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 2)

        # Compute overlap thresholds
        half_w = (widths[:, np.newaxis] + widths[np.newaxis, :]) / 2 + min_distance
        half_h = (heights[:, np.newaxis] + heights[np.newaxis, :]) / 2 + min_distance

        # Compute overlaps
        overlap_x = half_w - np.abs(diff[:, :, 0])
        overlap_y = half_h - np.abs(diff[:, :, 1])

        # Mask for actual overlaps (excluding self)
        is_overlap = (overlap_x > 0) & (overlap_y > 0)
        np.fill_diagonal(is_overlap, False)

        if not np.any(is_overlap):
            break

        # Compute push vectors
        push = np.zeros_like(pos)

        for i in range(n):
            for j in range(i + 1, n):
                if is_overlap[i, j]:
                    ox = overlap_x[i, j]
                    oy = overlap_y[i, j]
                    dx = diff[i, j, 0]
                    dy = diff[i, j, 1]

                    if ox < oy:
                        p = ox * damping / 2
                        if dx >= 0:
                            push[i, 0] += p
                            push[j, 0] -= p
                        else:
                            push[i, 0] -= p
                            push[j, 0] += p
                    else:
                        p = oy * damping / 2
                        if dy >= 0:
                            push[i, 1] += p
                            push[j, 1] -= p
                        else:
                            push[i, 1] -= p
                            push[j, 1] += p

        pos += push

    return {node_ids[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


# Aliases for Prolog binding interface
remove_overlaps = overlap_removal


def compute(positions, node_data=None, **options):
    """Compute optimization - matches Prolog binding interface."""
    return overlap_removal(positions, node_data, options)
