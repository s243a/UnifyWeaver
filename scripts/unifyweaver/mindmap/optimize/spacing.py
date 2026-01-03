# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Spacing adjustment optimizer.

"""
Spacing adjustment for mind map layouts.

Normalizes distances between nodes to improve visual balance
while preserving the overall structure.
"""

from typing import Dict, List, Tuple, Optional, Any
import math


def spacing(
    positions: Dict[str, Tuple[float, float]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Adjust spacing between nodes for visual balance.

    Args:
        positions: Dictionary mapping node IDs to (x, y) positions.
        options: Optional parameters:
            - target_spacing: Ideal distance between nodes (default: 100)
            - iterations: Number of adjustment iterations (default: 20)
            - strength: Adjustment strength (default: 0.1)
            - min_spacing: Minimum allowed spacing (default: 50)

    Returns:
        Dictionary mapping node IDs to adjusted (x, y) positions.
    """
    options = options or {}
    target_spacing = options.get('target_spacing', 100)
    iterations = options.get('iterations', 20)
    strength = options.get('strength', 0.1)
    min_spacing = options.get('min_spacing', 50)

    if not positions or len(positions) < 2:
        return positions.copy() if positions else {}

    node_ids = list(positions.keys())
    n = len(node_ids)

    # Copy positions
    pos = {nid: list(positions[nid]) for nid in node_ids}

    for _ in range(iterations):
        forces = {nid: [0.0, 0.0] for nid in node_ids}

        for i in range(n):
            nid_i = node_ids[i]
            xi, yi = pos[nid_i]

            # Find nearest neighbor
            min_dist = float('inf')
            nearest = None

            for j in range(n):
                if i == j:
                    continue
                nid_j = node_ids[j]
                xj, yj = pos[nid_j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = nid_j

            if nearest is None:
                continue

            xn, yn = pos[nearest]
            dx = xi - xn
            dy = yi - yn
            dist = max(1, math.sqrt(dx * dx + dy * dy))

            # Calculate adjustment
            diff = target_spacing - dist
            if abs(diff) > 5:  # Only adjust if significantly off
                adjustment = diff * strength
                # Normalize direction
                fx = (dx / dist) * adjustment
                fy = (dy / dist) * adjustment
                forces[nid_i][0] += fx
                forces[nid_i][1] += fy

        # Apply forces
        for nid in node_ids:
            pos[nid][0] += forces[nid][0]
            pos[nid][1] += forces[nid][1]

    return {nid: (pos[nid][0], pos[nid][1]) for nid in node_ids}


# Alias for Prolog binding interface
adjust_spacing = spacing


def compute(positions, **options):
    """Compute optimization - matches Prolog binding interface."""
    return spacing(positions, options)
