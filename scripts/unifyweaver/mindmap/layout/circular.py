# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Circular layout algorithm for mind maps.

"""
Circular layout placing all nodes on a circle.

Nodes are evenly distributed around the circumference,
which can help visualize cyclic relationships.
"""

from typing import Dict, List, Tuple, Optional, Any
import math


def circular(
    graph: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute circular layout for a mind map graph.

    Args:
        graph: Dictionary with 'nodes' and 'edges' keys.
        options: Optional parameters:
            - center_x: X coordinate of center (default: 400)
            - center_y: Y coordinate of center (default: 300)
            - radius: Circle radius (default: 200)
            - start_angle: Starting angle in radians (default: -pi/2, top)
            - clockwise: Direction of layout (default: False)

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    options = options or {}
    center_x = options.get('center_x', 400)
    center_y = options.get('center_y', 300)
    radius = options.get('radius', 200)
    start_angle = options.get('start_angle', -math.pi / 2)
    clockwise = options.get('clockwise', False)

    nodes = graph.get('nodes', [])

    if not nodes:
        return {}

    n = len(nodes)
    positions: Dict[str, Tuple[float, float]] = {}

    direction = -1 if clockwise else 1

    for i, node in enumerate(nodes):
        angle = start_angle + direction * (2 * math.pi * i / n)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions[node['id']] = (x, y)

    return positions


# Convenience function matching Prolog binding interface
def compute(nodes, edges, **options):
    """Compute layout - matches Prolog binding interface."""
    graph = {'nodes': nodes, 'edges': edges}
    return circular(graph, options)
