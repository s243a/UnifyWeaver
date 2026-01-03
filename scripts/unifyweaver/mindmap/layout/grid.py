# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Grid layout algorithm for mind maps.

"""
Grid layout placing nodes in a regular grid pattern.

Simple layout useful for debugging or as initial positions
for more sophisticated layout algorithms.
"""

from typing import Dict, List, Tuple, Optional, Any
import math


def grid(
    graph: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute grid layout for a mind map graph.

    Args:
        graph: Dictionary with 'nodes' and 'edges' keys.
        options: Optional parameters:
            - cell_width: Width of each grid cell (default: 120)
            - cell_height: Height of each grid cell (default: 80)
            - columns: Number of columns (default: auto based on sqrt(n))
            - margin_x: Left margin (default: 60)
            - margin_y: Top margin (default: 40)

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    options = options or {}
    cell_width = options.get('cell_width', 120)
    cell_height = options.get('cell_height', 80)
    columns = options.get('columns', 0)
    margin_x = options.get('margin_x', 60)
    margin_y = options.get('margin_y', 40)

    nodes = graph.get('nodes', [])

    if not nodes:
        return {}

    n = len(nodes)

    # Auto-calculate columns if not specified
    if columns <= 0:
        columns = math.ceil(math.sqrt(n))

    positions: Dict[str, Tuple[float, float]] = {}

    for i, node in enumerate(nodes):
        row = i // columns
        col = i % columns
        x = margin_x + col * cell_width + cell_width / 2
        y = margin_y + row * cell_height + cell_height / 2
        positions[node['id']] = (x, y)

    return positions


# Convenience function matching Prolog binding interface
def compute(nodes, edges, **options):
    """Compute layout - matches Prolog binding interface."""
    graph = {'nodes': nodes, 'edges': edges}
    return grid(graph, options)
