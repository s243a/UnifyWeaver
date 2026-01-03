# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Radial layout algorithm for mind maps.

"""
Radial layout placing nodes in concentric circles around a root.

The root node is placed at the center, and children are arranged
in rings at increasing distances based on their depth in the tree.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import math


def radial(
    graph: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute radial layout for a mind map graph.

    Args:
        graph: Dictionary with 'nodes' and 'edges' keys.
        options: Optional parameters:
            - center_x: X coordinate of center (default: 400)
            - center_y: Y coordinate of center (default: 300)
            - radius_step: Distance between rings (default: 100)
            - start_angle: Starting angle in radians (default: 0)
            - sweep: Angular sweep in radians (default: 2*pi)

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    options = options or {}
    center_x = options.get('center_x', 400)
    center_y = options.get('center_y', 300)
    radius_step = options.get('radius_step', 100)
    start_angle = options.get('start_angle', 0)
    sweep = options.get('sweep', 2 * math.pi)

    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    if not nodes:
        return {}

    # Build adjacency and find root
    node_ids = {node['id'] for node in nodes}
    children: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    parents: Dict[str, Optional[str]] = {nid: None for nid in node_ids}

    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in node_ids and tgt in node_ids:
            children[src].append(tgt)
            parents[tgt] = src

    # Find root (node with type 'root' or no parent)
    root_id = None
    for node in nodes:
        if node.get('type') == 'root':
            root_id = node['id']
            break
    if root_id is None:
        for nid in node_ids:
            if parents[nid] is None:
                root_id = nid
                break
    if root_id is None:
        root_id = nodes[0]['id']

    # BFS to assign levels
    levels: Dict[str, int] = {}
    queue = [(root_id, 0)]
    visited: Set[str] = set()

    while queue:
        nid, level = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        levels[nid] = level
        for child in children[nid]:
            if child not in visited:
                queue.append((child, level + 1))

    # Handle disconnected nodes
    for nid in node_ids:
        if nid not in levels:
            levels[nid] = max(levels.values(), default=0) + 1

    # Group nodes by level
    level_nodes: Dict[int, List[str]] = {}
    for nid, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(nid)

    # Position nodes
    positions: Dict[str, Tuple[float, float]] = {}

    for level, nids in level_nodes.items():
        if level == 0:
            # Root at center
            for nid in nids:
                positions[nid] = (center_x, center_y)
        else:
            # Arrange on circle
            radius = level * radius_step
            n = len(nids)
            for i, nid in enumerate(nids):
                angle = start_angle + (sweep * i / max(1, n))
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions[nid] = (x, y)

    return positions


# Convenience function matching Prolog binding interface
def compute(nodes, edges, **options):
    """Compute layout - matches Prolog binding interface."""
    graph = {'nodes': nodes, 'edges': edges}
    return radial(graph, options)
