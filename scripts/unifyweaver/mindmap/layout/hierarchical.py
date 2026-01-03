# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Hierarchical (tree) layout algorithm for mind maps.

"""
Hierarchical layout arranging nodes in tree structure.

Places nodes in horizontal levels with the root at the top,
using the Reingold-Tilford algorithm for aesthetic tree layout.
"""

from typing import Dict, List, Tuple, Optional, Any, Set


def hierarchical(
    graph: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute hierarchical tree layout for a mind map graph.

    Args:
        graph: Dictionary with 'nodes' and 'edges' keys.
        options: Optional parameters:
            - level_height: Vertical distance between levels (default: 80)
            - node_spacing: Horizontal spacing between siblings (default: 100)
            - direction: 'top-down', 'bottom-up', 'left-right', 'right-left' (default: 'top-down')
            - center_x: X coordinate of root (default: 400)
            - margin_top: Top margin (default: 50)

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    options = options or {}
    level_height = options.get('level_height', 80)
    node_spacing = options.get('node_spacing', 100)
    direction = options.get('direction', 'top-down')
    center_x = options.get('center_x', 400)
    margin_top = options.get('margin_top', 50)

    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    if not nodes:
        return {}

    # Build tree structure
    node_ids = {node['id'] for node in nodes}
    children: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    parents: Dict[str, Optional[str]] = {nid: None for nid in node_ids}

    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src in node_ids and tgt in node_ids:
            children[src].append(tgt)
            parents[tgt] = src

    # Find root
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

    # Assign levels via BFS
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
    max_level = max(levels.values(), default=0)
    for nid in node_ids:
        if nid not in levels:
            max_level += 1
            levels[nid] = max_level

    # Group by level
    level_nodes: Dict[int, List[str]] = {}
    for nid, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(nid)

    # Calculate subtree widths for better spacing
    subtree_width: Dict[str, int] = {}

    def calc_width(nid: str) -> int:
        child_list = [c for c in children[nid] if c in levels]
        if not child_list:
            subtree_width[nid] = 1
            return 1
        total = sum(calc_width(c) for c in child_list)
        subtree_width[nid] = max(1, total)
        return subtree_width[nid]

    calc_width(root_id)

    # Position nodes
    positions: Dict[str, Tuple[float, float]] = {}

    def position_subtree(nid: str, x_start: float, y: float) -> float:
        """Position node and its subtree, return width used."""
        child_list = [c for c in children[nid] if c in levels]

        if not child_list:
            positions[nid] = (x_start + node_spacing / 2, y)
            return node_spacing

        # Position children first
        total_width = sum(subtree_width.get(c, 1) for c in child_list) * node_spacing
        child_x = x_start
        child_y = y + level_height

        for child in child_list:
            child_width = subtree_width.get(child, 1) * node_spacing
            position_subtree(child, child_x, child_y)
            child_x += child_width

        # Position parent centered above children
        first_child_x = positions[child_list[0]][0]
        last_child_x = positions[child_list[-1]][0]
        parent_x = (first_child_x + last_child_x) / 2
        positions[nid] = (parent_x, y)

        return total_width

    # Start positioning from root
    total_width = subtree_width.get(root_id, 1) * node_spacing
    start_x = center_x - total_width / 2
    position_subtree(root_id, start_x, margin_top)

    # Handle disconnected nodes
    orphan_x = start_x
    orphan_y = margin_top + (max_level + 1) * level_height
    for nid in node_ids:
        if nid not in positions:
            positions[nid] = (orphan_x, orphan_y)
            orphan_x += node_spacing

    # Transform based on direction
    if direction == 'bottom-up':
        max_y = max(p[1] for p in positions.values())
        positions = {nid: (x, max_y - y + margin_top) for nid, (x, y) in positions.items()}
    elif direction == 'left-right':
        positions = {nid: (y, x) for nid, (x, y) in positions.items()}
    elif direction == 'right-left':
        max_y = max(p[1] for p in positions.values())
        positions = {nid: (max_y - y + margin_top, x) for nid, (x, y) in positions.items()}

    return positions


# Convenience function matching Prolog binding interface
def compute(nodes, edges, **options):
    """Compute layout - matches Prolog binding interface."""
    graph = {'nodes': nodes, 'edges': edges}
    return hierarchical(graph, options)
