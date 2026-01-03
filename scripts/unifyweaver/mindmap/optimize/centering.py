# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Graph centering optimizer.

"""
Centering adjustment for mind map layouts.

Centers the entire graph within the viewport while
maintaining relative positions.
"""

from typing import Dict, Tuple, Optional, Any


def centering(
    positions: Dict[str, Tuple[float, float]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Center the graph within a viewport.

    Args:
        positions: Dictionary mapping node IDs to (x, y) positions.
        options: Optional parameters:
            - viewport_width: Viewport width (default: 800)
            - viewport_height: Viewport height (default: 600)
            - padding: Padding from edges (default: 50)

    Returns:
        Dictionary mapping node IDs to centered (x, y) positions.
    """
    options = options or {}
    viewport_width = options.get('viewport_width', 800)
    viewport_height = options.get('viewport_height', 600)
    padding = options.get('padding', 50)

    if not positions:
        return {}

    # Find bounding box
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Calculate current center
    current_center_x = (min_x + max_x) / 2
    current_center_y = (min_y + max_y) / 2

    # Calculate target center
    target_center_x = viewport_width / 2
    target_center_y = viewport_height / 2

    # Calculate offset
    offset_x = target_center_x - current_center_x
    offset_y = target_center_y - current_center_y

    # Apply offset
    return {
        nid: (x + offset_x, y + offset_y)
        for nid, (x, y) in positions.items()
    }


def fit_to_viewport(
    positions: Dict[str, Tuple[float, float]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Scale and center the graph to fit within viewport.

    Args:
        positions: Dictionary mapping node IDs to (x, y) positions.
        options: Optional parameters:
            - viewport_width: Viewport width (default: 800)
            - viewport_height: Viewport height (default: 600)
            - padding: Padding from edges (default: 50)
            - max_scale: Maximum scale factor (default: 2.0)

    Returns:
        Dictionary mapping node IDs to fitted (x, y) positions.
    """
    options = options or {}
    viewport_width = options.get('viewport_width', 800)
    viewport_height = options.get('viewport_height', 600)
    padding = options.get('padding', 50)
    max_scale = options.get('max_scale', 2.0)

    if not positions:
        return {}

    # Find bounding box
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    graph_width = max_x - min_x
    graph_height = max_y - min_y

    if graph_width == 0:
        graph_width = 1
    if graph_height == 0:
        graph_height = 1

    # Calculate scale to fit
    available_width = viewport_width - 2 * padding
    available_height = viewport_height - 2 * padding

    scale_x = available_width / graph_width
    scale_y = available_height / graph_height
    scale = min(scale_x, scale_y, max_scale)

    # Calculate center offsets
    current_center_x = (min_x + max_x) / 2
    current_center_y = (min_y + max_y) / 2

    target_center_x = viewport_width / 2
    target_center_y = viewport_height / 2

    # Apply scale and translation
    return {
        nid: (
            target_center_x + (x - current_center_x) * scale,
            target_center_y + (y - current_center_y) * scale
        )
        for nid, (x, y) in positions.items()
    }


# Alias for Prolog binding interface
center_graph = centering


def compute(positions, **options):
    """Compute optimization - matches Prolog binding interface."""
    return centering(positions, options)
