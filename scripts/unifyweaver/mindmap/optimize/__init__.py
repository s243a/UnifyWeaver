# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Mind map layout optimization algorithms.

"""
Optimization passes for mind map layouts.

Provides algorithms to refine node positions after initial layout:
- Overlap removal (push apart overlapping nodes)
- Crossing minimization (reduce edge crossings)
- Spacing adjustment (normalize distances)
- Centering (center graph in viewport)
"""

from .overlap_removal import overlap_removal, remove_overlaps
from .crossing_minimization import crossing_minimization, minimize_crossings
from .spacing import spacing, adjust_spacing
from .centering import centering, center_graph

__all__ = [
    'overlap_removal',
    'remove_overlaps',
    'crossing_minimization',
    'minimize_crossings',
    'spacing',
    'adjust_spacing',
    'centering',
    'center_graph'
]
