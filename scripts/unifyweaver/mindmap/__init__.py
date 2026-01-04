# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Mind Map Python implementations for UnifyWeaver.

"""
Mind map layout, optimization, and rendering implementations.

These modules provide NumPy/SciPy-accelerated implementations of
layout algorithms and optimization passes for mind map visualization.

The io module provides JSON Lines I/O for cross-target communication
with the Prolog DSL via pipes.
"""

from . import layout
from . import optimize
from . import render
from . import io

__all__ = ['layout', 'optimize', 'render', 'io']
