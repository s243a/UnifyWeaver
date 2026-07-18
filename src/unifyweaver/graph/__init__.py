"""Reusable graph geometry and diffusion primitives."""

from .leaky_diffusion import (
    GroundedSemanticDiffusion,
    build_grounded_semantic_diffusion,
    combinatorial_laplacian,
    semantic_conductance_matrix,
)

__all__ = [
    "GroundedSemanticDiffusion",
    "build_grounded_semantic_diffusion",
    "combinatorial_laplacian",
    "semantic_conductance_matrix",
]
