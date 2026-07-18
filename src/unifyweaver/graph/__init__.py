"""Reusable graph geometry and diffusion primitives."""

from .leaky_diffusion import (
    GroundedSemanticDiffusion,
    build_grounded_semantic_diffusion,
    combinatorial_laplacian,
    semantic_conductance_matrix,
)
from .local_diffusion import (
    AnchorScreeningProvenance,
    LeakageCalibrationResult,
    LocalDiffusionDomain,
    LocalGroundedSemanticDiffusion,
    NestedDomainDiagnostics,
    build_local_grounded_semantic_diffusion,
    calibrate_uniform_leakage,
    compare_nested_domains,
    select_hop_local_domain,
)

__all__ = [
    "AnchorScreeningProvenance",
    "GroundedSemanticDiffusion",
    "build_grounded_semantic_diffusion",
    "LeakageCalibrationResult",
    "LocalDiffusionDomain",
    "LocalGroundedSemanticDiffusion",
    "NestedDomainDiagnostics",
    "build_local_grounded_semantic_diffusion",
    "calibrate_uniform_leakage",
    "compare_nested_domains",
    "select_hop_local_domain",
    "combinatorial_laplacian",
    "semantic_conductance_matrix",
]
