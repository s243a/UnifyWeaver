"""Reusable graph geometry and diffusion primitives."""

from .leaky_diffusion import (
    GroundedSemanticDiffusion,
    build_grounded_semantic_diffusion,
    combinatorial_laplacian,
    semantic_conductance_matrix,
)
from .local_diffusion import (
    AnchorLeakageCalibrationResult,
    AnchorScreeningProvenance,
    LeakageCalibrationMinimalityCertificate,
    LeakageCalibrationResult,
    LocalDiffusionDomain,
    LocalGroundedSemanticDiffusion,
    NestedDomainDiagnostics,
    PerAnchorLeakageCalibrationResult,
    build_local_grounded_semantic_diffusion,
    calibrate_uniform_leakage,
    calibrate_uniform_leakage_per_anchor,
    compare_nested_domains,
    select_hop_local_domain,
)

__all__ = [
    "AnchorLeakageCalibrationResult",
    "AnchorScreeningProvenance",
    "LeakageCalibrationMinimalityCertificate",
    "GroundedSemanticDiffusion",
    "build_grounded_semantic_diffusion",
    "LeakageCalibrationResult",
    "LocalDiffusionDomain",
    "LocalGroundedSemanticDiffusion",
    "NestedDomainDiagnostics",
    "PerAnchorLeakageCalibrationResult",
    "build_local_grounded_semantic_diffusion",
    "calibrate_uniform_leakage",
    "calibrate_uniform_leakage_per_anchor",
    "compare_nested_domains",
    "select_hop_local_domain",
    "combinatorial_laplacian",
    "semantic_conductance_matrix",
]
