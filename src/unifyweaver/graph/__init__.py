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
from .bounded_diffusion_fidelity import (
    ExactExteriorSchurReduction,
    ExteriorComponent,
    ExteriorComponentDiscovery,
    ExteriorTraversalLimitError,
    discover_exterior_components,
    reduce_exact_exterior_component,
)

__all__ = [
    "AnchorLeakageCalibrationResult",
    "AnchorScreeningProvenance",
    "LeakageCalibrationMinimalityCertificate",
    "GroundedSemanticDiffusion",
    "ExactExteriorSchurReduction",
    "ExteriorComponent",
    "ExteriorComponentDiscovery",
    "ExteriorTraversalLimitError",
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
    "discover_exterior_components",
    "reduce_exact_exterior_component",
    "select_hop_local_domain",
    "combinatorial_laplacian",
    "semantic_conductance_matrix",
]
