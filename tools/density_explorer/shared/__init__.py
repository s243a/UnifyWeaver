"""
Shared components for density manifold explorer.

Works across all frontends: Streamlit, Flask API, Pyodide.
"""

from .data_format import (
    DensityManifoldData,
    DensityGrid,
    TreeStructure,
    TreeNode,
    TreeEdge,
    DensityPeak,
    ProjectionInfo,
    JSON_SCHEMA
)

from .density_core import (
    load_embeddings,
    project_to_2d,
    compute_density_grid,
    build_mst_tree,
    find_density_peaks,
    compute_density_manifold,
    load_and_compute
)

__all__ = [
    # Data structures
    'DensityManifoldData',
    'DensityGrid',
    'TreeStructure',
    'TreeNode',
    'TreeEdge',
    'DensityPeak',
    'ProjectionInfo',
    'JSON_SCHEMA',
    # Functions
    'load_embeddings',
    'project_to_2d',
    'compute_density_grid',
    'build_mst_tree',
    'find_density_peaks',
    'compute_density_manifold',
    'load_and_compute',
]
