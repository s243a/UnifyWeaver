"""
Core density manifold computation.

This module works in:
- Direct Python (Streamlit)
- Flask API
- Pyodide (browser WASM)

Dependencies: numpy, scipy (available in all environments)
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Optional, Tuple, List, Dict, Any

from .data_format import (
    DensityManifoldData, DensityGrid, TreeStructure, TreeNode, TreeEdge,
    DensityPeak, ProjectionInfo
)


def load_embeddings(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load embeddings from .npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = data['embeddings']
    titles = list(data['titles']) if 'titles' in data else None
    return embeddings, titles


def project_to_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project embeddings to 2D via SVD.

    Returns:
        points_2d: (N, 2) coordinates
        singular_values: top 2 singular values
        variance_explained: [var1, var2] as percentages
    """
    # Center
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Project to top 2 components
    V_2d = Vt[:2].T
    points_2d = centered @ V_2d

    # Variance explained
    var = S[:2] ** 2
    var_explained = (var / var.sum() * 100).tolist()

    return points_2d, S[:2], var_explained


def compute_density_grid(
    points_2d: np.ndarray,
    bandwidth: Optional[float] = None,
    grid_size: int = 100,
    padding: float = 0.1
) -> DensityGrid:
    """
    Compute density grid using KDE.

    Args:
        points_2d: (N, 2) projected points
        bandwidth: KDE bandwidth (None for Scott's rule)
        grid_size: grid resolution
        padding: fraction of range to pad

    Returns:
        DensityGrid object
    """
    x, y = points_2d[:, 0], points_2d[:, 1]

    # Bounds with padding
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_min = x.min() - padding * x_range
    x_max = x.max() + padding * x_range
    y_min = y.min() - padding * y_range
    y_max = y.max() + padding * y_range

    # Grid
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xi, yi)

    # KDE
    values = np.vstack([x, y])
    if bandwidth is not None:
        kde = gaussian_kde(values, bw_method=bandwidth)
    else:
        kde = gaussian_kde(values)
        bandwidth = kde.factor  # Scott's rule factor

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    return DensityGrid(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        grid_size=grid_size,
        values=Z.tolist(),
        bandwidth=float(bandwidth)
    )


def build_mst_tree(
    embeddings: np.ndarray,
    points_2d: np.ndarray,
    titles: Optional[List[str]] = None
) -> TreeStructure:
    """Build minimum spanning tree."""
    # Normalize and compute cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)
    similarity = emb_norm @ emb_norm.T
    cos_dist = 1 - similarity
    np.fill_diagonal(cos_dist, 0)

    # MST
    mst = minimum_spanning_tree(cos_dist)
    cx = mst.tocoo()

    # Build adjacency
    adj = {}
    for i, j, w in zip(cx.row, cx.col, cx.data):
        adj.setdefault(i, []).append((j, w))
        adj.setdefault(j, []).append((i, w))

    # Root at highest degree
    degrees = [(len(adj.get(i, [])), i) for i in range(len(embeddings))]
    _, root = max(degrees)

    # BFS to build tree
    parent = {root: None}
    depth = {root: 0}
    visited = {root}
    queue = [root]

    while queue:
        node = queue.pop(0)
        for neighbor, _ in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)

    # Build nodes and edges
    nodes = []
    edges = []

    for i in range(len(embeddings)):
        nodes.append(TreeNode(
            id=i,
            title=titles[i] if titles else f"Node {i}",
            parent_id=parent.get(i),
            depth=depth.get(i, 0),
            x=float(points_2d[i, 0]),
            y=float(points_2d[i, 1])
        ))

        if parent.get(i) is not None:
            edges.append(TreeEdge(
                source_id=parent[i],
                target_id=i,
                weight=float(cos_dist[i, parent[i]])
            ))

    return TreeStructure(
        nodes=nodes,
        edges=edges,
        root_id=root,
        tree_type='mst'
    )


def find_density_peaks(
    density_grid: DensityGrid,
    points_2d: np.ndarray,
    titles: Optional[List[str]] = None,
    n_peaks: int = 5,
    min_distance: int = 5
) -> List[DensityPeak]:
    """Find local maxima in density field."""
    from scipy.ndimage import maximum_filter

    Z = np.array(density_grid.values)

    # Local maxima
    neighborhood_size = min_distance * 2 + 1
    local_max = maximum_filter(Z, size=neighborhood_size) == Z
    peak_mask = local_max & (Z > Z.mean())
    peak_indices = np.argwhere(peak_mask)

    if len(peak_indices) == 0:
        return []

    # Grid coordinates
    xi = np.linspace(density_grid.x_min, density_grid.x_max, density_grid.grid_size)
    yi = np.linspace(density_grid.y_min, density_grid.y_max, density_grid.grid_size)

    # Sort by density
    peak_densities = Z[peak_mask]
    sorted_idx = np.argsort(-peak_densities)[:n_peaks]

    peaks = []
    for idx in sorted_idx:
        i, j = peak_indices[idx]
        peak_x = xi[j]
        peak_y = yi[i]
        peak_density = Z[i, j]

        # Find nearest data point
        distances = np.sqrt((points_2d[:, 0] - peak_x)**2 + (points_2d[:, 1] - peak_y)**2)
        nearest_idx = int(np.argmin(distances))

        peaks.append(DensityPeak(
            x=float(peak_x),
            y=float(peak_y),
            density=float(peak_density),
            nearest_node_id=nearest_idx,
            title=titles[nearest_idx] if titles else f"Node {nearest_idx}"
        ))

    return peaks


def compute_density_manifold(
    embeddings: np.ndarray,
    titles: Optional[List[str]] = None,
    bandwidth: Optional[float] = None,
    grid_size: int = 100,
    include_tree: bool = True,
    tree_type: str = 'mst',
    include_peaks: bool = True,
    n_peaks: int = 5
) -> DensityManifoldData:
    """
    Main function: compute complete density manifold data.

    This is the primary entry point used by all frontends.

    Args:
        embeddings: (N, D) embedding matrix
        titles: optional node labels
        bandwidth: KDE bandwidth (None for auto)
        grid_size: density grid resolution
        include_tree: whether to compute tree overlay
        tree_type: 'mst' or 'j-guided'
        include_peaks: whether to find density peaks
        n_peaks: number of peaks to find

    Returns:
        DensityManifoldData ready for frontend
    """
    # Project to 2D
    points_2d, singular_values, var_explained = project_to_2d(embeddings)

    # Compute density
    density_grid = compute_density_grid(points_2d, bandwidth, grid_size)

    # Build tree if requested
    tree = None
    if include_tree:
        if tree_type == 'mst':
            tree = build_mst_tree(embeddings, points_2d, titles)
        # TODO: Add j-guided tree

    # Find peaks if requested
    peaks = None
    if include_peaks:
        peaks = find_density_peaks(density_grid, points_2d, titles, n_peaks)

    # Build points list
    points = []
    for i in range(len(embeddings)):
        points.append({
            'id': i,
            'title': titles[i] if titles else f"Node {i}",
            'x': float(points_2d[i, 0]),
            'y': float(points_2d[i, 1])
        })

    return DensityManifoldData(
        points=points,
        density_grid=density_grid,
        tree=tree,
        peaks=peaks,
        projection=ProjectionInfo(
            variance_explained=var_explained,
            singular_values=singular_values.tolist()
        ),
        n_points=len(embeddings)
    )


# Convenience function for loading and computing in one call
def load_and_compute(
    embeddings_path: str,
    top_k: Optional[int] = None,
    **kwargs
) -> DensityManifoldData:
    """Load embeddings and compute density manifold."""
    embeddings, titles = load_embeddings(embeddings_path)

    if top_k:
        embeddings = embeddings[:top_k]
        titles = titles[:top_k] if titles else None

    return compute_density_manifold(embeddings, titles, **kwargs)
