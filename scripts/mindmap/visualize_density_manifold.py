#!/usr/bin/env python3
"""
Visualize density manifolds of embeddings.

Projects embeddings onto the dominant two singular vectors (SVD) and plots:
- Contour lines: equipotential surfaces of the density function
- Gradient lines: direction of steepest ascent toward density peaks

The density is estimated using Kernel Density Estimation (KDE) with
Gaussian kernels at configurable bandwidth.

Usage:
    # Visualize Wikipedia physics embeddings
    python3 scripts/mindmap/visualize_density_manifold.py

    # With custom bandwidth
    python3 scripts/mindmap/visualize_density_manifold.py --bandwidth 0.1

    # Save to file
    python3 scripts/mindmap/visualize_density_manifold.py -o density_manifold.png

    # Use different dataset
    python3 scripts/mindmap/visualize_density_manifold.py --embeddings path/to/embeddings.npz
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import minimum_spanning_tree

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/mindmap"))


def load_embeddings(path: Path) -> Tuple[np.ndarray, Optional[list]]:
    """Load embeddings from .npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = data['embeddings']
    titles = list(data['titles']) if 'titles' in data else None
    return embeddings, titles


def project_to_svd_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project embeddings onto top 2 singular vectors.

    Args:
        embeddings: (N, D) embedding matrix

    Returns:
        projected: (N, 2) coordinates in SVD space
        V: (D, 2) the top 2 right singular vectors
        singular_values: top 2 singular values
    """
    # Center the data
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Top 2 components
    V_2d = Vt[:2].T  # (D, 2)
    projected = centered @ V_2d  # (N, 2)

    return projected, V_2d, S[:2]


def compute_density_grid(
    points_2d: np.ndarray,
    bandwidth: Optional[float] = None,
    grid_size: int = 100,
    padding: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute density on a 2D grid using KDE.

    Args:
        points_2d: (N, 2) projected points
        bandwidth: KDE bandwidth (None for automatic)
        grid_size: Number of grid points per axis
        padding: Fraction of range to add as padding

    Returns:
        X, Y: meshgrid coordinates
        Z: density values at each grid point
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # Grid bounds with padding
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_min, x_max = x.min() - padding * x_range, x.max() + padding * x_range
    y_min, y_max = y.min() - padding * y_range, y.max() + padding * y_range

    # Create grid
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xi, yi)

    # KDE
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])

    if bandwidth is not None:
        kde = gaussian_kde(values, bw_method=bandwidth)
    else:
        kde = gaussian_kde(values)  # Scott's rule

    Z = kde(positions).reshape(X.shape)

    return X, Y, Z


def compute_gradient(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient of the density field.

    Args:
        X, Y: meshgrid coordinates
        Z: density values

    Returns:
        dZ_dx, dZ_dy: gradient components
    """
    # Grid spacing
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # Gradient (numpy gradient returns [dy, dx] for 2D)
    dZ_dy, dZ_dx = np.gradient(Z, dy, dx)

    return dZ_dx, dZ_dy


def find_density_peaks(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    points_2d: np.ndarray,
    titles: Optional[list],
    n_peaks: int = 5,
    min_distance: int = 5
) -> list:
    """
    Find local maxima in density field and map to nearest data points.

    Args:
        X, Y: meshgrid coordinates
        Z: density values
        points_2d: (N, 2) projected data points
        titles: optional labels for points
        n_peaks: maximum number of peaks to return
        min_distance: minimum grid cells between peaks

    Returns:
        List of dicts with peak info: {'x', 'y', 'density', 'title', 'point_idx'}
    """
    from scipy.ndimage import maximum_filter

    # Find local maxima using maximum filter
    neighborhood_size = min_distance * 2 + 1
    local_max = maximum_filter(Z, size=neighborhood_size) == Z

    # Get peak coordinates
    peak_mask = local_max & (Z > Z.mean())  # Only peaks above mean density
    peak_indices = np.argwhere(peak_mask)

    if len(peak_indices) == 0:
        return []

    # Get density values at peaks
    peak_densities = Z[peak_mask]

    # Sort by density (descending) and take top n_peaks
    sorted_idx = np.argsort(-peak_densities)[:n_peaks]

    peaks = []
    for idx in sorted_idx:
        i, j = peak_indices[idx]
        peak_x = X[i, j]
        peak_y = Y[i, j]
        peak_density = Z[i, j]

        # Find nearest data point
        distances = np.sqrt((points_2d[:, 0] - peak_x)**2 + (points_2d[:, 1] - peak_y)**2)
        nearest_idx = np.argmin(distances)

        peak_info = {
            'x': peak_x,
            'y': peak_y,
            'density': peak_density,
            'point_idx': nearest_idx,
            'title': titles[nearest_idx] if titles else f"Point {nearest_idx}"
        }
        peaks.append(peak_info)

    return peaks


def build_tree_for_overlay(
    embeddings: np.ndarray,
    tree_type: Literal['mst', 'j-guided'] = 'mst',
    texts: Optional[list] = None,
    titles: Optional[list] = None
) -> dict:
    """
    Build a tree structure for overlay on the density manifold.

    Args:
        embeddings: (N, D) embedding matrix
        tree_type: 'mst' for minimum spanning tree, 'j-guided' for J-guided tree
        texts: optional texts for J-guided tree entropy
        titles: optional titles for nodes

    Returns:
        dict with 'parent', 'children', 'root' keys
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    # Cosine distance
    similarity = emb_norm @ emb_norm.T
    cos_dist = 1 - similarity
    np.fill_diagonal(cos_dist, 0)

    if tree_type == 'j-guided':
        try:
            from hierarchy_objective import JGuidedTreeBuilder
            builder = JGuidedTreeBuilder(
                embeddings=embeddings,
                texts=texts,
                titles=titles,
                use_bert_entropy=False,
                verbose=False
            )
            builder.build()
            return {
                'parent': builder.parent,
                'children': builder.children,
                'root': [i for i, p in builder.parent.items() if p is None][0]
            }
        except ImportError:
            print("Warning: J-guided tree not available, falling back to MST")
            tree_type = 'mst'

    # MST (default)
    mst = minimum_spanning_tree(cos_dist)
    cx = mst.tocoo()

    # Convert to adjacency
    adj = {}
    for i, j, w in zip(cx.row, cx.col, cx.data):
        if i not in adj:
            adj[i] = []
        if j not in adj:
            adj[j] = []
        adj[i].append((j, w))
        adj[j].append((i, w))

    # Root at highest degree node
    degrees = [(len(adj.get(i, [])), i) for i in range(len(embeddings))]
    _, root = max(degrees)

    # BFS to create rooted tree
    parent = {root: None}
    children = {root: []}
    visited = {root}
    queue = [root]

    while queue:
        node = queue.pop(0)
        for neighbor, _ in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                if node not in children:
                    children[node] = []
                children[node].append(neighbor)
                children[neighbor] = []
                queue.append(neighbor)

    return {'parent': parent, 'children': children, 'root': root}


def get_subtree_nodes(tree: dict, root_node: int) -> set:
    """Get all nodes in subtree rooted at root_node."""
    nodes = {root_node}
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        for child in tree.get('children', {}).get(node, []):
            if child not in nodes:
                nodes.add(child)
                queue.append(child)
    return nodes


def compute_depths(tree: dict) -> dict:
    """Compute depth of each node from root."""
    if 'depth' in tree:
        return tree['depth']

    root = tree.get('root')
    if root is None:
        # Find root
        for node, par in tree['parent'].items():
            if par is None:
                root = node
                break

    depths = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in tree.get('children', {}).get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def draw_tree_overlay(
    ax: plt.Axes,
    points_2d: np.ndarray,
    tree: dict,
    titles: Optional[list] = None,
    color: str = 'cyan',
    linewidth: float = 0.8,
    alpha: float = 0.6,
    draw_root: bool = True,
    show_arrows: bool = False,
    max_depth: Optional[int] = None,
    min_depth: int = 0,
    branch_filter: Optional[str] = None
):
    """
    Draw tree edges on an existing axes.

    Args:
        ax: matplotlib axes to draw on
        points_2d: (N, 2) projected points
        tree: dict with 'parent', 'children', 'root' keys
        titles: optional node titles for branch filtering
        color: edge color
        linewidth: edge width
        alpha: edge transparency
        draw_root: whether to highlight the root node
        show_arrows: whether to draw arrows on edges
        max_depth: only show edges where child depth <= max_depth
        min_depth: only show edges where child depth >= min_depth
        branch_filter: only show subtree rooted at node matching this title
    """
    # Compute depths
    depths = compute_depths(tree)

    # Find branch root if filtering
    subtree_nodes = None
    branch_root = None
    if branch_filter and titles:
        # Find node matching branch filter (partial match)
        branch_filter_lower = branch_filter.lower()
        for idx, title in enumerate(titles):
            if branch_filter_lower in title.lower():
                branch_root = idx
                subtree_nodes = get_subtree_nodes(tree, branch_root)
                break
        if subtree_nodes is None:
            print(f"Warning: No node found matching '{branch_filter}'")

    # Collect edges with filtering
    edges = []
    for node, par in tree['parent'].items():
        if par is None:
            continue

        node_depth = depths.get(node, 0)

        # Depth filtering
        if max_depth is not None and node_depth > max_depth:
            continue
        if node_depth < min_depth:
            continue

        # Branch filtering
        if subtree_nodes is not None and node not in subtree_nodes:
            continue

        edges.append({
            'start': (points_2d[par, 0], points_2d[par, 1]),
            'end': (points_2d[node, 0], points_2d[node, 1]),
            'depth': node_depth
        })

    # Draw edges
    if edges:
        if show_arrows:
            # Draw individual arrows
            for edge in edges:
                dx = edge['end'][0] - edge['start'][0]
                dy = edge['end'][1] - edge['start'][1]
                ax.annotate(
                    '', xy=edge['end'], xytext=edge['start'],
                    arrowprops=dict(
                        arrowstyle='->', color=color, lw=linewidth,
                        alpha=alpha, shrinkA=0, shrinkB=2
                    ),
                    zorder=6
                )
        else:
            # Use LineCollection for efficiency
            lines = [[e['start'], e['end']] for e in edges]
            lc = LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha, zorder=6)
            ax.add_collection(lc)

    # Highlight root (or branch root if filtering)
    display_root = branch_root if branch_root is not None else tree.get('root')
    if draw_root and display_root is not None:
        root_depth = depths.get(display_root, 0)
        if (max_depth is None or root_depth <= max_depth) and root_depth >= min_depth:
            ax.scatter(
                points_2d[display_root, 0], points_2d[display_root, 1],
                c='cyan', s=150, marker='s', edgecolors='white',
                linewidths=2, zorder=12, label='Root'
            )


def plot_density_manifold(
    points_2d: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    dZ_dx: np.ndarray,
    dZ_dy: np.ndarray,
    titles: Optional[list] = None,
    singular_values: Optional[np.ndarray] = None,
    n_contours: int = 20,
    show_contours: bool = True,
    streamline_density: float = 1.0,
    show_gradients: bool = True,
    show_points: bool = True,
    label_peaks: bool = False,
    n_peaks: int = 5,
    tree: Optional[dict] = None,
    tree_color: str = 'cyan',
    tree_arrows: bool = False,
    tree_max_depth: Optional[int] = None,
    tree_min_depth: int = 0,
    tree_branch: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Density Manifold"
) -> plt.Figure:
    """
    Create the density manifold visualization.

    Args:
        points_2d: (N, 2) projected points
        X, Y: meshgrid coordinates
        Z: density values
        dZ_dx, dZ_dy: gradient components
        titles: optional labels for points
        singular_values: for axis labels
        n_contours: number of contour levels
        show_contours: whether to show contour lines
        streamline_density: density of gradient streamlines
        show_gradients: whether to show gradient streamlines
        show_points: whether to scatter plot the data points
        label_peaks: whether to label density peaks with nearest point titles
        n_peaks: number of peaks to label (default 5)
        tree: optional tree dict with 'parent' key to overlay
        tree_color: color for tree edges
        tree_arrows: whether to show arrows on tree edges
        tree_max_depth: only show tree edges up to this depth
        tree_min_depth: only show tree edges at or below this depth
        tree_branch: only show subtree matching this title
        figsize: figure size
        title: plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Contour plot (filled) - always show for density background
    contourf = ax.contourf(X, Y, Z, levels=n_contours, cmap='viridis', alpha=0.7)

    # Contour lines (optional)
    if show_contours:
        ax.contour(X, Y, Z, levels=n_contours, colors='white', linewidths=0.5, alpha=0.5)

    # Gradient streamlines (optional)
    if show_gradients:
        # Normalize gradient for better visualization
        magnitude = np.sqrt(dZ_dx**2 + dZ_dy**2)
        dZ_dx_norm = dZ_dx / (magnitude + 1e-10)
        dZ_dy_norm = dZ_dy / (magnitude + 1e-10)

        # Streamplot shows flow toward density peaks
        ax.streamplot(
            X, Y, dZ_dx_norm, dZ_dy_norm,
            color=magnitude,
            cmap='hot',
            density=streamline_density,
            linewidth=0.8,
            arrowsize=0.8
        )

    # Scatter points
    if show_points:
        ax.scatter(
            points_2d[:, 0], points_2d[:, 1],
            c='red', s=20, alpha=0.7, edgecolors='white', linewidths=0.5,
            zorder=5, label=f'Data points (n={len(points_2d)})'
        )

    # Label density peaks
    if label_peaks and titles:
        peaks = find_density_peaks(X, Y, Z, points_2d, titles, n_peaks=n_peaks)
        for peak in peaks:
            # Draw marker at peak
            ax.scatter(peak['x'], peak['y'], c='yellow', s=100, marker='*',
                      edgecolors='black', linewidths=0.5, zorder=10)
            # Draw label with background
            ax.annotate(
                peak['title'][:30] + ('...' if len(peak['title']) > 30 else ''),
                xy=(peak['x'], peak['y']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                zorder=11
            )

    # Tree overlay
    if tree is not None:
        draw_tree_overlay(
            ax, points_2d, tree,
            titles=titles,
            color=tree_color,
            show_arrows=tree_arrows,
            max_depth=tree_max_depth,
            min_depth=tree_min_depth,
            branch_filter=tree_branch
        )

    # Colorbar for density
    cbar = plt.colorbar(contourf, ax=ax, label='Density')

    # Labels
    if singular_values is not None:
        var_explained = singular_values**2 / (singular_values**2).sum() * 100
        ax.set_xlabel(f'SVD Component 1 ({var_explained[0]:.1f}% variance)')
        ax.set_ylabel(f'SVD Component 2 ({var_explained[1]:.1f}% variance)')
    else:
        ax.set_xlabel('SVD Component 1')
        ax.set_ylabel('SVD Component 2')

    ax.set_title(title)
    ax.legend(loc='upper right')

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def plot_multiple_bandwidths(
    points_2d: np.ndarray,
    bandwidths: list,
    titles: Optional[list] = None,
    singular_values: Optional[np.ndarray] = None,
    grid_size: int = 100,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Plot density manifolds at multiple bandwidth scales.

    Args:
        points_2d: (N, 2) projected points
        bandwidths: list of bandwidth values to compare
        titles: optional labels for points
        singular_values: for axis labels
        grid_size: grid resolution
        figsize: figure size

    Returns:
        matplotlib Figure with subplots
    """
    n_bw = len(bandwidths)
    cols = min(3, n_bw)
    rows = (n_bw + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_bw == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, bw in enumerate(bandwidths):
        ax = axes[idx]

        # Compute density at this bandwidth
        X, Y, Z = compute_density_grid(points_2d, bandwidth=bw, grid_size=grid_size)
        dZ_dx, dZ_dy = compute_gradient(X, Y, Z)

        # Contours
        contourf = ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z, levels=15, colors='white', linewidths=0.3, alpha=0.5)

        # Gradient streamlines
        magnitude = np.sqrt(dZ_dx**2 + dZ_dy**2)
        dZ_dx_norm = dZ_dx / (magnitude + 1e-10)
        dZ_dy_norm = dZ_dy / (magnitude + 1e-10)

        ax.streamplot(
            X, Y, dZ_dx_norm, dZ_dy_norm,
            color='orange', density=0.8, linewidth=0.5, arrowsize=0.6
        )

        # Points
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c='red', s=10, alpha=0.5)

        ax.set_title(f'Bandwidth = {bw}')
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for idx in range(n_bw, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Density Manifolds at Different Smoothing Scales', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize density manifolds of embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--embeddings', '-e', type=Path,
        default=PROJECT_ROOT / 'datasets/wikipedia_physics.npz',
        help='Path to embeddings .npz file'
    )
    parser.add_argument(
        '--bandwidth', '-b', type=float, default=None,
        help='KDE bandwidth (None for automatic Scott\'s rule)'
    )
    parser.add_argument(
        '--multi-bandwidth', '-m', nargs='+', type=float,
        help='Plot multiple bandwidths for comparison (e.g., -m 0.05 0.1 0.2 0.5)'
    )
    parser.add_argument(
        '--grid-size', '-g', type=int, default=100,
        help='Grid resolution for density computation'
    )
    parser.add_argument(
        '--n-contours', '-c', type=int, default=20,
        help='Number of contour levels'
    )
    parser.add_argument(
        '--no-contours', action='store_true',
        help='Hide contour lines'
    )
    parser.add_argument(
        '--streamline-density', '-s', type=float, default=1.0,
        help='Density of gradient streamlines'
    )
    parser.add_argument(
        '--no-gradients', action='store_true',
        help='Hide gradient streamlines'
    )
    parser.add_argument(
        '--no-points', action='store_true',
        help='Hide data points'
    )
    parser.add_argument(
        '--label-peaks', '-l', action='store_true',
        help='Label density peaks with nearest point titles'
    )
    parser.add_argument(
        '--n-peaks', '-p', type=int, default=5,
        help='Number of peaks to label (default: 5)'
    )
    parser.add_argument(
        '--tree', '-t', type=str, choices=['mst', 'j-guided'],
        help='Overlay tree structure (mst or j-guided)'
    )
    parser.add_argument(
        '--tree-color', type=str, default='cyan',
        help='Color for tree edges (default: cyan)'
    )
    parser.add_argument(
        '--arrows', action='store_true',
        help='Show arrows on tree edges (parent→child direction)'
    )
    parser.add_argument(
        '--max-depth', type=int, default=None,
        help='Only show tree edges up to this depth from root'
    )
    parser.add_argument(
        '--min-depth', type=int, default=0,
        help='Only show tree edges at or below this depth (default: 0)'
    )
    parser.add_argument(
        '--branch', type=str, default=None,
        help='Show only the subtree rooted at node with this title (partial match)'
    )
    parser.add_argument(
        '--output', '-o', type=Path,
        help='Save figure to file (PNG, PDF, etc.)'
    )
    parser.add_argument(
        '--top-k', '-k', type=int, default=None,
        help='Use only top-k embeddings'
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for saved figure'
    )

    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings, titles = load_embeddings(args.embeddings)

    if args.top_k:
        embeddings = embeddings[:args.top_k]
        titles = titles[:args.top_k] if titles else None

    print(f"  Shape: {embeddings.shape}")

    # Project to 2D via SVD
    print("Projecting to 2D via SVD...")
    points_2d, V_2d, singular_values = project_to_svd_2d(embeddings)

    var_explained = singular_values**2
    total_var = var_explained.sum()
    print(f"  Top 2 singular values: {singular_values}")
    print(f"  Variance explained: {var_explained[0]/total_var*100:.1f}%, {var_explained[1]/total_var*100:.1f}%")

    if args.multi_bandwidth:
        # Multi-bandwidth comparison
        print(f"Computing density at bandwidths: {args.multi_bandwidth}")
        fig = plot_multiple_bandwidths(
            points_2d,
            args.multi_bandwidth,
            titles=titles,
            singular_values=singular_values,
            grid_size=args.grid_size
        )
    else:
        # Single bandwidth
        bw_str = f"{args.bandwidth}" if args.bandwidth else "auto (Scott's rule)"
        print(f"Computing density (bandwidth={bw_str})...")
        X, Y, Z = compute_density_grid(
            points_2d,
            bandwidth=args.bandwidth,
            grid_size=args.grid_size
        )

        print("Computing gradient field...")
        dZ_dx, dZ_dy = compute_gradient(X, Y, Z)

        # Build tree if requested
        tree = None
        if args.tree:
            print(f"Building {args.tree} tree...")
            tree = build_tree_for_overlay(embeddings, tree_type=args.tree, titles=titles)
            print(f"  Root: {titles[tree['root']] if titles else tree['root']}")

        print("Plotting...")
        fig = plot_density_manifold(
            points_2d, X, Y, Z, dZ_dx, dZ_dy,
            titles=titles,
            singular_values=singular_values,
            n_contours=args.n_contours,
            show_contours=not args.no_contours,
            streamline_density=args.streamline_density,
            show_gradients=not args.no_gradients,
            show_points=not args.no_points,
            label_peaks=args.label_peaks,
            n_peaks=args.n_peaks,
            tree=tree,
            tree_color=args.tree_color,
            tree_arrows=args.arrows,
            tree_max_depth=args.max_depth,
            tree_min_depth=args.min_depth,
            tree_branch=args.branch,
            title=f"Density Manifold ({len(embeddings)} points)"
        )

    if args.output:
        print(f"Saving to {args.output}...")
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print("Done.")
    else:
        plt.show()


if __name__ == '__main__':
    main()
