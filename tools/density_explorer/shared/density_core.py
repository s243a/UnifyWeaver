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

# Cached organizational metric model
_organizational_metric = None

# Cached Wikipedia physics distance model
_wikipedia_physics_distance = None


def load_organizational_metric(model_path: str = None):
    """Load the learned organizational metric model."""
    global _organizational_metric

    if _organizational_metric is not None:
        return _organizational_metric

    try:
        import torch
        import torch.nn as nn

        if model_path is None:
            model_path = 'models/organizational_metric.pt'

        # Define model class inline to avoid import issues
        class OrganizationalMetric(nn.Module):
            def __init__(self, embed_dim=768, weight_dim=64, hidden_dim=256, output_dim=64, weight_power=2.0):
                super().__init__()
                self.weight_power = weight_power
                input_dim = embed_dim * 2 + weight_dim + 1
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim)
                )

            def encode(self, input_emb, output_emb, weights):
                probs = torch.softmax(weights, dim=-1)
                w = probs ** self.weight_power
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
                p = probs + 1e-8
                entropy = -(p * torch.log(p)).sum(dim=-1, keepdim=True)
                x = torch.cat([input_emb, output_emb, w, entropy], dim=-1)
                return self.encoder(x)

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = OrganizationalMetric(
            embed_dim=checkpoint.get('embed_dim', 768),
            weight_dim=checkpoint.get('weight_dim', 64),
            hidden_dim=checkpoint.get('hidden_dim', 256),
            output_dim=checkpoint.get('output_dim', 64)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        _organizational_metric = model
        return model

    except Exception as e:
        print(f"Warning: Could not load organizational metric: {e}")
        return None


def compute_metric_embeddings(
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    weights: np.ndarray,
    metric_model=None,
    batch_size: int = 256
) -> np.ndarray:
    """
    Compute embeddings in the learned organizational metric space.

    Args:
        input_embeddings: (N, 768) query embeddings
        output_embeddings: (N, 768) projected embeddings
        weights: (N, 64) transformer blend weights
        metric_model: loaded OrganizationalMetric model
        batch_size: batch size for encoding

    Returns:
        metric_embeddings: (N, 64) embeddings in learned metric space
    """
    import torch

    if metric_model is None:
        metric_model = load_organizational_metric()

    if metric_model is None:
        raise ValueError("Organizational metric model not available")

    device = next(metric_model.parameters()).device
    n_samples = len(input_embeddings)
    metric_emb = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            inp = torch.from_numpy(input_embeddings[i:i+batch_size].astype(np.float32)).to(device)
            out = torch.from_numpy(output_embeddings[i:i+batch_size].astype(np.float32)).to(device)
            w = torch.from_numpy(weights[i:i+batch_size].astype(np.float32)).to(device)

            enc = metric_model.encode(inp, out, w)
            metric_emb.append(enc.cpu().numpy())

    return np.concatenate(metric_emb, axis=0)


def load_wikipedia_physics_distance(model_path: str = None):
    """Load the Wikipedia physics distance model."""
    global _wikipedia_physics_distance

    if _wikipedia_physics_distance is not None:
        return _wikipedia_physics_distance

    try:
        import torch
        import torch.nn as nn

        if model_path is None:
            model_path = 'models/wikipedia_physics_distance.pt'

        # Define model class matching the trained architecture
        class WikipediaPhysicsDistance(nn.Module):
            def __init__(self, embed_dim=768, hidden_dim=256, proj_dim=64):
                super().__init__()
                self.query_proj = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, proj_dim)
                )
                self.target_proj = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, proj_dim)
                )
                self.distance_head = nn.Sequential(
                    nn.Linear(proj_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )

            def forward(self, query_emb, target_emb):
                q = self.query_proj(query_emb)
                t = self.target_proj(target_emb)
                combined = torch.cat([q, t], dim=-1)
                return self.distance_head(combined).squeeze(-1)

            def get_query_projection(self, query_emb):
                """Get 64-dim projection for visualization."""
                return self.query_proj(query_emb)

            def get_target_projection(self, target_emb):
                """Get 64-dim projection for visualization."""
                return self.target_proj(target_emb)

        # Load state dict directly (not wrapped in checkpoint)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Handle both wrapped and unwrapped formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        model = WikipediaPhysicsDistance()
        model.load_state_dict(state_dict)
        model.eval()

        _wikipedia_physics_distance = model
        return model

    except Exception as e:
        print(f"Warning: Could not load Wikipedia physics distance model: {e}")
        return None


def compute_wikipedia_physics_distances(
    embeddings: np.ndarray,
    model=None,
    batch_size: int = 256
) -> np.ndarray:
    """
    Compute pairwise distances using Wikipedia physics distance model.

    Args:
        embeddings: (N, 768) Nomic embeddings
        model: loaded WikipediaPhysicsDistance model
        batch_size: batch size for computation

    Returns:
        distances: (N, N) pairwise distance matrix
    """
    import torch

    if model is None:
        model = load_wikipedia_physics_distance()

    if model is None:
        raise ValueError("Wikipedia physics distance model not available")

    device = next(model.parameters()).device
    n = len(embeddings)
    distances = np.zeros((n, n), dtype=np.float32)

    emb_tensor = torch.from_numpy(embeddings.astype(np.float32)).to(device)

    with torch.no_grad():
        for i in range(n):
            # Compute distances from point i to all points
            query = emb_tensor[i:i+1].expand(n, -1)  # (N, 768)
            dists = model(query, emb_tensor)  # (N,)
            distances[i] = dists.cpu().numpy()

    # Make symmetric (average of d(i,j) and d(j,i))
    distances = (distances + distances.T) / 2
    # Ensure diagonal is zero
    np.fill_diagonal(distances, 0)

    return distances


def compute_wikipedia_physics_projections(
    embeddings: np.ndarray,
    model=None,
    batch_size: int = 256
) -> np.ndarray:
    """
    Get 64-dim projections from Wikipedia physics distance model for visualization.

    Args:
        embeddings: (N, 768) Nomic embeddings
        model: loaded WikipediaPhysicsDistance model
        batch_size: batch size for computation

    Returns:
        projections: (N, 64) projected embeddings
    """
    import torch

    if model is None:
        model = load_wikipedia_physics_distance()

    if model is None:
        raise ValueError("Wikipedia physics distance model not available")

    device = next(model.parameters()).device
    n = len(embeddings)
    projections = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size].astype(np.float32)).to(device)
            # Use query projection (could also average query and target projections)
            proj = model.get_query_projection(batch)
            projections.append(proj.cpu().numpy())

    return np.concatenate(projections, axis=0)


def load_embeddings(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load embeddings from .npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = data['embeddings']
    titles = list(data['titles']) if 'titles' in data else None
    return embeddings, titles


def project_to_2d(
    embeddings: np.ndarray,
    mode: str = "embedding",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project to 2D via SVD.

    Args:
        embeddings: (N, d) embeddings
        mode: "embedding" (default) or "weights" (use transformer blend weights)
        weights: (N, n_basis) transformer blend weights (required for "weights" mode)

    Returns:
        points_2d: (N, 2) coordinates
        singular_values: top 2 singular values
        variance_explained: [var1, var2] as percentages
    """
    if mode == "weights" and weights is not None:
        # Project in weight space - shows which queries use similar transformation recipes
        return project_weights_to_2d(weights)

    # Default: project in embedding space
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


def project_weights_to_2d(
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project transformer blend weights to 2D via SVD.

    Each query has a weight vector (n_basis,) that determines how much
    of each bivector/bias to use. Projecting these to 2D shows which
    queries use similar transformation recipes.

    Args:
        weights: (N, n_basis) blend weights from transformer

    Returns:
        points_2d: (N, 2) coordinates in weight space
        singular_values: top 2 singular values
        variance_explained: [var1, var2] as percentages
    """
    # Center
    mean = weights.mean(axis=0)
    centered = weights - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Project to top 2 components
    V_2d = Vt[:2].T
    points_2d = centered @ V_2d

    # Variance explained
    var = S[:2] ** 2
    var_total = (S ** 2).sum()
    var_explained = (var / var_total * 100).tolist() if var_total > 0 else [50.0, 50.0]

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
    titles: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
    distance_metric: str = "embedding",
    input_embeddings: Optional[np.ndarray] = None,
    metric_model=None
) -> TreeStructure:
    """
    Build minimum spanning tree.

    Args:
        embeddings: (N, D) embedding vectors (output/projected)
        points_2d: (N, 2) projected coordinates for visualization
        titles: optional node labels
        weights: (N, n_basis) blend weights (for distance_metric="weights" or "learned")
        distance_metric: "embedding", "weights", or "learned"
        input_embeddings: (N, D) input embeddings (for "learned" metric)
        metric_model: loaded OrganizationalMetric model (for "learned" metric)

    Returns:
        TreeStructure with MST edges
    """
    # Choose distance space
    use_direct_distances = False

    if distance_metric == "wikipedia_physics":
        # Wikipedia physics distance model - predicts distances directly
        try:
            dist_matrix = compute_wikipedia_physics_distances(embeddings)
            use_direct_distances = True
            tree_type = 'mst-wikipedia-physics'
        except Exception as e:
            print(f"Warning: Wikipedia physics distance failed ({e}), falling back to embedding")
            vectors = embeddings
            tree_type = 'mst'
    elif distance_metric == "learned" and weights is not None and input_embeddings is not None:
        # Learned organizational metric space
        try:
            metric_emb = compute_metric_embeddings(
                input_embeddings, embeddings, weights, metric_model
            )
            vectors = metric_emb
            tree_type = 'mst-learned'
        except Exception as e:
            print(f"Warning: Learned metric failed ({e}), falling back to weights")
            vectors = weights
            tree_type = 'mst-weights'
    elif distance_metric == "weights" and weights is not None:
        # Weight-space distance: hierarchical/organizational relationships
        vectors = weights
        tree_type = 'mst-weights'
    else:
        # Embedding-space distance: semantic relationships
        vectors = embeddings
        tree_type = 'mst'

    # Compute distance matrix
    if use_direct_distances:
        # Already have distance matrix from model
        pass
    else:
        # Normalize and compute cosine distance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vec_norm = vectors / (norms + 1e-8)
        similarity = vec_norm @ vec_norm.T
        dist_matrix = 1 - similarity
        np.fill_diagonal(dist_matrix, 0)

    # MST
    mst = minimum_spanning_tree(dist_matrix)
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
                weight=float(dist_matrix[i, parent[i]])
            ))

    return TreeStructure(
        nodes=nodes,
        edges=edges,
        root_id=root,
        tree_type=tree_type
    )


def build_jguided_tree(
    embeddings: np.ndarray,
    points_2d: np.ndarray,
    titles: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
    distance_metric: str = "embedding",
    input_embeddings: Optional[np.ndarray] = None,
    metric_model=None,
    k_neighbors: int = 10
) -> TreeStructure:
    """
    Build J-guided tree (local connections based on k-nearest neighbors).

    J-guided trees connect each node to its nearest neighbor that hasn't
    been visited yet, creating more local connections than MST.

    Args:
        embeddings: (N, D) embedding vectors (output/projected)
        points_2d: (N, 2) projected coordinates for visualization
        titles: optional node labels
        weights: (N, n_basis) blend weights
        distance_metric: "embedding", "weights", or "learned"
        input_embeddings: (N, D) input embeddings (for "learned" metric)
        metric_model: loaded OrganizationalMetric model
        k_neighbors: number of neighbors to consider for connections

    Returns:
        TreeStructure with J-guided edges
    """
    # Choose distance space
    use_direct_distances = False

    if distance_metric == "wikipedia_physics":
        # Wikipedia physics distance model - predicts distances directly
        try:
            dist_matrix = compute_wikipedia_physics_distances(embeddings)
            np.fill_diagonal(dist_matrix, np.inf)
            use_direct_distances = True
            tree_type = 'jguided-wikipedia-physics'
        except Exception as e:
            print(f"Warning: Wikipedia physics distance failed ({e}), falling back to embedding")
            vectors = embeddings
            tree_type = 'jguided'
    elif distance_metric == "learned" and weights is not None and input_embeddings is not None:
        try:
            vectors = compute_metric_embeddings(
                input_embeddings, embeddings, weights, metric_model
            )
            tree_type = 'jguided-learned'
        except Exception as e:
            print(f"Warning: Learned metric failed ({e}), falling back to weights")
            vectors = weights
            tree_type = 'jguided-weights'
    elif distance_metric == "weights" and weights is not None:
        vectors = weights
        tree_type = 'jguided-weights'
    else:
        vectors = embeddings
        tree_type = 'jguided'

    n = len(embeddings)

    # Compute distance matrix if not already done
    if not use_direct_distances:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vec_norm = vectors / (norms + 1e-8)
        similarity = vec_norm @ vec_norm.T
        dist_matrix = 1 - similarity
        np.fill_diagonal(dist_matrix, np.inf)

    # Find k-nearest neighbors for each node
    knn_indices = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]

    # Build tree using greedy nearest-neighbor approach
    # Start from the node with highest average similarity to others (most central)
    centrality = similarity.sum(axis=1)
    root = int(np.argmax(centrality))

    visited = {root}
    parent = {root: None}
    depth = {root: 0}
    queue = [root]

    while len(visited) < n and queue:
        # Get next node to process
        current = queue.pop(0)

        # Find unvisited neighbors, prioritized by distance
        for neighbor in knn_indices[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)

        # If queue is empty but not all visited, find closest unvisited to any visited
        if not queue and len(visited) < n:
            best_dist = np.inf
            best_pair = None
            for v in visited:
                for u in range(n):
                    if u not in visited and dist_matrix[v, u] < best_dist:
                        best_dist = dist_matrix[v, u]
                        best_pair = (v, u)

            if best_pair:
                v, u = best_pair
                visited.add(u)
                parent[u] = v
                depth[u] = depth[v] + 1
                queue.append(u)

    # Build nodes and edges
    nodes = []
    edges = []

    for i in range(n):
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
                weight=float(dist_matrix[i, parent[i]] if dist_matrix[i, parent[i]] != np.inf else 1.0)
            ))

    return TreeStructure(
        nodes=nodes,
        edges=edges,
        root_id=root,
        tree_type=tree_type
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


def project_learned_metric_to_2d(
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    weights: np.ndarray,
    metric_model=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project learned organizational metric embeddings to 2D via SVD.

    The OrganizationalMetric model encodes items into a 64D space where
    Euclidean distance = organizational proximity. Projecting this space
    to 2D shows organizational structure rather than semantic similarity.

    Args:
        input_embeddings: (N, 768) query embeddings
        output_embeddings: (N, 768) projected embeddings
        weights: (N, 64) transformer blend weights
        metric_model: loaded OrganizationalMetric model

    Returns:
        points_2d: (N, 2) coordinates in learned metric space
        singular_values: top 2 singular values
        variance_explained: [var1, var2] as percentages
    """
    # Compute embeddings in learned metric space
    metric_emb = compute_metric_embeddings(
        input_embeddings, output_embeddings, weights, metric_model
    )

    # Center
    mean = metric_emb.mean(axis=0)
    centered = metric_emb - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Project to top 2 components
    V_2d = Vt[:2].T
    points_2d = centered @ V_2d

    # Variance explained
    var = S[:2] ** 2
    var_total = (S ** 2).sum()
    var_explained = (var / var_total * 100).tolist() if var_total > 0 else [50.0, 50.0]

    return points_2d, S[:2], var_explained


def compute_density_manifold(
    embeddings: np.ndarray,
    titles: Optional[List[str]] = None,
    bandwidth: Optional[float] = None,
    grid_size: int = 100,
    include_tree: bool = True,
    tree_type: str = 'mst',
    tree_distance_metric: str = "embedding",
    include_peaks: bool = True,
    n_peaks: int = 5,
    projection_mode: str = "embedding",
    weights: Optional[np.ndarray] = None,
    input_embeddings: Optional[np.ndarray] = None,
    metric_model=None
) -> DensityManifoldData:
    """
    Main function: compute complete density manifold data.

    This is the primary entry point used by all frontends.

    Args:
        embeddings: (N, D) embedding matrix (output/projected embeddings)
        titles: optional node labels
        bandwidth: KDE bandwidth (None for auto)
        grid_size: density grid resolution
        include_tree: whether to compute tree overlay
        tree_type: 'mst' or 'j-guided'
        tree_distance_metric: "embedding" (semantic), "weights" (hierarchical), or "learned"
        include_peaks: whether to find density peaks
        n_peaks: number of peaks to find
        projection_mode: "embedding", "weights", or "learned" (organizational metric space)
        weights: (N, n_basis) transformer blend weights
        input_embeddings: (N, D) input embeddings (for "learned" mode/metric)
        metric_model: loaded OrganizationalMetric model (auto-loaded if None)

    Returns:
        DensityManifoldData ready for frontend
    """
    # Project to 2D based on mode
    if projection_mode == "wikipedia_physics":
        # Use Wikipedia physics distance model's hidden layer (64-dim) for visualization
        inp_emb = input_embeddings if input_embeddings is not None else embeddings
        try:
            wiki_projections = compute_wikipedia_physics_projections(inp_emb)
            # SVD on the 64-dim space
            points_2d, singular_values, var_explained = project_to_2d(
                wiki_projections, mode="embedding"
            )
        except Exception as e:
            print(f"Warning: Wikipedia physics projection failed ({e}), falling back to embedding")
            points_2d, singular_values, var_explained = project_to_2d(
                embeddings, mode="embedding"
            )
            projection_mode = "embedding"  # Update mode for metadata
    elif projection_mode == "learned" and weights is not None:
        # Use learned organizational metric space for visualization
        # input_embeddings may be the same as output embeddings if not provided
        inp_emb = input_embeddings if input_embeddings is not None else embeddings
        try:
            points_2d, singular_values, var_explained = project_learned_metric_to_2d(
                inp_emb, embeddings, weights, metric_model
            )
        except Exception as e:
            print(f"Warning: Learned projection failed ({e}), falling back to weights")
            points_2d, singular_values, var_explained = project_to_2d(
                embeddings, mode="weights", weights=weights
            )
            projection_mode = "weights"  # Update mode for metadata
    else:
        points_2d, singular_values, var_explained = project_to_2d(
            embeddings, mode=projection_mode, weights=weights
        )

    # Compute density
    density_grid = compute_density_grid(points_2d, bandwidth, grid_size)

    # Build tree if requested
    tree = None
    if include_tree:
        # For learned distance metric, we need input embeddings
        inp_emb = input_embeddings if input_embeddings is not None else embeddings

        if tree_type in ('j-guided', 'jguided'):
            tree = build_jguided_tree(
                embeddings, points_2d, titles,
                weights=weights,
                distance_metric=tree_distance_metric,
                input_embeddings=inp_emb,
                metric_model=metric_model
            )
        else:
            # Default to MST
            tree = build_mst_tree(
                embeddings, points_2d, titles,
                weights=weights,
                distance_metric=tree_distance_metric,
                input_embeddings=inp_emb,
                metric_model=metric_model
            )

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
            singular_values=singular_values.tolist(),
            mode=projection_mode
        ),
        n_points=len(embeddings)
    )


# Convenience function for loading and computing in one call
def load_and_compute(
    embeddings_path: str,
    top_k: Optional[int] = None,
    weights_path: Optional[str] = None,
    **kwargs
) -> DensityManifoldData:
    """
    Load embeddings and compute density manifold.

    Args:
        embeddings_path: Path to .npz file with embeddings
        top_k: Limit to first k points
        weights_path: Optional path to .npz with blend weights
                      (for projection_mode="weights")
        **kwargs: Passed to compute_density_manifold
    """
    embeddings, titles = load_embeddings(embeddings_path)

    if top_k:
        embeddings = embeddings[:top_k]
        titles = titles[:top_k] if titles else None

    # Load weights if provided
    weights = None
    if weights_path:
        weight_data = np.load(weights_path, allow_pickle=True)
        weights = weight_data.get('weights')
        if top_k and weights is not None:
            weights = weights[:top_k]
        kwargs['weights'] = weights

    return compute_density_manifold(embeddings, titles, **kwargs)
