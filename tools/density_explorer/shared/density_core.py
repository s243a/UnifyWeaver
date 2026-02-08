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
            # Search multiple locations for the model file
            import os
            candidates = [
                'models/wikipedia_physics_distance.pt',  # CWD
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'wikipedia_physics_distance.pt'),  # Project root from shared/
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

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

    # Variance explained (relative to total variance, not just top 2)
    var = S[:2] ** 2
    var_total = (S ** 2).sum()
    var_explained = (var / var_total * 100).tolist() if var_total > 0 else [50.0, 50.0]

    return points_2d, S[:2], var_explained


def classical_mds_2d(
    dist_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical MDS: project distance matrix to 2D so Euclidean distances
    approximate the input distances.

    Uses eigendecomposition of the double-centered squared distance matrix.

    Args:
        dist_matrix: (N, N) symmetric distance matrix with zero diagonal

    Returns:
        points_2d: (N, 2) coordinates
        singular_values: top 2 eigenvalues (as proxy for singular values)
        variance_explained: [var1, var2] as percentages of total positive eigenvalue sum
    """
    n = dist_matrix.shape[0]

    # Squared distances
    D2 = dist_matrix ** 2

    # Double centering: B = -0.5 * H @ D2 @ H, where H = I - (1/n) * 11'
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Eigendecompose (B is symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # eigh returns ascending order; we want descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 2 positive eigenvalues
    top2_vals = np.maximum(eigenvalues[:2], 0)
    top2_vecs = eigenvectors[:, :2]

    # Coordinates = eigenvectors * sqrt(eigenvalues)
    points_2d = top2_vecs * np.sqrt(top2_vals)

    # Variance explained: fraction of total positive eigenvalue sum
    positive_sum = eigenvalues[eigenvalues > 0].sum()
    if positive_sum > 0:
        var_explained = (top2_vals / positive_sum * 100).tolist()
    else:
        var_explained = [50.0, 50.0]

    return points_2d, np.sqrt(top2_vals), var_explained


def blend_projections_2d(
    emb_input: np.ndarray,
    emb_output: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend two embedding spaces at the 2D projection level.

    Projects each space to 2D independently via SVD, normalizes coordinates
    to the same scale, then blends: pts = alpha * pts_input + (1-alpha) * pts_output.

    Args:
        emb_input: (N, D_in) raw/input embeddings
        emb_output: (N, D_out) model-transformed embeddings
        alpha: blend factor. 1.0 = pure input, 0.0 = pure output

    Returns:
        points_2d: (N, 2) blended coordinates
        singular_values: weighted blend of singular values
        variance_explained: weighted blend of variance explained
    """
    pts_input, sv_input, var_input = project_to_2d(emb_input)
    pts_output, sv_output, var_output = project_to_2d(emb_output)

    # Normalize both to same scale
    scale_input = np.abs(pts_input).max() + 1e-8
    scale_output = np.abs(pts_output).max() + 1e-8
    pts_input_norm = pts_input / scale_input
    pts_output_norm = pts_output / scale_output

    # Blend 2D positions
    pts_blend = alpha * pts_input_norm + (1 - alpha) * pts_output_norm

    # Weighted singular values and variance
    sv_blend = alpha * sv_input + (1 - alpha) * sv_output
    var_blend = [
        alpha * var_input[0] + (1 - alpha) * var_output[0],
        alpha * var_input[1] + (1 - alpha) * var_output[1],
    ]

    return pts_blend, sv_blend, var_blend


def blend_distance_matrices(
    emb_input: np.ndarray,
    emb_output: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Blend distance matrices from two embedding spaces.

    Computes cosine distance in each space, then blends:
    d = alpha * d_input + (1-alpha) * d_output.

    Args:
        emb_input: (N, D_in) raw/input embeddings
        emb_output: (N, D_out) model-transformed embeddings
        alpha: blend factor. 1.0 = pure input, 0.0 = pure output

    Returns:
        dist_blend: (N, N) blended distance matrix
    """
    dist_input = cosine_distance_matrix(emb_input)
    dist_output = cosine_distance_matrix(emb_output)

    dist_blend = alpha * dist_input + (1 - alpha) * dist_output
    np.fill_diagonal(dist_blend, 0)

    return dist_blend


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix.

    Args:
        embeddings: (N, D) embedding matrix

    Returns:
        dist: (N, N) cosine distance matrix with zero diagonal
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim = normed @ normed.T
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    return dist


def euclidean_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.

    Args:
        embeddings: (N, D) embedding matrix

    Returns:
        dist: (N, N) Euclidean distance matrix with zero diagonal
    """
    sq_norms = np.sum(embeddings ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * embeddings @ embeddings.T
    dist_sq = np.maximum(dist_sq, 0)  # numerical safety
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 0)
    return dist


def compute_custom_distance_matrix(
    embeddings: np.ndarray,
    input_embeddings: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    metric_model=None,
    space_weights: Dict[str, float] = None,
    embedding_alpha: Optional[float] = None,
    power_n: float = 1.0,
) -> np.ndarray:
    """
    Compute L^n power mean of distance matrices from multiple spaces.

    Normalizes each distance matrix to [0,1], raises to power n,
    computes weighted sum, then takes the (1/n)-th root.

    Args:
        embeddings: (N, D) output/projected embeddings
        input_embeddings: (N, D) input embeddings (for blending and learned metric)
        weights: (N, n_basis) transformer blend weights
        metric_model: loaded OrganizationalMetric model
        space_weights: dict mapping space name to raw weight,
            e.g. {"embedding": 3, "weights": 2, "learned": 2, "wiki": 0}
        embedding_alpha: blend alpha for input/output embedding mix
        power_n: exponent for L^n power mean (default 1.0 = linear)

    Returns:
        result: (N, N) custom distance matrix with zero diagonal
    """
    if space_weights is None:
        return cosine_distance_matrix(embeddings)

    dist_matrices = {}

    # 1. Compute each requested distance matrix
    if space_weights.get('embedding', 0) > 0:
        if embedding_alpha is not None and input_embeddings is not None:
            dist_matrices['embedding'] = blend_distance_matrices(
                input_embeddings, embeddings, embedding_alpha
            )
        else:
            dist_matrices['embedding'] = cosine_distance_matrix(embeddings)

    if space_weights.get('weights', 0) > 0 and weights is not None:
        dist_matrices['weights'] = cosine_distance_matrix(weights)

    if space_weights.get('learned', 0) > 0 and weights is not None:
        inp = input_embeddings if input_embeddings is not None else embeddings
        metric_emb = compute_metric_embeddings(
            inp, embeddings, weights, metric_model
        )
        dist_matrices['learned'] = euclidean_distance_matrix(metric_emb)

    if space_weights.get('wiki', 0) > 0:
        inp = input_embeddings if input_embeddings is not None else embeddings
        try:
            dist_matrices['wiki'] = compute_wikipedia_physics_distances(inp)
        except Exception as e:
            print(f"Warning: Wikipedia physics distances failed ({e}), skipping")

    if not dist_matrices:
        return cosine_distance_matrix(embeddings)

    # 2. Normalize each to [0, 1] (max normalization)
    for key in dist_matrices:
        d = dist_matrices[key]
        d_max = d.max()
        if d_max > 0:
            dist_matrices[key] = d / d_max

    # 3. Normalize weights to sum to 1 (only active spaces)
    active_weights = {k: space_weights[k] for k in dist_matrices}
    total = sum(active_weights.values())
    if total > 0:
        active_weights = {k: v / total for k, v in active_weights.items()}
    else:
        n_spaces = len(active_weights)
        active_weights = {k: 1.0 / n_spaces for k in active_weights}

    # 4. Compute L^n power mean
    n = power_n
    first_matrix = next(iter(dist_matrices.values()))

    if abs(n) < 1e-10:
        # Geometric mean: exp(sum(w_i * log(d_i)))
        log_sum = np.zeros_like(first_matrix)
        for key, d in dist_matrices.items():
            w = active_weights[key]
            log_sum += w * np.log(d + 1e-10)
        result = np.exp(log_sum)
    else:
        # General case: (sum(w_i * d_i^n))^(1/n)
        powered_sum = np.zeros_like(first_matrix)
        for key, d in dist_matrices.items():
            w = active_weights[key]
            powered_sum += w * np.power(d + 1e-10, n)
        result = np.power(powered_sum, 1.0 / n)

    np.fill_diagonal(result, 0)
    return result


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


def kde_gradient_hessian(
    x: np.ndarray,
    data: np.ndarray,
    bandwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient and Hessian of Gaussian KDE at point x.

    For Gaussian kernel K(u) = exp(-||u||^2/2):
      grad_f(x) = -(1/(n*h^(d+2))) * sum_i (x-x_i)/h * K((x-x_i)/h)
      H_f(x)    =  (1/(n*h^(d+2))) * sum_i [(x-x_i)(x-x_i)^T/h^2 - I] * K((x-x_i)/h)

    Args:
        x: (d,) point at which to evaluate
        data: (N, d) data points
        bandwidth: KDE bandwidth (scalar)

    Returns:
        gradient: (d,) gradient vector
        hessian: (d, d) Hessian matrix
    """
    n, d = data.shape
    h = bandwidth

    # Scaled differences: (N, d)
    diff = (x[None, :] - data) / h

    # Kernel values: (N,)
    sq_dist = np.sum(diff ** 2, axis=1)
    kernel_vals = np.exp(-0.5 * sq_dist)

    # Normalization constant
    norm = n * h ** (d + 2)

    # Gradient: -(1/norm) * sum_i diff_i * k_i
    gradient = -np.sum(diff * kernel_vals[:, None], axis=0) / norm

    # Hessian: (1/norm) * sum_i [(diff_i @ diff_i^T - I) * k_i]
    # Vectorized: (diff * k)^T @ diff gives sum of outer products weighted by kernel
    weighted_diff = diff * kernel_vals[:, None]  # (N, d)
    hessian = (weighted_diff.T @ diff - np.sum(kernel_vals) * np.eye(d)) / norm

    return gradient, hessian


def convexity_project_2d(
    embeddings: np.ndarray,
    root_idx: int = 0,
    alpha: float = 0.5,
    convexity_metric: str = "determinant",
    subspace_k: int = 20,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Find optimal 2D projection plane blending variance and convexity.

    Works in a reduced SVD subspace for tractability. Uses greedy selection:
    pick direction 1 maximizing blended score, then pick best orthogonal
    direction 2.

    Args:
        embeddings: (N, D) embedding matrix
        root_idx: index of root node for convexity evaluation
        alpha: blend factor. 0.0 = pure SVD (max variance), 1.0 = pure max convexity
        convexity_metric: "determinant", "min_eigenvalue", "trace", "geometric_mean"
        subspace_k: dimensionality of SVD subspace (default 20)
        bandwidth: KDE bandwidth (None for Scott's rule in k-dim)

    Returns:
        points_2d: (N, 2) projected coordinates
        singular_values: top 2 singular values from SVD (for reference)
        variance_explained: [var1, var2] as percentages
        convexity_info: dict with convexity_score, variance_score, svd comparison, etc.
    """
    n_pts, dim = embeddings.shape

    # Clamp root index
    root_idx = max(0, min(root_idx, n_pts - 1))

    # Center and SVD
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Reduce to top-k subspace
    k = min(subspace_k, len(S), n_pts, dim)
    X_k = centered @ Vt[:k].T  # (N, k)
    x_root_k = X_k[root_idx]   # (k,)

    # Bandwidth: Scott's rule in k-dim if not provided
    if bandwidth is None:
        bandwidth = n_pts ** (-1.0 / (k + 4))

    # Compute gradient and Hessian at root in k-dim subspace
    grad_k, H_k = kde_gradient_hessian(x_root_k, X_k, bandwidth)

    # Per-direction scores
    variances = S[:k] ** 2
    var_total = (S ** 2).sum()

    # Convexity: absolute diagonal of Hessian (curvature along each SVD direction)
    hessian_diag = np.abs(np.diag(H_k))
    hessian_trace = np.sum(hessian_diag)

    # Normalized scores [0, 1]
    norm_var = variances / var_total if var_total > 0 else np.ones(k) / k
    norm_conv = hessian_diag / hessian_trace if hessian_trace > 1e-10 else np.zeros(k)

    # Fall back to pure SVD if Hessian is flat
    if hessian_trace < 1e-10:
        j1, j2 = 0, 1
    else:
        # Blended score
        w_var = 1.0 - alpha
        w_conv = alpha
        scores = w_var * norm_var + w_conv * norm_conv

        # Greedy pick direction 1
        j1 = int(np.argmax(scores))

        # Greedy pick direction 2 (orthogonal — SVD directions are already orthogonal)
        scores_remaining = scores.copy()
        scores_remaining[j1] = -np.inf
        j2 = int(np.argmax(scores_remaining))

    # Ensure j1 < j2 for consistent ordering
    if j1 > j2:
        j1, j2 = j2, j1

    # Project all points onto chosen 2D plane
    points_2d = X_k[:, [j1, j2]]

    # Singular values for chosen directions
    sv_chosen = S[[j1, j2]]

    # Variance explained
    var_explained = [
        float(S[j1] ** 2 / var_total * 100) if var_total > 0 else 50.0,
        float(S[j2] ** 2 / var_total * 100) if var_total > 0 else 50.0,
    ]

    # Compute convexity metric for chosen plane
    H_2x2 = np.array([[H_k[j1, j1], H_k[j1, j2]],
                       [H_k[j2, j1], H_k[j2, j2]]])
    eigvals_2x2 = np.linalg.eigvalsh(H_2x2)

    conv_score = _compute_convexity_score(eigvals_2x2, convexity_metric)

    # Reference: SVD plane (directions 0, 1) convexity for comparison
    H_svd = np.array([[H_k[0, 0], H_k[0, 1]],
                       [H_k[1, 0], H_k[1, 1]]])
    svd_eigvals = np.linalg.eigvalsh(H_svd)
    svd_conv = _compute_convexity_score(svd_eigvals, convexity_metric)
    svd_var = float((S[0] ** 2 + S[1] ** 2) / var_total * 100) if var_total > 0 else 100.0

    convexity_info = {
        "convexity_score": float(conv_score),
        "variance_score": float(var_explained[0] + var_explained[1]),
        "svd_convexity": float(svd_conv),
        "svd_variance": svd_var,
        "direction_indices": [int(j1), int(j2)],
        "hessian_eigenvalues": [float(v) for v in np.sort(np.linalg.eigvalsh(H_k))[::-1][:6]],
    }

    return points_2d, sv_chosen, var_explained, convexity_info


def _compute_convexity_score(eigvals: np.ndarray, metric: str) -> float:
    """Compute convexity score from 2x2 projected Hessian eigenvalues."""
    e0, e1 = abs(eigvals[0]), abs(eigvals[1])
    if metric == "determinant":
        return e0 * e1
    elif metric == "min_eigenvalue":
        return min(e0, e1)
    elif metric == "trace":
        return e0 + e1
    elif metric == "geometric_mean":
        return float(np.sqrt(e0 * e1))
    else:
        return e0 * e1  # default to determinant


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
    metric_model=None,
    max_branching: Optional[int] = None,
    root_id: Optional[int] = None,
    dist_matrix_override: Optional[np.ndarray] = None
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
        max_branching: max children per node (None for unlimited)

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
    if dist_matrix_override is not None:
        dist_matrix = dist_matrix_override
        tree_type = tree_type + '-blended' if 'blended' not in tree_type else tree_type
    elif use_direct_distances:
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

    # Root selection: use explicit root_id if provided, else highest degree
    if root_id is not None and 0 <= root_id < len(embeddings):
        root = root_id
    else:
        degrees = [(len(adj.get(i, [])), i) for i in range(len(embeddings))]
        _, root = max(degrees)

    # BFS to build tree with optional branching limits
    n = len(embeddings)
    parent = {root: None}
    depth = {root: 0}
    child_count = {}
    visited = {root}
    queue = [root]

    while queue:
        node = queue.pop(0)
        # Sort neighbors by weight (closest first) for branching limit
        neighbors = sorted(adj.get(node, []), key=lambda x: x[1])
        for neighbor, w in neighbors:
            if neighbor not in visited:
                if max_branching is not None and child_count.get(node, 0) >= max_branching:
                    break
                visited.add(neighbor)
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                child_count[node] = child_count.get(node, 0) + 1
                queue.append(neighbor)

    # Phase 2: Attach orphaned nodes (unreachable due to full parents)
    # using distance matrix fallback
    while len(visited) < n:
        best_dist = np.inf
        best_from, best_to = -1, -1
        for v in visited:
            if max_branching is not None and child_count.get(v, 0) >= max_branching:
                continue
            for u in range(n):
                if u not in visited and dist_matrix[v, u] < best_dist:
                    best_dist = dist_matrix[v, u]
                    best_from, best_to = v, u
        if best_to == -1:
            break
        visited.add(best_to)
        parent[best_to] = best_from
        depth[best_to] = depth[best_from] + 1
        child_count[best_from] = child_count.get(best_from, 0) + 1
        queue.append(best_to)

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
    k_neighbors: int = 10,
    max_branching: Optional[int] = None,
    root_id: Optional[int] = None,
    dist_matrix_override: Optional[np.ndarray] = None
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
        max_branching: max children per node (None for unlimited)

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

    # Compute distance matrix
    if dist_matrix_override is not None:
        dist_matrix = dist_matrix_override.copy()
        np.fill_diagonal(dist_matrix, np.inf)
        tree_type = tree_type + '-blended' if 'blended' not in tree_type else tree_type
    elif not use_direct_distances:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vec_norm = vectors / (norms + 1e-8)
        similarity = vec_norm @ vec_norm.T
        dist_matrix = 1 - similarity
        np.fill_diagonal(dist_matrix, np.inf)

    # Find k-nearest neighbors for each node
    knn_indices = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]

    # Build tree using greedy nearest-neighbor approach
    # Start from explicit root_id if provided, else most central node
    if root_id is not None and 0 <= root_id < n:
        root = root_id
    else:
        if use_direct_distances:
            # For direct distance matrices, centrality = inverse of total distance
            finite_dists = np.where(dist_matrix == np.inf, 0, dist_matrix)
            centrality = -finite_dists.sum(axis=1)  # Negate so argmax gives smallest total distance
        else:
            centrality = similarity.sum(axis=1)
        root = int(np.argmax(centrality))

    visited = {root}
    parent = {root: None}
    depth = {root: 0}
    child_count = {}  # Track children per node for branching limits
    queue = [root]

    while len(visited) < n and queue:
        # Get next node to process
        current = queue.pop(0)

        # Find unvisited neighbors, prioritized by distance
        for neighbor in knn_indices[current]:
            if neighbor not in visited:
                # Check branching limit
                if max_branching is not None and child_count.get(current, 0) >= max_branching:
                    break
                visited.add(neighbor)
                parent[neighbor] = current
                depth[neighbor] = depth[current] + 1
                child_count[current] = child_count.get(current, 0) + 1
                queue.append(neighbor)

        # If queue is empty but not all visited, find closest unvisited to any visited
        if not queue and len(visited) < n:
            best_dist = np.inf
            best_pair = None
            for v in visited:
                # Skip nodes that have reached branching limit
                if max_branching is not None and child_count.get(v, 0) >= max_branching:
                    continue
                for u in range(n):
                    if u not in visited and dist_matrix[v, u] < best_dist:
                        best_dist = dist_matrix[v, u]
                        best_pair = (v, u)

            if best_pair:
                v, u = best_pair
                visited.add(u)
                parent[u] = v
                depth[u] = depth[v] + 1
                child_count[v] = child_count.get(v, 0) + 1
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


def build_density_greedy_tree(
    embeddings: np.ndarray,
    points_2d: np.ndarray,
    titles: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
    distance_metric: str = "embedding",
    input_embeddings: Optional[np.ndarray] = None,
    metric_model=None,
    k_neighbors: int = 10,
    max_branching: Optional[int] = None,
    root_id: Optional[int] = None,
    dist_matrix_override: Optional[np.ndarray] = None
) -> TreeStructure:
    """
    Build tree using density-ordered greedy insertion.

    This algorithm produces better semantic hierarchies because dense nodes
    (in high-probability regions / convex contours) naturally become hubs.

    Algorithm:
    1. Compute density for each node (inverse of k-th neighbor distance)
    2. Sort nodes by density (densest first)
    3. Place root first (user-selected or densest node)
    4. For each remaining node (in density order), connect to nearest already-placed node

    Args:
        embeddings: (N, D) embedding vectors (output/projected)
        points_2d: (N, 2) projected coordinates for visualization
        titles: optional node labels
        weights: (N, n_basis) blend weights
        distance_metric: "embedding", "weights", "learned", or "wikipedia_physics"
        input_embeddings: (N, D) input embeddings (for "learned" metric)
        metric_model: loaded OrganizationalMetric model
        k_neighbors: number of neighbors for density estimation
        max_branching: max children per node (None for unlimited)
        root_id: optional user-selected root (defaults to densest node)

    Returns:
        TreeStructure with density-greedy edges
    """
    # Choose distance space
    use_direct_distances = False

    if distance_metric == "wikipedia_physics":
        try:
            dist_matrix = compute_wikipedia_physics_distances(embeddings)
            np.fill_diagonal(dist_matrix, np.inf)
            use_direct_distances = True
            tree_type = 'density-greedy-wikipedia-physics'
        except Exception as e:
            print(f"Warning: Wikipedia physics distance failed ({e}), falling back to embedding")
            vectors = embeddings
            tree_type = 'density-greedy'
    elif distance_metric == "learned" and weights is not None and input_embeddings is not None:
        try:
            vectors = compute_metric_embeddings(
                input_embeddings, embeddings, weights, metric_model
            )
            tree_type = 'density-greedy-learned'
        except Exception as e:
            print(f"Warning: Learned metric failed ({e}), falling back to weights")
            vectors = weights
            tree_type = 'density-greedy-weights'
    elif distance_metric == "weights" and weights is not None:
        vectors = weights
        tree_type = 'density-greedy-weights'
    else:
        vectors = embeddings
        tree_type = 'density-greedy'

    n = len(embeddings)

    # Compute distance matrix
    if dist_matrix_override is not None:
        dist_matrix = dist_matrix_override.copy()
        np.fill_diagonal(dist_matrix, np.inf)
        tree_type = tree_type + '-blended' if 'blended' not in tree_type else tree_type
    elif not use_direct_distances:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vec_norm = vectors / (norms + 1e-8)
        similarity = vec_norm @ vec_norm.T
        dist_matrix = 1 - similarity
        np.fill_diagonal(dist_matrix, np.inf)

    # Step 1: Compute k-NN density for each node
    # density = 1 / (distance to k-th nearest neighbor)
    k_actual = min(k_neighbors, n - 1)
    density = np.zeros(n)
    for i in range(n):
        dists_sorted = np.sort(dist_matrix[i])
        # k-th neighbor (index k_actual-1 since 0-indexed, and first is self=inf after fill_diagonal)
        # Actually after fill_diagonal, self is inf, so sorted puts it last
        # We want k-th smallest non-inf distance
        finite_dists = dists_sorted[dists_sorted < np.inf]
        if len(finite_dists) >= k_actual:
            density[i] = 1.0 / (finite_dists[k_actual - 1] + 1e-8)
        else:
            density[i] = 1.0 / (finite_dists[-1] + 1e-8) if len(finite_dists) > 0 else 0

    # Step 2: Sort nodes by density (descending)
    order = np.argsort(-density)

    # Step 3: Determine root - user-selected or densest
    if root_id is None:
        root = int(order[0])  # Densest node
    else:
        root = root_id

    # Step 4: Place root first, then process remaining in density order
    placed = {root}
    parent = {root: None}
    depth_map = {root: 0}
    child_count = {}

    # Remove root from order (it's already placed)
    remaining = [int(idx) for idx in order if idx != root]

    # Step 5: Greedy insertion - each node connects to nearest placed node
    for idx in remaining:
        min_dist = np.inf
        best_parent = root

        for placed_node in placed:
            # Skip nodes that have reached branching limit
            if max_branching is not None and child_count.get(placed_node, 0) >= max_branching:
                continue
            d = dist_matrix[idx, placed_node]
            if d < min_dist:
                min_dist = d
                best_parent = placed_node

        placed.add(idx)
        parent[idx] = best_parent
        depth_map[idx] = depth_map[best_parent] + 1
        child_count[best_parent] = child_count.get(best_parent, 0) + 1

    # Build nodes and edges
    nodes = []
    edges = []

    for i in range(n):
        nodes.append(TreeNode(
            id=i,
            title=titles[i] if titles else f"Node {i}",
            parent_id=parent.get(i),
            depth=depth_map.get(i, 0),
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
    metric_model=None,
    max_branching: Optional[int] = None,
    root_id: Optional[int] = None,
    tree_embeddings: Optional[np.ndarray] = None,
    blend_layout_alpha: Optional[float] = None,
    blend_tree_alpha: Optional[float] = None,
    custom_viz_space_weights: Optional[Dict[str, float]] = None,
    custom_viz_power_n: float = 1.0,
    custom_tree_space_weights: Optional[Dict[str, float]] = None,
    custom_tree_power_n: float = 1.0,
    convexity_alpha: float = 0.5,
    convexity_metric: str = "determinant",
    convexity_subspace_k: int = 20,
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
        max_branching: max children per node in tree (None for unlimited)

    Returns:
        DensityManifoldData ready for frontend
    """
    # Resolve root_id if given as title string
    if isinstance(root_id, str) and titles is not None:
        for idx, t in enumerate(titles):
            if t == root_id:
                root_id = idx
                break
        else:
            root_id = None  # Title not found

    # Convexity info (populated only for convexity_blend mode)
    convexity_info = None

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
    elif projection_mode == "wikipedia_physics_mds":
        # MDS on predicted distances: 2D layout where Euclidean distances ≈ predicted distances
        inp_emb = input_embeddings if input_embeddings is not None else embeddings
        try:
            dist_matrix = compute_wikipedia_physics_distances(inp_emb)
            points_2d, singular_values, var_explained = classical_mds_2d(dist_matrix)
        except Exception as e:
            print(f"Warning: Wikipedia physics MDS failed ({e}), falling back to embedding")
            points_2d, singular_values, var_explained = project_to_2d(
                embeddings, mode="embedding"
            )
            projection_mode = "embedding"
    elif projection_mode == "convexity_blend":
        # Convexity-blended projection: find 2D plane balancing variance and convexity at root
        effective_root = root_id if isinstance(root_id, int) else 0
        points_2d, singular_values, var_explained, convexity_info = convexity_project_2d(
            embeddings,
            root_idx=effective_root,
            alpha=convexity_alpha,
            convexity_metric=convexity_metric,
            subspace_k=convexity_subspace_k,
            bandwidth=bandwidth,
        )
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
        # Check for visualization customization (L^n multi-space)
        if (custom_viz_space_weights is not None
                and projection_mode == "embedding"):
            custom_viz_dist = compute_custom_distance_matrix(
                embeddings=embeddings,
                input_embeddings=input_embeddings,
                weights=weights,
                metric_model=metric_model,
                space_weights=custom_viz_space_weights,
                embedding_alpha=blend_layout_alpha,
                power_n=custom_viz_power_n,
            )
            points_2d, singular_values, var_explained = classical_mds_2d(custom_viz_dist)
        # Check for layout blending (input↔output space)
        elif (blend_layout_alpha is not None and input_embeddings is not None
                and projection_mode == "embedding"):
            points_2d, singular_values, var_explained = blend_projections_2d(
                input_embeddings, embeddings, blend_layout_alpha
            )
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

        # Use separate tree_embeddings if provided (dual embedding mode)
        # Otherwise use the same embeddings as for projection
        tree_emb = tree_embeddings if tree_embeddings is not None else embeddings

        # Compute tree distance matrix override
        dist_matrix_for_tree = None
        if custom_tree_space_weights is not None:
            # L^n multi-space tree distance
            dist_matrix_for_tree = compute_custom_distance_matrix(
                embeddings=tree_emb,
                input_embeddings=input_embeddings,
                weights=weights,
                metric_model=metric_model,
                space_weights=custom_tree_space_weights,
                embedding_alpha=blend_tree_alpha,
                power_n=custom_tree_power_n,
            )
        elif blend_tree_alpha is not None and input_embeddings is not None:
            dist_matrix_for_tree = blend_distance_matrices(
                input_embeddings, tree_emb, blend_tree_alpha
            )

        if tree_type in ('j-guided', 'jguided'):
            tree = build_jguided_tree(
                tree_emb, points_2d, titles,
                weights=weights,
                distance_metric=tree_distance_metric,
                input_embeddings=inp_emb,
                metric_model=metric_model,
                max_branching=max_branching,
                root_id=root_id,
                dist_matrix_override=dist_matrix_for_tree
            )
        elif tree_type in ('density-greedy', 'density_greedy'):
            tree = build_density_greedy_tree(
                tree_emb, points_2d, titles,
                weights=weights,
                distance_metric=tree_distance_metric,
                input_embeddings=inp_emb,
                metric_model=metric_model,
                max_branching=max_branching,
                root_id=root_id,
                dist_matrix_override=dist_matrix_for_tree
            )
        else:
            # Default to MST
            tree = build_mst_tree(
                tree_emb, points_2d, titles,
                weights=weights,
                distance_metric=tree_distance_metric,
                input_embeddings=inp_emb,
                metric_model=metric_model,
                max_branching=max_branching,
                root_id=root_id,
                dist_matrix_override=dist_matrix_for_tree
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
            mode=projection_mode,
            extra=convexity_info,
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
