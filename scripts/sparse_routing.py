#!/usr/bin/env python3
"""
Sparse routing for memory-efficient embedding projections.

Implements hierarchical routing over cluster representatives instead of all items:
1. Route over centroids to select top-K clusters
2. Route over cluster representatives (centroid + trees + SVD-generated)
3. Compute weighted projection using only selected W matrices

Memory savings: 90%+ reduction compared to loading all embeddings.

See proposals/MEMORY_SAVINGS_HARVESTING.md for details.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from collections import OrderedDict


@dataclass
class RepresentativeConfig:
    """Configuration for cluster representative selection."""

    # Hard memory limit
    max_reps_per_cluster: int = 50        # Never exceed this many representatives

    # Subspace coverage target
    condition_threshold: float = 10.0      # Defines "dominant" subspace (κ ≤ threshold)
    subspace_multiple: float = 1.0         # Target N× the dominant subspace dimension

    # Priority weights (for ordering within trees)
    prioritize_by_size: bool = True        # Larger trees first
    prioritize_by_depth: bool = False      # Shallower trees first

    # Small cluster optimization
    use_real_for_small: bool = True        # Use real embeddings if cluster is small


def compute_target_reps(svd_S: np.ndarray, condition_threshold: float = 10.0,
                        subspace_multiple: float = 1.0) -> int:
    """Compute target number of representatives based on subspace dimension.

    Args:
        svd_S: Singular values of cluster (descending order)
        condition_threshold: Defines dominant subspace (κ ≤ threshold)
        subspace_multiple: Multiplier for target count

    Returns:
        target_reps: Target number of representatives
    """
    if len(svd_S) == 0:
        return 1

    # Count dominant singular values
    threshold = svd_S[0] / condition_threshold
    k_dominant = np.sum(svd_S >= threshold)

    # Apply multiplier
    target = int(np.ceil(k_dominant * subspace_multiple))

    return max(1, target)  # At least 1


def select_representatives(
    centroid: np.ndarray,
    tree_embeddings: Optional[np.ndarray],
    svd_Vt: Optional[np.ndarray],
    svd_S: Optional[np.ndarray],
    config: RepresentativeConfig,
    all_cluster_embeddings: Optional[np.ndarray] = None,
    cluster_size: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Select representatives respecting configuration limits.

    Priority order: centroid → trees → SVD-generated (or real embeddings if small)
    Stops when: target reached OR max_reps hit OR cluster exhausted

    Args:
        centroid: Cluster centroid (dim,)
        tree_embeddings: Tree node embeddings (n_trees, dim) or None
        svd_Vt: SVD right singular vectors (k, dim) or None
        svd_S: Singular values (k,) or None
        config: RepresentativeConfig with selection parameters
        all_cluster_embeddings: All embeddings in cluster (for small cluster optimization)
        cluster_size: Number of items in cluster

    Returns:
        representatives: Selected representatives (n_reps, dim)
        info: Dict with selection statistics
    """
    # Compute target based on subspace dimension
    if svd_S is not None and len(svd_S) > 0:
        target_reps = compute_target_reps(
            svd_S,
            config.condition_threshold,
            config.subspace_multiple
        )
    else:
        target_reps = config.max_reps_per_cluster

    # Cap by hard limit
    target_reps = min(target_reps, config.max_reps_per_cluster)

    # Cap by cluster size - don't generate more than we have
    if cluster_size is not None:
        target_reps = min(target_reps, cluster_size)

    # Small cluster optimization: if target >= cluster size, just use real embeddings
    if (config.use_real_for_small and
        cluster_size is not None and
        target_reps >= cluster_size and
        all_cluster_embeddings is not None):
        info = {
            'n_total': len(all_cluster_embeddings),
            'n_centroid': 0,
            'n_trees': 0,
            'n_svd': 0,
            'n_real': len(all_cluster_embeddings),
            'used_real': True,
        }
        return all_cluster_embeddings, info

    representatives = [centroid]  # Priority 1: always include centroid

    # Priority 2: Add trees
    if tree_embeddings is not None and len(tree_embeddings) > 0:
        for tree_emb in tree_embeddings:
            if len(representatives) >= target_reps:
                break
            representatives.append(tree_emb)

    # Priority 3: Fill remaining with SVD-generated
    n_svd_added = 0
    if svd_Vt is not None and svd_S is not None and len(svd_S) > 0:
        while len(representatives) < target_reps:
            weights = np.random.dirichlet(svd_S + 1e-8)  # Avoid zero
            rep = weights @ svd_Vt
            representatives.append(rep)
            n_svd_added += 1

    representatives = np.array(representatives)

    info = {
        'n_total': len(representatives),
        'n_centroid': 1,
        'n_trees': min(len(tree_embeddings) if tree_embeddings is not None else 0,
                       len(representatives) - 1),
        'n_svd': n_svd_added,
        'n_real': 0,
        'used_real': False,
        'target': target_reps,
    }

    return representatives, info


class LazyWMatrixLoader:
    """Load W matrices on-demand with LRU cache.

    Uses memory-mapped loading for fast access without loading entire file.
    """

    def __init__(self, w_stack_path: Path, cache_size: int = 10):
        """Initialize lazy loader.

        Args:
            w_stack_path: Path to npz file containing W_stack
            cache_size: Maximum number of W matrices to keep in cache
        """
        self.w_stack_path = Path(w_stack_path)
        self.cache_size = cache_size
        self._w_stack = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _ensure_loaded(self):
        """Lazy load the memory-mapped W_stack."""
        if self._w_stack is None:
            # Use mmap_mode='r' for memory-efficient read-only access
            data = np.load(self.w_stack_path, mmap_mode='r')
            self._w_stack = data['W_stack']

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return shape of W_stack (n_clusters, dim, dim)."""
        self._ensure_loaded()
        return self._w_stack.shape

    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        self._ensure_loaded()
        return self._w_stack.shape[0]

    def load(self, cluster_id: int) -> np.ndarray:
        """Load W matrix for cluster, using cache if available.

        Args:
            cluster_id: Cluster index

        Returns:
            W matrix (dim, dim)
        """
        if cluster_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cluster_id)
            return self._cache[cluster_id]

        self._ensure_loaded()

        # Load from mmap (copy to allow mmap to be released)
        W = self._w_stack[cluster_id].copy()

        # LRU cache management
        if len(self._cache) >= self.cache_size:
            # Remove oldest (first) item
            self._cache.popitem(last=False)

        self._cache[cluster_id] = W
        return W

    def preload(self, cluster_ids: List[int]):
        """Preload multiple W matrices into cache.

        Args:
            cluster_ids: List of cluster IDs to preload
        """
        for cid in cluster_ids[:self.cache_size]:
            self.load(cid)

    def memory_usage_mb(self) -> float:
        """Return current cache memory usage in MB."""
        if not self._cache:
            return 0.0
        # Assume all matrices same size
        sample = next(iter(self._cache.values()))
        return len(self._cache) * sample.nbytes / 1024 / 1024

    def clear_cache(self):
        """Clear the cache to free memory."""
        self._cache.clear()


def compute_routing_weights(
    query_emb: np.ndarray,
    cluster_reps: Dict[int, np.ndarray],
    cluster_ids: List[int],
    temperature: float = 1.0,
) -> Dict[int, float]:
    """Compute softmax routing weights over clusters.

    Args:
        query_emb: Query vector (dim,)
        cluster_reps: Dict[cluster_id -> representatives array (n_reps, dim)]
        cluster_ids: List of cluster IDs to route over
        temperature: Softmax temperature (lower = sharper)

    Returns:
        weights: Dict[cluster_id -> weight]
    """
    query_norm = np.linalg.norm(query_emb) + 1e-8

    # Compute raw scores for each cluster
    raw_scores = {}
    for cid in cluster_ids:
        reps = cluster_reps[cid]
        rep_norms = np.linalg.norm(reps, axis=1) + 1e-8

        # Sum of exp(cosine / temperature) over all representatives
        cosines = (reps @ query_emb) / (rep_norms * query_norm)
        raw_scores[cid] = np.sum(np.exp(cosines / temperature))

    # Softmax normalization
    total = sum(raw_scores.values())
    weights = {cid: score / total for cid, score in raw_scores.items()}

    return weights


def compute_projection_sparse(
    query_emb: np.ndarray,
    weights: Dict[int, float],
    w_loader: LazyWMatrixLoader,
    top_k: int = 5,
    weight_threshold: float = 0.01,
) -> np.ndarray:
    """Compute projection loading only significant W matrices.

    Args:
        query_emb: Query vector (dim,)
        weights: Dict[cluster_id -> weight] from routing
        w_loader: LazyWMatrixLoader for W matrices
        top_k: Maximum W matrices to load
        weight_threshold: Minimum weight to consider

    Returns:
        projected: Projected vector (dim,)
    """
    # Select top-K clusters by weight, filtering by threshold
    sorted_clusters = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    selected = [(cid, w) for cid, w in sorted_clusters[:top_k] if w >= weight_threshold]

    if not selected:
        # Fallback: use top cluster even if below threshold
        selected = [sorted_clusters[0]]

    # Renormalize weights for selected clusters
    selected_total = sum(w for _, w in selected)

    # Weighted projection
    projected = np.zeros_like(query_emb)
    for cid, weight in selected:
        W = w_loader.load(cid)
        projected += (weight / selected_total) * (query_emb @ W)

    return projected


def select_top_k_clusters(
    query_emb: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 5,
) -> List[int]:
    """Select top-K clusters by centroid similarity.

    Args:
        query_emb: Query vector (dim,)
        centroids: Cluster centroids (n_clusters, dim)
        top_k: Number of clusters to select

    Returns:
        cluster_ids: List of top-K cluster indices
    """
    query_norm = np.linalg.norm(query_emb) + 1e-8
    centroid_norms = np.linalg.norm(centroids, axis=1) + 1e-8

    # Cosine similarity to all centroids
    similarities = (centroids @ query_emb) / (centroid_norms * query_norm)

    # Select top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices.tolist()


class SparseRouter:
    """Full sparse routing pipeline for memory-efficient projections."""

    def __init__(
        self,
        centroids_path: Path,
        w_matrices_path: Path,
        representatives_path: Optional[Path] = None,
        config: Optional[RepresentativeConfig] = None,
        top_k_clusters: int = 5,
        top_k_projection: int = 5,
        w_cache_size: int = 10,
    ):
        """Initialize sparse router.

        Args:
            centroids_path: Path to centroids npz (must have 'centroids' key)
            w_matrices_path: Path to W matrices npz (must have 'W_stack' key)
            representatives_path: Path to pre-computed representatives (optional)
            config: RepresentativeConfig for on-demand representative generation
            top_k_clusters: Number of clusters to consider in routing
            top_k_projection: Number of W matrices to use in projection
            w_cache_size: LRU cache size for W matrices
        """
        self.centroids = np.load(centroids_path)['centroids']
        self.w_loader = LazyWMatrixLoader(w_matrices_path, cache_size=w_cache_size)
        self.config = config or RepresentativeConfig()
        self.top_k_clusters = top_k_clusters
        self.top_k_projection = top_k_projection

        # Load pre-computed representatives if available
        self._representatives: Optional[Dict[int, np.ndarray]] = None
        if representatives_path and representatives_path.exists():
            self._load_representatives(representatives_path)

    def _load_representatives(self, path: Path):
        """Load pre-computed representatives from file."""
        data = np.load(path, allow_pickle=True)
        self._representatives = {
            int(k): v for k, v in data.items()
        }

    def route(self, query_emb: np.ndarray) -> Dict[int, float]:
        """Route query to clusters, returning weights.

        Args:
            query_emb: Query vector (dim,)

        Returns:
            weights: Dict[cluster_id -> weight]
        """
        # Step 1: Select top-K clusters by centroid similarity
        cluster_ids = select_top_k_clusters(
            query_emb, self.centroids, self.top_k_clusters
        )

        # Step 2: Get representatives for selected clusters
        if self._representatives is not None:
            cluster_reps = {cid: self._representatives[cid] for cid in cluster_ids}
        else:
            # Fallback: use centroids only (fast but less accurate)
            cluster_reps = {cid: self.centroids[cid:cid+1] for cid in cluster_ids}

        # Step 3: Compute routing weights
        weights = compute_routing_weights(query_emb, cluster_reps, cluster_ids)

        return weights

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query using sparse routing.

        Args:
            query_emb: Query vector (dim,)

        Returns:
            projected: Projected vector (dim,)
        """
        weights = self.route(query_emb)
        return compute_projection_sparse(
            query_emb, weights, self.w_loader, self.top_k_projection
        )

    def project_batch(self, query_embs: np.ndarray) -> np.ndarray:
        """Project batch of queries.

        Args:
            query_embs: Query vectors (batch_size, dim)

        Returns:
            projected: Projected vectors (batch_size, dim)
        """
        return np.array([self.project(q) for q in query_embs])

    def memory_usage_mb(self) -> Dict[str, float]:
        """Return memory usage breakdown."""
        return {
            'centroids': self.centroids.nbytes / 1024 / 1024,
            'w_cache': self.w_loader.memory_usage_mb(),
            'representatives': (
                sum(r.nbytes for r in self._representatives.values()) / 1024 / 1024
                if self._representatives else 0
            ),
        }


if __name__ == '__main__':
    # Example usage
    print("Sparse Routing Module")
    print("=" * 40)

    # Test RepresentativeConfig
    config = RepresentativeConfig(
        max_reps_per_cluster=30,
        condition_threshold=10.0,
        subspace_multiple=1.5,
    )
    print(f"Config: {config}")

    # Test compute_target_reps
    test_S = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.01])
    target = compute_target_reps(test_S, condition_threshold=10.0, subspace_multiple=1.0)
    print(f"Test singular values: {test_S}")
    print(f"Target reps (1× subspace): {target}")

    target_2x = compute_target_reps(test_S, condition_threshold=10.0, subspace_multiple=2.0)
    print(f"Target reps (2× subspace): {target_2x}")
