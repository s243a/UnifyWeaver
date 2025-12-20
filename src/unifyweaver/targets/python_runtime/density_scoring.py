"""
Density-Based Confidence Scoring for Federated KG Queries.

This module implements kernel density estimation and clustering for
semantic consensus detection in distributed search results.

Key concepts:
- Density scoring: Measures how many results are semantically nearby
- Flux-based softmax: Density-weighted probability distribution
- Two-stage pipeline: Cluster first, then density within cluster
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import time
import hashlib
from enum import Enum


class BandwidthMethod(Enum):
    """Methods for kernel bandwidth selection."""
    SILVERMAN = "silverman"
    SCOTT = "scott"
    AUTO = "auto"  # Cross-validation (expensive)
    FIXED = "fixed"


class ClusterMethod(Enum):
    """Methods for semantic clustering."""
    GREEDY = "greedy"  # Simple greedy centroid-based
    HDBSCAN = "hdbscan"  # Hierarchical density-based


# Try to import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


@dataclass
class DensityConfig:
    """Configuration for density estimation."""
    bandwidth_method: BandwidthMethod = BandwidthMethod.SILVERMAN
    fixed_bandwidth: float = 0.1
    density_weight: float = 0.3  # Weight in flux-softmax (0 = ignore density)
    min_cluster_size: int = 2
    clustering_enabled: bool = True
    similarity_threshold: float = 0.7  # For greedy cluster assignment
    normalize_scores: bool = True
    # Phase 4d-ii: HDBSCAN options
    cluster_method: ClusterMethod = ClusterMethod.GREEDY
    hdbscan_min_samples: int = 2  # Core point neighborhood size
    hdbscan_cluster_selection_epsilon: float = 0.0  # Merge clusters below this distance
    # Phase 4d-iii: Adaptive bandwidth options
    use_adaptive_bandwidth: bool = False  # Enable per-point bandwidth
    adaptive_alpha: float = 0.5  # Sensitivity for adaptive bandwidth (0.5 is typical)
    cv_n_candidates: int = 10  # Number of candidates for cross-validation


@dataclass
class DensityResult:
    """Density information for a single result."""
    embedding: np.ndarray
    density_score: float = 0.0
    cluster_id: Optional[int] = None
    cluster_size: int = 1
    is_cluster_center: bool = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    return 1.0 - cosine_similarity(a, b)


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances for a set of embeddings.

    Args:
        embeddings: (n, d) array of n embeddings with dimension d

    Returns:
        (n, n) distance matrix
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Cosine similarity matrix
    similarities = normalized @ normalized.T

    # Cosine distance = 1 - similarity
    distances = 1.0 - similarities

    # Ensure diagonal is 0 (numerical precision)
    np.fill_diagonal(distances, 0.0)

    return distances


def silverman_bandwidth(distances: np.ndarray, n: int) -> float:
    """Compute bandwidth using Silverman's rule of thumb.

    h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)

    Args:
        distances: Flattened array of pairwise distances
        n: Number of points

    Returns:
        Bandwidth value h
    """
    if len(distances) == 0:
        return 0.1

    # Remove zeros (self-distances)
    nonzero = distances[distances > 0]
    if len(nonzero) == 0:
        return 0.1

    sigma = np.std(nonzero)
    q75, q25 = np.percentile(nonzero, [75, 25])
    iqr = q75 - q25

    # Silverman's rule
    scale = min(sigma, iqr / 1.34) if iqr > 0 else sigma
    h = 0.9 * scale * (n ** (-0.2))

    return max(h, 0.01)  # Minimum bandwidth


def scott_bandwidth(distances: np.ndarray, n: int, d: int = 1) -> float:
    """Compute bandwidth using Scott's rule.

    h = n^(-1/(d+4)) * σ

    Args:
        distances: Flattened array of pairwise distances
        n: Number of points
        d: Dimensionality (effective, typically 1 for distance-based)

    Returns:
        Bandwidth value h
    """
    if len(distances) == 0:
        return 0.1

    nonzero = distances[distances > 0]
    if len(nonzero) == 0:
        return 0.1

    sigma = np.std(nonzero)
    h = sigma * (n ** (-1.0 / (d + 4)))

    return max(h, 0.01)


def gaussian_kernel(distance: float, bandwidth: float) -> float:
    """Gaussian kernel function.

    K_h(u) = exp(-u² / 2h²)

    Args:
        distance: Distance value
        bandwidth: Kernel bandwidth h

    Returns:
        Kernel value
    """
    return np.exp(-(distance ** 2) / (2 * bandwidth ** 2))


# =============================================================================
# PHASE 4d-iii: ADAPTIVE BANDWIDTH METHODS
# =============================================================================

def leave_one_out_cv_score(distances: np.ndarray, bandwidth: float) -> float:
    """Compute leave-one-out cross-validation score for KDE bandwidth.

    Uses pseudo-likelihood: sum of log densities computed with each point
    left out.

    Args:
        distances: (n, n) pairwise distance matrix
        bandwidth: Bandwidth to evaluate

    Returns:
        Log-likelihood score (higher is better)
    """
    n = distances.shape[0]
    if n < 2:
        return 0.0

    total_score = 0.0
    for i in range(n):
        # Compute density at point i using all other points
        other_distances = np.delete(distances[i], i)
        kernel_vals = np.exp(-(other_distances ** 2) / (2 * bandwidth ** 2))
        density = kernel_vals.sum() / (n - 1)

        # Add log-likelihood (with small epsilon to avoid log(0))
        if density > 0:
            total_score += np.log(density + 1e-10)

    return total_score


def cross_validation_bandwidth(
    embeddings: np.ndarray,
    n_candidates: int = 10,
    bandwidth_range: Tuple[float, float] = (0.01, 1.0)
) -> float:
    """Select bandwidth via leave-one-out cross-validation.

    Tests multiple bandwidth values and selects the one with highest
    cross-validation score.

    Args:
        embeddings: (n, d) array of embeddings
        n_candidates: Number of bandwidth values to test
        bandwidth_range: (min, max) bandwidth range to search

    Returns:
        Optimal bandwidth
    """
    n = len(embeddings)
    if n < 2:
        return 0.1

    distances = pairwise_cosine_distances(embeddings)

    # Generate candidate bandwidths (log-spaced)
    candidates = np.logspace(
        np.log10(bandwidth_range[0]),
        np.log10(bandwidth_range[1]),
        n_candidates
    )

    best_bandwidth = candidates[0]
    best_score = float('-inf')

    for h in candidates:
        score = leave_one_out_cv_score(distances, h)
        if score > best_score:
            best_score = score
            best_bandwidth = h

    return best_bandwidth


def adaptive_local_bandwidth(
    embeddings: np.ndarray,
    global_bandwidth: float,
    k_neighbors: int = 5,
    alpha: float = 0.5
) -> np.ndarray:
    """Compute adaptive (balloon) bandwidth for each point.

    Uses local density to adjust bandwidth:
    h(x) = h₀ × (p̂_pilot(x))^(-α)

    Points in dense regions get smaller bandwidth (sharper),
    points in sparse regions get larger bandwidth (smoother).

    Args:
        embeddings: (n, d) array of embeddings
        global_bandwidth: Base bandwidth h₀
        k_neighbors: Number of neighbors for local density
        alpha: Sensitivity parameter (typically 0.5)

    Returns:
        (n,) array of per-point bandwidths
    """
    n = len(embeddings)
    if n < 2:
        return np.full(n, global_bandwidth)

    distances = pairwise_cosine_distances(embeddings)

    # Pilot density estimate using global bandwidth
    pilot_densities = np.zeros(n)
    for i in range(n):
        kernel_vals = np.exp(-(distances[i] ** 2) / (2 * global_bandwidth ** 2))
        pilot_densities[i] = kernel_vals.sum() / n

    # Normalize to prevent extreme values
    pilot_densities = np.clip(pilot_densities, 1e-10, None)
    geometric_mean = np.exp(np.mean(np.log(pilot_densities)))

    # Adaptive bandwidth: h(x) = h₀ × (p̂(x) / g)^(-α)
    # where g is geometric mean of pilot densities
    local_bandwidths = global_bandwidth * (pilot_densities / geometric_mean) ** (-alpha)

    # Clamp to reasonable range
    min_h = global_bandwidth * 0.1
    max_h = global_bandwidth * 10
    local_bandwidths = np.clip(local_bandwidths, min_h, max_h)

    return local_bandwidths


def compute_adaptive_density_scores(
    embeddings: np.ndarray,
    config: 'DensityConfig' = None
) -> np.ndarray:
    """Compute density scores using adaptive bandwidth.

    Each point has its own bandwidth based on local density,
    providing better estimates in both dense and sparse regions.

    Args:
        embeddings: (n, d) array of embeddings
        config: Density configuration

    Returns:
        (n,) array of normalized density scores
    """
    if config is None:
        config = DensityConfig()

    n = len(embeddings)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Get global bandwidth
    distances = pairwise_cosine_distances(embeddings)
    flat_distances = distances.flatten()

    if config.bandwidth_method == BandwidthMethod.AUTO:
        global_h = cross_validation_bandwidth(embeddings)
    elif config.bandwidth_method == BandwidthMethod.SCOTT:
        global_h = scott_bandwidth(flat_distances, n)
    elif config.bandwidth_method == BandwidthMethod.FIXED:
        global_h = config.fixed_bandwidth
    else:
        global_h = silverman_bandwidth(flat_distances, n)

    # Compute adaptive bandwidths
    local_bandwidths = adaptive_local_bandwidth(embeddings, global_h)

    # Compute density at each point using its local bandwidth
    densities = np.zeros(n)
    for i in range(n):
        h_i = local_bandwidths[i]
        kernel_vals = np.exp(-(distances[i] ** 2) / (2 * h_i ** 2))
        densities[i] = kernel_vals.sum() / n

    # Normalize to [0, 1]
    if config.normalize_scores and densities.max() > 0:
        densities = densities / densities.max()

    return densities


def compute_density_scores(
    embeddings: np.ndarray,
    config: DensityConfig = None
) -> np.ndarray:
    """Compute density scores for each embedding using KDE.

    density(eᵢ) = (1/n) Σⱼ K_h(d(eᵢ, eⱼ))

    Supports three bandwidth modes:
    - Fixed bandwidth (same for all points)
    - Adaptive bandwidth (per-point based on local density)
    - Cross-validation selected bandwidth

    Args:
        embeddings: (n, d) array of embeddings
        config: Density estimation configuration

    Returns:
        (n,) array of density scores
    """
    if config is None:
        config = DensityConfig()

    n = len(embeddings)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Use adaptive density if configured
    if config.use_adaptive_bandwidth:
        return compute_adaptive_density_scores(embeddings, config)

    # Compute pairwise distances
    distances = pairwise_cosine_distances(embeddings)
    flat_distances = distances[np.triu_indices(n, k=1)]

    # Select bandwidth
    if config.bandwidth_method == BandwidthMethod.FIXED:
        bandwidth = config.fixed_bandwidth
    elif config.bandwidth_method == BandwidthMethod.SCOTT:
        bandwidth = scott_bandwidth(flat_distances, n)
    elif config.bandwidth_method == BandwidthMethod.AUTO:
        bandwidth = cross_validation_bandwidth(
            embeddings, n_candidates=config.cv_n_candidates
        )
    else:  # SILVERMAN (default)
        bandwidth = silverman_bandwidth(flat_distances, n)

    # Compute density for each point
    densities = np.zeros(n)
    for i in range(n):
        # Apply Gaussian kernel to all distances from point i
        kernel_values = gaussian_kernel(distances[i], bandwidth)
        densities[i] = kernel_values.mean()

    # Normalize to [0, 1]
    if config.normalize_scores and densities.max() > 0:
        densities = densities / densities.max()

    return densities


def flux_softmax(
    scores: np.ndarray,
    densities: np.ndarray,
    density_weight: float = 0.3,
    temperature: float = 1.0
) -> np.ndarray:
    """Density-weighted softmax (flux-based).

    P(i) = exp(sᵢ/τ) * (1 + w * dᵢ) / Z

    where:
    - sᵢ is the score for result i
    - dᵢ is the density score for result i
    - w is the density weight
    - τ is the temperature
    - Z is the normalization constant

    The density acts as a "flux" that concentrates probability mass
    in semantically coherent regions.

    Args:
        scores: (n,) array of raw scores
        densities: (n,) array of density scores [0, 1]
        density_weight: How much density affects the distribution
        temperature: Softmax temperature (lower = sharper)

    Returns:
        (n,) array of probability values
    """
    if len(scores) == 0:
        return np.array([])
    if len(scores) == 1:
        return np.array([1.0])

    # Scale scores by temperature
    scaled = scores / temperature

    # Prevent overflow
    scaled = scaled - scaled.max()

    # Density modulation
    flux_factor = 1.0 + density_weight * densities

    # Flux-weighted exponentials
    exp_scores = np.exp(scaled) * flux_factor

    # Normalize
    return exp_scores / exp_scores.sum()


def cluster_by_similarity(
    embeddings: np.ndarray,
    threshold: float = 0.7,
    min_cluster_size: int = 2
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Simple greedy clustering by cosine similarity.

    This mimics the distributed cluster aggregator routing - results
    are assigned to the nearest existing cluster if similarity exceeds
    threshold, otherwise a new cluster is created.

    Args:
        embeddings: (n, d) array of embeddings
        threshold: Minimum similarity to join a cluster
        min_cluster_size: Clusters smaller than this are marked as noise (-1)

    Returns:
        labels: (n,) array of cluster labels (-1 for noise)
        centroids: List of cluster centroid arrays
    """
    n = len(embeddings)
    if n == 0:
        return np.array([]), []
    if n == 1:
        return np.array([-1]), []  # Single point is noise

    labels = np.full(n, -1, dtype=int)
    centroids = []
    cluster_members = []  # List of lists of indices

    for i in range(n):
        emb = embeddings[i]
        best_cluster = -1
        best_similarity = threshold

        # Find best matching cluster
        for c_idx, centroid in enumerate(centroids):
            sim = cosine_similarity(emb, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = c_idx

        if best_cluster >= 0:
            # Join existing cluster
            labels[i] = best_cluster
            cluster_members[best_cluster].append(i)

            # Update centroid (running average)
            members = cluster_members[best_cluster]
            new_centroid = embeddings[members].mean(axis=0)
            centroids[best_cluster] = new_centroid
        else:
            # Create new cluster
            new_label = len(centroids)
            labels[i] = new_label
            centroids.append(emb.copy())
            cluster_members.append([i])

    # Mark small clusters as noise
    for c_idx, members in enumerate(cluster_members):
        if len(members) < min_cluster_size:
            for idx in members:
                labels[idx] = -1

    # Renumber clusters (skip noise)
    unique_labels = sorted(set(labels) - {-1})
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = -1
    labels = np.array([label_map[l] for l in labels])

    # Filter centroids to only valid clusters
    valid_centroids = [centroids[old] for old in unique_labels]

    return labels, valid_centroids


def cluster_by_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 2,
    cluster_selection_epsilon: float = 0.0
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Cluster embeddings using HDBSCAN.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
    with Noise) is a density-based clustering algorithm that:
    - Doesn't require specifying number of clusters
    - Handles noise/outliers naturally
    - Works well with varying cluster densities
    - Produces soft cluster membership probabilities

    Args:
        embeddings: (n, d) array of embeddings
        min_cluster_size: Minimum points to form a cluster
        min_samples: Core point neighborhood size (higher = more conservative)
        cluster_selection_epsilon: Distance threshold for merging clusters

    Returns:
        labels: (n,) array of cluster labels (-1 for noise)
        centroids: List of cluster centroid arrays
    """
    n = len(embeddings)
    if n == 0:
        return np.array([]), []
    if n < min_cluster_size:
        return np.full(n, -1, dtype=int), []

    if not HDBSCAN_AVAILABLE:
        # Fallback to greedy clustering
        return cluster_by_similarity(
            embeddings,
            threshold=0.7,
            min_cluster_size=min_cluster_size
        )

    # Compute cosine distance matrix for HDBSCAN
    # HDBSCAN works better with precomputed distances for cosine
    distances = pairwise_cosine_distances(embeddings)

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='precomputed',
        cluster_selection_method='eom'  # Excess of Mass (default, better for varying sizes)
    )
    labels = clusterer.fit_predict(distances)

    # Compute centroids for each cluster
    unique_labels = sorted(set(labels) - {-1})
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)

    return labels, centroids


def get_hdbscan_probabilities(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Get soft cluster membership probabilities from HDBSCAN.

    Args:
        embeddings: (n, d) array of embeddings
        min_cluster_size: Minimum points to form a cluster
        min_samples: Core point neighborhood size

    Returns:
        labels: (n,) array of cluster labels
        probabilities: (n,) array of cluster membership probabilities
    """
    if not HDBSCAN_AVAILABLE:
        labels, _ = cluster_by_similarity(embeddings, min_cluster_size=min_cluster_size)
        # Assign probability 1.0 for clustered points, 0.0 for noise
        probs = np.where(labels >= 0, 1.0, 0.0)
        return labels, probs

    distances = pairwise_cosine_distances(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed'
    )
    labels = clusterer.fit_predict(distances)
    probabilities = clusterer.probabilities_

    return labels, probabilities


def compute_cluster_density(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    config: DensityConfig = None
) -> np.ndarray:
    """Compute density scores within each cluster.

    Two-stage approach: density is computed relative to cluster membership,
    not globally. This prevents unrelated answers from diluting the signal.

    Args:
        embeddings: (n, d) array of embeddings
        cluster_labels: (n,) array of cluster labels (-1 for noise)
        config: Density configuration

    Returns:
        (n,) array of intra-cluster density scores
    """
    if config is None:
        config = DensityConfig()

    n = len(embeddings)
    densities = np.zeros(n)

    # Get unique cluster labels (excluding noise)
    unique_clusters = set(cluster_labels) - {-1}

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) == 1:
            # Single-member cluster gets density 1.0
            densities[mask] = 1.0
        else:
            # Compute density within cluster
            cluster_densities = compute_density_scores(cluster_embeddings, config)
            densities[mask] = cluster_densities

    # Noise points get density 0 (or could use global density)
    # densities[cluster_labels == -1] = 0.0  # Already zero

    return densities


def two_stage_density_pipeline(
    embeddings: np.ndarray,
    scores: np.ndarray,
    config: DensityConfig = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Complete two-stage density scoring pipeline.

    Stage 1: Cluster results by semantic similarity
    Stage 2: Compute density within each cluster

    Args:
        embeddings: (n, d) array of embeddings
        scores: (n,) array of raw scores
        config: Density configuration

    Returns:
        flux_probs: Density-weighted probabilities
        densities: Per-result density scores
        labels: Cluster labels
        centroids: Cluster centroid embeddings
    """
    if config is None:
        config = DensityConfig()

    n = len(embeddings)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), []
    if n == 1:
        return np.array([1.0]), np.array([1.0]), np.array([-1]), []

    # Stage 1: Cluster
    if config.clustering_enabled:
        if config.cluster_method == ClusterMethod.HDBSCAN:
            labels, centroids = cluster_by_hdbscan(
                embeddings,
                min_cluster_size=config.min_cluster_size,
                min_samples=config.hdbscan_min_samples,
                cluster_selection_epsilon=config.hdbscan_cluster_selection_epsilon
            )
        else:
            # Default: greedy clustering
            labels, centroids = cluster_by_similarity(
                embeddings,
                threshold=config.similarity_threshold,
                min_cluster_size=config.min_cluster_size
            )
    else:
        # No clustering - treat all as one cluster
        labels = np.zeros(n, dtype=int)
        centroids = [embeddings.mean(axis=0)]

    # Stage 2: Intra-cluster density
    densities = compute_cluster_density(embeddings, labels, config)

    # Compute flux-softmax probabilities
    flux_probs = flux_softmax(scores, densities, config.density_weight)

    return flux_probs, densities, labels, centroids


@dataclass
class ClusterStats:
    """Statistics for a semantic cluster."""
    cluster_id: int
    size: int
    centroid: np.ndarray
    avg_density: float
    max_density: float
    variance: float  # Intra-cluster variance


def compute_cluster_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    densities: np.ndarray,
    centroids: List[np.ndarray]
) -> List[ClusterStats]:
    """Compute statistics for each cluster.

    Args:
        embeddings: (n, d) array of embeddings
        labels: (n,) array of cluster labels
        densities: (n,) array of density scores
        centroids: List of cluster centroids

    Returns:
        List of ClusterStats for each cluster
    """
    stats = []
    unique_labels = sorted(set(labels) - {-1})

    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_densities = densities[mask]

        # Compute variance (mean distance from centroid)
        centroid = centroids[i]
        distances = np.array([
            cosine_distance(emb, centroid)
            for emb in cluster_embeddings
        ])
        variance = np.var(distances)

        stats.append(ClusterStats(
            cluster_id=cluster_id,
            size=int(mask.sum()),
            centroid=centroid,
            avg_density=float(cluster_densities.mean()),
            max_density=float(cluster_densities.max()),
            variance=float(variance)
        ))

    return stats


# ============================================================================
# Transaction Management for Distributed Cluster Aggregators
# ============================================================================

@dataclass
class TransactionConfig:
    """Configuration for a federated query transaction."""
    transaction_id: str
    timeout_ms: int = 30000  # 30 second default
    max_aggregators: int = 50  # Prevent runaway spawning
    cleanup_on_complete: bool = True
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        elapsed = (time.time() - self.created_at) * 1000
        return elapsed > self.timeout_ms


@dataclass
class ClusterAggregator:
    """Aggregator for a semantic cluster within a transaction.

    Implements Freenet-style routing - accepts results that match
    its centroid, or redirects to better-matching peers.
    """
    cluster_id: str
    centroid: np.ndarray
    transaction_id: str
    results: List = field(default_factory=list)
    known_peers: Dict[str, np.ndarray] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    _shutdown: bool = False

    def should_accept(self, embedding: np.ndarray, threshold: float = 0.7) -> bool:
        """Check if result belongs to this cluster."""
        if self._shutdown:
            return False
        similarity = cosine_similarity(self.centroid, embedding)
        return similarity >= threshold

    def should_redirect(self, embedding: np.ndarray) -> Optional[str]:
        """Check if another aggregator is better suited for this result."""
        if self._shutdown:
            return None

        my_sim = cosine_similarity(embedding, self.centroid)

        for peer_id, peer_centroid in self.known_peers.items():
            peer_sim = cosine_similarity(embedding, peer_centroid)
            if peer_sim > my_sim + 0.1:  # Significant improvement threshold
                return peer_id

        return None  # Keep locally

    def add_result(self, result) -> bool:
        """Add result and update centroid. Returns False if shutdown."""
        if self._shutdown:
            return False
        self.results.append(result)
        self._update_centroid()
        return True

    def _update_centroid(self):
        """Update centroid as running average of result embeddings."""
        if not self.results:
            return
        # Assumes results have a semantic_centroid attribute
        embeddings = []
        for r in self.results:
            if hasattr(r, 'semantic_centroid') and r.semantic_centroid is not None:
                embeddings.append(r.semantic_centroid)
        if embeddings:
            self.centroid = np.mean(embeddings, axis=0)

    def notify_new_peer(self, peer_id: str, peer_centroid: np.ndarray):
        """Called when a new aggregator joins the transaction."""
        self.known_peers[peer_id] = peer_centroid

    def compute_density(self, config: DensityConfig = None) -> Dict[str, float]:
        """Compute density scores for all results in cluster."""
        if len(self.results) < 2:
            return {str(i): 1.0 for i in range(len(self.results))}

        embeddings = []
        for r in self.results:
            if hasattr(r, 'semantic_centroid') and r.semantic_centroid is not None:
                embeddings.append(r.semantic_centroid)

        if len(embeddings) < 2:
            return {str(i): 1.0 for i in range(len(self.results))}

        embeddings = np.array(embeddings)
        densities = compute_density_scores(embeddings, config)

        return {str(i): float(d) for i, d in enumerate(densities)}

    def get_stats(self) -> Dict:
        """Get cluster statistics."""
        densities = self.compute_density()
        density_values = list(densities.values())
        return {
            'cluster_id': self.cluster_id,
            'transaction_id': self.transaction_id,
            'size': len(self.results),
            'centroid_norm': float(np.linalg.norm(self.centroid)),
            'avg_density': float(np.mean(density_values)) if density_values else 0.0,
            'num_peers': len(self.known_peers),
            'is_active': not self._shutdown
        }

    def shutdown(self, force: bool = False):
        """Shutdown this aggregator."""
        self._shutdown = True

        if not force:
            # Graceful: compute final density before shutdown
            self.compute_density()

        # Clear results to free memory
        self.results.clear()
        self.known_peers.clear()

    def is_active(self) -> bool:
        return not self._shutdown


@dataclass
class AggregatorRegistry:
    """Transaction-scoped registry of cluster aggregators."""
    transaction_id: str
    aggregators: Dict[str, ClusterAggregator] = field(default_factory=dict)
    config: DensityConfig = field(default_factory=DensityConfig)

    def register_aggregator(self, aggregator: ClusterAggregator):
        """Register aggregator and broadcast to peers."""
        self.aggregators[aggregator.cluster_id] = aggregator

        # Broadcast to all other aggregators in this transaction
        for peer_id, peer in self.aggregators.items():
            if peer_id != aggregator.cluster_id:
                peer.notify_new_peer(aggregator.cluster_id, aggregator.centroid)
                aggregator.notify_new_peer(peer_id, peer.centroid)

    def find_best_aggregator(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find closest aggregator by cosine similarity."""
        if not self.aggregators:
            return None, 0.0

        best_id = None
        best_sim = -1.0

        for agg_id, agg in self.aggregators.items():
            if not agg.is_active():
                continue
            sim = cosine_similarity(embedding, agg.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = agg_id

        return best_id, best_sim

    def route_result(self, result, embedding: np.ndarray, max_aggregators: int = 50) -> str:
        """Route result to existing aggregator or spawn new one."""
        best_id, best_sim = self.find_best_aggregator(embedding)

        if best_id and best_sim >= self.config.similarity_threshold:
            # Route to existing aggregator
            self.aggregators[best_id].add_result(result)
            return best_id

        # Check if we can spawn new
        if len(self.aggregators) >= max_aggregators:
            # Force route to closest
            if best_id:
                self.aggregators[best_id].add_result(result)
                return best_id
            # No aggregators and at limit - shouldn't happen
            return ""

        # Spawn new aggregator
        new_id = f"cluster_{len(self.aggregators)}_{int(time.time() * 1000)}"
        new_agg = ClusterAggregator(
            cluster_id=new_id,
            centroid=embedding.copy(),
            transaction_id=self.transaction_id,
            results=[result]
        )
        self.register_aggregator(new_agg)
        return new_id

    def finalize(self) -> List[Dict]:
        """Compute final stats for all aggregators."""
        return [agg.get_stats() for agg in self.aggregators.values()]

    def shutdown_all(self, force: bool = False):
        """Shutdown all aggregators."""
        for agg in self.aggregators.values():
            agg.shutdown(force=force)


class TransactionManager:
    """Manages lifecycle of aggregator transactions."""

    def __init__(self):
        self.transactions: Dict[str, AggregatorRegistry] = {}
        self.configs: Dict[str, TransactionConfig] = {}

    def begin_transaction(
        self,
        transaction_id: str = None,
        timeout_ms: int = 30000,
        max_aggregators: int = 50,
        density_config: DensityConfig = None
    ) -> AggregatorRegistry:
        """Start a new aggregator transaction."""
        if transaction_id is None:
            transaction_id = hashlib.sha256(
                f"{time.time()}_{id(self)}".encode()
            ).hexdigest()[:16]

        config = TransactionConfig(
            transaction_id=transaction_id,
            timeout_ms=timeout_ms,
            max_aggregators=max_aggregators
        )

        registry = AggregatorRegistry(
            transaction_id=transaction_id,
            config=density_config or DensityConfig()
        )

        self.transactions[transaction_id] = registry
        self.configs[transaction_id] = config

        return registry

    def get_transaction(self, transaction_id: str) -> Optional[AggregatorRegistry]:
        """Get existing transaction registry."""
        return self.transactions.get(transaction_id)

    def close_transaction(self, transaction_id: str) -> Optional[List[Dict]]:
        """Explicitly close transaction and collect results."""
        if transaction_id not in self.transactions:
            return None

        registry = self.transactions[transaction_id]

        # Collect final stats from all aggregators
        results = registry.finalize()

        # Shutdown all aggregators
        registry.shutdown_all(force=False)

        # Remove from tracking
        del self.transactions[transaction_id]
        del self.configs[transaction_id]

        return results

    def kill_transaction(self, transaction_id: str):
        """Force-kill a transaction without collecting results."""
        if transaction_id not in self.transactions:
            return

        registry = self.transactions[transaction_id]

        # Immediate shutdown
        registry.shutdown_all(force=True)

        del self.transactions[transaction_id]
        del self.configs[transaction_id]

    def cleanup_expired(self) -> List[str]:
        """Clean up expired transactions. Returns list of killed transaction IDs."""
        expired = [
            tid for tid, config in self.configs.items()
            if config.is_expired()
        ]

        for tid in expired:
            self.kill_transaction(tid)

        return expired

    def get_stats(self) -> Dict:
        """Get overall transaction manager stats."""
        return {
            'active_transactions': len(self.transactions),
            'total_aggregators': sum(
                len(reg.aggregators) for reg in self.transactions.values()
            ),
            'transaction_ids': list(self.transactions.keys())
        }


# Global transaction manager instance
_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """Get or create the global transaction manager."""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager
