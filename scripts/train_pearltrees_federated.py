#!/usr/bin/env python3
"""
Federated Pearltrees Projection Training.

Clusters targets by materialized path depth or output embedding similarity,
then trains per-query Procrustes transforms within each cluster.

At inference:
1. Route query to cluster(s) via softmax
2. Within cluster, route to specific transform via softmax
3. Project and search

See docs/design/FEDERATED_MODEL_FORMAT.md for the output file format specification.
"""

import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the python_runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

from minimal_transform import compute_minimal_transform
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix


def suggest_cluster_count(embeddings: np.ndarray, method: str = "effective_rank", target_variance: float = 0.80) -> dict:
    """
    Suggest optimal cluster count based on embedding structure.

    Args:
        embeddings: Data embeddings (N x D)
        method: One of 'effective_rank', 'covering', 'sqrt_n', 'auto'
        target_variance: Variance threshold for covering method (e.g., 0.80 for 80%)

    Returns:
        Dict with 'suggested_k' and analysis details
    """
    N, D = embeddings.shape

    # SVD analysis
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)

    # 1. Effective rank (participation ratio)
    effective_rank = int((np.sum(S)**2) / np.sum(S**2))

    # 2. Covering number (2^r for target variance)
    r = int(np.searchsorted(cumvar, target_variance) + 1)
    covering_k = 2 ** min(r, 10)  # Cap at 2^10=1024

    # 3. √N heuristic
    sqrt_k = int(np.ceil(np.sqrt(N)))

    suggestions = {
        'effective_rank': effective_rank,
        'covering': covering_k,
        'sqrt_n': sqrt_k,
    }

    # Select based on method
    if method == "effective_rank":
        suggested_k = effective_rank
    elif method == "covering":
        suggested_k = covering_k
    elif method == "sqrt_n":
        suggested_k = sqrt_k
    elif method == "auto":
        suggested_k = effective_rank  # Default to effective rank
    else:
        suggested_k = effective_rank

    return {
        'suggested_k': suggested_k,
        'method': method,
        'target_variance': target_variance,
        'all_suggestions': suggestions,
        'variance_dims': {
            'r_50pct': int(np.searchsorted(cumvar, 0.50) + 1),
            'r_80pct': int(np.searchsorted(cumvar, 0.80) + 1),
            'r_90pct': int(np.searchsorted(cumvar, 0.90) + 1),
            f'r_{int(target_variance*100)}pct': r,
        }
    }


class SimpleEmbedder:
    """Wraps SentenceTransformer for embedding."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


@dataclass
class ClusterModel:
    """Model for a single cluster."""
    cluster_id: str
    W_stack: np.ndarray  # Per-query transforms (N, 768, 768)
    centroids: np.ndarray  # Query embeddings for routing (N, 768)
    target_embeddings: np.ndarray  # Target embeddings (N, 768)
    indices: List[int]  # Original indices in full dataset
    temperature: float = 0.1


@dataclass
class FederatedModel:
    """Complete federated model."""
    clusters: List[ClusterModel]
    cluster_centroids: np.ndarray  # Cluster centroids for first-level routing
    cluster_ids: List[str]
    temperature: float = 0.1
    global_target_ids: List[str] = field(default_factory=list)
    global_target_titles: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


def cluster_by_path_depth(data: List[Dict], max_clusters: int = 10) -> Dict[str, List[int]]:
    """
    Cluster targets by materialized path depth.
    
    Path depth = number of '/' in the path portion of target_text.
    """
    depth_groups = defaultdict(list)
    
    for i, d in enumerate(data):
        target = d.get("target_text", "")
        # Extract path (first line)
        path_line = target.split('\n')[0] if target else ""
        depth = path_line.count('/')
        depth_groups[depth].append(i)
    
    # If too many depth levels, merge small groups
    sorted_depths = sorted(depth_groups.keys())
    
    if len(sorted_depths) <= max_clusters:
        return {f"depth_{d}": indices for d, indices in depth_groups.items()}
    
    # Merge into max_clusters groups by combining adjacent depths
    items_per_cluster = len(data) / max_clusters
    clusters = {}
    current_cluster = []
    current_name = None
    cluster_idx = 0
    
    for depth in sorted_depths:
        if current_name is None:
            current_name = f"depth_{depth}"
        current_cluster.extend(depth_groups[depth])
        
        if len(current_cluster) >= items_per_cluster:
            clusters[f"cluster_{cluster_idx}"] = current_cluster
            current_cluster = []
            current_name = None
            cluster_idx += 1
    
    if current_cluster:
        clusters[f"cluster_{cluster_idx}"] = current_cluster
    
    return clusters


def cluster_by_tree(data: List[Dict]) -> Dict[str, List[int]]:
    """
    Cluster targets by their parent tree (cluster_id field).

    Each unique cluster_id (parent tree URI) becomes its own cluster.
    Items without cluster_id are grouped into "no_parent".

    This creates one cluster per Pearltrees folder, grouping items
    with their siblings (items sharing the same parent tree).
    """
    tree_groups = defaultdict(list)

    for i, d in enumerate(data):
        cluster_id = d.get("cluster_id", "")
        if cluster_id:
            # Use the tree ID from the URI as the cluster name
            # e.g., "https://www.pearltrees.com/.../id12345" -> "tree_12345"
            if "/id" in cluster_id:
                tree_id = cluster_id.split("/id")[-1].split("/")[0]
                cluster_name = f"tree_{tree_id}"
            else:
                cluster_name = f"tree_{hash(cluster_id) % 1000000}"
        else:
            cluster_name = "no_parent"

        tree_groups[cluster_name].append(i)

    logger.info(f"Created {len(tree_groups)} tree-based clusters")

    # Log distribution stats
    sizes = [len(indices) for indices in tree_groups.values()]
    logger.info(f"  Min size: {min(sizes)}, Max size: {max(sizes)}, Avg: {np.mean(sizes):.1f}")

    return dict(tree_groups)


def cluster_by_embedding(A_emb: np.ndarray, max_clusters: int = 10, max_items_per_cluster: int = 1000) -> Dict[str, List[int]]:
    """
    Cluster targets by output embedding similarity.
    Uses K-means then splits large clusters to enforce size constraint.
    """
    from sklearn.cluster import KMeans
    
    n = len(A_emb)
    
    # Start with more clusters than needed, will merge small ones
    initial_clusters = max(max_clusters, n // max_items_per_cluster + 1)
    initial_clusters = min(initial_clusters, n)  # Can't have more clusters than items
    
    logger.info(f"Initial K-means with {initial_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(A_emb)
    
    # Group by label
    initial_groups = defaultdict(list)
    for i, label in enumerate(labels):
        initial_groups[label].append(i)
    
    # Split large clusters, merge small ones
    final_clusters = {}
    cluster_idx = 0
    
    for label, indices in initial_groups.items():
        if len(indices) <= max_items_per_cluster:
            final_clusters[f"cluster_{cluster_idx}"] = indices
            cluster_idx += 1
        else:
            # Split large cluster into chunks
            for chunk_start in range(0, len(indices), max_items_per_cluster):
                chunk = indices[chunk_start:chunk_start + max_items_per_cluster]
                final_clusters[f"cluster_{cluster_idx}"] = chunk
                cluster_idx += 1
    
    logger.info(f"Created {len(final_clusters)} clusters (max size enforced: {max_items_per_cluster})")

    return final_clusters


def cluster_by_mst(
    A_emb: np.ndarray,
    max_clusters: int = 50,
    min_cluster_size: int = 10,
    max_cluster_size: int = 1000
) -> Dict[str, List[int]]:
    """
    Cluster targets by MST edge cutting.

    Builds minimum spanning tree on cosine distances between embeddings,
    then cuts the longest edges to form clusters. This preserves local
    neighborhood relationships better than K-means.

    Args:
        A_emb: Target embeddings (N x D)
        max_clusters: Maximum number of clusters to create
        min_cluster_size: Merge clusters smaller than this
        max_cluster_size: Split clusters larger than this

    Returns:
        Dict mapping cluster_id to list of item indices
    """
    n = len(A_emb)

    if n <= max_clusters:
        # Each item is its own cluster
        return {f"mst_{i}": [i] for i in range(n)}

    logger.info(f"Building MST for {n} items...")

    # Compute pairwise cosine distances
    if n > 10000:
        logger.warning(f"Large dataset ({n} items), distance matrix will use ~{n*n*8/1e9:.1f} GB")

    distances = squareform(pdist(A_emb, metric='cosine'))

    # Build MST
    mst = minimum_spanning_tree(csr_matrix(distances))

    # Extract edges with weights
    mst_coo = mst.tocoo()
    edges = list(zip(mst_coo.row, mst_coo.col, mst_coo.data))
    edges.sort(key=lambda x: x[2], reverse=True)  # Sort by weight descending

    # Build adjacency matrix from MST
    adj = mst.toarray()
    adj = adj + adj.T  # Make symmetric

    # Cut edges adaptively: keep cutting until all components <= max_cluster_size
    # or we've made max_clusters (respecting size constraint takes priority)
    cuts_made = 0
    edge_idx = 0

    while edge_idx < len(edges):
        # Find connected components with current cuts
        n_components, labels = connected_components(csr_matrix(adj), directed=False)

        # Count component sizes
        component_sizes = np.bincount(labels)
        max_component_size = component_sizes.max()

        # Stop if we have enough clusters AND all are under max_size
        if n_components >= max_clusters and max_component_size <= max_cluster_size:
            break

        # Stop if all components are under max_size (even if fewer clusters than requested)
        if max_component_size <= max_cluster_size and n_components >= 2:
            # But keep cutting if we haven't reached max_clusters yet
            if n_components >= max_clusters:
                break

        # Cut the next longest edge
        r, c, weight = edges[edge_idx]
        if adj[r, c] > 0:  # Edge still exists
            adj[r, c] = 0
            adj[c, r] = 0
            cuts_made += 1
        edge_idx += 1

    # Final component detection
    n_components, labels = connected_components(csr_matrix(adj), directed=False)

    logger.info(f"MST has {len(edges)} edges, cut {cuts_made} to create {n_components} clusters")

    # Group indices by component
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[f"mst_{label}"].append(idx)

    # Log size distribution
    sizes = [len(v) for v in clusters.values()]
    logger.info(f"  Size range: {min(sizes)}-{max(sizes)}, Avg: {np.mean(sizes):.1f}")

    # Merge small clusters, respecting max_cluster_size
    clusters = _mst_merge_small(dict(clusters), A_emb, min_cluster_size, max_cluster_size)

    logger.info(f"Final cluster count: {len(clusters)}")

    return clusters


def _mst_merge_small(
    clusters: Dict[str, List[int]],
    A_emb: np.ndarray,
    min_size: int,
    max_size: int = None
) -> Dict[str, List[int]]:
    """Merge clusters smaller than min_size with nearest neighbor, respecting max_size."""
    if min_size <= 1:
        return clusters

    clusters = dict(clusters)  # Copy

    # Compute centroids
    centroids = {cid: A_emb[indices].mean(axis=0) for cid, indices in clusters.items()}

    merged_count = 0
    while True:
        # Find smallest cluster below threshold
        small = [(cid, len(idx)) for cid, idx in clusters.items() if len(idx) < min_size]
        if not small:
            break

        smallest_cid = min(small, key=lambda x: x[1])[0]
        smallest_centroid = centroids[smallest_cid]
        smallest_size = len(clusters[smallest_cid])

        # Find nearest neighbor cluster (by cosine similarity) that won't exceed max_size
        best_sim = -1
        best_neighbor = None
        for cid, centroid in centroids.items():
            if cid != smallest_cid:
                # Check if merging would exceed max_size
                if max_size is not None and len(clusters[cid]) + smallest_size > max_size:
                    continue
                norm_product = np.linalg.norm(smallest_centroid) * np.linalg.norm(centroid)
                if norm_product > 1e-10:
                    sim = np.dot(smallest_centroid, centroid) / norm_product
                    if sim > best_sim:
                        best_sim = sim
                        best_neighbor = cid

        if best_neighbor is None:
            # No valid merge target - keep as small cluster
            break

        # Merge
        clusters[best_neighbor].extend(clusters[smallest_cid])
        del clusters[smallest_cid]
        centroids[best_neighbor] = A_emb[clusters[best_neighbor]].mean(axis=0)
        del centroids[smallest_cid]
        merged_count += 1

    if merged_count > 0:
        logger.info(f"  Merged {merged_count} small clusters (min_size={min_size})")

    return clusters


def _mst_split_large(
    clusters: Dict[str, List[int]],
    A_emb: np.ndarray,
    max_size: int
) -> Dict[str, List[int]]:
    """Split clusters larger than max_size."""
    result = {}
    split_count = 0

    for cid, indices in clusters.items():
        if len(indices) <= max_size:
            result[cid] = indices
        else:
            # Split into chunks
            n_splits = (len(indices) + max_size - 1) // max_size
            chunk_size = len(indices) // n_splits

            for i in range(n_splits):
                start = i * chunk_size
                end = start + chunk_size if i < n_splits - 1 else len(indices)
                result[f"{cid}_s{i}"] = indices[start:end]
            split_count += 1

    if split_count > 0:
        logger.info(f"  Split {split_count} large clusters (max_size={max_size})")

    return result


def estimate_memory_mb(n_items: int, dim: int = 768) -> float:
    """Estimate memory in MB for per-query transforms."""
    # W_stack: n × dim × dim × 4 bytes (float32)
    # centroids: n × dim × 4 bytes
    # target_embeddings: n × dim × 4 bytes
    w_bytes = n_items * dim * dim * 4
    cent_bytes = n_items * dim * 4
    target_bytes = n_items * dim * 4
    return (w_bytes + cent_bytes + target_bytes) / (1024 * 1024)


def train_cluster_perquery(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    indices: List[int],
    cluster_id: str
) -> ClusterModel:
    """Train per-query transforms for a single cluster (1 W matrix per query)."""
    
    n = len(indices)
    logger.info(f"Training cluster '{cluster_id}' with {n} items (per-query, ~{estimate_memory_mb(n):.1f} MB)...")
    
    W_list = []
    for i in range(n):
        q = Q_emb[i:i+1]
        a = A_emb[i:i+1]
        W, scale, _ = compute_minimal_transform(q, a, allow_scaling=True)
        W_list.append(W)
    
    W_stack = np.stack(W_list, axis=0)
    
    return ClusterModel(
        cluster_id=cluster_id,
        W_stack=W_stack,
        centroids=Q_emb.copy(),
        target_embeddings=A_emb.copy(),
        indices=indices
    )


def train_cluster_single(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    indices: List[int],
    cluster_id: str,
    verbose: bool = True
) -> ClusterModel:
    """Train a single global W matrix for the cluster (much smaller, ~2.4 MB per cluster)."""

    n = len(indices)
    if verbose:
        logger.info(f"Training cluster '{cluster_id}' with {n} items (single W, ~2.4 MB)...")
    
    # Compute single Procrustes for entire cluster
    W, scale, info = compute_minimal_transform(Q_emb, A_emb, allow_scaling=True)
    
    # Store as 1-element stack for API consistency
    W_stack = W[np.newaxis, :, :]  # (1, 768, 768)
    
    return ClusterModel(
        cluster_id=cluster_id,
        W_stack=W_stack,
        centroids=Q_emb.mean(axis=0, keepdims=True),  # Single centroid
        target_embeddings=A_emb.copy(),
        indices=indices
    )


def train_federated(
    data: List[Dict],
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    output_path: Path,
    cluster_method: str = "path_depth",
    transform_mode: str = "single",  # "single" (1 W per cluster) or "per-query" (N W per cluster)
    max_memory_mb: float = 3000.0,
    max_clusters: int = 10,
    min_cluster_size: int = 10
) -> Tuple['FederatedModel', Dict[str, np.ndarray], Path]:
    """
    Train federated model with clustering.

    Trains each cluster and saves to disk immediately to avoid OOM.

    Args:
        data: List of Q/A pairs
        Q_emb: Query embeddings (N, 768)
        A_emb: Target embeddings (N, 768)
        output_path: Path to save model (.pkl file, clusters saved to adjacent directory)
        cluster_method: "path_depth", "embedding" (K-means), "mst", or "per-tree"
        transform_mode: "single" (1 W per cluster, ~2.4 MB each) or "per-query" (N W per cluster)
        max_memory_mb: Max memory per cluster in MB (only used for per-query mode)
        max_clusters: Maximum number of clusters
        min_cluster_size: Minimum cluster size - smaller clusters merged (for MST method)

    Returns:
        Tuple of (model, cluster_target_embeddings, output_dir)
    """
    n = len(data)
    dim = Q_emb.shape[1]
    
    # Estimate max items per cluster based on memory
    # Memory per item ≈ dim × dim × 4 = 768 × 768 × 4 ≈ 2.36 MB
    bytes_per_item = dim * dim * 4 + dim * 4 * 2  # W + centroids + targets
    max_items = int(max_memory_mb * 1024 * 1024 / bytes_per_item)
    logger.info(f"Max items per cluster: {max_items} (based on {max_memory_mb:.0f} MB limit)")
    
    # Cluster the data
    if cluster_method == "path_depth":
        clusters = cluster_by_path_depth(data, max_clusters)
    elif cluster_method == "per-tree":
        clusters = cluster_by_tree(data)
    elif cluster_method == "mst":
        clusters = cluster_by_mst(A_emb, max_clusters, min_cluster_size=min_cluster_size, max_cluster_size=max_items)
    else:  # "embedding" (K-means, default)
        clusters = cluster_by_embedding(A_emb, max_clusters, max_items)
    
    logger.info(f"Created {len(clusters)} clusters")
    if len(clusters) <= 50:
        for cid, indices in clusters.items():
            logger.info(f"  {cid}: {len(indices)} items (~{estimate_memory_mb(len(indices)):.1f} MB)")
    else:
        sizes = [len(indices) for indices in clusters.values()]
        logger.info(f"  (too many to list individually)")
        logger.info(f"  Size range: {min(sizes)}-{max(sizes)}, Avg: {np.mean(sizes):.1f}")
    
    # Train each cluster and save immediately to disk
    # This avoids keeping all W_stacks in memory
    cluster_centroids_list = []
    cluster_ids = []
    cluster_target_embeddings = {}  # Only keep target embeddings for evaluation
    idx_to_cluster = {}  # Maps global index to cluster_id for routing
    
    output_dir = output_path.with_suffix('')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_clusters = len(clusters)
    for cluster_num, (cluster_id, indices) in enumerate(clusters.items(), 1):
        Q_subset = Q_emb[indices]
        A_subset = A_emb[indices]

        # Track index-to-cluster mapping for query-level routing
        for idx in indices:
            idx_to_cluster[int(idx)] = cluster_id

        # Progress logging for large cluster counts
        if total_clusters > 50 and cluster_num % 100 == 0:
            logger.info(f"Training progress: {cluster_num}/{total_clusters} clusters ({100*cluster_num/total_clusters:.0f}%)")

        # Train this cluster using selected mode
        verbose = total_clusters <= 50  # Suppress per-cluster logs for many clusters
        if transform_mode == "per-query":
            model = train_cluster_perquery(Q_subset, A_subset, indices, cluster_id)
        else:  # "single"
            model = train_cluster_single(Q_subset, A_subset, indices, cluster_id, verbose=verbose)
        
        # Immediately save to disk
        cluster_path = output_dir / f"{cluster_id}.npz"
        np.savez_compressed(
            cluster_path,
            W_stack=model.W_stack,
            centroids=model.centroids,
            target_embeddings=model.target_embeddings,
            indices=np.array(model.indices)
        )
        if verbose:
            logger.info(f"Saved {cluster_id} to {cluster_path}")
        
        # Keep only what we need for routing
        cluster_centroids_list.append(Q_subset.mean(axis=0))
        cluster_ids.append(cluster_id)
        cluster_target_embeddings[cluster_id] = A_subset
        
        # Free memory by deleting W_stack
        del model.W_stack
        del model
        import gc
        gc.collect()
    
    cluster_centroids = np.stack(cluster_centroids_list, axis=0)
    
    # Save query embeddings for query-level routing
    routing_path = output_dir / "routing_data.npz"
    np.savez_compressed(
        routing_path,
        query_embeddings=Q_emb,
        target_embeddings=A_emb,
        idx_to_cluster_keys=np.array(list(idx_to_cluster.keys())),
        idx_to_cluster_values=np.array([idx_to_cluster[k] for k in idx_to_cluster.keys()])
    )
    logger.info(f"Saved routing data to {routing_path}")
    
    # Return a lightweight model (no W_stacks in memory)
    return FederatedModel(
        clusters=[],  # Empty - clusters are on disk
        cluster_centroids=cluster_centroids,
        cluster_ids=cluster_ids,
        global_target_ids=[d.get("tree_id", d.get("uri", str(i))) for i, d in enumerate(data)],
        global_target_titles=[d.get("raw_title", d.get("query", "")) for d in data]
    ), cluster_target_embeddings, output_dir


def project_federated(q: np.ndarray, model: FederatedModel, temperature: float = 0.1) -> np.ndarray:
    """Project a single query using federated model."""
    
    # First level: route to cluster(s)
    cluster_sims = q @ model.cluster_centroids.T
    cluster_sims_shifted = (cluster_sims - np.max(cluster_sims)) / temperature
    cluster_weights = np.exp(cluster_sims_shifted)
    cluster_weights /= cluster_weights.sum()
    
    # Second level: weighted projection across clusters
    proj = np.zeros(q.shape[-1])
    
    for i, cluster in enumerate(model.clusters):
        if cluster_weights[i] < 0.01:  # Skip low-weight clusters
            continue
        
        # Route within cluster
        within_sims = q @ cluster.centroids.T
        within_sims_shifted = (within_sims - np.max(within_sims)) / temperature
        within_weights = np.exp(within_sims_shifted)
        within_weights /= within_weights.sum()
        
        # Weighted projection within cluster
        cluster_proj = sum(w * (q @ W) for w, W in zip(within_weights, cluster.W_stack))
        proj += cluster_weights[i] * cluster_proj
    
    return proj


def evaluate_federated(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    model: FederatedModel,
    leave_one_out: bool = False
) -> Dict:
    """Evaluate federated model."""
    logger.info("Evaluating federated model...")
    
    n = len(Q_emb)
    
    # Project all queries
    Q_proj = np.array([project_federated(q, model) for q in Q_emb])
    
    # Normalize
    Q_proj_norm = Q_proj / (np.linalg.norm(Q_proj, axis=1, keepdims=True) + 1e-8)
    A_norm = A_emb / (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    sims = (Q_proj_norm * A_norm).sum(axis=1)
    logger.info(f"Mean Cosine Similarity: {np.mean(sims):.4f}")
    
    # Full similarity matrix
    sim_matrix = Q_proj_norm @ A_norm.T
    
    # Compute ranks
    ranks = []
    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    metrics = {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_cosine_sim": float(np.mean(sims))
    }
    
    logger.info(f"Recall@1: {metrics['recall_at_1']*100:.2f}%")
    logger.info(f"Recall@5: {metrics['recall_at_5']*100:.2f}%")
    logger.info(f"Recall@10: {metrics['recall_at_10']*100:.2f}%")
    logger.info(f"MRR: {metrics['mrr']:.4f}")
    
    return metrics


def save_federated_model(model: FederatedModel, path: Path):
    """Save federated model to disk."""
    # Save each cluster separately to manage memory
    base_path = path.with_suffix('')
    
    # Save main model (without cluster W_stack)
    main_data = {
        "cluster_ids": model.cluster_ids,
        "cluster_centroids": model.cluster_centroids,
        "temperature": model.temperature,
        "global_target_ids": model.global_target_ids,
        "global_target_titles": model.global_target_titles,
        "metrics": model.metrics,
        "num_clusters": len(model.clusters)
    }
    
    with open(path, "wb") as f:
        pickle.dump(main_data, f)
    
    # Save each cluster
    for cluster in model.clusters:
        cluster_path = base_path / f"{cluster.cluster_id}.npz"
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cluster_path,
            W_stack=cluster.W_stack,
            centroids=cluster.centroids,
            target_embeddings=cluster.target_embeddings,
            indices=np.array(cluster.indices)
        )
    
    logger.info(f"Model saved to {path} and {base_path}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train federated Procrustes projection for Pearltrees.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL with Q/A pairs")
    parser.add_argument("output_model", type=Path, help="Output model file (.pkl)")
    parser.add_argument("--cluster-method", choices=["path_depth", "embedding", "per-tree", "mst"], default="embedding",
                       help="Clustering method: 'embedding' (K-means), 'mst' (MST-based), 'path_depth', or 'per-tree'")
    parser.add_argument("--transform-mode", choices=["single", "per-query"], default="single",
                       help="Transform mode: 'single' (1 W per cluster, ~70MB total) or 'per-query' (N W per cluster, ~15GB)")
    parser.add_argument("--max-memory-mb", type=float, default=800.0,
                       help="Max memory per cluster in MB (for per-query mode)")
    parser.add_argument("--max-clusters", type=int, default=None,
                       help="Maximum number of clusters (overrides --cluster-criterion)")
    parser.add_argument("--cluster-criterion",
                       choices=["effective_rank", "covering", "sqrt_n", "auto"],
                       default="effective_rank",
                       help="Method to determine cluster count: 'effective_rank' (recommended), 'covering' (2^r), 'sqrt_n', or 'auto'")
    parser.add_argument("--target-variance", type=float, default=0.80,
                       help="Target variance for 'covering' criterion (default: 0.80 = 80%%)")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                       help="Minimum cluster size - smaller clusters are merged (for MST method)")
    parser.add_argument("--model", type=str, default="nomic-ai/nomic-embed-text-v1.5",
                       help="Embedding model to use (default: nomic-ai/nomic-embed-text-v1.5)")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_jsonl}...")
    with open(args.input_jsonl) as f:
        data = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(data)} items.")
    
    queries = [d["query"] for d in data]
    answers = [d["target_text"] for d in data]
    
    # Embed with specified model
    logger.info(f"Using embedding model: {args.model}")
    embedder = SimpleEmbedder(args.model)
    
    logger.info("Embedding queries...")
    Q_emb = embedder.encode(queries).astype(np.float32)
    
    logger.info("Embedding answers (paths)...")
    A_emb = embedder.encode(answers).astype(np.float32)
    
    # Free embedder memory before training
    del embedder
    import gc
    gc.collect()
    
    # Try to free GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    logger.info("Embedder released, starting training...")

    # Determine cluster count
    if args.max_clusters is not None:
        max_clusters = args.max_clusters
        logger.info(f"Using specified max_clusters={max_clusters}")
    else:
        # Use cluster criterion to determine K
        logger.info(f"Analyzing embeddings to determine cluster count (method: {args.cluster_criterion})...")
        suggestion = suggest_cluster_count(A_emb, method=args.cluster_criterion, target_variance=args.target_variance)
        max_clusters = suggestion['suggested_k']

        var_pct = int(args.target_variance * 100)
        logger.info("=" * 50)
        logger.info("Cluster Count Suggestions:")
        logger.info(f"  Effective rank (recommended): {suggestion['all_suggestions']['effective_rank']}")
        logger.info(f"  Covering ({var_pct}% variance):      {suggestion['all_suggestions']['covering']}")
        logger.info(f"  √N heuristic:                 {suggestion['all_suggestions']['sqrt_n']}")
        logger.info(f"  → Using {args.cluster_criterion}: K = {max_clusters}")
        logger.info("=" * 50)

    # Train federated model
    model, cluster_targets, output_dir = train_federated(
        data, Q_emb, A_emb,
        output_path=args.output_model,
        cluster_method=args.cluster_method,
        transform_mode=args.transform_mode,
        max_memory_mb=args.max_memory_mb,
        max_clusters=max_clusters,
        min_cluster_size=args.min_cluster_size
    )
    
    # Skip evaluation for now (would need to load clusters from disk)
    # TODO: Implement disk-based evaluation
    logger.info("Training complete. Evaluation skipped (clusters saved to disk).")
    
    # Save main model metadata
    main_data = {
        "cluster_ids": model.cluster_ids,
        "cluster_centroids": model.cluster_centroids,
        "temperature": model.temperature,
        "global_target_ids": model.global_target_ids,
        "global_target_titles": model.global_target_titles,
        "metrics": {},
        "num_clusters": len(model.cluster_ids),
        "cluster_dir": str(output_dir),
        "data_path": str(args.input_jsonl.resolve())  # For account lookup at inference
    }
    
    with open(args.output_model, "wb") as f:
        pickle.dump(main_data, f)
    
    logger.info(f"Model metadata saved to {args.output_model}")
    logger.info(f"Cluster files saved to {output_dir}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
