#!/usr/bin/env python3
"""
Federated Pearltrees Projection Training.

Clusters targets by materialized path depth or output embedding similarity,
then trains per-query Procrustes transforms within each cluster.

At inference:
1. Route query to cluster(s) via softmax
2. Within cluster, route to specific transform via softmax
3. Project and search
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


def estimate_memory_mb(n_items: int, dim: int = 768) -> float:
    """Estimate memory in MB for per-query transforms."""
    # W_stack: n × dim × dim × 4 bytes (float32)
    # centroids: n × dim × 4 bytes
    # target_embeddings: n × dim × 4 bytes
    w_bytes = n_items * dim * dim * 4
    cent_bytes = n_items * dim * 4
    target_bytes = n_items * dim * 4
    return (w_bytes + cent_bytes + target_bytes) / (1024 * 1024)


def train_cluster(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    indices: List[int],
    cluster_id: str
) -> ClusterModel:
    """Train per-query transforms for a single cluster."""
    
    n = len(indices)
    logger.info(f"Training cluster '{cluster_id}' with {n} items (~{estimate_memory_mb(n):.1f} MB)...")
    
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


def train_federated(
    data: List[Dict],
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    output_path: Path,
    cluster_method: str = "path_depth",
    max_memory_mb: float = 3000.0,
    max_clusters: int = 10
) -> Tuple['FederatedModel', Dict[str, np.ndarray], Path]:
    """
    Train federated model with clustering.
    
    Trains each cluster and saves to disk immediately to avoid OOM.
    
    Args:
        data: List of Q/A pairs
        Q_emb: Query embeddings (N, 768)
        A_emb: Target embeddings (N, 768)
        output_path: Path to save model (.pkl file, clusters saved to adjacent directory)
        cluster_method: "path_depth" or "embedding"
        max_memory_mb: Max memory per cluster in MB
        max_clusters: Maximum number of clusters
        
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
    else:
        clusters = cluster_by_embedding(A_emb, max_clusters, max_items)
    
    logger.info(f"Created {len(clusters)} clusters:")
    for cid, indices in clusters.items():
        logger.info(f"  {cid}: {len(indices)} items (~{estimate_memory_mb(len(indices)):.1f} MB)")
    
    # Train each cluster and save immediately to disk
    # This avoids keeping all W_stacks in memory
    cluster_centroids_list = []
    cluster_ids = []
    cluster_target_embeddings = {}  # Only keep target embeddings for evaluation
    
    output_dir = output_path.with_suffix('')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cluster_id, indices in clusters.items():
        Q_subset = Q_emb[indices]
        A_subset = A_emb[indices]
        
        # Train this cluster
        model = train_cluster(Q_subset, A_subset, indices, cluster_id)
        
        # Immediately save to disk
        cluster_path = output_dir / f"{cluster_id}.npz"
        np.savez_compressed(
            cluster_path,
            W_stack=model.W_stack,
            centroids=model.centroids,
            target_embeddings=model.target_embeddings,
            indices=np.array(model.indices)
        )
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
    parser.add_argument("--cluster-method", choices=["path_depth", "embedding"], default="embedding",
                       help="Clustering method")
    parser.add_argument("--max-memory-mb", type=float, default=3000.0,
                       help="Max memory per cluster in MB")
    parser.add_argument("--max-clusters", type=int, default=10,
                       help="Maximum number of clusters")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_jsonl}...")
    with open(args.input_jsonl) as f:
        data = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(data)} items.")
    
    queries = [d["query"] for d in data]
    answers = [d["target_text"] for d in data]
    
    # Embed
    embedder = SimpleEmbedder()
    
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
    
    # Train federated model
    model, cluster_targets, output_dir = train_federated(
        data, Q_emb, A_emb,
        output_path=args.output_model,
        cluster_method=args.cluster_method,
        max_memory_mb=args.max_memory_mb,
        max_clusters=args.max_clusters
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
        "cluster_dir": str(output_dir)
    }
    
    with open(args.output_model, "wb") as f:
        pickle.dump(main_data, f)
    
    logger.info(f"Model metadata saved to {args.output_model}")
    logger.info(f"Cluster files saved to {output_dir}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
