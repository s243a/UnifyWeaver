#!/usr/bin/env python3
"""
Cluster Q/A pairs by answer embeddings and build federated W matrix model.

This creates a federated projection model where:
1. Q/A pairs are clustered by answer embedding similarity
2. Each cluster gets a single W matrix (Procrustes transform)
3. At inference, queries are routed to cluster(s) via softmax similarity

Usage:
    python3 scripts/cluster_qa_federated.py \
      --embeddings datasets/skills_qa/qa_embeddings_nomic.npz \
      --qa-data datasets/skills_qa/tailored_scored/skills_qa.jsonl \
      --output models/skills_qa_federated.pkl
"""

import argparse
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

from minimal_transform import compute_minimal_transform


@dataclass
class ClusterModel:
    """Model for a single cluster."""
    cluster_id: str
    W: np.ndarray  # Single W matrix (768, 768)
    centroid: np.ndarray  # Cluster centroid for routing (768,)
    target_embeddings: np.ndarray  # Answer embeddings in this cluster (N, 768)
    indices: List[int]  # Original indices in full dataset
    scale: float = 1.0


@dataclass
class FederatedQAModel:
    """Complete federated Q/A model."""
    clusters: List[ClusterModel]
    cluster_centroids: np.ndarray  # Cluster centroids for routing (K, 768)
    cluster_ids: List[str]
    temperature: float = 0.1
    embedding_dim: int = 768
    num_pairs: int = 0
    metrics: Dict = field(default_factory=dict)


def suggest_cluster_count(embeddings: np.ndarray, method: str = "effective_rank") -> dict:
    """
    Suggest optimal cluster count based on embedding structure.

    Args:
        embeddings: Data embeddings (N x D)
        method: One of 'effective_rank', 'sqrt_n', 'fixed'

    Returns:
        Dict with 'suggested_k' and analysis details
    """
    N, D = embeddings.shape

    # SVD analysis
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)

    # Effective rank (participation ratio)
    effective_rank = int((np.sum(S)**2) / np.sum(S**2))

    # √N heuristic
    sqrt_k = int(np.ceil(np.sqrt(N)))

    suggestions = {
        'effective_rank': effective_rank,
        'sqrt_n': sqrt_k,
        'fixed_20': 20,
    }

    if method == "effective_rank":
        suggested_k = effective_rank
    elif method == "sqrt_n":
        suggested_k = sqrt_k
    else:
        suggested_k = effective_rank

    # Clamp to reasonable range
    suggested_k = max(5, min(suggested_k, 100))

    return {
        'suggested_k': suggested_k,
        'method': method,
        'all_suggestions': suggestions,
        'variance_dims': {
            'r_50pct': int(np.searchsorted(cumvar, 0.50) + 1),
            'r_80pct': int(np.searchsorted(cumvar, 0.80) + 1),
            'r_90pct': int(np.searchsorted(cumvar, 0.90) + 1),
        }
    }


def cluster_by_kmeans(A_emb: np.ndarray, n_clusters: int) -> Dict[str, List[int]]:
    """Cluster answers using K-means."""
    from sklearn.cluster import KMeans

    logger.info(f"Running K-means with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(A_emb)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[f"cluster_{label}"].append(i)

    return dict(clusters)


def cluster_by_skill(qa_data: List[Dict]) -> Dict[str, List[int]]:
    """
    Cluster Q/A pairs by source skill.

    Uses the skill name from the pair metadata as the cluster key.
    """
    skill_groups = defaultdict(list)

    for i, pair in enumerate(qa_data):
        # Try different fields that might contain skill info
        skill = pair.get("skill", "")
        if not skill:
            skill = pair.get("source_skill", "")
        if not skill:
            # Extract from pair_id if present (e.g., "skill_networking_q1")
            pair_id = pair.get("pair_id", "")
            if pair_id.startswith("skill_"):
                parts = pair_id.split("_")
                if len(parts) >= 2:
                    skill = "_".join(parts[:2])  # e.g., "skill_networking"
        if not skill:
            skill = "unknown"

        skill_groups[skill].append(i)

    logger.info(f"Created {len(skill_groups)} skill-based clusters")

    # Log distribution
    sizes = [len(indices) for indices in skill_groups.values()]
    logger.info(f"  Size range: {min(sizes)}-{max(sizes)}, Avg: {np.mean(sizes):.1f}")

    return dict(skill_groups)


def train_cluster(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    indices: List[int],
    cluster_id: str
) -> ClusterModel:
    """Train a single W matrix for the cluster."""

    n = len(indices)
    logger.info(f"Training cluster '{cluster_id}' with {n} items...")

    # Compute single Procrustes transform for the cluster
    W, scale, info = compute_minimal_transform(Q_emb, A_emb, allow_scaling=True)

    # Compute cluster centroid (mean of question embeddings)
    centroid = Q_emb.mean(axis=0)

    return ClusterModel(
        cluster_id=cluster_id,
        W=W,
        centroid=centroid,
        target_embeddings=A_emb.copy(),
        indices=indices,
        scale=scale
    )


def train_federated_qa(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    qa_data: List[Dict],
    cluster_method: str = "kmeans",
    n_clusters: int = None,
) -> FederatedQAModel:
    """
    Train federated Q/A model.

    Args:
        Q_emb: Question embeddings (N, D)
        A_emb: Answer embeddings (N, D)
        qa_data: Original Q/A pair data
        cluster_method: "kmeans" or "skill"
        n_clusters: Number of clusters (for kmeans)

    Returns:
        Trained federated model
    """
    n = len(Q_emb)
    dim = Q_emb.shape[1]

    # Determine cluster count if not specified
    if n_clusters is None and cluster_method == "kmeans":
        suggestion = suggest_cluster_count(A_emb)
        n_clusters = suggestion['suggested_k']
        logger.info(f"Auto-selected {n_clusters} clusters (effective rank)")

    # Cluster the data
    if cluster_method == "skill":
        clusters = cluster_by_skill(qa_data)
    else:
        clusters = cluster_by_kmeans(A_emb, n_clusters)

    logger.info(f"Created {len(clusters)} clusters")

    # Train each cluster
    cluster_models = []
    cluster_centroids = []
    cluster_ids = []

    for cluster_id, indices in clusters.items():
        Q_subset = Q_emb[indices]
        A_subset = A_emb[indices]

        model = train_cluster(Q_subset, A_subset, indices, cluster_id)

        cluster_models.append(model)
        cluster_centroids.append(model.centroid)
        cluster_ids.append(cluster_id)

    cluster_centroids = np.stack(cluster_centroids, axis=0)

    return FederatedQAModel(
        clusters=cluster_models,
        cluster_centroids=cluster_centroids,
        cluster_ids=cluster_ids,
        embedding_dim=dim,
        num_pairs=n,
    )


def project_query(q: np.ndarray, model: FederatedQAModel, top_k_clusters: int = 3) -> np.ndarray:
    """
    Project a query embedding using the federated model.

    Uses soft routing across top-k clusters.
    """
    q = q.flatten()

    # Compute similarity to cluster centroids
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    centroid_norms = model.cluster_centroids / (np.linalg.norm(model.cluster_centroids, axis=1, keepdims=True) + 1e-8)

    similarities = q_norm @ centroid_norms.T

    # Get top-k clusters
    top_indices = np.argsort(similarities)[::-1][:top_k_clusters]
    top_sims = similarities[top_indices]

    # Softmax over top-k
    top_sims_shifted = (top_sims - np.max(top_sims)) / model.temperature
    weights = np.exp(top_sims_shifted)
    weights /= weights.sum()

    # Weighted projection
    projected = np.zeros(model.embedding_dim)
    for idx, weight in zip(top_indices, weights):
        cluster = model.clusters[idx]
        projected += weight * (q @ cluster.W)

    return projected


def evaluate_model(
    Q_emb: np.ndarray,
    A_emb: np.ndarray,
    model: FederatedQAModel,
) -> Dict:
    """Evaluate the federated model."""
    logger.info("Evaluating federated model...")

    n = len(Q_emb)

    # Project all queries
    Q_proj = np.array([project_query(q, model) for q in Q_emb])

    # Normalize
    Q_proj_norm = Q_proj / (np.linalg.norm(Q_proj, axis=1, keepdims=True) + 1e-8)
    A_norm = A_emb / (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-8)

    # Diagonal cosine similarity (projected query vs its answer)
    diag_sims = (Q_proj_norm * A_norm).sum(axis=1)

    # Full similarity matrix for ranking
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
        "mean_cosine_sim": float(np.mean(diag_sims)),
        "median_rank": float(np.median(ranks)),
    }

    logger.info(f"Recall@1: {metrics['recall_at_1']*100:.2f}%")
    logger.info(f"Recall@5: {metrics['recall_at_5']*100:.2f}%")
    logger.info(f"Recall@10: {metrics['recall_at_10']*100:.2f}%")
    logger.info(f"MRR: {metrics['mrr']:.4f}")
    logger.info(f"Mean Cosine Sim: {metrics['mean_cosine_sim']:.4f}")

    return metrics


def save_model(model: FederatedQAModel, output_path: Path):
    """Save the federated model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare serializable data
    model_data = {
        "cluster_centroids": model.cluster_centroids,
        "cluster_ids": model.cluster_ids,
        "temperature": model.temperature,
        "embedding_dim": model.embedding_dim,
        "num_pairs": model.num_pairs,
        "metrics": model.metrics,
        "clusters": []
    }

    for cluster in model.clusters:
        cluster_data = {
            "cluster_id": cluster.cluster_id,
            "W": cluster.W,
            "centroid": cluster.centroid,
            "target_embeddings": cluster.target_embeddings,
            "indices": cluster.indices,
            "scale": cluster.scale,
        }
        model_data["clusters"].append(cluster_data)

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Saved model to {output_path}")

    # Also save cluster files for memory-efficient loading
    cluster_dir = output_path.with_suffix('')
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for cluster in model.clusters:
        cluster_path = cluster_dir / f"{cluster.cluster_id}.npz"
        np.savez_compressed(
            cluster_path,
            W=cluster.W,
            centroid=cluster.centroid,
            target_embeddings=cluster.target_embeddings,
            indices=np.array(cluster.indices),
            scale=np.array([cluster.scale])
        )

    logger.info(f"Saved cluster files to {cluster_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Cluster Q/A pairs and build federated W matrix model")
    parser.add_argument("--embeddings", type=Path, required=True,
                       help="Path to Q/A embeddings (.npz with q_embeddings and a_embeddings)")
    parser.add_argument("--qa-data", type=Path, required=True,
                       help="Path to Q/A JSONL data")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output model path (.pkl)")
    parser.add_argument("--cluster-method", choices=["kmeans", "skill"], default="kmeans",
                       help="Clustering method")
    parser.add_argument("--n-clusters", type=int, default=None,
                       help="Number of clusters (auto-detected if not specified)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model after training")

    args = parser.parse_args()

    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings}...")
    emb_data = np.load(args.embeddings)
    Q_emb = emb_data["q_embeddings"]
    A_emb = emb_data["a_embeddings"]
    logger.info(f"Loaded {len(Q_emb)} Q/A pairs, dim={Q_emb.shape[1]}")

    # Load Q/A data
    logger.info(f"Loading Q/A data from {args.qa_data}...")
    qa_data = []
    with open(args.qa_data) as f:
        for line in f:
            if line.strip():
                qa_data.append(json.loads(line))
    logger.info(f"Loaded {len(qa_data)} Q/A pairs")

    # Verify counts match
    if len(qa_data) != len(Q_emb):
        logger.warning(f"Count mismatch: {len(qa_data)} Q/A pairs vs {len(Q_emb)} embeddings")
        # Truncate to smaller
        min_count = min(len(qa_data), len(Q_emb))
        qa_data = qa_data[:min_count]
        Q_emb = Q_emb[:min_count]
        A_emb = A_emb[:min_count]

    # Train model
    model = train_federated_qa(
        Q_emb=Q_emb,
        A_emb=A_emb,
        qa_data=qa_data,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
    )

    # Evaluate
    if args.evaluate:
        metrics = evaluate_model(Q_emb, A_emb, model)
        model.metrics = metrics

    # Save
    save_model(model, args.output)

    logger.info("Done!")
    logger.info(f"  Clusters: {len(model.clusters)}")
    logger.info(f"  Q/A pairs: {model.num_pairs}")
    logger.info(f"  Embedding dim: {model.embedding_dim}")


if __name__ == "__main__":
    main()
