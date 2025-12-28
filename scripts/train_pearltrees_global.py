#!/usr/bin/env python3
"""
Pearltrees Projection Training with Global, Per-Cluster, or Per-Query Procrustes.

Modes:
- global: Single 768×768 transform for all queries (fast, memory-efficient)
- per-cluster: Separate transform per parent tree (uses cluster_id field)
- per-query: Separate transform per query-answer pair (1/1 mapping, most accurate)
"""

import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add the python_runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

from minimal_transform import compute_minimal_transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleEmbedder:
    """Wraps SentenceTransformer for embedding."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def train_global(Q_emb, A_emb, data):
    """Train a single global Procrustes transform."""
    logger.info("Computing global Procrustes transformation...")
    W, scale, info = compute_minimal_transform(Q_emb, A_emb, allow_scaling=True)
    logger.info(f"Transform info: scale={scale:.4f}, residual={info.get('residual', 'N/A')}")
    
    return {
        "type": "global_procrustes",
        "W": W,
        "scale": scale,
    }


def train_per_cluster(Q_emb, A_emb, data):
    """Train separate Procrustes transforms per cluster (parent tree)."""
    # Group by cluster_id
    cluster_groups = defaultdict(list)
    for i, d in enumerate(data):
        cluster_id = d.get("cluster_id", "default")
        cluster_groups[cluster_id].append(i)
    
    logger.info(f"Found {len(cluster_groups)} unique clusters.")
    
    W_list = []
    centroids = []
    
    for cluster_id, indices in cluster_groups.items():
        Q_subset = Q_emb[indices]
        A_subset = A_emb[indices]
        
        # Compute per-cluster transform
        W, scale, _ = compute_minimal_transform(Q_subset, A_subset, allow_scaling=True)
        W_list.append(W)
        
        # Centroid for routing
        centroid = Q_subset.mean(axis=0)
        centroids.append(centroid)
    
    W_stack = np.stack(W_list, axis=0)  # (N_clusters, 768, 768)
    centroids = np.stack(centroids, axis=0)  # (N_clusters, 768)
    
    logger.info(f"Per-cluster training complete: {len(W_list)} transforms")
    
    return {
        "type": "per_cluster_procrustes",
        "W_stack": W_stack,
        "centroids": centroids,
        "temperature": 0.1,
    }


def train_per_query(Q_emb, A_emb, data):
    """Train separate Procrustes transform for each query-answer pair (1/1 mapping)."""
    logger.info(f"Computing per-query transforms for {len(Q_emb)} pairs...")
    
    W_list = []
    for i in range(len(Q_emb)):
        q = Q_emb[i:i+1]  # Keep as 2D
        a = A_emb[i:i+1]
        W, scale, _ = compute_minimal_transform(q, a, allow_scaling=True)
        W_list.append(W)
    
    W_stack = np.stack(W_list, axis=0)  # (N, 768, 768)
    centroids = Q_emb.copy()  # Each query is its own centroid
    
    logger.info(f"Per-query training complete: {len(W_list)} transforms")
    
    return {
        "type": "per_query_procrustes",
        "W_stack": W_stack,
        "centroids": centroids,
        "temperature": 0.1,
    }


def project_queries(Q_emb, model_data):
    """Project queries using the trained model."""
    if model_data["type"] == "global_procrustes":
        return Q_emb @ model_data["W"]
    else:
        # Per-cluster: use softmax routing
        centroids = model_data["centroids"]
        W_stack = model_data["W_stack"]
        temperature = model_data["temperature"]
        
        Q_proj = []
        for q in Q_emb:
            # Softmax routing (log-sum-exp trick to prevent overflow)
            sims = q @ centroids.T
            sims_shifted = (sims - np.max(sims)) / temperature
            weights = np.exp(sims_shifted)
            weights /= weights.sum()
            
            # Weighted projection
            proj = sum(w * (q @ W) for w, W in zip(weights, W_stack))
            Q_proj.append(proj)
        return np.array(Q_proj)


def evaluate(Q_emb, A_emb, model_data):
    """Evaluate the model on training set."""
    logger.info("Evaluating on training set...")
    
    Q_proj = project_queries(Q_emb, model_data)
    
    # Normalize for cosine similarity
    Q_proj_norm = Q_proj / (np.linalg.norm(Q_proj, axis=1, keepdims=True) + 1e-8)
    A_emb_norm = A_emb / (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    sims = (Q_proj_norm * A_emb_norm).sum(axis=1)
    logger.info(f"Mean Cosine Similarity (Train): {np.mean(sims):.4f}")
    
    # Compute ranking metrics
    logger.info("Computing ranking metrics...")
    sim_matrix = Q_proj_norm @ A_emb_norm.T
    
    ranks = []
    for i in range(len(Q_emb)):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    r_at_1 = np.mean(ranks == 1)
    r_at_5 = np.mean(ranks <= 5)
    r_at_10 = np.mean(ranks <= 10)
    mrr = np.mean(1.0 / ranks)
    
    logger.info(f"Recall@1: {r_at_1*100:.2f}%")
    logger.info(f"Recall@5: {r_at_5*100:.2f}%")
    logger.info(f"Recall@10: {r_at_10*100:.2f}%")
    logger.info(f"MRR: {mrr:.4f}")
    
    return {
        "recall_at_1": float(r_at_1),
        "recall_at_5": float(r_at_5),
        "recall_at_10": float(r_at_10),
        "mrr": float(mrr),
        "mean_cosine_sim": float(np.mean(sims)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Procrustes projection for Pearltrees.")
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL with Q/A pairs")
    parser.add_argument("output_model", type=Path, help="Output model file (.pkl)")
    parser.add_argument("--mode", choices=["global", "per-cluster", "per-query"], default="global",
                       help="Training mode: 'global' (single transform), 'per-cluster' (by parent tree), 'per-query' (1/1 mapping)")
    
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
    
    # Train
    logger.info(f"Training mode: {args.mode}")
    if args.mode == "global":
        model_data = train_global(Q_emb, A_emb, data)
    elif args.mode == "per-cluster":
        model_data = train_per_cluster(Q_emb, A_emb, data)
    else:  # per-query
        model_data = train_per_query(Q_emb, A_emb, data)
    
    # Evaluate
    metrics = evaluate(Q_emb, A_emb, model_data)
    
    # Add embeddings and metadata
    model_data["target_embeddings"] = A_emb
    model_data["query_embeddings"] = Q_emb
    model_data["target_ids"] = [d.get("tree_id", d.get("uri", str(i))) for i, d in enumerate(data)]
    model_data["target_titles"] = [d.get("raw_title", d.get("query", "")) for d in data]
    model_data["metrics"] = metrics
    
    # Save model
    logger.info(f"Saving model to {args.output_model}...")
    with open(args.output_model, "wb") as f:
        pickle.dump(model_data, f)
    
    # Also save npz for embeddings
    npz_path = args.output_model.with_suffix(".npz")
    npz_data = {
        "target_embeddings": A_emb,
        "query_embeddings": Q_emb,
    }
    if args.mode == "global":
        npz_data["W"] = model_data["W"]
    else:
        npz_data["W_stack"] = model_data["W_stack"]
        npz_data["centroids"] = model_data["centroids"]
    
    np.savez_compressed(npz_path, **npz_data)
    logger.info(f"Embeddings saved to {npz_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
