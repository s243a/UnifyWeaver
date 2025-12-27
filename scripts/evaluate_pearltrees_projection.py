#!/usr/bin/env python3
"""
Evaluate Pearltrees Projection Model.

Computes ranking metrics: R@1, R@5, R@10, MRR.
"""

import sys
import json
import logging
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))


def load_model(pkl_path: Path):
    """Load the saved model."""
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    npz_path = Path(metadata['npz_file'])
    data = np.load(npz_path)
    
    return {
        'W_stack': data['W_stack'],
        'centroids': data['centroids'],
        'temperature': float(data['temperature'][0]),
        'num_clusters': metadata['num_clusters'],
        'embedding_dim': metadata['embedding_dim'],
    }


def project_query(query_emb, model):
    """Project a query using softmax routing over clusters."""
    centroids = model['centroids']
    W_stack = model['W_stack']
    temperature = model['temperature']
    
    # Normalize for cosine similarity
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    c_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    similarities = q_norm @ c_norms.T
    
    # Softmax routing
    exp_sim = np.exp((similarities - np.max(similarities)) / temperature)
    weights = exp_sim / (np.sum(exp_sim) + 1e-8)
    
    # Weighted projection
    projected = np.zeros(model['embedding_dim'], dtype=np.float32)
    for i, w in enumerate(weights):
        projected += w * (query_emb @ W_stack[i])
    
    return projected


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_pearltrees_projection.py <model.pkl> <data.jsonl>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    data_path = Path(sys.argv[2])
    
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)
    logger.info(f"Model: {model['num_clusters']} clusters, {model['embedding_dim']}-dim")
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} items.")
    
    # Embed
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    queries = [d['query'] for d in data]
    answers = [d['target_text'] for d in data]
    
    logger.info("Embedding queries and answers...")
    Q_emb = embedder.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    A_emb = embedder.encode(answers, show_progress_bar=True, convert_to_numpy=True)
    
    # Evaluate
    logger.info("Evaluating...")
    ranks = []
    
    for i in range(len(data)):
        q_vec = Q_emb[i].astype(np.float32)
        target_vec = A_emb[i]
        
        # Project query
        proj_vec = project_query(q_vec, model)
        
        # Compute similarity to ALL answers
        proj_norm = proj_vec / (np.linalg.norm(proj_vec) + 1e-8)
        a_norms = A_emb / (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-8)
        sims = proj_norm @ a_norms.T
        
        # Find rank of correct answer (0-indexed)
        sorted_indices = np.argsort(-sims)  # Descending
        rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Compute metrics
    r_at_1 = np.mean(ranks <= 1)
    r_at_5 = np.mean(ranks <= 5)
    r_at_10 = np.mean(ranks <= 10)
    mrr = np.mean(1.0 / ranks)
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Queries: {len(ranks)}")
    print(f"Total Answer Pool: {len(A_emb)}")
    print("-" * 60)
    print(f"Recall@1:  {r_at_1:.2%}  (correct answer is #1)")
    print(f"Recall@5:  {r_at_5:.2%}  (correct answer in top 5)")
    print(f"Recall@10: {r_at_10:.2%}  (correct answer in top 10)")
    print(f"MRR:       {mrr:.4f}  (Mean Reciprocal Rank)")
    print(f"Mean Rank: {mean_rank:.1f}")
    print(f"Median Rank: {median_rank:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
