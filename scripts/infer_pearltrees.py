#!/usr/bin/env python3
"""
Pearltrees Query Inference.

Fast inference using pre-computed target embeddings.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
import time

def load_model(pkl_path: Path):
    """Load the saved model with cached target embeddings."""
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    npz_path = Path(metadata['npz_file'])
    data = np.load(npz_path)
    
    model = {
        'W_stack': data['W_stack'],
        'centroids': data['centroids'],
        'temperature': float(data['temperature'][0]),
        'num_clusters': metadata['num_clusters'],
        'embedding_dim': metadata['embedding_dim'],
        'target_embeddings': data['target_embeddings'],
        'target_ids': metadata.get('target_ids', []),
        'target_titles': metadata.get('target_titles', []),
    }
    
    # Pre-normalize target embeddings for fast cosine
    target_norms = np.linalg.norm(model['target_embeddings'], axis=1, keepdims=True) + 1e-8
    model['target_embeddings_normed'] = model['target_embeddings'] / target_norms
    
    return model


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


def search(projected_query, model, top_k=5):
    """Find top-k matches using cached target embeddings."""
    # Normalize projected query
    proj_norm = projected_query / (np.linalg.norm(projected_query) + 1e-8)
    
    # Cosine similarity to all targets (fast with pre-normalized targets)
    sims = proj_norm @ model['target_embeddings_normed'].T
    
    # Get top-k indices
    top_indices = np.argsort(-sims)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'score': float(sims[idx]),
            'tree_id': model['target_ids'][idx] if idx < len(model['target_ids']) else str(idx),
            'title': model['target_titles'][idx] if idx < len(model['target_titles']) else '',
        })
    
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python infer_pearltrees.py <model.pkl> <query_text>")
        print("       python infer_pearltrees.py <model.pkl> --interactive")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    
    # Load model
    print(f"Loading model from {model_path}...")
    start = time.perf_counter()
    model = load_model(model_path)
    load_time = time.perf_counter() - start
    print(f"Model loaded: {model['num_clusters']} clusters, {len(model['target_ids'])} targets ({load_time:.2f}s)")
    
    # Load embedder
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    if sys.argv[2] == "--interactive":
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            query = input("Query> ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query:
                continue
            
            # Embed query
            start = time.perf_counter()
            q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype(np.float32)
            embed_time = time.perf_counter() - start
            
            # Project and search
            start = time.perf_counter()
            proj = project_query(q_emb, model)
            results = search(proj, model, top_k=5)
            search_time = time.perf_counter() - start
            
            print(f"\nResults (embed: {embed_time*1000:.1f}ms, search: {search_time*1000:.1f}ms):")
            for r in results:
                print(f"  {r['rank']}. [{r['score']:.4f}] {r['title']} (id: {r['tree_id']})")
            print()
    else:
        query = " ".join(sys.argv[2:])
        
        # Embed query
        start = time.perf_counter()
        q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        embed_time = time.perf_counter() - start
        
        # Project and search
        start = time.perf_counter()
        proj = project_query(q_emb, model)
        results = search(proj, model, top_k=5)
        search_time = time.perf_counter() - start
        
        print(f"\nQuery: {query}")
        print(f"Timing: embed={embed_time*1000:.1f}ms, search={search_time*1000:.1f}ms")
        print("\nTop 5 Results:")
        for r in results:
            print(f"  {r['rank']}. [{r['score']:.4f}] {r['title']} (id: {r['tree_id']})")


if __name__ == "__main__":
    main()
