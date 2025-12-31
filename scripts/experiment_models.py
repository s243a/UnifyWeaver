#!/usr/bin/env python3
"""
Embedding Model Experiment Framework

Tests different model combinations for the three embedding points:
1. Raw Query (Input Objective) - pure semantic matching
2. Functionally Wrapped Query - tied to W matrices
3. Output/Target - tied to W matrices (same space as #2)

Usage:
    # Activate the hf-env first:
    source ~/.hf-env/bin/activate
    
    python3 scripts/experiment_models.py \
        --raw-model sentence-transformers/all-MiniLM-L6-v2 \
        --wrapped-model nomic-ai/modernbert-embed-base \
        --data reports/pearltrees_targets_full_pearls.jsonl \
        --queries "Hyphanet" "Feynman Lectures" "Quantum Mechanics"
"""

import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Model configurations to test
MODEL_CONFIGS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "context": 512,
        "notes": "Fast, small, good for short queries"
    },
    "nomic": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "context": 8192,
        "notes": "Good for long context, functional queries"
    },
    "modernbert": {
        "name": "nomic-ai/modernbert-embed-base",
        "dim": 768,
        "context": 8192,
        "notes": "Strong semantic understanding, requires transformers 4.48+"
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "context": 512,
        "notes": "Optimized for short text retrieval"
    },
    "gte-small": {
        "name": "Alibaba-NLP/gte-small",
        "dim": 384,
        "context": 512,
        "notes": "Strong MTEB scores for sentence similarity"
    }
}

TEST_QUERIES = [
    ("Hyphanet", "Freenet & Hyphanet"),  # Should find privacy/darknet folder
    ("Feynman Lectures", "The Feynman Lectures on Physics"),  # Physics
    ("Quantum Mechanics", "Quantum mechanics"),  # Physics
    ("wave", "Waves"),  # Disambiguation test
    ("uncertainty", "Uncertainty principle"),  # Disambiguation test
    ("neural network", "Artificial neural network"),  # Tech/AI
]


def load_model(model_name: str):
    """Load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, trust_remote_code=True)


def load_targets(data_path: Path, max_items: int = None) -> List[Dict]:
    """Load target items from JSONL."""
    items = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            line = line.strip()
            if line:
                item = json.loads(line)
                items.append(item)
    return items


def cosine_similarity(a, b):
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def search_raw(query_emb, target_embs, top_k=5):
    """Raw cosine similarity search."""
    scores = np.array([cosine_similarity(query_emb, t) for t in target_embs])
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def run_experiment(
    raw_model_name: str,
    wrapped_model_name: str,
    targets: List[Dict],
    queries: List[Tuple[str, str]],
    top_k: int = 5
) -> Dict:
    """
    Run experiment with specified model configuration.
    
    Returns results dict with timing and accuracy metrics.
    """
    results = {
        "raw_model": raw_model_name,
        "wrapped_model": wrapped_model_name,
        "num_targets": len(targets),
        "queries": []
    }
    
    print(f"\n{'='*60}")
    print(f"Raw Model: {raw_model_name}")
    print(f"Wrapped Model: {wrapped_model_name}")
    print(f"Targets: {len(targets)}")
    print(f"{'='*60}")
    
    # Load models
    print("\nLoading models...")
    t0 = time.time()
    raw_model = load_model(raw_model_name)
    results["raw_model_load_time"] = time.time() - t0
    
    t0 = time.time()
    if wrapped_model_name != raw_model_name:
        wrapped_model = load_model(wrapped_model_name)
    else:
        wrapped_model = raw_model
    results["wrapped_model_load_time"] = time.time() - t0
    
    # Embed targets (using raw titles with raw model)
    print("Embedding targets...")
    t0 = time.time()
    # Get titles - try 'title' first, fallback to parsing query_text
    target_titles = []
    for t in targets:
        title = t.get("title", "")
        if not title and "query_text" in t:
            # Parse from query_text like "locate_node('Title')"
            qt = t.get("query_text", "")
            if "'" in qt:
                title = qt.split("'")[1] if len(qt.split("'")) > 1 else qt
        target_titles.append(title)
    target_embs = raw_model.encode(target_titles, show_progress_bar=True)
    results["target_embed_time"] = time.time() - t0
    
    # Run queries
    hits = 0
    total = 0
    
    for query, expected in queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: '{expected}'")
        
        t0 = time.time()
        query_emb = raw_model.encode(query)
        query_time = time.time() - t0
        
        top_results = search_raw(query_emb, target_embs, top_k)
        
        query_result = {
            "query": query,
            "expected": expected,
            "query_time": query_time,
            "results": []
        }
        
        found = False
        for rank, (idx, score) in enumerate(top_results, 1):
            title = target_titles[idx]
            is_match = expected.lower() in title.lower() or title.lower() in expected.lower()
            
            query_result["results"].append({
                "rank": rank,
                "title": title,
                "score": score,
                "is_match": is_match
            })
            
            if is_match and not found:
                found = True
                hits += 1
                print(f"  ✓ #{rank} [{score:.4f}] {title}")
            else:
                print(f"  #{rank} [{score:.4f}] {title}")
        
        total += 1
        query_result["hit"] = found
        results["queries"].append(query_result)
    
    results["hit_rate"] = hits / total if total > 0 else 0
    print(f"\nHit Rate: {hits}/{total} = {results['hit_rate']:.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test embedding model combinations")
    parser.add_argument("--raw-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Model for raw query embedding (Input Objective)")
    parser.add_argument("--wrapped-model", type=str, default="nomic-ai/nomic-embed-text-v1.5",
                       help="Model for wrapped query embedding (tied to W)")
    parser.add_argument("--data", type=Path, default=Path("reports/pearltrees_targets_full_pearls.jsonl"),
                       help="Path to JSONL data")
    parser.add_argument("--max-targets", type=int, default=None,
                       help="Limit number of targets (for quick testing)")
    parser.add_argument("--queries", nargs="+", help="Custom queries to test")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Load targets
    print(f"Loading targets from {args.data}...")
    targets = load_targets(args.data, args.max_targets)
    print(f"Loaded {len(targets)} targets")
    
    # Use custom queries or defaults
    if args.queries:
        queries = [(q, q) for q in args.queries]  # No expected value
    else:
        queries = TEST_QUERIES
    
    # Run experiment
    results = run_experiment(
        args.raw_model,
        args.wrapped_model,
        targets,
        queries,
        args.top_k
    )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Hit Rate: {results['hit_rate']:.1%}")
    print(f"Target Embedding Time: {results['target_embed_time']:.2f}s")
    print(f"Avg Query Time: {np.mean([q['query_time'] for q in results['queries']]):.4f}s")


if __name__ == "__main__":
    main()
