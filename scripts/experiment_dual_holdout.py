#!/usr/bin/env python3
"""
Dual-Objective Experiment with Holdout

Tests dual-objective scoring with different models while properly
holding out the test query from routing/training.

Usage:
    source ~/.hf-env/bin/activate
    python3 scripts/experiment_dual_holdout.py \
        --model nomic-ai/modernbert-embed-base \
        --data reports/pearltrees_targets_physics.jsonl \
        --alpha 0.7
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple


def load_model(model_name: str):
    """Load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, trust_remote_code=True)


def load_targets(data_path: Path) -> List[Dict]:
    """Load target items from JSONL."""
    items = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items.append(item)
    return items


def extract_title(item: Dict) -> str:
    """Extract title from item."""
    # Try raw_title first (our format), then title, then parse query_text
    title = item.get("raw_title", "") or item.get("title", "")
    if not title and "query_text" in item:
        qt = item.get("query_text", "")
        if "'" in qt:
            title = qt.split("'")[1] if len(qt.split("'")) > 1 else qt
    if not title and "query" in item:
        qt = item.get("query", "")
        if "'" in qt:
            title = qt.split("'")[1] if len(qt.split("'")) > 1 else qt
    return title


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def create_functional_query(title: str, item_type: str = "tree") -> str:
    """Create functional query wrapper."""
    if item_type.lower() in ["pagepearl", "url", "bookmark"]:
        return f"locate_url('{title}')"
    else:
        return f"locate_node('{title}')"


TEST_QUERIES = [
    ("Hyphanet", "Freenet & Hyphanet", "tree"),
    ("Feynman Lectures", "The Feynman Lectures on Physics", "pagepearl"),
    ("Quantum Mechanics", "Quantum mechanics", "tree"),
    ("wave", "Waves", "tree"),
    ("uncertainty", "Uncertainty principle", "tree"),
    ("Schrodinger", "Schrödinger equation", "tree"),
]


def run_dual_objective_experiment(
    model_name: str,
    targets: List[Dict],
    queries: List[Tuple[str, str, str]],
    alpha: float = 0.7,
    top_k: int = 5
) -> Dict:
    """
    Run dual-objective experiment with holdout.
    
    For each query:
    1. Hold out the exact target from the dataset
    2. Compute Input Objective: raw query → raw titles
    3. Compute Output Objective: raw query → functional targets
    4. Blend with alpha
    5. Check if held-out target would rank in top-k
    """
    results = {
        "model": model_name,
        "alpha": alpha,
        "num_targets": len(targets),
        "queries": []
    }
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Alpha: {alpha} (Output) / {1-alpha:.1f} (Input)")
    print(f"Targets: {len(targets)}")
    print(f"{'='*60}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_name)
    
    # Extract all titles and create functional queries
    all_titles = [extract_title(t) for t in targets]
    all_types = [t.get("type", "tree") for t in targets]
    all_functional = [create_functional_query(title, typ) for title, typ in zip(all_titles, all_types)]
    
    hits = 0
    total = 0
    
    for query, expected, expected_type in queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: '{expected}' (holdout)")
        
        # Find and hold out the expected target
        holdout_idx = None
        for i, title in enumerate(all_titles):
            if expected.lower() in title.lower() or title.lower() in expected.lower():
                holdout_idx = i
                break
        
        if holdout_idx is None:
            print(f"  ⚠ Expected target not found in dataset, skipping")
            continue
        
        # Create dataset without holdout
        train_titles = [t for i, t in enumerate(all_titles) if i != holdout_idx]
        train_functional = [f for i, f in enumerate(all_functional) if i != holdout_idx]
        
        # Embed everything
        t0 = time.time()
        
        # Query embeddings
        query_emb = model.encode(query)
        
        # Input Objective: raw titles (train set only)
        train_title_embs = model.encode(train_titles)
        
        # Output Objective: functional queries (train set only)  
        train_func_embs = model.encode(train_functional)
        
        # Also embed the held-out item
        holdout_title_emb = model.encode(all_titles[holdout_idx])
        holdout_func_emb = model.encode(all_functional[holdout_idx])
        
        # Compute scores for train set
        input_scores = np.array([cosine_similarity(query_emb, e) for e in train_title_embs])
        output_scores = np.array([cosine_similarity(query_emb, e) for e in train_func_embs])
        
        # Normalize (ReLU + L1)
        p_input = np.maximum(input_scores, 0)
        p_input /= p_input.sum() + 1e-10
        p_output = np.maximum(output_scores, 0)
        p_output /= p_output.sum() + 1e-10
        
        # Blend
        blended_train = alpha * p_output + (1 - alpha) * p_input
        
        # Compute score for held-out item
        holdout_input = cosine_similarity(query_emb, holdout_title_emb)
        holdout_output = cosine_similarity(query_emb, holdout_func_emb)
        
        # Normalize holdout against train distribution
        all_input = np.append(input_scores, holdout_input)
        all_output = np.append(output_scores, holdout_output)
        
        all_p_input = np.maximum(all_input, 0)
        all_p_input /= all_p_input.sum() + 1e-10
        all_p_output = np.maximum(all_output, 0)
        all_p_output /= all_p_output.sum() + 1e-10
        
        all_blended = alpha * all_p_output + (1 - alpha) * all_p_input
        holdout_blended = all_blended[-1]
        
        query_time = time.time() - t0
        
        # Determine rank of held-out item
        rank = 1 + np.sum(all_blended[:-1] > holdout_blended)
        
        query_result = {
            "query": query,
            "expected": expected,
            "query_time": query_time,
            "holdout_input_score": float(holdout_input),
            "holdout_output_score": float(holdout_output),
            "holdout_blended_score": float(holdout_blended),
            "rank": int(rank)
        }
        
        hit = rank <= top_k
        if hit:
            hits += 1
            print(f"  ✓ Rank #{rank} (Input: {holdout_input:.4f}, Output: {holdout_output:.4f}, Blended: {holdout_blended:.6f})")
        else:
            print(f"  ✗ Rank #{rank} (Input: {holdout_input:.4f}, Output: {holdout_output:.4f}, Blended: {holdout_blended:.6f})")
        
        # Show top-3 from train set for context
        top_train = np.argsort(blended_train)[::-1][:3]
        for i, idx in enumerate(top_train, 1):
            print(f"    Train #{i}: {train_titles[idx][:40]} [{blended_train[idx]:.6f}]")
        
        total += 1
        query_result["hit"] = hit
        results["queries"].append(query_result)
    
    results["hit_rate"] = hits / total if total > 0 else 0
    print(f"\nHit Rate (top-{top_k}): {hits}/{total} = {results['hit_rate']:.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Dual-objective experiment with holdout")
    parser.add_argument("--model", type=str, default="nomic-ai/modernbert-embed-base",
                       help="Embedding model to use for all objectives")
    parser.add_argument("--data", type=Path, default=Path("reports/pearltrees_targets_physics.jsonl"),
                       help="Path to JSONL data")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Blend weight for Output Objective (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=5, help="Consider hit if in top-k")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Load targets
    print(f"Loading targets from {args.data}...")
    targets = load_targets(args.data)
    print(f"Loaded {len(targets)} targets")
    
    # Run experiment
    results = run_dual_objective_experiment(
        args.model,
        targets,
        TEST_QUERIES,
        args.alpha,
        args.top_k
    )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
