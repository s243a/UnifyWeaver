#!/usr/bin/env python3
"""
Mixed Model Dual-Objective Experiment

Tests dual-objective scoring with DIFFERENT models for Input vs Output objectives.

- Input Objective: Raw query → Raw title embeddings (Model A)
- Output Objective: Raw query → Functional query embeddings (Model B)

Usage:
    source ~/.hf-env/bin/activate
    python3 scripts/experiment_mixed_models.py \
        --input-model nomic-ai/modernbert-embed-base \
        --output-model nomic-ai/nomic-embed-text-v1.5 \
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
                items.append(json.loads(line))
    return items


def extract_title(item: Dict) -> str:
    """Extract title from item."""
    title = item.get("raw_title", "") or item.get("title", "")
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


def run_mixed_model_experiment(
    input_model_name: str,
    output_model_name: str,
    targets: List[Dict],
    queries: List[Tuple[str, str, str]],
    alpha: float = 0.7,
    top_k_values: List[int] = [1, 5, 10]
) -> Dict:
    """
    Run dual-objective experiment with mixed models.
    
    Input Objective: input_model encodes query → input_model encodes raw titles
    Output Objective: output_model encodes query → output_model encodes functional targets
    """
    results = {
        "input_model": input_model_name,
        "output_model": output_model_name,
        "alpha": alpha,
        "num_targets": len(targets),
        "queries": [],
        "hit_rates": {}
    }
    
    print(f"\n{'='*60}")
    print(f"Input Model:  {input_model_name}")
    print(f"Output Model: {output_model_name}")
    print(f"Alpha: {alpha} (Output) / {1-alpha:.1f} (Input)")
    print(f"Targets: {len(targets)}")
    print(f"{'='*60}")
    
    # Load models
    print("\nLoading input model...")
    input_model = load_model(input_model_name)
    
    if output_model_name != input_model_name:
        print("Loading output model...")
        output_model = load_model(output_model_name)
    else:
        output_model = input_model
    
    # Extract all titles and create functional queries
    all_titles = [extract_title(t) for t in targets]
    all_types = [t.get("type", "tree") for t in targets]
    all_functional = [create_functional_query(title, typ) for title, typ in zip(all_titles, all_types)]
    
    # Pre-embed all targets with both models
    print("Embedding targets with input model (raw titles)...")
    all_title_embs = input_model.encode(all_titles, show_progress_bar=True)
    
    print("Embedding targets with output model (functional)...")
    all_func_embs = output_model.encode(all_functional, show_progress_bar=True)
    
    ranks = []
    
    for query, expected, expected_type in queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: '{expected}'")
        
        # Find expected target
        holdout_idx = None
        for i, title in enumerate(all_titles):
            if expected.lower() in title.lower() or title.lower() in expected.lower():
                holdout_idx = i
                break
        
        if holdout_idx is None:
            print(f"  ⚠ Expected target not found, skipping")
            continue
        
        t0 = time.time()
        
        # Embed query with both models
        query_emb_input = input_model.encode(query)
        query_emb_output = output_model.encode(query)
        
        # Input Objective: query (input model) → titles (input model)
        input_scores = np.array([cosine_similarity(query_emb_input, e) for e in all_title_embs])
        
        # Output Objective: query (output model) → functional (output model)
        output_scores = np.array([cosine_similarity(query_emb_output, e) for e in all_func_embs])
        
        # Normalize (ReLU + L1)
        p_input = np.maximum(input_scores, 0)
        p_input /= p_input.sum() + 1e-10
        p_output = np.maximum(output_scores, 0)
        p_output /= p_output.sum() + 1e-10
        
        # Blend
        blended = alpha * p_output + (1 - alpha) * p_input
        
        query_time = time.time() - t0
        
        # Determine rank of target
        target_score = blended[holdout_idx]
        rank = 1 + np.sum(blended > target_score)
        ranks.append(rank)
        
        query_result = {
            "query": query,
            "expected": expected,
            "query_time": query_time,
            "input_score": float(input_scores[holdout_idx]),
            "output_score": float(output_scores[holdout_idx]),
            "blended_score": float(target_score),
            "rank": int(rank)
        }
        
        # Show result
        hit_5 = rank <= 5
        if hit_5:
            print(f"  ✓ Rank #{rank} (Input: {input_scores[holdout_idx]:.4f}, Output: {output_scores[holdout_idx]:.4f})")
        else:
            print(f"  ✗ Rank #{rank} (Input: {input_scores[holdout_idx]:.4f}, Output: {output_scores[holdout_idx]:.4f})")
        
        # Show top-3
        top_3 = np.argsort(blended)[::-1][:3]
        for i, idx in enumerate(top_3, 1):
            print(f"    #{i}: {all_titles[idx][:45]} [{blended[idx]:.6f}]")
        
        results["queries"].append(query_result)
    
    # Compute hit rates at different k
    for k in top_k_values:
        hits = sum(1 for r in ranks if r <= k)
        rate = hits / len(ranks) if ranks else 0
        results["hit_rates"][f"hit@{k}"] = rate
        print(f"\nHit@{k}: {hits}/{len(ranks)} = {rate:.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Mixed model dual-objective experiment")
    parser.add_argument("--input-model", type=str, default="nomic-ai/modernbert-embed-base",
                       help="Model for Input Objective (raw titles)")
    parser.add_argument("--output-model", type=str, default="nomic-ai/nomic-embed-text-v1.5",
                       help="Model for Output Objective (functional queries)")
    parser.add_argument("--data", type=Path, default=Path("reports/pearltrees_targets_physics.jsonl"))
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--output", type=Path, help="Save results to JSON")
    
    args = parser.parse_args()
    
    print(f"Loading targets from {args.data}...")
    targets = load_targets(args.data)
    print(f"Loaded {len(targets)} targets")
    
    results = run_mixed_model_experiment(
        args.input_model,
        args.output_model,
        targets,
        TEST_QUERIES,
        args.alpha
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
