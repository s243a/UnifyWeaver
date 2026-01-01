#!/usr/bin/env python3
"""
Evaluate Federated Models on Holdout Test Set

Computes Hit@K and MRR with bootstrap confidence intervals.

Usage:
    python scripts/evaluate_holdout.py \
        --test datasets/pearltrees_public/test.jsonl \
        --models public_minilm public_bge public_nomic
"""

import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from infer_pearltrees_federated import FederatedInferenceEngine

# Model configs
MODELS = {
    "public_minilm": {
        "pkl": "models/public_federated_minilm.pkl",
        "embedder": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "public_bge": {
        "pkl": "models/public_federated_bge.pkl",
        "embedder": "BAAI/bge-base-en-v1.5"
    },
    "public_nomic": {
        "pkl": "models/public_federated_nomic.pkl",
        "embedder": "nomic-ai/nomic-embed-text-v1.5"
    }
}


def load_test_set(path: Path) -> List[Dict]:
    """Load test set from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate_model(engine: FederatedInferenceEngine, test_items: List[Dict], 
                   k_values: List[int] = [1, 5, 10], top_k: int = 20) -> Dict:
    """
    Evaluate model on test set.
    Returns hit@k rates and MRR.
    """
    hits = {k: [] for k in k_values}
    reciprocal_ranks = []
    
    for item in test_items:
        query = item.get("raw_title", "")
        tree_id = str(item.get("tree_id", ""))
        
        if not query or not tree_id:
            continue
        
        # Search
        candidates = engine.search(query, top_k=top_k)
        
        # Find rank of expected target
        rank = None
        for i, c in enumerate(candidates, 1):
            if str(c.tree_id) == tree_id:
                rank = i
                break
        
        # Compute metrics
        for k in k_values:
            hit = 1 if (rank is not None and rank <= k) else 0
            hits[k].append(hit)
        
        rr = 1.0 / rank if rank is not None else 0.0
        reciprocal_ranks.append(rr)
    
    # Aggregate
    results = {}
    for k in k_values:
        results[f"hit@{k}"] = np.mean(hits[k])
    results["mrr"] = np.mean(reciprocal_ranks)
    results["n_queries"] = len(reciprocal_ranks)
    
    return results, hits, reciprocal_ranks


def bootstrap_ci(values: List[float], n_samples: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    means = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, 
                       default=Path("datasets/pearltrees_public/test.jsonl"))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--output", type=Path, 
                       default=Path("sandbox/paper-minimum-projection/statistical_results.json"))
    parser.add_argument("--limit", type=int, default=None, help="Limit test queries")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Holdout Evaluation")
    print("="*70)
    
    # Load test set
    print(f"\nLoading test set: {args.test}")
    test_items = load_test_set(args.test)
    if args.limit:
        test_items = test_items[:args.limit]
    print(f"  Test queries: {len(test_items)}")
    
    all_results = {}
    
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue
        
        config = MODELS[model_name]
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print("="*70)
        
        # Load model
        engine = FederatedInferenceEngine(Path(config["pkl"]), config["embedder"])
        
        # Evaluate
        t0 = time.time()
        results, hits, rrs = evaluate_model(engine, test_items)
        elapsed = time.time() - t0
        
        # Bootstrap CIs
        print(f"\nResults ({elapsed:.1f}s):")
        model_results = {"raw": results}
        
        for k in [1, 5, 10]:
            hit_rate = results[f"hit@{k}"]
            low, high = bootstrap_ci(hits[k])
            model_results[f"hit@{k}"] = {"mean": hit_rate, "ci_low": low, "ci_high": high}
            print(f"  Hit@{k}: {hit_rate:.3f} ({low:.3f} - {high:.3f})")
        
        mrr = results["mrr"]
        mrr_low, mrr_high = bootstrap_ci(rrs)
        model_results["mrr"] = {"mean": mrr, "ci_low": mrr_low, "ci_high": mrr_high}
        print(f"  MRR: {mrr:.3f} ({mrr_low:.3f} - {mrr_high:.3f})")
        
        all_results[model_name] = model_results
        
        del engine  # Free memory
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Summary table
    print("\n" + "="*70)
    print("Summary Table")
    print("="*70)
    print(f"{'Model':<15} {'Hit@1':>12} {'Hit@5':>12} {'Hit@10':>12} {'MRR':>12}")
    print("-"*63)
    for model, res in all_results.items():
        h1 = f"{res['hit@1']['mean']:.1%}"
        h5 = f"{res['hit@5']['mean']:.1%}"
        h10 = f"{res['hit@10']['mean']:.1%}"
        mrr = f"{res['mrr']['mean']:.3f}"
        print(f"{model:<15} {h1:>12} {h5:>12} {h10:>12} {mrr:>12}")


if __name__ == "__main__":
    main()
