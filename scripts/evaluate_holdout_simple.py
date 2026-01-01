#!/usr/bin/env python3
"""
Evaluate Holdout Federated Models

Evaluates models trained with train_holdout_federated.py on the test set.

Usage:
    python scripts/evaluate_holdout_simple.py --limit 100
"""

import argparse
import json
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Dict, List

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))
sys.path.insert(0, str(Path(__file__).parent))

from train_pearltrees_federated import SimpleEmbedder

# Model configs
MODELS = {
    "holdout_minilm": {
        "pkl": "models/holdout_federated_minilm.pkl",
        "embedder": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "holdout_bge": {
        "pkl": "models/holdout_federated_bge.pkl",
        "embedder": "BAAI/bge-base-en-v1.5"
    },
    "holdout_nomic": {
        "pkl": "models/holdout_federated_nomic.pkl",
        "embedder": "nomic-ai/nomic-embed-text-v1.5"
    }
}


class HoldoutInference:
    """Simple inference for holdout models."""
    
    def __init__(self, pkl_path: Path, embedder_name: str):
        print(f"Loading model from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.meta = pickle.load(f)
        
        self.model_dir = pkl_path.parent / pkl_path.stem
        
        # Load target embeddings
        target_file = self.model_dir / "target_embeddings.npz"
        data = np.load(target_file)
        self.target_embeddings = data['embeddings']
        self.target_tree_ids = self.meta.get('target_tree_ids', [])
        self.target_titles = self.meta.get('target_titles', [])
        
        print(f"  Targets: {len(self.target_tree_ids)}")
        
        # Load embedder
        self.embedder = SimpleEmbedder(embedder_name)
        
    def search(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        Search for query. Returns list of (tree_id, score, title).
        Uses cosine similarity against all target embeddings.
        """
        # Embed query
        query_emb = self.embedder.encode([query])[0]
        
        # Compute cosine similarity
        query_emb = query_emb / np.linalg.norm(query_emb)
        target_norms = np.linalg.norm(self.target_embeddings, axis=1, keepdims=True)
        target_normed = self.target_embeddings / (target_norms + 1e-9)
        
        scores = target_normed @ query_emb
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.target_tree_ids[idx],
                float(scores[idx]),
                self.target_titles[idx]
            ))
        
        return results


def load_test(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate(engine: HoldoutInference, test_items: List[Dict], k_values=[1, 5, 10]):
    """Evaluate model on test set."""
    hits = {k: [] for k in k_values}
    reciprocal_ranks = []
    
    for item in test_items:
        query = item.get("raw_title", "")
        expected_tree_id = str(item.get("tree_id", ""))
        
        if not query or not expected_tree_id:
            continue
        
        results = engine.search(query, top_k=max(k_values) * 2)
        
        # Find rank
        rank = None
        for i, (tid, score, title) in enumerate(results, 1):
            if str(tid) == expected_tree_id:
                rank = i
                break
        
        for k in k_values:
            hit = 1 if (rank is not None and rank <= k) else 0
            hits[k].append(hit)
        
        rr = 1.0 / rank if rank is not None else 0.0
        reciprocal_ranks.append(rr)
    
    return {
        "hit@1": np.mean(hits[1]),
        "hit@5": np.mean(hits[5]),
        "hit@10": np.mean(hits[10]),
        "mrr": np.mean(reciprocal_ranks),
        "n": len(reciprocal_ranks)
    }


def bootstrap_ci(values, n_samples=1000, ci=0.95):
    values = np.array(values)
    means = [np.mean(np.random.choice(values, len(values))) for _ in range(n_samples)]
    alpha = (1 - ci) / 2
    return np.percentile(means, [alpha * 100, (1 - alpha) * 100])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("datasets/pearltrees_public/test.jsonl"))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("sandbox/paper-minimum-projection/holdout_results.json"))
    
    args = parser.parse_args()
    
    print("="*70)
    print("Holdout Evaluation")
    print("="*70)
    
    test_items = load_test(args.test)
    if args.limit:
        test_items = test_items[:args.limit]
    print(f"Test queries: {len(test_items)}")
    
    all_results = {}
    
    for model_name in args.models:
        config = MODELS[model_name]
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print("="*70)
        
        engine = HoldoutInference(Path(config["pkl"]), config["embedder"])
        results = evaluate(engine, test_items)
        
        print(f"  Hit@1:  {results['hit@1']:.3f}")
        print(f"  Hit@5:  {results['hit@5']:.3f}")
        print(f"  Hit@10: {results['hit@10']:.3f}")
        print(f"  MRR:    {results['mrr']:.3f}")
        
        all_results[model_name] = results
        
        del engine
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Summary table
    print("\n" + "="*70)
    print(f"{'Model':<20} {'Hit@1':>10} {'Hit@5':>10} {'Hit@10':>10} {'MRR':>10}")
    print("-"*60)
    for m, r in all_results.items():
        print(f"{m:<20} {r['hit@1']:>10.1%} {r['hit@5']:>10.1%} {r['hit@10']:>10.1%} {r['mrr']:>10.3f}")


if __name__ == "__main__":
    main()
