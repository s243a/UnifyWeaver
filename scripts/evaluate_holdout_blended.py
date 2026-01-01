#!/usr/bin/env python3
"""
Evaluate Holdout with Score Blending and RRF

Extends evaluate_holdout_simple.py with ensemble methods.
"""

import argparse
import json
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))
sys.path.insert(0, str(Path(__file__).parent))

from train_pearltrees_federated import SimpleEmbedder


class HoldoutInference:
    """Simple inference for holdout models."""
    
    def __init__(self, pkl_path: Path, embedder_name: str):
        with open(pkl_path, 'rb') as f:
            self.meta = pickle.load(f)
        
        self.model_dir = pkl_path.parent / pkl_path.stem
        target_file = self.model_dir / "target_embeddings.npz"
        data = np.load(target_file)
        self.target_embeddings = data['embeddings']
        self.target_tree_ids = self.meta.get('target_tree_ids', [])
        self.embedder = SimpleEmbedder(embedder_name)
        
        # Normalize targets
        norms = np.linalg.norm(self.target_embeddings, axis=1, keepdims=True)
        self.target_normed = self.target_embeddings / (norms + 1e-9)
        
    def get_scores(self, query: str) -> Dict[str, float]:
        """Get scores for all targets."""
        query_emb = self.embedder.encode([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores = self.target_normed @ query_emb
        return {self.target_tree_ids[i]: float(scores[i]) for i in range(len(scores))}


def load_test(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def blend_scores(all_scores: Dict[str, Dict], weights: Dict[str, float]) -> Dict[str, float]:
    """Blend scores from multiple models."""
    blended = {}
    for tree_id in all_scores[list(all_scores.keys())[0]]:
        total = 0.0
        total_weight = 0.0
        for model, scores in all_scores.items():
            if model in weights and tree_id in scores:
                total += weights[model] * scores[tree_id]
                total_weight += weights[model]
        if total_weight > 0:
            blended[tree_id] = total / total_weight
    return blended


def rrf_blend(all_scores: Dict[str, Dict], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion."""
    rrf_scores = {}
    for model, scores in all_scores.items():
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (tree_id, score) in enumerate(sorted_items, 1):
            if tree_id not in rrf_scores:
                rrf_scores[tree_id] = 0.0
            rrf_scores[tree_id] += 1.0 / (k + rank)
    return rrf_scores


def evaluate(results: Dict[str, float], expected: str, k_values=[1, 5, 10]):
    """Evaluate single query."""
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    rank = None
    for i, (tid, score) in enumerate(sorted_items[:max(k_values)*2], 1):
        if tid == expected:
            rank = i
            break
    return rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("datasets/pearltrees_public/test.jsonl"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("sandbox/paper-minimum-projection/blended_results.json"))
    
    args = parser.parse_args()
    
    print("Loading models...")
    models = {
        "minilm": HoldoutInference(Path("models/holdout_federated_minilm.pkl"), "sentence-transformers/all-MiniLM-L6-v2"),
        "bge": HoldoutInference(Path("models/holdout_federated_bge.pkl"), "BAAI/bge-base-en-v1.5"),
        "nomic": HoldoutInference(Path("models/holdout_federated_nomic.pkl"), "nomic-ai/nomic-embed-text-v1.5"),
    }
    
    test_items = load_test(args.test)[:args.limit]
    print(f"Test queries: {len(test_items)}")
    
    # Configurations to test
    configs = {
        "minilm": {"minilm": 1.0},
        "bge": {"bge": 1.0},
        "nomic": {"nomic": 1.0},
        "blend_equal": {"minilm": 1.0, "bge": 1.0, "nomic": 1.0},
        "blend_minilm_bge": {"minilm": 1.0, "bge": 1.0},
        "blend_bge_nomic": {"bge": 1.0, "nomic": 1.0},
        "rrf_all": "rrf",
    }
    
    results = {cfg: {"hits@1": [], "hits@5": [], "hits@10": [], "rr": []} for cfg in configs}
    
    for i, item in enumerate(test_items):
        query = item.get("raw_title", "")
        expected = str(item.get("tree_id", ""))
        
        if not query or not expected:
            continue
        
        # Get scores from all models
        all_scores = {name: m.get_scores(query) for name, m in models.items()}
        
        for cfg_name, cfg in configs.items():
            if cfg == "rrf":
                scores = rrf_blend(all_scores)
            else:
                scores = blend_scores(all_scores, cfg)
            
            rank = evaluate(scores, expected)
            
            for k in [1, 5, 10]:
                hit = 1 if (rank and rank <= k) else 0
                results[cfg_name][f"hits@{k}"].append(hit)
            
            rr = 1.0/rank if rank else 0.0
            results[cfg_name]["rr"].append(rr)
        
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_items)}")
    
    # Aggregate
    final = {}
    for cfg, data in results.items():
        final[cfg] = {
            "hit@1": np.mean(data["hits@1"]),
            "hit@5": np.mean(data["hits@5"]),
            "hit@10": np.mean(data["hits@10"]),
            "mrr": np.mean(data["rr"])
        }
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(final, f, indent=2)
    
    # Print table
    print("\n" + "="*70)
    print(f"{'Config':<20} {'Hit@1':>10} {'Hit@5':>10} {'Hit@10':>10} {'MRR':>10}")
    print("-"*60)
    for cfg, r in final.items():
        print(f"{cfg:<20} {r['hit@1']:>10.1%} {r['hit@5']:>10.1%} {r['hit@10']:>10.1%} {r['mrr']:>10.3f}")


if __name__ == "__main__":
    main()
