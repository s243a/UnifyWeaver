#!/usr/bin/env python3
"""
Test per-pair routing with full answer pool evaluation.

Compares two evaluation modes:
1. Validation-only: Rank against held-out answers only
2. Full pool: Rank against ALL answers (train + val)

The full pool is a harder test - must find exact answer among all candidates.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from per_pair_routing import (
    PerPairRouting,
    RoutingConfig,
    train_val_split,
    build_full_answer_pool,
)
from training_data_loader import load_jsonl_pairs, embed_pairs


def run_comparison(embedder_name: str = "all-minilm", max_pairs: int = 500):
    """Run comparison between evaluation modes."""

    # Load data
    data_dir = "/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver/training-data"
    print(f"\nLoading data from tailored folders...")

    pairs = load_jsonl_pairs(data_dir, subdirs=["tailored"], max_pairs=max_pairs)
    print(f"Loaded {len(pairs)} Q/A pairs")

    if len(pairs) < 50:
        print("Not enough pairs for meaningful evaluation")
        return

    # Embed
    print(f"\nEmbedding with {embedder_name}...")
    qa_embeddings, cluster_ids = embed_pairs(pairs, embedder_name)

    # Split
    train_pairs, val_pairs, train_clusters, val_clusters = train_val_split(
        qa_embeddings, val_ratio=0.2, seed=42, cluster_ids=cluster_ids
    )
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Train router
    print("\nTraining per-pair routing...")
    router = PerPairRouting(
        config=RoutingConfig(temperature=0.1),
        allow_scaling=True,
        normalize_queries=True,
    )
    stats = router.train(train_pairs, train_clusters)
    print(f"Trained on {stats['num_pairs']} pairs, dim={stats['embedding_dim']}")

    # Build full answer pool
    full_pool = build_full_answer_pool(train_pairs, val_pairs)
    print(f"Full answer pool: {full_pool.shape[0]} answers")

    # Evaluate both modes
    print("\n" + "="*60)
    print("EVALUATION COMPARISON")
    print("="*60)

    # Mode 1: Validation answers only
    print("\n[1] Validation answers only (easier)")
    metrics_val = router.evaluate(val_pairs, answer_pool=None)
    print(f"    Pool size: {metrics_val['n_answers_in_pool']}")
    print(f"    MRR:  {metrics_val['mrr']:.4f}")
    print(f"    R@1:  {metrics_val['recall_at_1']*100:.1f}%")
    print(f"    R@5:  {metrics_val['recall_at_5']*100:.1f}%")
    print(f"    R@10: {metrics_val['recall_at_10']*100:.1f}%")

    # Mode 2: Full answer pool
    print("\n[2] Full answer pool - train + val (harder)")
    metrics_full = router.evaluate(val_pairs, answer_pool=full_pool)
    print(f"    Pool size: {metrics_full['n_answers_in_pool']}")
    print(f"    MRR:  {metrics_full['mrr']:.4f}")
    print(f"    R@1:  {metrics_full['recall_at_1']*100:.1f}%")
    print(f"    R@5:  {metrics_full['recall_at_5']*100:.1f}%")
    print(f"    R@10: {metrics_full['recall_at_10']*100:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {embedder_name}")
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    print()
    print(f"{'Metric':<12} {'Val-Only':<12} {'Full Pool':<12} {'Delta':<12}")
    print("-" * 48)
    for metric in ['mrr', 'recall_at_1', 'recall_at_5']:
        v1 = metrics_val[metric]
        v2 = metrics_full[metric]
        delta = v2 - v1
        if 'recall' in metric:
            print(f"{metric:<12} {v1*100:>10.1f}% {v2*100:>10.1f}% {delta*100:>+10.1f}%")
        else:
            print(f"{metric:<12} {v1:>11.4f} {v2:>11.4f} {delta:>+11.4f}")

    return metrics_val, metrics_full


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-minilm",
                       choices=["all-minilm", "modernbert"])
    parser.add_argument("--max-pairs", type=int, default=500)
    args = parser.parse_args()

    run_comparison(args.model, args.max_pairs)
