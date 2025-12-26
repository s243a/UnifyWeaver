#!/usr/bin/env python3
"""
Scale-up test: Run per-pair routing on the full dataset.

Uses embeddings cache to avoid recomputation.
"""

import sys
import time
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
from training_data_loader import load_and_embed_with_cache


DATA_DIR = "/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver/training-data"


def run_scale_test(
    embedder_name: str = "all-minilm",
    subdirs: list = None,
    max_pairs: int = None,
    val_ratio: float = 0.2,
):
    """Run full-scale evaluation with caching."""

    if subdirs is None:
        subdirs = ["tailored"]

    print("="*70)
    print(f"SCALE-UP TEST: {embedder_name}")
    print(f"Subdirs: {subdirs}, Max pairs: {max_pairs or 'all'}")
    print("="*70)

    # Load with cache
    start = time.time()
    qa_embeddings, cluster_ids, pair_ids = load_and_embed_with_cache(
        DATA_DIR,
        embedder_name=embedder_name,
        subdirs=subdirs,
        max_pairs=max_pairs,
    )
    load_time = time.time() - start
    print(f"Load time: {load_time:.2f}s")

    n_pairs = len(qa_embeddings)
    dim = qa_embeddings[0][0].shape[0]
    print(f"Pairs: {n_pairs}, Dimension: {dim}")

    # Split
    train_pairs, val_pairs, train_clusters, val_clusters = train_val_split(
        qa_embeddings, val_ratio=val_ratio, seed=42, cluster_ids=cluster_ids
    )
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Train
    print("\nTraining per-pair routing...")
    start = time.time()
    router = PerPairRouting(
        config=RoutingConfig(temperature=0.1),
        allow_scaling=True,
        normalize_queries=True,
    )
    stats = router.train(train_pairs, train_clusters)
    train_time = time.time() - start
    print(f"Train time: {train_time:.2f}s")

    # Build full answer pool
    full_pool = build_full_answer_pool(train_pairs, val_pairs)

    # Evaluate
    print("\nEvaluating...")
    start = time.time()

    # Validation-only
    metrics_val = router.evaluate(val_pairs, answer_pool=None)

    # Full pool
    metrics_full = router.evaluate(val_pairs, answer_pool=full_pool)

    eval_time = time.time() - start
    print(f"Eval time: {eval_time:.2f}s")

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nModel: {embedder_name}")
    print(f"Total pairs: {n_pairs}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    print(f"\n[Validation answers only ({len(val_pairs)} candidates)]")
    print(f"  MRR:  {metrics_val['mrr']:.4f}")
    print(f"  R@1:  {metrics_val['recall_at_1']*100:.1f}%")
    print(f"  R@5:  {metrics_val['recall_at_5']*100:.1f}%")
    print(f"  R@10: {metrics_val['recall_at_10']*100:.1f}%")

    print(f"\n[Full answer pool ({len(full_pool)} candidates)]")
    print(f"  MRR:  {metrics_full['mrr']:.4f}")
    print(f"  R@1:  {metrics_full['recall_at_1']*100:.1f}%")
    print(f"  R@5:  {metrics_full['recall_at_5']*100:.1f}%")
    print(f"  R@10: {metrics_full['recall_at_10']*100:.1f}%")

    print(f"\nTimings:")
    print(f"  Load:  {load_time:.2f}s")
    print(f"  Train: {train_time:.2f}s")
    print(f"  Eval:  {eval_time:.2f}s")

    return {
        "model": embedder_name,
        "n_pairs": n_pairs,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "val_only": metrics_val,
        "full_pool": metrics_full,
        "timings": {
            "load": load_time,
            "train": train_time,
            "eval": eval_time,
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-minilm",
                       choices=["all-minilm", "modernbert"])
    parser.add_argument("--max-pairs", type=int, default=None,
                       help="Max pairs (default: all)")
    parser.add_argument("--subdirs", nargs="+", default=["tailored"],
                       help="Subdirectories to include")
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare both models")
    args = parser.parse_args()

    if args.compare_models:
        results = {}
        for model in ["all-minilm", "modernbert"]:
            results[model] = run_scale_test(
                embedder_name=model,
                subdirs=args.subdirs,
                max_pairs=args.max_pairs,
            )
            print("\n")

        # Summary comparison
        print("="*70)
        print("MODEL COMPARISON (Full Pool)")
        print("="*70)
        print(f"{'Model':<15} {'Pairs':>8} {'MRR':>8} {'R@1':>8} {'R@5':>8}")
        print("-" * 50)
        for model, r in results.items():
            m = r["full_pool"]
            print(f"{model:<15} {r['n_pairs']:>8} {m['mrr']:>8.4f} "
                  f"{m['recall_at_1']*100:>7.1f}% {m['recall_at_5']*100:>7.1f}%")
    else:
        run_scale_test(
            embedder_name=args.model,
            subdirs=args.subdirs,
            max_pairs=args.max_pairs,
        )
