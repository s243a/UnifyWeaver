#!/usr/bin/env python3
"""
Test logit-flux routing vs standard softmax routing.

Logit-flux combines similarity and density in log-odds space:
    P(i) ∝ odds(s_i)^(1/τ) × odds(c_i)^(w/τ)

This should:
- Boost routing to dense (well-represented) training regions
- Penalize routing to sparse (unusual) training regions
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
    """Compare standard vs logit-flux routing."""

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

    # Build full answer pool for harder test
    full_pool = build_full_answer_pool(train_pairs, val_pairs)

    print("\n" + "="*70)
    print("ROUTING METHOD COMPARISON")
    print("="*70)

    results = {}

    # Test configurations
    configs = [
        ("Standard softmax (τ=0.1)", RoutingConfig(
            temperature=0.1,
            use_logit_flux=False,
        )),
        ("Standard softmax (τ=0.05)", RoutingConfig(
            temperature=0.05,
            use_logit_flux=False,
        )),
        ("Logit-flux (w=0.5, τ=0.1)", RoutingConfig(
            temperature=0.1,
            use_logit_flux=True,
            density_weight=0.5,
        )),
        ("Logit-flux (w=1.0, τ=0.1)", RoutingConfig(
            temperature=0.1,
            use_logit_flux=True,
            density_weight=1.0,
        )),
        ("Logit-flux (w=2.0, τ=0.1)", RoutingConfig(
            temperature=0.1,
            use_logit_flux=True,
            density_weight=2.0,
        )),
        ("Logit-flux (w=1.0, τ=0.05)", RoutingConfig(
            temperature=0.05,
            use_logit_flux=True,
            density_weight=1.0,
        )),
    ]

    for name, config in configs:
        print(f"\n[{name}]")

        # Validation-only pool
        metrics_val = router.evaluate(val_pairs, config, answer_pool=None)
        print(f"  Val-only (100):  MRR={metrics_val['mrr']:.4f}  "
              f"R@1={metrics_val['recall_at_1']*100:.1f}%  "
              f"R@5={metrics_val['recall_at_5']*100:.1f}%")

        # Full pool
        metrics_full = router.evaluate(val_pairs, config, answer_pool=full_pool)
        print(f"  Full pool (500): MRR={metrics_full['mrr']:.4f}  "
              f"R@1={metrics_full['recall_at_1']*100:.1f}%  "
              f"R@5={metrics_full['recall_at_5']*100:.1f}%")

        results[name] = {
            "val_only": metrics_val,
            "full_pool": metrics_full,
        }

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY (Full Pool - harder test)")
    print("="*70)
    print(f"{'Method':<30} {'MRR':>8} {'R@1':>8} {'R@5':>8}")
    print("-" * 56)
    for name, r in results.items():
        m = r["full_pool"]
        print(f"{name:<30} {m['mrr']:>8.4f} {m['recall_at_1']*100:>7.1f}% {m['recall_at_5']*100:>7.1f}%")

    return results


def analyze_density_distribution(embedder_name: str = "all-minilm", max_pairs: int = 500):
    """Analyze the density distribution of training queries."""

    data_dir = "/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver/training-data"
    pairs = load_jsonl_pairs(data_dir, subdirs=["tailored"], max_pairs=max_pairs)
    qa_embeddings, cluster_ids = embed_pairs(pairs, embedder_name)

    train_pairs, val_pairs, _, _ = train_val_split(
        qa_embeddings, val_ratio=0.2, seed=42, cluster_ids=cluster_ids
    )

    router = PerPairRouting()
    router.train(train_pairs)

    print("\n" + "="*70)
    print("DENSITY DISTRIBUTION ANALYSIS")
    print("="*70)

    confidences = router._train_confidences
    print(f"Training query confidences:")
    print(f"  Min:    {confidences.min():.4f}")
    print(f"  Max:    {confidences.max():.4f}")
    print(f"  Mean:   {confidences.mean():.4f}")
    print(f"  Std:    {confidences.std():.4f}")
    print(f"  Median: {np.median(confidences):.4f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(confidences, p):.4f}")

    # How many are above/below neutral (0.5)?
    above = (confidences > 0.5).sum()
    below = (confidences < 0.5).sum()
    print(f"\nAbove 0.5 (boost): {above} ({above/len(confidences)*100:.1f}%)")
    print(f"Below 0.5 (penalty): {below} ({below/len(confidences)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-minilm",
                       choices=["all-minilm", "modernbert"])
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--analyze-density", action="store_true",
                       help="Analyze density distribution instead of comparing methods")
    args = parser.parse_args()

    if args.analyze_density:
        analyze_density_distribution(args.model, args.max_pairs)
    else:
        run_comparison(args.model, args.max_pairs)
