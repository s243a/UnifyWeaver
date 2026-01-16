#!/usr/bin/env python3
"""
Analyze Federated Model Clusters and Suggest Refinement.

Analyzes centroid similarity and suggests optimal cluster count based on:
1. Effective rank (recommended) - spectral participation ratio
2. Covering number - 2^r where r captures 80% variance
3. √N heuristic - common K-means rule of thumb

Usage:
    python3 scripts/analyze_federated_clusters.py models/federated.pkl
    python3 scripts/analyze_federated_clusters.py models/federated.pkl --refine
    python3 scripts/analyze_federated_clusters.py models/federated.pkl --refine --target 100

See docs/proposals/DISTILLATION_MODES.md for theory.
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from types import ModuleType

# NumPy 2.0 compatibility shim
if not hasattr(np, '_core'):
    np_core = ModuleType('numpy._core')
    np_core.multiarray = np.core.multiarray
    np_core.umath = np.core.umath
    sys.modules['numpy._core'] = np_core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray


def analyze_centroids(centroids: np.ndarray, temperature: float = 0.1) -> dict:
    """
    Analyze centroid distribution and suggest optimal cluster count.

    Args:
        centroids: Cluster centroids (K x D)
        temperature: Softmax temperature for routing

    Returns:
        Analysis results with suggestions
    """
    K, D = centroids.shape

    # SVD analysis
    U, S, Vt = np.linalg.svd(centroids, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)

    # 1. Effective rank (participation ratio)
    effective_rank = int((np.sum(S)**2) / np.sum(S**2))

    # 2. Covering number (2^r for 80% variance)
    r_80 = int(np.searchsorted(cumvar, 0.80) + 1)
    covering_k = 2 ** min(r_80, 10)  # Cap at 2^10=1024

    # 3. √N heuristic
    sqrt_k = int(np.ceil(np.sqrt(K)))

    # Analyze softmax peakedness
    # Sample random queries and check weight distribution
    np.random.seed(42)
    sample_queries = centroids[np.random.choice(K, min(100, K), replace=False)]

    top1_weights = []
    top10_weights = []
    for q in sample_queries:
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        sims = q_norm @ centroids.T
        sims_shifted = (sims - np.max(sims)) / temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()
        sorted_w = np.sort(weights)[::-1]
        top1_weights.append(sorted_w[0])
        top10_weights.append(sorted_w[:10].sum())

    avg_top1 = np.mean(top1_weights)
    avg_top10 = np.mean(top10_weights)

    # Pairwise centroid similarity stats
    # Sample for efficiency
    n_sample = min(200, K)
    sample_idx = np.random.choice(K, n_sample, replace=False)
    sample_centroids = centroids[sample_idx]
    sample_centroids_norm = sample_centroids / (np.linalg.norm(sample_centroids, axis=1, keepdims=True) + 1e-8)
    sim_matrix = sample_centroids_norm @ sample_centroids_norm.T
    np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity

    return {
        'current_k': K,
        'embed_dim': D,
        'temperature': temperature,
        'suggestions': {
            'effective_rank': effective_rank,
            'covering_80pct': covering_k,
            'sqrt_n': sqrt_k,
            'recommended': effective_rank  # Default recommendation
        },
        'variance_analysis': {
            'r_50pct': int(np.searchsorted(cumvar, 0.50) + 1),
            'r_80pct': r_80,
            'r_90pct': int(np.searchsorted(cumvar, 0.90) + 1),
            'r_95pct': int(np.searchsorted(cumvar, 0.95) + 1),
        },
        'routing_analysis': {
            'avg_top1_weight': avg_top1,
            'avg_top10_weight': avg_top10,
            'centroid_sim_mean': float(sim_matrix.mean()),
            'centroid_sim_max': float(sim_matrix.max()),
        },
        'over_segmented': avg_top1 < 0.5,  # If top-1 weight < 50%, too many clusters
    }


def print_analysis(analysis: dict):
    """Pretty-print the analysis results."""
    print("=" * 60)
    print("Federated Model Cluster Analysis")
    print("=" * 60)

    print(f"\nCurrent model: K={analysis['current_k']} clusters, D={analysis['embed_dim']}")
    print(f"Temperature: {analysis['temperature']}")

    print("\n--- Suggested Cluster Counts ---")
    sugg = analysis['suggestions']
    print(f"  1. Effective rank (recommended): K = {sugg['effective_rank']}")
    print(f"  2. Covering (80% variance):      K = {sugg['covering_80pct']}")
    print(f"  3. √N heuristic:                 K = {sugg['sqrt_n']}")

    print("\n--- Variance Analysis ---")
    var = analysis['variance_analysis']
    print(f"  Dimensions for 50% variance: r = {var['r_50pct']}")
    print(f"  Dimensions for 80% variance: r = {var['r_80pct']}")
    print(f"  Dimensions for 90% variance: r = {var['r_90pct']}")
    print(f"  Dimensions for 95% variance: r = {var['r_95pct']}")

    print("\n--- Routing Analysis ---")
    route = analysis['routing_analysis']
    print(f"  Avg top-1 softmax weight:  {route['avg_top1_weight']*100:.1f}%")
    print(f"  Avg top-10 softmax weight: {route['avg_top10_weight']*100:.1f}%")
    print(f"  Centroid similarity mean:  {route['centroid_sim_mean']:.3f}")
    print(f"  Centroid similarity max:   {route['centroid_sim_max']:.3f}")

    print("\n--- Diagnosis ---")
    if analysis['over_segmented']:
        print("  ⚠ Model appears OVER-SEGMENTED (top-1 weight < 50%)")
        print(f"  → Consider reducing to K ≈ {sugg['recommended']} clusters")
    else:
        print("  ✓ Cluster count appears reasonable")

    print("=" * 60)


def metacluster_centroids(
    centroids: np.ndarray,
    target_k: int,
    method: str = "kmeans"
) -> tuple:
    """
    Meta-cluster centroids into fewer groups.

    Args:
        centroids: Original centroids (K x D)
        target_k: Target number of meta-clusters
        method: Clustering method ('kmeans', 'hierarchical')

    Returns:
        (meta_centroids, cluster_assignments)
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering

    if method == "kmeans":
        kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(centroids)
        meta_centroids = kmeans.cluster_centers_
    elif method == "hierarchical":
        clustering = AgglomerativeClustering(n_clusters=target_k)
        labels = clustering.fit_predict(centroids)
        # Compute centroids manually
        meta_centroids = np.zeros((target_k, centroids.shape[1]))
        for i in range(target_k):
            mask = labels == i
            if mask.sum() > 0:
                meta_centroids[i] = centroids[mask].mean(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return meta_centroids, labels


def merge_w_matrices(
    cluster_dir: Path,
    cluster_ids: list,
    meta_labels: np.ndarray,
    target_k: int,
    method: str = "weighted_avg"
) -> list:
    """
    Merge W matrices based on meta-clustering.

    Args:
        cluster_dir: Directory with cluster .npz files
        cluster_ids: Original cluster IDs
        meta_labels: Meta-cluster assignment for each original cluster
        target_k: Number of meta-clusters
        method: Merge method ('weighted_avg', 'first')

    Returns:
        List of merged W matrices
    """
    merged_W = []

    for meta_idx in range(target_k):
        # Find original clusters in this meta-cluster
        member_indices = np.where(meta_labels == meta_idx)[0]

        if len(member_indices) == 0:
            continue

        # Load and merge W matrices
        W_list = []
        sizes = []

        for orig_idx in member_indices:
            cid = cluster_ids[orig_idx]
            cluster_path = cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                data = np.load(cluster_path)
                W = data["W_stack"][0]
                n_samples = len(data["indices"])
                W_list.append(W)
                sizes.append(n_samples)

        if not W_list:
            continue

        if method == "weighted_avg":
            # Weighted average by cluster size
            total = sum(sizes)
            merged = sum(W * (s / total) for W, s in zip(W_list, sizes))
        elif method == "first":
            # Just use the first (largest) cluster's W
            merged = W_list[np.argmax(sizes)]
        else:
            merged = W_list[0]

        merged_W.append(merged)

    return merged_W


def main():
    parser = argparse.ArgumentParser(
        description="Analyze federated model clusters and suggest refinement"
    )
    parser.add_argument("input_model", type=Path,
                       help="Input federated model (.pkl)")
    parser.add_argument("--refine", action="store_true",
                       help="Perform meta-clustering refinement")
    parser.add_argument("--target", type=int, default=None,
                       help="Target cluster count (default: use effective rank)")
    parser.add_argument("--method", choices=["effective_rank", "covering", "sqrt_n"],
                       default="effective_rank",
                       help="Method for automatic target selection")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output refined model path")
    parser.add_argument("--merge-method", choices=["weighted_avg", "first"],
                       default="weighted_avg",
                       help="How to merge W matrices")

    args = parser.parse_args()

    if not args.input_model.exists():
        print(f"Error: Model not found: {args.input_model}")
        return 1

    # Load model
    print(f"Loading {args.input_model}...")
    with open(args.input_model, "rb") as f:
        meta = pickle.load(f)

    centroids = meta["cluster_centroids"]
    temperature = meta.get("temperature", 0.1)
    cluster_ids = meta["cluster_ids"]

    # Analyze
    analysis = analyze_centroids(centroids, temperature)
    print_analysis(analysis)

    if not args.refine:
        print("\nTo refine, run with --refine flag")
        return 0

    # Determine target K
    if args.target:
        target_k = args.target
    else:
        sugg = analysis['suggestions']
        if args.method == "effective_rank":
            target_k = sugg['effective_rank']
        elif args.method == "covering":
            target_k = sugg['covering_80pct']
        elif args.method == "sqrt_n":
            target_k = sugg['sqrt_n']
        else:
            target_k = sugg['recommended']

    if target_k >= analysis['current_k']:
        print(f"\nTarget K={target_k} >= current K={analysis['current_k']}, no refinement needed")
        return 0

    print(f"\n--- Refining to K={target_k} clusters ---")

    # Meta-cluster
    print("Meta-clustering centroids...")
    meta_centroids, meta_labels = metacluster_centroids(centroids, target_k)

    # Merge W matrices
    cluster_dir = Path(meta.get("cluster_dir", args.input_model.with_suffix('')))
    if cluster_dir.exists():
        print(f"Merging W matrices from {cluster_dir}...")
        merged_W = merge_w_matrices(
            cluster_dir, cluster_ids, meta_labels, target_k, args.merge_method
        )
        print(f"Created {len(merged_W)} merged W matrices")
    else:
        print(f"Warning: Cluster directory not found: {cluster_dir}")
        merged_W = None

    # Re-analyze refined model
    print("\nRe-analyzing refined centroids...")
    refined_analysis = analyze_centroids(meta_centroids, temperature)

    print(f"\nRefinement results:")
    print(f"  Original: K={analysis['current_k']}, top-1 weight={analysis['routing_analysis']['avg_top1_weight']*100:.1f}%")
    print(f"  Refined:  K={target_k}, top-1 weight={refined_analysis['routing_analysis']['avg_top1_weight']*100:.1f}%")

    # Save if output specified
    if args.output:
        output_dir = args.output.with_suffix('')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save refined model
        refined_meta = {
            "cluster_ids": [f"meta_{i}" for i in range(target_k)],
            "cluster_centroids": meta_centroids,
            "temperature": temperature,
            "global_target_ids": meta.get("global_target_ids", []),
            "global_target_titles": meta.get("global_target_titles", []),
            "metrics": {"refined_from": str(args.input_model), "original_k": analysis['current_k']},
            "num_clusters": target_k,
            "cluster_dir": str(output_dir),
        }

        with open(args.output, "wb") as f:
            pickle.dump(refined_meta, f)
        print(f"\nSaved refined model to {args.output}")

        # Save merged W matrices
        if merged_W:
            for i, W in enumerate(merged_W):
                np.savez_compressed(
                    output_dir / f"meta_{i}.npz",
                    W_stack=W[np.newaxis, :, :],  # Add batch dim
                    centroids=meta_centroids[i:i+1],
                    indices=np.array([])  # No original indices for merged
                )
            print(f"Saved {len(merged_W)} W matrices to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
