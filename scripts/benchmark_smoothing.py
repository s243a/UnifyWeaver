#!/usr/bin/env python3
"""
Benchmark smoothing approaches on real training data.

Measures:
1. Accuracy: MSE, cosine similarity, retrieval precision@k
2. Computational cost: training time, inference time
3. Scalability: varying K, varying cluster count

Usage:
    python scripts/benchmark_smoothing.py
    python scripts/benchmark_smoothing.py --embedding-model e5-small
    python scripts/benchmark_smoothing.py --max-clusters 100
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

from smoothing_basis import SmoothingBasisProjection, MultiHeadLDABaseline
from fft_smoothing import FFTSmoothingProjection, AdaptiveFFTSmoothing
from hierarchical_smoothing import HierarchicalSmoothing


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    params: Dict[str, Any]
    train_time_ms: float
    inference_time_us: float  # per query
    mse: float
    cosine_sim: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    num_clusters: int
    num_pairs: int


def load_training_data(data_dir: Path, max_clusters: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Load training data grouped by cluster_id.

    Returns:
        Dict mapping cluster_id -> list of Q-A pairs
    """
    clusters = defaultdict(list)

    for jsonl_file in sorted(data_dir.rglob("*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        pair = json.loads(line)
                        cluster_id = pair.get("cluster_id", "unknown")
                        clusters[cluster_id].append(pair)
                    except json.JSONDecodeError:
                        pass

    if max_clusters and len(clusters) > max_clusters:
        # Take first N clusters
        cluster_ids = list(clusters.keys())[:max_clusters]
        clusters = {k: clusters[k] for k in cluster_ids}

    return dict(clusters)


def get_text(field: Any) -> str:
    """Extract text from field that may be string or dict."""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict):
        return field.get("text", str(field))
    return str(field) if field else ""


def simple_embedding(text: str, dim: int = 64) -> np.ndarray:
    """
    Simple deterministic embedding for testing.
    Uses hash-based projection for reproducibility.
    """
    # Hash-based embedding (deterministic, fast)
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim)
    return emb / (np.linalg.norm(emb) + 1e-8)


def prepare_cluster_data(clusters: Dict[str, List[Dict]],
                         dim: int = 64) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
                                                  List[Tuple[np.ndarray, np.ndarray, str]],
                                                  List[Tuple[str, str, str]]]:
    """
    Convert cluster data to embeddings.

    Returns:
        tuple_format: List of (Q, A) for smoothing basis
        triple_format: List of (Q, A, cluster_id) for hierarchical
        text_data: List of (question_text, answer_text, cluster_id)
    """
    tuple_format = []
    triple_format = []
    text_data = []

    for cluster_id, pairs in clusters.items():
        questions = []
        answers = []
        q_texts = []
        a_texts = []

        for pair in pairs:
            q_text = get_text(pair.get("question", ""))
            a_text = get_text(pair.get("answer", ""))

            if q_text and a_text:
                q_emb = simple_embedding(q_text, dim)
                a_emb = simple_embedding(a_text, dim)
                questions.append(q_emb)
                answers.append(a_emb)
                q_texts.append(q_text)
                a_texts.append(a_text)
                text_data.append((q_text, a_text, cluster_id))

        if questions:
            Q = np.array(questions)
            # Use mean of answers as cluster answer
            A = np.mean(answers, axis=0).reshape(1, -1)

            tuple_format.append((Q, A))
            triple_format.append((Q, A.flatten(), cluster_id))

    return tuple_format, triple_format, text_data


def compute_retrieval_metrics(projector, test_queries: List[np.ndarray],
                              test_answers: List[np.ndarray],
                              cluster_ids: List[str],
                              all_answers: np.ndarray,
                              all_cluster_ids: List[str],
                              k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    Compute retrieval precision@k.

    For each query, project it, find nearest answers, check if correct cluster.
    """
    precisions = {k: [] for k in k_values}

    for q_emb, true_answer, true_cluster in zip(test_queries, test_answers, cluster_ids):
        # Project query
        projected = projector.project(q_emb)

        # Compute similarities to all answers
        sims = []
        for i, ans in enumerate(all_answers):
            ans_norm = ans / (np.linalg.norm(ans) + 1e-8)
            proj_norm = projected / (np.linalg.norm(projected) + 1e-8)
            sim = np.dot(proj_norm, ans_norm)
            sims.append((sim, all_cluster_ids[i]))

        # Sort by similarity
        sims.sort(reverse=True)

        # Check precision at each k
        for k in k_values:
            top_k_clusters = [c for _, c in sims[:k]]
            hits = sum(1 for c in top_k_clusters if c == true_cluster)
            precisions[k].append(hits / k)

    return {f"precision_at_{k}": np.mean(precisions[k]) for k in k_values}


def benchmark_method(name: str, projector, clusters_tuple: List[Tuple],
                     clusters_triple: List[Tuple], text_data: List[Tuple],
                     train_fn, params: Dict[str, Any],
                     num_test_queries: int = 100,
                     dim: int = 64) -> BenchmarkResult:
    """Run benchmark for a single method."""

    # Training
    start = time.perf_counter()
    train_fn()
    train_time = (time.perf_counter() - start) * 1000  # ms

    # Prepare test data
    np.random.seed(42)
    test_indices = np.random.choice(len(text_data), min(num_test_queries, len(text_data)), replace=False)

    test_queries = []
    test_answers = []
    test_clusters = []

    for idx in test_indices:
        q_text, a_text, cluster_id = text_data[idx]
        test_queries.append(simple_embedding(q_text, dim))
        test_answers.append(simple_embedding(a_text, dim))
        test_clusters.append(cluster_id)

    # All answers for retrieval
    all_answers = []
    all_cluster_ids = []
    for q_text, a_text, cluster_id in text_data:
        all_answers.append(simple_embedding(a_text, dim))
        all_cluster_ids.append(cluster_id)
    all_answers = np.array(all_answers)

    # Inference timing
    start = time.perf_counter()
    projections = [projector.project(q) for q in test_queries]
    inference_time = (time.perf_counter() - start) * 1e6 / len(test_queries)  # us per query

    # MSE and cosine similarity
    mse_values = []
    cosine_values = []

    for proj, true_ans in zip(projections, test_answers):
        mse_values.append(np.mean((proj - true_ans) ** 2))

        proj_norm = proj / (np.linalg.norm(proj) + 1e-8)
        ans_norm = true_ans / (np.linalg.norm(true_ans) + 1e-8)
        cosine_values.append(np.dot(proj_norm, ans_norm))

    # Retrieval metrics
    retrieval = compute_retrieval_metrics(
        projector, test_queries, test_answers, test_clusters,
        all_answers, all_cluster_ids
    )

    return BenchmarkResult(
        method=name,
        params=params,
        train_time_ms=train_time,
        inference_time_us=inference_time,
        mse=np.mean(mse_values),
        cosine_sim=np.mean(cosine_values),
        precision_at_1=retrieval["precision_at_1"],
        precision_at_3=retrieval["precision_at_3"],
        precision_at_5=retrieval["precision_at_5"],
        num_clusters=len(clusters_tuple),
        num_pairs=len(text_data)
    )


def run_benchmarks(clusters_tuple: List[Tuple], clusters_triple: List[Tuple],
                   text_data: List[Tuple], k_values: List[int] = [2, 4, 8, 16],
                   dim: int = 64) -> List[BenchmarkResult]:
    """Run all benchmarks."""
    results = []

    print(f"\nBenchmarking on {len(clusters_tuple)} clusters, {len(text_data)} pairs, dim={dim}")
    print("=" * 80)

    # 1. Baseline: Multi-head LDA (no smoothing)
    print("\n1. MultiHeadLDA Baseline...")
    baseline = MultiHeadLDABaseline()
    result = benchmark_method(
        "MultiHeadLDA", baseline, clusters_tuple, clusters_triple, text_data,
        lambda: baseline.train(clusters_tuple),
        {"type": "baseline"}, dim=dim
    )
    results.append(result)
    print(f"   Train: {result.train_time_ms:.1f}ms, Infer: {result.inference_time_us:.1f}us")
    print(f"   MSE: {result.mse:.4f}, Cosine: {result.cosine_sim:.4f}, P@1: {result.precision_at_1:.4f}")

    # 2. Smoothing Basis with varying K
    for K in k_values:
        if K > len(clusters_tuple):
            continue
        print(f"\n2. SmoothingBasis (K={K})...")
        sb = SmoothingBasisProjection(num_basis=K, cosine_weight=0.5)
        result = benchmark_method(
            f"SmoothingBasis_K{K}", sb, clusters_tuple, clusters_triple, text_data,
            lambda sb=sb: sb.train(clusters_tuple, num_iterations=50, log_interval=100),
            {"K": K, "iterations": 50}, dim=dim
        )
        results.append(result)
        print(f"   Train: {result.train_time_ms:.1f}ms, Infer: {result.inference_time_us:.1f}us")
        print(f"   MSE: {result.mse:.4f}, Cosine: {result.cosine_sim:.4f}, P@1: {result.precision_at_1:.4f}")

    # 3. FFT Smoothing with varying cutoff
    for cutoff in [0.3, 0.5, 0.7]:
        print(f"\n3. FFTSmoothing (cutoff={cutoff})...")
        fft_s = FFTSmoothingProjection(cutoff=cutoff, blend_factor=0.6)
        result = benchmark_method(
            f"FFT_c{cutoff}", fft_s, clusters_tuple, clusters_triple, text_data,
            lambda fft_s=fft_s: fft_s.train(clusters_tuple),
            {"cutoff": cutoff, "blend": 0.6}, dim=dim
        )
        results.append(result)
        print(f"   Train: {result.train_time_ms:.1f}ms, Infer: {result.inference_time_us:.1f}us")
        print(f"   MSE: {result.mse:.4f}, Cosine: {result.cosine_sim:.4f}, P@1: {result.precision_at_1:.4f}")

    # 4. Adaptive FFT
    print("\n4. AdaptiveFFT...")
    adaptive = AdaptiveFFTSmoothing(min_cutoff=0.2, max_cutoff=0.7)
    result = benchmark_method(
        "AdaptiveFFT", adaptive, clusters_tuple, clusters_triple, text_data,
        lambda: adaptive.train(clusters_tuple),
        {"min_cutoff": 0.2, "max_cutoff": 0.7}, dim=dim
    )
    results.append(result)
    print(f"   Train: {result.train_time_ms:.1f}ms, Infer: {result.inference_time_us:.1f}us")
    print(f"   MSE: {result.mse:.4f}, Cosine: {result.cosine_sim:.4f}, P@1: {result.precision_at_1:.4f}")

    # 5. Hierarchical Smoothing
    for num_levels in [2, 3]:
        print(f"\n5. Hierarchical (levels={num_levels})...")
        hs = HierarchicalSmoothing(num_levels=num_levels, num_basis=4)
        result = benchmark_method(
            f"Hierarchical_L{num_levels}", hs, clusters_tuple, clusters_triple, text_data,
            lambda hs=hs: hs.train(clusters_triple, num_iterations=30),
            {"levels": num_levels, "basis": 4}, dim=dim
        )
        results.append(result)
        print(f"   Train: {result.train_time_ms:.1f}ms, Infer: {result.inference_time_us:.1f}us")
        print(f"   MSE: {result.mse:.4f}, Cosine: {result.cosine_sim:.4f}, P@1: {result.precision_at_1:.4f}")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Header
    print(f"{'Method':<25} {'Train(ms)':<12} {'Infer(us)':<12} {'MSE':<10} {'Cosine':<10} {'P@1':<8} {'P@3':<8} {'P@5':<8}")
    print("-" * 100)

    # Sort by P@1 descending
    for r in sorted(results, key=lambda x: x.precision_at_1, reverse=True):
        print(f"{r.method:<25} {r.train_time_ms:<12.1f} {r.inference_time_us:<12.1f} "
              f"{r.mse:<10.4f} {r.cosine_sim:<10.4f} {r.precision_at_1:<8.4f} "
              f"{r.precision_at_3:<8.4f} {r.precision_at_5:<8.4f}")

    print("-" * 100)

    # Best by metric
    print("\nBest by metric:")
    print(f"  Lowest MSE:      {min(results, key=lambda x: x.mse).method}")
    print(f"  Highest Cosine:  {max(results, key=lambda x: x.cosine_sim).method}")
    print(f"  Highest P@1:     {max(results, key=lambda x: x.precision_at_1).method}")
    print(f"  Fastest Train:   {min(results, key=lambda x: x.train_time_ms).method}")
    print(f"  Fastest Infer:   {min(results, key=lambda x: x.inference_time_us).method}")


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save results to JSON."""
    data = []
    for r in results:
        data.append({
            "method": r.method,
            "params": r.params,
            "train_time_ms": r.train_time_ms,
            "inference_time_us": r.inference_time_us,
            "mse": r.mse,
            "cosine_sim": r.cosine_sim,
            "precision_at_1": r.precision_at_1,
            "precision_at_3": r.precision_at_3,
            "precision_at_5": r.precision_at_5,
            "num_clusters": r.num_clusters,
            "num_pairs": r.num_pairs
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark smoothing approaches")
    parser.add_argument("--data-dir", type=Path, default=Path("training-data/expanded"),
                        help="Training data directory")
    parser.add_argument("--max-clusters", type=int, default=None,
                        help="Limit number of clusters")
    parser.add_argument("--dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--output", type=Path, default=Path("reports/smoothing_benchmark.json"),
                        help="Output file for results")
    parser.add_argument("--k-values", type=str, default="2,4,8,16",
                        help="K values for smoothing basis (comma-separated)")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_dir}...")
    clusters = load_training_data(args.data_dir, args.max_clusters)
    print(f"Loaded {len(clusters)} clusters")

    # Prepare embeddings
    print("Generating embeddings...")
    clusters_tuple, clusters_triple, text_data = prepare_cluster_data(clusters, args.dim)
    print(f"Prepared {len(text_data)} Q-A pairs")

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Run benchmarks
    results = run_benchmarks(clusters_tuple, clusters_triple, text_data,
                             k_values=k_values, dim=args.dim)

    # Print summary
    print_summary(results)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
