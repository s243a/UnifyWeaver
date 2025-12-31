#!/usr/bin/env python3
"""
Benchmark for Phase 7-8 small-world implementation.

Measures:
- Network construction time
- Greedy routing performance (path length, comparisons)
- KNN search performance (accuracy, comparisons)
- Comparison of standard vs optimized lookup
"""

import sys
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Add the runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from small_world_proper import SmallWorldProper, cosine_similarity, cosine_distance


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    network_size: int
    k_local: int
    k_long: int
    construction_time_ms: float
    avg_path_length: float
    avg_comparisons: float
    avg_routing_time_ms: float
    knn_recall_at_5: float
    optimized_speedup: float


def brute_force_knn(nodes: dict, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
    """Brute force KNN for ground truth."""
    distances = []
    for node_id, node in nodes.items():
        dist = cosine_distance(query, node.vector)
        distances.append((node_id, dist))
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def run_benchmark(
    network_size: int,
    k_local: int,
    k_long: int,
    num_queries: int = 100,
    embedding_dim: int = 128,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    np.random.seed(seed)

    # Build network
    start = time.time()
    network = SmallWorldProper(k_local=k_local, k_long=k_long)

    for i in range(network_size):
        vec = np.random.randn(embedding_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        network.add_node(f"n{i}", vec)

    construction_time = (time.time() - start) * 1000

    # Generate queries
    queries = []
    for _ in range(num_queries):
        q = np.random.randn(embedding_dim).astype(np.float32)
        q = q / np.linalg.norm(q)
        queries.append(q)

    # Benchmark greedy routing
    path_lengths = []
    comparisons_list = []
    routing_times = []

    for query in queries:
        start = time.time()
        path, comparisons = network.route_greedy(query)
        routing_times.append((time.time() - start) * 1000)
        path_lengths.append(len(path))
        comparisons_list.append(comparisons)

    # Benchmark KNN search vs brute force (recall)
    recall_scores = []
    for query in queries[:20]:  # Limit for speed
        gt = brute_force_knn(network.nodes, query, 5)
        gt_ids = set(nid for nid, _ in gt)

        results, _ = network.search_knn(query, k=5)
        result_ids = set(nid for nid, _ in results)

        recall = len(gt_ids & result_ids) / len(gt_ids)
        recall_scores.append(recall)

    # Benchmark optimized vs standard
    standard_times = []
    optimized_times = []

    for query in queries[:20]:
        start = time.time()
        network.search_knn(query, k=5)
        standard_times.append(time.time() - start)

        start = time.time()
        network.search_knn_optimized(query, k=5, window_size=5)
        optimized_times.append(time.time() - start)

    avg_standard = sum(standard_times) / len(standard_times)
    avg_optimized = sum(optimized_times) / len(optimized_times)
    speedup = avg_standard / avg_optimized if avg_optimized > 0 else 1.0

    return BenchmarkResult(
        name=f"n={network_size}, k_local={k_local}, k_long={k_long}",
        network_size=network_size,
        k_local=k_local,
        k_long=k_long,
        construction_time_ms=construction_time,
        avg_path_length=sum(path_lengths) / len(path_lengths),
        avg_comparisons=sum(comparisons_list) / len(comparisons_list),
        avg_routing_time_ms=sum(routing_times) / len(routing_times),
        knn_recall_at_5=sum(recall_scores) / len(recall_scores),
        optimized_speedup=speedup,
    )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "=" * 100)
    print("PHASE 7-8 SMALL-WORLD BENCHMARK RESULTS")
    print("=" * 100)

    print(f"\n{'Config':<35} {'Build(ms)':<12} {'Path':<8} {'Comps':<10} "
          f"{'Route(ms)':<12} {'Recall@5':<10} {'Speedup':<10}")
    print("-" * 100)

    for r in results:
        print(f"{r.name:<35} {r.construction_time_ms:<12.2f} "
              f"{r.avg_path_length:<8.2f} {r.avg_comparisons:<10.2f} "
              f"{r.avg_routing_time_ms:<12.4f} {r.knn_recall_at_5:<10.2%} "
              f"{r.optimized_speedup:<10.2f}x")

    print("-" * 100)


def main():
    """Run benchmarks."""
    print("Running Phase 7-8 Small-World Benchmarks...")
    print("(This may take a minute)\n")

    configs = [
        # (network_size, k_local, k_long)
        (100, 5, 2),
        (100, 10, 5),
        (100, 15, 5),
        (500, 10, 5),
        (500, 20, 10),
        (1000, 10, 5),
        (1000, 20, 10),
    ]

    results = []
    for size, k_local, k_long in configs:
        print(f"  Running: n={size}, k_local={k_local}, k_long={k_long}...")
        result = run_benchmark(size, k_local, k_long, num_queries=50)
        results.append(result)

    print_results(results)

    # Summary
    print("\nKEY FINDINGS:")
    print("-" * 50)

    # Find best recall config
    best_recall = max(results, key=lambda r: r.knn_recall_at_5)
    print(f"Best Recall@5: {best_recall.knn_recall_at_5:.1%} ({best_recall.name})")

    # Find shortest paths
    shortest_path = min(results, key=lambda r: r.avg_path_length)
    print(f"Shortest Paths: {shortest_path.avg_path_length:.1f} ({shortest_path.name})")

    # Find fastest construction per node
    fastest = min(results, key=lambda r: r.construction_time_ms / r.network_size)
    print(f"Fastest Build: {fastest.construction_time_ms / fastest.network_size:.3f}ms/node ({fastest.name})")

    # Average speedup from optimization
    avg_speedup = sum(r.optimized_speedup for r in results) / len(results)
    print(f"Avg Optimized Speedup: {avg_speedup:.2f}x")

    print("\n" + "=" * 100)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
