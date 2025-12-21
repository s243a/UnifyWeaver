# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Comprehensive routing performance benchmarks.

Tests:
1. Scaling: O(log n) with shortcuts vs O(n) without
2. Backtrack vs greedy: Cost/benefit at different network sizes
3. Evolution speed: Queries until network matures
4. Cross-branch effectiveness: Semantic shortcuts reduce hops
5. Start-anywhere: Random-start success rate comparison
"""

import sys
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from unifyweaver.targets.python_runtime.small_world_evolution import (
    SmallWorldNetwork,
    cosine_distance,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    test_name: str
    parameters: Dict
    metrics: Dict
    duration_ms: float


def create_hierarchical_network(
    num_nodes: int,
    num_branches: int = 5,
    dim: int = 10,
    seed: int = 42,
) -> SmallWorldNetwork:
    """Create a hierarchical network with semantic clustering."""
    np.random.seed(seed)
    network = SmallWorldNetwork(max_shortcuts_per_node=10)

    # Root
    root_centroid = np.ones(dim) / np.sqrt(dim)
    network.add_node("root", root_centroid)

    # Create branches with distinct semantic regions
    nodes_per_branch = max(1, (num_nodes - 1) // num_branches)

    for b in range(num_branches):
        # Branch centroid - each branch focuses on different dimensions
        branch_centroid = np.zeros(dim)
        start_dim = (b * 2) % dim
        branch_centroid[start_dim] = 1.0
        branch_centroid[(start_dim + 1) % dim] = 0.5
        branch_centroid = branch_centroid / np.linalg.norm(branch_centroid)

        branch_id = f"branch_{b}"
        network.add_node(branch_id, branch_centroid, parent_id="root")
        network.nodes["root"].children_ids.append(branch_id)

        # Leaf nodes in branch
        for i in range(nodes_per_branch):
            leaf_centroid = branch_centroid + np.random.randn(dim) * 0.15
            leaf_centroid = leaf_centroid / np.linalg.norm(leaf_centroid)
            leaf_id = f"leaf_{b}_{i}"
            network.add_node(leaf_id, leaf_centroid, parent_id=branch_id)
            network.nodes[branch_id].children_ids.append(leaf_id)

    return network


def create_chain_network(length: int, dim: int = 10, seed: int = 42) -> SmallWorldNetwork:
    """Create a linear chain network for path folding tests."""
    np.random.seed(seed)
    network = SmallWorldNetwork(max_shortcuts_per_node=10)

    # Create progressive embeddings along a path
    start_dir = np.zeros(dim)
    start_dir[0] = 1.0

    end_dir = np.zeros(dim)
    end_dir[dim // 2] = 1.0

    nodes = [f"n{i}" for i in range(length)]

    for i, node_id in enumerate(nodes):
        t = i / max(1, length - 1)
        centroid = (1 - t) * start_dir + t * end_dir
        centroid = centroid / np.linalg.norm(centroid)

        parent = nodes[i - 1] if i > 0 else None
        network.add_node(node_id, centroid, parent_id=parent)
        if parent:
            network.nodes[parent].children_ids.append(node_id)

    return network


# =============================================================================
# BENCHMARK 1: Scaling (O(log n) vs O(n))
# =============================================================================

def benchmark_scaling(
    node_counts: List[int] = [10, 25, 50, 100, 200, 500],
    num_queries: int = 50,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Measure how comparisons scale with network size.

    Compares:
    - No shortcuts (pure hierarchy)
    - With evolution (shortcuts added)
    - With cross-branch links
    """
    start_time = time.time()
    results = []

    for num_nodes in node_counts:
        np.random.seed(seed)

        # Create network
        network = create_hierarchical_network(num_nodes, seed=seed)

        # Generate queries
        queries = []
        all_nodes = list(network.nodes.keys())
        for _ in range(num_queries):
            target_id = np.random.choice(all_nodes)
            target = network.nodes[target_id]
            query = target.centroid + np.random.randn(len(target.centroid)) * 0.1
            query = query / np.linalg.norm(query)
            queries.append((query, target_id))

        # Test 1: No shortcuts (hierarchical only)
        comps_no_shortcuts = []
        for query, _ in queries:
            _, comps = network.route_greedy(query, start_node_id="root")
            comps_no_shortcuts.append(comps)

        # Test 2: With evolution
        network.evolve(num_rounds=100, seed=seed)
        comps_evolved = []
        for query, _ in queries:
            _, comps = network.route_greedy(query, start_node_id="root")
            comps_evolved.append(comps)

        # Test 3: With cross-branch links
        network.discover_cross_branch_links(similarity_threshold=0.5)
        comps_cross_branch = []
        for query, _ in queries:
            _, comps = network.route_greedy(query, start_node_id="root")
            comps_cross_branch.append(comps)

        results.append({
            "num_nodes": num_nodes,
            "avg_comps_hierarchical": float(np.mean(comps_no_shortcuts)),
            "avg_comps_evolved": float(np.mean(comps_evolved)),
            "avg_comps_cross_branch": float(np.mean(comps_cross_branch)),
            "log_n": float(np.log2(num_nodes)),
        })

    duration = (time.time() - start_time) * 1000

    return BenchmarkResult(
        test_name="scaling",
        parameters={"node_counts": node_counts, "num_queries": num_queries},
        metrics={"results": results},
        duration_ms=duration,
    )


# =============================================================================
# BENCHMARK 2: Backtrack vs Greedy
# =============================================================================

def benchmark_backtrack_vs_greedy(
    node_counts: List[int] = [20, 50, 100, 200],
    num_queries: int = 50,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Compare backtracking vs greedy routing.

    Measures:
    - Success rate (found target)
    - Comparisons made
    - Path length
    """
    start_time = time.time()
    results = []

    for num_nodes in node_counts:
        np.random.seed(seed)

        network = create_hierarchical_network(num_nodes, seed=seed)

        # Generate queries from random start points to random targets
        all_nodes = list(network.nodes.keys())
        test_cases = []
        for _ in range(num_queries):
            start_id = np.random.choice(all_nodes)
            target_id = np.random.choice(all_nodes)
            target = network.nodes[target_id]
            query = target.centroid.copy()
            test_cases.append((query, start_id, target_id))

        # Test greedy (no backtrack)
        greedy_success = 0
        greedy_comps = []
        greedy_hops = []
        for query, start_id, target_id in test_cases:
            path, comps = network.route_greedy(
                query, start_node_id=start_id, use_backtrack=False
            )
            greedy_comps.append(comps)
            greedy_hops.append(len(path))
            if target_id in path:
                greedy_success += 1

        # Test backtracking
        backtrack_success = 0
        backtrack_comps = []
        backtrack_hops = []
        for query, start_id, target_id in test_cases:
            path, comps = network.route_greedy(
                query, start_node_id=start_id, use_backtrack=True
            )
            backtrack_comps.append(comps)
            backtrack_hops.append(len(path))
            if target_id in path:
                backtrack_success += 1

        results.append({
            "num_nodes": num_nodes,
            "greedy_success_rate": greedy_success / num_queries,
            "backtrack_success_rate": backtrack_success / num_queries,
            "greedy_avg_comps": float(np.mean(greedy_comps)),
            "backtrack_avg_comps": float(np.mean(backtrack_comps)),
            "greedy_avg_hops": float(np.mean(greedy_hops)),
            "backtrack_avg_hops": float(np.mean(backtrack_hops)),
        })

    duration = (time.time() - start_time) * 1000

    return BenchmarkResult(
        test_name="backtrack_vs_greedy",
        parameters={"node_counts": node_counts, "num_queries": num_queries},
        metrics={"results": results},
        duration_ms=duration,
    )


# =============================================================================
# BENCHMARK 3: Evolution Speed
# =============================================================================

def benchmark_evolution_speed(
    num_nodes: int = 100,
    max_queries: int = 500,
    target_maturity: float = 0.5,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Measure how many queries until network matures.

    Tracks maturity score as queries are processed and paths are folded.
    """
    start_time = time.time()
    np.random.seed(seed)

    network = create_hierarchical_network(num_nodes, seed=seed)

    # Track maturity over queries
    maturity_history = [(0, network.maturity_score)]
    all_nodes = list(network.nodes.keys())

    queries_to_mature = None

    for q in range(max_queries):
        # Generate query
        target_id = np.random.choice(all_nodes)
        target = network.nodes[target_id]
        query = target.centroid + np.random.randn(len(target.centroid)) * 0.05
        query = query / np.linalg.norm(query)

        # Route with backtracking
        path, _ = network.route_greedy(query, use_backtrack=True)

        # Path folding - create shortcuts from successful path
        if len(path) >= 3:
            network.record_successful_path(path, query)

        # Periodic evolution round
        if (q + 1) % 10 == 0:
            network.evolution_round()

        # Record maturity
        if (q + 1) % 10 == 0:
            maturity_history.append((q + 1, network.maturity_score))

        # Check if matured
        if queries_to_mature is None and network.maturity_score >= target_maturity:
            queries_to_mature = q + 1

    duration = (time.time() - start_time) * 1000

    return BenchmarkResult(
        test_name="evolution_speed",
        parameters={
            "num_nodes": num_nodes,
            "max_queries": max_queries,
            "target_maturity": target_maturity,
        },
        metrics={
            "queries_to_mature": queries_to_mature,
            "final_maturity": network.maturity_score,
            "maturity_history": maturity_history,
            "total_shortcuts": sum(len(n.shortcut_ids) for n in network.nodes.values()),
        },
        duration_ms=duration,
    )


# =============================================================================
# BENCHMARK 4: Cross-Branch Effectiveness
# =============================================================================

def benchmark_cross_branch_effectiveness(
    num_nodes: int = 100,
    num_branches: int = 5,
    num_queries: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Test how cross-branch links help for cross-topic queries.

    Creates queries that span multiple semantic topics.
    """
    start_time = time.time()
    np.random.seed(seed)
    dim = 10

    network = create_hierarchical_network(num_nodes, num_branches=num_branches, seed=seed)

    # Create cross-topic queries (blend of two branches)
    cross_topic_queries = []
    for _ in range(num_queries):
        # Pick two different branches
        b1, b2 = np.random.choice(num_branches, size=2, replace=False)

        # Create query that's between two branch centroids
        branch1 = network.nodes.get(f"branch_{b1}")
        branch2 = network.nodes.get(f"branch_{b2}")

        if branch1 and branch2:
            blend = 0.3 + np.random.random() * 0.4  # 30-70% blend
            query = blend * branch1.centroid + (1 - blend) * branch2.centroid
            query = query / np.linalg.norm(query)
            cross_topic_queries.append(query)

    # Test BEFORE cross-branch links
    hops_before = []
    comps_before = []
    for query in cross_topic_queries:
        path, comps = network.route_greedy(query, start_node_id="root")
        hops_before.append(len(path))
        comps_before.append(comps)

    # Add cross-branch links
    links_created = network.discover_cross_branch_links(
        similarity_threshold=0.4,
        max_links_per_node=3,
    )

    # Test AFTER cross-branch links
    hops_after = []
    comps_after = []
    for query in cross_topic_queries:
        path, comps = network.route_greedy(query, start_node_id="root")
        hops_after.append(len(path))
        comps_after.append(comps)

    duration = (time.time() - start_time) * 1000

    return BenchmarkResult(
        test_name="cross_branch_effectiveness",
        parameters={
            "num_nodes": num_nodes,
            "num_branches": num_branches,
            "num_queries": num_queries,
        },
        metrics={
            "links_created": links_created,
            "avg_hops_before": float(np.mean(hops_before)),
            "avg_hops_after": float(np.mean(hops_after)),
            "avg_comps_before": float(np.mean(comps_before)),
            "avg_comps_after": float(np.mean(comps_after)),
            "hop_reduction_pct": float((1 - np.mean(hops_after) / max(1, np.mean(hops_before))) * 100),
            "comp_reduction_pct": float((1 - np.mean(comps_after) / max(1, np.mean(comps_before))) * 100),
        },
        duration_ms=duration,
    )


# =============================================================================
# BENCHMARK 5: Start-Anywhere Success Rate
# =============================================================================

def benchmark_start_anywhere(
    num_nodes: int = 100,
    num_queries: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Compare success rate when starting from random nodes.

    Tests immature vs mature networks.
    """
    start_time = time.time()
    np.random.seed(seed)

    network = create_hierarchical_network(num_nodes, seed=seed)
    all_nodes = list(network.nodes.keys())

    # Generate test cases: random start -> random target
    test_cases = []
    for _ in range(num_queries):
        start_id = np.random.choice(all_nodes)
        target_id = np.random.choice(all_nodes)
        target = network.nodes[target_id]
        query = target.centroid.copy()
        test_cases.append((query, start_id, target_id))

    # Test IMMATURE network (no evolution)
    immature_success_greedy = 0
    immature_success_backtrack = 0

    for query, start_id, target_id in test_cases:
        # Greedy
        path, _ = network.route_greedy(query, start_node_id=start_id, use_backtrack=False)
        if target_id in path:
            immature_success_greedy += 1

        # Backtrack
        path, _ = network.route_greedy(query, start_node_id=start_id, use_backtrack=True)
        if target_id in path:
            immature_success_backtrack += 1

    # Evolve network
    network.evolve(num_rounds=200, seed=seed)
    network.discover_cross_branch_links(similarity_threshold=0.5)

    # Simulate query-driven evolution
    for _ in range(100):
        target_id = np.random.choice(all_nodes)
        target = network.nodes[target_id]
        query = target.centroid + np.random.randn(len(target.centroid)) * 0.1
        query = query / np.linalg.norm(query)
        path, _ = network.route_greedy(query, use_backtrack=True)
        network.record_successful_path(path, query)

    # Test MATURE network
    mature_success_greedy = 0
    mature_success_backtrack = 0

    for query, start_id, target_id in test_cases:
        # Greedy
        path, _ = network.route_greedy(query, start_node_id=start_id, use_backtrack=False)
        if target_id in path:
            mature_success_greedy += 1

        # Backtrack
        path, _ = network.route_greedy(query, start_node_id=start_id, use_backtrack=True)
        if target_id in path:
            mature_success_backtrack += 1

    duration = (time.time() - start_time) * 1000

    return BenchmarkResult(
        test_name="start_anywhere",
        parameters={"num_nodes": num_nodes, "num_queries": num_queries},
        metrics={
            "immature_greedy_success": immature_success_greedy / num_queries,
            "immature_backtrack_success": immature_success_backtrack / num_queries,
            "mature_greedy_success": mature_success_greedy / num_queries,
            "mature_backtrack_success": mature_success_backtrack / num_queries,
            "final_maturity": network.maturity_score,
            "total_shortcuts": sum(len(n.shortcut_ids) for n in network.nodes.values()),
        },
        duration_ms=duration,
    )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_benchmarks(output_dir: str = "reports") -> Dict:
    """Run all benchmarks and save results."""

    print("=" * 70)
    print("ROUTING PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    results = {}

    # Benchmark 1: Scaling
    print("Running: Scaling benchmark (O(log n) vs O(n))...")
    result = benchmark_scaling()
    results["scaling"] = asdict(result)
    print(f"  Duration: {result.duration_ms:.0f}ms")
    for r in result.metrics["results"]:
        print(f"  n={r['num_nodes']:3d}: hierarchical={r['avg_comps_hierarchical']:.1f}, "
              f"evolved={r['avg_comps_evolved']:.1f}, cross-branch={r['avg_comps_cross_branch']:.1f}")
    print()

    # Benchmark 2: Backtrack vs Greedy
    print("Running: Backtrack vs Greedy comparison...")
    result = benchmark_backtrack_vs_greedy()
    results["backtrack_vs_greedy"] = asdict(result)
    print(f"  Duration: {result.duration_ms:.0f}ms")
    for r in result.metrics["results"]:
        print(f"  n={r['num_nodes']:3d}: greedy_success={r['greedy_success_rate']:.0%}, "
              f"backtrack_success={r['backtrack_success_rate']:.0%}")
    print()

    # Benchmark 3: Evolution Speed
    print("Running: Evolution speed (queries until maturity)...")
    result = benchmark_evolution_speed()
    results["evolution_speed"] = asdict(result)
    print(f"  Duration: {result.duration_ms:.0f}ms")
    print(f"  Queries to mature: {result.metrics['queries_to_mature']}")
    print(f"  Final maturity: {result.metrics['final_maturity']:.3f}")
    print(f"  Total shortcuts: {result.metrics['total_shortcuts']}")
    print()

    # Benchmark 4: Cross-Branch Effectiveness
    print("Running: Cross-branch effectiveness...")
    result = benchmark_cross_branch_effectiveness()
    results["cross_branch"] = asdict(result)
    print(f"  Duration: {result.duration_ms:.0f}ms")
    print(f"  Links created: {result.metrics['links_created']}")
    print(f"  Hops: {result.metrics['avg_hops_before']:.1f} -> {result.metrics['avg_hops_after']:.1f} "
          f"({result.metrics['hop_reduction_pct']:.1f}% reduction)")
    print(f"  Comparisons: {result.metrics['avg_comps_before']:.1f} -> {result.metrics['avg_comps_after']:.1f}")
    print()

    # Benchmark 5: Start-Anywhere
    print("Running: Start-anywhere success rate...")
    result = benchmark_start_anywhere()
    results["start_anywhere"] = asdict(result)
    print(f"  Duration: {result.duration_ms:.0f}ms")
    print(f"  Immature: greedy={result.metrics['immature_greedy_success']:.0%}, "
          f"backtrack={result.metrics['immature_backtrack_success']:.0%}")
    print(f"  Mature:   greedy={result.metrics['mature_greedy_success']:.0%}, "
          f"backtrack={result.metrics['mature_backtrack_success']:.0%}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_time = sum(r["duration_ms"] for r in results.values())
    print(f"Total benchmark time: {total_time:.0f}ms ({total_time/1000:.1f}s)")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "routing_performance_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "benchmarks": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
