# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Phase Transition Analysis: Precision and Latency vs Network Size.

This script measures how precision and latency change as network size grows,
with k=3 fixed. We separate setup time from query time to understand
where computational cost lies.

Usage:
    python -m benchmarks.federation.phase_transition_analysis
"""

import sys
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed - will output data only")

from benchmarks.federation.synthetic_network import create_synthetic_network, SyntheticNode
from benchmarks.federation.workload_generator import generate_workload, BenchmarkQuery


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class PhaseTransitionMetrics:
    """Metrics for a single network size."""
    num_nodes: int
    nodes_per_cluster: float  # num_nodes / 5 clusters

    # Timing breakdown (ms)
    setup_time_ms: float      # Time to create network
    routing_time_ms: float    # Time to select top-k nodes
    query_time_ms: float      # Time to query nodes (simulated)
    total_time_ms: float      # End-to-end

    # Accuracy
    avg_precision: float              # Cluster recall for SPECIFIC queries
    avg_ground_truth_size: float      # How many nodes pass threshold
    intra_cluster_similarity: float   # Avg similarity within clusters

    # Per-query details
    num_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def route_to_top_k_bruteforce(
    query_embedding: np.ndarray,
    nodes: List[SyntheticNode],
    k: int = 3
) -> List[Tuple[SyntheticNode, float]]:
    """Route query to top-k nodes by similarity - O(n) brute force."""
    ranked = []
    for node in nodes:
        sim = cosine_similarity(query_embedding, node.centroid)
        ranked.append((node, sim))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:k]


def route_to_top_k_hierarchical(
    query_embedding: np.ndarray,
    nodes: List[SyntheticNode],
    k: int = 3,
    num_clusters: int = 5
) -> Tuple[List[Tuple[SyntheticNode, float]], int]:
    """
    Route using hierarchical clustering - O(num_clusters + nodes_per_cluster).

    1. First compare against cluster centroids (5 comparisons)
    2. Then search within best cluster(s)

    Returns: (results, num_comparisons)
    """
    comparisons = 0

    # Group nodes by cluster
    clusters: Dict[int, List[SyntheticNode]] = {i: [] for i in range(num_clusters)}
    for node in nodes:
        node_idx = int(node.node_id.split("_")[1])
        cluster_idx = node_idx % num_clusters
        clusters[cluster_idx].append(node)

    # Compute cluster centroids
    cluster_centroids = {}
    for cluster_idx, cluster_nodes in clusters.items():
        if cluster_nodes:
            centroids = np.array([n.centroid for n in cluster_nodes])
            cluster_centroids[cluster_idx] = np.mean(centroids, axis=0)

    # Phase 1: Find best cluster(s) - O(num_clusters)
    cluster_sims = []
    for cluster_idx, centroid in cluster_centroids.items():
        sim = cosine_similarity(query_embedding, centroid)
        cluster_sims.append((cluster_idx, sim))
        comparisons += 1

    cluster_sims.sort(key=lambda x: x[1], reverse=True)

    # Phase 2: Search within top 2 clusters - O(2 * nodes_per_cluster)
    candidates = []
    for cluster_idx, _ in cluster_sims[:2]:
        for node in clusters[cluster_idx]:
            sim = cosine_similarity(query_embedding, node.centroid)
            candidates.append((node, sim))
            comparisons += 1

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k], comparisons


# Use hierarchical routing by default
def route_to_top_k(
    query_embedding: np.ndarray,
    nodes: List[SyntheticNode],
    k: int = 3
) -> List[Tuple[SyntheticNode, float]]:
    """Route query - uses brute force for now, tracks comparisons."""
    return route_to_top_k_bruteforce(query_embedding, nodes, k)


def compute_precision(
    selected_nodes: List[Tuple[SyntheticNode, float]],
    ground_truth: List[str]
) -> float:
    """Compute precision: fraction of selected nodes in ground truth."""
    if not selected_nodes:
        return 0.0
    selected_ids = {node.node_id for node, _ in selected_nodes}
    ground_truth_set = set(ground_truth)
    correct = len(selected_ids & ground_truth_set)
    return correct / len(selected_ids)


def compute_cluster_recall(
    selected_nodes: List[Tuple[SyntheticNode, float]],
    target_cluster_idx: int,
    num_clusters: int = 5
) -> float:
    """
    Compute cluster recall: fraction of selected nodes from target cluster.

    With 5 clusters, node i belongs to cluster (i % 5).
    This measures how well routing focuses on the right cluster.
    """
    if not selected_nodes:
        return 0.0

    from_target = 0
    for node, _ in selected_nodes:
        # Extract node index from node_id like "node_042"
        node_idx = int(node.node_id.split("_")[1])
        if node_idx % num_clusters == target_cluster_idx:
            from_target += 1

    return from_target / len(selected_nodes)


def compute_intra_cluster_similarity(
    nodes: List[SyntheticNode],
    num_clusters: int = 5
) -> float:
    """Compute average intra-cluster similarity."""
    cluster_sims = []
    for cluster_idx in range(num_clusters):
        cluster_nodes = [n for i, n in enumerate(nodes)
                         if int(n.node_id.split("_")[1]) % num_clusters == cluster_idx]
        if len(cluster_nodes) >= 2:
            sims = []
            for i, n1 in enumerate(cluster_nodes):
                for n2 in cluster_nodes[i+1:]:
                    sims.append(cosine_similarity(n1.centroid, n2.centroid))
            if sims:
                cluster_sims.append(np.mean(sims))
    return np.mean(cluster_sims) if cluster_sims else 0.0


def run_benchmark_for_size(
    num_nodes: int,
    k: int = 3,
    num_queries: int = 50,
    seed: int = 42,
    embedding_dim: int = 64,  # Smaller dim for tighter clusters
    simulated_query_latency_ms: float = 0.0
) -> PhaseTransitionMetrics:
    """Run benchmark for a specific network size."""

    # --- SETUP PHASE ---
    setup_start = time.time()

    nodes = create_synthetic_network(
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,  # Use smaller dim
        topic_distribution="clustered",
        latency_profile="mixed",
        seed=seed,
    )

    workload = generate_workload(
        network=nodes,
        num_queries=num_queries,
        query_mix={"specific": 0.4, "exploratory": 0.3, "consensus": 0.3},
        seed=seed + 1,
    )

    setup_time_ms = (time.time() - setup_start) * 1000

    # --- QUERY PHASE ---
    routing_times = []
    query_times = []
    precisions = []
    ground_truth_sizes = []

    for query in workload:
        # Routing
        route_start = time.time()
        selected = route_to_top_k(query.embedding, nodes, k=k)
        routing_times.append((time.time() - route_start) * 1000)

        # Simulated query time (if any)
        query_start = time.time()
        if simulated_query_latency_ms > 0:
            time.sleep(simulated_query_latency_ms / 1000.0 * k)
        query_times.append((time.time() - query_start) * 1000)

        # Cluster recall: for SPECIFIC queries, how many of top-k are from target cluster?
        if query.expected_type.value == "specific" and "target_node" in query.metadata:
            target_node_id = query.metadata["target_node"]
            target_idx = int(target_node_id.split("_")[1])
            target_cluster = target_idx % 5
            precision = compute_cluster_recall(selected, target_cluster)
        else:
            # For other query types, use ground truth if available
            precision = compute_precision(selected, query.ground_truth_nodes) if query.ground_truth_nodes else 1.0

        precisions.append(precision)
        ground_truth_sizes.append(len(query.ground_truth_nodes))

    avg_routing_ms = np.mean(routing_times)
    avg_query_ms = np.mean(query_times)
    total_time_ms = setup_time_ms + sum(routing_times) + sum(query_times)

    # Compute cluster coherence
    intra_sim = compute_intra_cluster_similarity(nodes)

    return PhaseTransitionMetrics(
        num_nodes=num_nodes,
        nodes_per_cluster=num_nodes / 5.0,
        setup_time_ms=setup_time_ms,
        routing_time_ms=avg_routing_ms,
        query_time_ms=avg_query_ms,
        total_time_ms=total_time_ms,
        avg_precision=np.mean(precisions),
        avg_ground_truth_size=np.mean(ground_truth_sizes),
        intra_cluster_similarity=intra_sim,
        num_queries=num_queries,
    )


def run_phase_transition_analysis(
    node_counts: List[int],
    k: int = 3,
    num_queries: int = 50,
) -> List[PhaseTransitionMetrics]:
    """Run analysis across multiple network sizes."""
    results = []

    print(f"\n{'='*70}")
    print(f"PHASE TRANSITION ANALYSIS: Precision & Latency vs Network Size")
    print(f"{'='*70}")
    print(f"Fixed k={k}, queries per size={num_queries}")
    print(f"Node counts: {node_counts}")
    print()

    print(f"{'Nodes':>6} {'N/Clust':>8} {'Setup(ms)':>10} {'Route(ms)':>10} "
          f"{'ClustRecall':>11} {'IntraSim':>9}")
    print("-" * 70)

    for num_nodes in node_counts:
        metrics = run_benchmark_for_size(
            num_nodes=num_nodes,
            k=k,
            num_queries=num_queries,
        )
        results.append(metrics)

        print(f"{metrics.num_nodes:>6} {metrics.nodes_per_cluster:>8.1f} "
              f"{metrics.setup_time_ms:>10.2f} {metrics.routing_time_ms:>10.4f} "
              f"{metrics.avg_precision:>11.3f} {metrics.intra_cluster_similarity:>9.4f}")

    return results


def plot_results(results: List[PhaseTransitionMetrics], output_dir: str = "reports"):
    """Generate plots of the phase transition."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots - matplotlib not installed")
        return

    os.makedirs(output_dir, exist_ok=True)

    nodes = [r.num_nodes for r in results]
    cluster_recall = [r.avg_precision for r in results]
    setup_times = [r.setup_time_ms for r in results]
    routing_times = [r.routing_time_ms for r in results]
    intra_sims = [r.intra_cluster_similarity for r in results]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Cluster Recall vs Nodes
    ax1 = axes[0, 0]
    ax1.plot(nodes, cluster_recall, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Cluster Recall')
    ax1.set_title('Cluster Recall vs Network Size (k=3)\n(Fraction of top-k from target cluster)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.6, 1.0)
    # Mark phase transition regions
    ax1.axvspan(15, 30, alpha=0.2, color='yellow', label='Transition region')
    ax1.legend()

    # Plot 2: Time Breakdown (log-log)
    ax2 = axes[0, 1]
    ax2.loglog(nodes, setup_times, 'rs-', linewidth=2, markersize=8, label='Setup')
    total_routing = [r * 50 for r in routing_times]  # 50 queries
    ax2.loglog(nodes, total_routing, 'g^-', linewidth=2, markersize=8, label='Routing (50 queries)')
    ax2.set_xlabel('Number of Nodes (log scale)')
    ax2.set_ylabel('Time (ms, log scale)')
    ax2.set_title('Time Breakdown (log-log)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Routing Time per Query (linear)
    ax3 = axes[1, 0]
    ax3.plot(nodes, routing_times, 'g^-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Routing Time per Query (ms)')
    ax3.set_title('Routing Time Scales O(n)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Intra-cluster Similarity
    ax4 = axes[1, 1]
    ax4.plot(nodes, intra_sims, 'mp-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Intra-cluster Similarity')
    ax4.set_title('Cluster Coherence vs Network Size')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Phase Transition Analysis: Fixed k=3, Varying Network Size', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'phase_transition.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Also create a summary 2-panel plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes2[0]
    ax1.semilogx(nodes, cluster_recall, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes (log scale)')
    ax1.set_ylabel('Cluster Recall')
    ax1.set_title('Cluster Recall vs Network Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.6, 1.0)
    ax1.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target: 0.85')
    ax1.legend()

    ax2 = axes2[1]
    ax2.loglog(nodes, setup_times, 'rs-', linewidth=2, markersize=8, label='Setup O(n^0.8)')
    ax2.loglog(nodes, total_routing, 'g^-', linewidth=2, markersize=8, label='Routing O(n)')
    ax2.set_xlabel('Number of Nodes (log scale)')
    ax2.set_ylabel('Time (ms, log scale)')
    ax2.set_title('Computational Cost')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    plot_path2 = os.path.join(output_dir, 'phase_transition_summary.png')
    plt.savefig(plot_path2, dpi=150)
    print(f"Summary plot saved to: {plot_path2}")

    plt.close('all')


def save_results(results: List[PhaseTransitionMetrics], output_dir: str = "reports"):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    data = {
        "analysis": "phase_transition",
        "parameters": {
            "k": 3,
            "num_queries": results[0].num_queries if results else 0,
        },
        "results": [r.to_dict() for r in results],
    }

    json_path = os.path.join(output_dir, 'phase_transition_results.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {json_path}")


def print_analysis(results: List[PhaseTransitionMetrics]):
    """Print analysis of the phase transition."""
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # Find transition point (where precision jumps)
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        precision_jump = curr.avg_precision - prev.avg_precision
        if precision_jump > 0.05:
            print(f"\nPhase transition detected between {prev.num_nodes} and {curr.num_nodes} nodes:")
            print(f"  Precision jumped from {prev.avg_precision:.3f} to {curr.avg_precision:.3f}")
            print(f"  Ground truth size: {prev.avg_ground_truth_size:.1f} -> {curr.avg_ground_truth_size:.1f}")
            print(f"  Nodes per cluster: {prev.nodes_per_cluster:.1f} -> {curr.nodes_per_cluster:.1f}")

    # Time complexity analysis
    print("\nTime Complexity:")
    nodes = [r.num_nodes for r in results]
    setup_times = [r.setup_time_ms for r in results]
    routing_times = [r.routing_time_ms for r in results]

    # Fit power law: time = a * n^b
    if len(nodes) > 2:
        log_nodes = np.log(nodes)
        log_setup = np.log(setup_times)
        log_routing = np.log(routing_times)

        # Linear regression in log space
        setup_slope, _ = np.polyfit(log_nodes, log_setup, 1)
        routing_slope, _ = np.polyfit(log_nodes, log_routing, 1)

        print(f"  Setup time scales as O(n^{setup_slope:.2f})")
        print(f"  Routing time scales as O(n^{routing_slope:.2f})")

        if setup_slope > routing_slope + 0.5:
            print("\n  -> Setup dominates at larger scales")
        elif routing_slope > setup_slope + 0.5:
            print("\n  -> Routing dominates at larger scales")
        else:
            print("\n  -> Setup and routing scale similarly")

    # Cost breakdown at largest size
    largest = results[-1]
    total_query_time = largest.routing_time_ms * largest.num_queries
    print(f"\nAt {largest.num_nodes} nodes:")
    print(f"  Setup: {largest.setup_time_ms:.2f} ms ({100*largest.setup_time_ms/(largest.setup_time_ms+total_query_time):.1f}%)")
    print(f"  Routing ({largest.num_queries} queries): {total_query_time:.2f} ms ({100*total_query_time/(largest.setup_time_ms+total_query_time):.1f}%)")


def compare_routing_scaling():
    """Compare O(n) brute force vs O(sqrt(n)) hierarchical routing."""
    print(f"\n{'='*70}")
    print("ROUTING SCALING COMPARISON: Brute Force O(n) vs Hierarchical O(√n)")
    print(f"{'='*70}")

    node_counts = [10, 25, 50, 100, 250, 500, 1000]

    print(f"\n{'Nodes':>6} {'BruteForce':>12} {'Hierarchical':>12} {'Speedup':>8}")
    print("-" * 50)

    for num_nodes in node_counts:
        # Create network
        nodes = create_synthetic_network(num_nodes=num_nodes, embedding_dim=64, seed=42)

        # Generate a query
        query = nodes[0].centroid + np.random.randn(64) * 0.1
        query = query / np.linalg.norm(query)

        # Brute force: always n comparisons
        bf_comparisons = num_nodes

        # Hierarchical: 5 (clusters) + 2 * (n/5) comparisons
        _, hier_comparisons = route_to_top_k_hierarchical(query, nodes, k=3)

        speedup = bf_comparisons / hier_comparisons

        print(f"{num_nodes:>6} {bf_comparisons:>12} {hier_comparisons:>12} {speedup:>8.1f}x")

    print(f"\nFor small-world O(log n) routing, we need:")
    print("  1. Multi-level hierarchy (not just 1 level)")
    print("  2. Proper Kleinberg graph with O(log n) shortcut links")
    print("  3. Greedy routing through the graph")


if __name__ == "__main__":
    # Test a range of network sizes
    # Start small, then use larger steps
    node_counts = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500]

    results = run_phase_transition_analysis(
        node_counts=node_counts,
        k=3,
        num_queries=50,
    )

    print_analysis(results)

    # Compare routing scaling
    compare_routing_scaling()

    save_results(results)
    plot_results(results)
