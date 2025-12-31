# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Benchmark Runner for Federation Performance Testing.

Orchestrates benchmark runs with mock infrastructure.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Set
from unittest.mock import Mock, MagicMock
import numpy as np

from .synthetic_network import SyntheticNode
from .workload_generator import BenchmarkQuery, QueryType
from .metrics import QueryMetric, BenchmarkResults


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    engine_type: str  # "basic", "adaptive", "hierarchical", "streaming", "planned"
    strategy: str = "sum"  # Aggregation strategy
    k: int = 3  # Federation k (or base_k for adaptive)
    min_k: int = 1
    max_k: int = 10
    timeout_ms: int = 5000
    extra_options: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_options is None:
            self.extra_options = {}


# Default benchmark configurations
DEFAULT_CONFIGS = [
    BenchmarkConfig(name="baseline_sum_k3", engine_type="basic", strategy="sum", k=3),
    BenchmarkConfig(name="baseline_max_k3", engine_type="basic", strategy="max", k=3),
    BenchmarkConfig(name="adaptive_default", engine_type="adaptive", strategy="sum", k=3, min_k=1, max_k=10),
    BenchmarkConfig(name="hierarchical_2level", engine_type="hierarchical", strategy="sum", k=3),
    BenchmarkConfig(name="streaming_eager", engine_type="streaming", strategy="sum", k=5),
    BenchmarkConfig(name="planned_auto", engine_type="planned", strategy="sum", k=3),
]


class MockRouter:
    """Mock router that simulates node queries with configurable latency."""

    def __init__(self, nodes: List[SyntheticNode]):
        self.nodes = {n.node_id: n for n in nodes}
        self.node_list = nodes
        self.query_count = 0

    def get_nearest_nodes(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[SyntheticNode]:
        """Return k nearest nodes by cosine similarity."""
        similarities = []
        for node in self.node_list:
            sim = np.dot(query_embedding, node.centroid) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.centroid)
            )
            similarities.append((node, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in similarities[:k]]

    def query_node(
        self,
        node: SyntheticNode,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Simulate querying a node.

        Returns mock results with simulated latency.
        """
        self.query_count += 1

        # Simulate latency
        time.sleep(node.latency_ms / 1000.0)

        # Generate mock results
        similarity = np.dot(query_embedding, node.centroid) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(node.centroid)
        )

        results = []
        for i in range(min(node.result_count, top_k)):
            score = max(0.0, similarity - i * 0.05 + np.random.normal(0, 0.02))
            results.append({
                "answer_id": f"{node.node_id}_result_{i}",
                "answer_text": f"Result {i} from {node.node_id}",
                "raw_score": float(score),
                "exp_score": float(np.exp(score)),
            })

        return {
            "source_node": node.node_id,
            "results": results,
            "partition_sum": sum(r["exp_score"] for r in results),
            "response_time_ms": node.latency_ms,
        }


class FederationBenchmark:
    """Main benchmark runner."""

    def __init__(self, network: List[SyntheticNode]):
        self.network = network
        self.router = MockRouter(network)

    def run_single_query(
        self,
        query: BenchmarkQuery,
        config: BenchmarkConfig,
    ) -> QueryMetric:
        """Run a single benchmark query."""
        start_time = time.time()

        try:
            # Get nodes to query
            k = config.k
            if config.engine_type == "adaptive":
                # Simple adaptive: high similarity = lower k
                similarities = [
                    np.dot(query.embedding, n.centroid) / (
                        np.linalg.norm(query.embedding) * np.linalg.norm(n.centroid)
                    )
                    for n in self.network
                ]
                max_sim = max(similarities)
                # Higher similarity = lower k needed
                if max_sim > 0.8:
                    k = config.min_k
                elif max_sim > 0.6:
                    k = (config.min_k + config.max_k) // 2
                else:
                    k = config.max_k

            nodes = self.router.get_nearest_nodes(query.embedding, k)

            # Query nodes
            responses = []
            for node in nodes:
                resp = self.router.query_node(node, query.embedding)
                responses.append(resp)

            # Aggregate results (simplified)
            all_results = []
            for resp in responses:
                all_results.extend(resp["results"])

            # Sort by score and dedup
            all_results.sort(key=lambda x: x["exp_score"], reverse=True)
            seen_ids = set()
            unique_results = []
            for r in all_results:
                if r["answer_id"] not in seen_ids:
                    seen_ids.add(r["answer_id"])
                    unique_results.append(r)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Compute precision/recall vs ground truth
            returned_nodes = {r["answer_id"].split("_result_")[0] for r in unique_results[:10]}
            ground_truth_set = set(query.ground_truth_nodes)

            if ground_truth_set:
                precision = len(returned_nodes & ground_truth_set) / max(1, len(returned_nodes))
                recall = len(returned_nodes & ground_truth_set) / len(ground_truth_set)
            else:
                precision = 1.0 if returned_nodes else 0.0
                recall = 1.0

            return QueryMetric(
                query_id=query.query_id,
                query_type=query.expected_type,
                latency_ms=latency_ms,
                nodes_queried=len(nodes),
                results_returned=len(unique_results),
                precision_at_k=precision,
                recall_at_k=recall,
                config_name=config.name,
                k_used=k,
            )

        except Exception as e:
            end_time = time.time()
            return QueryMetric(
                query_id=query.query_id,
                query_type=query.expected_type,
                latency_ms=(end_time - start_time) * 1000,
                nodes_queried=0,
                results_returned=0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                config_name=config.name,
                error=str(e),
            )

    def run_config(
        self,
        config: BenchmarkConfig,
        workload: List[BenchmarkQuery],
        runs: int = 1,
    ) -> BenchmarkResults:
        """
        Run a benchmark configuration on a workload.

        Args:
            config: Benchmark configuration
            workload: List of queries to run
            runs: Number of runs for statistical significance

        Returns:
            BenchmarkResults with aggregated metrics
        """
        all_metrics = []

        for run_idx in range(runs):
            for query in workload:
                metric = self.run_single_query(query, config)
                all_metrics.append(metric)

        results = BenchmarkResults(
            config_name=config.name,
            network_size=len(self.network),
            queries=all_metrics,
            run_count=runs,
        )
        results.compute_aggregates()

        return results

    def compare_configs(
        self,
        configs: List[BenchmarkConfig],
        workload: List[BenchmarkQuery],
        runs: int = 1,
    ) -> Dict[str, BenchmarkResults]:
        """
        Run multiple configurations and compare.

        Args:
            configs: List of configurations to benchmark
            workload: List of queries to run
            runs: Number of runs per config

        Returns:
            Dict mapping config name to results
        """
        results = {}

        for config in configs:
            print(f"Running benchmark: {config.name}...")
            results[config.name] = self.run_config(config, workload, runs)
            print(f"  Avg latency: {results[config.name].avg_latency_ms:.1f}ms")
            print(f"  Avg precision: {results[config.name].avg_precision:.3f}")

        return results


def run_scalability_benchmark(
    sizes: List[int] = [10, 25, 50],
    queries_per_size: int = 30,
    seed: int = 42,
) -> Dict[int, Dict[str, BenchmarkResults]]:
    """
    Run benchmarks across different network sizes.

    Returns nested dict: size -> config_name -> results
    """
    from .synthetic_network import create_synthetic_network
    from .workload_generator import generate_workload

    results_by_size = {}

    for size in sizes:
        print(f"\n=== Network size: {size} nodes ===")

        network = create_synthetic_network(
            num_nodes=size,
            topic_distribution="clustered",
            latency_profile="mixed",
            seed=seed,
        )

        workload = generate_workload(
            network,
            num_queries=queries_per_size,
            seed=seed + size,
        )

        benchmark = FederationBenchmark(network)
        results_by_size[size] = benchmark.compare_configs(
            DEFAULT_CONFIGS,
            workload,
            runs=1,
        )

    return results_by_size
