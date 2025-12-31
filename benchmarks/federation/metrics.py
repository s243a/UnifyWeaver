# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Metrics Collection for Federation Benchmarks.

Tracks per-query and aggregate performance metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import numpy as np

from .workload_generator import QueryType


@dataclass
class QueryMetric:
    """Metrics for a single query execution."""

    query_id: str
    query_type: QueryType
    latency_ms: float
    nodes_queried: int
    results_returned: int
    precision_at_k: float
    recall_at_k: float
    config_name: str = ""
    k_used: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "query_type": self.query_type.value,
            "latency_ms": self.latency_ms,
            "nodes_queried": self.nodes_queried,
            "results_returned": self.results_returned,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "config_name": self.config_name,
            "k_used": self.k_used,
            "error": self.error,
        }


@dataclass
class BenchmarkResults:
    """Aggregated results from a benchmark run."""

    config_name: str
    network_size: int
    queries: List[QueryMetric] = field(default_factory=list)
    run_count: int = 1

    # Computed aggregates (populated by compute_aggregates)
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_nodes_queried: float = 0.0
    avg_results_returned: float = 0.0
    total_time_s: float = 0.0
    error_rate: float = 0.0

    # Per-query-type breakdowns
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from query metrics."""
        if not self.queries:
            return

        successful = [q for q in self.queries if q.error is None]
        if not successful:
            self.error_rate = 1.0
            return

        latencies = [q.latency_ms for q in successful]
        self.p50_latency_ms = float(np.percentile(latencies, 50))
        self.p90_latency_ms = float(np.percentile(latencies, 90))
        self.p99_latency_ms = float(np.percentile(latencies, 99))
        self.avg_latency_ms = float(np.mean(latencies))

        self.avg_precision = float(np.mean([q.precision_at_k for q in successful]))
        self.avg_recall = float(np.mean([q.recall_at_k for q in successful]))
        self.avg_nodes_queried = float(np.mean([q.nodes_queried for q in successful]))
        self.avg_results_returned = float(np.mean([q.results_returned for q in successful]))
        self.total_time_s = sum(latencies) / 1000.0
        self.error_rate = 1.0 - (len(successful) / len(self.queries))

        # Per-type breakdown
        for query_type in QueryType:
            type_queries = [q for q in successful if q.query_type == query_type]
            if type_queries:
                type_latencies = [q.latency_ms for q in type_queries]
                self.metrics_by_type[query_type.value] = {
                    "count": len(type_queries),
                    "avg_latency_ms": float(np.mean(type_latencies)),
                    "avg_precision": float(np.mean([q.precision_at_k for q in type_queries])),
                    "avg_recall": float(np.mean([q.recall_at_k for q in type_queries])),
                    "avg_nodes_queried": float(np.mean([q.nodes_queried for q in type_queries])),
                }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_name": self.config_name,
            "network_size": self.network_size,
            "run_count": self.run_count,
            "query_count": len(self.queries),
            "p50_latency_ms": self.p50_latency_ms,
            "p90_latency_ms": self.p90_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_precision": self.avg_precision,
            "avg_recall": self.avg_recall,
            "avg_nodes_queried": self.avg_nodes_queried,
            "avg_results_returned": self.avg_results_returned,
            "total_time_s": self.total_time_s,
            "error_rate": self.error_rate,
            "metrics_by_type": self.metrics_by_type,
            "queries": [q.to_dict() for q in self.queries],
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dict without individual queries."""
        d = self.to_dict()
        del d["queries"]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResults":
        """Create from dictionary."""
        queries = [
            QueryMetric(
                query_id=q["query_id"],
                query_type=QueryType(q["query_type"]),
                latency_ms=q["latency_ms"],
                nodes_queried=q["nodes_queried"],
                results_returned=q["results_returned"],
                precision_at_k=q["precision_at_k"],
                recall_at_k=q["recall_at_k"],
                config_name=q.get("config_name", ""),
                k_used=q.get("k_used", 0),
                error=q.get("error"),
            )
            for q in data.get("queries", [])
        ]
        result = cls(
            config_name=data["config_name"],
            network_size=data["network_size"],
            queries=queries,
            run_count=data.get("run_count", 1),
        )
        result.compute_aggregates()
        return result


def save_results(
    results: Dict[str, BenchmarkResults],
    output_path: str,
) -> None:
    """Save benchmark results to JSON."""
    data = {name: r.to_dict() for name, r in results.items()}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(input_path: str) -> Dict[str, BenchmarkResults]:
    """Load benchmark results from JSON."""
    with open(input_path, "r") as f:
        data = json.load(f)
    return {name: BenchmarkResults.from_dict(d) for name, d in data.items()}


def compare_results(
    results: Dict[str, BenchmarkResults],
) -> Dict[str, Dict[str, float]]:
    """
    Compare results across configurations.

    Returns a dict with comparison metrics.
    """
    if not results:
        return {}

    comparison = {}
    baseline_name = list(results.keys())[0]
    baseline = results[baseline_name]

    for name, result in results.items():
        comparison[name] = {
            "latency_vs_baseline": (
                result.avg_latency_ms / baseline.avg_latency_ms
                if baseline.avg_latency_ms > 0 else 1.0
            ),
            "precision_vs_baseline": (
                result.avg_precision / baseline.avg_precision
                if baseline.avg_precision > 0 else 1.0
            ),
            "nodes_vs_baseline": (
                result.avg_nodes_queried / baseline.avg_nodes_queried
                if baseline.avg_nodes_queried > 0 else 1.0
            ),
        }

    return comparison
