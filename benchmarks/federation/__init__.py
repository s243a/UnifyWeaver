# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Federation Benchmark Suite for KG Topology Phase 6b.

Measures performance of federated semantic search across different
configurations, network sizes, and query patterns.
"""

from .synthetic_network import SyntheticNode, create_synthetic_network
from .workload_generator import BenchmarkQuery, generate_workload
from .metrics import QueryMetric, BenchmarkResults
from .benchmark_runner import FederationBenchmark
from .visualizations import (
    plot_latency_comparison,
    plot_precision_latency_tradeoff,
    plot_scalability,
    generate_report,
)

__all__ = [
    "SyntheticNode",
    "create_synthetic_network",
    "BenchmarkQuery",
    "generate_workload",
    "QueryMetric",
    "BenchmarkResults",
    "FederationBenchmark",
    "plot_latency_comparison",
    "plot_precision_latency_tradeoff",
    "plot_scalability",
    "generate_report",
]
