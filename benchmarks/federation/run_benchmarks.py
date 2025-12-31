#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
CLI Entry Point for Federation Benchmarks.

Usage:
    python -m benchmarks.federation.run_benchmarks --nodes 10,25,50 --queries 50 --output reports/
"""

import argparse
import os
import sys
import time

from .synthetic_network import create_synthetic_network, create_network_suite
from .workload_generator import generate_workload
from .metrics import BenchmarkResults, save_results
from .benchmark_runner import FederationBenchmark, DEFAULT_CONFIGS, BenchmarkConfig
from .visualizations import generate_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run federation benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick benchmark with 10 nodes
    python -m benchmarks.federation.run_benchmarks --nodes 10 --queries 30

    # Scalability test
    python -m benchmarks.federation.run_benchmarks --nodes 10,25,50 --queries 50

    # Full benchmark with custom output
    python -m benchmarks.federation.run_benchmarks --nodes 10,25,50 --queries 100 --output reports/
        """,
    )

    parser.add_argument(
        "--nodes",
        type=str,
        default="10,25,50",
        help="Comma-separated network sizes (default: 10,25,50)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of queries per size (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/",
        help="Output directory for reports (default: reports/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per configuration (default: 1)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated config names to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer queries, single run",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse node sizes
    sizes = [int(s.strip()) for s in args.nodes.split(",")]

    # Quick mode overrides
    if args.quick:
        args.queries = min(args.queries, 20)
        args.runs = 1

    # Filter configs if specified
    configs = DEFAULT_CONFIGS
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = [c for c in DEFAULT_CONFIGS if c.name in config_names]
        if not configs:
            print(f"Error: No matching configs found. Available: {[c.name for c in DEFAULT_CONFIGS]}")
            sys.exit(1)

    print("=" * 60)
    print("Federation Benchmark Suite")
    print("=" * 60)
    print(f"Network sizes: {sizes}")
    print(f"Queries per size: {args.queries}")
    print(f"Runs per config: {args.runs}")
    print(f"Configurations: {[c.name for c in configs]}")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    start_time = time.time()

    # Run benchmarks for each size
    results_by_size = {}

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Network Size: {size} nodes")
        print("=" * 60)

        # Create network
        network = create_synthetic_network(
            num_nodes=size,
            topic_distribution="clustered",
            latency_profile="mixed",
            seed=args.seed,
        )

        # Generate workload
        workload = generate_workload(
            network,
            num_queries=args.queries,
            seed=args.seed + size,
        )

        print(f"Generated {len(workload)} queries")

        # Run benchmarks
        benchmark = FederationBenchmark(network)
        results = benchmark.compare_configs(configs, workload, runs=args.runs)

        results_by_size[size] = results

        # Print summary
        print(f"\nSummary for {size} nodes:")
        print("-" * 40)
        for name, result in results.items():
            print(f"  {name}:")
            print(f"    P50 latency: {result.p50_latency_ms:.1f}ms")
            print(f"    Avg precision: {result.avg_precision:.3f}")
            print(f"    Avg nodes queried: {result.avg_nodes_queried:.1f}")

    # Generate report with the largest size as primary
    primary_size = max(sizes)
    primary_results = results_by_size[primary_size]

    print(f"\n{'='*60}")
    print("Generating reports...")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    # Generate HTML report
    html_path = generate_report(
        primary_results,
        args.output,
        results_by_size if len(sizes) > 1 else None,
    )

    # Save raw results
    json_path = os.path.join(args.output, "results.json")
    save_results(primary_results, json_path)

    elapsed = time.time() - start_time

    print(f"\nBenchmark complete in {elapsed:.1f}s")
    print(f"Reports saved to: {args.output}")
    print(f"  - HTML report: {html_path}")
    print(f"  - JSON data: {json_path}")
    print(f"  - Charts: {args.output}/*.png")


if __name__ == "__main__":
    main()
