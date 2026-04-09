#!/usr/bin/env python3
"""
Benchmark A* dimensionality parameter for semantic shortest path.

Compares Dijkstra (no heuristic) vs A* with different power factors D
in the priority function f(n) = g(n)^D + h(n)^D.

Measures:
  - Nodes expanded (lower = better pruning)
  - Wall-clock time
  - Path cost (should be identical for admissible heuristics)

Usage:
    python benchmark_astar_dimensionality.py <category_parent.tsv> [--edge-weights edge_weights.tsv]

    If --edge-weights is not provided, uses uniform weights of 0.3 for testing.
    For real benchmarks, precompute with precompute_edge_weights.py first.
"""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


def load_edges(path: Path) -> list[tuple[str, str]]:
    edges = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    return edges


def load_weighted_edges(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load edge weights as adjacency list: node -> [(neighbor, weight)]"""
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                adj[parts[0]].append((parts[1], float(parts[2])))
    return dict(adj)


def build_uniform_adj(edges: list[tuple[str, str]], weight: float = 0.3) -> dict[str, list[tuple[str, float]]]:
    """Build adjacency list with uniform weights (for testing without embeddings)."""
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for child, parent in edges:
        adj[child].append((parent, weight))
    return dict(adj)


def compute_direct_distances(
    adj: dict[str, list[tuple[str, float]]], targets: list[str]
) -> dict[tuple[str, str], float]:
    """Compute direct semantic distance from each node to each target.

    Uses the edge weight as a proxy for semantic distance where available.
    For nodes not directly connected, uses 1.0 (maximum distance).
    """
    # Collect all nodes
    nodes = set()
    for src, neighbors in adj.items():
        nodes.add(src)
        for dst, _ in neighbors:
            nodes.add(dst)

    # For each target, run reverse Dijkstra to get true shortest distances
    # (these serve as perfect heuristics — makes A* optimal AND fast)
    direct: dict[tuple[str, str], float] = {}
    rev_adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for src, neighbors in adj.items():
        for dst, w in neighbors:
            rev_adj[dst].append((src, w))

    for target in targets:
        dist = dijkstra_all(dict(rev_adj), target)
        for node, d in dist.items():
            direct[(node, target)] = d

    return direct


def dijkstra_all(adj: dict[str, list[tuple[str, float]]], start: str) -> dict[str, float]:
    """Standard Dijkstra from start to all reachable nodes."""
    dist: dict[str, float] = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        for next_node, weight in adj.get(node, []):
            next_cost = cost + weight
            if next_cost < dist.get(next_node, float("inf")):
                dist[next_node] = next_cost
                heapq.heappush(heap, (next_cost, next_node))
    return dist


def astar_search(
    adj: dict[str, list[tuple[str, float]]],
    start: str,
    target: str,
    direct_dist: dict[tuple[str, str], float],
    dim: float,
) -> tuple[float, int]:
    """A* search with f(n) = g^D + h^D priority.

    Returns (min_cost, nodes_expanded).
    """
    def heuristic(node: str) -> float:
        return direct_dist.get((node, target), 1.0)

    def f_cost(g: float, h: float) -> float:
        return g ** dim + h ** dim

    dist: dict[str, float] = {start: 0.0}
    heap = [(f_cost(0.0, heuristic(start)), 0.0, start)]
    expanded = 0

    while heap:
        _, g_cost, node = heapq.heappop(heap)
        if g_cost > dist.get(node, float("inf")):
            continue
        expanded += 1
        if node == target:
            return dist[target], expanded
        for next_node, weight in adj.get(node, []):
            next_g = g_cost + weight
            if next_g < dist.get(next_node, float("inf")):
                dist[next_node] = next_g
                h = heuristic(next_node)
                heapq.heappush(heap, (f_cost(next_g, h), next_g, next_node))

    return dist.get(target, float("inf")), expanded


def dijkstra_search(
    adj: dict[str, list[tuple[str, float]]],
    start: str,
    target: str,
) -> tuple[float, int]:
    """Dijkstra (no heuristic) as baseline. Returns (min_cost, nodes_expanded)."""
    dist: dict[str, float] = {start: 0.0}
    heap = [(0.0, start)]
    expanded = 0

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        expanded += 1
        if node == target:
            return dist[target], expanded
        for next_node, weight in adj.get(node, []):
            next_cost = cost + weight
            if next_cost < dist.get(next_node, float("inf")):
                dist[next_node] = next_cost
                heapq.heappush(heap, (next_cost, next_node))

    return dist.get(target, float("inf")), expanded


def find_root_categories(edges: list[tuple[str, str]]) -> list[str]:
    """Find nodes that are parents but never children (roots)."""
    children = set(e[0] for e in edges)
    parents = set(e[1] for e in edges)
    roots = parents - children
    return sorted(roots)[:5]  # limit for tractability


def find_leaf_categories(edges: list[tuple[str, str]]) -> list[str]:
    """Find nodes that are children but never parents (leaves)."""
    children = set(e[0] for e in edges)
    parents = set(e[1] for e in edges)
    leaves = children - parents
    return sorted(leaves)[:20]  # sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edges_tsv", type=Path)
    parser.add_argument("--edge-weights", type=Path, default=None)
    parser.add_argument("--dims", default="1,2,3,4,5,7,10", help="Comma-separated D values")
    parser.add_argument("--queries", type=int, default=20, help="Number of start→target queries")
    args = parser.parse_args()

    edges = load_edges(args.edges_tsv)
    print(f"Loaded {len(edges)} edges", file=sys.stderr)

    if args.edge_weights and args.edge_weights.exists():
        adj = load_weighted_edges(args.edge_weights)
        print(f"Loaded weighted edges from {args.edge_weights}", file=sys.stderr)
    else:
        adj = build_uniform_adj(edges)
        print("Using uniform weights (0.3) — pass --edge-weights for real benchmark", file=sys.stderr)

    dims = [float(d) for d in args.dims.split(",")]
    roots = find_root_categories(edges)
    leaves = find_leaf_categories(edges)

    if not roots:
        print("No root categories found", file=sys.stderr)
        return 1

    print(f"Roots: {roots}", file=sys.stderr)
    print(f"Leaves (sample): {leaves[:5]}...", file=sys.stderr)

    # Precompute direct distances (reverse Dijkstra from each root)
    print("Precomputing direct distances...", file=sys.stderr)
    t0 = time.perf_counter()
    direct_dist = compute_direct_distances(adj, roots)
    precompute_time = time.perf_counter() - t0
    print(f"  {len(direct_dist)} direct distances in {precompute_time:.3f}s", file=sys.stderr)

    # Build query pairs
    queries = []
    for leaf in leaves[:args.queries]:
        for root in roots[:2]:  # limit root targets
            queries.append((leaf, root))
    print(f"\nRunning {len(queries)} queries...\n", file=sys.stderr)

    # Header
    print("dim\ttotal_expanded\ttotal_time_ms\tavg_expanded\tpath_agreement")

    # Dijkstra baseline
    total_exp = 0
    total_time = 0.0
    baseline_costs: dict[tuple[str, str], float] = {}
    for start, target in queries:
        t0 = time.perf_counter()
        cost, exp = dijkstra_search(adj, start, target)
        total_time += time.perf_counter() - t0
        total_exp += exp
        baseline_costs[(start, target)] = cost

    print(f"dijkstra\t{total_exp}\t{total_time*1000:.1f}\t{total_exp/max(len(queries),1):.1f}\t-")

    # A* with each D value
    for dim in dims:
        total_exp = 0
        total_time = 0.0
        agree = 0
        for start, target in queries:
            t0 = time.perf_counter()
            cost, exp = astar_search(adj, start, target, direct_dist, dim)
            total_time += time.perf_counter() - t0
            total_exp += exp
            baseline = baseline_costs.get((start, target), float("inf"))
            if abs(cost - baseline) < 1e-9 or (cost == float("inf") and baseline == float("inf")):
                agree += 1

        pct = 100 * agree / max(len(queries), 1)
        print(f"D={dim:.0f}\t{total_exp}\t{total_time*1000:.1f}\t{total_exp/max(len(queries),1):.1f}\t{pct:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
