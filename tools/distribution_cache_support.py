#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Shared parent-only distribution cache fixtures and exact semantics."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field


ROOT = "R"
EXPONENT = 2.0


FIXTURES = {
    "chain": {
        "parents": {
            "A": [ROOT],
            "B": ["A"],
            "C": ["B"],
        },
        "targets": [ROOT, "A", "B", "C"],
        "known": {
            ROOT: {0: 1},
            "A": {1: 1},
            "B": {2: 1},
            "C": {3: 1},
        },
    },
    "diamond": {
        "parents": {
            "A": [ROOT],
            "B": [ROOT],
            "C": ["A", "B"],
        },
        "targets": [ROOT, "A", "B", "C"],
        "known": {
            ROOT: {0: 1},
            "A": {1: 1},
            "B": {1: 1},
            "C": {2: 2},
        },
    },
    "shared_ancestor": {
        "parents": {
            "A": [ROOT],
            "B": ["A"],
            "C": ["A"],
            "D": ["B", "C"],
        },
        "targets": [ROOT, "A", "B", "C", "D"],
        "known": {
            ROOT: {0: 1},
            "A": {1: 1},
            "B": {2: 1},
            "C": {2: 1},
            "D": {3: 2},
        },
    },
    "shortcut_parent": {
        "parents": {
            "A": [ROOT],
            "B": [ROOT, "A"],
            "C": ["B"],
        },
        "targets": [ROOT, "A", "B", "C"],
        "known": {
            ROOT: {0: 1},
            "A": {1: 1},
            "B": {1: 1, 2: 1},
            "C": {2: 1, 3: 1},
        },
    },
    "disconnected": {
        "parents": {
            "A": [ROOT],
            "X": ["Y"],
            "Y": [],
        },
        "targets": [ROOT, "A", "X", "Y", "Z"],
        "known": {
            ROOT: {0: 1},
            "A": {1: 1},
            "X": {},
            "Y": {},
            "Z": {},
        },
    },
}


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    edges_examined: int = 0
    cache_hits: int = 0
    histogram_bins_scanned: int = 0
    cumulative_basis_lookups: int = 0
    hit_remaining_budgets: list[int] = field(default_factory=list)
    hit_depths_from_target: list[int] = field(default_factory=list)


def add_hist(left, right):
    out = defaultdict(int)
    for hist in (left, right):
        for length, count in hist.items():
            out[length] += count
    return dict(sorted(out.items()))


def shift_hist(hist, step=1):
    return {length + step: count for length, count in hist.items()}


def truncate_hist(hist, budget):
    return {length: count for length, count in hist.items() if length <= budget}


def exact_histogram(node, parents, root=ROOT, memo=None, visiting=None):
    """Exact unbounded parent-only path-length histogram from node to root."""
    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()
    if node in memo:
        return memo[node]
    if node == root:
        memo[node] = {0: 1}
        return memo[node]
    if node in visiting:
        raise ValueError(f"cycle detected at {node}")
    visiting.add(node)
    out = {}
    for parent in parents.get(node, []):
        out = add_hist(out, shift_hist(exact_histogram(parent, parents, root, memo, visiting)))
    visiting.remove(node)
    memo[node] = out
    return out


def full_exact_bounded(node, parents, budget, root=ROOT):
    return truncate_hist(exact_histogram(node, parents, root), budget)


def full_exact_search(node, parents, budget, root=ROOT, stats=None):
    """Budgeted parent-only search with no distribution cache."""
    if budget < 0:
        return {}
    if stats is None:
        stats = SearchStats()
    stats.nodes_expanded += 1
    if node == root:
        return {0: 1}
    if budget == 0:
        return {}
    out = {}
    for parent in parents.get(node, []):
        stats.edges_examined += 1
        out = add_hist(out, shift_hist(full_exact_search(parent, parents, budget - 1, root, stats)))
    return truncate_hist(out, budget)


def min_parent_distance(node, parents, root=ROOT, memo=None, visiting=None):
    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()
    if node in memo:
        return memo[node]
    if node == root:
        memo[node] = 0
        return 0
    if node in visiting:
        return math.inf
    visiting.add(node)
    distances = [min_parent_distance(p, parents, root, memo, visiting) for p in parents.get(node, [])]
    visiting.remove(node)
    best = min(distances, default=math.inf)
    memo[node] = math.inf if best == math.inf else best + 1
    return memo[node]


def all_nodes(parents, root=ROOT):
    nodes = {root}
    for child, ps in parents.items():
        nodes.add(child)
        nodes.update(ps)
    return nodes


def build_cache(parents, precompute_depth, root=ROOT):
    cache = {}
    distance_memo = {}
    hist_memo = {}
    for node in sorted(all_nodes(parents, root)):
        if min_parent_distance(node, parents, root, distance_memo) <= precompute_depth:
            cache[node] = exact_histogram(node, parents, root, hist_memo)
    return cache


def cached_histogram_search(node, parents, budget, cache, root=ROOT, stats=None, depth_from_target=0):
    if budget < 0:
        return {}
    if stats is None:
        stats = SearchStats()
    if node in cache:
        stats.cache_hits += 1
        stats.hit_remaining_budgets.append(budget)
        stats.hit_depths_from_target.append(depth_from_target)
        stats.histogram_bins_scanned += len(cache[node])
        return truncate_hist(cache[node], budget)
    stats.nodes_expanded += 1
    if node == root:
        return {0: 1}
    if budget == 0:
        return {}
    out = {}
    for parent in parents.get(node, []):
        stats.edges_examined += 1
        out = add_hist(
            out,
            shift_hist(
                cached_histogram_search(parent, parents, budget - 1, cache, root, stats, depth_from_target + 1)
            ),
        )
    return truncate_hist(out, budget)


def mass_cdf(hist, budget):
    return sum(count for length, count in hist.items() if length <= budget)


def first_moment_cdf(hist, budget):
    return sum(length * count for length, count in hist.items() if length <= budget)


def weighted_power_cdf(hist, budget, exponent=EXPONENT):
    return sum(((length + 1) ** (-exponent)) * count for length, count in hist.items() if length <= budget)


def interval_mass(hist, low_exclusive, high_inclusive):
    return mass_cdf(hist, high_inclusive) - mass_cdf(hist, low_exclusive)


def entropy(hist, budget):
    bounded = truncate_hist(hist, budget)
    total = sum(bounded.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in bounded.values())


def histogram_bytes(hist):
    return sys.getsizeof(hist) + sum(sys.getsizeof(length) + sys.getsizeof(count) for length, count in hist.items())


def cache_bytes(cache):
    return sys.getsizeof(cache) + sum(sys.getsizeof(node) + histogram_bytes(hist) for node, hist in cache.items())


def support_sizes(cache):
    return [len(hist) for hist in cache.values()]
