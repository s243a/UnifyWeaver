#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tiny parent-only distribution cache parity fixtures.

These tests exercise the first slice of docs/design/DISTRIBUTION_CACHE_BENCHMARK_PLAN.md:
exact parent-only path histograms, cached suffix cutoffs, and cumulative-basis
lookups on hand-checkable DAGs. They intentionally do not touch Wikipedia data
or fitted tails.
"""
import math
import unittest
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
    """Exact parent-only path-length histogram from node to root on a DAG."""
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
    for node in all_nodes(parents, root):
        if min_parent_distance(node, parents, root, distance_memo) <= precompute_depth:
            cache[node] = exact_histogram(node, parents, root, hist_memo)
    return cache


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    cache_hits: int = 0
    histogram_bins_scanned: int = 0
    cumulative_basis_lookups: int = 0
    hit_remaining_budgets: list = field(default_factory=list)


def cached_histogram_search(node, parents, budget, cache, root=ROOT, stats=None):
    if budget < 0:
        return {}
    if stats is None:
        stats = SearchStats()
    if node in cache:
        stats.cache_hits += 1
        stats.hit_remaining_budgets.append(budget)
        stats.histogram_bins_scanned += len(cache[node])
        return truncate_hist(cache[node], budget)
    stats.nodes_expanded += 1
    if node == root:
        return {0: 1}
    out = {}
    for parent in parents.get(node, []):
        out = add_hist(out, shift_hist(cached_histogram_search(parent, parents, budget - 1, cache, root, stats)))
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


class ParentDistributionCacheParityTests(unittest.TestCase):
    def test_exact_histograms_match_hand_checked_fixtures(self):
        for fixture_name, fixture in FIXTURES.items():
            parents = fixture["parents"]
            with self.subTest(fixture=fixture_name):
                for node, expected in fixture["known"].items():
                    self.assertEqual(exact_histogram(node, parents), expected)

    def test_cached_cutoff_matches_full_exact_for_depth_budget_grid(self):
        for fixture_name, fixture in FIXTURES.items():
            parents = fixture["parents"]
            for precompute_depth in range(0, 4):
                cache = build_cache(parents, precompute_depth)
                for budget in range(0, 6):
                    for target in fixture["targets"]:
                        with self.subTest(fixture=fixture_name, d_pre=precompute_depth, budget=budget, target=target):
                            stats = SearchStats()
                            cached = cached_histogram_search(target, parents, budget, cache, stats=stats)
                            exact = full_exact_bounded(target, parents, budget)
                            self.assertEqual(cached, exact)

    def test_cached_search_records_suffix_cache_hits(self):
        parents = FIXTURES["diamond"]["parents"]
        cache = build_cache(parents, precompute_depth=1)
        stats = SearchStats()
        got = cached_histogram_search("C", parents, budget=2, cache=cache, stats=stats)
        self.assertEqual(got, {2: 2})
        self.assertEqual(stats.cache_hits, 2)
        self.assertEqual(stats.hit_remaining_budgets, [1, 1])
        self.assertEqual(stats.nodes_expanded, 1)

    def test_cumulative_bases_match_raw_histogram_scans(self):
        for fixture_name, fixture in FIXTURES.items():
            parents = fixture["parents"]
            for target in fixture["targets"]:
                hist = exact_histogram(target, parents)
                total_budget = max(hist.keys(), default=0) + 2
                for budget in range(0, total_budget + 1):
                    with self.subTest(fixture=fixture_name, target=target, budget=budget):
                        bounded = truncate_hist(hist, budget)
                        self.assertEqual(mass_cdf(hist, budget), sum(bounded.values()))
                        self.assertEqual(first_moment_cdf(hist, budget), sum(k * v for k, v in bounded.items()))
                        self.assertAlmostEqual(
                            weighted_power_cdf(hist, budget),
                            sum(((k + 1) ** (-EXPONENT)) * v for k, v in bounded.items()),
                            places=12,
                        )
                        if budget > 0:
                            self.assertEqual(
                                interval_mass(hist, budget - 1, budget),
                                hist.get(budget, 0),
                            )

    def test_functionals_are_equal_for_cached_and_full_exact_results(self):
        parents = FIXTURES["shortcut_parent"]["parents"]
        cache = build_cache(parents, precompute_depth=1)
        budget = 3
        exact = full_exact_bounded("C", parents, budget)
        cached = cached_histogram_search("C", parents, budget, cache)
        self.assertEqual(cached, exact)
        self.assertEqual(mass_cdf(cached, budget), 2)
        self.assertEqual(first_moment_cdf(cached, budget), 5)
        self.assertAlmostEqual(weighted_power_cdf(cached, budget), (3 ** -2) + (4 ** -2), places=12)
        self.assertAlmostEqual(entropy(cached, budget), math.log(2), places=12)


if __name__ == "__main__":
    unittest.main()
