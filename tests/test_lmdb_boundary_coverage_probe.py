#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for boundary coverage probe helpers."""

import unittest

from scripts.lmdb_boundary_coverage_probe import (
    RootReachabilityFilter,
    exact_boundary_coverage,
    sample_boundary_coverage,
)


class DictGraph:
    def __init__(self, parents):
        self._parents = parents

    def parents(self, node):
        return self._parents.get(node, [])


class BoundaryCoverageProbeTests(unittest.TestCase):
    def test_exact_boundary_coverage_counts_root_and_boundary_prefixes(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "R"],
        })

        record = exact_boundary_coverage(graph.parents, "C", "R", 3, {"B"})

        self.assertEqual(record["terminal_prefixes"], 2)
        self.assertEqual(record["root_paths"], 1)
        self.assertEqual(record["boundary_hit_prefixes"], 1)
        self.assertEqual(record["budget_exhausted_prefixes"], 0)
        self.assertEqual(record["boundary_hit_fraction"], 0.5)
        self.assertEqual(record["boundary_hits_by_depth"], {1: 1})
        self.assertEqual(record["boundary_hits_by_remaining_budget"], {2: 1})
        self.assertEqual(record["boundary_suffix_path_mass_sum"], 1)
        self.assertTrue(record["completed"])

    def test_exact_boundary_coverage_skips_cycles(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A", "C"],
            "C": ["B"],
        })

        record = exact_boundary_coverage(graph.parents, "C", "R", 4, {"A"})

        self.assertEqual(record["terminal_prefixes"], 1)
        self.assertEqual(record["boundary_hit_prefixes"], 1)
        self.assertEqual(record["cycle_skips"], 1)
        self.assertTrue(record["completed"])

    def test_exact_boundary_coverage_can_filter_to_root_reachable_parents(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "X"],
            "X": ["Y"],
            "Y": [],
        })
        root_filter = RootReachabilityFilter(graph.parents, "R")

        record = exact_boundary_coverage(
            graph.parents,
            "C",
            "R",
            3,
            set(),
            reachability_filter=root_filter.can_reach,
            parent_filter_name="root-reachable",
        )

        self.assertEqual(record["parent_filter"], "root-reachable")
        self.assertEqual(record["terminal_prefixes"], 1)
        self.assertEqual(record["root_paths"], 1)
        self.assertEqual(record["budget_exhausted_prefixes"], 0)
        self.assertEqual(record["root_unreachable_parent_skips"], 1)
        self.assertEqual(record["filtered_dead_end_prefixes"], 0)

    def test_exact_boundary_coverage_unfiltered_keeps_off_root_prefixes(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "X"],
            "X": ["Y"],
            "Y": [],
        })

        record = exact_boundary_coverage(graph.parents, "C", "R", 3, set(), parent_filter_name="all")

        self.assertEqual(record["parent_filter"], "all")
        self.assertEqual(record["terminal_prefixes"], 2)
        self.assertEqual(record["root_paths"], 1)
        self.assertEqual(record["dead_end_prefixes"], 1)
        self.assertEqual(record["root_unreachable_parent_skips"], 0)

    def test_sample_boundary_coverage_estimates_weighted_prefix_counts(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "R"],
        })

        record = sample_boundary_coverage(graph.parents, "C", "R", 3, {"B"}, samples=200, seed="fixture")

        self.assertEqual(record["samples"], 200)
        self.assertGreater(record["estimated_terminal_prefixes"], 0.0)
        self.assertGreater(record["estimated_boundary_hit_prefixes"], 0.0)
        self.assertGreater(record["estimated_root_paths"], 0.0)
        self.assertIsNotNone(record["estimated_boundary_hit_fraction"])


if __name__ == "__main__":
    unittest.main()
