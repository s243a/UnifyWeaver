#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for boundary-cache parent histogram helpers."""

import unittest

from scripts.lmdb_parent_boundary_cache_benchmark import cached_parent_histogram, histogram_distribution_error
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram


class BoundaryCacheBenchmarkTests(unittest.TestCase):
    def test_boundary_cache_matches_dag_histogram(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        }
        cache = {"B": {2: 1}}

        full, _full_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)
        cached, stats = cached_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3, cache)

        self.assertEqual(full, cached)
        self.assertEqual(stats.cache_hits, 1)

    def test_boundary_cache_can_differ_when_suffix_violates_visited_state(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
            "D": ["C", "B"],
        }
        # This boundary histogram is locally valid for B, but when reached from
        # D through C, the suffix B->C->... would revisit C.  A node-only cache
        # cannot see that visited-state constraint.
        cache = {"B": {2: 1, 3: 1}}

        full, _full_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "D", "R", 4)
        cached, stats = cached_parent_histogram(lambda node: parents.get(node, []), "D", "R", 4, cache)

        self.assertNotEqual(full, cached)
        self.assertGreater(stats.cache_hits, 0)
        l1, cdf = histogram_distribution_error(full, cached)
        self.assertGreater(l1, 0.0)
        self.assertGreater(cdf, 0.0)


if __name__ == "__main__":
    unittest.main()
