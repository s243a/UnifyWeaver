#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for bounded LMDB parent histogram benchmark helpers."""

import unittest

from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram


class BoundedParentHistogramTests(unittest.TestCase):
    def test_counts_shortcut_paths_within_budget(self):
        parents = {
            "A": ["R"],
            "B": ["R", "A"],
            "C": ["B"],
        }

        hist, stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)

        self.assertEqual(hist, {2: 1, 3: 1})
        self.assertGreater(stats.edges_examined, 0)
        self.assertFalse(stats.path_cap_hit)

    def test_budget_cuts_off_longer_paths(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        }

        hist, stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 2)

        self.assertEqual(hist, {})
        self.assertGreater(stats.budget_cutoffs, 0)

    def test_cycle_policy_skips_revisited_nodes(self):
        parents = {
            "A": ["R"],
            "B": ["A", "C"],
            "C": ["B"],
        }

        hist, stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 4)

        self.assertEqual(hist, {3: 1})
        self.assertEqual(stats.cycle_skips, 1)

    def test_path_cap_is_reported(self):
        parents = {
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        }

        hist, stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 2, path_cap=1)

        self.assertEqual(sum(hist.values()), 1)
        self.assertTrue(stats.path_cap_hit)


if __name__ == "__main__":
    unittest.main()
