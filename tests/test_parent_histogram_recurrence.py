#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for parent histogram recurrence helpers."""

import unittest

from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram
from scripts.parent_histogram_recurrence import recurrence_parent_histogram


class ParentHistogramRecurrenceTests(unittest.TestCase):
    def test_recurrence_matches_dag_path_histogram(self):
        parents = {
            "A": ["R"],
            "B": ["R", "A"],
            "C": ["A", "B"],
        }

        dfs_hist, _dfs_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)
        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)

        self.assertEqual(dfs_hist, {2: 2, 3: 1})
        self.assertEqual(rec_hist, dfs_hist)
        self.assertFalse(rec_stats.cycle_approximation)

    def test_recurrence_respects_budget_horizon(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        }

        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 2)

        self.assertEqual(rec_hist, {})
        self.assertGreater(rec_stats.budget_cutoffs, 0)

    def test_recurrence_marks_cycles_as_approximate(self):
        parents = {
            "A": ["R"],
            "B": ["A", "C"],
            "C": ["B"],
        }

        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 4)
        dfs_hist, _dfs_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 4)

        self.assertEqual(rec_hist, dfs_hist)
        self.assertTrue(rec_stats.cycle_approximation)
        self.assertGreater(rec_stats.cycle_edges, 0)

    def test_path_cap_marks_recurrence_truncated(self):
        parents = {
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        }

        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 2, path_cap=1)

        self.assertTrue(rec_stats.path_cap_hit)
        self.assertEqual(sum(rec_hist.values()), 1)


if __name__ == "__main__":
    unittest.main()
