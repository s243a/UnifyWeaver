#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for bounded LMDB parent histogram benchmark helpers."""

import unittest

from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram, summarize, target_budget_records


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

    def test_target_budget_records_report_representation_selection(self):
        parents = {
            "A": ["R"],
            "B": ["R", "A"],
            "C": ["B"],
        }

        class Graph:
            def parents(self, node):
                return parents.get(node, [])

        records = target_budget_records(
            "fixture",
            Graph(),
            "R",
            "C",
            child_depth=2,
            budget=3,
            tail_epsilon=0.01,
            prune_thresholds=[0.01],
            path_cap=None,
            expansion_cap=None,
        )
        fit_rows = [row for row in records if row["record_type"] == "lmdb_parent_histogram_fit"]
        summary = summarize(records)

        self.assertTrue(fit_rows)
        self.assertTrue(all(row["selected_prefix_representation"] for row in fit_rows))
        self.assertTrue(all(row["selected_functional_representation"] for row in fit_rows))
        self.assertIn("Representation Policy Selection", summary)
        self.assertIn("Representation Policy By Budget And Depth", summary)


if __name__ == "__main__":
    unittest.main()
