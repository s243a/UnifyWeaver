#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for exact sparse depth sweep helpers."""

import unittest

from scripts.lmdb_exact_sparse_depth_sweep import (
    break_even_hits,
    histogram_metrics,
    markdown_summary,
    summary_row,
)


class ExactSparseDepthSweepTests(unittest.TestCase):
    def test_histogram_metrics_reports_effective_support_and_tail_pruning(self):
        metrics = histogram_metrics({2: 100, 3: 1}, 0.01, [0.01])

        self.assertTrue(metrics["reachable"])
        self.assertEqual(metrics["path_count"], 101)
        self.assertEqual(metrics["support_bins"], 2)
        self.assertEqual(metrics["effective_support_bins"], 1)
        self.assertEqual(metrics["tail_pruning"]["0.01"]["kept_bins"], 1)

    def test_break_even_hits_uses_cached_bins_as_eval_cost(self):
        self.assertAlmostEqual(break_even_hits(5, 20, 10), 0.5)
        self.assertIsNone(break_even_hits(5, 10, 10))

    def test_summary_bucket_classifies_exact_sparse_rows(self):
        row = {
            "reachable": True,
            "exact_match": True,
            "exact_sparse_under_point_cap": True,
            "recurrence_cycle_approximation": False,
            "dfs_path_cap_hit": False,
            "dfs_expansion_cap_hit": False,
            "recurrence_path_cap_hit": False,
            "recurrence_expansion_cap_hit": False,
            "path_count": 2,
            "support_bins": 1,
            "effective_support_bins": 1,
            "exact_histogram_bytes": 16,
            "dfs_nodes_expanded": 10,
            "recurrence_states_evaluated": 3,
            "state_expansion_ratio": 0.3,
            "time_ratio": 0.4,
            "hits_to_break_even_states": 3 / 9,
        }

        summary = summary_row((2, 10), [row], 50)

        self.assertEqual(summary["exact_sparse_under_point_cap_rows"], 1)
        self.assertEqual(summary["exact_match_rows"], 1)
        self.assertEqual(summary["max_effective_support_bins"], 1)
        self.assertAlmostEqual(summary["mean_hits_to_break_even_states"], 3 / 9)

    def test_markdown_summary_mentions_break_even_and_point_cap(self):
        summary = {
            "graph": "fixture",
            "root": 1,
            "point_cap": 50,
            "tail_epsilon": 0.01,
            "selection": {
                "selection_counts": {0: 1, 2: 1},
                "selected_targets": [{"child_sample_depth": 2}],
                "root_reachable_targets": [{"child_sample_depth": 2}],
                "filtered_targets": [],
            },
            "buckets": [
                {
                    "child_sample_depth": 2,
                    "budget": 10,
                    "rows": 1,
                    "exact_sparse_under_point_cap_rows": 1,
                    "exact_match_rows": 1,
                    "mean_path_count": 1.0,
                    "max_path_count": 1,
                    "mean_effective_support_bins": 1.0,
                    "max_effective_support_bins": 1,
                    "pct_effective_bins_le_point_cap": 100.0,
                    "mean_dfs_nodes_expanded": 3.0,
                    "mean_recurrence_states_evaluated": 1.0,
                    "mean_state_expansion_ratio": 0.333,
                    "mean_time_ratio": 0.5,
                    "mean_hits_to_break_even_states": 0.5,
                }
            ],
        }

        markdown = markdown_summary(summary)

        self.assertIn("Point cap: `50`", markdown)
        self.assertIn("mean_break_even_hits", markdown)
        self.assertIn("exact_sparse", markdown)


if __name__ == "__main__":
    unittest.main()
