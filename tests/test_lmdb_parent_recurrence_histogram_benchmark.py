#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for parent recurrence histogram benchmark reporting helpers."""

import unittest

from scripts.lmdb_parent_recurrence_histogram_benchmark import histogram_error, markdown_summary, summarize


class ParentRecurrenceHistogramBenchmarkTests(unittest.TestCase):
    def test_histogram_error_reports_l1_cdf_and_w1(self):
        l1, cdf, w1 = histogram_error({1: 1, 2: 1}, {1: 2})

        self.assertAlmostEqual(l1, 1.0)
        self.assertAlmostEqual(cdf, 0.5)
        self.assertAlmostEqual(w1, 0.5)

    def test_summary_reports_w1_and_certificate_fields(self):
        summary = summarize([
            {
                "record_type": "parent_recurrence_histogram_comparison",
                "graph": "fixture",
                "budget": 4,
                "dfs_histogram": {1: 1},
                "recurrence_histogram": {1: 1},
                "recurrence_cycle_approximation": False,
                "l1_error": 0.0,
                "max_cdf_error": 0.0,
                "w1_cdf_error": 0.0,
                "total_error_max_cdf": 0.0,
                "total_error_w1_cdf": 0.0,
                "state_expansion_ratio": 0.25,
                "time_ratio": 0.5,
                "dfs_path_cap_hit": False,
                "dfs_expansion_cap_hit": False,
                "recurrence_path_cap_hit": False,
                "recurrence_expansion_cap_hit": False,
            }
        ])

        row = summary["budget_rows"][0]
        self.assertEqual(row["mean_w1_cdf_error"], 0.0)
        self.assertEqual(row["mean_total_error_max_cdf"], 0.0)
        self.assertEqual(row["mean_total_error_w1_cdf"], 0.0)
        self.assertIn("mean_w1", markdown_summary(summary))


if __name__ == "__main__":
    unittest.main()
