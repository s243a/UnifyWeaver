#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Smoke tests for the distribution cache benchmark runner."""

import unittest

from scripts.distribution_cache_benchmark import run_fixture_benchmark, summarize
from tools.distribution_cache_support import FIXTURES


class DistributionCacheBenchmarkTests(unittest.TestCase):
    def test_tiny_fixture_benchmark_records_exact_cached_parity(self):
        records = run_fixture_benchmark("diamond", FIXTURES["diamond"], depths=[0, 1], budgets=[2])
        query_records = [record for record in records if record["record_type"] == "query"]
        cache_records = [record for record in records if record["record_type"] == "cache_build"]

        self.assertEqual(len(cache_records), 2)
        self.assertTrue(query_records)
        self.assertTrue(all(record["histogram_exact_match"] for record in query_records))
        self.assertTrue(any(record["mode"] == "cached_histogram" and record["cache_hits"] > 0 for record in query_records))

    def test_summary_reports_grid_cells_and_exact_failures(self):
        records = run_fixture_benchmark("shortcut_parent", FIXTURES["shortcut_parent"], depths=[0, 1], budgets=[2, 4])
        summary = summarize(records)

        self.assertIn("| fixture | D_pre | B_search |", summary)
        self.assertIn("shortcut_parent", summary)
        self.assertIn("| 0 |", summary)


if __name__ == "__main__":
    unittest.main()
