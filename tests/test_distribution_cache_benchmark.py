#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Smoke tests for the distribution cache benchmark runner."""

import unittest
from pathlib import Path

from scripts.distribution_cache_benchmark import run_fixture_benchmark, run_graph_benchmark, summarize
from tools.distribution_cache_support import FIXTURES, load_parent_edges_tsv, reachable_nodes_by_parent_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLEWIKI_SAMPLE = REPO_ROOT / "tests" / "fixtures" / "simplewiki_articles_parent_sample.tsv"
SIMPLEWIKI_ROOT = "Category:Articles"


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

    def test_file_backed_simplewiki_subtree_sample_records_exact_parity(self):
        parents = load_parent_edges_tsv(SIMPLEWIKI_SAMPLE)
        targets = reachable_nodes_by_parent_distance(parents, SIMPLEWIKI_ROOT, max_depth=2)
        records = run_graph_benchmark(
            "simplewiki_articles_parent_sample",
            "simplewiki_articles_parent_sample",
            parents,
            targets,
            depths=[0, 1, 2],
            budgets=[2],
            root=SIMPLEWIKI_ROOT,
        )
        query_records = [record for record in records if record["record_type"] == "query"]

        self.assertIn("Category:Articles", targets)
        self.assertIn("Category:Science", targets)
        self.assertNotIn("Category:Physics", targets)
        self.assertTrue(query_records)
        self.assertTrue(all(record["root"] == SIMPLEWIKI_ROOT for record in query_records))
        self.assertTrue(all(record["histogram_exact_match"] for record in query_records))


if __name__ == "__main__":
    unittest.main()
