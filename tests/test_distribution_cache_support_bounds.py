#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Smoke tests for scalar parent-path support bounds."""

import unittest
from pathlib import Path

from scripts.distribution_cache_support_bounds import (
    parent_branching_moments,
    run_graph_support_bounds,
    summarize,
)
from tools.distribution_cache_support import (
    FIXTURES,
    ROOT,
    load_parent_edges_tsv,
    max_parent_distance,
    reachable_nodes_by_parent_distance,
    support_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLEWIKI_SAMPLE = REPO_ROOT / "tests" / "fixtures" / "simplewiki_articles_parent_sample.tsv"
SIMPLEWIKI_ROOT = "Category:Articles"


class DistributionCacheSupportBoundsTests(unittest.TestCase):
    def test_support_bounds_match_fixture_histogram_support(self):
        for fixture_name, fixture in FIXTURES.items():
            parents = fixture["parents"]
            with self.subTest(fixture=fixture_name):
                for node, hist in fixture["known"].items():
                    expected = (min(hist), max(hist)) if hist else (None, None)
                    self.assertEqual(support_bounds(node, parents), expected)

    def test_max_parent_distance_keeps_shortcut_width(self):
        parents = FIXTURES["shortcut_parent"]["parents"]

        self.assertEqual(support_bounds("B", parents), (1, 2))
        self.assertEqual(max_parent_distance("C", parents), 3)

    def test_parent_branching_moments_report_size_biased_parent_degree(self):
        records = run_graph_support_bounds(
            "tiny_fixture", "diamond", FIXTURES["diamond"]["parents"], FIXTURES["diamond"]["targets"], [2], ROOT
        )
        target_records = [record for record in records if record["record_type"] == "target_bounds"]
        moments = parent_branching_moments(target_records)

        self.assertEqual(moments["max_parent_degree"], 2)
        self.assertAlmostEqual(moments["size_biased_parent_branching"], 1.5)

    def test_support_bounds_runner_reports_budget_pruning_categories(self):
        records = run_graph_support_bounds(
            "tiny_fixture",
            "chain",
            FIXTURES["chain"]["parents"],
            FIXTURES["chain"]["targets"],
            budgets=[1, 3],
            root=ROOT,
            narrow_width=0,
            wide_width=2,
        )
        target_records = [record for record in records if record["record_type"] == "target_bounds"]
        signal_records = [record for record in records if record["record_type"] == "budget_signal"]

        self.assertTrue(all(record["bounds_match_histogram"] for record in target_records))
        self.assertTrue(any(record["target_node"] == "C" and record["zero_by_min_budget"] for record in signal_records))
        self.assertTrue(any(record["target_node"] == "A" and record["fully_covered_by_max_budget"] for record in signal_records))

        summary = summarize(records)
        self.assertIn("# Parent Support Bounds Benchmark Summary", summary)
        self.assertIn("zero_by_min", summary)
        self.assertIn("E[p^2]/E[p]", summary)
        self.assertIn("Root-Distance Buckets", summary)

    def test_file_backed_simplewiki_sample_bounds_match_exact_histograms(self):
        parents = load_parent_edges_tsv(SIMPLEWIKI_SAMPLE)
        targets = reachable_nodes_by_parent_distance(parents, SIMPLEWIKI_ROOT, max_depth=2)
        records = run_graph_support_bounds(
            "simplewiki_articles_parent_sample",
            "simplewiki_articles_parent_sample",
            parents,
            targets,
            budgets=[2],
            root=SIMPLEWIKI_ROOT,
        )
        target_records = [record for record in records if record["record_type"] == "target_bounds"]

        self.assertTrue(target_records)
        self.assertTrue(all(record["bounds_match_histogram"] for record in target_records))


if __name__ == "__main__":
    unittest.main()
