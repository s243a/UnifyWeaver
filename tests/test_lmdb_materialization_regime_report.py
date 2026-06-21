#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for LMDB materialization regime report helpers."""

import unittest

from scripts.lmdb_materialization_regime_report import (
    markdown_report,
    summarize_cache,
    summarize_profile,
)


class MaterializationRegimeReportTests(unittest.TestCase):
    def test_profile_summary_preserves_raw_and_root_conditioned_branching(self):
        records = [
            {
                "record_type": "root_conditioned_branching_selection",
                "graph": "fixture",
                "root": 1,
                "retained_nodes": 4,
                "max_observed_child_depth": 2,
                "truncated_by_nodes": False,
                "truncated_by_depth": False,
            },
            {
                "record_type": "root_conditioned_branching_overall",
                "full_parent_degree": {
                    "nodes": 4,
                    "mean_parent_degree": 2.0,
                    "size_biased_parent_branching": 4.0,
                    "mean_excess": 3.0,
                    "max_parent_degree": 6,
                },
                "root_conditioned_parent_degree": {
                    "nodes": 4,
                    "mean_parent_degree": 1.25,
                    "size_biased_parent_branching": 1.8,
                    "mean_excess": 0.8,
                    "max_parent_degree": 2,
                },
            },
            {
                "record_type": "root_conditioned_branching_depth_bucket",
                "child_depth": 2,
                "nodes": 2,
                "full_parent_degree": {
                    "nodes": 2,
                    "mean_parent_degree": 3.0,
                    "size_biased_parent_branching": 5.0,
                },
                "root_conditioned_parent_degree": {
                    "nodes": 2,
                    "mean_parent_degree": 1.5,
                    "size_biased_parent_branching": 2.0,
                },
                "mean_outside_parent_fraction": 0.5,
            },
        ]

        summary = summarize_profile("fixture", ["profile.jsonl"], records)

        self.assertEqual(summary["label"], "fixture")
        self.assertEqual(summary["root_conditioned_parent_degree"]["size_biased_parent_branching"], 1.8)
        self.assertEqual(summary["full_parent_degree"]["size_biased_parent_branching"], 4.0)
        self.assertEqual(summary["depth_rows"][0]["child_depth"], 2)

    def test_cache_summary_classifies_sparse_low_mass_histograms(self):
        records = [
            {
                "record_type": "boundary_cache_selection",
                "graph": "simple",
                "root": 1,
                "boundary_counts": {"0": 1, "1": 3},
                "target_counts": {"2": 2},
                "cached_boundary_nodes": 2,
                "targets": 2,
                "budgets": [4, 6],
                "boundary_budget": 6,
                "boundary_builder": "recurrence",
                "admission_policy": "baseline",
            },
            {
                "record_type": "boundary_cache_entry",
                "cached": True,
                "parametric_cached": False,
                "path_count": 1,
                "support_bins": 1,
                "effective_support_bins_after_trim": 1,
                "recurrence_states_evaluated": 3,
                "cache_payload_bytes": 40,
            },
            {
                "record_type": "boundary_cache_entry",
                "cached": True,
                "parametric_cached": False,
                "path_count": 2,
                "support_bins": 2,
                "effective_support_bins_after_trim": 2,
                "recurrence_states_evaluated": 5,
                "cache_payload_bytes": 52,
            },
            {
                "record_type": "boundary_cache_comparison",
                "cache_hits": 1,
                "histogram_cache_hits": 1,
                "parametric_cache_hits": 0,
                "full_path_count": 2,
                "cached_path_count": 2,
                "full_time_ns": 100,
                "cached_time_ns": 120,
                "l1_error": 0.0,
                "max_cdf_error": 0.0,
            },
        ]

        summary = summarize_cache("simplewiki", ["cache.jsonl"], records, point_cap=50)

        self.assertEqual(summary["histogram_regime"], "exact_sparse_low_mass")
        self.assertEqual(summary["max_effective_support_bins"], 2)
        self.assertEqual(summary["pct_effective_bins_le_point_cap"], 100.0)
        self.assertFalse(summary["measured_cache_faster"])

    def test_cache_summary_classifies_sparse_high_mass_histograms(self):
        records = [
            {"record_type": "boundary_cache_selection", "graph": "enwiki", "root": 7},
            {
                "record_type": "boundary_cache_entry",
                "cached": True,
                "parametric_cached": False,
                "path_count": 250,
                "support_bins": 6,
                "effective_support_bins_after_trim": 6,
                "recurrence_states_evaluated": 400,
            },
        ]

        summary = summarize_cache("enwiki", ["cache.jsonl"], records, point_cap=50)

        self.assertEqual(summary["histogram_regime"], "exact_sparse_high_mass")
        self.assertEqual(summary["mean_path_count"], 250.0)

    def test_markdown_calls_out_point_cap_as_upper_bound(self):
        report = {
            "point_cap": 50,
            "profiles": [
                {
                    "label": "fixture",
                    "graph": "profile",
                    "root": 1,
                    "retained_nodes": 2,
                    "max_observed_child_depth": 1,
                    "truncated_by_depth": False,
                    "truncated_by_nodes": False,
                    "full_parent_degree": {
                        "mean_parent_degree": 2.0,
                        "size_biased_parent_branching": 3.0,
                        "max_parent_degree": 4,
                    },
                    "root_conditioned_parent_degree": {
                        "mean_parent_degree": 1.0,
                        "size_biased_parent_branching": 1.0,
                        "max_parent_degree": 1,
                    },
                    "depth_rows": [],
                    "source_paths": ["profile.jsonl"],
                }
            ],
            "caches": [
                {
                    "label": "fixture",
                    "graph": "cache",
                    "entries": 1,
                    "exact_histogram_entries": 1,
                    "parametric_entries": 0,
                    "mean_effective_support_bins": 1.0,
                    "max_effective_support_bins": 1,
                    "pct_effective_bins_le_point_cap": 100.0,
                    "mean_path_count": 1.0,
                    "max_path_count": 1,
                    "mean_recurrence_states": 2.0,
                    "max_recurrence_states": 2,
                    "mean_payload_bytes": 40.0,
                    "histogram_regime": "exact_sparse_low_mass",
                    "targets": 1,
                    "budgets": [4],
                    "comparison_rows": 1,
                    "mean_cache_hits": 1.0,
                    "positive_cache_hit_rows": 1,
                    "mean_full_path_count": 1.0,
                    "mean_time_ratio": 1.2,
                    "measured_cache_faster": False,
                    "mean_l1_error": 0.0,
                    "mean_max_cdf_error": 0.0,
                    "source_paths": ["cache.jsonl"],
                }
            ],
        }

        markdown = markdown_report(report, "fixture_report")

        self.assertIn("Point cap: `50`", markdown)
        self.assertIn("representation upper bound", markdown)
        self.assertIn("exact_sparse_low_mass", markdown)


if __name__ == "__main__":
    unittest.main()
