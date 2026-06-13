#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for distribution precompute depth estimator helpers."""

import unittest

from scripts.distribution_precompute_depth_estimator import (
    build_records,
    calibration_from_payload_recurrence_summaries,
    cumulative_branching,
    parse_args,
    parse_depth_float_map,
)


class DistributionPrecomputeDepthEstimatorTests(unittest.TestCase):
    def test_cumulative_branching_uses_depth_overrides(self):
        overrides = parse_depth_float_map("1:2,2:3")

        self.assertEqual(cumulative_branching(0, 4.0, overrides), 1.0)
        self.assertEqual(cumulative_branching(1, 4.0, overrides), 2.0)
        self.assertEqual(cumulative_branching(2, 4.0, overrides), 6.0)
        self.assertEqual(cumulative_branching(3, 4.0, overrides), 24.0)

    def test_expected_hits_decay_by_branching_factor(self):
        args = parse_args([
            "--expected-queries", "1000",
            "--branching-factor", "4",
            "--max-depth", "2",
        ])

        rows = [
            row for row in build_records(args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["representation"] == "exact_sparse_histogram"
        ]

        self.assertAlmostEqual(rows[0]["expected_hits"], 1000.0)
        self.assertAlmostEqual(rows[1]["expected_hits"], 250.0)
        self.assertAlmostEqual(rows[2]["expected_hits"], 62.5)

    def test_fifty_point_model_can_break_even_below_fifty_hits(self):
        args = parse_args([
            "--expected-queries", "10",
            "--branching-factor", "4",
            "--max-depth", "4",
        ])

        row = next(
            row for row in build_records(args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["depth"] == 4
            and row["representation"] == "sampled_50_point_distribution"
        )

        self.assertEqual(row["points"], 50)
        self.assertLess(row["hits_to_break_even"], 50.0)

    def test_depth_specific_query_reach_probability_scales_hits(self):
        args = parse_args([
            "--expected-queries", "1000",
            "--branching-factor", "2",
            "--max-depth", "2",
            "--query-reach-probability", "2:0.25",
        ])

        row = next(
            row for row in build_records(args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["depth"] == 2
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertAlmostEqual(row["expected_hits"], 62.5)

    def test_calibration_reads_payload_recurrence_summary(self):
        args = parse_args([
            "--branching-factor", "4",
            "--max-depth", "4",
        ])

        calibration = calibration_from_payload_recurrence_summaries(
            ["docs/reports/enwiki_mtc_payload_recurrence_layer_depth3_smoke_payload_recurrence_layer_summary.json"],
            args,
        )

        self.assertEqual(calibration["source_summaries"], 1)
        self.assertGreater(calibration["source_budget_rows"], 0)
        self.assertGreater(calibration["uncached_cost_per_state"], 0.0)
        self.assertGreater(calibration["cached_eval_cost_per_point"], 0.0)
        self.assertGreater(calibration["decode_cost_per_byte"], 0.0)


if __name__ == "__main__":
    unittest.main()
