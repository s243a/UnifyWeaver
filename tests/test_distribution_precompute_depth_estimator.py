#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for distribution precompute depth estimator helpers."""

import unittest

from scripts.distribution_precompute_depth_estimator import (
    apply_branching_profile,
    branching_profile_from_records,
    build_records,
    calibration_from_payload_recurrence_summaries,
    cumulative_branching,
    parse_args,
    parse_depth_float_map,
    path_value_sweep_measurements_from_records,
    validation_measurements_from_records,
)


class DistributionPrecomputeDepthEstimatorTests(unittest.TestCase):

    def test_branching_profile_extracts_overall_and_depth_priors(self):
        profile = branching_profile_from_records([
            {
                "record_type": "root_conditioned_branching_selection",
                "graph": "fixture",
                "max_nodes": 100,
                "truncated": True,
            },
            {
                "record_type": "root_conditioned_branching_overall",
                "root_conditioned_parent_degree": {
                    "nodes": 10,
                    "mean_parent_degree": 1.25,
                    "second_parent_degree_moment": 3.0,
                    "size_biased_parent_branching": 2.4,
                    "mean_excess": 1.4,
                    "max_parent_degree": 5,
                    "p95_parent_degree": 4,
                    "p99_parent_degree": 5,
                },
            },
            {
                "record_type": "root_conditioned_branching_depth_bucket",
                "child_depth": 1,
                "root_conditioned_parent_degree": {
                    "nodes": 4,
                    "mean_parent_degree": 1.0,
                    "size_biased_parent_branching": 1.5,
                    "mean_excess": 0.5,
                    "max_parent_degree": 2,
                },
            },
        ], source_paths=["profile.jsonl"])

        self.assertEqual(profile["source_paths"], ["profile.jsonl"])
        self.assertAlmostEqual(profile["branching_factor"], 2.4)
        self.assertEqual(profile["overall"]["nodes"], 10)
        self.assertEqual(profile["selection"]["max_nodes"], 100)
        self.assertEqual(profile["depth_branching"], {1: 1.5})

    def test_apply_branching_profile_preserves_manual_overrides(self):
        args = parse_args([
            "--branching-factor", "9",
            "--depth-branching", "2:7",
            "--max-depth", "2",
        ])
        profile = {
            "branching_factor": 2.4,
            "depth_branching": {1: 1.5, 2: 2.5},
        }

        original_loader = apply_branching_profile.__globals__["branching_profile_from_jsonl"]
        try:
            apply_branching_profile.__globals__["branching_profile_from_jsonl"] = lambda paths, degree_scope: dict(profile)
            args.branching_profile_jsonl = ["fixture.jsonl"]
            args.branching_profile_degree_scope = "root_conditioned_parent_degree"
            applied = apply_branching_profile(args)
        finally:
            apply_branching_profile.__globals__["branching_profile_from_jsonl"] = original_loader

        self.assertAlmostEqual(args.branching_factor, 9.0)
        self.assertEqual(parse_depth_float_map(args.depth_branching), {1: 1.5, 2: 7.0})
        self.assertTrue(applied["manual_branching_factor_override"])
        self.assertEqual(applied["manual_depth_branching_overrides"], {2: 7.0})

    def test_apply_branching_profile_updates_estimator_default_branching(self):
        args = parse_args(["--expected-queries", "1000", "--max-depth", "1"])
        profile = {
            "branching_factor": 2.0,
            "depth_branching": {},
        }

        original_loader = apply_branching_profile.__globals__["branching_profile_from_jsonl"]
        try:
            apply_branching_profile.__globals__["branching_profile_from_jsonl"] = lambda paths, degree_scope: dict(profile)
            args.branching_profile_jsonl = ["fixture.jsonl"]
            args.branching_profile_degree_scope = "root_conditioned_parent_degree"
            applied = apply_branching_profile(args)
        finally:
            apply_branching_profile.__globals__["branching_profile_from_jsonl"] = original_loader

        rows = [
            row for row in build_records(args, branching_profile=applied)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["representation"] == "exact_sparse_histogram"
        ]

        self.assertAlmostEqual(rows[1]["expected_hits"], 500.0)
        self.assertAlmostEqual(rows[1]["branching_factor"], 2.0)

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

        self.assertEqual([row["boundary_depth"] for row in rows], [0, 1, 2])
        self.assertAlmostEqual(rows[0]["expected_hits"], 1000.0)
        self.assertAlmostEqual(rows[1]["expected_hits"], 250.0)
        self.assertAlmostEqual(rows[2]["expected_hits"], 62.5)

    def test_sampled_point_limit_is_upper_bound_and_break_even_can_be_lower(self):
        args = parse_args([
            "--expected-queries", "10",
            "--branching-factor", "4",
            "--max-depth", "4",
            "--target-depth", "8",
        ])

        row = next(
            row for row in build_records(args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["boundary_depth"] == 4
            and row["representation"] == "sampled_up_to_50_point_distribution"
        )

        self.assertEqual(row["suffix_hops"], 4)
        self.assertEqual(row["points"], 5)
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
            and row["boundary_depth"] == 2
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertAlmostEqual(row["expected_hits"], 62.5)

    def test_suffix_cost_uses_target_depth_not_boundary_depth(self):
        args = parse_args([
            "--branching-factor", "4",
            "--max-depth", "2",
            "--target-depth", "5",
        ])

        row = next(
            row for row in build_records(args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["boundary_depth"] == 2
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertEqual(row["suffix_hops"], 3)
        self.assertAlmostEqual(row["expected_build_states"], 16.0)
        self.assertAlmostEqual(row["expected_suffix_states"], 64.0)



    def test_validation_measurements_clip_negative_saved_per_hit(self):
        measurements = validation_measurements_from_records([
            {
                "record_type": "boundary_cache_selection",
                "graph": "fixture_b3_recurrence",
                "boundary_counts": {"0": 1, "1": 2, "2": 3, "3": 4},
                "target_counts": {"0": 1, "1": 2, "4": 5},
                "boundary_nodes": 4,
                "cached_boundary_nodes": 4,
            },
            {
                "record_type": "boundary_cache_comparison",
                "graph": "fixture_b3_recurrence",
                "full_time_ns": 100.0,
                "cached_time_ns": 120.0,
                "cache_hits": 10,
                "full_nodes_expanded": 50,
                "cached_nodes_expanded": 50,
                "cache_payload_bytes_read": 64,
                "cache_decode_ns": 8,
                "full_expansion_cap_hit": True,
                "cached_expansion_cap_hit": True,
            },
            {
                "record_type": "boundary_cache_comparison",
                "graph": "fixture_b3_recurrence",
                "full_time_ns": 200.0,
                "cached_time_ns": 240.0,
                "cache_hits": 0,
                "full_nodes_expanded": 60,
                "cached_nodes_expanded": 60,
                "cache_payload_bytes_read": 0,
                "cache_decode_ns": 0,
                "full_expansion_cap_hit": False,
                "cached_expansion_cap_hit": False,
            },
        ])

        measurement = measurements[3]
        self.assertEqual(measurement["target_depth"], 4)
        self.assertEqual(measurement["rows"], 2)
        self.assertAlmostEqual(measurement["mean_cache_hits"], 10.0)
        self.assertAlmostEqual(measurement["all_rows_mean_cache_hits"], 5.0)
        self.assertEqual(measurement["positive_cache_hit_rows"], 1)
        self.assertEqual(measurement["zero_cache_hit_rows"], 1)
        self.assertTrue(measurement["validation_usable_for_cap"])
        self.assertAlmostEqual(measurement["measured_saved_per_hit_ns"], -2.0)
        self.assertEqual(measurement["clipped_saved_per_hit_ns"], 0.0)
        self.assertFalse(measurement["measured_pays"])

    def test_validation_measurements_require_cache_hits_to_pay(self):
        measurements = validation_measurements_from_records([
            {
                "record_type": "boundary_cache_comparison",
                "graph": "fixture_b7_recurrence",
                "full_time_ns": 120.0,
                "cached_time_ns": 100.0,
                "cache_hits": 0,
            },
        ])

        measurement = measurements[7]
        self.assertEqual(measurement["positive_cache_hit_rows"], 0)
        self.assertEqual(measurement["zero_cache_hit_rows"], 1)
        self.assertFalse(measurement["validation_usable_for_cap"])
        self.assertIsNone(measurement["measured_saved_per_hit_ns"])
        self.assertEqual(measurement["clipped_saved_per_hit_ns"], 0.0)
        self.assertFalse(measurement["measured_pays"])

    def test_validation_cap_mode_uses_measured_saved_per_hit(self):
        args = parse_args([
            "--expected-queries", "1000",
            "--branching-factor", "2",
            "--max-depth", "3",
            "--target-depth", "5",
            "--cap-mode", "validation",
            "--uncached-cost-per-state", "10",
            "--cached-eval-cost-per-point", "0",
        ])
        validation = {
            2: {
                "rows": 2,
                "mean_time_ratio": 1.2,
                "mean_full_time_ns": 100.0,
                "mean_cached_time_ns": 120.0,
                "mean_cache_hits": 5.0,
                "positive_cache_hit_rows": 2,
                "zero_cache_hit_rows": 0,
                "validation_usable_for_cap": True,
                "all_rows_mean_cache_hits": 5.0,
                "mean_payload_bytes_read": 64.0,
                "mean_decode_ns": 8.0,
                "measured_pays": False,
                "measured_saved_per_hit_ns": -4.0,
                "clipped_saved_per_hit_ns": 0.0,
            }
        }

        row = next(
            row for row in build_records(args, validation_measurements=validation)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["boundary_depth"] == 2
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertEqual(row["saved_per_hit"], 0.0)
        self.assertIsNone(row["hits_to_break_even"])
        self.assertFalse(row["precompute_pays"])
        self.assertTrue(row["validation_prediction_matches_measured"])

    def test_measured_cap_limits_suffix_work(self):
        args = parse_args([
            "--branching-factor", "4",
            "--max-depth", "2",
            "--target-depth", "5",
            "--cap-mode", "measured",
            "--estimated-full-work", "10",
        ])

        row = next(
            row for row in build_records(args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["boundary_depth"] == 2
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertAlmostEqual(row["expected_suffix_states"], 64.0)
        self.assertAlmostEqual(row["cap_limited_suffix_states"], 10.0)
        self.assertAlmostEqual(row["uncached_suffix_cost"], 10.0)

    def test_decode_cost_is_per_hit_unless_memoized(self):
        uncached_args = parse_args([
            "--branching-factor", "4",
            "--max-depth", "1",
            "--target-depth", "3",
            "--decode-cost-per-byte", "2",
        ])
        memoized_args = parse_args([
            "--branching-factor", "4",
            "--max-depth", "1",
            "--target-depth", "3",
            "--decode-cost-per-byte", "2",
            "--decode-memoized",
        ])

        uncached = next(
            row for row in build_records(uncached_args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["boundary_depth"] == 1
            and row["representation"] == "exact_sparse_histogram"
        )
        memoized = next(
            row for row in build_records(memoized_args)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["boundary_depth"] == 1
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertGreater(uncached["per_hit_decode_cost"], 0.0)
        self.assertEqual(memoized["per_hit_decode_cost"], 0.0)
        self.assertGreater(memoized["saved_per_hit"], uncached["saved_per_hit"])

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

    def test_path_value_sweep_measurements_group_by_variant_and_depth(self):
        measurements = path_value_sweep_measurements_from_records([
            {
                "record_type": "boundary_coverage_selection",
                "kernel_variant": "bp_decay_auto",
                "graph": "fixture",
                "mode": "all",
                "parent_filter": "all",
                "selection_source": "root-cone",
                "boundary_counts": {"1": 4},
                "target_counts": {"3": 2},
                "path_value_kernel": "bp-decay",
                "path_value_branching_factor": 2.0,
                "path_value_branching_factor_source": "test",
                "path_value_power": None,
            },
            {
                "record_type": "boundary_coverage_target",
                "kernel_variant": "bp_decay_auto",
                "graph": "fixture",
                "mode": "root-sample",
                "path_length_budget": 6,
                "estimated_boundary_hit_fraction": 0.5,
                "estimated_root_boundary_hit_fraction": 0.25,
                "estimated_root_value_boundary_hit_fraction": 0.125,
                "estimated_root_paths": 8.0,
                "estimated_root_value_sum": 2.0,
            },
            {
                "record_type": "boundary_coverage_target",
                "kernel_variant": "bp_decay_auto",
                "graph": "fixture",
                "mode": "root-sample",
                "path_length_budget": 6,
                "estimated_boundary_hit_fraction": 0.25,
                "estimated_root_boundary_hit_fraction": 0.5,
                "estimated_root_value_boundary_hit_fraction": 0.25,
                "estimated_root_paths": 4.0,
                "estimated_root_value_sum": 1.0,
            },
        ])

        measurement = measurements[("bp_decay_auto", 1)]
        self.assertEqual(measurement["boundary_depth"], 1)
        self.assertEqual(measurement["target_depth"], 3)
        self.assertEqual(measurement["rows"], 2)
        self.assertEqual(measurement["path_value_kernel"], "bp-decay")
        self.assertAlmostEqual(measurement["mean_estimated_boundary_hit_fraction"], 0.375)
        self.assertAlmostEqual(measurement["mean_estimated_root_boundary_hit_fraction"], 0.375)
        self.assertAlmostEqual(measurement["mean_estimated_root_value_boundary_hit_fraction"], 0.1875)
        self.assertAlmostEqual(measurement["mean_estimated_root_paths"], 6.0)
        self.assertAlmostEqual(measurement["mean_estimated_root_value_sum"], 1.5)

    def test_value_weighted_score_scales_planning_hits(self):
        args = parse_args([
            "--expected-queries", "100",
            "--branching-factor", "2",
            "--max-depth", "1",
            "--target-depth", "3",
            "--recommendation-score", "value-weighted",
            "--uncached-cost-per-state", "1",
            "--cached-eval-cost-per-point", "0",
            "--storage-cost-per-byte", "0",
            "--decode-cost-per-byte", "0",
            "--sample-fit-cost-per-point", "0",
            "--parametric-fit-cost", "0",
        ])
        path_value_measurements = {
            ("bp_decay_auto", 1): {
                "mode": "root-sample",
                "budget": 6,
                "boundary_depth": 1,
                "target_depth": 3,
                "rows": 2,
                "path_value_kernel": "bp-decay",
                "path_value_branching_factor": 2.0,
                "path_value_power": None,
                "mean_estimated_boundary_hit_fraction": 0.5,
                "mean_estimated_root_boundary_hit_fraction": 0.25,
                "mean_estimated_root_value_boundary_hit_fraction": 0.1,
                "mean_estimated_root_paths": 8.0,
                "mean_estimated_root_value_sum": 2.0,
            }
        }

        row = next(
            row for row in build_records(args, path_value_measurements=path_value_measurements)
            if row["record_type"] == "distribution_precompute_depth_estimate"
            and row["kernel_variant"] == "bp_decay_auto"
            and row["boundary_depth"] == 1
            and row["representation"] == "exact_sparse_histogram"
        )

        self.assertAlmostEqual(row["expected_hits"], 50.0)
        self.assertAlmostEqual(row["path_value_hit_scale"], 0.1)
        self.assertAlmostEqual(row["planning_hits"], 5.0)
        self.assertGreater(row["expected_net_value"], row["value_weighted_net_value"])
        self.assertEqual(row["recommendation_net_value"], row["value_weighted_net_value"])
        self.assertEqual(row["path_value_hit_scale_source"], "mean_estimated_root_value_boundary_hit_fraction")


if __name__ == "__main__":
    unittest.main()
