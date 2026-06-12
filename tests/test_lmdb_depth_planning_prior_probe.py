#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for depth-conditioned planning prior helpers."""

import argparse
import unittest

from scripts.lmdb_depth_planning_prior_probe import (
    admission_decision,
    build_records,
    compare_prior_to_hist,
    planning_prior_for_bucket,
)


class LmdbDepthPlanningPriorProbeTests(unittest.TestCase):
    def test_planning_prior_mean_scales_with_depth(self):
        prior = planning_prior_for_bucket([1, 3], 2, 0.001)

        # Size-biased excess distribution: excess 0 has weight 1, excess 2 has
        # weight 3, so E[excess] = 1.5 and the depth-2 prior has mean 3.0.
        self.assertAlmostEqual(prior["base_mean_excess"], 1.5)
        self.assertAlmostEqual(prior["prior_mean_excess"], 3.0)
        self.assertEqual(prior["binomial_trials"], 2)
        self.assertEqual(prior["prior_support_bins"], 5)

    def test_compare_prior_to_degenerate_histogram(self):
        prior = planning_prior_for_bucket([1, 1, 1], 3, 0.001)

        comparison = compare_prior_to_hist({3: 7}, prior, 0.001)

        self.assertTrue(comparison["comparable"])
        self.assertEqual(comparison["histogram_L_min"], 3)
        self.assertEqual(comparison["support_bins"], 1)
        self.assertEqual(comparison["l1_error"], 0.0)
        self.assertEqual(comparison["max_cdf_error"], 0.0)
        self.assertEqual(comparison["empirical_storage_prediction_ratio"], 1.0)

    def test_admission_marks_capped_recurrence_risky(self):
        prior = planning_prior_for_bucket([1, 1, 1], 3, 0.001)
        comparison = compare_prior_to_hist({3: 7}, prior, 0.001)

        decision = admission_decision(
            {"recurrence_capped": True, "recurrence_cycle_approximation": False},
            prior,
            comparison,
            1.25,
        )

        self.assertEqual(decision, "risky_try_capped_or_approx")

    def test_build_records_adds_family_ratios_and_decision(self):
        args = argparse.Namespace(
            graph_name="fixture",
            root="R",
            tail_epsilon=0.001,
            max_prior_depth=3,
            safety_factor=1.25,
        )
        target_rows = [{
            "target_node": "A",
            "child_sample_depth": 1,
            "L_min": 3,
            "L_max": 3,
            "distance_truncated": False,
            "cycle_skipped": False,
            "full_parent_degree": 1,
            "root_reaching_parent_degree": 1,
            "recurrence_histogram": {3: 2},
            "recurrence_cycle_approximation": False,
            "recurrence_capped": False,
            "recurrence_states_evaluated": 1,
        }]

        records = build_records(args, target_rows, {0: 1, 1: 1})
        target_record = next(row for row in records if row["record_type"] == "depth_planning_prior_target")

        self.assertEqual(target_record["admission_decision"], "exact_recurrence_likely_cheap")
        self.assertIn("binomial_storage_prediction_ratio", target_record)
        self.assertIn("gamma_storage_prediction_ratio", target_record)
        self.assertIn("safety_storage_prediction_ratio", target_record)


if __name__ == "__main__":
    unittest.main()
