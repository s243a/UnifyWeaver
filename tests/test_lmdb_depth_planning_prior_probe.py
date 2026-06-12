#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for depth-conditioned planning prior helpers."""

import unittest

from scripts.lmdb_depth_planning_prior_probe import compare_prior_to_hist, planning_prior_for_bucket


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


if __name__ == "__main__":
    unittest.main()
