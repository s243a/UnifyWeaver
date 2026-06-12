#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for distribution fit comparison helpers."""

import unittest

from scripts.distribution_fit_comparison import (
    binomial_pmf,
    convolve,
    exact_excess_distribution,
    depth_prior_records,
    fitted_binomial_pmf,
    fitted_gamma_pmf,
    l1_error,
    max_cdf_error,
    nfold_convolution,
    append_realized_support_table,
    run_graph_fit_comparison,
    size_biased_excess_pmf,
    summarize,
)
from tools.distribution_cache_support import FIXTURES, ROOT


class DistributionFitComparisonTests(unittest.TestCase):
    def test_exact_excess_distribution_shifts_to_minimum_length(self):
        distribution, origin = exact_excess_distribution({2: 2, 4: 2})

        self.assertEqual(origin, 2)
        self.assertEqual(distribution, [0.5, 0.0, 0.5])

    def test_binomial_pmf_handles_bernoulli_case(self):
        self.assertEqual(binomial_pmf(1, 0.25), [0.75, 0.25])

    def test_fitted_binomial_matches_one_step_bernoulli_histogram(self):
        model, params = fitted_binomial_pmf([0.75, 0.25])

        self.assertEqual(params["trials"], 1)
        self.assertAlmostEqual(params["probability"], 0.25)
        self.assertLess(l1_error([0.75, 0.25], model), 1e-12)
        self.assertLess(max_cdf_error([0.75, 0.25], model), 1e-12)

    def test_fitted_gamma_degenerates_for_one_point_histogram(self):
        model, params = fitted_gamma_pmf([1.0])

        self.assertEqual(model, [1.0])
        self.assertEqual(params["family"], "degenerate")

    def test_size_biased_excess_pmf_weights_by_parent_degree(self):
        pmf = size_biased_excess_pmf([0, 1, 2, 2])

        self.assertEqual(pmf, [0.2, 0.8])

    def test_nfold_convolution_matches_binomial_for_bernoulli_base(self):
        empirical = nfold_convolution([0.75, 0.25], 2)

        self.assertEqual(len(empirical), 3)
        self.assertAlmostEqual(empirical[0], 0.5625)
        self.assertAlmostEqual(empirical[1], 0.375)
        self.assertAlmostEqual(empirical[2], 0.0625)
        self.assertEqual(convolve([1.0], [0.75, 0.25]), [0.75, 0.25])

    def test_depth_prior_records_report_binomial_and_gamma_models(self):
        records = depth_prior_records(
            "tiny_fixture",
            "synthetic",
            parent_degrees=[1, 1, 2],
            depths=[2],
            tail_epsilon=0.001,
            continuous_sample_points=32,
            prune_thresholds=[0.001],
        )
        models = {record["model"] for record in records}
        roles = {record["distribution_role"] for record in records}

        self.assertEqual(models, {"binomial_prior", "shifted_gamma_prior"})
        self.assertEqual(roles, {"depth_prior"})
        self.assertTrue(all(record["effective_support_bins"] >= 1 for record in records))

    def test_realized_support_table_counts_targets_once(self):
        lines = []
        rows = [
            {
                "target_node": "A",
                "L_min": 1,
                "support_bins": 2,
                "effective_support_bins": 2,
                "path_count": 2,
                "parent_degree": 1,
            },
            {
                "target_node": "A",
                "L_min": 1,
                "support_bins": 2,
                "effective_support_bins": 2,
                "path_count": 2,
                "parent_degree": 1,
            },
        ]
        append_realized_support_table(lines, rows)
        self.assertEqual(lines[0].split("|")[2].strip(), "1")

    def test_fixture_comparison_reports_both_roles(self):
        fixture = FIXTURES["shortcut_parent"]
        records = run_graph_fit_comparison(
            "tiny_fixture",
            "shortcut_parent",
            fixture["parents"],
            fixture["targets"],
            ROOT,
            depths=[2],
        )
        fit_records = [record for record in records if record["record_type"] == "distribution_fit"]
        realized_models = {record["model"] for record in fit_records if record["distribution_role"] == "realized_fit"}
        prior_models = {record["model"] for record in fit_records if record["distribution_role"] == "depth_prior"}
        summary = summarize(records)

        self.assertEqual(realized_models, {"binomial_fit", "shifted_gamma_fit"})
        self.assertEqual(prior_models, {"binomial_prior", "shifted_gamma_prior"})
        self.assertIn("Realized Histogram Fits", summary)
        self.assertIn("Depth-Conditioned Prior Distributions", summary)
        self.assertIn("Realized Support By Root Distance", summary)
        self.assertIn("Prior Support By Depth", summary)


if __name__ == "__main__":
    unittest.main()
