#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for distribution fit comparison helpers."""

import unittest

from scripts.distribution_fit_comparison import (
    binomial_pmf,
    convolve,
    exact_excess_distribution,
    choose_distribution_representation,
    cheapest_candidate_within,
    depth_prior_records,
    fitted_binomial_pmf,
    fitted_gamma_pmf,
    l1_error,
    max_cdf_error,
    nfold_convolution,
    append_realized_support_table,
    packed_exact_candidates,
    packed_sparse_histogram_bytes,
    parametric_candidate_from_model,
    quantized_cdf_table_pmf,
    run_graph_fit_comparison,
    shift_distribution,
    size_biased_excess_pmf,
    summarize,
    total_error_certificate,
    unique_distribution_records,
    w1_cdf_error,
    weighted_parent_certificate,
    representation_policy_candidates,
    tail_pruned_pmf,
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

    def test_cdf_error_certificates_are_shift_invariant(self):
        exact = [0.2, 0.3, 0.5]
        model = [0.1, 0.4, 0.5]

        self.assertAlmostEqual(max_cdf_error(exact, model), max_cdf_error(shift_distribution(exact), shift_distribution(model)))
        self.assertAlmostEqual(w1_cdf_error(exact, model), w1_cdf_error(shift_distribution(exact), shift_distribution(model)))

    def test_weighted_parent_certificate_uses_mass_weights(self):
        inherited = weighted_parent_certificate([2.0, 6.0], [0.1, 0.3])

        self.assertAlmostEqual(inherited, 0.25)

    def test_error_certificate_adds_inherited_and_fit_terms(self):
        self.assertAlmostEqual(total_error_certificate(0.2, 0.05), 0.25)
        self.assertAlmostEqual(total_error_certificate(-0.2, 0.05), 0.05)

    def test_weighted_parent_certificate_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            weighted_parent_certificate([1.0], [0.1, 0.2])

    def test_tail_pruned_pmf_drops_suffix_with_error_certificate(self):
        approximation, summary = tail_pruned_pmf([0.7, 0.2, 0.07, 0.03], 0.05)

        self.assertEqual(approximation, [0.7, 0.2, 0.07, 0.0])
        self.assertEqual(summary["kept_bins"], 3)
        self.assertEqual(summary["dropped_bins"], 1)
        self.assertAlmostEqual(summary["dropped_mass"], 0.03)
        self.assertAlmostEqual(max_cdf_error([0.7, 0.2, 0.07, 0.03], approximation), 0.03)

    def test_quantized_cdf_table_roundtrips_with_bounded_error(self):
        approximation, summary = quantized_cdf_table_pmf([0.25, 0.25, 0.5], bits=8)

        self.assertEqual(len(approximation), 3)
        self.assertAlmostEqual(sum(approximation), 1.0)
        self.assertLessEqual(max_cdf_error([0.25, 0.25, 0.5], approximation), summary["quantization_step"])

    def test_packed_exact_candidates_include_sparse_tail_and_cdf(self):
        candidates = packed_exact_candidates([0.7, 0.2, 0.07, 0.03], [0.05], cdf_bits=8)
        representations = {candidate["representation"] for candidate in candidates}

        self.assertEqual(representations, {"packed_sparse_histogram", "tail_pruned_histogram", "quantized_cdf_table"})
        tail = [candidate for candidate in candidates if candidate["representation"] == "tail_pruned_histogram"][0]
        self.assertAlmostEqual(tail["max_cdf_error"], 0.03)

    def test_cheapest_candidate_within_uses_error_gate_then_bytes(self):
        candidates = [
            {"representation": "bad_small", "bytes_estimate": 4, "max_cdf_error": 0.2, "w1_cdf_error": 0.2},
            {"representation": "good_large", "bytes_estimate": 20, "max_cdf_error": 0.0, "w1_cdf_error": 0.0},
            {"representation": "good_small", "bytes_estimate": 12, "max_cdf_error": 0.01, "w1_cdf_error": 0.01},
        ]

        self.assertEqual(cheapest_candidate_within(candidates, max_cdf=0.05)["representation"], "good_small")
        self.assertEqual(cheapest_candidate_within(candidates, max_cdf=0.005, max_w1=0.005)["representation"], "good_large")

    def test_packed_sparse_histogram_bytes_counts_nonzero_bins(self):
        self.assertLess(packed_sparse_histogram_bytes(1), packed_sparse_histogram_bytes(2))

    def test_representation_selector_chooses_cheapest_candidate_under_policy(self):
        candidates = [
            {
                "representation": "exact_histogram",
                "candidate_source": "exact",
                "bytes_estimate": 100,
                "max_cdf_error": 0.0,
                "w1_cdf_error": 0.0,
                "workloads": ["prefix_mass", "arbitrary_functional"],
            },
            {
                "representation": "quantized_cdf_table",
                "candidate_source": "packed_exact",
                "bytes_estimate": 20,
                "max_cdf_error": 0.001,
                "w1_cdf_error": 0.004,
                "workloads": ["prefix_mass"],
            },
            {
                "representation": "parametric:binomial_fit",
                "candidate_source": "parametric",
                "bytes_estimate": 16,
                "max_cdf_error": 0.2,
                "w1_cdf_error": 0.2,
                "workloads": ["prefix_mass", "arbitrary_functional"],
            },
        ]

        selected = choose_distribution_representation(candidates, max_cdf=0.01, max_w1=0.01, workload="prefix_mass")

        self.assertEqual(selected["selected_representation"], "quantized_cdf_table")
        rejected = {row["representation"]: row["rejection_reason"] for row in selected["evaluated_candidates"]}
        self.assertEqual(rejected["parametric:binomial_fit"], "max_cdf_error_exceeds_policy")

    def test_representation_selector_respects_workload_support(self):
        candidates = [
            {
                "representation": "quantized_cdf_table",
                "candidate_source": "packed_exact",
                "bytes_estimate": 10,
                "max_cdf_error": 0.0,
                "w1_cdf_error": 0.0,
                "workloads": ["prefix_mass"],
            },
            {
                "representation": "exact_histogram",
                "candidate_source": "exact",
                "bytes_estimate": 40,
                "max_cdf_error": 0.0,
                "w1_cdf_error": 0.0,
                "workloads": ["prefix_mass", "arbitrary_functional"],
            },
        ]

        selected = choose_distribution_representation(candidates, max_cdf=0.0, workload="arbitrary_functional")

        self.assertEqual(selected["selected_representation"], "exact_histogram")

    def test_policy_candidates_add_exact_packed_and_parametric_sources(self):
        model = {"model": "binomial_fit", "max_cdf_error": 0.01, "w1_cdf_error": 0.02, "l1_error": 0.03}
        candidates = representation_policy_candidates(
            100,
            [{"representation": "quantized_cdf_table", "bytes_estimate": 12, "max_cdf_error": 0.0, "w1_cdf_error": 0.0}],
            parametric_candidate_from_model(model, 32),
        )

        self.assertEqual([candidate["candidate_source"] for candidate in candidates], ["exact", "packed_exact", "parametric"])
        self.assertEqual(candidates[-1]["representation"], "parametric:binomial_fit")

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
        self.assertTrue(all(record["best_packed_exact_cdf"] is not None for record in records))
        self.assertTrue(all(record["selected_prefix_representation"] is not None for record in records))
        self.assertTrue(all(record["selected_functional_representation"] is not None for record in records))

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
        self.assertIn("Packed Exact Candidates", summary)
        self.assertIn("Representation Policy Selection", summary)

    def test_unique_distribution_records_keeps_lmdb_target_budget_models(self):
        rows = [
            {"record_type": "lmdb_parent_histogram_fit", "target_node": 1, "budget": 4, "model": "a"},
            {"record_type": "lmdb_parent_histogram_fit", "target_node": 1, "budget": 4, "model": "b"},
            {"record_type": "lmdb_parent_histogram_fit", "target_node": 1, "budget": 6, "model": "a"},
            {"record_type": "lmdb_parent_histogram_fit", "target_node": 1, "budget": 4, "model": "a"},
        ]

        unique = unique_distribution_records(rows)

        self.assertEqual(len(unique), 3)


if __name__ == "__main__":
    unittest.main()
