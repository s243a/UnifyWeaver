#!/usr/bin/env python3
"""Tests for recurrence threshold calibration grid helpers."""

import unittest
from types import SimpleNamespace

from scripts.lmdb_recurrence_threshold_grid import grid_cases, summarize, summarize_case


class RecurrenceThresholdGridTests(unittest.TestCase):
    def test_grid_cases_expands_thresholds_and_blends(self):
        args = SimpleNamespace(
            max_recurrence_states_grid="25,50",
            max_effective_bins_grid="3",
            mean_models="midpoint,blend",
            blend_values="0.25,0.75",
            parametric_mean_blend=0.5,
        )

        cases = grid_cases(args)

        self.assertEqual(len(cases), 6)
        self.assertIn({
            "max_recurrence_states": 25,
            "max_effective_bins_after_trim": 3,
            "mean_model": "midpoint",
            "mean_blend": 0.5,
        }, cases)
        self.assertIn({
            "max_recurrence_states": 50,
            "max_effective_bins_after_trim": 3,
            "mean_model": "blend",
            "mean_blend": 0.75,
        }, cases)

    def test_summarize_case_reports_threshold_and_error_metrics(self):
        case = {
            "max_recurrence_states": 50,
            "max_effective_bins_after_trim": 4,
            "mean_model": "midpoint",
            "mean_blend": 0.5,
        }
        records = [
            {
                "record_type": "boundary_cache_selection",
                "graph": "fixture",
                "root": "R",
                "boundary_nodes": 2,
                "cached_boundary_nodes": 1,
                "parametric_boundary_nodes": 1,
            },
            {
                "record_type": "boundary_cache_entry",
                "parametric_cached": False,
                "approximation_forced_by_threshold": False,
                "recurrence_states_over_limit": False,
                "effective_bins_over_limit": False,
                "recurrence_states_evaluated": 10,
                "effective_support_bins_after_trim": 2,
                "parametric_support_bins": 0,
            },
            {
                "record_type": "boundary_cache_entry",
                "parametric_cached": True,
                "approximation_forced_by_threshold": True,
                "recurrence_states_over_limit": True,
                "effective_bins_over_limit": False,
                "recurrence_states_evaluated": 75,
                "effective_support_bins_after_trim": 3,
                "parametric_support_bins": 3,
                "parametric_mass_ratio": 1.0,
            },
            {
                "record_type": "boundary_cache_comparison",
                "budget": 6,
                "l1_error": 0.25,
                "max_cdf_error": 0.1,
                "path_count_relative_error": 0.5,
                "abs_path_count_delta": 2,
                "node_expansion_ratio": 0.75,
                "parametric_cache_hits": 1,
                "parametric_bins_spliced": 3,
            },
        ]

        row = summarize_case(case, records)

        self.assertEqual(row["forced_parametric"], 1)
        self.assertEqual(row["states_over_limit"], 1)
        self.assertEqual(row["bins_over_limit"], 0)
        self.assertEqual(row["threshold_disagree"], 1)
        self.assertAlmostEqual(row["mean_recurrence_states"], 42.5)
        self.assertAlmostEqual(row["mean_forced_recurrence_states"], 75.0)
        self.assertAlmostEqual(row["budget_6_mean_l1"], 0.25)
        self.assertAlmostEqual(row["budget_6_mean_path_count_relative_error"], 0.5)

    def test_summary_renders_budget_table(self):
        records = [
            {
                "record_type": "recurrence_threshold_grid_selection",
                "graph": "fixture",
                "root": "R",
            },
            {
                "record_type": "recurrence_threshold_grid_case",
                "max_recurrence_states": 50,
                "max_effective_bins_after_trim": 4,
                "mean_model": "midpoint",
                "mean_blend": 0.5,
                "boundary_nodes": 2,
                "histogram_cached": 1,
                "parametric_cached": 1,
                "forced_parametric": 1,
                "states_over_limit": 1,
                "bins_over_limit": 0,
                "threshold_disagree": 1,
                "mean_recurrence_states": 42.5,
                "mean_effective_bins_after_trim": 2.5,
                "mean_forced_recurrence_states": 75.0,
                "budget_6_mean_l1": 0.25,
                "budget_6_max_l1": 0.25,
                "budget_6_mean_cdf": 0.1,
                "budget_6_mean_path_count_relative_error": 0.5,
                "budget_6_mean_abs_path_delta": 2.0,
                "budget_6_mean_node_ratio": 0.75,
                "budget_6_mean_param_hits": 1.0,
                "budget_6_mean_param_bins_spliced": 3.0,
            },
        ]

        rendered = summarize(records)

        self.assertIn("# Recurrence Approximation Threshold Grid", rendered)
        self.assertIn("## Budget 6", rendered)
        self.assertIn("| 50 | 4 | midpoint | n/a | 1 | 1 | 0.250000 |", rendered)


if __name__ == "__main__":
    unittest.main()
