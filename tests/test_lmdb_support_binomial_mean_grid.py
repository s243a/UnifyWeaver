#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for support-binomial mean grid helpers."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.lmdb_support_binomial_mean_grid import (
    benchmark_args_for_case,
    grid_cases,
    summarize,
)


class SupportBinomialMeanGridTests(unittest.TestCase):
    def base_args(self):
        return SimpleNamespace(
            lmdb_dir=Path("unused"),
            root=1,
            graph_name="grid",
            boundary_depth_grid="1,2",
            target_depth_grid="3",
            mean_models="midpoint,blend",
            blend_values="0.0,0.05",
            children_per_node=8,
            frontier_limit=32,
            boundaries_per_depth=4,
            targets_per_depth=2,
            boundary_budget=6,
            budgets="6,8",
            admission_policy="depth-prior",
            safety_factor=1.25,
            max_histogram_bytes=64,
            parametric_bytes=64,
            parametric_mean_blend=0.5,
            parametric_mass_model="oracle",
            parametric_mass_cap=1000,
            tail_epsilon=0.001,
            max_parent_depth=12,
            path_cap=100,
            expansion_cap=200,
            seed="seed",
            output_dir=None,
        )

    def test_grid_cases_expands_blend_values(self):
        cases = grid_cases(self.base_args())

        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[0]["mean_model"], "midpoint")
        self.assertEqual(cases[1]["mean_blend"], 0.0)
        self.assertEqual(cases[2]["mean_blend"], 0.05)

    def test_benchmark_args_for_case_sets_support_binomial(self):
        args = self.base_args()
        case = {"boundary_depth": 2, "target_depth": 4, "mean_model": "blend", "mean_blend": 0.05}

        projected = benchmark_args_for_case(args, case)

        self.assertEqual(projected.boundary_depths, "2")
        self.assertEqual(projected.target_depths, "4")
        self.assertEqual(projected.parametric_shape_model, "support-binomial")
        self.assertEqual(projected.parametric_mean_model, "blend")
        self.assertEqual(projected.parametric_mean_blend, 0.05)

    def test_summarize_includes_budget_table(self):
        records = [
            {
                "record_type": "support_binomial_mean_grid_selection",
                "graph": "grid",
                "root": 1,
            },
            {
                "record_type": "support_binomial_mean_grid_case",
                "graph": "grid_bd1_td3_midpoint",
                "root": 1,
                "boundary_depth": 1,
                "target_depth": 3,
                "mean_model": "midpoint",
                "mean_blend": 0.5,
                "boundary_nodes": 4,
                "parametric_cached": 2,
                "mean_shape_probability": 0.5,
                "mean_parametric_bins": 3.0,
                "budget_6_rows": 2,
                "budget_6_mean_l1": 0.25,
                "budget_6_mean_cdf": 0.125,
                "budget_6_mean_path_count_relative_error": 0.5,
                "budget_6_mean_abs_path_delta": 4.0,
                "budget_6_mean_param_hits": 1.0,
                "budget_6_mean_param_bins_spliced": 2.0,
            },
        ]

        summary = summarize(records)

        self.assertIn("Support-Binomial Mean Calibration Grid", summary)
        self.assertIn("Budget 6", summary)
        self.assertIn("midpoint", summary)


if __name__ == "__main__":
    unittest.main()
