#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for the path-value kernel sweep wrapper."""

from types import SimpleNamespace
import unittest

from scripts.lmdb_path_value_kernel_sweep import (
    grouped_estimate_rows,
    parse_kernel_variants,
    probe_args_for_variant,
    summarize_sweep,
)


class PathValueKernelSweepTests(unittest.TestCase):
    def test_parse_kernel_variants_expands_supported_grid(self):
        variants = parse_kernel_variants(
            "count,bp-decay:auto,bp-decay:explicit,bp-decay:1.5,weighted-power:2",
            explicit_branching_factor=2.0,
        )

        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "count",
                "bp_decay_auto",
                "bp_decay_explicit_2p0",
                "bp_decay_1p5",
                "weighted_power_2p0",
            ],
        )
        self.assertEqual(variants[0]["path_value_kernel"], "count")
        self.assertEqual(variants[1]["path_value_kernel"], "bp-decay")
        self.assertIsNone(variants[1]["path_value_branching_factor"])
        self.assertEqual(variants[2]["path_value_branching_factor"], 2.0)
        self.assertEqual(variants[3]["path_value_branching_factor"], 1.5)
        self.assertEqual(variants[4]["path_value_kernel"], "weighted-power")
        self.assertEqual(variants[4]["path_value_power"], 2.0)

    def test_probe_args_for_variant_overrides_only_kernel_fields(self):
        args = SimpleNamespace(
            lmdb_dir="/fixture",
            root=1,
            graph_name="fixture",
            mode="all",
            parent_filter="root-cone",
            selection_source="root-cone",
            root_cone_depth=3,
            root_cone_children_per_node=None,
            root_cone_frontier_limit=None,
            require_targets_in_root_cone=False,
            require_boundaries_in_root_cone=True,
            skip_boundary_suffix_mass=False,
            boundary_depths="1",
            target_depths="2",
            children_per_node=64,
            frontier_limit=500,
            boundaries_per_depth=0,
            targets_per_depth=2,
            target_selection="boundary-descendants",
            include_target_ancestor_boundaries=False,
            target_ancestor_boundary_limit=500,
            max_parent_depth=24,
            budgets="4",
            path_count_cap=0,
            expansion_cap=0,
            samples=50,
            seed="fixture",
        )
        variant = {
            "label": "bp_decay_2p0",
            "path_value_kernel": "bp-decay",
            "path_value_branching_factor": 2.0,
            "path_value_power": 1.0,
        }

        probe_args = probe_args_for_variant(args, variant)

        self.assertEqual(probe_args.graph_name, "fixture_bp_decay_2p0")
        self.assertEqual(probe_args.path_value_kernel, "bp-decay")
        self.assertEqual(probe_args.path_value_branching_factor, 2.0)
        self.assertEqual(probe_args.path_value_power, 1.0)
        self.assertEqual(probe_args.root_cone_depth, 3)
        self.assertTrue(probe_args.require_boundaries_in_root_cone)

    def test_grouped_estimate_rows_summarizes_by_variant_mode_and_budget(self):
        records = [
            {
                "record_type": "boundary_coverage_target",
                "kernel_variant": "count",
                "mode": "root-sample",
                "path_length_budget": 4,
                "completed": True,
                "terminal_prefixes": 1,
                "root_paths": 1,
                "boundary_hit_prefixes": 0,
                "estimated_root_paths": 2.0,
                "estimated_root_value_sum": 2.0,
                "estimated_kernel_mean_root_path_length": 2.0,
                "estimated_spliced_total_root_paths": None,
                "estimated_spliced_total_value_sum": None,
                "estimated_boundary_hit_fraction": 0.0,
                "elapsed_ns": 1_000_000,
            },
            {
                "record_type": "boundary_coverage_target",
                "kernel_variant": "count",
                "mode": "root-sample",
                "path_length_budget": 4,
                "completed": True,
                "terminal_prefixes": 1,
                "root_paths": 1,
                "boundary_hit_prefixes": 1,
                "estimated_root_paths": 4.0,
                "estimated_root_value_sum": 4.0,
                "estimated_kernel_mean_root_path_length": 3.0,
                "estimated_spliced_total_root_paths": None,
                "estimated_spliced_total_value_sum": None,
                "estimated_boundary_hit_fraction": 1.0,
                "elapsed_ns": 2_000_000,
            },
        ]

        grouped = grouped_estimate_rows(records)

        self.assertEqual(len(grouped), 1)
        row = grouped[0]
        self.assertEqual(row["variant"], "count")
        self.assertEqual(row["mode"], "root-sample")
        self.assertEqual(row["budget"], 4)
        self.assertEqual(row["targets"], 2)
        self.assertEqual(row["completed_targets"], 2)
        self.assertEqual(row["observed_root_paths"], 2)
        self.assertEqual(row["observed_boundary_hit_prefixes"], 1)
        self.assertEqual(row["mean_estimated_root_paths"], 3.0)
        self.assertEqual(row["mean_estimated_root_value_sum"], 3.0)
        self.assertEqual(row["mean_estimated_kernel_mean_root_path_length"], 2.5)
        self.assertEqual(row["mean_estimated_boundary_hit_fraction"], 0.5)
        self.assertEqual(row["elapsed_ms"], 3.0)

    def test_summarize_sweep_includes_variant_and_value_fields(self):
        args = SimpleNamespace(
            graph_name="fixture",
            root=1,
            lmdb_dir="/fixture",
            mode="root-sample",
            parent_filter="root-cone",
            selection_source="root-cone",
            budgets="4",
            samples=10,
        )
        records = [
            {
                "record_type": "path_value_kernel_sweep_variant",
                "variant": "bp_decay_auto",
                "path_value_kernel": "bp-decay",
                "path_value_branching_factor": 1.5,
                "path_value_branching_factor_source": "root_cone_eligible_parent_e_p2_over_e_p",
                "path_value_power": None,
                "targets": 1,
                "boundary_nodes": 2,
                "elapsed_ns": 1_000_000,
            },
            {
                "record_type": "boundary_coverage_target",
                "kernel_variant": "bp_decay_auto",
                "mode": "root-sample",
                "path_length_budget": 4,
                "completed": True,
                "terminal_prefixes": 1,
                "root_paths": 1,
                "boundary_hit_prefixes": 0,
                "estimated_root_paths": 2.0,
                "estimated_root_value_sum": 0.5,
                "estimated_kernel_mean_root_path_length": 2.0,
                "estimated_spliced_total_root_paths": None,
                "estimated_spliced_total_value_sum": None,
                "estimated_boundary_hit_fraction": 0.0,
                "elapsed_ns": 1_000_000,
            },
        ]

        summary = summarize_sweep(records, args)

        self.assertIn("# LMDB Path-Value Kernel Sweep", summary)
        self.assertIn("bp_decay_auto", summary)
        self.assertIn("mean_est_root_paths", summary)
        self.assertIn("mean_est_root_value_sum", summary)


if __name__ == "__main__":
    unittest.main()
