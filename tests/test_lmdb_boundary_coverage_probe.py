#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for boundary coverage probe helpers."""

from types import SimpleNamespace
import unittest

from scripts.lmdb_boundary_coverage_probe import (
    PathValueKernel,
    RootConeFilter,
    RootReachabilityFilter,
    build_root_cone,
    estimate_parent_branching_factor,
    exact_boundary_coverage,
    sample_boundary_coverage,
    sample_root_path_space,
    select_nodes_by_root_cone_depth,
    resolve_path_value_kernel,
)


class DictGraph:
    def __init__(self, parents):
        self._parents = parents

    def parents(self, node):
        return self._parents.get(node, [])


class BoundaryCoverageProbeTests(unittest.TestCase):
    def test_exact_boundary_coverage_counts_root_and_boundary_prefixes(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "R"],
        })

        record = exact_boundary_coverage(graph.parents, "C", "R", 3, {"B"})

        self.assertEqual(record["terminal_prefixes"], 2)
        self.assertEqual(record["root_paths"], 1)
        self.assertEqual(record["boundary_hit_prefixes"], 1)
        self.assertEqual(record["budget_exhausted_prefixes"], 0)
        self.assertEqual(record["boundary_hit_fraction"], 0.5)
        self.assertEqual(record["boundary_hits_by_depth"], {1: 1})
        self.assertEqual(record["boundary_hits_by_remaining_budget"], {2: 1})
        self.assertEqual(record["boundary_suffix_path_mass_sum"], 1)
        self.assertTrue(record["completed"])

    def test_exact_boundary_coverage_skips_cycles(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A", "C"],
            "C": ["B"],
        })

        record = exact_boundary_coverage(graph.parents, "C", "R", 4, {"A"})

        self.assertEqual(record["terminal_prefixes"], 1)
        self.assertEqual(record["boundary_hit_prefixes"], 1)
        self.assertEqual(record["cycle_skips"], 1)
        self.assertTrue(record["completed"])

    def test_exact_boundary_coverage_can_filter_to_root_reachable_parents(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "X"],
            "X": ["Y"],
            "Y": [],
        })
        root_filter = RootReachabilityFilter(graph.parents, "R")

        record = exact_boundary_coverage(
            graph.parents,
            "C",
            "R",
            3,
            set(),
            reachability_filter=root_filter.can_reach,
            parent_filter_name="root-reachable",
        )

        self.assertEqual(record["parent_filter"], "root-reachable")
        self.assertEqual(record["terminal_prefixes"], 1)
        self.assertEqual(record["root_paths"], 1)
        self.assertEqual(record["budget_exhausted_prefixes"], 0)
        self.assertEqual(record["root_unreachable_parent_skips"], 1)
        self.assertEqual(record["filtered_dead_end_prefixes"], 0)

    def test_exact_boundary_coverage_unfiltered_keeps_off_root_prefixes(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "X"],
            "X": ["Y"],
            "Y": [],
        })

        record = exact_boundary_coverage(graph.parents, "C", "R", 3, set(), parent_filter_name="all")

        self.assertEqual(record["parent_filter"], "all")
        self.assertEqual(record["terminal_prefixes"], 2)
        self.assertEqual(record["root_paths"], 1)
        self.assertEqual(record["dead_end_prefixes"], 1)
        self.assertEqual(record["root_unreachable_parent_skips"], 0)

    def test_root_cone_filter_uses_precomputed_depth_bound(self):
        depth_by_node = {"R": 0, "A": 1, "B": 2}
        root_filter = RootConeFilter(depth_by_node)

        self.assertTrue(root_filter.can_reach("B", 2))
        self.assertFalse(root_filter.can_reach("B", 1))
        self.assertFalse(root_filter.can_reach("X", 3))
        self.assertEqual(root_filter.remaining_misses, 1)
        self.assertEqual(root_filter.depth_misses, 1)

    def test_build_root_cone_records_minimum_child_depths(self):
        graph = DictGraph({
            "R": ["A", "B"],
            "A": ["C"],
            "B": ["C", "D"],
            "C": ["E"],
        })

        depth_by_node, counts = build_root_cone(
            graph.parents,
            "R",
            3,
            children_per_node=0,
            frontier_limit=0,
            seed="fixture",
        )

        self.assertEqual(depth_by_node["R"], 0)
        self.assertEqual(depth_by_node["A"], 1)
        self.assertEqual(depth_by_node["B"], 1)
        self.assertEqual(depth_by_node["C"], 2)
        self.assertEqual(depth_by_node["D"], 2)
        self.assertEqual(depth_by_node["E"], 3)
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 2)

    def test_select_nodes_by_root_cone_depth_samples_requested_depths(self):
        selected, depths, counts = select_nodes_by_root_cone_depth(
            {"R": 0, "A": 1, "B": 1, "C": 2},
            [1, 2],
            nodes_per_depth=1,
            seed="fixture",
        )

        self.assertEqual(counts, {1: 2, 2: 1})
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(depths.values()), {1, 2})

    def test_estimate_parent_branching_factor_uses_e_p2_over_e_p(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        })

        stats = estimate_parent_branching_factor(graph.parents, ["A", "B", "C"])

        self.assertAlmostEqual(stats["mean_parent_degree"], 4.0 / 3.0)
        self.assertAlmostEqual(stats["second_parent_degree_moment"], 2.0)
        self.assertAlmostEqual(stats["branching_factor"], 1.5)

    def test_resolve_path_value_kernel_defaults_bp_from_root_cone(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        })
        args = SimpleNamespace(
            path_value_kernel="bp-decay",
            path_value_branching_factor=None,
            path_value_power=1.0,
            parent_filter="root-cone",
            selection_source="root-cone",
        )

        kernel, stats = resolve_path_value_kernel(
            args,
            graph,
            "R",
            ["C"],
            [],
            {"R": 0, "A": 1, "B": 1, "C": 2},
        )

        self.assertEqual(kernel.name, "bp-decay")
        self.assertEqual(kernel.branching_factor_source, "root_cone_eligible_parent_e_p2_over_e_p")
        self.assertAlmostEqual(kernel.branching_factor, 1.5)
        self.assertAlmostEqual(stats["branching_factor"], 1.5)

    def test_sample_boundary_coverage_estimates_weighted_prefix_counts(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B", "R"],
        })

        record = sample_boundary_coverage(graph.parents, "C", "R", 3, {"B"}, samples=200, seed="fixture")

        self.assertEqual(record["samples"], 200)
        self.assertGreater(record["estimated_terminal_prefixes"], 0.0)
        self.assertGreater(record["estimated_boundary_hit_prefixes"], 0.0)
        self.assertGreater(record["estimated_root_paths"], 0.0)
        self.assertIsNotNone(record["estimated_boundary_hit_fraction"])

    def test_sample_boundary_coverage_splices_boundary_suffix_mass(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        })

        record = sample_boundary_coverage(graph.parents, "C", "R", 3, {"B"}, samples=10, seed="fixture")

        self.assertEqual(record["boundary_hit_prefixes"], 10)
        self.assertEqual(record["estimated_boundary_hit_prefixes"], 1.0)
        self.assertEqual(record["estimated_boundary_spliced_root_paths"], 1.0)
        self.assertEqual(record["estimated_spliced_total_root_paths"], 1.0)

    def test_sample_boundary_coverage_splices_bp_decay_value_sum(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        })
        kernel = PathValueKernel(name="bp-decay", branching_factor=2.0, branching_factor_source="test")

        record = sample_boundary_coverage(
            graph.parents,
            "C",
            "R",
            3,
            {"B"},
            samples=10,
            seed="fixture",
            path_value_kernel=kernel,
        )

        self.assertEqual(record["estimated_spliced_total_root_paths"], 1.0)
        self.assertAlmostEqual(record["estimated_boundary_spliced_value_sum"], 0.125)
        self.assertAlmostEqual(record["estimated_spliced_total_value_sum"], 0.125)
        self.assertAlmostEqual(record["mean_boundary_suffix_path_value"], 0.125)

    def test_sample_boundary_coverage_splices_weighted_power_value_sum(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        })
        kernel = PathValueKernel(name="weighted-power", power=2.0)

        record = sample_boundary_coverage(
            graph.parents,
            "C",
            "R",
            3,
            {"B"},
            samples=10,
            seed="fixture",
            path_value_kernel=kernel,
        )

        self.assertAlmostEqual(record["estimated_spliced_total_value_sum"], 1.0 / 16.0)

    def test_sample_boundary_coverage_disables_suffix_splice_with_filter(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        })

        record = sample_boundary_coverage(
            graph.parents,
            "C",
            "R",
            3,
            {"B"},
            samples=10,
            seed="fixture",
            reachability_filter=lambda _node, _remaining: True,
            parent_filter_name="test-filter",
        )

        self.assertEqual(record["boundary_hit_prefixes"], 10)
        self.assertEqual(record["estimated_boundary_hit_prefixes"], 1.0)
        self.assertIsNone(record["estimated_boundary_spliced_root_paths"])
        self.assertIsNone(record["estimated_spliced_total_root_paths"])
        self.assertFalse(record["boundary_suffix_mass_measured"])

    def test_sample_root_path_space_estimates_root_count_and_mean_length(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        })

        record = sample_root_path_space(graph.parents, "C", "R", 3, set(), samples=200, seed="fixture")

        self.assertEqual(record["samples"], 200)
        self.assertEqual(record["root_paths"], 200)
        self.assertAlmostEqual(record["estimated_root_paths"], 2.0)
        self.assertAlmostEqual(record["estimated_mean_root_path_length"], 2.0)
        self.assertAlmostEqual(record["estimated_root_path_length_sum"], 4.0)

    def test_sample_root_path_space_estimates_bp_decay_value_sum(self):
        graph = DictGraph({
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        })
        kernel = PathValueKernel(name="bp-decay", branching_factor=2.0, branching_factor_source="test")

        record = sample_root_path_space(
            graph.parents,
            "C",
            "R",
            3,
            set(),
            samples=200,
            seed="fixture",
            path_value_kernel=kernel,
        )

        self.assertAlmostEqual(record["estimated_root_paths"], 2.0)
        self.assertAlmostEqual(record["estimated_root_value_sum"], 0.5)
        self.assertAlmostEqual(record["estimated_kernel_mean_root_path_length"], 2.0)


if __name__ == "__main__":
    unittest.main()
