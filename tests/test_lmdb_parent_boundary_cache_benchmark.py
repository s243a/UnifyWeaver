#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for boundary-cache parent histogram helpers."""

import unittest

from scripts.lmdb_parent_boundary_cache_benchmark import (
    build_boundary_cache,
    cached_parent_histogram,
    estimate_parametric_total_count,
    histogram_distribution_error,
    parametric_shape_distribution,
    scaled_distribution_histogram,
    support_binomial_mean,
)
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram


class DictGraph:
    def __init__(self, parents):
        self._parents = parents

    def parents(self, node):
        return self._parents.get(node, [])


class BoundaryCacheBenchmarkTests(unittest.TestCase):
    def test_boundary_cache_matches_dag_histogram(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        }
        cache = {"B": {2: 1}}

        full, _full_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)
        cached, stats = cached_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3, cache)

        self.assertEqual(full, cached)
        self.assertEqual(stats.cache_hits, 1)

    def test_boundary_cache_can_differ_when_suffix_violates_visited_state(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
            "D": ["C", "B"],
        }
        # This boundary histogram is locally valid for B, but when reached from
        # D through C, the suffix B->C->... would revisit C.  A node-only cache
        # cannot see that visited-state constraint.
        cache = {"B": {2: 1, 3: 1}}

        full, _full_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "D", "R", 4)
        cached, stats = cached_parent_histogram(lambda node: parents.get(node, []), "D", "R", 4, cache)

        self.assertNotEqual(full, cached)
        self.assertGreater(stats.cache_hits, 0)
        l1, cdf = histogram_distribution_error(full, cached)
        self.assertGreater(l1, 0.0)
        self.assertGreater(cdf, 0.0)

    def test_baseline_boundary_cache_materializes_uncapped_histogram(self):
        graph = DictGraph({"A": ["R"]})

        cache, parametric_cache, rows = build_boundary_cache(graph, "R", ["A"], 1, None, None)

        self.assertEqual(cache, {"A": {1: 1}})
        self.assertEqual(parametric_cache, {})
        self.assertEqual(rows[0]["cache_admission_action"], "materialize_exact")
        self.assertEqual(rows[0]["cache_admission_reason"], "baseline_uncapped_histogram")

    def test_depth_prior_boundary_policy_materializes_within_budget(self):
        graph = DictGraph({"A": ["R"]})

        cache, parametric_cache, rows = build_boundary_cache(
            graph,
            "R",
            ["A"],
            1,
            None,
            None,
            admission_policy="depth-prior",
            safety_factor=1.25,
            max_histogram_bytes=1024,
            parametric_bytes=64,
            max_parent_depth=4,
        )

        self.assertEqual(cache, {"A": {1: 1}})
        self.assertEqual(parametric_cache, {})
        self.assertEqual(rows[0]["cache_admission_action"], "materialize_exact")
        self.assertEqual(rows[0]["predicted_prior_bytes"], 16)

    def test_depth_prior_boundary_policy_records_parametric_without_cache_hit(self):
        graph = DictGraph({"A": ["R"]})

        cache, parametric_cache, rows = build_boundary_cache(
            graph,
            "R",
            ["A"],
            1,
            None,
            None,
            admission_policy="depth-prior",
            safety_factor=2.0,
            max_histogram_bytes=24,
            parametric_bytes=8,
            max_parent_depth=4,
        )

        self.assertEqual(cache, {})
        self.assertEqual(parametric_cache, {"A": {1: 1}})
        self.assertEqual(rows[0]["cache_admission_action"], "use_parametric_prior")
        self.assertFalse(rows[0]["cached"])
        self.assertTrue(rows[0]["parametric_cached"])

    def test_parametric_unit_mass_model_records_mass_error(self):
        graph = DictGraph({"A": ["R"], "B": ["A", "R"]})

        cache, parametric_cache, rows = build_boundary_cache(
            graph,
            "R",
            ["B"],
            2,
            None,
            None,
            admission_policy="depth-prior",
            safety_factor=2.0,
            max_histogram_bytes=24,
            parametric_bytes=8,
            parametric_mass_model="unit",
            max_parent_depth=4,
        )

        self.assertEqual(cache, {})
        self.assertEqual(sum(parametric_cache["B"].values()), 1)
        self.assertEqual(rows[0]["path_count"], 2)
        self.assertEqual(rows[0]["parametric_path_count"], 1)
        self.assertEqual(rows[0]["parametric_mass_model"], "unit")
        self.assertEqual(rows[0]["parametric_mass_delta"], -1)
        self.assertAlmostEqual(rows[0]["parametric_mass_ratio"], 0.5)

    def test_support_binomial_shape_stays_inside_boundary_support(self):
        probabilities, origin, params = parametric_shape_distribution(
            {"histogram_L_min": 2, "histogram_L_max": 5},
            {"prior_mean_excess": 1.5},
            "support-binomial",
        )

        self.assertEqual(origin, 2)
        self.assertEqual(len(probabilities), 4)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(params["mean_model"], "prior-clipped")
        self.assertAlmostEqual(params["mean_excess"], 1.5)

        midpoint, midpoint_origin, midpoint_params = parametric_shape_distribution(
            {"histogram_L_min": 2, "histogram_L_max": 5},
            {"prior_mean_excess": 99.0},
            "support-binomial-midpoint",
        )
        self.assertEqual(midpoint_origin, 2)
        self.assertEqual(len(midpoint), 4)
        self.assertGreater(midpoint[1], 0.0)
        self.assertEqual(midpoint_params["mean_model"], "midpoint")

    def test_support_binomial_mean_blends_prior_and_midpoint(self):
        self.assertEqual(support_binomial_mean(4, 99.0, "prior-clipped", 0.5), 4.0)
        self.assertEqual(support_binomial_mean(4, 99.0, "midpoint", 0.5), 2.0)
        self.assertEqual(support_binomial_mean(4, 4.0, "blend", 0.25), 2.5)

    def test_support_binomial_boundary_cache_records_support_interval(self):
        graph = DictGraph({"A": ["R"], "B": ["A", "R"]})

        cache, parametric_cache, rows = build_boundary_cache(
            graph,
            "R",
            ["B"],
            2,
            None,
            None,
            admission_policy="depth-prior",
            safety_factor=2.0,
            max_histogram_bytes=24,
            parametric_bytes=8,
            parametric_shape_model="support-binomial",
            parametric_mean_model="blend",
            parametric_mean_blend=0.25,
            parametric_mass_model="oracle",
            max_parent_depth=4,
        )

        self.assertEqual(cache, {})
        self.assertTrue(rows[0]["parametric_cached"])
        self.assertEqual(rows[0]["parametric_shape_model"], "support-binomial")
        self.assertEqual(rows[0]["parametric_mean_model"], "blend")
        self.assertEqual(rows[0]["parametric_mean_blend"], 0.25)
        self.assertGreaterEqual(rows[0]["parametric_support_min"], rows[0]["histogram_L_min"])
        self.assertLessEqual(rows[0]["parametric_support_max"], rows[0]["histogram_L_max"])
        self.assertEqual(sum(parametric_cache["B"].values()), rows[0]["path_count"])

    def test_depth_prior_mass_model_caps_branching_pressure(self):
        estimate = estimate_parametric_total_count(
            {"path_count": 2, "histogram_L_max": 5},
            {"base_mean_excess": 3.0},
            "depth-prior",
            mass_cap=10,
        )

        self.assertEqual(estimate["estimated_path_count"], 10)
        self.assertTrue(estimate["mass_capped"])

    def test_scaled_distribution_histogram_preserves_total_count(self):
        hist = scaled_distribution_histogram([0.2, 0.3, 0.5], 4, 11)

        self.assertEqual(sum(hist.values()), 11)
        self.assertEqual(min(hist), 4)

    def test_parametric_cache_hit_splices_approximate_histogram(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        }

        hist, stats = cached_parent_histogram(
            lambda node: parents.get(node, []),
            "C",
            "R",
            3,
            {},
            parametric_boundary_cache={"B": {2: 3}},
        )

        self.assertEqual(hist, {3: 3})
        self.assertEqual(stats.cache_hits, 1)
        self.assertEqual(stats.histogram_cache_hits, 0)
        self.assertEqual(stats.parametric_cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
