#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for policy-selected boundary-cache runtime helpers."""

import unittest

from scripts.lmdb_ancestor_cache_policy_benchmark import AncestorSpace
from scripts.lmdb_ancestor_cache_policy_sweep import CacheConfig, QueryInput
from scripts.lmdb_policy_cache_runtime_benchmark import (
    build_policy_boundary_cache,
    markdown_summary,
    policy_cache_entries,
    summarize_runtime,
)


class PolicyCacheRuntimeBenchmarkTests(unittest.TestCase):
    def test_policy_cache_entries_retains_admitted_nodes(self):
        query = QueryInput(target=3, child_depth=1, space=AncestorSpace(nodes={0, 1, 2, 3}))
        distances = {
            0: {"L_min": 0, "L_max": 0, "truncated": False},
            1: {"L_min": 1, "L_max": 1, "truncated": False},
            2: {"L_min": 1, "L_max": 1, "truncated": False},
            3: {"L_min": 2, "L_max": 2, "truncated": False},
        }

        cache, actions = policy_cache_entries([query], distances, CacheConfig(8, 2, 2))

        self.assertEqual({entry.node for entry in cache.values()}, {0, 1, 2, 3})
        self.assertEqual(actions["insert"], 4)

    def test_build_policy_boundary_cache_materializes_histograms(self):
        query = QueryInput(target=3, child_depth=1, space=AncestorSpace(nodes={0, 1, 2, 3}))
        distances = {
            0: {"L_min": 0, "L_max": 0, "truncated": False},
            1: {"L_min": 1, "L_max": 1, "truncated": False},
            2: {"L_min": 1, "L_max": 1, "truncated": False},
            3: {"L_min": 2, "L_max": 2, "truncated": False},
        }
        cache, _actions = policy_cache_entries([query], distances, CacheConfig(8, 2, 2))
        parents = {3: [1, 2], 2: [0], 1: [0]}

        boundary_cache, rows = build_policy_boundary_cache(
            lambda node: parents.get(node, []),
            0,
            cache,
            boundary_budget=3,
            path_cap=None,
            expansion_cap=None,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(boundary_cache[3], {2: 2})
        self.assertEqual(boundary_cache[1], {1: 1})
        self.assertEqual(rows[0]["cache_payload_representation"], "packed_sparse_histogram")
        self.assertGreater(rows[0]["cache_payload_bytes"], 0)

    def test_summarize_runtime_reports_budget_rows(self):
        class Args:
            graph_name = "tiny"
            root = 0
            target_depths = "1"
            boundary_budget = 2
            path_cap = 100
            expansion_cap = 1000

        query = QueryInput(target=3, child_depth=1, space=AncestorSpace(nodes={0, 1, 2, 3}))
        cache_rows = [
            {
                "cached": True,
                "nodes_expanded": 1,
                "path_cap_hit": False,
                "expansion_cap_hit": False,
                "raw_histogram_bytes": 16,
                "cache_payload_bytes": 32,
                "cache_payload_decoded_max_cdf_error": 0.0,
            }
        ]
        comparison_rows = [
            {
                "budget": 2,
                "l1_error": 0.0,
                "max_cdf_error": 0.0,
                "node_expansion_ratio": 0.5,
                "cache_hits": 1,
                "full_time_ns": 100,
                "cached_time_ns": 50,
                "cache_payload_bytes_read": 32,
                "cache_decode_ns": 5,
                "full_path_cap_hit": False,
                "full_expansion_cap_hit": False,
                "cached_path_cap_hit": False,
                "cached_expansion_cap_hit": False,
            }
        ]

        summary = summarize_runtime(
            Args(),
            {0: 1, 1: 1},
            [query],
            CacheConfig(8, 2, 2),
            {"insert": 1},
            cache_rows,
            comparison_rows,
        )

        self.assertEqual(summary["materialized_boundary_entries"], 1)
        self.assertEqual(summary["budget_rows"][0]["mean_node_expansion_ratio"], 0.5)
        self.assertEqual(summary["budget_rows"][0]["mean_time_ratio"], 0.5)
        self.assertIn("| slots |", markdown_summary([summary]))


if __name__ == "__main__":
    unittest.main()
