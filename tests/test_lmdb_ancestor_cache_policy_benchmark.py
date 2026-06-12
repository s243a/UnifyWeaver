#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for fixed-size ancestor cache policy helpers."""

import unittest

from scripts.lmdb_ancestor_cache_policy_benchmark import (
    CacheEntry,
    collect_ancestor_space,
    slot_for,
    update_cache,
)
from scripts.lmdb_ancestor_cache_policy_sweep import (
    CacheConfig,
    QueryInput,
    parse_grid,
    simulate_config,
)


class AncestorCachePolicyBenchmarkTests(unittest.TestCase):
    def test_slot_for_is_direct_mapped_by_node_id(self):
        self.assertEqual(slot_for(11, 8), 3)

    def test_update_cache_overwrites_existing_entry_outside_current_cone(self):
        cache = {1: CacheEntry(node=1, l_min=1, l_max=1)}
        candidate = CacheEntry(node=9, l_min=2, l_max=2)

        action = update_cache(cache, candidate, ancestor_nodes={9}, cache_slots=8)

        self.assertEqual(action, "overwrite_outside_cone")
        self.assertEqual(cache[1].node, 9)

    def test_update_cache_keeps_better_existing_entry_inside_current_cone(self):
        cache = {1: CacheEntry(node=1, l_min=1, l_max=2)}
        candidate = CacheEntry(node=9, l_min=1, l_max=4)

        action = update_cache(cache, candidate, ancestor_nodes={1, 9}, cache_slots=8)

        self.assertEqual(action, "keep_existing")
        self.assertEqual(cache[1].node, 1)

    def test_update_cache_replaces_lower_priority_existing_entry_inside_current_cone(self):
        cache = {1: CacheEntry(node=1, l_min=1, l_max=5)}
        candidate = CacheEntry(node=9, l_min=1, l_max=2)

        action = update_cache(cache, candidate, ancestor_nodes={1, 9}, cache_slots=8)

        self.assertEqual(action, "overwrite_lower_priority")
        self.assertEqual(cache[1].node, 9)

    def test_update_cache_refreshes_same_node(self):
        cache = {1: CacheEntry(node=9, l_min=1, l_max=2)}

        action = update_cache(cache, CacheEntry(node=9, l_min=1, l_max=2), ancestor_nodes={9}, cache_slots=8)

        self.assertEqual(action, "refresh")
        self.assertEqual(cache[1].hits, 1)
        self.assertEqual(cache[1].stores, 2)

    def test_collect_ancestor_space_skips_cycles_and_reports_caps(self):
        parents = {
            "A": ["B", "C"],
            "B": ["A", "R"],
            "C": ["R"],
        }

        space = collect_ancestor_space(lambda node: parents.get(node, []), "A", max_nodes=3, max_edges=10)

        self.assertEqual(space.nodes, {"A", "B", "C"})
        self.assertEqual(space.cycle_edges, 1)
        self.assertTrue(space.capped)

    def test_parse_grid_defaults_or_expands_ranges(self):
        self.assertEqual(parse_grid(None, 7), [7])
        self.assertEqual(parse_grid("2,4-6", 7), [2, 4, 5, 6])

    def test_sweep_simulates_config_without_trace_records(self):
        class Args:
            graph_name = "tiny"
            root = 0
            target_depths = "1"

        space = collect_ancestor_space(lambda node: {3: [1, 2], 2: [0], 1: [0]}.get(node, []), 3, 10, 20)
        distances = {
            0: {"L_min": 0, "L_max": 0, "truncated": False},
            1: {"L_min": 1, "L_max": 1, "truncated": False},
            2: {"L_min": 1, "L_max": 1, "truncated": False},
            3: {"L_min": 2, "L_max": 2, "truncated": False},
        }

        summary = simulate_config(Args(), {0: 1, 1: 1}, [QueryInput(3, 1, space)], distances, CacheConfig(8, 2, 2))

        self.assertEqual(summary["record_type"], "ancestor_cache_policy_sweep_summary")
        self.assertEqual(summary["targets"], 1)
        self.assertEqual(summary["final_cache_entries"], 4)
        self.assertEqual(summary["cache_actions"]["insert"], 4)


if __name__ == "__main__":
    unittest.main()
