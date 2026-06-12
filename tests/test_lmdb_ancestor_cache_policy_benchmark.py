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


if __name__ == "__main__":
    unittest.main()
