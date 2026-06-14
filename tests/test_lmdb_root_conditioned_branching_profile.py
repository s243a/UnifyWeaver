#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for root-conditioned branching profile helpers."""

import unittest

from scripts.lmdb_root_conditioned_branching_profile import (
    collect_root_conditioned_nodes,
    parent_degree_record,
    profile_rows,
)


class FakeGraph:
    def __init__(self, children, parents):
        self._children = children
        self._parents = parents

    def children(self, node):
        return self._children.get(node, [])

    def parents(self, node):
        return self._parents.get(node, [])


class RootConditionedBranchingProfileTests(unittest.TestCase):
    def test_collect_root_conditioned_nodes_tracks_min_depth(self):
        graph = FakeGraph(
            children={"R": ["A", "B"], "A": ["C"], "B": ["C"]},
            parents={},
        )

        result = collect_root_conditioned_nodes(graph.children, "R")

        self.assertEqual(result["depth_by_node"], {"R": 0, "A": 1, "B": 1, "C": 2})
        self.assertEqual(result["child_edges_examined"], 4)
        self.assertFalse(result["truncated_by_nodes"])

    def test_parent_degree_record_counts_only_retained_parents(self):
        row = parent_degree_record("C", 2, ["A", "External"], {"R", "A", "C"})

        self.assertEqual(row["full_parent_degree"], 2)
        self.assertEqual(row["root_conditioned_parent_degree"], 1)
        self.assertEqual(row["outside_root_parent_degree"], 1)

    def test_profile_rows_reports_raw_and_conditioned_branching(self):
        graph = FakeGraph(
            children={"R": ["A", "B"], "A": ["C"], "B": ["C"]},
            parents={
                "R": [],
                "A": ["R"],
                "B": ["R"],
                "C": ["A", "B", "External"],
            },
        )

        rows = profile_rows(graph, "R", "fixture")
        overall = next(row for row in rows if row["record_type"] == "root_conditioned_branching_overall")
        c_row = next(row for row in rows if row.get("node") == "C")

        self.assertEqual(c_row["full_parent_degree"], 3)
        self.assertEqual(c_row["root_conditioned_parent_degree"], 2)
        self.assertGreater(
            overall["full_parent_degree"]["size_biased_parent_branching"],
            overall["root_conditioned_parent_degree"]["size_biased_parent_branching"],
        )


if __name__ == "__main__":
    unittest.main()
