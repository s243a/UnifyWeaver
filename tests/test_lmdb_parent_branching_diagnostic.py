#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for LMDB parent-branching diagnostic helpers."""

import unittest

from scripts.lmdb_parent_branching_diagnostic import (
    bucket_records,
    deterministic_sample,
    root_distances,
    size_biased_branching,
)


class LmdbParentBranchingDiagnosticTests(unittest.TestCase):
    def test_size_biased_branching_reports_excess(self):
        stats = size_biased_branching([1, 3, 5])

        self.assertAlmostEqual(stats["mean_parent_degree"], 3.0)
        self.assertAlmostEqual(stats["second_parent_degree_moment"], 35.0 / 3.0)
        self.assertAlmostEqual(stats["size_biased_parent_branching"], 35.0 / 9.0)
        self.assertAlmostEqual(stats["mean_excess"], 35.0 / 9.0 - 1.0)

    def test_root_distances_tracks_maximum_parent_distance(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["R", "B"],
        }

        result = root_distances("C", "R", lambda node: parents.get(node, []), 8)

        self.assertEqual(result["L_min"], 1)
        self.assertEqual(result["L_max"], 3)
        self.assertFalse(result["truncated"])

    def test_bucket_records_uses_l_max_bucket(self):
        records = [
            {
                "record_type": "lmdb_parent_branching_target",
                "L_max": 2,
                "full_parent_degree": 1,
                "root_reaching_parent_degree": 1,
                "distance_truncated": False,
                "cycle_skipped": False,
            },
            {
                "record_type": "lmdb_parent_branching_target",
                "L_max": 2,
                "full_parent_degree": 4,
                "root_reaching_parent_degree": 3,
                "distance_truncated": False,
                "cycle_skipped": False,
            },
        ]

        buckets = bucket_records(records)

        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0]["L_max_bucket"], 2)
        self.assertEqual(buckets[0]["full_parent_degree"]["max_parent_degree"], 4)
        self.assertEqual(buckets[0]["root_reaching_parent_degree"]["max_parent_degree"], 3)

    def test_deterministic_sample_is_stable(self):
        values = list(range(100))

        self.assertEqual(
            deterministic_sample(values, 10, "seed"),
            deterministic_sample(values, 10, "seed"),
        )


if __name__ == "__main__":
    unittest.main()
