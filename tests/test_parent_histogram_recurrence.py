#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for parent histogram recurrence helpers."""

import unittest

from scripts.distribution_serialization import decode_distribution_payload, encode_selected_distribution
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram
from scripts.parent_histogram_recurrence import (
    histogram_distribution,
    payload_to_histogram,
    recurrence_parent_histogram,
    serialize_shifted_parent_payload_histogram,
    shifted_parent_payload_histogram,
)



class ParentHistogramRecurrenceTests(unittest.TestCase):
    def test_recurrence_matches_dag_path_histogram(self):
        parents = {
            "A": ["R"],
            "B": ["R", "A"],
            "C": ["A", "B"],
        }

        dfs_hist, _dfs_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)
        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 3)

        self.assertEqual(dfs_hist, {2: 2, 3: 1})
        self.assertEqual(rec_hist, dfs_hist)
        self.assertFalse(rec_stats.cycle_approximation)

    def test_recurrence_respects_budget_horizon(self):
        parents = {
            "A": ["R"],
            "B": ["A"],
            "C": ["B"],
        }

        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 2)

        self.assertEqual(rec_hist, {})
        self.assertGreater(rec_stats.budget_cutoffs, 0)

    def test_recurrence_marks_cycles_as_approximate(self):
        parents = {
            "A": ["R"],
            "B": ["A", "C"],
            "C": ["B"],
        }

        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 4)
        dfs_hist, _dfs_stats = bounded_parent_histogram(lambda node: parents.get(node, []), "C", "R", 4)

        self.assertEqual(rec_hist, dfs_hist)
        self.assertTrue(rec_stats.cycle_approximation)
        self.assertGreater(rec_stats.cycle_edges, 0)

    def test_path_cap_marks_recurrence_truncated(self):
        parents = {
            "A": ["R"],
            "B": ["R"],
            "C": ["A", "B"],
        }

        rec_hist, rec_stats = recurrence_parent_histogram(lambda node: parents.get(node, []), "C", "R", 2, path_cap=1)

        self.assertTrue(rec_stats.path_cap_hit)
        self.assertEqual(sum(rec_hist.values()), 1)

    def test_payload_to_histogram_restores_unnormalized_mass(self):
        probabilities, origin, total_count = histogram_distribution({2: 1, 3: 2})
        payload, _meta = encode_selected_distribution(probabilities, "packed_sparse_histogram", origin=origin, total_mass=total_count)

        hist, decoded_meta, decode_ns, payload_bytes = payload_to_histogram(payload)

        self.assertEqual(hist, {2: 1, 3: 2})
        self.assertEqual(decoded_meta["total_mass"], 3)
        self.assertGreaterEqual(decode_ns, 0)
        self.assertEqual(payload_bytes, len(payload))

    def test_shifted_parent_payload_histogram_sums_parent_distributions(self):
        left_prob, left_origin, left_count = histogram_distribution({1: 1, 2: 2})
        right_prob, right_origin, right_count = histogram_distribution({1: 3})
        left_payload, _left_meta = encode_selected_distribution(left_prob, "packed_sparse_histogram", origin=left_origin, total_mass=left_count)
        right_payload, _right_meta = encode_selected_distribution(right_prob, "packed_sparse_histogram", origin=right_origin, total_mass=right_count)

        hist, stats = shifted_parent_payload_histogram([left_payload, right_payload], remaining=4)

        self.assertEqual(hist, {2: 4, 3: 2})
        self.assertEqual(stats.parent_payloads, 2)
        self.assertEqual(stats.payloads_decoded, 2)
        self.assertEqual(stats.output_path_count, 6)
        self.assertGreater(stats.payload_bytes_read, 0)

    def test_serialize_shifted_parent_payload_histogram_round_trips_child(self):
        probabilities, origin, total_count = histogram_distribution({1: 2, 3: 1})
        parent_payload, _parent_meta = encode_selected_distribution(probabilities, "packed_sparse_histogram", origin=origin, total_mass=total_count)

        child_payload, child_meta, child_hist, stats = serialize_shifted_parent_payload_histogram([parent_payload])
        child_probabilities, decoded_meta = decode_distribution_payload(child_payload)
        decoded_hist, _decoded_meta, _decode_ns, _payload_bytes = payload_to_histogram(child_payload)

        self.assertEqual(child_hist, {2: 2, 4: 1})
        self.assertEqual(decoded_hist, child_hist)
        self.assertEqual(decoded_meta["origin"], 2)
        self.assertAlmostEqual(sum(child_probabilities), 1.0)
        self.assertEqual(child_meta["payload_bytes"], stats.output_payload_bytes)


if __name__ == "__main__":
    unittest.main()
