#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for payload recurrence layer benchmark helpers."""

import unittest

from scripts.lmdb_payload_recurrence_layer_benchmark import (
    PayloadDecodeCache,
    build_parent_payloads,
    markdown_summary,
    parse_decode_cache_modes,
    payload_layer_records,
    summarize,
)


class DictGraph:
    def __init__(self, parents):
        self._parents = parents

    def parents(self, node):
        return self._parents.get(node, [])


class PayloadRecurrenceLayerBenchmarkTests(unittest.TestCase):
    def test_parent_payloads_serialize_recurrence_histograms(self):
        graph = DictGraph({"A": ["R"], "B": ["R"]})

        payloads, rows = build_parent_payloads(
            graph.parents,
            "R",
            ["R", "A", "B"],
            parent_budget=1,
            path_cap=None,
            expansion_cap=None,
        )

        self.assertEqual(set(payloads), {"R", "A", "B"})
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["payload_bytes"] > 0 for row in rows))
        self.assertEqual(next(row for row in rows if row["node"] == "A")["histogram"], {1: 1})

    def test_payload_layer_matches_recurrence_when_parent_payloads_available(self):
        graph = DictGraph({"A": ["R"], "B": ["R"], "C": ["A", "B"]})
        payloads, _rows = build_parent_payloads(
            graph.parents,
            "R",
            ["R", "A", "B"],
            parent_budget=1,
            path_cap=None,
            expansion_cap=None,
        )

        rows = payload_layer_records(
            graph.parents,
            "R",
            ["C"],
            {"C": 2},
            payloads,
            budgets=[2],
            path_cap=None,
            expansion_cap=None,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recurrence_histogram"], {2: 2})
        self.assertEqual(rows[0]["payload_histogram"], {2: 2})
        self.assertTrue(rows[0]["exact_match"])
        self.assertEqual(rows[0]["parent_payload_nodes"], ["A", "B"])
        self.assertEqual(rows[0]["missing_parent_payloads"], 0)
        self.assertGreater(rows[0]["payload_bytes_read"], 0)

    def test_payload_layer_memo_cache_reuses_parent_decodes_across_children(self):
        graph = DictGraph({"A": ["R"], "B": ["R"], "C": ["A", "B"], "D": ["A", "B"]})
        payloads, _rows = build_parent_payloads(
            graph.parents,
            "R",
            ["R", "A", "B"],
            parent_budget=1,
            path_cap=None,
            expansion_cap=None,
        )

        rows = payload_layer_records(
            graph.parents,
            "R",
            ["C", "D"],
            {"C": 2, "D": 2},
            payloads,
            budgets=[2],
            path_cap=None,
            expansion_cap=None,
            decode_cache_mode="memo",
        )

        self.assertEqual(sum(row["payload_references"] for row in rows), 4)
        self.assertEqual(sum(row["payloads_decoded"] for row in rows), 2)
        self.assertEqual(sum(row["decode_cache_hits"] for row in rows), 2)

    def test_payload_layer_reports_missing_parent_payloads(self):
        graph = DictGraph({"A": ["R"], "B": ["R"], "C": ["A", "B"]})
        payloads, _rows = build_parent_payloads(
            graph.parents,
            "R",
            ["A"],
            parent_budget=1,
            path_cap=None,
            expansion_cap=None,
        )

        rows = payload_layer_records(
            graph.parents,
            "R",
            ["C"],
            {"C": 2},
            payloads,
            budgets=[2],
            path_cap=None,
            expansion_cap=None,
        )

        self.assertEqual(rows[0]["missing_parent_payloads"], 1)
        self.assertFalse(rows[0]["exact_match"])
        self.assertLess(rows[0]["path_count_delta"], 0)

    def test_summary_renders_payload_metrics(self):
        records = [
            {
                "record_type": "payload_recurrence_layer_selection",
                "graph": "tiny",
                "root": "R",
                "parent_depths": [0, 1],
                "child_depths": [2],
            },
            {
                "record_type": "payload_recurrence_parent_payload",
                "payload_bytes": 32,
                "support_bins": 1,
            },
            {
                "record_type": "payload_recurrence_layer_comparison",
                "budget": 2,
                "exact_match": True,
                "missing_parent_payloads": 0,
                "parent_payloads_available": 2,
                "parent_payload_nodes": ["A", "B"],
                "l1_error": 0.0,
                "max_cdf_error": 0.0,
                "w1_cdf_error": 0.0,
                "decode_cache_mode": "memo",
                "payload_references": 2,
                "payloads_decoded": 1,
                "decode_cache_hits": 1,
                "decode_cache_misses": 1,
                "decode_cache_hit_rate": 0.5,
                "payload_bytes_read": 64,
                "payload_decode_ns": 10,
                "payload_decoded_bins": 2,
                "payload_output_bins": 1,
                "payload_output_bytes": 32,
                "recurrence_time_ns": 100,
                "payload_time_ns": 50,
                "time_ratio": 0.5,
                "recurrence_path_cap_hit": False,
                "recurrence_expansion_cap_hit": False,
                "payload_path_cap_hit": False,
            },
        ]

        summary = summarize(records)
        rendered = markdown_summary(summary)

        self.assertEqual(summary["budget_rows"][0]["exact_match_rows"], 1)
        self.assertEqual(summary["budget_rows"][0]["unique_parent_payloads_referenced"], 2)
        self.assertEqual(summary["budget_rows"][0]["total_parent_payload_references"], 2)
        self.assertEqual(summary["budget_rows"][0]["max_children_per_parent_payload"], 1)
        self.assertEqual(summary["budget_rows"][0]["mean_payload_bytes_read"], 64)
        self.assertIn("mean_payload_read", rendered)
        self.assertIn("unique_refs", rendered)
        self.assertIn("max_share", rendered)
        self.assertIn("decode_cache", rendered)

    def test_parse_decode_cache_modes_deduplicates_modes(self):
        self.assertEqual(parse_decode_cache_modes("none,memo,memo"), ["none", "memo"])
        self.assertEqual(PayloadDecodeCache("memo").mode, "memo")


if __name__ == "__main__":
    unittest.main()
