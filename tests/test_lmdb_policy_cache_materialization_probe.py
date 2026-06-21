#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for policy-cache materialization classification helpers."""

import unittest

from scripts.lmdb_ancestor_cache_policy_benchmark import CacheEntry
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram
from scripts.lmdb_policy_cache_materialization_probe import (
    binomial_support_prior,
    classify_materialization,
    markdown_summary,
    probe_entry,
)


class PolicyCacheMaterializationProbeTests(unittest.TestCase):
    def test_classifies_exact_histogram(self):
        entry = CacheEntry(node=1, l_min=1, l_max=1)
        hist, stats = bounded_parent_histogram(lambda node: {1: [0]}.get(node, []), 1, 0, 2)

        classification, fallback = classify_materialization(entry, hist, stats, 2)

        self.assertEqual(classification, "exact_histogram")
        self.assertIsNone(fallback)

    def test_classifies_budget_too_short(self):
        entry = CacheEntry(node=2, l_min=3, l_max=5)
        hist, stats = bounded_parent_histogram(lambda node: {}, 2, 0, 2)

        classification, fallback = classify_materialization(entry, hist, stats, 2)

        self.assertEqual(classification, "budget_too_short")
        self.assertIsNone(fallback)

    def test_capped_histogram_becomes_binomial_candidate(self):
        entry = CacheEntry(node=2, l_min=1, l_max=3)
        hist, stats = bounded_parent_histogram(lambda node: {2: [1], 1: [0]}.get(node, []), 2, 0, 3, expansion_cap=1)

        classification, fallback = classify_materialization(entry, hist, stats, 3)

        self.assertEqual(classification, "closed_form_candidate")
        self.assertEqual(fallback["family"], "binomial_support_prior")
        self.assertEqual(fallback["support_min"], 1)
        self.assertEqual(fallback["support_max"], 3)

    def test_binomial_prior_uses_partial_histogram_when_present(self):
        entry = CacheEntry(node=2, l_min=1, l_max=3)
        fallback = binomial_support_prior(entry, {1: 1, 3: 1}, 3)

        self.assertEqual(fallback["source"], "partial_histogram")
        self.assertEqual(fallback["probability"], 0.5)
        self.assertTrue(fallback["distribution_normalized"])
        self.assertEqual(fallback["normalization_count"], 2)
        self.assertEqual(fallback["normalization_count_source"], "partial_histogram_lower_bound")

    def test_probe_entry_reports_classification(self):
        entry = CacheEntry(node=1, l_min=1, l_max=1)
        row = probe_entry(lambda node: {1: [0]}.get(node, []), 0, 7, entry, 2, None, None)

        self.assertEqual(row["classification"], "exact_histogram")
        self.assertEqual(row["slot"], 7)

    def test_markdown_summary_contains_closed_form_column(self):
        summary = {
            "cache_slots": 8,
            "admit_l_min": 2,
            "admit_l_max": 4,
            "budget_rows": [{
                "boundary_budget": 2,
                "resident_entries": 3,
                "classification_counts": {"exact_histogram": 1, "closed_form_candidate": 2},
                "expansion_cap_entries": 2,
                "mean_nodes_expanded": 10.0,
                "mean_fallback_support_width": 2.0,
            }],
        }

        self.assertIn("closed_form", markdown_summary([summary]))


if __name__ == "__main__":
    unittest.main()
