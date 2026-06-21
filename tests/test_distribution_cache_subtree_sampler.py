#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for distribution-cache subtree sampling."""

import unittest
from pathlib import Path

from scripts.sample_distribution_cache_subtree import (
    DEFAULT_EXCLUDE_PATTERNS,
    build_children_index,
    compile_patterns,
    sample_edges,
    select_subtree_nodes,
)
from tools.distribution_cache_support import load_parent_edges_tsv


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLEWIKI_SAMPLE = REPO_ROOT / "tests" / "fixtures" / "simplewiki_articles_parent_sample.tsv"
SIMPLEWIKI_ROOT = "Category:Articles"


class DistributionCacheSubtreeSamplerTests(unittest.TestCase):
    def test_selects_shallow_articles_subtree(self):
        parents = load_parent_edges_tsv(SIMPLEWIKI_SAMPLE)
        children = build_children_index(parents, [], SIMPLEWIKI_ROOT)
        selected, distances = select_subtree_nodes(children, SIMPLEWIKI_ROOT, max_depth=2)
        rows = sample_edges(parents, selected)

        self.assertEqual(distances[SIMPLEWIKI_ROOT], 0)
        self.assertEqual(distances["Category:Science"], 2)
        self.assertIn(("Category:Science", "Category:Main_topic_classifications"), rows)
        self.assertNotIn("Category:Physics", selected)

    def test_default_filters_exclude_registry_and_admin_regions(self):
        parents = {
            "Category:Container_categories": [SIMPLEWIKI_ROOT],
            "Category:Topic_container": ["Category:Container_categories"],
            "Category:Wikipedia_maintenance": [SIMPLEWIKI_ROOT],
            "Category:Science": [SIMPLEWIKI_ROOT],
        }
        patterns = compile_patterns(DEFAULT_EXCLUDE_PATTERNS)
        children = build_children_index(parents, patterns, SIMPLEWIKI_ROOT)
        selected, _distances = select_subtree_nodes(children, SIMPLEWIKI_ROOT, max_depth=2)

        self.assertIn("Category:Science", selected)
        self.assertNotIn("Category:Container_categories", selected)
        self.assertNotIn("Category:Topic_container", selected)
        self.assertNotIn("Category:Wikipedia_maintenance", selected)


if __name__ == "__main__":
    unittest.main()
