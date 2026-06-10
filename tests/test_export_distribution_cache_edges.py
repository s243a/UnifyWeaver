#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for resolved category edge export."""

import sqlite3
import unittest

from scripts.export_distribution_cache_edges import category_node, iter_category_parent_edges


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE page (
            page_id INTEGER PRIMARY KEY,
            page_title TEXT,
            page_namespace INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE categorylinks (
            cl_from INTEGER,
            cl_to TEXT,
            cl_type TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO page (page_id, page_title, page_namespace) VALUES (?, ?, ?)",
        [
            (1, "Articles", 14),
            (2, "Science", 14),
            (3, "Physics", 14),
            (4, "Albert_Einstein", 0),
        ],
    )
    conn.executemany(
        "INSERT INTO categorylinks (cl_from, cl_to, cl_type) VALUES (?, ?, ?)",
        [
            (2, "Articles", "subcat"),
            (3, "Science", "subcat"),
            (4, "Physics", "page"),
        ],
    )
    return conn


class ExportDistributionCacheEdgesTests(unittest.TestCase):
    def test_category_node_adds_prefix_once(self):
        self.assertEqual(category_node("Science"), "Category:Science")
        self.assertEqual(category_node("Category:Science"), "Category:Science")

    def test_iter_category_parent_edges_exports_resolved_subcat_edges(self):
        conn = make_conn()
        try:
            rows = list(iter_category_parent_edges(conn))
        finally:
            conn.close()

        self.assertEqual(
            rows,
            [
                ("Category:Physics", "Category:Science"),
                ("Category:Science", "Category:Articles"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
