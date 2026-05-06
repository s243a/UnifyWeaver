#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from generate_pipeline import (  # noqa: E402
    generate_csharp_query_category_influence,
    generate_csharp_query_dependency_depth,
    generate_csharp_query_dependency_longest_depth,
    generate_csharp_query_effective_distance,
    generate_csharp_query_shortest_path,
    generate_csharp_query_weighted_shortest_path,
)


class CSharpQueryGraphSourceModePolicyTests(unittest.TestCase):
    def test_graph_benchmark_generators_use_source_mode_policy_helper(self) -> None:
        article_cats = [("Article", "Physics")]
        category_parents = [("Physics", "Science")]
        root_cats = ["Science"]
        cases = [
            (
                "dependency-depth",
                generate_csharp_query_dependency_depth(article_cats, category_parents, root_cats),
            ),
            (
                "dependency-longest-depth",
                generate_csharp_query_dependency_longest_depth(article_cats, category_parents, root_cats),
            ),
            (
                "category-influence",
                generate_csharp_query_category_influence(article_cats, category_parents, root_cats),
            ),
            (
                "effective-distance",
                generate_csharp_query_effective_distance(article_cats, category_parents, root_cats),
            ),
            (
                "shortest-path",
                generate_csharp_query_shortest_path(article_cats, category_parents, root_cats),
            ),
            (
                "weighted-shortest-path",
                generate_csharp_query_weighted_shortest_path(article_cats, category_parents, root_cats),
            ),
        ]

        for workload, source in cases:
            with self.subTest(workload=workload):
                self.assertIn(
                    f'RelationSourceModePolicy.ResolveGraphBenchmarkMode(configuredSourceMode, "{workload}")',
                    source,
                )
                self.assertNotIn(
                    "configuredSourceMode == RelationSourceMode.Auto ? RelationSourceMode.Preload : configuredSourceMode",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
