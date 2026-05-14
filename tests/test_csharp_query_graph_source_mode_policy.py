#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

import benchmark_scan_materialization  # noqa: E402
from generate_pipeline import (  # noqa: E402
    generate_csharp_query_category_influence,
    generate_csharp_query_dependency_depth,
    generate_csharp_query_dependency_longest_depth,
    generate_csharp_query_effective_distance,
    generate_csharp_query_shortest_path,
    generate_csharp_query_weighted_shortest_path,
)
from benchmark_csharp_query_source_mode_sweep import load_calibration_artifact  # noqa: E402


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
                self.assertIn(
                    'Console.Error.WriteLine($"source_mode={RelationSourceModePolicy.ToConfigValue(configuredSourceMode)}");',
                    source,
                )
                self.assertIn(
                    'Console.Error.WriteLine($"resolved_source_mode={RelationSourceModePolicy.ToConfigValue(sourceMode)}");',
                    source,
                )
                self.assertNotIn(
                    "configuredSourceMode == RelationSourceMode.Auto ? RelationSourceMode.Preload : configuredSourceMode",
                    source,
                )

    def test_scan_materialization_uses_source_mode_policy_helper(self) -> None:
        source = benchmark_scan_materialization.PROGRAM

        self.assertIn(
            "RelationSourceModePolicy.ResolveScanBenchmarkMode(configuredSourceMode, mode, articleRowCount, edgeRowCount)",
            source,
        )
        self.assertIn(
            'Console.Error.WriteLine($"source_mode={RelationSourceModePolicy.ToConfigValue(configuredSourceMode)}");',
            source,
        )
        self.assertIn(
            'Console.Error.WriteLine($"resolved_source_mode={RelationSourceModePolicy.ToConfigValue(sourceMode)}");',
            source,
        )
        self.assertNotIn(
            "configuredSourceMode == RelationSourceMode.Auto ? RelationSourceMode.Preload : configuredSourceMode",
            source,
        )

    def test_scan_materialization_output_hash_ignores_row_order(self) -> None:
        left = "value\nb\na\n"
        right = "value\na\nb\n"

        self.assertEqual(
            benchmark_scan_materialization.normalized_output_hash(left),
            benchmark_scan_materialization.normalized_output_hash(right),
        )

    def test_runtime_scan_source_mode_policy_matches_documented_cases(self) -> None:
        runtime_source = (
            ROOT / "src" / "unifyweaver" / "targets" / "csharp_query_runtime" / "QueryRuntime.cs"
        ).read_text()

        expected_cases = [
            '"bound_scan" => RelationSourceMode.Artifact',
            (
                '"selective_join" when totalRows <= SmallPrebuiltArtifactRowThreshold '
                "=> RelationSourceMode.ArtifactPrebuilt"
            ),
            '"selective_join" => RelationSourceMode.Artifact',
            (
                '"join" when totalRows <= SmallPrebuiltArtifactRowThreshold '
                "=> RelationSourceMode.ArtifactPrebuilt"
            ),
            '"join" => RelationSourceMode.Artifact',
            '"nary_join" => RelationSourceMode.ArtifactPrebuilt',
            "_ => RelationSourceMode.Preload",
        ]

        for expected_case in expected_cases:
            with self.subTest(expected_case=expected_case):
                self.assertIn(expected_case, runtime_source)

    def test_runtime_source_mode_policy_matches_calibration_artifact(self) -> None:
        runtime_source = (
            ROOT / "src" / "unifyweaver" / "targets" / "csharp_query_runtime" / "QueryRuntime.cs"
        ).read_text()

        for row in load_calibration_artifact():
            with self.subTest(workload=row.workload):
                self.assertIn(
                    f'"{row.workload}" => RelationSourceMode.{self._source_mode_member(row.current_auto_resolved_source_mode)}',
                    runtime_source,
                )

    @staticmethod
    def _source_mode_member(source_mode: str) -> str:
        return {
            "preload": "Preload",
            "delimited": "Delimited",
            "artifact": "Artifact",
            "artifact-prebuilt": "ArtifactPrebuilt",
        }[source_mode]


if __name__ == "__main__":
    unittest.main()
