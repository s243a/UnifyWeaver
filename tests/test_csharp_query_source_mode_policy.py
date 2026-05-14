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

SCAN_CALIBRATION_ARTIFACT = ROOT / "examples" / "benchmark" / "csharp_query_scan_source_mode_calibration.tsv"
SCAN_WORKLOAD_PREFIX = "scan-materialization:"
SMALL_PREBUILT_ARTIFACT_ROW_THRESHOLD = 7_500


class CSharpQuerySourceModePolicyTests(unittest.TestCase):
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
            "private const long SmallPrebuiltArtifactRowThreshold = 7_500L",
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

    def test_runtime_scan_source_mode_policy_matches_scan_calibration_artifact(self) -> None:
        rows = load_calibration_artifact(SCAN_CALIBRATION_ARTIFACT)

        for row in rows:
            with self.subTest(workload=row.workload, scale=row.scale):
                self.assertTrue(row.workload.startswith(SCAN_WORKLOAD_PREFIX))
                scan_mode = row.workload[len(SCAN_WORKLOAD_PREFIX):]
                expected_mode = self._expected_scan_source_mode(scan_mode, row.scale)

                self.assertEqual(row.current_auto_resolved_source_mode, expected_mode)
                self.assertIn(
                    f"auto:{expected_mode}",
                    row.resolved_source_mode_summary,
                )

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

    @staticmethod
    def _expected_scan_source_mode(scan_mode: str, scale: str) -> str:
        total_rows = (
            CSharpQuerySourceModePolicyTests._data_row_count(
                ROOT / "data" / "benchmark" / scale / "article_category.tsv"
            )
            + CSharpQuerySourceModePolicyTests._data_row_count(
                ROOT / "data" / "benchmark" / scale / "category_parent.tsv"
            )
        )
        if scan_mode == "bound_scan":
            return "artifact"
        if scan_mode in {"selective_join", "join"}:
            if total_rows <= SMALL_PREBUILT_ARTIFACT_ROW_THRESHOLD:
                return "artifact-prebuilt"
            return "artifact"
        if scan_mode == "nary_join":
            return "artifact-prebuilt"
        return "preload"

    @staticmethod
    def _data_row_count(path: Path) -> int:
        with path.open(encoding="utf-8") as handle:
            next(handle, None)
            return sum(1 for _ in handle)


if __name__ == "__main__":
    unittest.main()
