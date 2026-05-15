#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_csharp_query_source_mode_sweep import (  # noqa: E402
    CalibrationArtifactRow,
    write_calibration_artifact,
)
from report_csharp_query_source_mode_actions import (  # noqa: E402
    actionable_reasons,
    actionable_rows,
    actionable_rows_from_artifacts,
    main,
    render_markdown,
    render_tsv,
)


class CSharpQuerySourceModeActionReportTests(unittest.TestCase):
    def test_actionable_reasons_flags_policy_diff_and_slow_auto(self) -> None:
        row = self._row(
            current_auto_resolved_source_mode="preload",
            observed_best_source_mode="artifact-prebuilt",
            observed_auto_vs_best="1.25x",
        )

        self.assertEqual(
            actionable_reasons(row, slow_ratio=1.10),
            ["policy-diff", "slow-auto"],
        )

    def test_actionable_rows_ignores_matching_fast_rows(self) -> None:
        rows = actionable_rows(
            "graph",
            [
                self._row(
                    current_auto_resolved_source_mode="preload",
                    observed_best_source_mode="preload",
                    observed_auto_vs_best="1.01x",
                )
            ],
        )

        self.assertEqual(rows, [])

    def test_actionable_rows_from_artifacts_sorts_by_artifact_workload_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "graph.tsv"
            scan_path = Path(tmp_dir) / "scan.tsv"
            write_calibration_artifact(
                graph_path,
                [
                    self._row(workload="weighted-shortest-path", scale="10k"),
                    self._row(workload="weighted-shortest-path", scale="1k"),
                ],
            )
            write_calibration_artifact(scan_path, [self._row(workload="scan-materialization:join")])

            rows = actionable_rows_from_artifacts(
                [("scan", scan_path), ("graph", graph_path)],
                slow_ratio=1.10,
            )

        self.assertEqual(
            [(row.artifact, row.workload, row.scale) for row in rows],
            [
                ("graph", "weighted-shortest-path", "1k"),
                ("graph", "weighted-shortest-path", "10k"),
                ("scan", "scan-materialization:join", "300"),
            ],
        )

    def test_render_tsv_includes_reason_and_modes(self) -> None:
        report = render_tsv(actionable_rows("graph", [self._row()]))

        self.assertIn("artifact\tworkload\tscale\treason", report)
        self.assertIn("graph\tcategory-influence\t300\tpolicy-diff", report)
        self.assertIn("preload\tauto\t1.00x", report)

    def test_render_markdown_includes_compact_table(self) -> None:
        report = render_markdown(actionable_rows("graph", [self._row()]))

        self.assertIn("| Artifact | Workload | Scale | Reason | Auto | Best | Auto vs Best |", report)
        self.assertIn("| graph | category-influence | 300 | policy-diff | preload | auto | 1.00x |", report)

    def test_main_accepts_temp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "graph.tsv"
            scan_path = Path(tmp_dir) / "scan.tsv"
            write_calibration_artifact(graph_path, [self._row()])
            write_calibration_artifact(scan_path, [])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "--graph-artifact",
                            str(graph_path),
                            "--scan-artifact",
                            str(scan_path),
                            "--format",
                            "markdown",
                        ]
                    ),
                    0,
                )
            self.assertIn("| graph | category-influence | 300 |", stdout.getvalue())

    @staticmethod
    def _row(
        *,
        workload: str = "category-influence",
        scale: str = "300",
        current_auto_resolved_source_mode: str = "preload",
        observed_best_source_mode: str = "auto",
        observed_auto_vs_best: str = "1.00x",
    ) -> CalibrationArtifactRow:
        return CalibrationArtifactRow(
            workload=workload,
            scale=scale,
            observed_best_source_mode=observed_best_source_mode,
            current_auto_resolved_source_mode=current_auto_resolved_source_mode,
            observed_auto_vs_best=observed_auto_vs_best,
            output_agreement="match",
            median_summary="artifact-prebuilt:0.904,auto:0.467,preload:0.360",
            resolved_source_mode_summary="artifact-prebuilt:artifact-prebuilt,auto:preload,preload:preload",
            source_registration_summary="auto:preloaded:preload:arity2=3",
        )


if __name__ == "__main__":
    unittest.main()
