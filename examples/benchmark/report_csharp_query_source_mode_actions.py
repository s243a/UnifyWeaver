#!/usr/bin/env python3
"""
Report actionable C# query source-mode calibration rows.

The calibration TSVs keep every measured row. This report filters them down to
rows where the runtime's current auto policy differs from the observed best
source mode, or where auto is slower than the best mode by a configured ratio.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from benchmark_common import scale_sort_key
from benchmark_csharp_query_source_mode_sweep import (
    CALIBRATION_ARTIFACT,
    CalibrationArtifactRow,
    load_calibration_artifact,
    parse_ratio,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
SCAN_CALIBRATION_ARTIFACT = BENCHMARK_DIR / "csharp_query_scan_source_mode_calibration.tsv"


@dataclass(frozen=True)
class ActionableCalibrationRow:
    artifact: str
    workload: str
    scale: str
    reason: str
    current_auto_resolved_source_mode: str
    observed_best_source_mode: str
    observed_auto_vs_best: str
    median_summary: str
    source_registration_summary: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-artifact",
        type=Path,
        default=CALIBRATION_ARTIFACT,
        help="Graph source-mode calibration TSV.",
    )
    parser.add_argument(
        "--scan-artifact",
        type=Path,
        default=SCAN_CALIBRATION_ARTIFACT,
        help="Scan-materialization source-mode calibration TSV.",
    )
    parser.add_argument(
        "--slow-ratio",
        type=float,
        default=1.10,
        help="Report rows whose observed auto-vs-best ratio is at or above this value.",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "markdown"),
        default="tsv",
        help="Output format.",
    )
    return parser.parse_args(argv)


def actionable_rows_from_artifacts(
    artifacts: list[tuple[str, Path]],
    *,
    slow_ratio: float = 1.10,
) -> list[ActionableCalibrationRow]:
    rows: list[ActionableCalibrationRow] = []
    for artifact_name, path in artifacts:
        rows.extend(
            actionable_rows(
                artifact_name,
                load_calibration_artifact(path),
                slow_ratio=slow_ratio,
            )
        )
    return sorted(rows, key=_action_sort_key)


def actionable_rows(
    artifact_name: str,
    calibration_rows: list[CalibrationArtifactRow],
    *,
    slow_ratio: float = 1.10,
) -> list[ActionableCalibrationRow]:
    actions: list[ActionableCalibrationRow] = []
    for row in calibration_rows:
        reasons = actionable_reasons(row, slow_ratio=slow_ratio)
        if not reasons:
            continue
        actions.append(
            ActionableCalibrationRow(
                artifact=artifact_name,
                workload=row.workload,
                scale=row.scale,
                reason=",".join(reasons),
                current_auto_resolved_source_mode=row.current_auto_resolved_source_mode,
                observed_best_source_mode=row.observed_best_source_mode,
                observed_auto_vs_best=row.observed_auto_vs_best,
                median_summary=row.median_summary,
                source_registration_summary=row.source_registration_summary,
            )
        )
    return actions


def actionable_reasons(row: CalibrationArtifactRow, *, slow_ratio: float) -> list[str]:
    reasons: list[str] = []
    if row.current_auto_resolved_source_mode != row.observed_best_source_mode:
        reasons.append("policy-diff")

    ratio = parse_ratio(row.observed_auto_vs_best)
    if ratio is not None and ratio >= slow_ratio:
        reasons.append("slow-auto")
    return reasons


def render_tsv(rows: list[ActionableCalibrationRow]) -> str:
    lines = [
        (
            "artifact\tworkload\tscale\treason\tcurrent_auto_resolved_source_mode\t"
            "observed_best_source_mode\tobserved_auto_vs_best\tmedian_s_by_mode\t"
            "source_registrations_by_mode"
        )
    ]
    for row in rows:
        lines.append(
            (
                f"{row.artifact}\t{row.workload}\t{row.scale}\t{row.reason}\t"
                f"{row.current_auto_resolved_source_mode}\t{row.observed_best_source_mode}\t"
                f"{row.observed_auto_vs_best}\t{row.median_summary}\t"
                f"{row.source_registration_summary}"
            )
        )
    return "\n".join(lines) + "\n"


def render_markdown(rows: list[ActionableCalibrationRow]) -> str:
    lines = [
        "| Artifact | Workload | Scale | Reason | Auto | Best | Auto vs Best |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            (
                f"| {row.artifact} | {row.workload} | {row.scale} | {row.reason} | "
                f"{row.current_auto_resolved_source_mode} | {row.observed_best_source_mode} | "
                f"{row.observed_auto_vs_best} |"
            )
        )
    return "\n".join(lines) + "\n"


def _action_sort_key(row: ActionableCalibrationRow) -> tuple[str, str, tuple[int, str]]:
    return (row.artifact, row.workload, scale_sort_key(row.scale))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = actionable_rows_from_artifacts(
        [
            ("graph", args.graph_artifact),
            ("scan", args.scan_artifact),
        ],
        slow_ratio=args.slow_ratio,
    )
    if args.format == "markdown":
        print(render_markdown(rows), end="")
    else:
        print(render_tsv(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
