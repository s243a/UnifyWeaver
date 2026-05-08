#!/usr/bin/env python3
"""
Run C# query source-mode sweeps across generated benchmark runners and emit a
compact comparison table.

The generated cross-target runners already support `--csharp-query-source-modes`;
this wrapper makes the result easier to compare across workloads by extracting
the best mode, auto-vs-best ratio, output hash agreement, and per-mode medians.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmark_common import run_command, scale_sort_key, split_csv, validate_csharp_query_source_modes


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"

WORKLOAD_SCRIPTS = {
    "category-influence": BENCHMARK_DIR / "benchmark_category_influence_cross_target.py",
    "dependency-depth": BENCHMARK_DIR / "benchmark_dependency_depth_cross_target.py",
    "dependency-longest-depth": BENCHMARK_DIR / "benchmark_dependency_longest_depth_cross_target.py",
    "effective-distance": BENCHMARK_DIR / "benchmark_effective_distance.py",
    "shortest-path": BENCHMARK_DIR / "benchmark_shortest_path_cross_target.py",
    "weighted-shortest-path": BENCHMARK_DIR / "benchmark_weighted_shortest_path_cross_target.py",
}

FILE_BACKED_GRAPH_SCALES = ("dev", "300", "1k", "5k", "10k")
GENERATED_GRAPH_SCALES = ("300", "1k", "5k", "10k")
WORKLOAD_SUPPORTED_SCALES = {
    "category-influence": FILE_BACKED_GRAPH_SCALES,
    "dependency-depth": GENERATED_GRAPH_SCALES,
    "dependency-longest-depth": GENERATED_GRAPH_SCALES,
    "effective-distance": FILE_BACKED_GRAPH_SCALES,
    "shortest-path": FILE_BACKED_GRAPH_SCALES,
    "weighted-shortest-path": FILE_BACKED_GRAPH_SCALES,
}

DEFAULT_WORKLOADS = "all"
CALIBRATION_ARTIFACT = BENCHMARK_DIR / "csharp_query_graph_source_mode_calibration.tsv"


@dataclass
class SourceModeSummary:
    workload: str
    scale: str
    best_source_mode: str
    auto_vs_best: str
    output_agreement: str
    median_summary: str
    resolved_source_mode_summary: str
    source_registration_summary: str


@dataclass
class CalibrationArtifactRow:
    workload: str
    scale: str
    observed_best_source_mode: str
    current_auto_resolved_source_mode: str
    observed_auto_vs_best: str
    output_agreement: str
    median_summary: str
    resolved_source_mode_summary: str
    source_registration_summary: str


@dataclass
class CalibrationDrift:
    critical: list[str]
    timing: list[str]


@dataclass
class SourceModeStabilitySummary:
    workload: str
    scale: str
    runs: int
    output_agreement: str
    stable_best_source_mode: str
    best_source_mode_counts: str
    stable_auto_resolved_source_mode: str
    auto_resolved_source_mode_counts: str
    auto_vs_best_median: str
    median_summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        default=DEFAULT_WORKLOADS,
        help="Comma-separated workloads, or 'all' for every registered graph workload.",
    )
    parser.add_argument("--scales", default="300")
    parser.add_argument(
        "--source-modes",
        default="auto,preload,artifact-prebuilt",
        help="Comma-separated C# query source modes to sweep.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=1,
        help=(
            "Run the wrapper-level sweep this many times and report stable winners. "
            "Values above 1 print a stability summary instead of the raw per-run table."
        ),
    )
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Set UNIFYWEAVER_BENCH_TRACE=1 for the child benchmark runners.",
    )
    parser.add_argument(
        "--max-auto-vs-best-ratio",
        type=float,
        default=None,
        help=(
            "Fail when any output-matching summary reports auto_vs_best above this ratio. "
            "Omit to report only."
        ),
    )
    parser.add_argument(
        "--fail-on-output-mismatch",
        action="store_true",
        help="Fail when any swept C# query source mode produces a different output hash.",
    )
    parser.add_argument(
        "--compare-calibration",
        action="store_true",
        help=(
            "Compare the fresh sweep with the checked-in calibration artifact. "
            "Policy/output drift fails; timing drift is reported as warnings."
        ),
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=CALIBRATION_ARTIFACT,
        help="TSV calibration artifact used by --compare-calibration.",
    )
    parser.add_argument(
        "--timing-drift-ratio",
        type=float,
        default=1.50,
        help="Warn when a per-mode median or auto-vs-best ratio changes by more than this ratio.",
    )
    return parser.parse_args()


def parse_workloads(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(WORKLOAD_SCRIPTS)
    workloads = split_csv(value)
    unknown = [workload for workload in workloads if workload not in WORKLOAD_SCRIPTS]
    if unknown:
        raise SystemExit(
            "unknown workload(s): "
            + ", ".join(unknown)
            + "; expected one of "
            + ", ".join(WORKLOAD_SCRIPTS)
        )
    if not workloads:
        raise SystemExit("expected at least one workload")
    return workloads


def supported_scales_for_workload(workload: str) -> tuple[str, ...]:
    try:
        return WORKLOAD_SUPPORTED_SCALES[workload]
    except KeyError as exc:
        raise SystemExit(
            f"unknown workload {workload!r}; expected one of {', '.join(WORKLOAD_SCRIPTS)}"
        ) from exc


def filter_workload_scales(workload: str, scales: str) -> tuple[str, list[str]]:
    requested = split_csv(scales)
    supported = set(supported_scales_for_workload(workload))
    selected = [scale for scale in requested if scale in supported]
    skipped = [scale for scale in requested if scale not in supported]
    return ",".join(selected), skipped


def load_calibration_artifact(path: Path = CALIBRATION_ARTIFACT) -> list[CalibrationArtifactRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[CalibrationArtifactRow] = []
        for row in reader:
            rows.append(
                CalibrationArtifactRow(
                    workload=row["workload"],
                    scale=row["scale"],
                    observed_best_source_mode=row["observed_best_source_mode"],
                    current_auto_resolved_source_mode=row["current_auto_resolved_source_mode"],
                    observed_auto_vs_best=row["observed_auto_vs_best"],
                    output_agreement=row["output_agreement"],
                    median_summary=row["median_s_by_mode"],
                    resolved_source_mode_summary=row["resolved_source_modes_by_mode"],
                    source_registration_summary=row["source_registrations_by_mode"],
                )
            )
    return rows


def parse_runner_output(workload: str, output: str) -> list[SourceModeSummary]:
    medians: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    resolved_source_modes: dict[str, dict[str, str]] = {}
    source_registrations: dict[str, dict[str, str]] = {}
    best_modes: dict[str, str] = {}
    auto_vs_best: dict[str, str] = {}

    for line in output.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 7 and parts[0] != "scale" and parts[1].startswith("csharp-query:"):
            scale = parts[0]
            source_mode = parts[1].split(":", 1)[1]
            medians.setdefault(scale, {})[source_mode] = parts[2]
            hashes.setdefault(scale, {})[source_mode] = parts[6]
        elif len(parts) >= 3 and parts[1].startswith("csharp-query:") and parts[1].endswith("-metrics"):
            scale = parts[0]
            target = parts[1][:-len("-metrics")]
            source_mode = target.split(":", 1)[1]
            metrics = parse_metric_tokens(parts[2])
            resolved_mode = metrics.get("resolved_source_mode")
            if resolved_mode:
                resolved_source_modes.setdefault(scale, {})[source_mode] = resolved_mode
            registrations = summarize_source_registration_tokens(parts[2])
            if registrations:
                source_registrations.setdefault(scale, {})[source_mode] = registrations
        elif len(parts) >= 3 and parts[1] == "csharp_query_best_source_mode":
            best_modes[parts[0]] = parts[2]
        elif len(parts) >= 3 and parts[1] == "csharp_query_auto_vs_best_source_mode":
            auto_vs_best[parts[0]] = parts[2]

    summaries: list[SourceModeSummary] = []
    for scale in sorted(medians, key=scale_sort_key):
        mode_medians = medians[scale]
        mode_hashes = hashes.get(scale, {})
        unique_hashes = set(mode_hashes.values())
        output_agreement = "match" if len(unique_hashes) == 1 else "MISMATCH"
        median_summary = ",".join(
            f"{mode}:{mode_medians[mode]}"
            for mode in sorted(mode_medians)
        )
        registration_summary = ",".join(
            f"{mode}:{registrations}"
            for mode, registrations in sorted(source_registrations.get(scale, {}).items())
        )
        resolved_summary = ",".join(
            f"{mode}:{resolved}"
            for mode, resolved in sorted(resolved_source_modes.get(scale, {}).items())
        )
        summaries.append(
            SourceModeSummary(
                workload=workload,
                scale=scale,
                best_source_mode=best_modes.get(scale, ""),
                auto_vs_best=auto_vs_best.get(scale, ""),
                output_agreement=output_agreement,
                median_summary=median_summary,
                resolved_source_mode_summary=resolved_summary,
                source_registration_summary=registration_summary,
            )
        )
    return summaries


def parse_metric_tokens(metrics: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in metrics.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def summarize_source_registration_tokens(metrics: str) -> str:
    registrations: list[str] = []
    for key, value in parse_metric_tokens(metrics).items():
        if not key.startswith("source_registration_"):
            continue
        registrations.append(f"{key[len('source_registration_'):]}={value}")
    return "|".join(sorted(registrations))


def parse_ratio(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("x"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def parse_mode_summary(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in split_csv(value):
        if ":" not in part:
            continue
        mode, mode_value = part.split(":", 1)
        parsed[mode] = mode_value
    return parsed


def summarize_stability(sweep_runs: list[list[SourceModeSummary]]) -> list[SourceModeStabilitySummary]:
    by_key: dict[tuple[str, str], list[SourceModeSummary]] = {}
    for run in sweep_runs:
        for summary in run:
            by_key.setdefault((summary.workload, summary.scale), []).append(summary)

    stability: list[SourceModeStabilitySummary] = []
    for key in sorted(by_key, key=_calibration_key_sort_key):
        summaries = by_key[key]
        best_modes = [summary.best_source_mode for summary in summaries if summary.best_source_mode]
        auto_resolved_modes = [
            parse_mode_summary(summary.resolved_source_mode_summary).get("auto", "")
            for summary in summaries
        ]
        auto_resolved_modes = [mode for mode in auto_resolved_modes if mode]
        output_agreement = "match" if all(
            summary.output_agreement == "match" for summary in summaries
        ) else "MISMATCH"
        stability.append(
            SourceModeStabilitySummary(
                workload=key[0],
                scale=key[1],
                runs=len(summaries),
                output_agreement=output_agreement,
                stable_best_source_mode=_majority_value(best_modes, len(summaries)),
                best_source_mode_counts=_value_counts(best_modes),
                stable_auto_resolved_source_mode=_majority_value(auto_resolved_modes, len(summaries)),
                auto_resolved_source_mode_counts=_value_counts(auto_resolved_modes),
                auto_vs_best_median=_median_ratio(
                    summary.auto_vs_best for summary in summaries
                ),
                median_summary=_median_mode_summary(
                    summary.median_summary for summary in summaries
                ),
            )
        )
    return stability


def _value_counts(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ",".join(
        f"{value}:{count}"
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _majority_value(values: list[str], total_count: int) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return value if count > total_count / 2 else ""


def _median_ratio(values: Iterable[str]) -> str:
    ratios = [
        ratio
        for ratio in (parse_ratio(str(value)) for value in values)
        if ratio is not None
    ]
    if not ratios:
        return ""
    return f"{statistics.median(ratios):.2f}x"


def _median_mode_summary(values: Iterable[str]) -> str:
    by_mode: dict[str, list[float]] = {}
    for value in values:
        for mode, raw_mode_value in parse_mode_summary(str(value)).items():
            parsed = parse_ratio(raw_mode_value)
            if parsed is not None:
                by_mode.setdefault(mode, []).append(parsed)
    return ",".join(
        f"{mode}:{statistics.median(mode_values):.3f}"
        for mode, mode_values in sorted(by_mode.items())
    )


def compare_calibration(
    summaries: list[SourceModeSummary],
    baseline_rows: list[CalibrationArtifactRow],
    *,
    timing_drift_ratio: float = 1.50,
) -> CalibrationDrift:
    critical: list[str] = []
    timing: list[str] = []
    fresh_by_key = {
        (summary.workload, summary.scale): summary
        for summary in summaries
    }
    baseline_by_key = {
        (row.workload, row.scale): row
        for row in baseline_rows
    }

    for key in sorted(set(baseline_by_key) - set(fresh_by_key), key=_calibration_key_sort_key):
        critical.append(f"{key[0]}/{key[1]}: missing fresh sweep row")
    for key in sorted(set(fresh_by_key) - set(baseline_by_key), key=_calibration_key_sort_key):
        critical.append(f"{key[0]}/{key[1]}: no calibration baseline row")

    for key in sorted(set(fresh_by_key) & set(baseline_by_key), key=_calibration_key_sort_key):
        fresh = fresh_by_key[key]
        baseline = baseline_by_key[key]
        label = f"{fresh.workload}/{fresh.scale}"
        fresh_resolved = parse_mode_summary(fresh.resolved_source_mode_summary)
        if fresh.output_agreement != "match":
            critical.append(f"{label}: fresh source-mode outputs {fresh.output_agreement}")
        if baseline.output_agreement != "match":
            critical.append(f"{label}: calibration baseline records output {baseline.output_agreement}")
        if fresh_resolved.get("auto") != baseline.current_auto_resolved_source_mode:
            critical.append(
                f"{label}: auto resolved source mode changed from "
                f"{baseline.current_auto_resolved_source_mode} to {fresh_resolved.get('auto', '<missing>')}"
            )
        if fresh.source_registration_summary != baseline.source_registration_summary:
            critical.append(f"{label}: source registration shape changed")

        if fresh.best_source_mode != baseline.observed_best_source_mode:
            timing.append(
                f"{label}: best source mode changed from "
                f"{baseline.observed_best_source_mode} to {fresh.best_source_mode}"
            )
        _append_ratio_drift(
            timing,
            label,
            "auto_vs_best",
            baseline.observed_auto_vs_best,
            fresh.auto_vs_best,
            timing_drift_ratio,
        )
        baseline_medians = parse_mode_summary(baseline.median_summary)
        fresh_medians = parse_mode_summary(fresh.median_summary)
        for mode in sorted(set(baseline_medians) | set(fresh_medians)):
            _append_ratio_drift(
                timing,
                label,
                f"median[{mode}]",
                baseline_medians.get(mode, ""),
                fresh_medians.get(mode, ""),
                timing_drift_ratio,
            )

    return CalibrationDrift(critical=critical, timing=timing)


def _calibration_key_sort_key(key: tuple[str, str]) -> tuple[str, tuple[int, object]]:
    return (key[0], scale_sort_key(key[1]))


def _append_ratio_drift(
    drift: list[str],
    label: str,
    field: str,
    baseline_value: str,
    fresh_value: str,
    threshold: float,
) -> None:
    baseline = parse_ratio(baseline_value)
    fresh = parse_ratio(fresh_value)
    if baseline is None or fresh is None:
        if baseline_value != fresh_value:
            drift.append(
                f"{label}: {field} changed from "
                f"{baseline_value or '<missing>'} to {fresh_value or '<missing>'}"
            )
        return
    if baseline == 0.0 or fresh == 0.0:
        if baseline != fresh:
            drift.append(f"{label}: {field} changed from {baseline_value} to {fresh_value}")
        return
    ratio = max(baseline, fresh) / min(baseline, fresh)
    if ratio > threshold:
        drift.append(
            f"{label}: {field} changed from {baseline_value} to {fresh_value} "
            f"({ratio:.2f}x)"
        )


def calibration_failures(
    summaries: list[SourceModeSummary],
    *,
    max_auto_vs_best_ratio: float | None = None,
    fail_on_output_mismatch: bool = False,
) -> list[str]:
    failures: list[str] = []
    for summary in summaries:
        label = f"{summary.workload}/{summary.scale}"
        if fail_on_output_mismatch and summary.output_agreement != "match":
            failures.append(f"{label}: source-mode outputs {summary.output_agreement}")
        ratio = parse_ratio(summary.auto_vs_best)
        if (
            max_auto_vs_best_ratio is not None
            and ratio is not None
            and summary.output_agreement == "match"
            and ratio > max_auto_vs_best_ratio
        ):
            failures.append(
                f"{label}: auto_vs_best {summary.auto_vs_best} exceeds "
                f"{max_auto_vs_best_ratio:.2f}x"
            )
    return failures


def run_workload(
    workload: str,
    *,
    scales: str,
    source_modes: str,
    repetitions: int,
    trace: bool,
) -> list[SourceModeSummary]:
    env = dict(os.environ)
    if trace:
        env["UNIFYWEAVER_BENCH_TRACE"] = "1"
    else:
        env.pop("UNIFYWEAVER_BENCH_TRACE", None)

    result = run_command(
        [
            sys.executable,
            str(WORKLOAD_SCRIPTS[workload]),
            "--scales",
            scales,
            "--targets",
            "csharp-query",
            "--repetitions",
            str(repetitions),
            "--csharp-query-source-modes",
            source_modes,
        ],
        cwd=ROOT,
        env=env,
    )
    return parse_runner_output(workload, result.stdout)


def print_tsv(summaries: list[SourceModeSummary]) -> None:
    print("workload\tscale\tbest_source_mode\tauto_vs_best\toutput_agreement\tmedian_s_by_mode\tresolved_source_modes_by_mode\tsource_registrations_by_mode")
    for summary in summaries:
        print(
            f"{summary.workload}\t{summary.scale}\t{summary.best_source_mode}\t"
            f"{summary.auto_vs_best}\t{summary.output_agreement}\t{summary.median_summary}\t"
            f"{summary.resolved_source_mode_summary}\t"
            f"{summary.source_registration_summary}"
        )


def print_markdown(summaries: list[SourceModeSummary]) -> None:
    print("| Workload | Scale | Best source mode | Auto vs best | Outputs | Median seconds by mode | Resolved source modes by mode | Source registrations by mode |")
    print("| --- | --- | --- | ---: | --- | --- | --- | --- |")
    for summary in summaries:
        print(
            f"| {summary.workload} | {summary.scale} | {summary.best_source_mode} | "
            f"{summary.auto_vs_best} | {summary.output_agreement} | `{summary.median_summary}` | "
            f"`{summary.resolved_source_mode_summary}` | "
            f"`{summary.source_registration_summary}` |"
        )


def print_stability_tsv(summaries: list[SourceModeStabilitySummary]) -> None:
    print("workload\tscale\truns\toutput_agreement\tstable_best_source_mode\tbest_source_mode_counts\tstable_auto_resolved_source_mode\tauto_resolved_source_mode_counts\tauto_vs_best_median\tmedian_s_by_mode_median")
    for summary in summaries:
        print(
            f"{summary.workload}\t{summary.scale}\t{summary.runs}\t"
            f"{summary.output_agreement}\t{summary.stable_best_source_mode}\t"
            f"{summary.best_source_mode_counts}\t"
            f"{summary.stable_auto_resolved_source_mode}\t"
            f"{summary.auto_resolved_source_mode_counts}\t"
            f"{summary.auto_vs_best_median}\t{summary.median_summary}"
        )


def print_stability_markdown(summaries: list[SourceModeStabilitySummary]) -> None:
    print("| Workload | Scale | Runs | Outputs | Stable best source mode | Best mode counts | Stable auto resolved mode | Auto resolved counts | Auto vs best median | Median seconds by mode |")
    print("| --- | --- | ---: | --- | --- | --- | --- | --- | ---: | --- |")
    for summary in summaries:
        print(
            f"| {summary.workload} | {summary.scale} | {summary.runs} | "
            f"{summary.output_agreement} | {summary.stable_best_source_mode} | "
            f"`{summary.best_source_mode_counts}` | {summary.stable_auto_resolved_source_mode} | "
            f"`{summary.auto_resolved_source_mode_counts}` | {summary.auto_vs_best_median} | "
            f"`{summary.median_summary}` |"
        )


def main() -> int:
    args = parse_args()
    workloads = parse_workloads(args.workloads)
    if args.stability_runs < 1:
        raise SystemExit("--stability-runs must be at least 1")
    try:
        validate_csharp_query_source_modes(args.source_modes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sweep_runs: list[list[SourceModeSummary]] = []
    for _ in range(args.stability_runs):
        run_summaries: list[SourceModeSummary] = []
        for workload in workloads:
            selected_scales, skipped_scales = filter_workload_scales(workload, args.scales)
            for scale in skipped_scales:
                supported = ", ".join(supported_scales_for_workload(workload))
                print(
                    (
                        f"WARNING: skipping unsupported scale {scale!r} "
                        f"for {workload}; supported scales: {supported}"
                    ),
                    file=sys.stderr,
                )
            if not selected_scales:
                continue
            run_summaries.extend(
                run_workload(
                    workload,
                    scales=selected_scales,
                    source_modes=args.source_modes,
                    repetitions=args.repetitions,
                    trace=args.trace,
                )
            )
        sweep_runs.append(run_summaries)
    summaries = [
        summary
        for run_summaries in sweep_runs
        for summary in run_summaries
    ]
    if not summaries:
        raise SystemExit("no supported workload/scale combinations selected")

    if args.stability_runs > 1:
        stability_summaries = summarize_stability(sweep_runs)
        if args.format == "markdown":
            print_stability_markdown(stability_summaries)
        else:
            print_stability_tsv(stability_summaries)
    else:
        if args.format == "markdown":
            print_markdown(summaries)
        else:
            print_tsv(summaries)

    failures = calibration_failures(
        summaries,
        max_auto_vs_best_ratio=args.max_auto_vs_best_ratio,
        fail_on_output_mismatch=args.fail_on_output_mismatch,
    )
    if args.compare_calibration:
        baseline_rows = load_calibration_artifact(args.calibration_artifact)
        for run_index, run_summaries in enumerate(sweep_runs, start=1):
            fresh_keys = {
                (summary.workload, summary.scale)
                for summary in run_summaries
            }
            selected_baseline_rows = [
                row
                for row in baseline_rows
                if (row.workload, row.scale) in fresh_keys
            ]
            drift = compare_calibration(
                run_summaries,
                selected_baseline_rows,
                timing_drift_ratio=args.timing_drift_ratio,
            )
            prefix = f"run {run_index}: " if args.stability_runs > 1 else ""
            for warning in drift.timing:
                print(f"WARNING: calibration timing drift: {prefix}{warning}", file=sys.stderr)
            for failure in drift.critical:
                failures.append(f"calibration drift: {prefix}{failure}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
