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
import subprocess
import sys
import tempfile
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


@dataclass
class CompetingProcess:
    pid: int
    cpu_percent: float
    command: str


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
    parser.add_argument("--format", choices=["tsv", "markdown", "calibration-tsv"], default="tsv")
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
        help="TSV calibration artifact used by --compare-calibration and --write-calibration-artifact.",
    )
    parser.add_argument(
        "--write-calibration-artifact",
        action="store_true",
        help=(
            "Write the fresh calibration TSV to --calibration-artifact. "
            "Use with --format calibration-tsv and usually --stability-runs > 1."
        ),
    )
    parser.add_argument(
        "--timing-drift-ratio",
        type=float,
        default=1.50,
        help="Warn when a per-mode median or auto-vs-best ratio changes by more than this ratio.",
    )
    parser.add_argument(
        "--require-idle",
        action="store_true",
        help=(
            "Before timing-sensitive runs, fail fast if competing high-CPU processes "
            "are active or available memory is below --min-free-memory-mib."
        ),
    )
    parser.add_argument(
        "--max-competing-cpu-percent",
        type=float,
        default=50.0,
        help="CPU-percent threshold for --require-idle competing-process detection.",
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=int,
        default=1024,
        help="Minimum MemAvailable MiB required by --require-idle.",
    )
    args = parser.parse_args()
    if args.write_calibration_artifact and args.format != "calibration-tsv":
        raise SystemExit("--write-calibration-artifact requires --format calibration-tsv")
    return args


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


def parse_mem_available_mib(meminfo_text: str) -> int | None:
    for line in meminfo_text.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1]) // 1024
        except ValueError:
            return None
    return None


def read_mem_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        return parse_mem_available_mib(meminfo_path.read_text(encoding="utf-8"))
    except OSError:
        return None


def parse_competing_processes(
    ps_output: str,
    *,
    current_pid: int,
    cpu_threshold: float,
) -> list[CompetingProcess]:
    processes: list[CompetingProcess] = []
    for line in ps_output.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            cpu_percent = float(parts[1])
        except ValueError:
            continue
        if pid == current_pid or cpu_percent < cpu_threshold:
            continue
        processes.append(
            CompetingProcess(
                pid=pid,
                cpu_percent=cpu_percent,
                command=parts[2],
            )
        )
    return processes


def collect_competing_processes(
    *,
    cpu_threshold: float,
    current_pid: int | None = None,
) -> list[CompetingProcess]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,%cpu=,args="],
            check=True,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    header = "PID %CPU COMMAND\n"
    return parse_competing_processes(
        header + result.stdout,
        current_pid=current_pid or os.getpid(),
        cpu_threshold=cpu_threshold,
    )


def resource_preflight_failures(
    *,
    min_free_memory_mib: int,
    max_competing_cpu_percent: float,
    available_memory_mib: int | None = None,
    competing_processes: list[CompetingProcess] | None = None,
) -> list[str]:
    failures: list[str] = []
    memory_mib = read_mem_available_mib() if available_memory_mib is None else available_memory_mib
    if memory_mib is None:
        failures.append("could not determine available memory from /proc/meminfo")
    elif memory_mib < min_free_memory_mib:
        failures.append(
            f"available memory {memory_mib} MiB is below required {min_free_memory_mib} MiB"
        )

    processes = (
        collect_competing_processes(cpu_threshold=max_competing_cpu_percent)
        if competing_processes is None
        else competing_processes
    )
    for process in processes:
        failures.append(
            (
                f"competing process {process.pid} uses {process.cpu_percent:.1f}% CPU "
                f"(threshold {max_competing_cpu_percent:.1f}%): {process.command}"
            )
        )
    return failures


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


def parse_runner_output(
    workload: str,
    output: str,
    *,
    default_source_mode: str = "auto",
) -> list[SourceModeSummary]:
    medians: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    resolved_source_modes: dict[str, dict[str, str]] = {}
    source_registrations: dict[str, dict[str, str]] = {}
    best_modes: dict[str, str] = {}
    auto_vs_best: dict[str, str] = {}

    for line in output.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 7 and parts[0] != "scale" and _is_csharp_query_result(parts[1]):
            scale = parts[0]
            source_mode = _source_mode_from_result_target(parts[1], default_source_mode)
            medians.setdefault(scale, {})[source_mode] = parts[2]
            hashes.setdefault(scale, {})[source_mode] = parts[6]
        elif len(parts) >= 3 and _is_csharp_query_metrics(parts[1]):
            scale = parts[0]
            target = parts[1][:-len("-metrics")]
            metrics = parse_metric_tokens(parts[2])
            source_mode = metrics.get("source_mode") or _source_mode_from_result_target(
                target,
                default_source_mode,
            )
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
        best_source_mode = best_modes.get(scale, "")
        auto_vs_best_ratio = auto_vs_best.get(scale, "")
        inferred_single_mode = False
        if not best_source_mode and len(mode_medians) == 1:
            best_source_mode = next(iter(mode_medians))
            inferred_single_mode = True
        if (
            inferred_single_mode
            and not auto_vs_best_ratio
            and "auto" in mode_medians
            and best_source_mode in mode_medians
        ):
            best_median = parse_ratio(mode_medians[best_source_mode])
            auto_median = parse_ratio(mode_medians["auto"])
            if best_median and auto_median is not None:
                auto_vs_best_ratio = f"{auto_median / best_median:.2f}x"
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
                best_source_mode=best_source_mode,
                auto_vs_best=auto_vs_best_ratio,
                output_agreement=output_agreement,
                median_summary=median_summary,
                resolved_source_mode_summary=resolved_summary,
                source_registration_summary=registration_summary,
            )
        )
    return summaries


def _is_csharp_query_result(target: str) -> bool:
    return target == "csharp-query" or target.startswith("csharp-query:")


def _is_csharp_query_metrics(target: str) -> bool:
    return target == "csharp-query-metrics" or (
        target.startswith("csharp-query:") and target.endswith("-metrics")
    )


def _source_mode_from_result_target(target: str, default_source_mode: str) -> str:
    if target.startswith("csharp-query:"):
        return target.split(":", 1)[1]
    return default_source_mode


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


def calibration_rows_from_summaries(summaries: list[SourceModeSummary]) -> list[CalibrationArtifactRow]:
    rows: list[CalibrationArtifactRow] = []
    for summary in sorted(summaries, key=lambda item: _calibration_key_sort_key((item.workload, item.scale))):
        rows.append(
            CalibrationArtifactRow(
                workload=summary.workload,
                scale=summary.scale,
                observed_best_source_mode=summary.best_source_mode,
                current_auto_resolved_source_mode=parse_mode_summary(
                    summary.resolved_source_mode_summary
                ).get("auto", ""),
                observed_auto_vs_best=summary.auto_vs_best,
                output_agreement=summary.output_agreement,
                median_summary=summary.median_summary,
                resolved_source_mode_summary=summary.resolved_source_mode_summary,
                source_registration_summary=summary.source_registration_summary,
            )
        )
    return rows


def calibration_rows_from_stability(
    stability_summaries: list[SourceModeStabilitySummary],
    sweep_runs: list[list[SourceModeSummary]],
) -> list[CalibrationArtifactRow]:
    summaries_by_key: dict[tuple[str, str], list[SourceModeSummary]] = {}
    for run in sweep_runs:
        for summary in run:
            summaries_by_key.setdefault((summary.workload, summary.scale), []).append(summary)

    rows: list[CalibrationArtifactRow] = []
    for summary in stability_summaries:
        key = (summary.workload, summary.scale)
        raw_summaries = summaries_by_key.get(key, [])
        rows.append(
            CalibrationArtifactRow(
                workload=summary.workload,
                scale=summary.scale,
                observed_best_source_mode=(
                    summary.stable_best_source_mode
                    or _first_count_value(summary.best_source_mode_counts)
                ),
                current_auto_resolved_source_mode=(
                    summary.stable_auto_resolved_source_mode
                    or _first_count_value(summary.auto_resolved_source_mode_counts)
                ),
                observed_auto_vs_best=summary.auto_vs_best_median,
                output_agreement=summary.output_agreement,
                median_summary=summary.median_summary,
                resolved_source_mode_summary=_most_common_nonempty(
                    item.resolved_source_mode_summary for item in raw_summaries
                ),
                source_registration_summary=_most_common_nonempty(
                    item.source_registration_summary for item in raw_summaries
                ),
            )
        )
    return rows


def _value_counts(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ",".join(
        f"{value}:{count}"
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _first_count_value(counts: str) -> str:
    first = split_csv(counts)
    if not first or ":" not in first[0]:
        return ""
    return first[0].split(":", 1)[0]


def _most_common_nonempty(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


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
    return compare_calibration_rows(
        calibration_rows_from_summaries(summaries),
        baseline_rows,
        timing_drift_ratio=timing_drift_ratio,
    )


def compare_calibration_rows(
    fresh_rows: list[CalibrationArtifactRow],
    baseline_rows: list[CalibrationArtifactRow],
    *,
    timing_drift_ratio: float = 1.50,
) -> CalibrationDrift:
    critical: list[str] = []
    timing: list[str] = []
    fresh_by_key = {
        (row.workload, row.scale): row
        for row in fresh_rows
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
        if fresh.output_agreement != "match":
            critical.append(f"{label}: fresh source-mode outputs {fresh.output_agreement}")
        if baseline.output_agreement != "match":
            critical.append(f"{label}: calibration baseline records output {baseline.output_agreement}")
        if fresh.current_auto_resolved_source_mode != baseline.current_auto_resolved_source_mode:
            critical.append(
                f"{label}: auto resolved source mode changed from "
                f"{baseline.current_auto_resolved_source_mode} "
                f"to {fresh.current_auto_resolved_source_mode or '<missing>'}"
            )
        if fresh.source_registration_summary != baseline.source_registration_summary:
            critical.append(f"{label}: source registration shape changed")

        if (
            fresh.observed_best_source_mode != baseline.observed_best_source_mode
            and _best_mode_change_exceeds_threshold(
                baseline.observed_auto_vs_best,
                fresh.observed_auto_vs_best,
                timing_drift_ratio,
            )
        ):
            timing.append(
                f"{label}: best source mode changed from "
                f"{baseline.observed_best_source_mode} to {fresh.observed_best_source_mode}"
            )
        _append_ratio_drift(
            timing,
            label,
            "auto_vs_best",
            baseline.observed_auto_vs_best,
            fresh.observed_auto_vs_best,
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


def _best_mode_change_exceeds_threshold(
    baseline_auto_vs_best: str,
    fresh_auto_vs_best: str,
    threshold: float,
) -> bool:
    ratios = [
        ratio
        for ratio in (
            parse_ratio(baseline_auto_vs_best),
            parse_ratio(fresh_auto_vs_best),
        )
        if ratio is not None
    ]
    return not ratios or max(ratios) >= threshold


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
    source_mode_values = split_csv(source_modes)
    default_source_mode = source_mode_values[0] if len(source_mode_values) == 1 else "auto"
    return parse_runner_output(
        workload,
        result.stdout,
        default_source_mode=default_source_mode,
    )


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


def print_calibration_tsv(rows: list[CalibrationArtifactRow]) -> None:
    print(render_calibration_tsv(rows), end="")


def render_calibration_tsv(rows: list[CalibrationArtifactRow]) -> str:
    lines = [
        (
            "workload\tscale\tobserved_best_source_mode\t"
            "current_auto_resolved_source_mode\tobserved_auto_vs_best\t"
            "output_agreement\tmedian_s_by_mode\tresolved_source_modes_by_mode\t"
            "source_registrations_by_mode"
        )
    ]
    for row in rows:
        lines.append(
            f"{row.workload}\t{row.scale}\t{row.observed_best_source_mode}\t"
            f"{row.current_auto_resolved_source_mode}\t{row.observed_auto_vs_best}\t"
            f"{row.output_agreement}\t{row.median_summary}\t"
            f"{row.resolved_source_mode_summary}\t{row.source_registration_summary}"
        )
    return "\n".join(lines) + "\n"


def merge_calibration_rows(
    baseline_rows: list[CalibrationArtifactRow],
    fresh_rows: list[CalibrationArtifactRow],
) -> list[CalibrationArtifactRow]:
    merged_by_key = {
        (row.workload, row.scale): row
        for row in baseline_rows
    }
    for row in fresh_rows:
        merged_by_key[(row.workload, row.scale)] = row
    return [
        merged_by_key[key]
        for key in sorted(merged_by_key, key=_calibration_key_sort_key)
    ]


def write_calibration_artifact(path: Path, rows: list[CalibrationArtifactRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(render_calibration_tsv(rows))
    tmp_path.replace(path)


def write_merged_calibration_artifact(
    path: Path,
    fresh_rows: list[CalibrationArtifactRow],
) -> list[CalibrationArtifactRow]:
    baseline_rows = load_calibration_artifact(path) if path.exists() else []
    merged_rows = merge_calibration_rows(baseline_rows, fresh_rows)
    write_calibration_artifact(path, merged_rows)
    return merged_rows


def main() -> int:
    args = parse_args()
    workloads = parse_workloads(args.workloads)
    if args.stability_runs < 1:
        raise SystemExit("--stability-runs must be at least 1")
    try:
        validate_csharp_query_source_modes(args.source_modes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.require_idle:
        preflight_failures = resource_preflight_failures(
            min_free_memory_mib=args.min_free_memory_mib,
            max_competing_cpu_percent=args.max_competing_cpu_percent,
        )
        if preflight_failures:
            for failure in preflight_failures:
                print(f"ERROR: resource preflight: {failure}", file=sys.stderr)
            return 1

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

    calibration_rows: list[CalibrationArtifactRow]
    if args.stability_runs > 1:
        stability_summaries = summarize_stability(sweep_runs)
        calibration_rows = calibration_rows_from_stability(stability_summaries, sweep_runs)
        if args.format == "calibration-tsv":
            print_calibration_tsv(calibration_rows)
        elif args.format == "markdown":
            print_stability_markdown(stability_summaries)
        else:
            print_stability_tsv(stability_summaries)
    else:
        calibration_rows = calibration_rows_from_summaries(summaries)
        if args.format == "calibration-tsv":
            print_calibration_tsv(calibration_rows)
        elif args.format == "markdown":
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
        fresh_keys = {
            (row.workload, row.scale)
            for row in calibration_rows
        }
        selected_baseline_rows = [
            row
            for row in baseline_rows
            if (row.workload, row.scale) in fresh_keys
        ]
        drift = compare_calibration_rows(
            calibration_rows,
            selected_baseline_rows,
            timing_drift_ratio=args.timing_drift_ratio,
        )
        for warning in drift.timing:
            print(f"WARNING: calibration timing drift: {warning}", file=sys.stderr)
        for failure in drift.critical:
            failures.append(f"calibration drift: {failure}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    if args.write_calibration_artifact:
        write_merged_calibration_artifact(args.calibration_artifact, calibration_rows)
        print(f"Wrote calibration artifact: {args.calibration_artifact}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
