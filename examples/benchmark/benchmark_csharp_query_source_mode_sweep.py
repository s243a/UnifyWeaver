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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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

DEFAULT_WORKLOADS = "all"


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


def main() -> int:
    args = parse_args()
    workloads = parse_workloads(args.workloads)
    try:
        validate_csharp_query_source_modes(args.source_modes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summaries: list[SourceModeSummary] = []
    for workload in workloads:
        summaries.extend(
            run_workload(
                workload,
                scales=args.scales,
                source_modes=args.source_modes,
                repetitions=args.repetitions,
                trace=args.trace,
            )
        )

    if args.format == "markdown":
        print_markdown(summaries)
    else:
        print_tsv(summaries)

    failures = calibration_failures(
        summaries,
        max_auto_vs_best_ratio=args.max_auto_vs_best_ratio,
        fail_on_output_mismatch=args.fail_on_output_mismatch,
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
