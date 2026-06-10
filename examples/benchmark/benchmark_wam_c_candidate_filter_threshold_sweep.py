#!/usr/bin/env python3
"""
Sweep WAM-C candidate-root filtering thresholds for effective distance.

The generated WAM-C runner now has an `auto` threshold for candidate-root
filtering. This wrapper keeps threshold calibration repeatable by running the
existing effective-distance matrix over a small set of root-count profiles and
threshold policies, then reporting hash agreement and the relevant WAM-C
runtime counters.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "examples" / "benchmark" / "benchmark_effective_distance_matrix.py"
TARGET = "c-wam-accumulated-child-csr"
DEFAULT_SCALE = "50k_cats"
DEFAULT_PROFILES = "low,medium,high-capped"
DEFAULT_THRESHOLDS = "auto,always,off"
DEFAULT_REPETITIONS = 1
DEFAULT_TIMEOUT_SECONDS = 180.0
OFF_THRESHOLD = 999_999_999
METRIC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s;]+)")


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    matrix_args: tuple[str, ...]


@dataclass(frozen=True)
class ThresholdSpec:
    label: str
    min_roots: int | None


@dataclass(frozen=True)
class SweepRow:
    scale: str
    profile: str
    threshold_label: str
    threshold_min_roots: int | None
    target: str
    status: str
    median_s: float | None
    rows: int | None
    stdout_sha256: str
    message: str
    metrics: dict[str, str]


PROFILES: dict[str, Profile] = {
    "low": Profile(
        "low",
        "sampled articles and sparse roots; validates the dense low-root path",
        (
            "--wam-c-article-stride",
            "1000",
            "--wam-c-root-stride",
            "100",
        ),
    ),
    "medium": Profile(
        "medium",
        "sampled articles and medium root fanout; probes the auto threshold boundary",
        (
            "--wam-c-article-stride",
            "1000",
            "--wam-c-root-stride",
            "10",
        ),
    ),
    "boundary-250": Profile(
        "boundary-250",
        "sampled articles and about 250 roots; probes below the noisy boundary",
        (
            "--wam-c-article-stride",
            "1000",
            "--wam-c-root-stride",
            "16",
        ),
    ),
    "boundary-500": Profile(
        "boundary-500",
        "sampled articles and about 500 roots; probes just above the noisy boundary",
        (
            "--wam-c-article-stride",
            "1000",
            "--wam-c-root-stride",
            "8",
        ),
    ),
    "boundary-800": Profile(
        "boundary-800",
        "sampled articles and about 800 roots; probes the upper boundary band",
        (
            "--wam-c-article-stride",
            "1000",
            "--wam-c-root-stride",
            "5",
        ),
    ),
    "high-capped": Profile(
        "high-capped",
        "all roots with a result cap; validates sparse scheduling at high fanout",
        (
            "--wam-c-max-results",
            "50",
            "--wam-c-progress-queries",
            "5000",
        ),
    ),
}

SUMMARY_FIELDS = [
    "scale",
    "profile",
    "threshold",
    "threshold_min_roots",
    "status",
    "policy",
    "selected_articles",
    "selected_roots",
    "rows",
    "stdout_sha256",
    "hash_agreement",
    "median_s",
    "query_ms",
    "query_ms_vs_dense",
    "candidate_filter_ms",
    "candidate_filter_articles",
    "candidate_filter_skips",
    "candidate_schedule_roots",
    "category_visits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default=DEFAULT_SCALE)
    parser.add_argument(
        "--profiles",
        default=DEFAULT_PROFILES,
        help=f"Comma-separated profiles. Available: {','.join(PROFILES)}.",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help=(
            "Comma-separated thresholds. Use auto, always, off, or a nonnegative "
            f"integer. Default: {DEFAULT_THRESHOLDS}."
        ),
    )
    parser.add_argument("--repetitions", type=nonnegative_int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--run-timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--format", choices=("tsv", "markdown"), default="tsv")
    parser.add_argument("--dry-run", action="store_true", help="Print matrix commands without running them.")
    parser.add_argument(
        "matrix_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after -- to the matrix runner.",
    )
    return parser.parse_args()


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def csv_values(values: str) -> list[str]:
    return [part.strip() for part in values.split(",") if part.strip()]


def resolve_profiles(profiles: str) -> list[Profile]:
    resolved: list[Profile] = []
    for name in csv_values(profiles):
        try:
            resolved.append(PROFILES[name])
        except KeyError as exc:
            known = ",".join(PROFILES)
            raise argparse.ArgumentTypeError(f"unknown profile {name!r}; expected one of {known}") from exc
    return resolved


def parse_thresholds(thresholds: str) -> list[ThresholdSpec]:
    specs: list[ThresholdSpec] = []
    for raw in csv_values(thresholds):
        lower = raw.lower()
        if lower == "auto":
            specs.append(ThresholdSpec("auto", None))
        elif lower == "always":
            specs.append(ThresholdSpec("always", 1))
        elif lower in {"off", "never"}:
            specs.append(ThresholdSpec("off", OFF_THRESHOLD))
        else:
            min_roots = nonnegative_int(raw)
            label = "auto" if min_roots == 0 else str(min_roots)
            specs.append(ThresholdSpec(label, None if min_roots == 0 else min_roots))
    return specs


def matrix_command(
    scale: str,
    profile: Profile,
    threshold: ThresholdSpec,
    repetitions: int = DEFAULT_REPETITIONS,
    run_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX),
        "--scales",
        scale,
        "--targets",
        TARGET,
        "--repetitions",
        str(repetitions),
        "--baseline-target",
        TARGET,
        "--run-timeout-seconds",
        f"{run_timeout_seconds:g}",
        "--timeout-targets",
        TARGET,
        *profile.matrix_args,
    ]
    if threshold.min_roots is not None:
        command.extend(["--wam-c-candidate-filter-min-roots", str(threshold.min_roots)])
    if extra_args:
        command.extend(extra_args)
    return command


def run_matrix_row(
    scale: str,
    profile: Profile,
    threshold: ThresholdSpec,
    repetitions: int,
    timeout_seconds: float,
    extra_args: list[str],
) -> list[SweepRow]:
    command = matrix_command(scale, profile, threshold, repetitions, timeout_seconds, extra_args)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    rows = parse_matrix_output(result.stdout, profile.name, threshold)
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"matrix command failed for {scale}/{profile.name}/{threshold.label}: {result.stderr}")
    return rows


def parse_matrix_output(output: str, profile: str, threshold: ThresholdSpec) -> list[SweepRow]:
    lines = [line for line in output.splitlines() if line.strip() and "\t" in line]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
    rows: list[SweepRow] = []
    for raw in reader:
        message = raw.get("message", "")
        rows.append(
            SweepRow(
                scale=raw.get("scale", ""),
                profile=profile,
                threshold_label=threshold.label,
                threshold_min_roots=threshold.min_roots,
                target=raw.get("target", ""),
                status=raw.get("status", ""),
                median_s=parse_optional_float(raw.get("median_s")),
                rows=parse_optional_int(raw.get("rows")),
                stdout_sha256=raw.get("stdout_sha256", ""),
                message=message,
                metrics=parse_message_metrics(message),
            )
        )
    return rows


def parse_message_metrics(message: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in METRIC_RE.finditer(message)}


def parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, "", "nan"):
        return None
    return float(value)


def row_policy(row: SweepRow) -> str:
    if row.status != "ok":
        return row.status
    if metric_int(row, "candidate_schedule_roots") > 0:
        return "sparse"
    if metric_int(row, "candidate_filter_articles") > 0:
        return "filtered-dense"
    return "dense"


def metric_int(row: SweepRow, key: str) -> int:
    value = row.metrics.get(key)
    return int(value) if value not in (None, "") else 0


def metric_float(row: SweepRow, key: str) -> float | None:
    value = row.metrics.get(key)
    return float(value) if value not in (None, "") else None


def summarize_rows(rows: list[SweepRow]) -> list[dict[str, str]]:
    dense_baselines = dense_baselines_by_group(rows)
    summaries: list[dict[str, str]] = []
    for row in rows:
        group = (row.scale, row.profile)
        dense = dense_baselines.get(group)
        query_ms = metric_float(row, "query_ms")
        dense_query_ms = metric_float(dense, "query_ms") if dense is not None else None
        summaries.append(
            {
                "scale": row.scale,
                "profile": row.profile,
                "threshold": row.threshold_label,
                "threshold_min_roots": threshold_min_roots_text(row),
                "status": row.status,
                "policy": row_policy(row),
                "selected_articles": row.metrics.get("selected_articles", ""),
                "selected_roots": row.metrics.get("selected_roots", ""),
                "rows": "" if row.rows is None else str(row.rows),
                "stdout_sha256": row.stdout_sha256,
                "hash_agreement": hash_agreement(row, dense),
                "median_s": format_optional_float(row.median_s, 3),
                "query_ms": format_optional_float(query_ms, 3),
                "query_ms_vs_dense": ratio_text(query_ms, dense_query_ms),
                "candidate_filter_ms": row.metrics.get("candidate_filter_ms", ""),
                "candidate_filter_articles": row.metrics.get("candidate_filter_articles", ""),
                "candidate_filter_skips": row.metrics.get("candidate_filter_skips", ""),
                "candidate_schedule_roots": row.metrics.get("candidate_schedule_roots", ""),
                "category_visits": row.metrics.get("category_visits", ""),
            }
        )
    return summaries


def threshold_min_roots_text(row: SweepRow) -> str:
    if row.threshold_min_roots is not None:
        return str(row.threshold_min_roots)
    return row.metrics.get("candidate_filter_min_roots", "")


def dense_baselines_by_group(rows: list[SweepRow]) -> dict[tuple[str, str], SweepRow]:
    grouped: dict[tuple[str, str], list[SweepRow]] = {}
    for row in rows:
        grouped.setdefault((row.scale, row.profile), []).append(row)

    baselines: dict[tuple[str, str], SweepRow] = {}
    for key, group_rows in grouped.items():
        off_rows = [row for row in group_rows if row.threshold_label == "off"]
        if off_rows:
            baselines[key] = off_rows[0]
            continue
        dense_rows = [row for row in group_rows if row_policy(row) == "dense"]
        if dense_rows:
            baselines[key] = dense_rows[0]
            continue
        baselines[key] = group_rows[0]
    return baselines


def hash_agreement(row: SweepRow, dense: SweepRow | None) -> str:
    if dense is None or not dense.stdout_sha256 or not row.stdout_sha256:
        return ""
    return "match" if row.stdout_sha256 == dense.stdout_sha256 else "diff"


def ratio_text(value: float | None, baseline: float | None) -> str:
    if value is None or baseline in (None, 0.0):
        return ""
    return f"{value / baseline:.2f}x"


def format_optional_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def render_tsv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in SUMMARY_FIELDS) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    profiles = resolve_profiles(args.profiles)
    thresholds = parse_thresholds(args.thresholds)
    scales = csv_values(args.scales)
    extra_args = args.matrix_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    commands = [
        (scale, profile, threshold, matrix_command(scale, profile, threshold,
                                                   args.repetitions,
                                                   args.run_timeout_seconds,
                                                   extra_args))
        for scale in scales
        for profile in profiles
        for threshold in thresholds
    ]
    if args.dry_run:
        for scale, profile, threshold, command in commands:
            print(f"{scale}\t{profile.name}\t{threshold.label}\t{' '.join(command)}")
        return 0

    rows: list[SweepRow] = []
    for scale, profile, threshold, _command in commands:
        rows.extend(run_matrix_row(scale, profile, threshold,
                                   args.repetitions,
                                   args.run_timeout_seconds,
                                   extra_args))
    summaries = summarize_rows(rows)
    if args.format == "markdown":
        print(render_markdown(summaries), end="")
    else:
        print(render_tsv(summaries), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
