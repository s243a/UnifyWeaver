#!/usr/bin/env python3
"""
Run or re-render the C# query effective-distance backend policy review.

This is a thin wrapper around
benchmark_csharp_query_effective_distance_artifact_backends.py that keeps the
scripted policy-actionable invocation in one place.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"
BENCHMARK_SCRIPT = BENCHMARK_DIR / "benchmark_csharp_query_effective_distance_artifact_backends.py"
OUTPUT_DIR = ROOT / "output"
DEFAULT_SUMMARY_OUTPUT = OUTPUT_DIR / "csharp-query-effective-distance-policy-summary.tsv"
DEFAULT_ARTIFACT_ROOT = OUTPUT_DIR / "csharp-query-effective-distance-artifacts"
POLICY_FORMATS = (
    "policy-tsv",
    "policy-markdown",
    "policy-compare-tsv",
    "policy-compare-markdown",
    "policy-actionable-tsv",
    "policy-actionable-markdown",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="dev", help="comma-separated scales for a live benchmark run")
    parser.add_argument(
        "--relation",
        choices=("category_parent", "article_category"),
        default="category_parent",
        help="support relation to benchmark during live runs",
    )
    parser.add_argument("--lookup-keys", type=int, default=4)
    parser.add_argument("--lookup-repetitions", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="persistent artifact directory for live runs",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="summary TSV written during live runs",
    )
    parser.add_argument(
        "--summary-input",
        type=Path,
        default=None,
        help="existing summary TSV to render offline instead of running benchmarks",
    )
    parser.add_argument(
        "--format",
        choices=POLICY_FORMATS,
        default="policy-actionable-markdown",
        help="policy report format to render",
    )
    parser.add_argument(
        "--policy-action-threshold",
        type=float,
        default=1.10,
        help="minimum policy-vs-best ratio included by actionable policy reports",
    )
    parser.add_argument(
        "--no-fail-on-policy-actions",
        action="store_true",
        help="do not fail when actionable policy rows remain",
    )
    parser.add_argument(
        "--no-require-idle",
        action="store_true",
        help="do not pass --require-idle to the underlying live benchmark",
    )
    parser.add_argument(
        "--skip-resource-check",
        action="store_true",
        help="forward --skip-resource-check to the underlying live benchmark",
    )
    parser.add_argument(
        "--skip-missing-scales",
        action="store_true",
        help="forward --skip-missing-scales to the underlying live benchmark",
    )
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="rebuild artifacts under --artifact-root instead of reusing manifests",
    )
    parser.add_argument(
        "--no-use-scale-lmdb-artifact",
        action="store_true",
        help="do not prefer scale-local LMDB manifests during live runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the underlying command without running it",
    )
    args = parser.parse_args(argv)

    if args.lookup_keys <= 0:
        parser.error("--lookup-keys must be positive")
    if args.lookup_repetitions <= 0:
        parser.error("--lookup-repetitions must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.policy_action_threshold < 1.0:
        parser.error("--policy-action-threshold must be at least 1.0")
    if args.summary_input is not None and args.summary_output != DEFAULT_SUMMARY_OUTPUT:
        parser.error("--summary-output cannot be used with --summary-input")
    return args


def build_review_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
    ]

    if args.summary_input is not None:
        command.extend(["--summary-input", str(args.summary_input)])
    else:
        command.extend(
            [
                "--scales",
                args.scales,
                "--relation",
                args.relation,
                "--lookup-keys",
                str(args.lookup_keys),
                "--lookup-repetitions",
                str(args.lookup_repetitions),
                "--repetitions",
                str(args.repetitions),
                "--artifact-root",
                str(args.artifact_root),
                "--summary-output",
                str(args.summary_output),
            ]
        )
        if not args.no_require_idle:
            command.append("--require-idle")
        if args.skip_resource_check:
            command.append("--skip-resource-check")
        if args.skip_missing_scales:
            command.append("--skip-missing-scales")
        if args.refresh_artifacts:
            command.append("--refresh-artifacts")
        if not args.no_use_scale_lmdb_artifact:
            command.append("--use-scale-lmdb-artifact")

    command.extend(
        [
            "--policy-action-threshold",
            f"{args.policy_action_threshold:g}",
            "--format",
            args.format,
        ]
    )
    if not args.no_fail_on_policy_actions:
        command.append("--fail-on-policy-actions")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_review_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
