#!/usr/bin/env python3
"""
Run an end-to-end WAM-C bounded child-search runtime sweep.

This wrapper exercises the generated effective-distance programs instead of
only compiling CSR artifacts. The default run is intentionally small; larger
scales should be requested explicitly because bounded child expansion can make
the full query body expensive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "examples" / "benchmark" / "benchmark_effective_distance_matrix.py"
CHILD_SEARCH_TARGETS = [
    "c-wam-accumulated-child-scan",
    "c-wam-accumulated-child-csr",
    "c-wam-accumulated-child-csr-drop",
    "c-wam-accumulated-child-csr-lmdb-offset",
]
PARENT_ONLY_TARGET = "c-wam-accumulated"
DEFAULT_SCALES = "dev"
DEFAULT_REPETITIONS = 1
DEFAULT_TIMEOUT_SECONDS = 180.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default=DEFAULT_SCALES,
        help=(
            f"Comma-separated benchmark scales. Default: {DEFAULT_SCALES}. "
            "Use --scales dev,10x explicitly for the first larger runtime check."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--run-timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--include-parent-only",
        action="store_true",
        help="Also run the parent-only WAM-C accumulated target for semantic/runtime comparison.",
    )
    parser.add_argument(
        "--baseline-target",
        default="c-wam-accumulated-child-csr",
        help="Matrix baseline for speedup rows. Default: c-wam-accumulated-child-csr.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix command without running it.")
    parser.add_argument(
        "matrix_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after -- to the matrix runner.",
    )
    return parser.parse_args()


def matrix_command(
    scales: str,
    repetitions: int = DEFAULT_REPETITIONS,
    run_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    include_parent_only: bool = False,
    baseline_target: str = "c-wam-accumulated-child-csr",
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX),
        "--scales",
        scales,
        "--target-sets",
        "c-wam-child-search-layouts",
        "--repetitions",
        str(repetitions),
        "--baseline-target",
        baseline_target,
        "--run-timeout-seconds",
        f"{run_timeout_seconds:g}",
    ]
    if include_parent_only:
        command.extend(["--include-targets", PARENT_ONLY_TARGET])
    if extra_args:
        command.extend(extra_args)
    return command


def main() -> int:
    args = parse_args()
    extra_args = args.matrix_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = matrix_command(
        args.scales,
        args.repetitions,
        args.run_timeout_seconds,
        args.include_parent_only,
        args.baseline_target,
        extra_args,
    )
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
