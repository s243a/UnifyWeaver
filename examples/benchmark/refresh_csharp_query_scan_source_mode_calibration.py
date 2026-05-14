#!/usr/bin/env python3
"""
Refresh the checked-in C# query scan source-mode calibration artifact.

This is a thin wrapper around benchmark_csharp_query_source_mode_sweep.py that
keeps the scan-materialization artifact separate from the graph calibration
artifact.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"
SWEEP_SCRIPT = BENCHMARK_DIR / "benchmark_csharp_query_source_mode_sweep.py"
CALIBRATION_ARTIFACT = BENCHMARK_DIR / "csharp_query_scan_source_mode_calibration.tsv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="dev")
    parser.add_argument("--source-modes", default="auto,preload,artifact-prebuilt")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--stability-runs", type=int, default=3)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=CALIBRATION_ARTIFACT,
        help="Scan calibration TSV to compare and update.",
    )
    parser.add_argument(
        "--no-require-idle",
        action="store_true",
        help="Do not pass --require-idle to the underlying sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying command without running it.",
    )
    return parser.parse_args(argv)


def build_refresh_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--workloads",
        "scan-materialization",
        "--scales",
        args.scales,
        "--source-modes",
        args.source_modes,
        "--repetitions",
        str(args.repetitions),
        "--stability-runs",
        str(args.stability_runs),
        "--compare-calibration",
        "--format",
        "none",
        "--calibration-artifact",
        str(args.calibration_artifact),
        "--write-calibration-artifact",
    ]
    if not args.no_require_idle:
        command.insert(2, "--require-idle")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_refresh_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
