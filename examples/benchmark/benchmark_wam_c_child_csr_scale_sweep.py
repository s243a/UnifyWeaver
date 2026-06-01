#!/usr/bin/env python3
"""
Run the WAM-C child-search CSR layout scale sweep.

This wrapper keeps the routine sweep compile-only: it builds the generated C
projects and reports artifact byte sizes without executing the expensive
child-search query body. Use the underlying matrix directly for full runtime
rows when a longer benchmark window is intentional.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "examples" / "benchmark" / "benchmark_effective_distance_matrix.py"
CSR_TARGETS = [
    "c-wam-accumulated-child-csr",
    "c-wam-accumulated-child-csr-drop",
    "c-wam-accumulated-child-csr-lmdb-offset",
]
DEFAULT_SCALES = "10x,1k,5k,10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default=DEFAULT_SCALES,
        help=(
            f"Comma-separated benchmark scales. Default: {DEFAULT_SCALES}. "
            "Use --scales 50k_cats,100k_cats explicitly for the largest local artifacts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix command without running it.",
    )
    parser.add_argument(
        "matrix_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after -- to the matrix runner.",
    )
    return parser.parse_args()


def matrix_command(scales: str, extra_args: list[str] | None = None) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX),
        "--scales",
        scales,
        "--target-sets",
        "c-wam-child-csr-layouts",
        "--compile-only-targets",
        ",".join(CSR_TARGETS),
        "--baseline-target",
        "c-wam-accumulated-child-csr",
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def main() -> int:
    args = parse_args()
    extra_args = args.matrix_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = matrix_command(args.scales, extra_args)
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
