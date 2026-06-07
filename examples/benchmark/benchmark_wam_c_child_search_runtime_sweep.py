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
BENCH_DIR = ROOT / "data" / "benchmark"
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
        "--skip-cache-input-summary",
        action="store_true",
        help="Do not print root-distance cache input-size summary rows after the matrix run.",
    )
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


def iter_tsv_fields(path: Path, header: tuple[str, ...]):
    with path.open("r", encoding="utf-8") as handle:
        first_data_row = True
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if first_data_row and tuple(fields) == header:
                first_data_row = False
                continue
            first_data_row = False
            yield fields


def cache_input_summary(scale: str, bench_dir: Path = BENCH_DIR) -> dict[str, int]:
    scale_dir = bench_dir / scale
    roots = {
        fields[0]
        for fields in iter_tsv_fields(scale_dir / "root_categories.tsv", ("category",))
        if fields
    }

    category_ids = set(roots)
    parent_edge_rows = 0
    for fields in iter_tsv_fields(scale_dir / "category_parent.tsv", ("child", "parent")):
        if len(fields) < 2:
            continue
        parent_edge_rows += 1
        category_ids.add(fields[0])
        category_ids.add(fields[1])

    article_category_rows = 0
    for fields in iter_tsv_fields(scale_dir / "article_category.tsv", ("article", "category")):
        if len(fields) < 2:
            continue
        article_category_rows += 1
        category_ids.add(fields[1])

    category_id_count = len(category_ids)
    root_count = len(roots)
    return {
        "roots": root_count,
        "category_ids": category_id_count,
        "parent_edges": parent_edge_rows,
        "article_category_rows": article_category_rows,
        "max_cache_maps": root_count,
        "max_distance_entries_upper_bound": root_count * category_id_count,
    }


def cache_input_summary_line(scale: str, bench_dir: Path = BENCH_DIR) -> str:
    summary = cache_input_summary(scale, bench_dir)
    fields = " ".join(f"{key}={value}" for key, value in summary.items())
    return f"{scale}\twam_c_child_search_cache_inputs\t{fields}"


def print_cache_input_summary(scales_csv: str, bench_dir: Path = BENCH_DIR) -> None:
    for scale in [part.strip() for part in scales_csv.split(",") if part.strip()]:
        print(cache_input_summary_line(scale, bench_dir))


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
    result = subprocess.run(command, cwd=ROOT, check=False)
    if not args.skip_cache_input_summary:
        print_cache_input_summary(args.scales)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
