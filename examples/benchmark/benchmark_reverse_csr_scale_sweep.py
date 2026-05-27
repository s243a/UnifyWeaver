#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
benchmark_reverse_csr_scale_sweep.py — sweep synthetic reverse CSR scales.

This is a thin wrapper around generate_synthetic_phase1_lmdb.py and
benchmark_reverse_csr_lookup.py. It records how CSR bytes, parent-only
LMDB bytes, build time, and lookup time move as parent count and fanout
change.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"
GENERATOR = BENCHMARK_DIR / "generate_synthetic_phase1_lmdb.py"
LOOKUP_BENCHMARK = BENCHMARK_DIR / "benchmark_reverse_csr_lookup.py"


HEADERS = [
    "parents",
    "children_per_parent",
    "edge_count",
    "backend",
    "index_backend",
    "sample_parents",
    "iterations",
    "total_children",
    "median_ms",
    "min_ms",
    "max_ms",
    "csr_artifact_bytes",
    "csr_bytes_per_edge",
    "csr_bytes_per_parent",
    "csr_build_seconds",
    "offset_index_bytes",
    "parent_lmdb_env_bytes",
    "parent_lmdb_bytes_per_edge",
    "phase1_lmdb_env_bytes",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep synthetic reverse CSR lookup scales.")
    parser.add_argument(
        "--scale",
        action="append",
        default=[],
        help="scale as PARENTSxCHILDREN_PER_PARENT, e.g. 10000x8; may be repeated",
    )
    parser.add_argument("--sample-parents", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--work-dir", type=Path, default=None, help="keep generated artifacts under this directory")
    parser.add_argument("--refresh", action="store_true", help="replace generated artifacts under --work-dir")
    return parser.parse_args(argv)


def parse_scale(value: str) -> tuple[int, int]:
    try:
        parents_text, children_text = value.lower().split("x", 1)
        parents = int(parents_text)
        children_per_parent = int(children_text)
    except ValueError as exc:
        raise ValueError(f"invalid --scale {value!r}; expected PARENTSxCHILDREN_PER_PARENT") from exc
    if parents <= 0 or children_per_parent <= 0:
        raise ValueError(f"invalid --scale {value!r}; both values must be positive")
    return parents, children_per_parent


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def generate_fixture(out_dir: Path, parents: int, children_per_parent: int, refresh: bool) -> None:
    command = [
        sys.executable,
        str(GENERATOR),
        str(out_dir),
        "--parents",
        str(parents),
        "--children-per-parent",
        str(children_per_parent),
    ]
    if refresh:
        command.append("--refresh")
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"fixture generation failed for {parents}x{children_per_parent}:\n{result.stderr}")


def run_lookup_benchmark(
    phase1_dir: Path,
    csr_dir: Path,
    parent_lmdb_dir: Path,
    sample_parents: int,
    iterations: int,
    seed: int,
    refresh: bool,
) -> list[dict[str, str]]:
    command = [
        sys.executable,
        str(LOOKUP_BENCHMARK),
        str(phase1_dir),
        "--csr-dir",
        str(csr_dir),
        "--parent-lmdb-dir",
        str(parent_lmdb_dir),
        "--csr-index-backends",
        "sorted_array,lmdb_offset",
        "--sample-parents",
        str(sample_parents),
        "--iterations",
        str(iterations),
        "--seed",
        str(seed),
    ]
    if refresh:
        command.extend(["--refresh-csr", "--refresh-parent-lmdb"])
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"lookup benchmark failed for {phase1_dir}:\n{result.stderr}")
    return list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))


def row_with_scale(row: dict[str, str], parents: int, children_per_parent: int) -> dict[str, str]:
    edge_count = parents * children_per_parent
    csr_artifact_bytes = int(row["csr_artifact_bytes"])
    parent_lmdb_env_bytes = int(row["parent_lmdb_env_bytes"])
    return {
        "parents": str(parents),
        "children_per_parent": str(children_per_parent),
        "edge_count": str(edge_count),
        **row,
        "csr_bytes_per_edge": f"{csr_artifact_bytes / edge_count:.6f}",
        "csr_bytes_per_parent": f"{csr_artifact_bytes / parents:.6f}",
        "parent_lmdb_bytes_per_edge": f"{parent_lmdb_env_bytes / edge_count:.6f}",
    }


def print_tsv(rows: list[dict[str, str]]) -> None:
    print("\t".join(HEADERS))
    for row in rows:
        print("\t".join(row[h] for h in HEADERS))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        scales = [parse_scale(value) for value in (args.scale or ["1000x8", "10000x8"])]
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        work_dir = Path(temp_dir.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    try:
        for parents, children_per_parent in scales:
            scale_dir = work_dir / f"{parents}x{children_per_parent}"
            phase1_dir = scale_dir / "phase1.lmdb"
            csr_dir = scale_dir / "csr"
            parent_lmdb_dir = scale_dir / "parent_only.lmdb"
            generate_fixture(phase1_dir, parents, children_per_parent, args.refresh)
            benchmark_rows = run_lookup_benchmark(
                phase1_dir,
                csr_dir,
                parent_lmdb_dir,
                args.sample_parents,
                args.iterations,
                args.seed,
                args.refresh,
            )
            rows.extend(row_with_scale(row, parents, children_per_parent) for row in benchmark_rows)
        print_tsv(rows)
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
