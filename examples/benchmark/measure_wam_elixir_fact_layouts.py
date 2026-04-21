#!/usr/bin/env python3
"""Measure WAM-Elixir fact-shape layouts across scales.

Thin driver around ``measure_wam_elixir_fact_layouts.pl``. Follows the
C# hybrid-WAM matrix-harness pattern
(``benchmark_effective_distance_matrix.py``): run each explicit layout
plus the auto-policies, then print a summary table with the layout the
planner actually picked and a ``ratio_vs_min_bytes`` column showing how
close each non-winner got to the cheapest-by-bytes choice.

Host-compile / query-latency measurements are left out here
intentionally. The cheap signals (codegen time, module size) are what
actually runs in CI / on the development box; the richer profiling
belongs on a desktop environment and can plug into the same output
format later.

Example::

    python3 measure_wam_elixir_fact_layouts.py \\
        --scales dev,300 --predicates category_parent/2

Emits a TSV block per scale/predicate plus a `ratio` summary that
makes it obvious which layout is cheapest and how much worse auto /
cost_aware is.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
PL_SCRIPT = Path(__file__).with_suffix(".pl")


@dataclass
class Row:
    scale: str
    predicate: str
    layout: str
    codegen_ms: int
    module_bytes: int
    resolved_as: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default="dev,300",
        help="Comma-separated scale directories under data/benchmark/",
    )
    parser.add_argument(
        "--predicates",
        default="category_parent/2,article_category/2",
        help="Comma-separated predicate indicators to measure (pred/arity)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-invocation swipl timeout in seconds",
    )
    return parser.parse_args()


def run_measurement(scale: str, predicate: str, timeout_s: int) -> list[Row]:
    facts_path = BENCH_DIR / scale / "facts.pl"
    if not facts_path.exists():
        raise FileNotFoundError(f"facts not found: {facts_path}")
    cmd = [
        "swipl",
        "-q",
        "-s",
        str(PL_SCRIPT),
        "--",
        str(facts_path),
        predicate,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    rows: list[Row] = []
    for line in result.stdout.splitlines():
        if not line or line.startswith("scale\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        rows.append(
            Row(
                scale=parts[0],
                predicate=parts[1],
                layout=parts[2],
                codegen_ms=int(parts[3]),
                module_bytes=int(parts[4]),
                resolved_as=parts[5],
            )
        )
    return rows


def print_summary(all_rows: list[Row]) -> None:
    print("scale\tpredicate\tlayout\tcodegen_ms\tmodule_bytes\tresolved_as\tratio_vs_min_bytes")
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in all_rows:
        grouped[(row.scale, row.predicate)].append(row)
    for (scale, predicate), entries in sorted(grouped.items()):
        # Ratio vs cheapest-by-bytes among the EXPLICIT layouts — the
        # auto-policies themselves don't count as candidates, they
        # reflect a choice AMONG the explicit ones.
        explicit_bytes = [
            row.module_bytes
            for row in entries
            if row.layout not in ("auto", "cost_aware")
        ]
        min_bytes = min(explicit_bytes) if explicit_bytes else 0
        for row in sorted(entries, key=lambda r: (r.layout,)):
            ratio = (row.module_bytes / min_bytes) if min_bytes else float("nan")
            print(
                f"{row.scale}\t{row.predicate}\t{row.layout}\t"
                f"{row.codegen_ms}\t{row.module_bytes}\t{row.resolved_as}\t"
                f"{ratio:.2f}x"
            )


def main() -> int:
    args = parse_args()
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    predicates = [p.strip() for p in args.predicates.split(",") if p.strip()]

    all_rows: list[Row] = []
    for scale in scales:
        for predicate in predicates:
            try:
                rows = run_measurement(scale, predicate, args.timeout)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as err:
                sys.stderr.write(f"[skip] {scale}/{predicate}: {err}\n")
                continue
            except FileNotFoundError as err:
                sys.stderr.write(f"[skip] {err}\n")
                continue
            all_rows.extend(rows)

    if not all_rows:
        sys.stderr.write("no measurements collected\n")
        return 1
    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
