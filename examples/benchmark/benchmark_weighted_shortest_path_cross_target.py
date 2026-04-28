#!/usr/bin/env python3
"""
Benchmark positive weighted shortest-path-to-root across the C# query engine
and generated DFS binaries for C#, Rust, and Go, plus seeded Prolog `min` closure.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_common import (
    available_targets,
    build_csharp_package,
    build_go_binary,
    build_rust_binary,
    digest_normalized_output,
    find_result,
    group_results_by_scale,
    normalize_three_column_float_rows,
    print_bucket_strategy_metrics,
    print_match_status,
    print_pair_match_status,
    print_phase_metrics,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_weighted_shortest_path_benchmark.pl"
FACTS_PATH = BENCH_DIR / "10k" / "facts.pl"


@dataclass
class RunResult:
    target: str
    scale: str
    times: list[float]
    stdout_sha256: str
    row_count: int
    stderr: str

    @property
    def median(self) -> float:
        return statistics.median(self.times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="300,1k,5k,10k")
    parser.add_argument(
        "--targets",
        default="csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def build_csharp_query(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, FACTS_PATH, "weighted_shortest_path", "csharp_query", root / "csharp_query", root="Physics"
    )


def build_csharp_dfs(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, FACTS_PATH, "weighted_shortest_path", "csharp", root / "csharp_dfs", root="Physics"
    )


def build_rust_dfs(root: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, FACTS_PATH, "weighted_shortest_path", root / "rust_dfs", "weighted_shortest_path_rust", root="Physics"
    )


def build_go_dfs(root: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, FACTS_PATH, "weighted_shortest_path", root / "go_dfs", "weighted_shortest_path_go", root="Physics"
    )


def build_prolog_min(root: Path, scale: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    script_path = root / "prolog_min" / scale / "weighted_shortest_path_min.pl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(PROLOG_GENERATOR),
            "--",
            str(facts_path),
            str(script_path),
            "min",
        ],
        cwd=ROOT,
    )
    return ["swipl", "-q", "-s", str(script_path)]


def normalize_output(output: str) -> str:
    return normalize_three_column_float_rows(output, decimals=9)


def print_head_to_head(scale: str, left: RunResult | None, right: RunResult | None, label: str) -> None:
    if left and right:
        faster = left if left.median < right.median else right
        slower = right if faster is left else left
        print(f"{scale}\t{label}_faster\t{faster.target}")
        print(f"{scale}\t{label}_speedup\t{slower.median / faster.median:.2f}x")


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target == "prolog-min":
            result = run_command(command, cwd=ROOT)
        else:
            scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
            edge_path = scale_dir / "category_parent.tsv"
            article_path = scale_dir / "article_category.tsv"
            result = run_command(command + [str(edge_path), str(article_path)])
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr

    normalized = normalize_output(stdout)
    digest, row_count = digest_normalized_output(normalized)
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results):
        print_result_table(entries, scale)

        qe = find_result(entries, "csharp-query")
        csharp_dfs = find_result(entries, "csharp-dfs")
        rust_dfs = find_result(entries, "rust-dfs")
        go_dfs = find_result(entries, "go-dfs")
        prolog_min = find_result(entries, "prolog-min")
        dfs_like = [item for item in entries if item.target in {"csharp-dfs", "rust-dfs", "go-dfs"}]

        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)
        print_pair_match_status(scale, "query_vs_csharp_dfs", qe, csharp_dfs)
        print_pair_match_status(scale, "query_vs_prolog_min", qe, prolog_min)
        print_speedup(scale, "speedup_vs_csharp_dfs", csharp_dfs, qe)
        print_speedup(scale, "speedup_vs_rust_dfs", rust_dfs, qe)
        print_speedup(scale, "speedup_vs_go_dfs", go_dfs, qe)
        print_head_to_head(scale, qe, prolog_min, "query_vs_prolog_min")
        print_phase_metrics(scale, "csharp-query-metrics", qe)
        print_bucket_strategy_metrics(scale, "csharp-query-bucket-strategies", qe)
        print_phase_metrics(scale, "prolog-min-metrics", prolog_min)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-weighted-cross-target-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-weighted-cross-target-")
        temp_root = Path(temp_ctx.name)

    try:
        commands: dict[str, list[str]] = {}
        for target in targets:
            if target == "csharp-query":
                commands[target] = build_csharp_query(temp_root)
            elif target == "csharp-dfs":
                commands[target] = build_csharp_dfs(temp_root)
            elif target == "rust-dfs":
                commands[target] = build_rust_dfs(temp_root)
            elif target == "go-dfs":
                commands[target] = build_go_dfs(temp_root)
            elif target != "prolog-min":
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            for target in targets:
                if target == "prolog-min":
                    command = build_prolog_min(temp_root, scale)
                else:
                    command = commands[target]
                results.append(benchmark_target(command, scale, args.repetitions, target))

        print_summary(results)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
