#!/usr/bin/env python3
"""
Benchmark the effective-distance workload for the C# query engine, seeded
Prolog, optional direct article/root Prolog, and the compiled DFS pipelines.

Default targets:
  - csharp-query  : current C# query runtime using PathAwareTransitiveClosureNode
  - prolog-seeded : generated Prolog using seeded counted-closure reuse
  - prolog-accumulated : generated Prolog using seeded pre-aggregated weight sums
  - csharp-dfs    : generated C# DFS pipeline
  - rust-dfs      : generated Rust DFS pipeline

The script builds temporary binaries locally, runs them against one or more
benchmark scales, and reports median wall-clock times plus output agreement.
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
PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_distance_benchmark.pl"


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
    parser.add_argument(
        "--scales",
        default="300,1k,5k,10k",
        help="Comma-separated benchmark scales from data/benchmark/",
    )
    parser.add_argument(
        "--targets",
        default="csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-accumulated",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-seeded,prolog-pruned,prolog-accumulated,prolog-article-accumulated",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of timed runs per target/scale",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary build directory for inspection",
    )
    return parser.parse_args()


def build_csharp_query(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", "csharp_query", root / "csharp_query", root="Physics"
    )


def build_csharp_dfs(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", "csharp", root / "csharp_dfs", root="Physics"
    )


def build_rust_dfs(root: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", root / "rust_dfs", "effective_distance_rust", root="Physics"
    )


def build_go_dfs(root: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", root / "go_dfs", "effective_distance_go", root="Physics"
    )


def build_prolog_effective_distance(root: Path, scale: str, variant: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    script_path = root / f"prolog_{variant}" / scale / f"effective_distance_{variant}.pl"
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
            variant,
        ]
    )
    return ["swipl", "-q", "-s", str(script_path)]


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target.startswith("prolog-"):
            result = run_command(command)
        else:
            scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
            edge_path = scale_dir / "category_parent.tsv"
            article_path = scale_dir / "article_category.tsv"
            result = run_command(command + [str(edge_path), str(article_path)])
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        last_stdout = result.stdout
        last_stderr = result.stderr

    normalized = normalize_three_column_float_rows(last_stdout, decimals=6)
    digest, rows = digest_normalized_output(normalized)
    return RunResult(target=target, scale=scale, times=times, stdout_sha256=digest, row_count=rows, stderr=last_stderr)


def print_summary(results: list[RunResult]) -> None:
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results, sort_key=scale_sort_key):
        print_result_table(entries, scale)

        qe = find_result(entries, "csharp-query")
        csharp_dfs = find_result(entries, "csharp-dfs")
        rust_dfs = find_result(entries, "rust-dfs")
        prolog_seeded = find_result(entries, "prolog-seeded")
        prolog_pruned = find_result(entries, "prolog-pruned")
        prolog_accumulated = find_result(entries, "prolog-accumulated")
        prolog_article_accumulated = find_result(entries, "prolog-article-accumulated")
        dfs_like = [item for item in entries if item.target in {"csharp-dfs", "rust-dfs", "go-dfs"}]

        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)
        print_pair_match_status(scale, "query_vs_csharp_dfs", qe, csharp_dfs)
        print_pair_match_status(scale, "query_vs_prolog_seeded", qe, prolog_seeded)
        print_pair_match_status(scale, "query_vs_prolog_pruned", qe, prolog_pruned)
        print_pair_match_status(scale, "query_vs_prolog_accumulated", qe, prolog_accumulated)
        print_pair_match_status(scale, "query_vs_prolog_article_accumulated", qe, prolog_article_accumulated)
        print_speedup(scale, "speedup_vs_csharp_dfs", csharp_dfs, qe)
        print_speedup(scale, "speedup_vs_rust_dfs", rust_dfs, qe)
        print_speedup(scale, "speedup_vs_prolog_seeded", prolog_seeded, qe)
        print_speedup(scale, "speedup_vs_prolog_pruned", prolog_pruned, qe)
        print_speedup(scale, "speedup_vs_prolog_accumulated", prolog_accumulated, qe)
        print_speedup(scale, "speedup_vs_prolog_article_accumulated", prolog_article_accumulated, qe)
        print_phase_metrics(scale, "csharp-query-metrics", qe)
        print_phase_metrics(scale, "prolog-seeded-metrics", prolog_seeded)
        print_phase_metrics(scale, "prolog-pruned-metrics", prolog_pruned)
        print_phase_metrics(scale, "prolog-accumulated-metrics", prolog_accumulated)
        print_phase_metrics(scale, "prolog-article-accumulated-metrics", prolog_article_accumulated)


def scale_sort_key(scale: str) -> tuple[int, str]:
    digits = "".join(ch for ch in scale if ch.isdigit())
    suffix = "".join(ch for ch in scale if not ch.isdigit())
    if scale == "dev":
        return (0, scale)
    if not digits:
        return (10**9, scale)
    value = int(digits)
    multiplier = 1000 if suffix.lower() == "k" else 1
    return (value * multiplier, scale)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    requested_targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    targets = available_targets(requested_targets)
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-effective-distance-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-effective-distance-")
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
            elif target == "prolog-seeded":
                continue
            elif target == "prolog-pruned":
                continue
            elif target == "prolog-accumulated":
                continue
            elif target == "prolog-article-accumulated":
                continue
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            for target in targets:
                if target == "prolog-seeded":
                    command = build_prolog_effective_distance(temp_root, scale, "seeded")
                elif target == "prolog-pruned":
                    command = build_prolog_effective_distance(temp_root, scale, "pruned")
                elif target == "prolog-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "accumulated")
                elif target == "prolog-article-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "article_accumulated")
                else:
                    command = commands[target]
                results.append(benchmark_target(command, scale, args.repetitions, target))

        print_summary(results)
        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
