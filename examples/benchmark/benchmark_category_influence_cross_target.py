#!/usr/bin/env python3
"""
Benchmark category influence propagation across the C# query engine,
generated Prolog, and generated Rust/Go binaries.
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
    normalize_two_column_float_rows,
    print_bucket_strategy_metrics,
    print_match_status,
    print_phase_metrics,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
FACTS_PATH = BENCH_DIR / "10k" / "facts.pl"
PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_category_influence_benchmark.pl"


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
        default="csharp-query,rust-dfs,go-dfs",
        help="Comma-separated targets: csharp-query,rust-dfs,go-dfs,prolog-accumulated",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def build_csharp_query(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, FACTS_PATH, "category_influence", "csharp_query", root / "csharp_query"
    )


def build_rust_dfs(root: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, FACTS_PATH, "category_influence", root / "rust_dfs", "category_influence_rust"
    )


def build_go_dfs(root: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, FACTS_PATH, "category_influence", root / "go_dfs", "category_influence_go"
    )


def build_prolog_accumulated(root: Path, scale: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    script_path = root / "prolog_accumulated" / scale / "category_influence_accumulated.pl"
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
            "accumulated",
        ],
        cwd=ROOT,
    )
    return ["swipl", "-q", "-s", str(script_path)]


def normalize_output(output: str) -> str:
    return normalize_two_column_float_rows(output, decimals=9, descending_numeric=True)


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target == "prolog-accumulated":
            result = run_command(command, cwd=ROOT)
        else:
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
        csharp = find_result(entries, "csharp-query")
        if len(entries) > 1:
            print_match_status(scale, "outputs", entries)
            rust = find_result(entries, "rust-dfs")
            go = find_result(entries, "go-dfs")
            prolog = find_result(entries, "prolog-accumulated")
            print_speedup(scale, "speedup_vs_rust_dfs", rust, csharp)
            print_speedup(scale, "speedup_vs_go_dfs", go, csharp)
            print_speedup(scale, "speedup_vs_prolog_accumulated", prolog, csharp)
            if rust and go:
                faster = "rust-dfs" if rust.median < go.median else "go-dfs"
                speedup = (go.median / rust.median) if faster == "rust-dfs" else (rust.median / go.median)
                print(f"{scale}\tfaster_target\t{faster}")
                print(f"{scale}\tspeedup\t{speedup:.2f}x")
            print_phase_metrics(scale, "prolog-accumulated-metrics", prolog)
        print_phase_metrics(scale, "csharp-query-metrics", csharp)
        print_bucket_strategy_metrics(scale, "csharp-query-bucket-strategies", csharp)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-category-influence-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-category-influence-")
        temp_root = Path(temp_ctx.name)

    try:
        static_commands: dict[str, list[str]] = {}
        for target in targets:
            if target == "csharp-query":
                static_commands[target] = build_csharp_query(temp_root)
            elif target == "rust-dfs":
                static_commands[target] = build_rust_dfs(temp_root)
            elif target == "go-dfs":
                static_commands[target] = build_go_dfs(temp_root)
            elif target == "prolog-accumulated":
                continue
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            commands = dict(static_commands)
            if "prolog-accumulated" in targets:
                commands["prolog-accumulated"] = build_prolog_accumulated(temp_root, scale)

            for target in targets:
                results.append(benchmark_target(commands[target], scale, args.repetitions, target))

        print_summary(results)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
