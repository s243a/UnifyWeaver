#!/usr/bin/env python3
"""
Benchmark shortest-path-to-root across the C# query engine and generated
DFS binaries for C#, Rust, and Go.
"""

from __future__ import annotations

import argparse
import statistics
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
    normalize_sorted_lines,
    print_match_status,
    print_pair_match_status,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
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
        default="csharp-query,csharp-dfs,rust-dfs,go-dfs",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def build_csharp_query(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, FACTS_PATH, "shortest_path_to_root", "csharp_query", root / "csharp_query", root="Physics"
    )


def build_csharp_dfs(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, FACTS_PATH, "shortest_path_to_root", "csharp", root / "csharp_dfs", root="Physics"
    )


def build_rust_dfs(root: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, FACTS_PATH, "shortest_path_to_root", root / "rust_dfs", "shortest_path_rust", root="Physics"
    )


def build_go_dfs(root: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, FACTS_PATH, "shortest_path_to_root", root / "go_dfs", "shortest_path_go", root="Physics"
    )


def normalize_output(output: str) -> str:
    return normalize_sorted_lines(output)


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
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
        dfs_like = [item for item in entries if item.target != "csharp-query"]

        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)
        print_pair_match_status(scale, "query_vs_csharp_dfs", qe, csharp_dfs)
        print_speedup(scale, "speedup_vs_csharp_dfs", csharp_dfs, qe)
        print_speedup(scale, "speedup_vs_rust_dfs", rust_dfs, qe)
        print_speedup(scale, "speedup_vs_go_dfs", go_dfs, qe)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-shortest-cross-target-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-shortest-cross-target-")
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
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            for target in targets:
                results.append(benchmark_target(commands[target], scale, args.repetitions, target))

        print_summary(results)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
