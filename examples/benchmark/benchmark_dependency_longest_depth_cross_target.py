#!/usr/bin/env python3
"""
Benchmark dependency longest-depth analysis across generated DFS binaries
for C#, Rust, and Go.
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
    normalize_sorted_lines,
    print_match_status,
    print_pair_match_status,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
DATA_GENERATOR = ROOT / "examples" / "benchmark" / "generate_dependency_benchmark_data.py"
SCALES = {"300": 300, "1k": 1000, "5k": 5000, "10k": 10000}


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
        default="csharp-dfs,rust-dfs,go-dfs",
        help="Comma-separated targets: csharp-dfs,rust-dfs,go-dfs",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def build_dataset(root: Path, scale: str) -> Path:
    dataset_dir = root / "datasets" / scale
    run_command(
        [
            sys.executable,
            str(DATA_GENERATOR),
            "--output-dir",
            str(dataset_dir),
            "--scale",
            scale,
        ]
    )
    return dataset_dir


def build_csharp_query(root: Path, facts_path: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, facts_path, "dependency_longest_depth", "csharp_query", root / "csharp_query"
    )


def build_csharp_dfs(root: Path, facts_path: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, facts_path, "dependency_longest_depth", "csharp", root / "csharp_dfs"
    )


def build_rust_dfs(root: Path, facts_path: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, facts_path, "dependency_longest_depth", root / "rust_dfs", "dependency_longest_depth_rust"
    )


def build_go_dfs(root: Path, facts_path: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, facts_path, "dependency_longest_depth", root / "go_dfs", "dependency_longest_depth_go"
    )


def benchmark_target(command: list[str], dataset_dir: Path, repetitions: int, target: str, scale: str) -> RunResult:
    edge_path = require_file(dataset_dir / "category_parent.tsv")
    article_path = require_file(dataset_dir / "article_category.tsv")

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run_command(command + [str(edge_path), str(article_path)])
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr

    digest, row_count = digest_normalized_output(normalize_sorted_lines(stdout))
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results):
        print_result_table(entries, scale)
        csharp_dfs = find_result(entries, "csharp-dfs")
        rust_dfs = find_result(entries, "rust-dfs")
        go_dfs = find_result(entries, "go-dfs")
        dfs_like = list(entries)
        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    for scale in scales:
        if scale not in SCALES:
            raise SystemExit(f"Unsupported scale {scale!r}; expected one of {', '.join(SCALES)}")
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available")
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-dependency-longest-depth-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-dependency-longest-depth-")
        temp_root = Path(temp_ctx.name)

    try:
        seed_dataset = build_dataset(temp_root, scales[-1])
        seed_facts = seed_dataset / "facts.pl"

        commands: dict[str, list[str]] = {}
        for target in targets:
            if target == "csharp-dfs":
                commands[target] = build_csharp_dfs(temp_root, seed_facts)
            elif target == "rust-dfs":
                commands[target] = build_rust_dfs(temp_root, seed_facts)
            elif target == "go-dfs":
                commands[target] = build_go_dfs(temp_root, seed_facts)
            elif target == "csharp-query":
                raise ValueError("csharp-query longest-depth mode is not ready yet; use csharp-dfs,rust-dfs,go-dfs")
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            dataset_dir = build_dataset(temp_root, scale)
            for target in targets:
                results.append(benchmark_target(commands[target], dataset_dir, args.repetitions, target, scale))

        print_summary(results)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
