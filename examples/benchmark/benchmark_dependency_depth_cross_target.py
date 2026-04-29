#!/usr/bin/env python3
"""
Benchmark dependency reach-count analysis across the C# query engine and
generated DFS binaries for C#, Rust, and Go.
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
    add_csharp_query_source_mode_arg,
    append_csharp_query_source_mode_metric,
    csharp_query_env,
    csharp_query_results,
    csharp_query_source_modes_from_args,
    csharp_query_target_label,
    digest_normalized_output,
    find_result,
    find_csharp_query_result,
    group_results_by_scale,
    normalize_sorted_lines,
    print_bucket_strategy_metrics,
    print_csharp_query_source_mode_summary,
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
        default="csharp-query,csharp-dfs,rust-dfs,go-dfs",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    add_csharp_query_source_mode_arg(parser)
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
        GENERATOR, facts_path, "dependency_depth", "csharp_query", root / "csharp_query"
    )


def build_csharp_dfs(root: Path, facts_path: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, facts_path, "dependency_depth", "csharp", root / "csharp_dfs"
    )


def build_rust_dfs(root: Path, facts_path: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, facts_path, "dependency_depth", root / "rust_dfs", "dependency_depth_rust"
    )


def build_go_dfs(root: Path, facts_path: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, facts_path, "dependency_depth", root / "go_dfs", "dependency_depth_go"
    )


def benchmark_target(
    command: list[str],
    dataset_dir: Path,
    repetitions: int,
    target: str,
    scale: str,
    csharp_query_source_mode: str = "auto",
    artifact_dir: Path | None = None,
    result_target: str | None = None,
) -> RunResult:
    edge_path = require_file(dataset_dir / "category_parent.tsv")
    article_path = require_file(dataset_dir / "article_category.tsv")

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        env = csharp_query_env(csharp_query_source_mode, artifact_dir) if target == "csharp-query" else None
        result = run_command(command + [str(edge_path), str(article_path)], env=env)
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr
        if target == "csharp-query":
            stderr = append_csharp_query_source_mode_metric(stderr, csharp_query_source_mode)

    digest, row_count = digest_normalized_output(normalize_sorted_lines(stdout))
    return RunResult(result_target or target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results):
        print_result_table(entries, scale)
        qe = find_csharp_query_result(entries)
        csharp_dfs = find_result(entries, "csharp-dfs")
        rust_dfs = find_result(entries, "rust-dfs")
        go_dfs = find_result(entries, "go-dfs")
        dfs_like = [item for item in entries if not item.target.startswith("csharp-query")]
        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)
        print_pair_match_status(scale, "query_vs_csharp_dfs", qe, csharp_dfs)
        print_speedup(scale, "speedup_vs_csharp_dfs", csharp_dfs, qe)
        print_speedup(scale, "speedup_vs_rust_dfs", rust_dfs, qe)
        print_speedup(scale, "speedup_vs_go_dfs", go_dfs, qe)
        for csharp_entry in csharp_query_results(entries):
            bucket_label = (
                f"{csharp_entry.target}-bucket-strategies"
                if csharp_entry.target != "csharp-query"
                else "csharp-query-bucket-strategies"
            )
            print_bucket_strategy_metrics(scale, bucket_label, csharp_entry)
        print_csharp_query_source_mode_summary(scale, entries)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    csharp_query_source_modes = csharp_query_source_modes_from_args(args)
    for scale in scales:
        if scale not in SCALES:
            raise SystemExit(f"Unsupported scale {scale!r}; expected one of {', '.join(SCALES)}")
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available")
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-dependency-depth-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-dependency-depth-")
        temp_root = Path(temp_ctx.name)

    try:
        seed_dataset = build_dataset(temp_root, scales[-1])
        seed_facts = seed_dataset / "facts.pl"

        commands: dict[str, list[str]] = {}
        for target in targets:
            if target == "csharp-query":
                commands[target] = build_csharp_query(temp_root, seed_facts)
            elif target == "csharp-dfs":
                commands[target] = build_csharp_dfs(temp_root, seed_facts)
            elif target == "rust-dfs":
                commands[target] = build_rust_dfs(temp_root, seed_facts)
            elif target == "go-dfs":
                commands[target] = build_go_dfs(temp_root, seed_facts)
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            dataset_dir = build_dataset(temp_root, scale)
            for target in targets:
                if target == "csharp-query":
                    for source_mode in csharp_query_source_modes:
                        artifact_dir = temp_root / "artifacts" / target / source_mode / scale
                        results.append(
                            benchmark_target(
                                commands[target],
                                dataset_dir,
                                args.repetitions,
                                target,
                                scale,
                                source_mode,
                                artifact_dir,
                                csharp_query_target_label(source_mode, csharp_query_source_modes),
                            )
                        )
                else:
                    results.append(
                        benchmark_target(
                            commands[target],
                            dataset_dir,
                            args.repetitions,
                            target,
                            scale,
                        )
                    )

        print_summary(results)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
