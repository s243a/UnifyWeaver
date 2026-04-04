#!/usr/bin/env python3
"""
Benchmark the Prolog shortest-path-to-root workload with branch pruning
enabled and disabled.

Targets:
  - prolog-source    : original handwritten Prolog benchmark source
  - prolog-pruned    : generated Prolog target with branch_pruning(auto)
  - prolog-unpruned  : generated Prolog target with branch_pruning(false)
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_common import (
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
GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_shortest_path_benchmark.pl"
SOURCE_WORKLOAD = ROOT / "examples" / "benchmark" / "shortest_path_to_root.pl"


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
        default="prolog-source,prolog-pruned,prolog-unpruned",
        help="Comma-separated targets: prolog-source,prolog-pruned,prolog-unpruned",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def available_targets(requested: list[str]) -> list[str]:
    if shutil.which("swipl") is None:
        print("skip prolog benchmark: swipl not found", file=sys.stderr)
        return []
    supported = {"prolog-source", "prolog-pruned", "prolog-unpruned"}
    targets: list[str] = []
    for target in requested:
        if target not in supported:
            raise ValueError(f"unsupported target: {target}")
        targets.append(target)
    return targets


def scale_facts_path(scale: str) -> Path:
    return require_file(BENCH_DIR / scale / "facts.pl")


def build_generated_script(root: Path, scale: str, branch_setting: str, script_name: str) -> list[str]:
    facts_path = scale_facts_path(scale)
    script_path = root / scale / script_name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(GENERATOR),
            "--",
            str(facts_path),
            str(script_path),
            branch_setting,
        ],
        cwd=ROOT,
    )
    return ["swipl", "-q", "-s", str(script_path)]


def build_source_command(scale: str) -> list[str]:
    facts_path = scale_facts_path(scale)
    return [
        "swipl",
        "-q",
        "-s",
        str(facts_path),
        "-s",
        str(SOURCE_WORKLOAD),
    ]


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run_command(command, cwd=ROOT)
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr

    normalized = normalize_sorted_lines(stdout)
    digest, row_count = digest_normalized_output(normalized)
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results):
        print_result_table(entries, scale)
        print_match_status(scale, "all_outputs", entries)

        source = find_result(entries, "prolog-source")
        pruned = find_result(entries, "prolog-pruned")
        unpruned = find_result(entries, "prolog-unpruned")

        print_pair_match_status(scale, "pruned_vs_unpruned", pruned, unpruned)
        print_pair_match_status(scale, "pruned_vs_source", pruned, source)
        print_pair_match_status(scale, "unpruned_vs_source", unpruned, source)
        print_speedup(scale, "speedup_pruned_vs_unpruned", unpruned, pruned)
        print_speedup(scale, "speedup_pruned_vs_source", source, pruned)
        print_speedup(scale, "speedup_unpruned_vs_source", source, unpruned)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-prolog-pruning-bench-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-prolog-pruning-bench-")
        temp_root = Path(temp_ctx.name)

    try:
        results: list[RunResult] = []
        for scale in scales:
            commands: dict[str, list[str]] = {}
            for target in targets:
                if target == "prolog-source":
                    commands[target] = build_source_command(scale)
                elif target == "prolog-pruned":
                    commands[target] = build_generated_script(temp_root, scale, "auto", "shortest_path_pruned.pl")
                elif target == "prolog-unpruned":
                    commands[target] = build_generated_script(temp_root, scale, "false", "shortest_path_unpruned.pl")
                else:
                    raise ValueError(f"unsupported target: {target}")

            for target in targets:
                results.append(benchmark_target(commands[target], scale, args.repetitions, target))

        print_summary(results)
        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
