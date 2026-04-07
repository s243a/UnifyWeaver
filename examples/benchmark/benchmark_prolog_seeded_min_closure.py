#!/usr/bin/env python3
"""
Benchmark seeded shortest-path closure generation for the Prolog target.

Targets:
  - prolog-all : generated Prolog using seeded all-path closure per (seed, root)
  - prolog-min : generated Prolog using seeded mode-directed $min closure
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
    print_pair_match_status,
    print_phase_metrics,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_shortest_path_seeded_benchmark.pl"


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
        default="prolog-all,prolog-min",
        help="Comma-separated targets: prolog-all,prolog-min",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def available_targets(requested: list[str]) -> list[str]:
    if shutil.which("swipl") is None:
        print("skip prolog seeded benchmark: swipl not found", file=sys.stderr)
        return []
    supported = {"prolog-all", "prolog-min"}
    targets: list[str] = []
    for target in requested:
        if target not in supported:
            raise ValueError(f"unsupported target: {target}")
        targets.append(target)
    return targets


def scale_facts_path(scale: str) -> Path:
    return require_file(BENCH_DIR / scale / "facts.pl")


def build_generated_script(root: Path, scale: str, variant: str, script_name: str) -> list[str]:
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
            variant,
        ],
        cwd=ROOT,
    )
    return ["swipl", "-q", "-s", str(script_path)]


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
        all_mode = find_result(entries, "prolog-all")
        min_mode = find_result(entries, "prolog-min")
        print_pair_match_status(scale, "all_vs_min", all_mode, min_mode)
        print_speedup(scale, "speedup_min_vs_all", all_mode, min_mode)
        print_phase_metrics(scale, "all_metrics", all_mode)
        print_phase_metrics(scale, "min_metrics", min_mode)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-prolog-seeded-min-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-prolog-seeded-min-")
        temp_root = Path(temp_ctx.name)

    try:
        results: list[RunResult] = []
        for scale in scales:
            commands: dict[str, list[str]] = {}
            for target in targets:
                if target == "prolog-all":
                    commands[target] = build_generated_script(temp_root, scale, "all", "shortest_path_all.pl")
                elif target == "prolog-min":
                    commands[target] = build_generated_script(temp_root, scale, "min", "shortest_path_min.pl")
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
