#!/usr/bin/env python3
"""
Benchmark seeded effective-distance closure generation for the Prolog target.

Targets:
  - prolog-seeded : generated Prolog using seeded counted closure reuse
  - prolog-pruned : generated Prolog using seeded closure reuse plus branch pruning
  - prolog-accumulated : generated Prolog using seeded pre-aggregated weight sums
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
    normalize_three_column_float_rows,
    print_pair_match_status,
    print_phase_metrics,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_distance_benchmark.pl"


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
        default="prolog-seeded,prolog-pruned,prolog-accumulated",
        help="Comma-separated targets: prolog-seeded,prolog-pruned,prolog-accumulated",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def available_targets(requested: list[str]) -> list[str]:
    if shutil.which("swipl") is None:
        print("skip prolog effective-distance benchmark: swipl not found", file=sys.stderr)
        return []
    supported = {"prolog-seeded", "prolog-pruned", "prolog-accumulated"}
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

    normalized = normalize_three_column_float_rows(stdout, decimals=6)
    digest, row_count = digest_normalized_output(normalized)
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results):
        print_result_table(entries, scale)
        seeded = find_result(entries, "prolog-seeded")
        pruned = find_result(entries, "prolog-pruned")
        accumulated = find_result(entries, "prolog-accumulated")
        print_pair_match_status(scale, "seeded_vs_pruned", seeded, pruned)
        print_pair_match_status(scale, "seeded_vs_accumulated", seeded, accumulated)
        print_pair_match_status(scale, "pruned_vs_accumulated", pruned, accumulated)
        print_speedup(scale, "speedup_pruned_vs_seeded", seeded, pruned)
        print_speedup(scale, "speedup_accumulated_vs_seeded", seeded, accumulated)
        print_speedup(scale, "speedup_accumulated_vs_pruned", pruned, accumulated)
        print_phase_metrics(scale, "seeded_metrics", seeded)
        print_phase_metrics(scale, "pruned_metrics", pruned)
        print_phase_metrics(scale, "accumulated_metrics", accumulated)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-prolog-effective-distance-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-prolog-effective-distance-")
        temp_root = Path(temp_ctx.name)

    try:
        results: list[RunResult] = []
        for scale in scales:
            commands: dict[str, list[str]] = {}
            for target in targets:
                if target == "prolog-seeded":
                    commands[target] = build_generated_script(
                        temp_root, scale, "seeded", "effective_distance_seeded.pl"
                    )
                elif target == "prolog-pruned":
                    commands[target] = build_generated_script(
                        temp_root, scale, "pruned", "effective_distance_pruned.pl"
                    )
                elif target == "prolog-accumulated":
                    commands[target] = build_generated_script(
                        temp_root, scale, "accumulated", "effective_distance_accumulated.pl"
                    )
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
