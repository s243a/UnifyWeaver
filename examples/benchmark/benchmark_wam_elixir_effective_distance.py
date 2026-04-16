#!/usr/bin/env python3
"""
Benchmark the effective-distance workload for the lowered WAM-Elixir target.

This script builds the Elixir project, runs it against one or more
benchmark scales, and reports median wall-clock times.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_common import (
    available_targets,
    digest_normalized_output,
    group_results_by_scale,
    normalize_three_column_float_rows,
    print_result_table,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
WAM_ELIXIR_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_elixir_effective_distance_benchmark.pl"


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
        default="300,1k,5k",
        help="Comma-separated benchmark scales from data/benchmark/",
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


def benchmark_temp_parent() -> Path:
    candidates: list[Path] = []
    for var in ("TMPDIR", "TMP", "TEMP"):
        raw = os.environ.get(var)
        if raw:
            candidates.append(Path(raw))
    prefix = os.environ.get("PREFIX")
    if prefix:
        candidates.append(Path(prefix) / "tmp")
    candidates.append(ROOT / "output")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".uw_tmp_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("no writable temporary directory found")


def build_wam_elixir_effective_distance(root: Path, scale: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir = root / f"wam_elixir" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure erl shebang is fixed for Termux
    erl_path = run_command(["which", "erl"]).stdout.strip()
    run_command(["termux-fix-shebang", erl_path])

    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_ELIXIR_GENERATOR),
            "--",
            str(facts_path),
            str(project_dir),
        ],
        cwd=ROOT,
    )
    
    driver_script = project_dir / "test_bench.exs"
    scale_dir = facts_path.parent
    return ["elixir", str(driver_script), str(scale_dir)]


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run_command(command)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        last_stdout = result.stdout
        last_stderr = result.stderr

    normalized = normalize_three_column_float_rows(last_stdout, decimals=6)
    digest, rows = digest_normalized_output(normalized)
    return RunResult(target=target, scale=scale, times=times, stdout_sha256=digest, row_count=rows, stderr=last_stderr)


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
    
    temp_ctx = None
    temp_parent = benchmark_temp_parent()
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-elixir-bench-", dir=temp_parent))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-elixir-bench-", dir=temp_parent)
        temp_root = Path(temp_ctx.name)
        
    try:
        results: list[RunResult] = []
        for scale in scales:
            print(f"Benchmarking scale {scale}...", file=sys.stderr)
            command = build_wam_elixir_effective_distance(temp_root, scale)
            results.append(benchmark_target(command, scale, args.repetitions, "wam-elixir-lowered"))

        print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
        for scale, entries in group_results_by_scale(results, sort_key=scale_sort_key):
            print_result_table(entries, scale)

        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
