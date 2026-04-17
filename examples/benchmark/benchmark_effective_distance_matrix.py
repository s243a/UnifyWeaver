#!/usr/bin/env python3
"""
Benchmark effective-distance across multiple execution families without
mixing them implicitly.

The important distinction is the compilation path:

  - optimized-prolog:
      Prolog workload -> prolog_target optimization -> generated benchmark
      surface or lowered Haskell with WAM fallback available
  - hybrid-wam:
      optimized Prolog helpers compiled to WAM-backed targets
  - direct-pipeline:
      direct target-language pipeline generators, not the optimized-Prolog path
  - query-engine:
      the C# parameterized query runtime

On Termux, the default target set excludes C# because running it through
proot Debian would impose an unfair benchmark penalty.
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
    build_csharp_package,
    build_go_binary,
    build_haskell_project,
    build_rust_binary,
    digest_normalized_output,
    find_result,
    group_results_by_scale,
    normalize_three_column_float_rows,
    print_match_status,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)
from benchmark_target_matrix import (
    TARGETS,
    default_target_set_name,
    list_targets_text,
    parse_csv,
    resolve_targets,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_distance_benchmark.pl"
WAM_RUST_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_effective_distance_benchmark.pl"
WAM_HASKELL_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_haskell_matrix_benchmark.pl"
DEFAULT_FACTS = BENCH_DIR / "10k" / "facts.pl"
HASKELL_EXE = "wam-haskell-matrix-bench"


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
        "--target-sets",
        default="",
        help=f"Comma-separated target sets. Defaults to {default_target_set_name()} for this environment.",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Comma-separated explicit targets. Overrides the default target set selection.",
    )
    parser.add_argument("--include-targets", default="")
    parser.add_argument("--exclude-targets", default="")
    parser.add_argument("--baseline-target", default="prolog-accumulated")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
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
    candidates.extend([ROOT / "output", Path("/tmp")])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".uw_matrix_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("no writable temporary directory found")


def build_csharp_query(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, DEFAULT_FACTS, "effective_distance", "csharp_query", root / "csharp_query", root="Physics"
    )


def build_csharp_dfs(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, DEFAULT_FACTS, "effective_distance", "csharp", root / "csharp_dfs", root="Physics"
    )


def build_rust_dfs(root: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, DEFAULT_FACTS, "effective_distance", root / "rust_dfs", "effective_distance_rust", root="Physics"
    )


def build_go_dfs(root: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, DEFAULT_FACTS, "effective_distance", root / "go_dfs", "effective_distance_go", root="Physics"
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
        ],
        cwd=ROOT,
    )
    return ["swipl", "-q", "-s", str(script_path)]


def build_wam_rust_effective_distance(root: Path, scale: str, variant: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir = root / f"wam_rust_{variant}" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_RUST_GENERATOR),
            "--",
            str(facts_path),
            str(project_dir),
            variant,
        ],
        cwd=ROOT,
    )
    run_command(["cargo", "build", "--release"], cwd=project_dir)
    binary = project_dir / "target" / "release" / "hybrid_ed_bench"
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    return [str(binary), str(scale_dir)]


def build_haskell_effective_distance(root: Path, mode: str, kernel_mode: str) -> list[str]:
    project_dir = root / f"haskell_{mode}_{kernel_mode}"
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_HASKELL_GENERATOR),
            "--",
            str(DEFAULT_FACTS),
            str(project_dir),
            "accumulated",
            mode,
            kernel_mode,
        ],
        cwd=ROOT,
    )
    return build_haskell_project(project_dir, HASKELL_EXE)


def build_scale_independent_commands(root: Path, targets: list[str]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for target in targets:
        if target == "csharp-query":
            commands[target] = build_csharp_query(root)
        elif target == "csharp-dfs":
            commands[target] = build_csharp_dfs(root)
        elif target == "rust-dfs":
            commands[target] = build_rust_dfs(root)
        elif target == "go-dfs":
            commands[target] = build_go_dfs(root)
        elif target == "haskell-pure-interp":
            commands[target] = build_haskell_effective_distance(root, "interpreter", "kernels_off")
        elif target == "haskell-interp-ffi":
            commands[target] = build_haskell_effective_distance(root, "interpreter", "kernels_on")
        elif target == "haskell-lowered-only":
            commands[target] = build_haskell_effective_distance(root, "functions", "kernels_off")
        elif target == "haskell-lowered-ffi":
            commands[target] = build_haskell_effective_distance(root, "functions", "kernels_on")
    return commands


def normalize_output(output: str) -> str:
    return normalize_three_column_float_rows(output, decimals=9)


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target.startswith("prolog-") or target.startswith("wam-"):
            result = run_command(command, cwd=ROOT)
        else:
            scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
            edge_path = scale_dir / "category_parent.tsv"
            article_path = scale_dir / "article_category.tsv"
            result = run_command(command + [str(edge_path), str(article_path)], cwd=ROOT)
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr

    normalized = normalize_output(stdout)
    digest, row_count = digest_normalized_output(normalized)
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult], baseline_target: str) -> None:
    print("scale\ttarget\tcategory\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results):
        for result in sorted(entries, key=lambda item: item.target):
            category = TARGETS[result.target].category
            print(
                f"{scale}\t{result.target}\t{category}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}\t"
                f"{result.row_count}\t{result.stdout_sha256[:12]}"
            )

        if len(entries) > 1:
            print_match_status(scale, "all_outputs", entries)

        for category in sorted({TARGETS[item.target].category for item in entries}):
            category_entries = [item for item in entries if TARGETS[item.target].category == category]
            if len(category_entries) > 1:
                print_match_status(scale, f"{category}_outputs", category_entries)

        baseline = find_result(entries, baseline_target)
        if baseline:
            for result in sorted(entries, key=lambda item: item.target):
                if result.target == baseline_target:
                    continue
                print_speedup(scale, f"speedup_vs_{baseline_target}_{result.target}", baseline, result)


def resolve_requested_targets(args: argparse.Namespace) -> list[str]:
    explicit_targets = parse_csv(args.targets) if args.targets else None
    target_set_names = parse_csv(args.target_sets) if args.target_sets else None
    include_targets = parse_csv(args.include_targets) if args.include_targets else None
    exclude_targets = parse_csv(args.exclude_targets) if args.exclude_targets else None
    resolved = resolve_targets(
        explicit_targets=explicit_targets,
        target_set_names=target_set_names,
        include_targets=include_targets,
        exclude_targets=exclude_targets,
    )
    return available_targets(resolved)


def main() -> int:
    args = parse_args()
    if args.list_targets:
        print(list_targets_text())
        return 0

    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = resolve_requested_targets(args)
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_parent = benchmark_temp_parent()
    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-effective-matrix-", dir=temp_parent))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-effective-matrix-", dir=temp_parent)
        temp_root = Path(temp_ctx.name)

    try:
        commands = build_scale_independent_commands(temp_root, targets)
        results: list[RunResult] = []
        for scale in scales:
            for target in targets:
                if target == "prolog-seeded":
                    command = build_prolog_effective_distance(temp_root, scale, "seeded")
                elif target == "prolog-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "accumulated")
                elif target == "wam-rust-seeded":
                    command = build_wam_rust_effective_distance(temp_root, scale, "seeded")
                elif target == "wam-rust-accumulated":
                    command = build_wam_rust_effective_distance(temp_root, scale, "accumulated")
                else:
                    command = commands[target]
                results.append(benchmark_target(command, scale, args.repetitions, target))

        print_summary(results, args.baseline_target)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
