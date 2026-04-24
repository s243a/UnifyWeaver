#!/usr/bin/env python3
"""
Benchmark the effective-distance workload for the C# query engine, seeded
Prolog, optional direct article/root and bound-root Prolog variants, and
the compiled DFS pipelines.

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
WAM_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_effective_distance_benchmark.pl"
SEMANTIC_PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_min_semantic_distance_benchmark.pl"
EFF_SEMANTIC_PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_semantic_distance_benchmark.pl"
EDGE_WEIGHT_SCRIPT = ROOT / "examples" / "benchmark" / "precompute_edge_weights.py"


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
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-seeded,prolog-pruned,prolog-accumulated,prolog-article-accumulated,prolog-root-accumulated,wam-rust-seeded,wam-rust-accumulated,wam-rust-seeded-no-kernels,wam-rust-accumulated-no-kernels",
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
    """Pick a writable temp parent, including Termux's $PREFIX/tmp."""
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


def build_prolog_semantic_min(root: Path, scale: str) -> list[str]:
    """Build the min semantic distance Prolog benchmark.

    Requires precomputed edge weights in data/benchmark/<scale>/edge_weights.pl.
    Generate them first with:
        python precompute_edge_weights.py data/benchmark/<scale>/category_parent.tsv data/benchmark/<scale>/
    """
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    weights_path = BENCH_DIR / scale / "edge_weights.pl"
    if not weights_path.exists():
        # Try to precompute if sentence-transformers is available
        edges_path = BENCH_DIR / scale / "category_parent.tsv"
        if edges_path.exists():
            print(f"  Precomputing edge weights for {scale}...", file=sys.stderr)
            run_command(
                [sys.executable, str(EDGE_WEIGHT_SCRIPT), str(edges_path), str(BENCH_DIR / scale)],
                check=False,
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Edge weights not found: {weights_path}\n"
                f"Run: python precompute_edge_weights.py {BENCH_DIR / scale / 'category_parent.tsv'} {BENCH_DIR / scale}/"
            )
    weights_path = require_file(weights_path)

    script_path = root / "prolog_semantic_min" / scale / "min_semantic_distance.pl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl", "-q", "-s", str(SEMANTIC_PROLOG_GENERATOR),
            "--", str(facts_path), str(weights_path), str(script_path),
        ]
    )
    return ["swipl", "-q", "-s", str(script_path)]


def build_prolog_effective_semantic(root: Path, scale: str) -> list[str]:
    """Build the effective semantic distance Prolog benchmark (power-mean over weighted paths)."""
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    weights_path = BENCH_DIR / scale / "edge_weights.pl"
    if not weights_path.exists():
        edges_path = BENCH_DIR / scale / "category_parent.tsv"
        if edges_path.exists():
            print(f"  Precomputing edge weights for {scale}...", file=sys.stderr)
            run_command(
                [sys.executable, str(EDGE_WEIGHT_SCRIPT), str(edges_path), str(BENCH_DIR / scale)],
                check=False,
            )
        if not weights_path.exists():
            raise FileNotFoundError(f"Edge weights not found: {weights_path}")
    weights_path = require_file(weights_path)

    script_path = root / "prolog_eff_semantic" / scale / "effective_semantic_distance.pl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl", "-q", "-s", str(EFF_SEMANTIC_PROLOG_GENERATOR),
            "--", str(facts_path), str(weights_path), str(script_path),
        ]
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
            str(WAM_GENERATOR),
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


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target.startswith("prolog-") or target.startswith("wam-"):
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
    seed_subset_probe = wam_seed_subset_probe_enabled()
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
        prolog_root_accumulated = find_result(entries, "prolog-root-accumulated")
        wam_rust_seeded = find_result(entries, "wam-rust-seeded")
        wam_rust_accumulated = find_result(entries, "wam-rust-accumulated")
        wam_rust_seeded_no_kernels = find_result(entries, "wam-rust-seeded-no-kernels")
        wam_rust_accumulated_no_kernels = find_result(entries, "wam-rust-accumulated-no-kernels")
        prolog_semantic_min = find_result(entries, "prolog-semantic-min")
        prolog_eff_semantic = find_result(entries, "prolog-eff-semantic")
        dfs_like = [item for item in entries if item.target in {"csharp-dfs", "rust-dfs", "go-dfs"}]

        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)
        print_pair_match_status(scale, "query_vs_csharp_dfs", qe, csharp_dfs)
        print_pair_match_status(scale, "query_vs_prolog_seeded", qe, prolog_seeded)
        print_pair_match_status(scale, "query_vs_prolog_pruned", qe, prolog_pruned)
        print_pair_match_status(scale, "query_vs_prolog_accumulated", qe, prolog_accumulated)
        print_pair_match_status(scale, "query_vs_prolog_article_accumulated", qe, prolog_article_accumulated)
        print_pair_match_status(scale, "query_vs_prolog_root_accumulated", qe, prolog_root_accumulated)
        print_pair_match_status(scale, "query_vs_wam_rust_seeded", qe, wam_rust_seeded)
        print_pair_match_status(scale, "query_vs_wam_rust_accumulated", qe, wam_rust_accumulated)
        print_no_kernel_match_status(scale, "query_vs_wam_rust_seeded_no_kernels", qe, wam_rust_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "query_vs_wam_rust_accumulated_no_kernels", qe, wam_rust_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "prolog_vs_wam_rust_seeded", prolog_accumulated, wam_rust_seeded)
        print_pair_match_status(scale, "prolog_vs_wam_rust_accumulated", prolog_accumulated, wam_rust_accumulated)
        print_no_kernel_match_status(scale, "prolog_vs_wam_rust_seeded_no_kernels", prolog_accumulated, wam_rust_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "prolog_vs_wam_rust_accumulated_no_kernels", prolog_accumulated, wam_rust_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "query_vs_prolog_semantic_min", qe, prolog_semantic_min)
        print_pair_match_status(scale, "query_vs_prolog_eff_semantic", qe, prolog_eff_semantic)
        print_speedup(scale, "speedup_vs_csharp_dfs", csharp_dfs, qe)
        print_speedup(scale, "speedup_vs_rust_dfs", rust_dfs, qe)
        print_speedup(scale, "speedup_vs_prolog_seeded", prolog_seeded, qe)
        print_speedup(scale, "speedup_vs_prolog_pruned", prolog_pruned, qe)
        print_speedup(scale, "speedup_vs_prolog_accumulated", prolog_accumulated, qe)
        print_speedup(scale, "speedup_vs_prolog_article_accumulated", prolog_article_accumulated, qe)
        print_speedup(scale, "speedup_vs_prolog_root_accumulated", prolog_root_accumulated, qe)
        print_speedup(scale, "speedup_vs_wam_rust_seeded", wam_rust_seeded, qe)
        print_speedup(scale, "speedup_vs_wam_rust_accumulated", wam_rust_accumulated, qe)
        print_no_kernel_speedup(scale, "speedup_vs_wam_rust_seeded_no_kernels", wam_rust_seeded_no_kernels, qe, seed_subset_probe)
        print_no_kernel_speedup(scale, "speedup_vs_wam_rust_accumulated_no_kernels", wam_rust_accumulated_no_kernels, qe, seed_subset_probe)
        print_speedup(scale, "speedup_vs_prolog_semantic_min", prolog_semantic_min, qe)
        print_speedup(scale, "speedup_vs_prolog_eff_semantic", prolog_eff_semantic, qe)
        print_phase_metrics(scale, "csharp-query-metrics", qe)
        print_phase_metrics(scale, "prolog-seeded-metrics", prolog_seeded)
        print_phase_metrics(scale, "prolog-pruned-metrics", prolog_pruned)
        print_phase_metrics(scale, "prolog-accumulated-metrics", prolog_accumulated)
        print_phase_metrics(scale, "prolog-article-accumulated-metrics", prolog_article_accumulated)
        print_phase_metrics(scale, "prolog-root-accumulated-metrics", prolog_root_accumulated)
        print_phase_metrics(scale, "wam-rust-seeded-metrics", wam_rust_seeded)
        print_phase_metrics(scale, "wam-rust-accumulated-metrics", wam_rust_accumulated)
        print_phase_metrics(scale, "wam-rust-seeded-no-kernels-metrics", wam_rust_seeded_no_kernels)
        print_phase_metrics(scale, "wam-rust-accumulated-no-kernels-metrics", wam_rust_accumulated_no_kernels)
        print_phase_metrics(scale, "prolog-semantic-min-metrics", prolog_semantic_min)
        print_phase_metrics(scale, "prolog-eff-semantic-metrics", prolog_eff_semantic)


def wam_seed_subset_probe_enabled() -> bool:
    return bool(os.environ.get("WAM_SEED_LIMIT") or os.environ.get("WAM_SEED_FILTER"))


def print_no_kernel_match_status(
    scale: str,
    label: str,
    left: RunResult | None,
    right: RunResult | None,
    seed_subset_probe: bool,
) -> None:
    if not (left and right):
        return
    if seed_subset_probe:
        print(f"{scale}\t{label}\tSKIPPED_SEED_SUBSET")
        return
    print_pair_match_status(scale, label, left, right)


def print_no_kernel_speedup(
    scale: str,
    label: str,
    faster_baseline: RunResult | None,
    measured: RunResult | None,
    seed_subset_probe: bool,
) -> None:
    if seed_subset_probe:
        return
    print_speedup(scale, label, faster_baseline, measured)


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
    temp_parent = benchmark_temp_parent()
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-effective-distance-", dir=temp_parent))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-effective-distance-", dir=temp_parent)
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
            elif target == "prolog-root-accumulated":
                continue
            # WAM-Rust variants are generated per scale because facts and
            # optional optimized helpers are loaded into the generated project.
            elif target == "wam-rust-seeded":
                continue
            elif target == "wam-rust-accumulated":
                continue
            elif target == "wam-rust-seeded-no-kernels":
                continue
            elif target == "wam-rust-accumulated-no-kernels":
                continue
            elif target == "prolog-semantic-min":
                continue
            elif target == "prolog-eff-semantic":
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
                elif target == "prolog-root-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "root_accumulated")
                elif target == "wam-rust-seeded":
                    command = build_wam_rust_effective_distance(temp_root, scale, "seeded")
                elif target == "wam-rust-accumulated":
                    command = build_wam_rust_effective_distance(temp_root, scale, "accumulated")
                elif target == "wam-rust-seeded-no-kernels":
                    command = build_wam_rust_effective_distance(temp_root, scale, "seeded_no_kernels")
                elif target == "wam-rust-accumulated-no-kernels":
                    command = build_wam_rust_effective_distance(temp_root, scale, "accumulated_no_kernels")
                elif target == "prolog-semantic-min":
                    command = build_prolog_semantic_min(temp_root, scale)
                elif target == "prolog-eff-semantic":
                    command = build_prolog_effective_semantic(temp_root, scale)
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
