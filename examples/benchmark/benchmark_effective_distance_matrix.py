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
import re
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
    is_termux_environment,
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
WAM_RUST_MATRIX_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_rust_matrix_benchmark.pl"
WAM_HASKELL_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_haskell_matrix_benchmark.pl"
WAM_GO_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_go_effective_distance_benchmark.pl"
WAM_CLOJURE_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_clojure_optimized_benchmark.pl"
DEFAULT_FACTS = BENCH_DIR / "10k" / "facts.pl"
HASKELL_EXE = "wam-haskell-matrix-bench"
RUST_MATRIX_EXE = "wam_rust_matrix_bench"


def default_scales_csv() -> str:
    return "dev,10x" if is_termux_environment() else "300,1k,5k,10k"


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
    parser.add_argument("--scales", default=default_scales_csv())
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
    parser.add_argument(
        "--allow-large-termux-scales",
        action="store_true",
        help="Allow scales beyond the Termux-safe smoke set (dev,10x).",
    )
    return parser.parse_args()


def validate_termux_scales(scales: list[str], allow_large: bool) -> None:
    if not is_termux_environment() or allow_large:
        return
    safe_scales = {"dev", "10x"}
    large_scales = [scale for scale in scales if scale not in safe_scales]
    if large_scales:
        joined = ",".join(large_scales)
        raise ValueError(
            f"large scales are disabled on Termux by default: {joined}. "
            "Use --allow-large-termux-scales to override."
        )


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


def build_wam_go_effective_distance(root: Path, scale: str, kernel_mode: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir = root / f"wam_go_{kernel_mode}" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    variant = "accumulated"
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_GO_GENERATOR),
            "--",
            str(facts_path),
            str(project_dir),
            variant,
            kernel_mode,
        ],
        cwd=ROOT,
    )
    go_cache = project_dir / ".gocache"
    go_cache.mkdir(exist_ok=True)
    env = dict(os.environ, GOCACHE=str(go_cache))
    run_command(["go", "build", "-o", str(project_dir / "hybrid_ed_bench_go")], cwd=project_dir, env=env)
    return [str(project_dir / "hybrid_ed_bench_go")]


def clojure_classpath(project_dir: Path) -> str:
    env_classpath = os.environ.get("CLASSPATH", "")
    helper_jar = project_dir / "lib" / "lmdb-artifact-reader.jar"
    if env_classpath:
        parts = [str(project_dir / "src")]
        if helper_jar.exists():
            parts.append(str(helper_jar))
        parts.append(env_classpath)
        return ":".join(parts)
    jar_paths = [
        Path.home() / ".m2" / "repository" / "org" / "clojure" / "clojure" / "1.11.1" / "clojure-1.11.1.jar",
        Path.home() / ".m2" / "repository" / "org" / "clojure" / "spec.alpha" / "0.3.218" / "spec.alpha-0.3.218.jar",
        Path.home()
        / ".m2"
        / "repository"
        / "org"
        / "clojure"
        / "core.specs.alpha"
        / "0.2.62"
        / "core.specs.alpha-0.2.62.jar",
        Path("/data/data/com.termux/files/home/.m2/repository/org/clojure/clojure/1.11.1/clojure-1.11.1.jar"),
        Path("/data/data/com.termux/files/home/.m2/repository/org/clojure/spec.alpha/0.3.218/spec.alpha-0.3.218.jar"),
        Path("/data/data/com.termux/files/home/.m2/repository/org/clojure/core.specs.alpha/0.2.62/core.specs.alpha-0.2.62.jar"),
    ]
    existing_jars = [path for path in jar_paths if path.exists()]
    if not existing_jars:
        raise RuntimeError("Clojure jars not found; cannot run clojure-wam target")
    parts = [str(project_dir / "src")]
    if helper_jar.exists():
        parts.append(str(helper_jar))
    parts.extend(str(path) for path in existing_jars)
    return ":".join(parts)


def build_wam_clojure_effective_distance(
    root: Path, scale: str, variant: str, kernel_mode: str, data_mode: str = "sidecar"
) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir = root / f"wam_clojure_{variant}_{kernel_mode}" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_CLOJURE_GENERATOR),
            "--",
            str(facts_path),
            str(project_dir),
            variant,
            kernel_mode,
            data_mode,
        ],
        cwd=ROOT,
    )
    command = ["java"]
    native_lib_dir = project_dir / "lib"
    if native_lib_dir.exists():
        command.append(f"-Djava.library.path={native_lib_dir}")
    command.extend(
        [
            "-cp",
            clojure_classpath(project_dir),
            "clojure.main",
            "-m",
            "generated.wam_clojure_optimized_bench.core",
        ]
    )
    return command


def parse_effective_distance_facts(facts_path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    article_categories: list[tuple[str, str]] = []
    roots: list[str] = []
    article_re = re.compile(r"^article_category\('((?:\\.|[^'])*)', '((?:\\.|[^'])*)'\)\.$")
    root_re = re.compile(r"^root_category\('((?:\\.|[^'])*)'\)\.$")
    with facts_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            article_match = article_re.match(line)
            if article_match:
                article_categories.append(
                    (unescape_prolog_atom(article_match.group(1)), unescape_prolog_atom(article_match.group(2)))
                )
                continue
            root_match = root_re.match(line)
            if root_match:
                roots.append(unescape_prolog_atom(root_match.group(1)))
    article_categories = sorted(set(article_categories))
    roots = sorted(set(roots))
    return article_categories, roots


def unescape_prolog_atom(value: str) -> str:
    return value.replace("\\'", "'").replace("\\\\", "\\")


def go_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'


def extract_shared_wam_label(project_dir: Path, label: str) -> int:
    lib_path = require_file(project_dir / "lib.go")
    pattern = re.compile(rf'"{re.escape(label)}":\s*(\d+),')
    content = lib_path.read_text(encoding="utf-8")
    match = pattern.search(content)
    if not match:
        raise RuntimeError(f"could not find sharedWam label {label!r} in {lib_path}")
    return int(match.group(1))


def write_go_effective_distance_main(project_dir: Path, facts_path: Path) -> None:
    article_categories, roots = parse_effective_distance_facts(facts_path)
    category_ancestor_pc = extract_shared_wam_label(project_dir, "category_ancestor/4")
    article_rows = "\n".join(
        f"    {{Article: {go_string_literal(article)}, Category: {go_string_literal(category)}}},"
        for article, category in article_categories
    )
    roots_literal = ", ".join(go_string_literal(root) for root in roots)
    main_code = f"""package main

import (
    "fmt"
    "math"
    "os"
    "sort"
    "time"
)

const (
    categoryAncestorStartPC = {category_ancestor_pc}
    dimensionN = 5.0
    inverseDimensionN = -1.0 / dimensionN
)

type articleCategoryPair struct {{
    Article string
    Category string
}}

type resultRow struct {{
    Article string
    Root string
    Distance float64
}}

var benchmarkArticleCategories = []articleCategoryPair{{
{article_rows}
}}

var benchmarkRoots = []string{{{roots_literal}}}

func newBenchmarkVM(startPC int, args ...Value) *WamState {{
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = startPC
    for i, arg := range args {{
        vm.Regs[fmt.Sprintf("A%d", i+1)] = arg
    }}
    return vm
}}

func collectSolutions(startPC int, args ...Value) [][]Value {{
    vm := newBenchmarkVM(startPC, args...)
    solutions := make([][]Value, 0)
    if !vm.Run() {{
        return solutions
    }}
    solutions = append(solutions, append([]Value(nil), vm.CollectResults()...))
    for vm.backtrack() {{
        if !vm.Run() {{
            break
        }}
        solutions = append(solutions, append([]Value(nil), vm.CollectResults()...))
    }}
    return solutions
}}

func atomName(v Value) string {{
    if a, ok := v.(*Atom); ok {{
        return a.Name
    }}
    if v == nil {{
        return ""
    }}
    return v.String()
}}

func floatValue(v Value) (float64, bool) {{
    switch t := v.(type) {{
    case *Float:
        return t.Val, true
    case *Integer:
        return float64(t.Val), true
    default:
        return 0, false
    }}
}}

func hopsForCategoryRoot(category string, root string) []int {{
    rows := collectSolutions(
        categoryAncestorStartPC,
        &Atom{{Name: category}},
        &Atom{{Name: root}},
        &Unbound{{Name: "hops"}},
        &List{{Elements: []Value{{&Atom{{Name: category}}}}}},
    )
    hops := make([]int, 0, len(rows))
    for _, row := range rows {{
        if len(row) < 3 {{
            continue
        }}
        switch t := row[2].(type) {{
        case *Integer:
            hops = append(hops, int(t.Val))
        case *Float:
            hops = append(hops, int(t.Val))
        }}
    }}
    return hops
}}

func main() {{
    started := time.Now()
    articleCats := append([]articleCategoryPair(nil), benchmarkArticleCategories...)
    roots := append([]string(nil), benchmarkRoots...)
    loadMs := time.Since(started).Milliseconds()
    if len(roots) == 0 {{
        fmt.Fprintln(os.Stderr, "no root categories")
        os.Exit(1)
    }}

    queryStart := time.Now()
    articleToCategories := make(map[string][]string)
    for _, pair := range articleCats {{
        articleToCategories[pair.Article] = append(articleToCategories[pair.Article], pair.Category)
    }}
    articles := make([]string, 0, len(articleToCategories))
    for article := range articleToCategories {{
        articles = append(articles, article)
    }}
    sort.Strings(articles)
    rows := make([]resultRow, 0)
    rootCount := 0
    tupleCount := 0
    for _, root := range roots {{
        rootCount++
        for _, article := range articles {{
            weightSum := 0.0
            for _, category := range articleToCategories[article] {{
                if category == root {{
                    weightSum += 1.0
                    continue
                }}
                hops := hopsForCategoryRoot(category, root)
                tupleCount += len(hops)
                for _, hop := range hops {{
                    weightSum += math.Pow(float64(hop+1), -dimensionN)
                }}
            }}
            if weightSum <= 0 {{
                continue
            }}
            rows = append(rows, resultRow{{
                Article: article,
                Root: root,
                Distance: math.Pow(weightSum, inverseDimensionN),
            }})
        }}
    }}
    queryMs := time.Since(queryStart).Milliseconds()

    aggregationStart := time.Now()
    sort.Slice(rows, func(i, j int) bool {{
        if rows[i].Distance != rows[j].Distance {{
            return rows[i].Distance < rows[j].Distance
        }}
        if rows[i].Root != rows[j].Root {{
            return rows[i].Root < rows[j].Root
        }}
        return rows[i].Article < rows[j].Article
    }})
    aggregationMs := time.Since(aggregationStart).Milliseconds()
    totalMs := time.Since(started).Milliseconds()

    fmt.Println("article\\troot_category\\teffective_distance")
    for _, row := range rows {{
        fmt.Printf("%s\\t%s\\t%.6f\\n", row.Article, row.Root, row.Distance)
    }}

    fmt.Fprintf(os.Stderr, "mode=accumulated_go_wam\\n")
    fmt.Fprintf(os.Stderr, "load_ms=%d\\n", loadMs)
    fmt.Fprintf(os.Stderr, "query_ms=%d\\n", queryMs)
    fmt.Fprintf(os.Stderr, "aggregation_ms=%d\\n", aggregationMs)
    fmt.Fprintf(os.Stderr, "total_ms=%d\\n", totalMs)
    fmt.Fprintf(os.Stderr, "root_count=%d\\n", rootCount)
    fmt.Fprintf(os.Stderr, "tuple_count=%d\\n", tupleCount)
    fmt.Fprintf(os.Stderr, "article_count=%d\\n", len(rows))
}}
"""
    (project_dir / "main.go").write_text(main_code, encoding="utf-8")


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


def build_rust_matrix_effective_distance(root: Path, mode: str, kernel_mode: str) -> list[str]:
    project_dir = root / f"rust_{mode}_{kernel_mode}"
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_RUST_MATRIX_GENERATOR),
            "--",
            str(DEFAULT_FACTS),
            str(project_dir),
            "accumulated",
            mode,
            kernel_mode,
        ],
        cwd=ROOT,
    )
    run_command(["cargo", "build", "--release"], cwd=project_dir)
    return [str(project_dir / "target" / "release" / RUST_MATRIX_EXE)]


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
        elif target == "rust-interp-ffi":
            commands[target] = build_rust_matrix_effective_distance(root, "interpreter", "kernels_on")
        elif target == "rust-lowered-ffi":
            commands[target] = build_rust_matrix_effective_distance(root, "functions", "kernels_on")
    return commands


def normalize_output(output: str) -> str:
    return normalize_three_column_float_rows(output, decimals=9)


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target.startswith("prolog-") or target.startswith("wam-") or target.startswith("clojure-wam-"):
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
    runnable = []
    for target in resolved:
        if TARGETS[target].category == "hybrid-wam-scaffold":
            print(
                f"skip {target}: scaffold-only target; no result-producing effective-distance runner yet",
                file=sys.stderr,
            )
            continue
        runnable.append(target)
    return available_targets(runnable)


def main() -> int:
    args = parse_args()
    if args.list_targets:
        print(list_targets_text())
        return 0

    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    validate_termux_scales(scales, args.allow_large_termux_scales)
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
                elif target == "wam-rust-seeded-no-kernels":
                    command = build_wam_rust_effective_distance(temp_root, scale, "seeded_no_kernels")
                elif target == "wam-rust-accumulated-no-kernels":
                    command = build_wam_rust_effective_distance(temp_root, scale, "accumulated_no_kernels")
                elif target == "go-wam-accumulated":
                    command = build_wam_go_effective_distance(temp_root, scale, "kernels_on")
                elif target == "go-wam-accumulated-no-kernels":
                    command = build_wam_go_effective_distance(temp_root, scale, "kernels_off")
                elif target == "clojure-wam-accumulated":
                    command = build_wam_clojure_effective_distance(temp_root, scale, "accumulated", "kernels_on")
                elif target == "clojure-wam-accumulated-no-kernels":
                    command = build_wam_clojure_effective_distance(temp_root, scale, "accumulated", "kernels_off")
                elif target == "clojure-wam-accumulated-artifact":
                    command = build_wam_clojure_effective_distance(
                        temp_root, scale, "accumulated", "kernels_on", "artifact"
                    )
                elif target == "clojure-wam-accumulated-no-kernels-artifact":
                    command = build_wam_clojure_effective_distance(
                        temp_root, scale, "accumulated", "kernels_off", "artifact"
                    )
                elif target == "clojure-wam-seeded":
                    command = build_wam_clojure_effective_distance(temp_root, scale, "seeded", "kernels_on")
                elif target == "clojure-wam-seeded-no-kernels":
                    command = build_wam_clojure_effective_distance(temp_root, scale, "seeded", "kernels_off")
                elif target == "clojure-wam-seeded-artifact":
                    command = build_wam_clojure_effective_distance(
                        temp_root, scale, "seeded", "kernels_on", "artifact"
                    )
                elif target == "clojure-wam-seeded-no-kernels-artifact":
                    command = build_wam_clojure_effective_distance(
                        temp_root, scale, "seeded", "kernels_off", "artifact"
                    )
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
