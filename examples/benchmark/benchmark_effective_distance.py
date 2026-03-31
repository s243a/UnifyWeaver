#!/usr/bin/env python3
"""
Benchmark the effective-distance workload for the C# query engine and the
compiled DFS pipelines.

Default targets:
  - csharp-query  : current C# query runtime using PathAwareTransitiveClosureNode
  - csharp-dfs    : generated C# DFS pipeline
  - rust-dfs      : generated Rust DFS pipeline

The script builds temporary binaries locally, runs them against one or more
benchmark scales, and reports median wall-clock times plus output agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
QRY_RUNTIME = ROOT / "src" / "unifyweaver" / "targets" / "csharp_query_runtime" / "QueryRuntime.cs"

QE_PROGRAM = """\
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnifyWeaver.QueryRuntime;

class Program
{
    const string ROOT_CATEGORY = "Physics";
    const double N = 5.0;
    const int MAX_DEPTH = 10;

    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("Usage: program <category_parent.tsv> <article_category.tsv>");
            Environment.Exit(1);
        }

        var swTotal = Stopwatch.StartNew();

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var predId = new PredicateId("category_ancestor", 3);
        var articleCategories = new Dictionary<string, List<string>>();

        var swLoad = Stopwatch.StartNew();
        foreach (var (line, i) in File.ReadLines(args[0]).Select((l, i) => (l, i)))
        {
            if (i == 0 && (line.StartsWith("child") || line.StartsWith("article")))
            {
                continue;
            }

            var parts = line.Split('\\t', 2);
            if (parts.Length == 2)
            {
                provider.AddFact(edgeId, parts[0], parts[1]);
            }
        }

        foreach (var (line, i) in File.ReadLines(args[1]).Select((l, i) => (l, i)))
        {
            if (i == 0 && (line.StartsWith("article") || line.StartsWith("child")))
            {
                continue;
            }

            var parts = line.Split('\\t', 2);
            if (parts.Length != 2)
            {
                continue;
            }

            if (!articleCategories.TryGetValue(parts[0], out var categories))
            {
                categories = new List<string>();
                articleCategories[parts[0]] = categories;
            }

            categories.Add(parts[1]);
        }
        swLoad.Stop();

        var plan = new QueryPlan(
            predId,
            new PathAwareTransitiveClosureNode(edgeId, predId, 1, 1, MAX_DEPTH),
            true,
            new int[] { 0 }
        );

        var uniqueSeeds = new HashSet<string>();
        foreach (var categories in articleCategories.Values)
        {
            foreach (var category in categories)
            {
                uniqueSeeds.Add(category);
            }
        }

        var seedParams = uniqueSeeds.Select(category => new object[] { category }).ToList();

        var swExec = Stopwatch.StartNew();
        var executor = new QueryExecutor(provider);
        var rows = executor.Execute(plan, seedParams).ToList();
        swExec.Stop();

        var swAgg = Stopwatch.StartNew();
        var ancestorIndex = new Dictionary<string, List<(string Ancestor, int Hops)>>();
        foreach (var row in rows)
        {
            var source = row[0]?.ToString() ?? "";
            var ancestor = row[1]?.ToString() ?? "";
            var hops = Convert.ToInt32(row[2]);
            if (!ancestorIndex.TryGetValue(source, out var bucket))
            {
                bucket = new List<(string, int)>();
                ancestorIndex[source] = bucket;
            }

            bucket.Add((ancestor, hops));
        }

        var results = new List<(double Deff, string Article)>();
        foreach (var article in articleCategories.Keys.OrderBy(x => x))
        {
            var allHops = new List<int>();
            foreach (var category in articleCategories[article])
            {
                if (category == ROOT_CATEGORY)
                {
                    allHops.Add(1);
                }

                if (ancestorIndex.TryGetValue(category, out var ancestors))
                {
                    foreach (var (ancestor, hops) in ancestors)
                    {
                        if (ancestor == ROOT_CATEGORY)
                        {
                            allHops.Add(hops + 1);
                        }
                    }
                }
            }

            if (allHops.Count > 0)
            {
                results.Add((ComputeDeff(allHops), article));
            }
        }
        swAgg.Stop();

        results.Sort((a, b) =>
        {
            var cmp = a.Deff.CompareTo(b.Deff);
            return cmp != 0 ? cmp : string.Compare(a.Article, b.Article, StringComparison.Ordinal);
        });

        Console.WriteLine("article\\troot_category\\teffective_distance");
        foreach (var (deff, article) in results)
        {
            Console.WriteLine($"{article}\\t{ROOT_CATEGORY}\\t{deff:F6}");
        }

        swTotal.Stop();
        Console.Error.WriteLine($"load_ms={swLoad.ElapsedMilliseconds}");
        Console.Error.WriteLine($"query_ms={swExec.ElapsedMilliseconds}");
        Console.Error.WriteLine($"aggregation_ms={swAgg.ElapsedMilliseconds}");
        Console.Error.WriteLine($"total_ms={swTotal.ElapsedMilliseconds}");
        Console.Error.WriteLine($"seed_count={seedParams.Count}");
        Console.Error.WriteLine($"tuple_count={rows.Count}");
        Console.Error.WriteLine($"article_count={results.Count}");
    }

    static double ComputeDeff(List<int> hops)
    {
        if (hops.Count == 0)
        {
            return double.PositiveInfinity;
        }

        var weightSum = hops.Sum(h => Math.Pow(h, -N));
        return weightSum > 0 ? Math.Pow(weightSum, -1.0 / N) : double.PositiveInfinity;
    }
}
"""

CSHARP_PROJECT = """\
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
"""


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
        default="csharp-query,csharp-dfs,rust-dfs",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs",
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


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def available_targets(requested: list[str]) -> list[str]:
    targets: list[str] = []
    for target in requested:
        if target == "rust-dfs" and shutil.which("rustc") is None:
            print("skip rust-dfs: rustc not found", file=sys.stderr)
            continue
        if target == "go-dfs" and shutil.which("go") is None:
            print("skip go-dfs: go not found", file=sys.stderr)
            continue
        if target.startswith("csharp-") and shutil.which("dotnet") is None:
            print(f"skip {target}: dotnet not found", file=sys.stderr)
            continue
        targets.append(target)
    return targets


def build_csharp_query(root: Path) -> list[str]:
    project_dir = root / "csharp_query"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(QE_PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_qe.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_qe.csproj", "-c", "Release"], cwd=project_dir)
    return [
        "dotnet",
        str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_qe.dll"),
    ]


def build_csharp_dfs(root: Path) -> list[str]:
    project_dir = root / "csharp_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = require_file(BENCH_DIR / "10k" / "pipelines" / "EffectiveDistance.cs")
    (project_dir / "Program.cs").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_dfs.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_dfs.csproj", "-c", "Release"], cwd=project_dir)
    return [
        "dotnet",
        str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_dfs.dll"),
    ]


def build_rust_dfs(root: Path) -> list[str]:
    project_dir = root / "rust_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = require_file(BENCH_DIR / "10k" / "pipelines" / "effective_distance.rs")
    binary = project_dir / "effective_distance_rust"
    run(["rustc", "-O", str(source), "-o", str(binary)])
    return [str(binary)]


def build_go_dfs(root: Path) -> list[str]:
    project_dir = root / "go_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = require_file(BENCH_DIR / "10k" / "pipelines" / "effective_distance.go")
    binary = project_dir / "effective_distance_go"
    run(["go", "build", "-o", str(binary), str(source)])
    return [str(binary)]


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [str(edge_path), str(article_path)])
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        last_stdout = result.stdout
        last_stderr = result.stderr

    lines = last_stdout.splitlines()
    header = lines[:1]
    body = sorted(lines[1:])
    normalized = "\n".join(header + body)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    rows = len(body)
    return RunResult(target=target, scale=scale, times=times, stdout_sha256=digest, row_count=rows, stderr=last_stderr)


def print_summary(results: list[RunResult]) -> None:
    by_scale: dict[str, list[RunResult]] = {}
    for result in results:
        by_scale.setdefault(result.scale, []).append(result)

    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale in sorted(by_scale.keys(), key=scale_sort_key):
        for result in sorted(by_scale[scale], key=lambda item: item.target):
            print(
                f"{scale}\t{result.target}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}\t"
                f"{result.row_count}\t{result.stdout_sha256[:12]}"
            )

        qe = next((item for item in by_scale[scale] if item.target == "csharp-query"), None)
        csharp_dfs = next((item for item in by_scale[scale] if item.target == "csharp-dfs"), None)
        rust_dfs = next((item for item in by_scale[scale] if item.target == "rust-dfs"), None)
        dfs_like = [item for item in by_scale[scale] if item.target != "csharp-query"]

        if len(dfs_like) > 1:
            dfs_hashes = {item.stdout_sha256 for item in dfs_like}
            status = "match" if len(dfs_hashes) == 1 else "MISMATCH"
            print(f"{scale}\tdfs_outputs\t{status}")

        if qe and csharp_dfs:
            same = "match" if qe.stdout_sha256 == csharp_dfs.stdout_sha256 else "DIFFERENT"
            print(f"{scale}\tquery_vs_csharp_dfs\t{same}")

        if qe and csharp_dfs:
            print(f"{scale}\tspeedup_vs_csharp_dfs\t{csharp_dfs.median / qe.median:.2f}x")
        if qe and rust_dfs:
            print(f"{scale}\tspeedup_vs_rust_dfs\t{rust_dfs.median / qe.median:.2f}x")

        if qe and qe.stderr:
            phase_lines = [line.strip() for line in qe.stderr.splitlines() if "=" in line]
            if phase_lines:
                print(f"{scale}\tcsharp-query-metrics\t" + " ".join(phase_lines))


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
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-effective-distance-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-effective-distance-")
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
        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
