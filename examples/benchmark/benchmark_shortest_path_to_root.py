#!/usr/bin/env python3
"""
Benchmark shortest-path-to-root on the C# query engine with two runtime modes:

  - all : path-aware counted closure preserving all simple paths
  - min : mode-directed tabling with minimum-path pruning

This isolates the benefit of the new min-tabling runtime support while
keeping the aggregation workload identical. Both modes compute the same
final shortest-path answers; they differ only in how much recursive work
they retain before aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
QRY_RUNTIME = ROOT / "src" / "unifyweaver" / "targets" / "csharp_query_runtime" / "QueryRuntime.cs"

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

PROGRAM = """\
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnifyWeaver.QueryRuntime;

class Program
{
    const string ROOT_CATEGORY = "Physics";
    const int MAX_DEPTH = 10;

    static void Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: program <all|min> <category_parent.tsv> <article_category.tsv>");
            Environment.Exit(1);
        }

        var modeName = args[0];
        var mode = modeName switch
        {
            "all" => TableMode.All,
            "min" => TableMode.Min,
            _ => throw new ArgumentException($"Unknown mode: {modeName}")
        };

        var swTotal = Stopwatch.StartNew();
        var swLoad = Stopwatch.StartNew();

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var predId = new PredicateId("category_ancestor", 3);
        var articleCategories = new Dictionary<string, List<string>>();
        var rootCategories = new HashSet<string>(StringComparer.Ordinal);

        foreach (var (line, i) in File.ReadLines(args[1]).Select((l, i) => (l, i)))
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

        foreach (var (line, i) in File.ReadLines(args[2]).Select((l, i) => (l, i)))
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

        foreach (var categories in articleCategories.Values)
        {
            foreach (var category in categories)
            {
                if (category == ROOT_CATEGORY)
                {
                    rootCategories.Add(category);
                }
            }
        }

        swLoad.Stop();

        var plan = new QueryPlan(
            predId,
            new PathAwareTransitiveClosureNode(edgeId, predId, 1, 1, MAX_DEPTH, mode),
            true,
            new int[] { 0 }
        );

        var uniqueSeeds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var categories in articleCategories.Values)
        {
            foreach (var category in categories)
            {
                uniqueSeeds.Add(category);
            }
        }

        var seedParams = uniqueSeeds
            .OrderBy(category => category, StringComparer.Ordinal)
            .Select(category => new object[] { category })
            .ToList();

        var swQuery = Stopwatch.StartNew();
        var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: false));
        var rows = executor.Execute(plan, seedParams).ToList();
        swQuery.Stop();

        var swAgg = Stopwatch.StartNew();
        var ancestorIndex = new Dictionary<string, List<(string Ancestor, int Hops)>>(StringComparer.Ordinal);
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

        var results = new List<(int Distance, string Article)>();
        foreach (var article in articleCategories.Keys.OrderBy(x => x, StringComparer.Ordinal))
        {
            int? best = null;
            foreach (var category in articleCategories[article])
            {
                if (category == ROOT_CATEGORY)
                {
                    best = best is null ? 1 : Math.Min(best.Value, 1);
                }

                if (!ancestorIndex.TryGetValue(category, out var ancestors))
                {
                    continue;
                }

                foreach (var (ancestor, hops) in ancestors)
                {
                    if (ancestor != ROOT_CATEGORY)
                    {
                        continue;
                    }

                    var dist = hops + 1;
                    best = best is null ? dist : Math.Min(best.Value, dist);
                }
            }

            if (best is not null)
            {
                results.Add((best.Value, article));
            }
        }
        swAgg.Stop();
        swTotal.Stop();

        results.Sort((a, b) =>
        {
            var cmp = a.Distance.CompareTo(b.Distance);
            return cmp != 0 ? cmp : string.Compare(a.Article, b.Article, StringComparison.Ordinal);
        });

        Console.WriteLine("article\\troot_category\\tshortest_path");
        foreach (var (distance, article) in results)
        {
            Console.WriteLine($"{article}\\t{ROOT_CATEGORY}\\t{distance}");
        }

        Console.Error.WriteLine($"mode={modeName}");
        Console.Error.WriteLine($"load_ms={swLoad.ElapsedMilliseconds}");
        Console.Error.WriteLine($"query_ms={swQuery.ElapsedMilliseconds}");
        Console.Error.WriteLine($"aggregation_ms={swAgg.ElapsedMilliseconds}");
        Console.Error.WriteLine($"total_ms={swTotal.ElapsedMilliseconds}");
        Console.Error.WriteLine($"seed_count={seedParams.Count}");
        Console.Error.WriteLine($"tuple_count={rows.Count}");
        Console.Error.WriteLine($"article_count={results.Count}");
    }
}
"""


@dataclass
class RunResult:
    scale: str
    mode: str
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


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


def build_harness(root: Path) -> list[str]:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet not found")

    project_dir = root / "shortest_path_to_root"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_shortest_path.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_shortest_path.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_shortest_path.dll")]


def benchmark_mode(command: list[str], scale: str, mode: str, repetitions: int) -> RunResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [mode, str(edge_path), str(article_path)])
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
    return RunResult(scale=scale, mode=mode, times=times, stdout_sha256=digest, row_count=rows, stderr=last_stderr)


def print_summary(results: list[RunResult]) -> None:
    by_scale: dict[str, list[RunResult]] = {}
    for result in results:
        by_scale.setdefault(result.scale, []).append(result)

    print("scale\tmode\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale in sorted(by_scale.keys(), key=scale_sort_key):
        for result in sorted(by_scale[scale], key=lambda item: item.mode):
            print(
                f"{scale}\t{result.mode}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}\t"
                f"{result.row_count}\t{result.stdout_sha256[:12]}"
            )

        all_mode = next((item for item in by_scale[scale] if item.mode == "all"), None)
        min_mode = next((item for item in by_scale[scale] if item.mode == "min"), None)
        if all_mode and min_mode:
            status = "match" if all_mode.stdout_sha256 == min_mode.stdout_sha256 else "DIFFERENT"
            print(f"{scale}\tall_vs_min\t{status}")
            print(f"{scale}\tspeedup_min_vs_all\t{all_mode.median / min_mode.median:.2f}x")

        if min_mode and min_mode.stderr:
            phase_lines = [line.strip() for line in min_mode.stderr.splitlines() if "=" in line]
            if phase_lines:
                print(f"{scale}\tmin_metrics\t" + " ".join(phase_lines))
        if all_mode and all_mode.stderr:
            phase_lines = [line.strip() for line in all_mode.stderr.splitlines() if "=" in line]
            if phase_lines:
                print(f"{scale}\tall_metrics\t" + " ".join(phase_lines))


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-shortest-path-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-shortest-path-")
        temp_root = Path(temp_ctx.name)

    try:
        command = build_harness(temp_root)
        results: list[RunResult] = []
        for scale in scales:
            for mode in ("all", "min"):
                results.append(benchmark_mode(command, scale, mode, args.repetitions))
        print_summary(results)
        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}")
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
