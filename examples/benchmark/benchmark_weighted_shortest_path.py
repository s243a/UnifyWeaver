#!/usr/bin/env python3
"""
Benchmark weighted shortest-path-to-root on the C# query engine with two
PathAwareAccumulationNode runtime modes:

  - all : preserve all simple weighted paths
  - min : mode-directed pruning on the accumulator

The harness derives a positive source weight from category out-degree so the
recursive increment remains monotone and min-pruning is sound.
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
using System.Globalization;
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
        var predId = new PredicateId("category_weighted_shortest", 3);
        var weightId = new PredicateId("category_weight", 2);
        var articleCategories = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var outDegree = new Dictionary<string, int>(StringComparer.Ordinal);
        var sourceNodes = new HashSet<string>(StringComparer.Ordinal);

        foreach (var (line, i) in File.ReadLines(args[1]).Select((l, i) => (l, i)))
        {
            if (i == 0 && (line.StartsWith("child") || line.StartsWith("article")))
            {
                continue;
            }

            var parts = line.Split('\\t', 2);
            if (parts.Length != 2)
            {
                continue;
            }

            provider.AddFact(edgeId, parts[0], parts[1]);
            sourceNodes.Add(parts[0]);
            outDegree[parts[0]] = outDegree.TryGetValue(parts[0], out var degree) ? degree + 1 : 1;
        }

        foreach (var source in sourceNodes)
        {
            outDegree.TryGetValue(source, out var degree);
            var weight = 1.0 + Math.Log(Math.Max(1, degree), 5.0);
            provider.AddFact(weightId, source, weight);
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

        swLoad.Stop();

        var plan = new QueryPlan(
            predId,
            new PathAwareAccumulationNode(
                edgeId,
                predId,
                weightId,
                new ColumnExpression(3),
                new BinaryArithmeticExpression(
                    ArithmeticBinaryOperator.Add,
                    new ColumnExpression(2),
                    new ColumnExpression(3)),
                MAX_DEPTH,
                mode),
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
        var ancestorIndex = new Dictionary<string, List<(string Ancestor, double Weight)>>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            var source = row[0]?.ToString() ?? "";
            var ancestor = row[1]?.ToString() ?? "";
            var weight = Convert.ToDouble(row[2], CultureInfo.InvariantCulture);
            if (!ancestorIndex.TryGetValue(source, out var bucket))
            {
                bucket = new List<(string, double)>();
                ancestorIndex[source] = bucket;
            }

            bucket.Add((ancestor, weight));
        }

        var results = new List<(double Distance, string Article)>();
        foreach (var article in articleCategories.Keys.OrderBy(x => x, StringComparer.Ordinal))
        {
            double? best = null;
            foreach (var category in articleCategories[article])
            {
                if (category == ROOT_CATEGORY)
                {
                    best = best is null ? 0.0 : Math.Min(best.Value, 0.0);
                }

                if (!ancestorIndex.TryGetValue(category, out var ancestors))
                {
                    continue;
                }

                foreach (var (ancestor, weight) in ancestors)
                {
                    if (ancestor != ROOT_CATEGORY)
                    {
                        continue;
                    }

                    best = best is null ? weight : Math.Min(best.Value, weight);
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

        Console.WriteLine("article\\troot_category\\tweighted_shortest_path");
        foreach (var (distance, article) in results)
        {
            Console.WriteLine($"{article}\\t{ROOT_CATEGORY}\\t{distance.ToString("G17", CultureInfo.InvariantCulture)}");
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
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def scale_sort_key(scale: str) -> tuple[int, str]:
    digits = "".join(ch for ch in scale if ch.isdigit())
    suffix = "".join(ch for ch in scale if not ch.isdigit())
    if not digits:
        return (0, scale)
    value = int(digits)
    if suffix.lower() == "k":
        value *= 1000
    return (value, scale)


def build_harness(root: Path) -> list[str]:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet not found")

    project_dir = root / "weighted_shortest_path"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_weighted_shortest_path.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_weighted_shortest_path.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_weighted_shortest_path.dll")]


def benchmark_mode(command: list[str], scale: str, mode: str, repetitions: int) -> RunResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [mode, str(edge_path), str(article_path)])
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        stdout = result.stdout
        stderr = result.stderr

    row_count = max(0, len(stdout.splitlines()) - 1)
    digest = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    return RunResult(scale, mode, times, digest, row_count, stderr)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-weighted-shortest-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-weighted-shortest-")
        temp_root = Path(temp_ctx.name)

    try:
        command = build_harness(temp_root)
        results: list[RunResult] = []
        for scale in scales:
            results.append(benchmark_mode(command, scale, "all", args.repetitions))
            results.append(benchmark_mode(command, scale, "min", args.repetitions))

        print("scale\tmode\tmedian_s\tmin_s\tmax_s\trows")
        grouped: dict[str, dict[str, RunResult]] = {}
        for result in results:
            grouped.setdefault(result.scale, {})[result.mode] = result
            print(
                f"{result.scale}\t{result.mode}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}\t{result.row_count}"
            )

        for scale in sorted(grouped.keys(), key=scale_sort_key):
            all_result = grouped[scale]["all"]
            min_result = grouped[scale]["min"]
            outputs_match = "match" if all_result.stdout_sha256 == min_result.stdout_sha256 else "DIFFERENT"
            speedup = all_result.median / min_result.median if min_result.median else float("inf")
            print(f"{scale}\tall_vs_min\t{outputs_match}")
            print(f"{scale}\tspeedup_min_vs_all\t{speedup:.2f}x")
            if all_result.stderr:
                print(f"{scale}\tall_metrics\t{' '.join(line.strip() for line in all_result.stderr.splitlines())}")
            if min_result.stderr:
                print(f"{scale}\tmin_metrics\t{' '.join(line.strip() for line in min_result.stderr.splitlines())}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
