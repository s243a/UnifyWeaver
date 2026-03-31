#!/usr/bin/env python3
"""
Measure the runtime overhead of the generalized PathAwareAccumulationNode
relative to the counted PathAwareTransitiveClosureNode on the same graph.

The benchmark derives a unit weight for every source category from the
benchmark graph, so both plans enumerate the same seeded reachability
shape while the accumulation plan pays the extra auxiliary lookup and
arithmetic-expression cost.
"""

from __future__ import annotations

import argparse
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
using System.IO;
using System.Linq;
using UnifyWeaver.QueryRuntime;

class Program
{
    const int MAX_DEPTH = 10;

    static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: program <counted|weighted> <category_parent.tsv> <article_category.tsv>");
            return 1;
        }

        var mode = args[0];
        var edgePath = args[1];
        var articlePath = args[2];

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var countedId = new PredicateId("category_ancestor", 3);
        var weightedId = new PredicateId("category_weighted_ancestor", 3);
        var weightId = new PredicateId("category_weight", 2);
        var sourceNodes = new HashSet<string>();
        var articleCategories = new HashSet<string>();

        foreach (var (line, i) in File.ReadLines(edgePath).Select((l, i) => (l, i)))
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
        }

        foreach (var source in sourceNodes)
        {
            provider.AddFact(weightId, source, 1);
        }

        foreach (var (line, i) in File.ReadLines(articlePath).Select((l, i) => (l, i)))
        {
            if (i == 0 && (line.StartsWith("article") || line.StartsWith("child")))
            {
                continue;
            }

            var parts = line.Split('\\t', 2);
            if (parts.Length == 2)
            {
                articleCategories.Add(parts[1]);
            }
        }

        var parameters = articleCategories
            .OrderBy(x => x, StringComparer.Ordinal)
            .Select(category => new object[] { category })
            .ToList();

        QueryPlan plan = mode switch
        {
            "counted" => new QueryPlan(
                countedId,
                new PathAwareTransitiveClosureNode(edgeId, countedId, 1, 1, MAX_DEPTH),
                true,
                new int[] { 0 }),
            "weighted" => new QueryPlan(
                weightedId,
                new PathAwareAccumulationNode(
                    edgeId,
                    weightedId,
                    weightId,
                    new ColumnExpression(3),
                    new BinaryArithmeticExpression(
                        ArithmeticBinaryOperator.Add,
                        new ColumnExpression(2),
                        new ColumnExpression(3)),
                    MAX_DEPTH),
                true,
                new int[] { 0 }),
            _ => throw new ArgumentException($"Unknown mode: {mode}")
        };

        var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: false));
        var rows = executor.Execute(plan, parameters).ToList();
        Console.Error.WriteLine($"mode={mode}");
        Console.Error.WriteLine($"seed_count={parameters.Count}");
        Console.Error.WriteLine($"row_count={rows.Count}");
        return 0;
    }
}
"""


@dataclass
class BenchResult:
    scale: str
    mode: str
    times: list[float]
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

    project_dir = root / "path_aware_accumulation"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_path_aware.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_path_aware.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_path_aware.dll")]


def benchmark_mode(command: list[str], scale: str, mode: str, repetitions: int) -> BenchResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [mode, str(edge_path), str(article_path)])
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        stderr = result.stderr

    return BenchResult(scale=scale, mode=mode, times=times, stderr=stderr)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-path-aware-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-path-aware-")
        temp_root = Path(temp_ctx.name)
    try:
        command = build_harness(temp_root)
        results: list[BenchResult] = []
        for scale in scales:
            results.append(benchmark_mode(command, scale, "counted", args.repetitions))
            results.append(benchmark_mode(command, scale, "weighted", args.repetitions))

        print("scale\tmode\tmedian_s\tmin_s\tmax_s")
        grouped: dict[str, dict[str, BenchResult]] = {}
        for result in results:
            grouped.setdefault(result.scale, {})[result.mode] = result
            print(
                f"{result.scale}\t{result.mode}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}"
            )

        for scale in sorted(grouped.keys(), key=scale_sort_key):
            counted = grouped[scale]["counted"]
            weighted = grouped[scale]["weighted"]
            overhead = weighted.median / counted.median if counted.median else float("inf")
            print(f"{scale}\tweighted_overhead\t{overhead:.2f}x")
            if weighted.stderr:
                metrics = " ".join(line.strip() for line in weighted.stderr.splitlines())
                print(f"{scale}\tweighted_metrics\t{metrics}")

        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
