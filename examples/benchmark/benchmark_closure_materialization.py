#!/usr/bin/env python3
"""
Measure generic closure-family materialization planning in the C# query runtime.

This harness exercises two representative closure paths against the benchmark
TSVs using streamed relation sources inside a temporary C# project:

- `seeded`: generic `TransitiveClosureNode` with seeded execution
- `weighted`: `PathAwareAccumulationNode` with a streamed auxiliary relation

It is intended as a focused validation tool for the closure materialization
planner rather than a cross-target benchmark.
"""

from __future__ import annotations

import argparse
import os
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

CSHARP_PROJECT = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
"""

PROGRAM = r"""using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using UnifyWeaver.QueryRuntime;

class Program
{
    static ClosureRelationRetentionStrategy ParseClosureStrategy(string value) => value.ToLowerInvariant() switch
    {
        "streaming" => ClosureRelationRetentionStrategy.StreamingDirect,
        "replayable" => ClosureRelationRetentionStrategy.ReplayableBuffer,
        "external" => ClosureRelationRetentionStrategy.ExternalMaterialized,
        _ => ClosureRelationRetentionStrategy.Auto,
    };

    static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: program <seeded|weighted> <category_parent.tsv> <article_category.tsv>");
            return 1;
        }

        var mode = args[0];
        var edgePath = args[1];
        var articlePath = args[2];
        var traceEnabled = Environment.GetEnvironmentVariable("UNIFYWEAVER_BENCH_TRACE") == "1";
        var closureStrategy = ParseClosureStrategy(Environment.GetEnvironmentVariable("UNIFYWEAVER_CLOSURE_RETENTION_STRATEGY") ?? "auto");

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var predicateId = new PredicateId(mode == "seeded" ? "category_ancestor_seeded" : "category_weighted_ancestor", 3);
        var weightId = new PredicateId("category_weight", 2);

        provider.RegisterDelimitedSource(edgeId, new DelimitedRelationSource(edgePath));

        var sourceNodes = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var line in File.ReadLines(edgePath).Skip(1))
        {
            var parts = line.Split('	', 2);
            if (parts.Length == 2)
            {
                sourceNodes.Add(parts[0]);
            }
        }

        var weightPath = Path.GetTempFileName();
        using (var writer = new StreamWriter(weightPath, false))
        {
            writer.WriteLine("child	weight");
            foreach (var source in sourceNodes)
            {
                writer.Write(source);
                writer.Write('	');
                writer.WriteLine("1");
            }
        }
        provider.RegisterDelimitedSource(weightId, new DelimitedRelationSource(weightPath));

        var parameters = File.ReadLines(articlePath)
            .Skip(1)
            .Select(line => line.Split('	', 2))
            .Where(parts => parts.Length == 2)
            .Select(parts => parts[1])
            .Distinct(StringComparer.Ordinal)
            .OrderBy(value => value, StringComparer.Ordinal)
            .Select(value => new object[] { value })
            .ToList();

        QueryPlan plan = mode switch
        {
            "seeded" => new QueryPlan(
                predicateId,
                new TransitiveClosureNode(edgeId, predicateId),
                true,
                new int[] { 0 }),
            "weighted" => new QueryPlan(
                predicateId,
                new PathAwareAccumulationNode(
                    edgeId,
                    predicateId,
                    weightId,
                    new ColumnExpression(3),
                    new BinaryArithmeticExpression(
                        ArithmeticBinaryOperator.Add,
                        new ColumnExpression(2),
                        new ColumnExpression(3)),
                    10),
                true,
                new int[] { 0 }),
            _ => throw new ArgumentException($"Unknown mode: {mode}")
        };

        var trace = traceEnabled ? new QueryExecutionTrace() : null;
        var executor = new QueryExecutor(provider, new QueryExecutorOptions(
            ReuseCaches: false,
            ClosureRelationRetentionStrategy: closureStrategy));

        var stopwatch = Stopwatch.StartNew();
        var rows = executor.Execute(plan, parameters, trace).ToList();
        stopwatch.Stop();

        Console.Error.WriteLine($"mode={mode}");
        Console.Error.WriteLine($"closure_strategy_setting={closureStrategy}");
        Console.Error.WriteLine($"seed_count={parameters.Count}");
        Console.Error.WriteLine($"row_count={rows.Count}");
        Console.Error.WriteLine($"elapsed_ms={stopwatch.ElapsedMilliseconds}");

        if (traceEnabled && trace is not null)
        {
            foreach (var strategy in trace.SnapshotStrategies().OrderBy(s => s.NodeId).ThenBy(s => s.Strategy, StringComparer.Ordinal))
            {
                Console.Error.WriteLine($"strategy_{strategy.Strategy}={strategy.Count}");
            }

            foreach (var phase in trace.SnapshotPhases().OrderBy(p => p.NodeId).ThenBy(p => p.Phase, StringComparer.Ordinal))
            {
                Console.Error.WriteLine($"phase_{phase.NodeType}_{phase.Phase}={phase.Elapsed.TotalMilliseconds.ToString("F3", CultureInfo.InvariantCulture)}");
            }
        }

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
    parser.add_argument("--scales", default="300,10k")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)


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

    project_dir = root / "closure_materialization"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_closure_materialization.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_closure_materialization.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_closure_materialization.dll")]


def benchmark_mode(command: list[str], scale: str, mode: str, repetitions: int) -> BenchResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"
    env = os.environ.copy()

    times: list[float] = []
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [mode, str(edge_path), str(article_path)], env=env)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        stderr = result.stderr

    return BenchResult(scale=scale, mode=mode, times=times, stderr=stderr)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-closure-materialization-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-closure-materialization-")
        temp_root = Path(temp_ctx.name)

    try:
        command = build_harness(temp_root)
        results: list[BenchResult] = []
        for scale in scales:
            results.append(benchmark_mode(command, scale, "seeded", args.repetitions))
            results.append(benchmark_mode(command, scale, "weighted", args.repetitions))

        print("scale	mode	median_s	min_s	max_s")
        for result in sorted(results, key=lambda item: (scale_sort_key(item.scale), item.mode)):
            print(
                f"{result.scale}	{result.mode}	{result.median:.3f}	"
                f"{min(result.times):.3f}	{max(result.times):.3f}"
            )
            for line in result.stderr.splitlines():
                if line.startswith(("mode=", "closure_strategy_setting=", "seed_count=", "row_count=", "elapsed_ms=")):
                    print(f"{result.scale}	{result.mode}_metrics	{line}")
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
