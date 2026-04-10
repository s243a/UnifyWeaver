#!/usr/bin/env python3
"""
Measure generic scan-family materialization planning in the C# query runtime.

This harness exercises representative scan-heavy plans against the benchmark
TSVs using streamed relation sources inside a temporary C# project:

- `scan`: plain `RelationScanNode`
- `pattern`: `PatternScanNode` with a bound category
- `join`: `KeyJoinNode` over article/category and category-parent scans
- `negation`: unary negation over scanned support relations
- `aggregate`: grouped count aggregate over a scanned relation

It is intended as a focused validation tool for the scan materialization
planner rather than a cross-target benchmark.
"""

from __future__ import annotations

import argparse
import os
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
    static ScanRelationRetentionStrategy ParseScanStrategy(string value) => value.ToLowerInvariant() switch
    {
        "streaming" => ScanRelationRetentionStrategy.StreamingDirect,
        "replayable" => ScanRelationRetentionStrategy.ReplayableBuffer,
        "external" => ScanRelationRetentionStrategy.ExternalMaterialized,
        _ => ScanRelationRetentionStrategy.Auto,
    };

    static List<string[]> ReadDelimitedRows(string path, int expectedWidth)
    {
        return File.ReadLines(path)
            .Skip(1)
            .Select(line => line.Split('	'))
            .Where(parts => parts.Length == expectedWidth)
            .ToList();
    }

    static void RegisterBinaryRelation(InMemoryRelationProvider provider, PredicateId predicate, string path)
    {
        provider.RegisterDelimitedSource(predicate, new DelimitedRelationSource(path, '	', 1, 2));
        foreach (var parts in ReadDelimitedRows(path, 2))
        {
            provider.AddFact(predicate, parts[0], parts[1]);
        }
    }

    static string SummarizeStrategies(QueryExecutionTrace trace, Func<QueryStrategyTrace, bool> predicate)
    {
        return string.Join(
            "|",
            trace.SnapshotStrategies()
                .Where(predicate)
                .OrderBy(s => s.NodeId)
                .ThenBy(s => s.Strategy, StringComparer.Ordinal)
                .Select(s => $"{s.NodeType}:{s.Strategy}={s.Count}"));
    }

    static string SummarizePhases(QueryExecutionTrace trace, string prefix)
    {
        return string.Join(
            "|",
            trace.SnapshotPhases()
                .Where(p => p.Phase.StartsWith(prefix, StringComparison.Ordinal))
                .GroupBy(p => p.Phase, StringComparer.Ordinal)
                .OrderBy(group => group.Key, StringComparer.Ordinal)
                .Select(group => $"{group.Key}:{group.Sum(p => p.Elapsed.TotalMilliseconds).ToString("F3", CultureInfo.InvariantCulture)}"));
    }

    static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: program <scan|pattern|join|negation|aggregate> <category_parent.tsv> <article_category.tsv>");
            return 1;
        }

        var mode = args[0];
        var edgePath = args[1];
        var articlePath = args[2];
        var traceEnabled = Environment.GetEnvironmentVariable("UNIFYWEAVER_BENCH_TRACE") == "1";
        var scanStrategy = ParseScanStrategy(Environment.GetEnvironmentVariable("UNIFYWEAVER_SCAN_RETENTION_STRATEGY") ?? "auto");

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var articleId = new PredicateId("article_category", 2);
        var blockedId = new PredicateId("blocked_article_category", 2);

        RegisterBinaryRelation(provider, edgeId, edgePath);
        RegisterBinaryRelation(provider, articleId, articlePath);

        var articleRows = ReadDelimitedRows(articlePath, 2);
        var patternCategory = articleRows.Count > 0
            ? articleRows[0][1]
            : string.Empty;

        var blockedPath = Path.Combine(Path.GetTempPath(), $"uw-scan-{Guid.NewGuid():N}.tsv");
        using (var writer = new StreamWriter(blockedPath, false))
        {
            writer.WriteLine("article	category");
            for (var i = 0; i < articleRows.Count; i += 3)
            {
                writer.Write(articleRows[i][0]);
                writer.Write('	');
                writer.WriteLine(articleRows[i][1]);
            }
        }

        try
        {
            RegisterBinaryRelation(provider, blockedId, blockedPath);

            var scanOutId = new PredicateId("scan_rows", 2);
            var patternOutId = new PredicateId("pattern_rows", 2);
            var joinOutId = new PredicateId("join_rows", 4);
            var negationOutId = new PredicateId("negation_rows", 2);
            var aggregateOutId = new PredicateId("aggregate_rows", 2);

            QueryPlan plan = mode switch
            {
                "scan" => new QueryPlan(
                    scanOutId,
                    new RelationScanNode(articleId)),
                "pattern" => new QueryPlan(
                    patternOutId,
                    new PatternScanNode(articleId, new object[] { Wildcard.Value, patternCategory })),
                "join" => new QueryPlan(
                    joinOutId,
                    new KeyJoinNode(
                        new RelationScanNode(articleId),
                        new RelationScanNode(edgeId),
                        new int[] { 1 },
                        new int[] { 0 },
                        2,
                        2,
                        4)),
                "negation" => new QueryPlan(
                    negationOutId,
                    new NegationNode(
                        new RelationScanNode(articleId),
                        blockedId,
                        tuple => new object[] { tuple[0], tuple[1] })),
                "aggregate" => new QueryPlan(
                    aggregateOutId,
                    new AggregateNode(
                        new UnitNode(0),
                        articleId,
                        AggregateOperation.Count,
                        _ => new object[] { Wildcard.Value, Wildcard.Value },
                        new int[] { 1 },
                        0,
                        2)),
                _ => throw new ArgumentException($"Unknown mode: {mode}")
            };

            var trace = traceEnabled ? new QueryExecutionTrace() : null;
            var executor = new QueryExecutor(provider, new QueryExecutorOptions(
                ReuseCaches: false,
                ScanRelationRetentionStrategy: scanStrategy));

            var stopwatch = Stopwatch.StartNew();
            var rows = executor.Execute(plan, trace: trace).ToList();
            stopwatch.Stop();

            Console.Error.WriteLine($"mode={mode}");
            Console.Error.WriteLine($"scan_retention_strategy_setting={scanStrategy}");
            Console.Error.WriteLine($"pattern_category={patternCategory}");
            Console.Error.WriteLine($"row_count={rows.Count}");
            Console.Error.WriteLine($"elapsed_ms={stopwatch.ElapsedMilliseconds}");

            if (traceEnabled && trace is not null)
            {
                Console.Error.WriteLine(
                    "scan_planner_strategies=" +
                    SummarizeStrategies(
                        trace,
                        strategy => strategy.Strategy.Contains("ScanRelationRetention", StringComparison.Ordinal) ||
                                    strategy.Strategy.Contains("ScanMaterializationPlan", StringComparison.Ordinal)));

                Console.Error.WriteLine(
                    "scan_operator_strategies=" +
                    SummarizeStrategies(
                        trace,
                        strategy => strategy.Strategy.StartsWith("KeyJoin", StringComparison.Ordinal) ||
                                    strategy.Strategy.StartsWith("ScanRelationRetention", StringComparison.Ordinal) ||
                                    strategy.Strategy.StartsWith("ScanMaterializationPlan", StringComparison.Ordinal)));

                Console.Error.WriteLine("scan_phase_summary=" + SummarizePhases(trace, "scan_"));
            }

            return 0;
        }
        finally
        {
            try { File.Delete(blockedPath); } catch { }
        }
    }
}
"""


@dataclass
class BenchResult:
    scale: str
    mode: str
    strategy: str
    times: list[float]
    stderr: str

    @property
    def median(self) -> float:
        return statistics.median(self.times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="300,10k")
    parser.add_argument("--modes", default="scan,pattern,join,negation,aggregate")
    parser.add_argument("--strategies", default="auto,streaming,replayable,external")
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

    project_dir = root / "scan_materialization"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_scan_materialization.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_scan_materialization.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_scan_materialization.dll")]


def parse_metrics(stderr: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in stderr.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def benchmark_mode(command: list[str], scale: str, mode: str, strategy: str, repetitions: int) -> BenchResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"
    env = os.environ.copy()
    env["UNIFYWEAVER_BENCH_TRACE"] = "1"
    env["UNIFYWEAVER_SCAN_RETENTION_STRATEGY"] = strategy

    times: list[float] = []
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [mode, str(edge_path), str(article_path)], env=env)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        stderr = result.stderr

    return BenchResult(scale=scale, mode=mode, strategy=strategy, times=times, stderr=stderr)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    strategies = [part.strip() for part in args.strategies.split(",") if part.strip()]

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-scan-materialization-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-scan-materialization-")
        temp_root = Path(temp_ctx.name)

    try:
        command = build_harness(temp_root)
        results: list[BenchResult] = []
        for scale in scales:
            for mode in modes:
                for strategy in strategies:
                    results.append(benchmark_mode(command, scale, mode, strategy, args.repetitions))

        print("scale	mode	strategy	median_s	min_s	max_s	rows	planner	phases")
        grouped: dict[tuple[str, str], dict[str, BenchResult]] = {}
        for result in sorted(results, key=lambda item: (scale_sort_key(item.scale), item.mode, item.strategy)):
            grouped.setdefault((result.scale, result.mode), {})[result.strategy] = result
            metrics = parse_metrics(result.stderr)
            planner = metrics.get("scan_planner_strategies", "")
            phases = metrics.get("scan_phase_summary", "")
            rows = metrics.get("row_count", "")
            print(
                f"{result.scale}	{result.mode}	{result.strategy}	{result.median:.3f}	"
                f"{min(result.times):.3f}	{max(result.times):.3f}	{rows}	{planner}	{phases}"
            )

        for scale, mode in sorted(grouped.keys(), key=lambda item: (scale_sort_key(item[0]), item[1])):
            by_strategy = grouped[(scale, mode)]
            auto = by_strategy.get("auto")
            best = min(by_strategy.values(), key=lambda item: item.median)
            if auto is not None:
                print(f"{scale}	{mode}	best_strategy	{best.strategy}")
                ratio = auto.median / best.median if best.median else float('inf')
                print(f"{scale}	{mode}	auto_vs_best	{ratio:.2f}x")
                auto_metrics = parse_metrics(auto.stderr)
                print(f"{scale}	{mode}	auto_planner	{auto_metrics.get('scan_planner_strategies', '')}")

        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
