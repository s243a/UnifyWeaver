#!/usr/bin/env python3
"""
Measure generic scan-family materialization planning in the C# query runtime.

This harness exercises representative scan-heavy plans against the benchmark
TSVs using preloaded, streamed, binary-artifact, or prebuilt binary-artifact
relation sources inside a temporary C# project:

- `scan`: plain `RelationScanNode`
- `bound_scan`: parameterized `RelationScanNode` over one bound column
- `pattern`: `PatternScanNode` with a bound category
- `join`: `KeyJoinNode` over article/category and category-parent scans
- `nary_join`: `KeyJoinNode` over synthetic width-3 delimited relations
- `selective_join`: parameter seed joined against category-parent
- `negation`: unary negation over scanned support relations
- `aggregate`: grouped count aggregate over a scanned relation

It is intended as a focused validation tool for the scan materialization
planner rather than a cross-target benchmark. It can also exercise a simple
source-mode planner via `--source-modes auto,...`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
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

    static long CountDataRows(string path)
    {
        long count = 0;
        foreach (var _ in File.ReadLines(path).Skip(1))
        {
            count++;
        }

        return count;
    }

    static List<string[]> ReadDelimitedRows(string path, int expectedWidth)
    {
        return File.ReadLines(path)
            .Skip(1)
            .Select(line => line.Split('	'))
            .Where(parts => parts.Length == expectedWidth)
            .ToList();
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

    static bool IsArtifactBucketStrategy(QueryStrategyTrace strategy)
    {
        return strategy.Strategy.StartsWith("KeyJoinIndexedRelationProviderBucket", StringComparison.Ordinal) &&
               !string.Equals(strategy.Strategy, "KeyJoinIndexedRelationProviderBuckets", StringComparison.Ordinal);
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

    static string SummarizeSourceRegistrations(ConfiguredDelimitedRelationProvider configuredProvider)
    {
        return string.Join(
            "|",
            configuredProvider.SnapshotRegistrations()
                .GroupBy(registration => new
                {
                    registration.StorageKind,
                    registration.SourceMode,
                    registration.Arity
                })
                .OrderBy(group => group.Key.StorageKind, StringComparer.Ordinal)
                .ThenBy(group => group.Key.SourceMode.ToString(), StringComparer.Ordinal)
                .ThenBy(group => group.Key.Arity)
                .Select(group =>
                    $"{group.Key.StorageKind}:{RelationSourceModePolicy.ToConfigValue(group.Key.SourceMode)}:arity{group.Key.Arity}={group.Count()}"));
    }

    static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: program <scan|pattern|join|nary_join|negation|aggregate> <category_parent.tsv> <article_category.tsv>");
            return 1;
        }

        var mode = args[0];
        var edgePath = args[1];
        var articlePath = args[2];
        var traceEnabled = Environment.GetEnvironmentVariable("UNIFYWEAVER_BENCH_TRACE") == "1";
        var scanStrategy = ParseScanStrategy(Environment.GetEnvironmentVariable("UNIFYWEAVER_SCAN_RETENTION_STRATEGY") ?? "auto");
        if (!RelationSourceModePolicy.TryParse(Environment.GetEnvironmentVariable("UNIFYWEAVER_SCAN_SOURCE_MODE"), out var configuredSourceMode))
        {
            Console.Error.WriteLine($"Unknown UNIFYWEAVER_SCAN_SOURCE_MODE: {Environment.GetEnvironmentVariable("UNIFYWEAVER_SCAN_SOURCE_MODE")}");
            return 1;
        }

        var articleRowCount = CountDataRows(articlePath);
        var edgeRowCount = CountDataRows(edgePath);
        var sourceMode = RelationSourceModePolicy.ResolveScanBenchmarkMode(configuredSourceMode, mode, articleRowCount, edgeRowCount);

        var artifactDir = Environment.GetEnvironmentVariable("UNIFYWEAVER_SCAN_ARTIFACT_DIR");
        var configuredProvider = new ConfiguredDelimitedRelationProvider(sourceMode, artifactDir);
        var memoryProvider = configuredProvider.MemoryProvider;
        IRelationProvider provider = configuredProvider.Provider;
        var edgeId = new PredicateId("category_parent", 2);
        var articleId = new PredicateId("article_category", 2);
        var blockedId = new PredicateId("blocked_article_category", 2);
        var naryLeftId = new PredicateId("nary_left", 3);
        var naryRightId = new PredicateId("nary_right", 3);

        configuredProvider.RegisterBinaryRelation(edgeId, new DelimitedRelationSource(edgePath, '	', 1, 2), $"{edgeId.Name}_{edgeId.Arity}");
        configuredProvider.RegisterBinaryRelation(articleId, new DelimitedRelationSource(articlePath, '	', 1, 2), $"{articleId.Name}_{articleId.Arity}");

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

        var naryLeftPath = Path.Combine(Path.GetTempPath(), $"uw-scan-nary-left-{Guid.NewGuid():N}.tsv");
        var naryRightPath = Path.Combine(Path.GetTempPath(), $"uw-scan-nary-right-{Guid.NewGuid():N}.tsv");
        using (var writer = new StreamWriter(naryLeftPath, false))
        {
            writer.WriteLine("key	item	rank");
            writer.WriteLine("Keyboard	left-keyboard	1");
            writer.WriteLine("Laptop	left-laptop	2");
            writer.WriteLine("Mouse	left-mouse	3");
        }

        using (var writer = new StreamWriter(naryRightPath, false))
        {
            writer.WriteLine("key	item	score");
            writer.WriteLine("Keyboard	right-keyboard	10");
            writer.WriteLine("Laptop	right-laptop	20");
            writer.WriteLine("Mouse	right-mouse	30");
        }

        try
        {
            configuredProvider.RegisterBinaryRelation(blockedId, new DelimitedRelationSource(blockedPath, '	', 1, 2), $"{blockedId.Name}_{blockedId.Arity}");
            if (mode == "nary_join")
            {
                configuredProvider.RegisterDelimitedRelation(naryLeftId, new DelimitedRelationSource(naryLeftPath, '	', 1, 3), $"{naryLeftId.Name}_{naryLeftId.Arity}");
                configuredProvider.RegisterDelimitedRelation(naryRightId, new DelimitedRelationSource(naryRightPath, '	', 1, 3), $"{naryRightId.Name}_{naryRightId.Arity}");
            }

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
                "bound_scan" => new QueryPlan(
                    patternOutId,
                    new RelationScanNode(articleId),
                    InputPositions: new int[] { 1 }),
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
                "nary_join" => new QueryPlan(
                    joinOutId,
                    new KeyJoinNode(
                        new RelationScanNode(naryLeftId),
                        new RelationScanNode(naryRightId),
                        new int[] { 0 },
                        new int[] { 0 },
                        3,
                        3,
                        6)),
                "selective_join" => new QueryPlan(
                    joinOutId,
                    new KeyJoinNode(
                        new ParamSeedNode(new PredicateId("category_seed", 1), new int[] { 0 }, 1),
                        new RelationScanNode(edgeId),
                        new int[] { 0 },
                        new int[] { 0 },
                        1,
                        2,
                        3)),
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
            var parameters = mode == "bound_scan" || mode == "selective_join"
                ? new object[][] { new object[] { patternCategory } }
                : null;
            var rows = executor.Execute(plan, parameters: parameters, trace: trace).ToList();
            stopwatch.Stop();

            Console.Error.WriteLine($"mode={mode}");
            Console.Error.WriteLine($"source_mode={RelationSourceModePolicy.ToConfigValue(configuredSourceMode)}");
            Console.Error.WriteLine($"resolved_source_mode={RelationSourceModePolicy.ToConfigValue(sourceMode)}");
            Console.Error.WriteLine($"scan_retention_strategy_setting={scanStrategy}");
            Console.Error.WriteLine($"pattern_category={patternCategory}");
            Console.Error.WriteLine($"row_count={rows.Count}");
            Console.Error.WriteLine($"elapsed_ms={stopwatch.ElapsedMilliseconds}");
            Console.Error.WriteLine($"source_registrations={SummarizeSourceRegistrations(configuredProvider)}");

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
                                    strategy.Strategy.StartsWith("IndexedRelationProvider", StringComparison.Ordinal) ||
                                    strategy.Strategy.StartsWith("ScanRelationRetention", StringComparison.Ordinal) ||
                                    strategy.Strategy.StartsWith("ScanMaterializationPlan", StringComparison.Ordinal)));

                Console.Error.WriteLine("bucket_strategies=" + SummarizeStrategies(trace, IsArtifactBucketStrategy));
                Console.Error.WriteLine("scan_phase_summary=" + SummarizePhases(trace, "scan_"));
            }

            return 0;
        }
        finally
        {
            try { File.Delete(blockedPath); } catch { }
            try { File.Delete(naryLeftPath); } catch { }
            try { File.Delete(naryRightPath); } catch { }
            if (sourceMode != RelationSourceMode.ArtifactPrebuilt)
            {
                if (!string.IsNullOrWhiteSpace(configuredProvider.ArtifactDirectory))
                {
                    try { Directory.Delete(configuredProvider.ArtifactDirectory, recursive: true); } catch { }
                }
            }
        }
    }
}
"""


@dataclass
class BenchResult:
    scale: str
    mode: str
    source_mode: str
    resolved_source_mode: str
    strategy: str
    times: list[float]
    stderr: str

    @property
    def median(self) -> float:
        return statistics.median(self.times)


RUNTIME_CACHE_VERSION = hashlib.sha256(QRY_RUNTIME.read_bytes()).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="300,10k")
    parser.add_argument("--modes", default="scan,pattern,join,negation,aggregate")
    parser.add_argument("--source-modes", default="preload,artifact")
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


def select_planner_summary(metrics: dict[str, str]) -> str:
    operator = metrics.get("scan_operator_strategies", "")
    if "KeyJoinIndexedRelationProvider" in operator or "IndexedRelationProviderLookup" in operator:
        return operator
    return metrics.get("scan_planner_strategies", "") or operator


def benchmark_mode(
    command: list[str],
    scale: str,
    mode: str,
    source_mode: str,
    strategy: str,
    repetitions: int,
) -> BenchResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"
    env = os.environ.copy()
    env["UNIFYWEAVER_BENCH_TRACE"] = "1"
    env["UNIFYWEAVER_SCAN_RETENTION_STRATEGY"] = strategy
    env["UNIFYWEAVER_SCAN_SOURCE_MODE"] = source_mode
    if source_mode in {"artifact-prebuilt", "auto"}:
        artifact_dir = Path(tempfile.gettempdir()) / f"uw-scan-prebuilt-artifacts-{RUNTIME_CACHE_VERSION}" / scale
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env["UNIFYWEAVER_SCAN_ARTIFACT_DIR"] = str(artifact_dir)

    times: list[float] = []
    stderr = ""
    if repetitions > 0:
        run(command + [mode, str(edge_path), str(article_path)], env=env)
    for _ in range(repetitions):
        result = run(command + [mode, str(edge_path), str(article_path)], env=env)
        stderr = result.stderr
        metrics = parse_metrics(stderr)
        elapsed_ms = metrics.get("elapsed_ms")
        elapsed = float(elapsed_ms) / 1000.0 if elapsed_ms is not None else 0.0
        times.append(elapsed)
    metrics = parse_metrics(stderr)
    resolved_source_mode = metrics.get("resolved_source_mode", source_mode)

    return BenchResult(
        scale=scale,
        mode=mode,
        source_mode=source_mode,
        resolved_source_mode=resolved_source_mode,
        strategy=strategy,
        times=times,
        stderr=stderr,
    )


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    source_modes = [part.strip() for part in args.source_modes.split(",") if part.strip()]
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
                for source_mode in source_modes:
                    for strategy in strategies:
                        results.append(benchmark_mode(command, scale, mode, source_mode, strategy, args.repetitions))

        print("scale	mode	source_mode	resolved_source_mode	strategy	median_s	min_s	max_s	rows	source_registrations	planner	bucket_strategies	phases")
        grouped: dict[tuple[str, str, str], dict[str, BenchResult]] = {}
        by_source_mode: dict[tuple[str, str, str], BenchResult] = {}
        for result in sorted(results, key=lambda item: (scale_sort_key(item.scale), item.mode, item.source_mode, item.strategy)):
            grouped.setdefault((result.scale, result.mode, result.source_mode), {})[result.strategy] = result
            if result.strategy == "auto":
                by_source_mode[(result.scale, result.mode, result.source_mode)] = result
            metrics = parse_metrics(result.stderr)
            planner = select_planner_summary(metrics)
            bucket_strategies = metrics.get("bucket_strategies", "")
            phases = metrics.get("scan_phase_summary", "")
            rows = metrics.get("row_count", "")
            source_registrations = metrics.get("source_registrations", "")
            print(
                f"{result.scale}	{result.mode}	{result.source_mode}	{result.resolved_source_mode}	{result.strategy}	{result.median:.3f}	"
                f"{min(result.times):.3f}	{max(result.times):.3f}	{rows}	{source_registrations}	{planner}	{bucket_strategies}	{phases}"
            )

        for scale, mode, source_mode in sorted(grouped.keys(), key=lambda item: (scale_sort_key(item[0]), item[1], item[2])):
            by_strategy = grouped[(scale, mode, source_mode)]
            auto = by_strategy.get("auto")
            best = min(by_strategy.values(), key=lambda item: item.median)
            if auto is not None:
                print(f"{scale}	{mode}	{source_mode}	best_strategy	{best.strategy}")
                ratio = auto.median / best.median if best.median else float('inf')
                print(f"{scale}	{mode}	{source_mode}	auto_vs_best	{ratio:.2f}x")
                auto_metrics = parse_metrics(auto.stderr)
                auto_planner = select_planner_summary(auto_metrics)
                print(f"{scale}	{mode}	{source_mode}	auto_planner	{auto_planner}")

        if "auto" in source_modes:
            for scale in scales:
                for mode in modes:
                    auto = by_source_mode.get((scale, mode, "auto"))
                    if auto is None:
                        continue

                    concrete = [
                        result
                        for (result_scale, result_mode, result_source_mode), result in by_source_mode.items()
                        if result_scale == scale and result_mode == mode and result_source_mode != "auto"
                    ]
                    if not concrete:
                        continue

                    best = min(concrete, key=lambda item: item.median)
                    ratio = auto.median / best.median if best.median else float("inf")
                    print(f"{scale}	{mode}	auto	best_source_mode	{best.source_mode}")
                    print(f"{scale}	{mode}	auto	chosen_source_mode	{auto.resolved_source_mode}")
                    print(f"{scale}	{mode}	auto	auto_vs_best_source_mode	{ratio:.2f}x")

        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
