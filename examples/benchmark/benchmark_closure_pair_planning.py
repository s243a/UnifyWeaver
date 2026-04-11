#!/usr/bin/env python3
"""
Measure closure-pair strategy planning in the C# query runtime.

This harness exercises representative transitive-closure pair workloads against
benchmark TSVs using a temporary C# project built around `QueryRuntime.cs`:

- `single`: one concrete source/target pair
- `source_fanout`: one source with many bound targets
- `target_fanin`: one target with many bound sources
- `mixed`: a blend of forward- and backward-oriented concrete pairs
- `grouped_mixed`: the grouped closure analogue over a duplicated grouped edge relation

It is intended as a focused validation tool for the closure-pair planner rather
than a cross-target benchmark.
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
    static readonly string[] Groups = new[] { "all", "focus" };

    static ClosurePairStrategy ParsePairStrategy(string value) => value.ToLowerInvariant() switch
    {
        "forward" => ClosurePairStrategy.Forward,
        "backward" => ClosurePairStrategy.Backward,
        "memo-source" => ClosurePairStrategy.MemoizedBySource,
        "memo-target" => ClosurePairStrategy.MemoizedByTarget,
        "mixed" => ClosurePairStrategy.MixedDirection,
        "mixed-cache" => ClosurePairStrategy.MixedDirectionWithPairProbeCache,
        _ => ClosurePairStrategy.Auto,
    };

    static List<string[]> ReadBinaryRows(string path)
    {
        return File.ReadLines(path)
            .Skip(1)
            .Select(line => line.Split('	'))
            .Where(parts => parts.Length == 2)
            .ToList();
    }

    static Dictionary<string, List<string>> BuildIndex(IEnumerable<string[]> rows, int keyIndex, int valueIndex)
    {
        var index = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var parts in rows)
        {
            var key = parts[keyIndex];
            var value = parts[valueIndex];
            if (!index.TryGetValue(key, out var bucket))
            {
                bucket = new List<string>();
                index.Add(key, bucket);
            }

            bucket.Add(value);
        }

        foreach (var bucket in index.Values)
        {
            bucket.Sort(StringComparer.Ordinal);
        }

        return index;
    }

    static List<string> ExpandReachable(Dictionary<string, List<string>> graph, string seed, int limit)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var queue = new Queue<string>();
        if (graph.TryGetValue(seed, out var initial))
        {
            foreach (var next in initial)
            {
                if (seen.Add(next))
                {
                    queue.Enqueue(next);
                }
            }
        }

        var ordered = new List<string>();
        while (queue.Count > 0 && ordered.Count < limit)
        {
            var node = queue.Dequeue();
            ordered.Add(node);
            if (!graph.TryGetValue(node, out var bucket))
            {
                continue;
            }

            foreach (var next in bucket)
            {
                if (seen.Add(next))
                {
                    queue.Enqueue(next);
                }
            }
        }

        return ordered;
    }

    static (string Seed, List<string> Others) FindBestReachable(Dictionary<string, List<string>> graph, int desired)
    {
        var bestSeed = string.Empty;
        var best = new List<string>();
        foreach (var seed in graph.Keys.OrderBy(value => value, StringComparer.Ordinal))
        {
            var reachable = ExpandReachable(graph, seed, desired * 4);
            if (reachable.Count > best.Count)
            {
                bestSeed = seed;
                best = reachable;
            }

            if (best.Count >= desired)
            {
                break;
            }
        }

        if (best.Count == 0)
        {
            throw new InvalidOperationException("Failed to derive closure-pair seeds from benchmark graph.");
        }

        return (bestSeed, best.Take(Math.Min(desired, best.Count)).ToList());
    }

    static (string Seed, List<string> Others) FindBestDirect(Dictionary<string, List<string>> graph, int desired)
    {
        var bestSeed = string.Empty;
        var best = new List<string>();
        foreach (var seed in graph.Keys.OrderBy(value => value, StringComparer.Ordinal))
        {
            if (!graph.TryGetValue(seed, out var bucket) || bucket.Count == 0)
            {
                continue;
            }

            if (bucket.Count > best.Count)
            {
                bestSeed = seed;
                best = bucket.Take(Math.Min(desired, bucket.Count)).ToList();
            }

            if (best.Count >= desired)
            {
                break;
            }
        }

        if (best.Count == 0)
        {
            throw new InvalidOperationException("Failed to derive direct grouped closure-pair seeds from benchmark graph.");
        }

        return (bestSeed, best);
    }

    static List<object[]> BuildUngroupedParameters(string mode, Dictionary<string, List<string>> succ, Dictionary<string, List<string>> pred)
    {
        const int FanoutSize = 8;
        var fanout = FindBestReachable(succ, FanoutSize);
        var fanin = FindBestReachable(pred, FanoutSize);

        return mode switch
        {
            "single" => new List<object[]>
            {
                new object[] { fanout.Seed, fanout.Others.Last() }
            },
            "source_fanout" => fanout.Others.Select(target => new object[] { fanout.Seed, target }).ToList<object[]>(),
            "target_fanin" => fanin.Others.Select(source => new object[] { source, fanin.Seed }).ToList<object[]>(),
            "mixed" => fanout.Others.Take(4).Select(target => new object[] { fanout.Seed, target })
                .Concat(fanin.Others.Take(4).Select(source => new object[] { source, fanin.Seed }))
                .GroupBy(tuple => string.Join("	", tuple.Cast<object?>().Select(value => value?.ToString() ?? "<null>")), StringComparer.Ordinal)
                .Select(group => group.First())
                .ToList(),
            _ => throw new ArgumentException($"Unknown ungrouped mode: {mode}")
        };
    }

    static List<object[]> BuildGroupedParameters(Dictionary<string, List<string>> succ, Dictionary<string, List<string>> pred)
    {
        var outgoing = FindBestDirect(succ, 6);
        var incoming = FindBestDirect(pred, 6);
        var rows = new List<object[]>();

        foreach (var target in outgoing.Others.Take(3))
        {
            rows.Add(new object[] { outgoing.Seed, target, Groups[0] });
            rows.Add(new object[] { outgoing.Seed, target, Groups[1] });
        }

        foreach (var source in incoming.Others.Take(3))
        {
            rows.Add(new object[] { source, incoming.Seed, Groups[0] });
            rows.Add(new object[] { source, incoming.Seed, Groups[1] });
        }

        return rows
            .GroupBy(tuple => string.Join("	", tuple.Cast<object?>().Select(value => value?.ToString() ?? "<null>")), StringComparer.Ordinal)
            .Select(group => group.First())
            .ToList();
    }

    static string BuildGroupedEdgeFile(IEnumerable<string[]> edges)
    {
        var path = Path.Combine(Path.GetTempPath(), $"uw-closure-pairs-{Guid.NewGuid():N}.tsv");
        using var writer = new StreamWriter(path, false);
        writer.WriteLine("from	to	group");
        foreach (var parts in edges)
        {
            foreach (var group in Groups)
            {
                writer.Write(parts[0]);
                writer.Write('	');
                writer.Write(parts[1]);
                writer.Write('	');
                writer.WriteLine(group);
            }
        }

        return path;
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
        if (args.Length < 2)
        {
            Console.Error.WriteLine("Usage: program <single|source_fanout|target_fanin|mixed|grouped_mixed> <category_parent.tsv>");
            return 1;
        }

        var mode = args[0];
        var edgePath = args[1];
        var traceEnabled = Environment.GetEnvironmentVariable("UNIFYWEAVER_BENCH_TRACE") == "1";
        var pairStrategy = ParsePairStrategy(Environment.GetEnvironmentVariable("UNIFYWEAVER_CLOSURE_PAIR_STRATEGY") ?? "auto");

        var edgeRows = ReadBinaryRows(edgePath);
        var succ = BuildIndex(edgeRows, 0, 1);
        var pred = BuildIndex(edgeRows, 1, 0);

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var predicateId = new PredicateId("category_ancestor", 2);
        var groupedEdgeId = new PredicateId("grouped_category_parent", 3);
        var groupedPredicateId = new PredicateId("grouped_category_ancestor", 3);

        provider.RegisterDelimitedSource(edgeId, new DelimitedRelationSource(edgePath));

        var groupedEdgePath = BuildGroupedEdgeFile(edgeRows);
        provider.RegisterDelimitedSource(groupedEdgeId, new DelimitedRelationSource(groupedEdgePath, '	', 1, 3));

        try
        {
            QueryPlan plan;
            List<object[]> parameters;
            if (mode == "grouped_mixed")
            {
                plan = new QueryPlan(
                    groupedPredicateId,
                    new GroupedTransitiveClosureNode(groupedEdgeId, groupedPredicateId, new int[] { 2 }),
                    true,
                    new int[] { 0, 1, 2 });
                parameters = BuildGroupedParameters(succ, pred);
            }
            else
            {
                plan = new QueryPlan(
                    predicateId,
                    new TransitiveClosureNode(edgeId, predicateId),
                    true,
                    new int[] { 0, 1 });
                parameters = BuildUngroupedParameters(mode, succ, pred);
            }

            var trace = traceEnabled ? new QueryExecutionTrace() : null;
            var executor = new QueryExecutor(provider, new QueryExecutorOptions(
                ReuseCaches: true,
                ClosurePairStrategy: pairStrategy));

            var stopwatch = Stopwatch.StartNew();
            var rows = executor.Execute(plan, parameters, trace).ToList();
            stopwatch.Stop();

            Console.Error.WriteLine($"mode={mode}");
            Console.Error.WriteLine($"closure_pair_strategy_setting={pairStrategy}");
            Console.Error.WriteLine($"request_count={parameters.Count}");
            Console.Error.WriteLine($"row_count={rows.Count}");
            Console.Error.WriteLine($"elapsed_ms={stopwatch.ElapsedMilliseconds}");

            if (traceEnabled && trace is not null)
            {
                Console.Error.WriteLine(
                    "closure_pair_planner_strategies=" +
                    SummarizeStrategies(
                        trace,
                        strategy => strategy.Strategy.Contains("TransitiveClosurePairsMaterializationPlan", StringComparison.Ordinal) ||
                                    strategy.Strategy.Contains("GroupedTransitiveClosurePairsMaterializationPlan", StringComparison.Ordinal)));
                Console.Error.WriteLine(
                    "closure_pair_operator_strategies=" +
                    SummarizeStrategies(
                        trace,
                        strategy => strategy.Strategy.StartsWith("TransitiveClosurePairs", StringComparison.Ordinal) ||
                                    strategy.Strategy.StartsWith("GroupedTransitiveClosurePairs", StringComparison.Ordinal)));
                Console.Error.WriteLine("closure_pair_phase_summary=" + SummarizePhases(trace, "closure_pair_"));
            }

            return 0;
        }
        finally
        {
            try { File.Delete(groupedEdgePath); } catch { }
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
    parser.add_argument("--modes", default="single,source_fanout,target_fanin,mixed,grouped_mixed")
    parser.add_argument("--strategies", default="auto,forward,backward,memo-source,memo-target,mixed,mixed-cache")
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

    project_dir = root / "closure_pair_planning"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_closure_pair_planning.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_closure_pair_planning.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_closure_pair_planning.dll")]


def parse_metrics(stderr: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in stderr.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def parse_phase_summary(summary: str) -> dict[str, float]:
    phases: dict[str, float] = {}
    for part in summary.split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        try:
            phases[key] = float(value)
        except ValueError:
            continue
    return phases


def extract_effective_pair_plan(planner_summary: str) -> str:
    for part in planner_summary.split("|"):
        marker = "MaterializationPlanPairs"
        index = part.find(marker)
        if index >= 0:
            return part[index + len(marker):].split("=", 1)[0]
    return planner_summary


def benchmark_mode(command: list[str], scale: str, mode: str, strategy: str, repetitions: int) -> BenchResult:
    scale_dir = BENCH_DIR / scale
    edge_path = scale_dir / "category_parent.tsv"
    env = os.environ.copy()
    env["UNIFYWEAVER_BENCH_TRACE"] = "1"
    env["UNIFYWEAVER_CLOSURE_PAIR_STRATEGY"] = strategy

    times: list[float] = []
    stderr = ""
    if repetitions > 0:
        run(command + [mode, str(edge_path)], env=env)
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [mode, str(edge_path)], env=env)
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
        temp_root = Path(tempfile.mkdtemp(prefix="uw-closure-pairs-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-closure-pairs-")
        temp_root = Path(temp_ctx.name)

    try:
        command = build_harness(temp_root)
        results: list[BenchResult] = []
        for scale in scales:
            for mode in modes:
                for strategy in strategies:
                    results.append(benchmark_mode(command, scale, mode, strategy, args.repetitions))

        print("scale	mode	strategy	median_s	min_s	max_s	requests	rows	planner	phases")
        grouped: dict[tuple[str, str], dict[str, BenchResult]] = {}
        for result in sorted(results, key=lambda item: (scale_sort_key(item.scale), item.mode, item.strategy)):
            grouped.setdefault((result.scale, result.mode), {})[result.strategy] = result
            metrics = parse_metrics(result.stderr)
            print(
                f"{result.scale}	{result.mode}	{result.strategy}	{result.median:.3f}	"
                f"{min(result.times):.3f}	{max(result.times):.3f}	"
                f"{metrics.get('request_count', '')}	{metrics.get('row_count', '')}	"
                f"{metrics.get('closure_pair_planner_strategies', '')}	"
                f"{metrics.get('closure_pair_phase_summary', '')}"
            )

        for scale, mode in sorted(grouped.keys(), key=lambda item: (scale_sort_key(item[0]), item[1])):
            by_strategy = grouped[(scale, mode)]
            metrics_by_strategy = {strategy: parse_metrics(result.stderr) for strategy, result in by_strategy.items()}
            row_counts = {
                strategy: int(metrics.get("row_count", "0") or 0)
                for strategy, metrics in metrics_by_strategy.items()
            }
            max_row_count = max(row_counts.values(), default=0)
            eligible_results = [
                result
                for strategy, result in by_strategy.items()
                if row_counts.get(strategy, 0) == max_row_count
            ] or list(by_strategy.values())

            best = min(eligible_results, key=lambda item: item.median)
            best_by_effective_plan: dict[str, BenchResult] = {}
            for result in eligible_results:
                effective_plan = extract_effective_pair_plan(
                    metrics_by_strategy[result.strategy].get("closure_pair_planner_strategies", ""))
                existing = best_by_effective_plan.get(effective_plan)
                if existing is None or result.median < existing.median:
                    best_by_effective_plan[effective_plan] = result
            best_effective = min(best_by_effective_plan.values(), key=lambda item: item.median)

            auto = by_strategy.get("auto")
            if auto is not None:
                auto_metrics = metrics_by_strategy["auto"]
                ratio = auto.median / best.median if best.median else float("inf")
                effective_ratio = auto.median / best_effective.median if best_effective.median else float("inf")
                print(f"{scale}	{mode}	best_strategy	{best.strategy}")
                print(f"{scale}	{mode}	auto_vs_best	{ratio:.2f}x")
                print(f"{scale}	{mode}	best_effective_plan	{extract_effective_pair_plan(metrics_by_strategy[best_effective.strategy].get('closure_pair_planner_strategies', ''))}")
                print(f"{scale}	{mode}	auto_vs_best_effective	{effective_ratio:.2f}x")
                print(f"{scale}	{mode}	auto_planner	{auto_metrics.get('closure_pair_planner_strategies', '')}")
                auto_phases = parse_phase_summary(auto_metrics.get("closure_pair_phase_summary", ""))
                probe_ms = sum(
                    value
                    for phase, value in auto_phases.items()
                    if phase.startswith("closure_pair_probe_"))
                print(f"{scale}	{mode}	auto_strategy_select_ms	{auto_phases.get('closure_pair_strategy_select', 0.0):.3f}")
                print(f"{scale}	{mode}	auto_probe_ms	{probe_ms:.3f}")

        if args.keep_temp:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
