#!/usr/bin/env python3
"""
Benchmark seeded transitive-closure cache-hit paths in the C# query runtime.

This is intentionally narrower than the end-to-end cache smoke sequence: it
builds a temporary C# harness once, warms a seeded closure cache once, then
measures repeated same-key cache-hit executions inside the process.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import tempfile
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

PROGRAM = r"""using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using UnifyWeaver.QueryRuntime;

class Program
{
    static List<string[]> ReadEdges(string path)
    {
        return File.ReadLines(path)
            .Skip(1)
            .Select(line => line.Split('\t', 2))
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

        return index;
    }

    static List<object[]> SelectSeedParameters(Dictionary<string, List<string>> index, int seedCount)
    {
        return index
            .Where(entry => entry.Value.Count > 0)
            .OrderByDescending(entry => entry.Value.Count)
            .ThenBy(entry => entry.Key, StringComparer.Ordinal)
            .Take(seedCount)
            .Select(entry => new object[] { entry.Key })
            .ToList();
    }

    static QueryPlan BuildPlan(string mode, PredicateId edgeId, PredicateId predicateId)
    {
        return mode switch
        {
            "source" => new QueryPlan(
                predicateId,
                new TransitiveClosureNode(edgeId, predicateId),
                true,
                new int[] { 0 }),
            "target" => new QueryPlan(
                predicateId,
                new TransitiveClosureNode(edgeId, predicateId),
                true,
                new int[] { 1 }),
            _ => throw new ArgumentException($"Unknown mode: {mode}")
        };
    }

    static long ForceFullCollectionAndGetMemory()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        return GC.GetTotalMemory(forceFullCollection: true);
    }

    static long EstimateCacheStorageBytes(QueryExecutor executor, string mode)
    {
        var executorType = typeof(QueryExecutor);
        var cacheContextField = executorType.GetField("_cacheContext", BindingFlags.Instance | BindingFlags.NonPublic);
        var cacheContext = cacheContextField?.GetValue(executor);
        if (cacheContext is null)
        {
            return 0;
        }

        var cacheFieldName = mode == "source"
            ? "TransitiveClosureSeededResults"
            : "TransitiveClosureSeededByTargetResults";
        var cacheField = cacheContext.GetType().GetProperty(cacheFieldName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        var outerCache = cacheField?.GetValue(cacheContext) as System.Collections.IEnumerable;
        if (outerCache is null)
        {
            return 0;
        }

        long bytes = 0;
        foreach (var outerEntry in outerCache)
        {
            var innerCache = outerEntry.GetType().GetProperty("Value")?.GetValue(outerEntry) as System.Collections.IEnumerable;
            if (innerCache is null)
            {
                continue;
            }

            foreach (var innerEntry in innerCache)
            {
                var cachedRows = innerEntry.GetType().GetProperty("Value")?.GetValue(innerEntry);
                if (cachedRows is not null)
                {
                    bytes += EstimateCachedResultRowsStorageBytes(cachedRows);
                }
            }
        }

        return bytes;
    }

    static long EstimateCachedResultRowsStorageBytes(object cachedRows)
    {
        var type = cachedRows.GetType();
        var objectRows = type.GetField("_objectRows", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(cachedRows);
        if (objectRows is not null)
        {
            return EstimateObjectRowsStorageBytes(objectRows);
        }

        var leftValues = type.GetField("_leftValues", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(cachedRows);
        if (leftValues is not null)
        {
            var rightValues = type.GetField("_rightValues", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(cachedRows);
            return EstimateReferenceListStorageBytes(leftValues) + EstimateReferenceListStorageBytes(rightValues);
        }

        var targetNodeIds = type.GetField("_targetNodeIds", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(cachedRows);
        var depths = type.GetField("_depths", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(cachedRows);
        return EstimateIntListStorageBytes(targetNodeIds) + EstimateIntListStorageBytes(depths);
    }

    static long EstimateObjectRowsStorageBytes(object objectRows)
    {
        if (objectRows is not IReadOnlyCollection<object[]> rows)
        {
            return 0;
        }

        long bytes = EstimateReferenceListStorageBytes(objectRows);
        foreach (var row in rows)
        {
            bytes += EstimateArrayStorageBytes(row?.Length ?? 0, IntPtr.Size);
        }

        return bytes;
    }

    static long EstimateReferenceListStorageBytes(object? values)
    {
        if (values is null)
        {
            return 0;
        }

        if (values is Array array)
        {
            return EstimateArrayStorageBytes(array.Length, IntPtr.Size);
        }

        if (values is System.Collections.ICollection collection)
        {
            return EstimateArrayStorageBytes(collection.Count, IntPtr.Size);
        }

        return 0;
    }

    static long EstimateIntListStorageBytes(object? values)
    {
        if (values is null)
        {
            return 0;
        }

        if (values is Array array)
        {
            return EstimateArrayStorageBytes(array.Length, sizeof(int));
        }

        if (values is System.Collections.ICollection collection)
        {
            return EstimateArrayStorageBytes(collection.Count, sizeof(int));
        }

        return 0;
    }

    static long EstimateArrayStorageBytes(int count, int elementSize)
    {
        var bytes = 24L + ((long)count * elementSize);
        var alignment = IntPtr.Size;
        var remainder = bytes % alignment;
        return remainder == 0 ? bytes : bytes + alignment - remainder;
    }

    static void Main(string[] args)
    {
        if (args.Length < 4)
        {
            Console.Error.WriteLine("Usage: program <source|target> <category_parent.tsv> <seed-count> <repetitions> [compact-cache-rows]");
            Environment.Exit(1);
        }

        var mode = args[0];
        var edgePath = args[1];
        var seedCount = int.Parse(args[2], CultureInfo.InvariantCulture);
        var repetitions = int.Parse(args[3], CultureInfo.InvariantCulture);
        var compactSeededCacheRows = args.Length >= 5 && bool.Parse(args[4]);

        var edgeRows = ReadEdges(edgePath);
        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var predicateId = new PredicateId("category_ancestor", 2);
        provider.RegisterDelimitedSource(edgeId, new DelimitedRelationSource(edgePath));

        var index = mode == "source"
            ? BuildIndex(edgeRows, 0, 1)
            : BuildIndex(edgeRows, 1, 0);
        var parameters = SelectSeedParameters(index, seedCount);
        if (parameters.Count == 0)
        {
            throw new InvalidOperationException("No cache benchmark parameters were selected.");
        }

        var plan = BuildPlan(mode, edgeId, predicateId);
        var executor = new QueryExecutor(provider, new QueryExecutorOptions(
            ReuseCaches: true,
            SeededCacheMaxEntries: Math.Max(seedCount * 2, 8),
            SeededCacheAdmissionMinRows: 0,
            SeededCacheAdmissionMinRowsPerSeed: 0,
            CompactSeededCacheRows: compactSeededCacheRows));
        var cacheName = mode == "source"
            ? "TransitiveClosureSeeded"
            : "TransitiveClosureSeededByTarget";

        var retainedBeforeWarm = ForceFullCollectionAndGetMemory();
        var allocatedBeforeWarm = GC.GetTotalAllocatedBytes(precise: true);
        var warmTrace = new QueryExecutionTrace();
        List<object[]>? warmRows = executor.Execute(plan, parameters, warmTrace).ToList();
        var warmRowCount = warmRows.Count;
        var warmAllocatedBytes = GC.GetTotalAllocatedBytes(precise: true) - allocatedBeforeWarm;
        warmRows = null;
        warmTrace = null;
        var retainedAfterWarm = ForceFullCollectionAndGetMemory();
        var warmRetainedBytes = Math.Max(0, retainedAfterWarm - retainedBeforeWarm);
        var cacheStorageEstimateBytes = EstimateCacheStorageBytes(executor, mode);

        var elapsedMs = new List<double>();
        long lastRows = 0;
        long totalHits = 0;
        long totalBuilds = 0;
        for (var i = 0; i < repetitions; i++)
        {
            var trace = new QueryExecutionTrace();
            var stopwatch = Stopwatch.StartNew();
            var rows = executor.Execute(plan, parameters, trace).ToList();
            stopwatch.Stop();

            lastRows = rows.Count;
            elapsedMs.Add(stopwatch.Elapsed.TotalMilliseconds);
            foreach (var cache in trace.SnapshotCaches().Where(entry => entry.Cache == cacheName))
            {
                totalHits += cache.Hits;
                totalBuilds += cache.Builds;
            }
        }

        Console.WriteLine(string.Join('\t', new[]
        {
            mode,
            compactSeededCacheRows ? "compact" : "object",
            parameters.Count.ToString(CultureInfo.InvariantCulture),
            repetitions.ToString(CultureInfo.InvariantCulture),
            warmRowCount.ToString(CultureInfo.InvariantCulture),
            lastRows.ToString(CultureInfo.InvariantCulture),
            statistics(elapsedMs, values => values.Min()),
            statistics(elapsedMs, values => values.Max()),
            statistics(elapsedMs, values => Median(values)),
            totalHits.ToString(CultureInfo.InvariantCulture),
            totalBuilds.ToString(CultureInfo.InvariantCulture),
            warmRetainedBytes.ToString(CultureInfo.InvariantCulture),
            warmAllocatedBytes.ToString(CultureInfo.InvariantCulture),
            cacheStorageEstimateBytes.ToString(CultureInfo.InvariantCulture)
        }));
    }

    static string statistics(List<double> values, Func<List<double>, double> selector)
    {
        return selector(values).ToString("F3", CultureInfo.InvariantCulture);
    }

    static double Median(List<double> values)
    {
        var ordered = values.OrderBy(value => value).ToList();
        var midpoint = ordered.Count / 2;
        return ordered.Count % 2 == 1
            ? ordered[midpoint]
            : (ordered[midpoint - 1] + ordered[midpoint]) / 2.0;
    }
}
"""


@dataclass(frozen=True)
class BenchmarkResult:
    scale: str
    mode: str
    cache_rows: str
    seed_count: int
    repetitions: int
    warm_rows: int
    rows: int
    min_ms: float
    max_ms: float
    median_ms: float
    cache_hits: int
    cache_builds: int
    warm_retained_bytes: int
    warm_allocated_bytes: int
    cache_storage_estimate_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="300,1k")
    parser.add_argument("--modes", default="source,target")
    parser.add_argument("--cache-rows", choices=("object", "compact", "both"), default="object")
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument(
        "--runtime-source",
        type=Path,
        default=QRY_RUNTIME,
        help="QueryRuntime.cs to compile into the temporary benchmark harness",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def build_harness(root: Path, runtime_source: Path) -> list[str]:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet not found")
    if not runtime_source.exists():
        raise FileNotFoundError(runtime_source)

    project_dir = root / "seeded_cache_hits"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(runtime_source.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "seeded_cache_hits.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "seeded_cache_hits.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "seeded_cache_hits.dll")]


def run_benchmark(command: list[str], scale: str, mode: str, seed_count: int, repetitions: int, cache_rows: str) -> BenchmarkResult:
    edge_path = BENCH_DIR / scale / "category_parent.tsv"
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    compact_cache_rows = "true" if cache_rows == "compact" else "false"
    result = run(command + [mode, str(edge_path), str(seed_count), str(repetitions), compact_cache_rows])
    fields = result.stdout.strip().split("\t")
    if len(fields) != 14:
        raise RuntimeError(f"Unexpected benchmark output: {result.stdout!r}\n{result.stderr}")

    return BenchmarkResult(
        scale=scale,
        mode=fields[0],
        cache_rows=fields[1],
        seed_count=int(fields[2]),
        repetitions=int(fields[3]),
        warm_rows=int(fields[4]),
        rows=int(fields[5]),
        min_ms=float(fields[6]),
        max_ms=float(fields[7]),
        median_ms=float(fields[8]),
        cache_hits=int(fields[9]),
        cache_builds=int(fields[10]),
        warm_retained_bytes=int(fields[11]),
        warm_allocated_bytes=int(fields[12]),
        cache_storage_estimate_bytes=int(fields[13]),
    )


def main() -> int:
    args = parse_args()
    scales = [scale.strip() for scale in args.scales.split(",") if scale.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    cache_row_modes = ["object", "compact"] if args.cache_rows == "both" else [args.cache_rows]

    temp_root = Path(tempfile.mkdtemp(prefix="unifyweaver_seeded_cache_hits_"))
    try:
        command = build_harness(temp_root, args.runtime_source)
        print("scale\tmode\tcache_rows\tseeds\trepetitions\twarm_rows\trows\tmedian_ms\tmin_ms\tmax_ms\tcache_hits\tcache_builds\twarm_retained_bytes\twarm_allocated_bytes\tcache_storage_estimate_bytes")
        for scale in scales:
            for mode in modes:
                for cache_rows in cache_row_modes:
                    result = run_benchmark(command, scale, mode, args.seed_count, args.repetitions, cache_rows)
                    print(
                        f"{result.scale}\t{result.mode}\t{result.cache_rows}\t"
                        f"{result.seed_count}\t{result.repetitions}\t"
                        f"{result.warm_rows}\t{result.rows}\t{result.median_ms:.3f}\t"
                        f"{result.min_ms:.3f}\t{result.max_ms:.3f}\t"
                        f"{result.cache_hits}\t{result.cache_builds}\t"
                        f"{result.warm_retained_bytes}\t{result.warm_allocated_bytes}\t"
                        f"{result.cache_storage_estimate_bytes}"
                    )
    finally:
        if args.keep_temp:
            print(f"kept temp: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
