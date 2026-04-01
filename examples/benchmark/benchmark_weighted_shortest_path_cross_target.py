#!/usr/bin/env python3
"""
Benchmark positive weighted shortest-path-to-root across the C# query engine
and generated DFS binaries for C#, Rust, and Go.
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
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
QRY_RUNTIME = ROOT / "src" / "unifyweaver" / "targets" / "csharp_query_runtime" / "QueryRuntime.cs"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
FACTS_PATH = BENCH_DIR / "10k" / "facts.pl"

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

QE_PROGRAM = """\
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
        if (args.Length < 2)
        {
            Console.Error.WriteLine("Usage: program <category_parent.tsv> <article_category.tsv>");
            Environment.Exit(1);
        }

        var swTotal = Stopwatch.StartNew();
        var swLoad = Stopwatch.StartNew();

        var provider = new InMemoryRelationProvider();
        var edgeId = new PredicateId("category_parent", 2);
        var predId = new PredicateId("category_weighted_shortest", 3);
        var weightId = new PredicateId("category_weight", 2);
        var articleCategories = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var outDegree = new Dictionary<string, int>(StringComparer.Ordinal);
        var sourceNodes = new HashSet<string>(StringComparer.Ordinal);

        foreach (var (line, i) in File.ReadLines(args[0]).Select((l, i) => (l, i)))
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
                TableMode.Min,
                true),
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
            Console.WriteLine($"{article}\\t{ROOT_CATEGORY}\\t{distance.ToString("F12", CultureInfo.InvariantCulture)}");
        }

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
    parser.add_argument("--scales", default="300,1k,5k,10k")
    parser.add_argument(
        "--targets",
        default="csharp-query,csharp-dfs,rust-dfs,go-dfs",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, env=env)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def scale_sort_key(scale: str) -> tuple[int, str]:
    digits = "".join(ch for ch in scale if ch.isdigit())
    suffix = "".join(ch for ch in scale if not ch.isdigit())
    if not digits:
        return (0, scale)
    value = int(digits)
    if suffix.lower() == "k":
        value *= 1000
    return (value, scale)


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


def generate_pipeline_source(root: Path, target: str) -> Path:
    ext = {"csharp": ".cs", "rust": ".rs", "go": ".go"}[target]
    filename = "Program.cs" if target == "csharp" else f"weighted_shortest_path{ext}"
    output = root / filename
    run(
        [
            sys.executable,
            str(GENERATOR),
            "--facts",
            str(FACTS_PATH),
            "--root",
            "Physics",
            "--workload",
            "weighted_shortest_path",
            "--target",
            target,
            "--output",
            str(output),
        ]
    )
    return output


def build_csharp_query(root: Path) -> list[str]:
    project_dir = root / "csharp_query"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Program.cs").write_text(QE_PROGRAM, encoding="utf-8")
    (project_dir / "QueryRuntime.cs").write_text(QRY_RUNTIME.read_text(encoding="utf-8"), encoding="utf-8")
    (project_dir / "benchmark_weighted_shortest_path_query.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_weighted_shortest_path_query.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_weighted_shortest_path_query.dll")]


def build_csharp_dfs(root: Path) -> list[str]:
    project_dir = root / "csharp_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    generate_pipeline_source(project_dir, "csharp")
    (project_dir / "benchmark_weighted_shortest_path_dfs.csproj").write_text(CSHARP_PROJECT, encoding="utf-8")
    run(["dotnet", "build", "benchmark_weighted_shortest_path_dfs.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_weighted_shortest_path_dfs.dll")]


def build_rust_dfs(root: Path) -> list[str]:
    project_dir = root / "rust_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = generate_pipeline_source(project_dir, "rust")
    binary = project_dir / "weighted_shortest_path_rust"
    run(["rustc", "-O", str(source), "-o", str(binary)])
    return [str(binary)]


def build_go_dfs(root: Path) -> list[str]:
    project_dir = root / "go_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = generate_pipeline_source(project_dir, "go")
    binary = project_dir / "weighted_shortest_path_go"
    go_cache = project_dir / ".gocache"
    go_cache.mkdir(exist_ok=True)
    env = dict(os.environ, GOCACHE=str(go_cache))
    run(["go", "build", "-o", str(binary), str(source)], env=env)
    return [str(binary)]


def normalize_output(output: str) -> str:
    lines = output.splitlines()
    if not lines:
        return ""
    header = lines[0]
    rows: list[tuple[str, str, float]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rows.append((parts[0], parts[1], round(float(parts[2]), 12)))
    rows.sort(key=lambda item: (item[2], item[0], item[1]))
    normalized = [header]
    for article, root, value in rows:
        normalized.append(f"{article}\t{root}\t{value:.12f}")
    return "\n".join(normalized)


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [str(edge_path), str(article_path)])
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr

    normalized = normalize_output(stdout)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    row_count = max(0, len(normalized.splitlines()) - 1)
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    by_scale: dict[str, list[RunResult]] = {}
    for result in results:
        by_scale.setdefault(result.scale, []).append(result)

    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale in sorted(by_scale.keys(), key=scale_sort_key):
        entries = sorted(by_scale[scale], key=lambda item: item.target)
        for result in entries:
            print(
                f"{scale}\t{result.target}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}\t"
                f"{result.row_count}\t{result.stdout_sha256[:12]}"
            )

        qe = next((item for item in entries if item.target == "csharp-query"), None)
        csharp_dfs = next((item for item in entries if item.target == "csharp-dfs"), None)
        rust_dfs = next((item for item in entries if item.target == "rust-dfs"), None)
        go_dfs = next((item for item in entries if item.target == "go-dfs"), None)
        dfs_like = [item for item in entries if item.target != "csharp-query"]

        if len(dfs_like) > 1:
            dfs_hashes = {item.stdout_sha256 for item in dfs_like}
            print(f"{scale}\tdfs_outputs\t{'match' if len(dfs_hashes) == 1 else 'MISMATCH'}")
        if qe and csharp_dfs:
            print(f"{scale}\tquery_vs_csharp_dfs\t{'match' if qe.stdout_sha256 == csharp_dfs.stdout_sha256 else 'DIFFERENT'}")
            print(f"{scale}\tspeedup_vs_csharp_dfs\t{csharp_dfs.median / qe.median:.2f}x")
        if qe and rust_dfs:
            print(f"{scale}\tspeedup_vs_rust_dfs\t{rust_dfs.median / qe.median:.2f}x")
        if qe and go_dfs:
            print(f"{scale}\tspeedup_vs_go_dfs\t{go_dfs.median / qe.median:.2f}x")


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-weighted-cross-target-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-weighted-cross-target-")
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
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
