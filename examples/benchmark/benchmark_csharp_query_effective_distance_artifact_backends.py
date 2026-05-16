#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
LMDB_PROJECT = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime_lmdb"
    / "UnifyWeaver.QueryRuntime.Lmdb.csproj"
)
CORE_DLL = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime"
    / "bin"
    / "Debug"
    / "net9.0"
    / "UnifyWeaver.QueryRuntime.Core.dll"
)
LMDB_DLL = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime_lmdb"
    / "bin"
    / "Debug"
    / "net9.0"
    / "UnifyWeaver.QueryRuntime.Lmdb.dll"
)
LIGHTNINGDB_PACKAGE = Path.home() / ".nuget" / "packages" / "lightningdb" / "0.21.0"
LIGHTNINGDB_DLL = LIGHTNINGDB_PACKAGE / "lib" / "net9.0" / "LightningDB.dll"


HEADERS = [
    "scale",
    "run",
    "mode",
    "rows",
    "distinct_categories",
    "lookup_keys",
    "artifact_bytes",
    "open_ms",
    "lookup_ms",
    "bucket_ms",
    "scan_ms",
    "retained_bytes",
    "scan_hash",
    "lookup_hash",
    "bucket_hash",
]

RAW_HEADERS = [column for column in HEADERS if column != "run"]

SUMMARY_HEADERS = [
    "scale",
    "rows",
    "distinct_categories",
    "lookup_keys",
    "best_lookup_mode",
    "best_bucket_mode",
    "best_scan_mode",
    "smallest_artifact_mode",
    "lookup_ms_by_mode",
    "bucket_ms_by_mode",
    "scan_ms_by_mode",
    "artifact_bytes_by_mode",
]


@dataclass(frozen=True)
class BenchmarkRow:
    values: dict[str, str]


@dataclass(frozen=True)
class SummaryRow:
    values: dict[str, str]


def run_checked(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def build_lmdb_runtime() -> None:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet is not available")
    if not LIGHTNINGDB_PACKAGE.exists():
        raise RuntimeError("LightningDB 0.21.0 package is not available in the local NuGet cache")

    run_checked(["dotnet", "build", str(LMDB_PROJECT)], cwd=ROOT, timeout=120)
    for path in (CORE_DLL, LMDB_DLL, LIGHTNINGDB_DLL):
        if not path.exists():
            raise RuntimeError(f"required C# benchmark dependency is missing: {path}")


def write_benchmark_project(project_dir: Path) -> Path:
    project_path = project_dir / "CSharpQueryEffectiveDistanceArtifactBackends.csproj"
    program_path = project_dir / "Program.cs"
    project_path.write_text(
        textwrap.dedent(
            f"""\
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <OutputType>Exe</OutputType>
                <TargetFramework>net9.0</TargetFramework>
                <Nullable>enable</Nullable>
                <ImplicitUsings>enable</ImplicitUsings>
                <NuGetAudit>false</NuGetAudit>
              </PropertyGroup>
              <ItemGroup>
                <Reference Include="UnifyWeaver.QueryRuntime.Core">
                  <HintPath>{CORE_DLL}</HintPath>
                </Reference>
                <Reference Include="UnifyWeaver.QueryRuntime.Lmdb">
                  <HintPath>{LMDB_DLL}</HintPath>
                </Reference>
                <Reference Include="LightningDB">
                  <HintPath>{LIGHTNINGDB_DLL}</HintPath>
                </Reference>
              </ItemGroup>
            </Project>
            """
        ),
        encoding="utf-8",
    )
    program_path.write_text(BENCHMARK_PROGRAM, encoding="utf-8")
    return project_path


def parse_rows(output: str, run_index: int) -> list[BenchmarkRow]:
    reader = csv.DictReader(output.splitlines(), delimiter="\t")
    if reader.fieldnames != RAW_HEADERS:
        raise RuntimeError(f"unexpected benchmark headers: {reader.fieldnames}")
    rows = []
    for row in reader:
        values = dict(row)
        values["run"] = str(run_index)
        rows.append(BenchmarkRow(values))
    return rows


def run_scale(
    scale: str,
    lookup_keys: int,
    lookup_repetitions: int,
    run_index: int,
    keep_temp: bool,
) -> list[BenchmarkRow]:
    scale_dir = BENCH_DIR / scale
    if not (scale_dir / "category_parent.tsv").exists():
        raise RuntimeError(f"scale has no category_parent.tsv: {scale_dir}")

    build_lmdb_runtime()
    temp_path = Path(tempfile.mkdtemp(prefix=f"uw-csharp-effective-distance-artifacts-{scale}-"))
    try:
        project_path = write_benchmark_project(temp_path)
        run_checked(["dotnet", "build", str(project_path)], cwd=temp_path, timeout=120)

        output_dir = temp_path / "bin" / "Debug" / "net9.0"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            str(output_dir)
            if not env.get("LD_LIBRARY_PATH")
            else f"{output_dir}:{env['LD_LIBRARY_PATH']}"
        )
        result = run_checked(
            [
                "dotnet",
                str(output_dir / "CSharpQueryEffectiveDistanceArtifactBackends.dll"),
                "--scale",
                scale,
                "--scale-dir",
                str(scale_dir),
                "--lookup-keys",
                str(lookup_keys),
                "--lookup-repetitions",
                str(lookup_repetitions),
            ],
            cwd=temp_path,
            timeout=240,
        )
        return parse_rows(result.stdout, run_index)
    finally:
        if keep_temp:
            print(f"kept benchmark project: {temp_path}", file=sys.stderr)
        else:
            shutil.rmtree(temp_path, ignore_errors=True)


def render_tsv(rows: list[BenchmarkRow]) -> str:
    lines = ["\t".join(HEADERS)]
    lines.extend("\t".join(row.values[column] for column in HEADERS) for row in rows)
    return "\n".join(lines)


def render_markdown(rows: list[BenchmarkRow]) -> str:
    lines = [
        "| Scale | Run | Mode | Rows | Categories | Lookup keys | Artifact bytes | Open ms | Lookup ms | Bucket ms | Scan ms | Retained bytes |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        value = row.values
        lines.append(
            "| {scale} | {run} | {mode} | {rows} | {distinct_categories} | {lookup_keys} | {artifact_bytes} | {open_ms} | {lookup_ms} | {bucket_ms} | {scan_ms} | {retained_bytes} |".format(
                **value
            )
        )
    return "\n".join(lines)


def numeric(row: BenchmarkRow, column: str) -> float:
    return float(row.values[column])


def scale_sort_key(scale: str) -> tuple[int, str]:
    label = scale.strip().lower()
    multiplier = 1
    if label.endswith("k"):
        multiplier = 1_000
        label = label[:-1]
    elif label.endswith("m"):
        multiplier = 1_000_000
        label = label[:-1]

    try:
        return (int(float(label) * multiplier), scale)
    except ValueError:
        return (sys.maxsize, scale)


def summarize(rows: list[BenchmarkRow]) -> list[SummaryRow]:
    grouped: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.values["scale"], []).append(row)

    summaries: list[SummaryRow] = []
    for scale in sorted(grouped, key=scale_sort_key):
        scale_rows = grouped[scale]
        modes = sorted({row.values["mode"] for row in scale_rows})

        def median_by_mode(column: str) -> dict[str, float]:
            return {
                mode: statistics.median(numeric(row, column) for row in scale_rows if row.values["mode"] == mode)
                for mode in modes
            }

        lookup = median_by_mode("lookup_ms")
        bucket = median_by_mode("bucket_ms")
        scan = median_by_mode("scan_ms")
        artifact = median_by_mode("artifact_bytes")
        row0 = scale_rows[0].values
        summaries.append(
            SummaryRow(
                {
                    "scale": scale,
                    "rows": row0["rows"],
                    "distinct_categories": row0["distinct_categories"],
                    "lookup_keys": row0["lookup_keys"],
                    "best_lookup_mode": min(lookup, key=lookup.get),
                    "best_bucket_mode": min(bucket, key=bucket.get),
                    "best_scan_mode": min(scan, key=scan.get),
                    "smallest_artifact_mode": min(
                        (mode for mode in artifact if mode != "preload"),
                        key=artifact.get,
                    ),
                    "lookup_ms_by_mode": format_mode_values(lookup),
                    "bucket_ms_by_mode": format_mode_values(bucket),
                    "scan_ms_by_mode": format_mode_values(scan),
                    "artifact_bytes_by_mode": format_mode_values(artifact, digits=0),
                }
            )
        )
    return summaries


def format_mode_values(values: dict[str, float], digits: int = 3) -> str:
    def format_value(value: float) -> str:
        return f"{value:.{digits}f}" if digits > 0 else str(int(value))

    return "|".join(f"{mode}:{format_value(values[mode])}" for mode in sorted(values))


def render_summary_tsv(rows: list[SummaryRow]) -> str:
    lines = ["\t".join(SUMMARY_HEADERS)]
    lines.extend("\t".join(row.values[column] for column in SUMMARY_HEADERS) for row in rows)
    return "\n".join(lines)


def render_summary_markdown(rows: list[SummaryRow]) -> str:
    lines = [
        "| Scale | Rows | Categories | Lookup keys | Best lookup | Best bucket | Best scan | Smallest artifact |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        value = row.values
        lines.append(
            "| {scale} | {rows} | {distinct_categories} | {lookup_keys} | {best_lookup_mode} | {best_bucket_mode} | {best_scan_mode} | {smallest_artifact_mode} |".format(
                **value
            )
        )
    lines.append("")
    lines.append("Median timing/detail fields are available in `summary-tsv`.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare C# query artifact backends on the real effective-distance category_parent/2 relation."
    )
    parser.add_argument("--scales", default="300,1k,5k,10k", help="comma-separated scales from data/benchmark")
    parser.add_argument("--lookup-keys", type=int, default=64, help="number of category IDs to probe")
    parser.add_argument("--lookup-repetitions", type=int, default=5, help="lookup passes per mode")
    parser.add_argument("--repetitions", type=int, default=1, help="independent benchmark runs per scale")
    parser.add_argument(
        "--format",
        choices=("tsv", "markdown", "summary-tsv", "summary-markdown"),
        default="tsv",
    )
    parser.add_argument("--keep-temp", action="store_true", help="keep the generated C# benchmark project")
    args = parser.parse_args(argv)

    if args.lookup_keys <= 0:
        parser.error("--lookup-keys must be positive")
    if args.lookup_repetitions <= 0:
        parser.error("--lookup-repetitions must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    scales = [scale.strip() for scale in args.scales.split(",") if scale.strip()]
    if not scales:
        parser.error("--scales must include at least one scale")

    rows: list[BenchmarkRow] = []
    for scale in scales:
        for run_index in range(1, args.repetitions + 1):
            rows.extend(run_scale(scale, args.lookup_keys, args.lookup_repetitions, run_index, args.keep_temp))

    if args.format == "summary-tsv":
        print(render_summary_tsv(summarize(rows)))
    elif args.format == "summary-markdown":
        print(render_summary_markdown(summarize(rows)))
    elif args.format == "markdown":
        print(render_markdown(rows))
    else:
        print(render_tsv(rows))
    return 0


BENCHMARK_PROGRAM = r"""
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LightningDB;
using UnifyWeaver.QueryRuntime;

static string ReadArg(string[] args, string name, string defaultValue)
{
    for (var index = 0; index < args.Length - 1; index++)
    {
        if (args[index] == name)
        {
            return args[index + 1];
        }
    }

    return defaultValue;
}

static int ReadIntArg(string[] args, string name, int defaultValue)
{
    return int.TryParse(ReadArg(args, name, ""), out var parsed) ? parsed : defaultValue;
}

static long DirectorySize(string path)
{
    if (!Directory.Exists(path))
    {
        return File.Exists(path) ? new FileInfo(path).Length : 0L;
    }

    return Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories)
        .Select(file => new FileInfo(file).Length)
        .Sum();
}

static string HashRows(IEnumerable<object[]> rows)
{
    var texts = rows.Select(row => string.Join("\t", row.Select(value => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "")))
        .OrderBy(value => value, StringComparer.Ordinal);
    using var sha = SHA256.Create();
    foreach (var text in texts)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        sha.TransformBlock(bytes, 0, bytes.Length, null, 0);
        sha.TransformBlock(new byte[] { 10 }, 0, 1, null, 0);
    }
    sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
    return Convert.ToHexString(sha.Hash!).ToLowerInvariant();
}

static double Measure(Action action)
{
    var stopwatch = Stopwatch.StartNew();
    action();
    stopwatch.Stop();
    return stopwatch.Elapsed.TotalMilliseconds;
}

static int Intern(Dictionary<string, int> ids, string value)
{
    if (ids.TryGetValue(value, out var id))
    {
        return id;
    }

    id = ids.Count;
    ids[value] = id;
    return id;
}

static IReadOnlyList<(int Child, int Parent)> ReadAndInternEdges(string scaleDir, out int distinctCategories)
{
    var ids = new Dictionary<string, int>(StringComparer.Ordinal);
    var rows = new List<(int Child, int Parent)>();
    foreach (var line in File.ReadLines(Path.Combine(scaleDir, "category_parent.tsv")).Skip(1))
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            continue;
        }

        var tab = line.IndexOf('\t');
        if (tab < 0)
        {
            throw new InvalidDataException($"category_parent.tsv row has no tab: {line}");
        }

        var child = line.Substring(0, tab);
        var parent = line.Substring(tab + 1);
        rows.Add((Intern(ids, child), Intern(ids, parent)));
    }

    distinctCategories = ids.Count;
    return rows;
}

static void WriteInput(string path, IReadOnlyList<(int Child, int Parent)> rows)
{
    using var writer = new StreamWriter(path, append: false, Encoding.UTF8);
    writer.WriteLine("child\tparent");
    foreach (var row in rows)
    {
        writer.Write(row.Child.ToString(System.Globalization.CultureInfo.InvariantCulture));
        writer.Write('\t');
        writer.Write(row.Parent.ToString(System.Globalization.CultureInfo.InvariantCulture));
        writer.WriteLine();
    }
}

static string WriteLmdbArtifact(
    string root,
    PredicateId predicate,
    IReadOnlyList<(int Child, int Parent)> rows,
    string sourcePath)
{
    var envPath = Path.Combine(root, "lmdb-category-parent");
    Directory.CreateDirectory(envPath);
    using (var env = new LightningEnvironment(envPath) { MaxDatabases = 4, MapSize = 256L * 1024L * 1024L })
    {
        env.Open();
        using var tx = env.BeginTransaction();
        using var db = tx.OpenDatabase("main", new DatabaseConfiguration
        {
            Flags = DatabaseOpenFlags.Create | DatabaseOpenFlags.DuplicatesSort,
        });

        foreach (var row in rows)
        {
            tx.Put(
                db,
                Encoding.UTF8.GetBytes(row.Child.ToString(System.Globalization.CultureInfo.InvariantCulture)),
                Encoding.UTF8.GetBytes(row.Parent.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }

        tx.Commit();
    }

    var manifest = new LmdbRelationArtifactManifest
    {
        PredicateName = predicate.Name,
        Arity = predicate.Arity,
        EnvironmentPath = Path.GetFileName(envPath),
        DatabaseName = "main",
        DupSort = true,
        KeyEncoding = "utf8",
        ValueEncoding = "utf8",
        RowCount = rows.Count,
        SourcePath = sourcePath,
        SourceLength = new FileInfo(sourcePath).Length,
        SourceSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(sourcePath))).ToLowerInvariant(),
    };
    var manifestPath = Path.Combine(root, "category_parent.lmdb.manifest.json");
    File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
    return manifestPath;
}

static string HashBuckets(IEnumerable<IndexedRelationBucket> buckets)
{
    return HashRows(buckets.Select(bucket => new object[]
    {
        bucket.Key,
        bucket.Rows.Count,
        string.Join(",", bucket.Rows.Select(row => Convert.ToString(row[1], System.Globalization.CultureInfo.InvariantCulture)).OrderBy(value => value, StringComparer.Ordinal))
    }));
}

static void Emit(
    string scale,
    string mode,
    int rowCount,
    int distinctCategories,
    int lookupKeys,
    long artifactBytes,
    double openMs,
    double lookupMs,
    double bucketMs,
    double scanMs,
    long retainedBytes,
    string scanHash,
    string lookupHash,
    string bucketHash)
{
    Console.WriteLine(string.Join('\t', new[]
    {
        scale,
        mode,
        rowCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
        distinctCategories.ToString(System.Globalization.CultureInfo.InvariantCulture),
        lookupKeys.ToString(System.Globalization.CultureInfo.InvariantCulture),
        artifactBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
        openMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        lookupMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        bucketMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        scanMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        retainedBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
        scanHash,
        lookupHash,
        bucketHash,
    }));
}

static void BenchmarkProvider(
    string scale,
    string mode,
    Func<IRelationProvider> openProvider,
    PredicateId predicate,
    IReadOnlyList<object> lookupKeys,
    int lookupRepetitions,
    long artifactBytes,
    int rowCount,
    int distinctCategories)
{
    GC.Collect();
    GC.WaitForPendingFinalizers();
    GC.Collect();
    var before = GC.GetTotalMemory(forceFullCollection: true);
    IRelationProvider? provider = null;
    var openMs = Measure(() => provider = openProvider());
    var after = GC.GetTotalMemory(forceFullCollection: true);
    var retainedBytes = Math.Max(0L, after - before);

    var lookupRows = new List<object[]>();
    var lookupMs = Measure(() =>
    {
        for (var iteration = 0; iteration < lookupRepetitions; iteration++)
        {
            if (provider is IIndexedRelationProvider indexed &&
                indexed.TryLookupFacts(predicate, 0, lookupKeys, out var indexedRows))
            {
                lookupRows.AddRange(indexedRows);
            }
            else
            {
                var keySet = lookupKeys.Select(Convert.ToString).ToHashSet(StringComparer.Ordinal);
                lookupRows.AddRange(provider!.GetFacts(predicate).Where(row => keySet.Contains(Convert.ToString(row[0]))));
            }
        }
    });

    string bucketHash = "";
    var bucketMs = Measure(() =>
    {
        if (provider is IIndexedRelationBucketProvider bucketProvider &&
            bucketProvider.TryReadIndexedBuckets(predicate, 0, out var buckets))
        {
            bucketHash = HashBuckets(buckets);
        }
        else
        {
            bucketHash = HashBuckets(provider!.GetFacts(predicate)
                .GroupBy(row => row[0])
                .OrderBy(group => Convert.ToString(group.Key, System.Globalization.CultureInfo.InvariantCulture), StringComparer.Ordinal)
                .Select(group => new IndexedRelationBucket(group.Key, group.ToArray())));
        }
    });

    object[][] scanRows = Array.Empty<object[]>();
    var scanMs = Measure(() => scanRows = provider!.GetFacts(predicate).ToArray());

    Emit(
        scale,
        mode,
        rowCount,
        distinctCategories,
        lookupKeys.Count,
        artifactBytes,
        openMs,
        lookupMs,
        bucketMs,
        scanMs,
        retainedBytes,
        HashRows(scanRows),
        HashRows(lookupRows),
        bucketHash);
}

var scale = ReadArg(args, "--scale", "dev");
var scaleDir = ReadArg(args, "--scale-dir", "");
var lookupKeyCount = Math.Max(1, ReadIntArg(args, "--lookup-keys", 64));
var lookupRepetitions = Math.Max(1, ReadIntArg(args, "--lookup-repetitions", 5));
var root = Directory.GetCurrentDirectory();
var predicate = new PredicateId("category_parent", 2);
var rows = ReadAndInternEdges(scaleDir, out var distinctCategories);
var lookupKeys = rows.Select(row => row.Child)
    .Distinct()
    .OrderBy(value => value)
    .Take(Math.Min(lookupKeyCount, rows.Count))
    .Select(value => (object)value.ToString(System.Globalization.CultureInfo.InvariantCulture))
    .ToArray();

var inputPath = Path.Combine(root, "category_parent_ids.tsv");
WriteInput(inputPath, rows);
var source = new DelimitedRelationSource(inputPath, '\t', 1, 2);

var binaryDir = Path.Combine(root, "binary-artifact");
var binaryManifest = BinaryRelationArtifactBuilder.BuildFromDelimited(predicate, source, binaryDir);
var delimitedDir = Path.Combine(root, "delimited-artifact");
var delimitedManifest = DelimitedRelationArtifactBuilder.BuildFromDelimited(predicate, source, delimitedDir);
var lmdbManifest = WriteLmdbArtifact(root, predicate, rows, inputPath);
var lmdbDir = Path.Combine(root, "lmdb-category-parent");
var mmapDir = Path.Combine(root, "mmap-array-artifact");
var mmapManifest = MmapArrayRelationArtifactBuilder.BuildFromDelimited(predicate, source, mmapDir);

Console.WriteLine("scale\tmode\trows\tdistinct_categories\tlookup_keys\tartifact_bytes\topen_ms\tlookup_ms\tbucket_ms\tscan_ms\tretained_bytes\tscan_hash\tlookup_hash\tbucket_hash");
BenchmarkProvider(
    scale,
    "preload",
    () =>
    {
        var provider = new InMemoryRelationProvider();
        provider.AddFacts(predicate, rows.Select(row => new object[] { row.Child, row.Parent }));
        return provider;
    },
    predicate,
    lookupKeys,
    lookupRepetitions,
    0,
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "binary-artifact",
    () =>
    {
        var provider = new BinaryRelationArtifactProvider();
        provider.RegisterArtifact(predicate, binaryManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupRepetitions,
    DirectorySize(binaryDir),
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "delimited-artifact",
    () =>
    {
        var provider = new DelimitedRelationArtifactProvider();
        provider.RegisterArtifact(predicate, delimitedManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupRepetitions,
    DirectorySize(delimitedDir),
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "lmdb",
    () =>
    {
        var provider = new LmdbRelationProvider();
        provider.RegisterArtifact(predicate, lmdbManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupRepetitions,
    DirectorySize(lmdbDir) + new FileInfo(lmdbManifest).Length,
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "mmap-array",
    () =>
    {
        var provider = new MmapArrayRelationArtifactProvider();
        provider.RegisterArtifact(predicate, mmapManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupRepetitions,
    DirectorySize(mmapDir),
    rows.Count,
    distinctCategories);
"""


if __name__ == "__main__":
    raise SystemExit(main())
