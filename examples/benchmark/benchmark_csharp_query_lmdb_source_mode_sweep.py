#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    "mode",
    "rows",
    "lookup_keys",
    "artifact_bytes",
    "open_ms",
    "lookup_ms",
    "scan_ms",
    "retained_bytes",
    "scan_hash",
    "lookup_hash",
]


@dataclass(frozen=True)
class BenchmarkRow:
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
    project_path = project_dir / "CSharpQueryLmdbSourceModeBench.csproj"
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


def parse_rows(output: str) -> list[BenchmarkRow]:
    reader = csv.DictReader(output.splitlines(), delimiter="\t")
    rows = [BenchmarkRow(dict(row)) for row in reader]
    if reader.fieldnames != HEADERS:
        raise RuntimeError(f"unexpected benchmark headers: {reader.fieldnames}")
    return rows


def run_benchmark(rows: int, keys: int, lookup_repetitions: int, keep_temp: bool) -> list[BenchmarkRow]:
    build_lmdb_runtime()
    temp_path = Path(tempfile.mkdtemp(prefix="uw-csharp-lmdb-source-mode-"))
    try:
        project_dir = temp_path
        project_path = write_benchmark_project(project_dir)
        run_checked(["dotnet", "build", str(project_path)], cwd=project_dir, timeout=120)

        output_dir = project_dir / "bin" / "Debug" / "net9.0"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            str(output_dir)
            if not env.get("LD_LIBRARY_PATH")
            else f"{output_dir}:{env['LD_LIBRARY_PATH']}"
        )
        result = run_checked(
            [
                "dotnet",
                str(output_dir / "CSharpQueryLmdbSourceModeBench.dll"),
                "--rows",
                str(rows),
                "--keys",
                str(keys),
                "--lookup-repetitions",
                str(lookup_repetitions),
            ],
            cwd=project_dir,
            timeout=180,
        )
        return parse_rows(result.stdout)
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
        "| Mode | Rows | Lookup keys | Artifact bytes | Open ms | Lookup ms | Scan ms | Retained bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        value = row.values
        lines.append(
            "| {mode} | {rows} | {lookup_keys} | {artifact_bytes} | {open_ms} | {lookup_ms} | {scan_ms} | {retained_bytes} |".format(
                **value
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare C# query relation source providers against one LMDB-friendly arity-2 workload."
    )
    parser.add_argument("--rows", type=int, default=1000, help="synthetic edge/2 rows to generate")
    parser.add_argument("--keys", type=int, default=100, help="distinct arg0 keys to probe")
    parser.add_argument(
        "--lookup-repetitions",
        type=int,
        default=10,
        help="number of full lookup-key passes to time per mode",
    )
    parser.add_argument("--format", choices=("tsv", "markdown"), default="tsv")
    parser.add_argument("--keep-temp", action="store_true", help="keep the generated C# benchmark project")
    args = parser.parse_args(argv)

    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.keys <= 0:
        parser.error("--keys must be positive")
    if args.lookup_repetitions <= 0:
        parser.error("--lookup-repetitions must be positive")

    rows = run_benchmark(args.rows, args.keys, args.lookup_repetitions, args.keep_temp)
    if args.format == "markdown":
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

static int ReadIntArg(string[] args, string name, int defaultValue)
{
    for (var index = 0; index < args.Length - 1; index++)
    {
        if (args[index] == name && int.TryParse(args[index + 1], out var parsed))
        {
            return parsed;
        }
    }

    return defaultValue;
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

static void WriteInput(string path, IReadOnlyList<(string Key, string Value)> rows)
{
    using var writer = new StreamWriter(path, append: false, Encoding.UTF8);
    writer.WriteLine("from\tto");
    foreach (var row in rows)
    {
        writer.Write(row.Key);
        writer.Write('\t');
        writer.Write(row.Value);
        writer.WriteLine();
    }
}

static string WriteLmdbArtifact(
    string root,
    PredicateId predicate,
    IReadOnlyList<(string Key, string Value)> rows,
    string sourcePath)
{
    var envPath = Path.Combine(root, "lmdb-edge");
    Directory.CreateDirectory(envPath);
    using (var env = new LightningEnvironment(envPath) { MaxDatabases = 4, MapSize = 128L * 1024L * 1024L })
    {
        env.Open();
        using var tx = env.BeginTransaction();
        using var db = tx.OpenDatabase("main", new DatabaseConfiguration
        {
            Flags = DatabaseOpenFlags.Create | DatabaseOpenFlags.DuplicatesSort,
        });

        foreach (var row in rows)
        {
            tx.Put(db, Encoding.UTF8.GetBytes(row.Key), Encoding.UTF8.GetBytes(row.Value));
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
    var manifestPath = Path.Combine(root, "edge.lmdb.manifest.json");
    File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
    return manifestPath;
}

static void Emit(
    string mode,
    int rowCount,
    int lookupKeys,
    long artifactBytes,
    double openMs,
    double lookupMs,
    double scanMs,
    long retainedBytes,
    string scanHash,
    string lookupHash)
{
    Console.WriteLine(string.Join('\t', new[]
    {
        mode,
        rowCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
        lookupKeys.ToString(System.Globalization.CultureInfo.InvariantCulture),
        artifactBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
        openMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        lookupMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        scanMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        retainedBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
        scanHash,
        lookupHash,
    }));
}

static void BenchmarkProvider(
    string mode,
    Func<IRelationProvider> openProvider,
    PredicateId predicate,
    IReadOnlyList<object> lookupKeys,
    int lookupRepetitions,
    long artifactBytes,
    int rowCount)
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

    object[][] scanRows = Array.Empty<object[]>();
    var scanMs = Measure(() => scanRows = provider!.GetFacts(predicate).ToArray());

    Emit(
        mode,
        rowCount,
        lookupKeys.Count,
        artifactBytes,
        openMs,
        lookupMs,
        scanMs,
        retainedBytes,
        HashRows(scanRows),
        HashRows(lookupRows));
}

var rowCount = Math.Max(1, ReadIntArg(args, "--rows", 1000));
var keyCount = Math.Max(1, ReadIntArg(args, "--keys", 100));
var lookupRepetitions = Math.Max(1, ReadIntArg(args, "--lookup-repetitions", 10));
var root = Directory.GetCurrentDirectory();
var predicate = new PredicateId("edge", 2);
var rows = Enumerable.Range(0, rowCount)
    .Select(index => (Key: $"k{index % keyCount:D6}", Value: $"v{index:D8}"))
    .ToArray();
var lookupKeys = Enumerable.Range(0, Math.Min(keyCount, rowCount))
    .Select(index => (object)$"k{index:D6}")
    .ToArray();

var inputPath = Path.Combine(root, "edge.tsv");
WriteInput(inputPath, rows);
var source = new DelimitedRelationSource(inputPath, '\t', 1, 2);

var binaryDir = Path.Combine(root, "binary-artifact");
var binaryManifest = BinaryRelationArtifactBuilder.BuildFromDelimited(predicate, source, binaryDir);
var delimitedDir = Path.Combine(root, "delimited-artifact");
var delimitedManifest = DelimitedRelationArtifactBuilder.BuildFromDelimited(predicate, source, delimitedDir);
var lmdbManifest = WriteLmdbArtifact(root, predicate, rows, inputPath);
var lmdbDir = Path.Combine(root, "lmdb-edge");

Console.WriteLine("mode\trows\tlookup_keys\tartifact_bytes\topen_ms\tlookup_ms\tscan_ms\tretained_bytes\tscan_hash\tlookup_hash");
BenchmarkProvider(
    "preload",
    () =>
    {
        var provider = new InMemoryRelationProvider();
        provider.AddFacts(predicate, rows.Select(row => new object[] { row.Key, row.Value }));
        return provider;
    },
    predicate,
    lookupKeys,
    lookupRepetitions,
    0,
    rowCount);
BenchmarkProvider(
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
    rowCount);
BenchmarkProvider(
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
    rowCount);
BenchmarkProvider(
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
    rowCount);
"""


if __name__ == "__main__":
    raise SystemExit(main())
