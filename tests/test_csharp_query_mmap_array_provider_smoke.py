#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PROJECT = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime"
    / "UnifyWeaver.QueryRuntime.Core.csproj"
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


class CSharpQueryMmapArrayProviderSmokeTests(unittest.TestCase):
    def test_mmap_array_provider_scans_lookups_and_buckets_match_expected_rows(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")

        build_provider = subprocess.run(
            ["dotnet", "build", str(CORE_PROJECT)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        self.assertEqual(
            build_provider.returncode,
            0,
            msg=f"stdout:\n{build_provider.stdout}\nstderr:\n{build_provider.stderr}",
        )

        with tempfile.TemporaryDirectory(prefix="uw-csharp-mmap-array-smoke-") as tmp:
            tmp_path = Path(tmp)
            project_path = tmp_path / "MmapArrayProviderSmoke.csproj"
            program_path = tmp_path / "Program.cs"
            output_dir = tmp_path / "bin" / "Debug" / "net9.0"
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
                      </ItemGroup>
                    </Project>
                    """
                ),
                encoding="utf-8",
            )
            program_path.write_text(SMOKE_PROGRAM, encoding="utf-8")

            build_result = subprocess.run(
                ["dotnet", "build", str(project_path)],
                cwd=tmp_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(
                build_result.returncode,
                0,
                msg=f"stdout:\n{build_result.stdout}\nstderr:\n{build_result.stderr}",
            )

            result = subprocess.run(
                ["dotnet", str(output_dir / "MmapArrayProviderSmoke.dll")],
                cwd=tmp_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("MMAP_ARRAY_PROVIDER_SMOKE_OK", result.stdout)


SMOKE_PROGRAM = r"""
using UnifyWeaver.QueryRuntime;

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static string RowText(object[] row) => string.Join(">", row.Select(Convert.ToString));

var root = Directory.GetCurrentDirectory();
var sourcePath = Path.Combine(root, "edge.tsv");
File.WriteAllLines(sourcePath, new[]
{
    "from\tto",
    "2\t4",
    "1\t3",
    "1\t2",
});

var predicate = new PredicateId("edge", 2);
var source = new DelimitedRelationSource(sourcePath, '\t', 1, 2);
var manifestPath = MmapArrayRelationArtifactBuilder.BuildFromDelimited(predicate, source, Path.Combine(root, "edge-mmap"));

var provider = new MmapArrayRelationArtifactProvider();
provider.RegisterArtifact(predicate, manifestPath);

var scanned = provider.GetFacts(predicate).Select(RowText).ToArray();
Assert(scanned.SequenceEqual(new[] { "1>2", "1>3", "2>4" }), "scan rows did not match");

Assert(provider.TryLookupFacts(predicate, 0, new object[] { "1" }, out var lookupRows), "arg0 lookup was not served");
var lookedUp = lookupRows.Select(RowText).ToArray();
Assert(lookedUp.SequenceEqual(new[] { "1>2", "1>3" }), "arg0 lookup rows did not match");

Assert(provider.TryReadIndexedBuckets(predicate, 0, out var buckets), "arg0 bucket stream was not served");
var bucketText = buckets.Select(bucket => $"{bucket.Key}:{string.Join(",", bucket.Rows.Select(RowText))}").ToArray();
Assert(bucketText.SequenceEqual(new[] { "1:1>2,1>3", "2:2>4" }), "bucket rows did not match");

Assert(provider.TryGetRelationCardinality(predicate, out var rowCount), "row count was not served");
Assert(rowCount == 3, $"row count was {rowCount}, expected 3");

var factory = new DefaultRelationArtifactProviderFactory();
Assert(factory.TryOpen(predicate, manifestPath, fallback: null, out var opened), "factory did not open mmap array manifest");
Assert(opened.StorageKind == "mmap_array_artifact", $"storage kind was {opened.StorageKind}");
var factoryRows = opened.Provider.GetFacts(predicate).Select(RowText).ToArray();
Assert(factoryRows.SequenceEqual(scanned), "factory provider scan rows did not match");

Console.WriteLine("MMAP_ARRAY_PROVIDER_SMOKE_OK");
"""


if __name__ == "__main__":
    unittest.main()
