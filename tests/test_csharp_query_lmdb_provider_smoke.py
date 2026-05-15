#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
import unittest
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


class CSharpQueryLmdbProviderSmokeTests(unittest.TestCase):
    def test_lmdb_provider_scans_and_arg0_lookups_match_expected_rows(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        build_provider = subprocess.run(
            ["dotnet", "build", str(LMDB_PROJECT)],
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

        with tempfile.TemporaryDirectory(prefix="uw-csharp-lmdb-smoke-") as tmp:
            tmp_path = Path(tmp)
            project_path = tmp_path / "LmdbProviderSmoke.csproj"
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

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = (
                str(output_dir)
                if not env.get("LD_LIBRARY_PATH")
                else f"{output_dir}:{env['LD_LIBRARY_PATH']}"
            )
            result = subprocess.run(
                ["dotnet", str(output_dir / "LmdbProviderSmoke.dll")],
                cwd=tmp_path,
                env=env,
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
            self.assertIn("LMDB_PROVIDER_SMOKE_OK", result.stdout)


SMOKE_PROGRAM = r"""
using System.Text;
using System.Text.Json;
using LightningDB;
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
var envPath = Path.Combine(root, "edges.lmdb");
Directory.CreateDirectory(envPath);

using (var env = new LightningEnvironment(envPath) { MaxDatabases = 4 })
{
    env.Open();
    using var tx = env.BeginTransaction();
    using var db = tx.OpenDatabase("main", new DatabaseConfiguration
    {
        Flags = DatabaseOpenFlags.Create | DatabaseOpenFlags.DuplicatesSort,
    });

    tx.Put(db, Encoding.UTF8.GetBytes("alice"), Encoding.UTF8.GetBytes("bob"));
    tx.Put(db, Encoding.UTF8.GetBytes("alice"), Encoding.UTF8.GetBytes("carol"));
    tx.Put(db, Encoding.UTF8.GetBytes("bob"), Encoding.UTF8.GetBytes("dave"));
    tx.Commit();
}

var manifestPath = Path.Combine(root, "edge.lmdb.manifest.json");
var manifest = new LmdbRelationArtifactManifest
{
    PredicateName = "edge",
    Arity = 2,
    EnvironmentPath = "edges.lmdb",
    DatabaseName = "main",
    DupSort = true,
    KeyEncoding = "utf8",
    ValueEncoding = "utf8",
    RowCount = 3,
};
File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest));

var predicate = new PredicateId("edge", 2);
var provider = new LmdbRelationProvider();
provider.RegisterArtifact(predicate, manifestPath);

var scanned = provider.GetFacts(predicate).Select(RowText).OrderBy(value => value).ToArray();
Assert(scanned.SequenceEqual(new[] { "alice>bob", "alice>carol", "bob>dave" }), "scan rows did not match");

Assert(provider.TryLookupFacts(predicate, 0, new object[] { "alice" }, out var lookupRows), "arg0 lookup was not served");
var lookedUp = lookupRows.Select(RowText).OrderBy(value => value).ToArray();
Assert(lookedUp.SequenceEqual(new[] { "alice>bob", "alice>carol" }), "arg0 lookup rows did not match");

Assert(provider.TryGetRelationCardinality(predicate, out var rowCount), "row count was not served");
Assert(rowCount == 3, $"row count was {rowCount}, expected 3");

var factory = new LmdbRelationArtifactProviderFactory();
Assert(factory.TryOpen(predicate, manifestPath, fallback: null, out var opened), "factory did not open LMDB manifest");
Assert(opened.StorageKind == "lmdb_artifact", $"storage kind was {opened.StorageKind}");
var factoryRows = opened.Provider.GetFacts(predicate).Select(RowText).OrderBy(value => value).ToArray();
Assert(factoryRows.SequenceEqual(scanned), "factory provider scan rows did not match");

Console.WriteLine("LMDB_PROVIDER_SMOKE_OK");
"""


if __name__ == "__main__":
    unittest.main()
