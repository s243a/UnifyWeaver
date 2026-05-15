#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_lmdb_source_mode_sweep.py"
LIGHTNINGDB_PACKAGE = Path.home() / ".nuget" / "packages" / "lightningdb" / "0.21.0"


class CSharpQueryLmdbSourceModeBenchmarkTests(unittest.TestCase):
    def test_small_sweep_reports_all_modes_with_matching_hashes(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--rows",
                "30",
                "--keys",
                "5",
                "--lookup-repetitions",
                "2",
                "--format",
                "tsv",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

        lines = result.stdout.strip().splitlines()
        self.assertGreaterEqual(len(lines), 6)
        headers = lines[0].split("\t")
        rows = [dict(zip(headers, line.split("\t"))) for line in lines[1:]]

        self.assertEqual(
            [row["mode"] for row in rows],
            ["preload", "binary-artifact", "delimited-artifact", "lmdb", "mmap-array"],
        )
        self.assertEqual({row["scan_hash"] for row in rows}, {rows[0]["scan_hash"]})
        self.assertEqual({row["lookup_hash"] for row in rows}, {rows[0]["lookup_hash"]})

        for row in rows:
            self.assertEqual(row["rows"], "30")
            self.assertEqual(row["lookup_keys"], "5")
            for column in ("artifact_bytes", "open_ms", "lookup_ms", "scan_ms", "retained_bytes"):
                float(row[column])


if __name__ == "__main__":
    unittest.main()
