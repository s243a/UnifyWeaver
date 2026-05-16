#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_effective_distance_artifact_backends.py"
LIGHTNINGDB_PACKAGE = Path.home() / ".nuget" / "packages" / "lightningdb" / "0.21.0"


class CSharpQueryEffectiveDistanceArtifactBackendTests(unittest.TestCase):
    def test_dev_scale_reports_all_backends_with_matching_hashes(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
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
        self.assertEqual(len(lines), 6)
        headers = lines[0].split("\t")
        rows = [dict(zip(headers, line.split("\t"))) for line in lines[1:]]

        self.assertEqual(
            [row["mode"] for row in rows],
            ["preload", "binary-artifact", "delimited-artifact", "lmdb", "mmap-array"],
        )
        self.assertEqual({row["scale"] for row in rows}, {"dev"})
        self.assertEqual({row["run"] for row in rows}, {"1"})
        self.assertEqual({row["scan_hash"] for row in rows}, {rows[0]["scan_hash"]})
        self.assertEqual({row["lookup_hash"] for row in rows}, {rows[0]["lookup_hash"]})
        self.assertEqual({row["bucket_hash"] for row in rows}, {rows[0]["bucket_hash"]})

        for row in rows:
            self.assertGreater(int(row["rows"]), 0)
            self.assertGreater(int(row["distinct_categories"]), 0)
            self.assertEqual(row["lookup_keys"], "4")
            for column in ("artifact_bytes", "open_ms", "lookup_ms", "bucket_ms", "scan_ms", "retained_bytes"):
                float(row[column])

    def test_summary_markdown_reports_best_modes(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
                "--format",
                "summary-markdown",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("| Scale | Rows | Categories | Lookup keys | Best lookup |", result.stdout)
        self.assertIn("| dev |", result.stdout)
        self.assertIn("Smallest artifact", result.stdout)


if __name__ == "__main__":
    unittest.main()
