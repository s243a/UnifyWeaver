#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_effective_distance_artifact_backends.py"
LIGHTNINGDB_PACKAGE = Path.home() / ".nuget" / "packages" / "lightningdb" / "0.21.0"

SPEC = importlib.util.spec_from_file_location("effective_distance_artifact_backends", SCRIPT)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CSharpQueryEffectiveDistanceArtifactBackendTests(unittest.TestCase):
    def test_large_scale_helpers_recognize_category_scales(self) -> None:
        self.assertFalse(MODULE.is_large_scale("dev"))
        self.assertFalse(MODULE.is_large_scale("10k"))
        self.assertTrue(MODULE.is_large_scale("50k_cats"))
        self.assertTrue(MODULE.is_large_scale("100k_cats"))
        self.assertTrue(MODULE.is_large_scale("500k_cats"))
        self.assertTrue(MODULE.is_large_scale("1m_cats"))
        self.assertTrue(MODULE.is_large_scale("1m"))
        self.assertEqual(MODULE.scale_seed_cap("50k_cats"), 50_000)
        self.assertIsNone(MODULE.scale_seed_cap("100k_cats"))

    def test_parse_mem_available_mib(self) -> None:
        self.assertEqual(
            MODULE.parse_mem_available_mib("MemTotal: 4096000 kB\nMemAvailable: 2097152 kB\n"),
            2048,
        )
        self.assertIsNone(MODULE.parse_mem_available_mib("MemTotal: 4096000 kB\n"))

    def test_unsupported_auto_preparation_scale_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "automatic fixture preparation only supports"):
            MODULE.scale_seed_cap("1m_cats")

    def test_skip_missing_scale_errors_when_nothing_remains(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "__definitely_missing_scale__",
                "--skip-missing-scales",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all requested scales were missing", result.stderr)

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
        self.assertEqual({row["lookup_col1_hash"] for row in rows}, {rows[0]["lookup_col1_hash"]})
        self.assertEqual({row["bucket_hash"] for row in rows}, {rows[0]["bucket_hash"]})
        self.assertEqual({row["bucket_col1_hash"] for row in rows}, {rows[0]["bucket_col1_hash"]})

        for row in rows:
            self.assertGreater(int(row["rows"]), 0)
            self.assertGreater(int(row["distinct_categories"]), 0)
            self.assertEqual(row["lookup_keys"], "4")
            for column in ("artifact_bytes", "open_ms", "lookup_ms", "lookup_col1_ms", "bucket_ms", "bucket_col1_ms", "scan_ms", "retained_bytes"):
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
        self.assertIn("| Scale | Rows | Categories | Lookup keys | Best lookup c0 | Best lookup c1 |", result.stdout)
        self.assertIn("| dev |", result.stdout)
        self.assertIn("Smallest artifact", result.stdout)

    def test_summary_full_markdown_reports_timing_details(self) -> None:
        row = MODULE.SummaryRow(
            {
                "scale": "rust_lmdb_500k",
                "rows": "500000",
                "distinct_categories": "225697",
                "lookup_keys": "64",
                "best_lookup_mode": "mmap-array",
                "best_lookup_col1_mode": "mmap-array",
                "best_bucket_mode": "mmap-array",
                "best_bucket_col1_mode": "mmap-array",
                "best_scan_mode": "preload",
                "smallest_artifact_mode": "mmap-array",
                "lookup_ms_by_mode": "lmdb:2.087|mmap-array:1.250",
                "lookup_col1_ms_by_mode": "lmdb:2.500|mmap-array:1.750",
                "bucket_ms_by_mode": "lmdb:718.217|mmap-array:250.000",
                "bucket_col1_ms_by_mode": "lmdb:701.000|mmap-array:240.000",
                "scan_ms_by_mode": "lmdb:129.326|preload:1.000",
                "artifact_bytes_by_mode": "lmdb:27971980|mmap-array:4000000",
            }
        )
        output = MODULE.render_summary_full_markdown([row])
        self.assertIn("Lookup c0 ms by mode", output)
        self.assertIn("Lookup c1 ms by mode", output)
        self.assertIn("lmdb:2.087|mmap-array:1.250", output)
        self.assertIn("lmdb:2.500|mmap-array:1.750", output)
        self.assertIn("Artifact bytes by mode", output)

    def test_artifact_root_reuses_existing_manifests(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            command = [
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
                "--artifact-root",
                str(artifact_root),
                "--format",
                "summary-markdown",
            ]
            first = subprocess.run(
                command + ["--refresh-artifacts"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            self.assertEqual(first.returncode, 0, msg=f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}")
            manifest = artifact_root / "dev" / "category_parent.lmdb.manifest.json"
            self.assertTrue(manifest.exists())
            first_mtime = manifest.stat().st_mtime_ns

            second = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            self.assertEqual(second.returncode, 0, msg=f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}")
            self.assertEqual(manifest.stat().st_mtime_ns, first_mtime)


if __name__ == "__main__":
    unittest.main()
