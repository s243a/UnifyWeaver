#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_read_reverse_csr_artifact import make_phase1_lmdb
from test_build_reverse_csr_artifact import ROOT


BENCHMARK = ROOT / "examples" / "benchmark" / "benchmark_reverse_csr_lookup.py"


class BenchmarkReverseCsrLookupTest(unittest.TestCase):
    def test_benchmark_reports_csr_and_lmdb_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            phase1 = tmp_path / "phase1.lmdb"
            csr_dir = tmp_path / "category_child_csr"
            make_phase1_lmdb(phase1)

            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK),
                    str(phase1),
                    "--csr-dir",
                    str(csr_dir),
                    "--sample-parents",
                    "2",
                    "--iterations",
                    "2",
                    "--seed",
                    "7",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
            self.assertEqual([row["backend"] for row in rows], ["csr", "lmdb"])
            for row in rows:
                self.assertEqual(row["sample_parents"], "2")
                self.assertEqual(row["iterations"], "2")
                self.assertEqual(row["total_children"], "4")
                self.assertGreater(float(row["median_ms"]), 0.0)
                self.assertGreater(int(row["csr_artifact_bytes"]), 0)
                self.assertGreater(int(row["phase1_lmdb_env_bytes"]), 0)

    def test_benchmark_detects_missing_phase1_lmdb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.lmdb"
            result = subprocess.run(
                [sys.executable, str(BENCHMARK), str(missing)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
