#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_build_reverse_csr_artifact import ROOT


SWEEP = ROOT / "examples" / "benchmark" / "benchmark_reverse_csr_scale_sweep.py"


class BenchmarkReverseCsrScaleSweepTest(unittest.TestCase):
    def test_scale_sweep_reports_size_and_lookup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SWEEP),
                    "--scale",
                    "4x2",
                    "--sample-parents",
                    "3",
                    "--iterations",
                    "1",
                    "--work-dir",
                    str(Path(tmp) / "work"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
            self.assertEqual(
                [row["backend"] for row in rows],
                ["csr_sorted_array", "csr_lmdb_offset", "lmdb"],
            )
            for row in rows:
                self.assertEqual(row["parents"], "4")
                self.assertEqual(row["children_per_parent"], "2")
                self.assertEqual(row["edge_count"], "8")
                self.assertEqual(row["sample_parents"], "3")
                self.assertEqual(row["iterations"], "1")
                self.assertGreater(int(row["csr_artifact_bytes"]), 0)
                self.assertGreater(float(row["csr_bytes_per_edge"]), 0.0)
                self.assertGreater(float(row["csr_bytes_per_parent"]), 0.0)
                self.assertGreater(int(row["parent_lmdb_env_bytes"]), 0)
                self.assertGreater(float(row["parent_lmdb_bytes_per_edge"]), 0.0)

    def test_scale_sweep_rejects_bad_scale(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SWEEP), "--scale", "bad"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid --scale", result.stderr)


if __name__ == "__main__":
    unittest.main()
