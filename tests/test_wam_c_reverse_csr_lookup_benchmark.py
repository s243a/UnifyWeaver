#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "examples" / "benchmark" / "benchmark_wam_c_reverse_csr_lookup.py"


class WamCReverseCsrLookupBenchmarkTests(unittest.TestCase):
    def test_dry_run_reports_scales_and_modes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--dry-run",
                "--scales",
                "dev,10k",
                "--modes",
                "sorted_array,lmdb_offset",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scale=dev", result.stdout)
        self.assertIn("scale=10k", result.stdout)
        self.assertIn("modes=sorted_array,lmdb_offset", result.stdout)

    def test_rejects_unknown_mode(self) -> None:
        result = subprocess.run(
            [sys.executable, str(BENCHMARK), "--dry-run", "--modes", "unknown"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported mode", result.stderr)

    def test_dev_benchmark_reports_wam_c_runtime_lookup_rows(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--scales",
                "dev",
                "--modes",
                "sorted_array,lmdb_offset",
                "--sample-parents",
                "3",
                "--iterations",
                "1",
                "--warmups",
                "0",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
        self.assertEqual([row["mode"] for row in rows], ["sorted_array", "lmdb_offset"])
        for row in rows:
            self.assertEqual(row["scale"], "dev")
            self.assertEqual(row["sample_parents"], "3")
            self.assertEqual(row["iterations"], "1")
            self.assertGreater(int(row["parent_count"]), 0)
            self.assertGreater(int(row["edge_count"]), 0)
            self.assertGreater(int(row["total_children"]), 0)
            self.assertGreater(int(row["checksum"]), 0)
            self.assertGreater(float(row["median_ms"]), 0.0)
            self.assertGreater(float(row["lookup_us_per_parent"]), 0.0)
            self.assertGreater(int(row["reverse_csr_index_bytes"]), 0)
            self.assertGreater(int(row["reverse_csr_values_bytes"]), 0)
        self.assertEqual(int(rows[0]["reverse_csr_offsets_lmdb_bytes"]), 0)
        self.assertGreater(int(rows[1]["reverse_csr_offsets_lmdb_bytes"]), 0)


if __name__ == "__main__":
    unittest.main()
