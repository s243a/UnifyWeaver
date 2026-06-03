#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "examples" / "benchmark" / "benchmark_wam_c_bidirectional_kernel.py"


class WamCBidirectionalKernelBenchmarkTests(unittest.TestCase):
    def test_dry_run_reports_scales_modes_and_sampling(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--dry-run",
                "--scales",
                "dev,10k",
                "--modes",
                "scan,sorted_array",
                "--sample-queries",
                "7",
                "--sample-roots",
                "2",
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
        self.assertIn("modes=scan,sorted_array", result.stdout)
        self.assertIn("sample_queries=7", result.stdout)
        self.assertIn("sample_roots=2", result.stdout)

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

    def test_dev_benchmark_reports_kernel_rows(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--scales",
                "dev",
                "--modes",
                "scan,sorted_array",
                "--sample-queries",
                "3",
                "--sample-roots",
                "1",
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
        self.assertEqual([row["mode"] for row in rows], ["scan", "sorted_array"])
        for row in rows:
            self.assertEqual(row["scale"], "dev")
            self.assertEqual(row["sample_queries"], "3")
            self.assertEqual(row["sample_roots"], "1")
            self.assertEqual(row["iterations"], "1")
            self.assertEqual(row["warmups"], "0")
            self.assertGreater(int(row["parent_edge_count"]), 0)
            self.assertGreater(int(row["category_count"]), 0)
            self.assertGreater(float(row["setup_ms"]), 0.0)
            self.assertGreater(float(row["parent_edge_tsv_load_ms"]), 0.0)
            self.assertGreater(float(row["parent_edge_register_ms"]), 0.0)
            self.assertGreater(float(row["category_id_load_ms"]), 0.0)
            self.assertGreater(float(row["query_load_ms"]), 0.0)
            self.assertGreaterEqual(float(row["reverse_csr_load_ms"]), 0.0)
            self.assertGreaterEqual(float(row["csr_attach_ms"]), 0.0)
            self.assertGreater(float(row["kernel_register_ms"]), 0.0)
            self.assertGreater(float(row["median_ms"]), 0.0)
            self.assertGreater(float(row["query_us_per_query"]), 0.0)
            self.assertGreaterEqual(int(row["total_results"]), 0)
            self.assertGreater(float(row["harness_compile_s"]), 0.0)
            self.assertGreaterEqual(float(row["query_sample_s"]), 0.0)
        self.assertEqual(int(rows[0]["reverse_csr_index_bytes"]), 0)
        self.assertGreater(int(rows[1]["reverse_csr_index_bytes"]), 0)
        self.assertGreater(int(rows[1]["reverse_csr_values_bytes"]), 0)
        self.assertEqual(rows[0]["checksum"], rows[1]["checksum"])


if __name__ == "__main__":
    unittest.main()
