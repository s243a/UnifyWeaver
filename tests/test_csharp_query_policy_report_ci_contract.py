#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_effective_distance_artifact_backends.py"
WRAPPER_SCRIPT = ROOT / "examples" / "benchmark" / "review_csharp_query_effective_distance_policy.py"
SUMMARY_HEADERS = [
    "scale",
    "relation",
    "rows",
    "distinct_categories",
    "lookup_keys",
    "best_lookup_mode",
    "best_lookup_col1_mode",
    "best_bucket_mode",
    "best_bucket_col1_mode",
    "best_scan_mode",
    "smallest_artifact_mode",
    "lookup_ms_by_mode",
    "lookup_col1_ms_by_mode",
    "bucket_ms_by_mode",
    "bucket_col1_ms_by_mode",
    "scan_ms_by_mode",
    "artifact_bytes_by_mode",
]


class CSharpQueryPolicyReportCiContractTests(unittest.TestCase):
    def test_help_exposes_policy_report_contract_flags(self) -> None:
        result = subprocess.run(
            ["python3", str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        for expected in (
            "--summary-input",
            "--summary-output",
            "--policy-action-threshold",
            "--fail-on-policy-actions",
            "policy-actionable-markdown",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stdout)

    def test_policy_review_wrapper_is_in_ci_contract(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()

        self.assertIn("python3 -m unittest tests/test_csharp_query_policy_review_wrapper.py", workflow)
        self.assertIn(
            "python3 examples/benchmark/review_csharp_query_effective_distance_policy.py --dry-run",
            workflow,
        )

    def test_policy_review_wrapper_help_exposes_safe_contract(self) -> None:
        result = subprocess.run(
            ["python3", str(WRAPPER_SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        for expected in (
            "--summary-input",
            "--summary-output",
            "--policy-action-threshold",
            "--no-fail-on-policy-actions",
            "--dry-run",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stdout)

    def test_summary_input_fail_gate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.tsv"
            summary_path.write_text(
                "\t".join(SUMMARY_HEADERS)
                + "\n"
                + "\t".join(
                    [
                        "enwiki_page_5m",
                        "article_category",
                        "5000000",
                        "1918305",
                        "128",
                        "lmdb",
                        "mmap-array",
                        "mmap-array",
                        "mmap-array",
                        "mmap-array",
                        "mmap-array",
                        "lmdb:3.251|mmap-array:3.288",
                        "lmdb:1219.110|mmap-array:236.382",
                        "delimited-artifact:400.000|mmap-array:350.000",
                        "mmap-array:350.000",
                        "mmap-array:850.000",
                        "lmdb:142443019|mmap-array:80000643",
                    ]
                )
                + "\n"
            )

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--summary-input",
                    str(summary_path),
                    "--format",
                    "policy-actionable-markdown",
                    "--policy-action-threshold",
                    "1.10",
                    "--fail-on-policy-actions",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        self.assertEqual(result.returncode, 2, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("| Policy mode | Policy artifact value | Policy vs best |", result.stdout)
        self.assertIn("| bucket_c0 | ms | delimited-artifact | 400.000 | 1.143x | mmap-array | 350.000 | diff |", result.stdout)


if __name__ == "__main__":
    unittest.main()
