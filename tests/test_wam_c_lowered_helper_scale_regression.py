#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_effective_distance_matrix.py"


class WamCLoweredHelperScaleRegressionTests(unittest.TestCase):
    def test_dev_and_10x_scales_preserve_lowered_helper_output_parity(self) -> None:
        if shutil.which("swipl") is None:
            self.skipTest("swipl is not available")
        if shutil.which("gcc") is None:
            self.skipTest("gcc is not available")

        result = subprocess.run(
            [
                sys.executable,
                str(MATRIX_SCRIPT),
                "--scales",
                "dev,10x",
                "--target-sets",
                "c-wam-lowered-helper",
                "--repetitions",
                "1",
                "--baseline-target",
                "c-wam-lowered-helper-interpreted",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )

        if result.returncode != 0:
            self.fail(
                "lowered-helper scale matrix failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        target_rows = self._parse_target_rows(result.stdout)
        targets = {
            "c-wam-lowered-helper",
            "c-wam-lowered-helper-interpreted",
        }

        for scale, expected_rows in {"dev": 16, "10x": 160}.items():
            rows_for_scale = {
                row["target"]: row
                for row in target_rows
                if row["scale"] == scale and row["target"] in targets
            }
            self.assertEqual(set(rows_for_scale), targets)

            for target, row in rows_for_scale.items():
                with self.subTest(scale=scale, target=target):
                    self.assertEqual(row["status"], "ok")
                    self.assertEqual(row["rows"], str(expected_rows))

            self.assertEqual(
                rows_for_scale["c-wam-lowered-helper"]["stdout_sha256"],
                rows_for_scale["c-wam-lowered-helper-interpreted"]["stdout_sha256"],
            )

        self.assertNotEqual(
            {
                row["stdout_sha256"]
                for row in target_rows
                if row["scale"] == "dev" and row["target"] in targets
            },
            {
                row["stdout_sha256"]
                for row in target_rows
                if row["scale"] == "10x" and row["target"] in targets
            },
        )
        self.assertIn("dev\tall_outputs\tmatch", result.stdout)
        self.assertIn("10x\tall_outputs\tmatch", result.stdout)
        self.assertIn("dev\thybrid-wam-lowered-helper_outputs\tmatch", result.stdout)
        self.assertIn("10x\thybrid-wam-lowered-helper_outputs\tmatch", result.stdout)

    def _parse_target_rows(self, stdout: str) -> list[dict[str, str]]:
        lines = [line for line in stdout.splitlines() if line]
        self.assertTrue(lines, "matrix output should not be empty")
        header = lines[0].split("\t")
        self.assertEqual(
            header,
            [
                "scale",
                "target",
                "category",
                "status",
                "median_s",
                "min_s",
                "max_s",
                "rows",
                "stdout_sha256",
                "message",
            ],
        )

        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) > len(header):
                continue
            parts.extend([""] * (len(header) - len(parts)))
            row = dict(zip(header, parts))
            if row["target"].startswith("c-wam-lowered-helper"):
                rows.append(row)
        return rows


if __name__ == "__main__":
    unittest.main()
