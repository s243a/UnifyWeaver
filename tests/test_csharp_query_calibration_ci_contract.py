#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


class CSharpQueryCalibrationCiContractTests(unittest.TestCase):
    def test_ci_runs_calibration_artifact_and_wrapper_contracts(self) -> None:
        workflow = WORKFLOW.read_text()

        required_commands = [
            "python3 -m unittest tests/test_csharp_query_source_mode_policy.py",
            "python3 -m unittest tests/test_csharp_query_calibration_refresh_wrapper.py",
            "python3 examples/benchmark/refresh_csharp_query_source_mode_calibration.py --dry-run",
            "python3 -m unittest tests/test_csharp_query_scan_calibration_refresh_wrapper.py",
            "python3 examples/benchmark/refresh_csharp_query_scan_source_mode_calibration.py --dry-run",
            "python3 -m unittest tests/test_csharp_query_calibration_ci_contract.py",
        ]

        for command in required_commands:
            with self.subTest(command=command):
                self.assertIn(command, workflow)


if __name__ == "__main__":
    unittest.main()
