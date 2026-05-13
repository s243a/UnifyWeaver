#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from refresh_csharp_query_source_mode_calibration import (  # noqa: E402
    CALIBRATION_ARTIFACT,
    SWEEP_SCRIPT,
    build_refresh_command,
    main,
    parse_args,
)


class CSharpQueryCalibrationRefreshWrapperTests(unittest.TestCase):
    def test_default_command_uses_guarded_quiet_artifact_writer(self) -> None:
        command = build_refresh_command(parse_args([]))

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(SWEEP_SCRIPT))
        self.assertIn("--require-idle", command)
        self.assertEqual(self._option_value(command, "--workloads"), "all")
        self.assertEqual(self._option_value(command, "--scales"), "300,1k,5k,10k")
        self.assertEqual(
            self._option_value(command, "--source-modes"),
            "auto,preload,artifact-prebuilt",
        )
        self.assertEqual(self._option_value(command, "--repetitions"), "1")
        self.assertEqual(self._option_value(command, "--stability-runs"), "3")
        self.assertIn("--compare-calibration", command)
        self.assertEqual(self._option_value(command, "--format"), "none")
        self.assertEqual(
            self._option_value(command, "--calibration-artifact"),
            str(CALIBRATION_ARTIFACT),
        )
        self.assertIn("--write-calibration-artifact", command)

    def test_no_require_idle_removes_idle_guard(self) -> None:
        command = build_refresh_command(parse_args(["--no-require-idle"]))

        self.assertNotIn("--require-idle", command)

    def test_dry_run_prints_command_without_running(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            result = main(["--dry-run", "--no-require-idle"])

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn(str(SWEEP_SCRIPT), output)
        self.assertIn("--format none", output)
        self.assertIn("--write-calibration-artifact", output)

    @staticmethod
    def _option_value(command: list[str], option: str) -> str:
        return command[command.index(option) + 1]


if __name__ == "__main__":
    unittest.main()
