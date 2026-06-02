#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_wam_c_child_search_runtime_sweep import (  # noqa: E402
    CHILD_SEARCH_TARGETS,
    DEFAULT_REPETITIONS,
    DEFAULT_TIMEOUT_SECONDS,
    PARENT_ONLY_TARGET,
    matrix_command,
)


class WamCChildSearchRuntimeSweepTests(unittest.TestCase):
    def test_matrix_command_runs_child_search_target_set(self) -> None:
        command = matrix_command("dev")

        self.assertEqual(command[0], sys.executable)
        self.assertIn("benchmark_effective_distance_matrix.py", command[1])
        self.assertIn("--target-sets", command)
        self.assertEqual(command[command.index("--target-sets") + 1], "c-wam-child-search-layouts")
        self.assertIn("--repetitions", command)
        self.assertEqual(command[command.index("--repetitions") + 1], str(DEFAULT_REPETITIONS))
        self.assertIn("--run-timeout-seconds", command)
        self.assertEqual(command[command.index("--run-timeout-seconds") + 1], f"{DEFAULT_TIMEOUT_SECONDS:g}")
        self.assertEqual(command[command.index("--baseline-target") + 1], "c-wam-accumulated-child-csr")
        self.assertNotIn("--compile-only-targets", command)
        self.assertNotIn("--include-targets", command)

    def test_matrix_command_can_include_parent_only_baseline(self) -> None:
        command = matrix_command("dev", include_parent_only=True)

        self.assertIn("--include-targets", command)
        self.assertEqual(command[command.index("--include-targets") + 1], PARENT_ONLY_TARGET)

    def test_matrix_command_accepts_runtime_controls_and_extra_args(self) -> None:
        command = matrix_command(
            "dev,10x",
            repetitions=2,
            run_timeout_seconds=45.0,
            baseline_target=CHILD_SEARCH_TARGETS[0],
            extra_args=["--keep-temp"],
        )

        self.assertEqual(command[command.index("--scales") + 1], "dev,10x")
        self.assertEqual(command[command.index("--repetitions") + 1], "2")
        self.assertEqual(command[command.index("--run-timeout-seconds") + 1], "45")
        self.assertEqual(command[command.index("--baseline-target") + 1], CHILD_SEARCH_TARGETS[0])
        self.assertEqual(command[-1], "--keep-temp")


if __name__ == "__main__":
    unittest.main()
