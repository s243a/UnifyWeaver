#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_wam_c_child_csr_scale_sweep import (  # noqa: E402
    CSR_TARGETS,
    DEFAULT_SCALES,
    matrix_command,
)


class WamCChildCsrScaleSweepTests(unittest.TestCase):
    def test_matrix_command_uses_compile_only_csr_layout_targets(self) -> None:
        command = matrix_command("10x,1k")

        self.assertEqual(command[0], sys.executable)
        self.assertIn("benchmark_effective_distance_matrix.py", command[1])
        self.assertIn("--target-sets", command)
        self.assertEqual(command[command.index("--target-sets") + 1], "c-wam-child-csr-layouts")
        self.assertIn("--compile-only-targets", command)
        self.assertEqual(command[command.index("--compile-only-targets") + 1], ",".join(CSR_TARGETS))
        self.assertEqual(command[command.index("--baseline-target") + 1], "c-wam-accumulated-child-csr")

    def test_matrix_command_accepts_extra_matrix_args(self) -> None:
        command = matrix_command(DEFAULT_SCALES, ["--keep-temp"])

        self.assertEqual(command[-1], "--keep-temp")


if __name__ == "__main__":
    unittest.main()
