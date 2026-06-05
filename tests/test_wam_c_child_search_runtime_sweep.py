#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_wam_c_child_search_runtime_sweep import (  # noqa: E402
    CHILD_SEARCH_TARGETS,
    DEFAULT_REPETITIONS,
    DEFAULT_TIMEOUT_SECONDS,
    PARENT_ONLY_TARGET,
    cache_input_summary,
    cache_input_summary_line,
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

    def test_cache_input_summary_reports_root_cache_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scale_dir = Path(tmp) / "toy"
            scale_dir.mkdir()
            (scale_dir / "root_categories.tsv").write_text("category\nroot\nother_root\n", encoding="utf-8")
            (scale_dir / "category_parent.tsv").write_text(
                "child\tparent\nchild\troot\nleaf\tchild\nside\tother_root\n",
                encoding="utf-8",
            )
            (scale_dir / "article_category.tsv").write_text(
                "article\tcategory\narticle_a\tleaf\narticle_b\tside\n",
                encoding="utf-8",
            )

            summary = cache_input_summary("toy", Path(tmp))
            self.assertEqual(summary["roots"], 2)
            self.assertEqual(summary["parent_edges"], 3)
            self.assertEqual(summary["article_category_rows"], 2)
            self.assertEqual(summary["max_cache_maps"], 2)
            self.assertEqual(summary["category_ids"], 5)
            self.assertEqual(summary["max_distance_entries_upper_bound"], 10)

            line = cache_input_summary_line("toy", Path(tmp))
            self.assertTrue(line.startswith("toy\twam_c_child_search_cache_inputs\t"))
            self.assertIn("roots=2", line)
            self.assertIn("max_distance_entries_upper_bound=10", line)


if __name__ == "__main__":
    unittest.main()
