#!/usr/bin/env python3
from __future__ import annotations

import sys
import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_effective_distance_matrix import (  # noqa: E402
    RunResult,
    benchmark_target,
    kernel_pair_delta_rows,
    parse_args,
    print_kernel_pair_deltas,
    print_summary,
    resolve_requested_targets,
)
from benchmark_target_matrix import (  # noqa: E402
    KERNEL_TARGET_PAIRS,
    TARGETS,
    list_kernel_pairs_text,
    list_targets_text,
    resolve_targets,
)


class BenchmarkTargetMatrixTests(unittest.TestCase):
    def test_clojure_targets_are_registered(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["clojure-wam"],
        )

        self.assertEqual(
            targets,
            [
                "clojure-wam-accumulated",
                "clojure-wam-seeded",
                "clojure-wam-seeded-no-kernels",
                "clojure-wam-accumulated-no-kernels",
            ],
        )
        self.assertEqual(TARGETS["clojure-wam-accumulated"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-accumulated-no-kernels"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-seeded"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-seeded-no-kernels"].category, "hybrid-wam")

    def test_clojure_artifact_targets_are_registered_separately(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["clojure-wam-artifact"],
        )

        self.assertEqual(
            targets,
            [
                "clojure-wam-accumulated",
                "clojure-wam-accumulated-artifact",
                "clojure-wam-accumulated-no-kernels",
                "clojure-wam-accumulated-no-kernels-artifact",
                "clojure-wam-seeded",
                "clojure-wam-seeded-artifact",
                "clojure-wam-seeded-no-kernels",
                "clojure-wam-seeded-no-kernels-artifact",
            ],
        )
        self.assertEqual(TARGETS["clojure-wam-accumulated-artifact"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-accumulated-no-kernels-artifact"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-seeded-artifact"].category, "hybrid-wam")
        self.assertEqual(TARGETS["clojure-wam-seeded-no-kernels-artifact"].category, "hybrid-wam")

    def test_default_hybrid_wam_excludes_clojure_scaffolds(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["hybrid-wam"],
        )

        self.assertIn("go-wam-accumulated", targets)
        self.assertIn("haskell-interp-ffi", targets)
        self.assertIn("scala-wam-seeded", targets)
        self.assertIn("scala-wam-seeded-no-kernels", targets)
        self.assertIn("scala-wam-accumulated", targets)
        self.assertIn("scala-wam-accumulated-no-kernels", targets)
        self.assertIn("c-wam-accumulated", targets)
        self.assertIn("c-wam-accumulated-no-kernels", targets)
        self.assertIn("c-wam-accumulated-lmdb", targets)
        self.assertIn("c-wam-accumulated-no-kernels-lmdb", targets)
        self.assertIn("clojure-wam-accumulated", targets)
        self.assertIn("clojure-wam-accumulated-no-kernels", targets)
        self.assertIn("clojure-wam-seeded", targets)
        self.assertIn("clojure-wam-seeded-no-kernels", targets)
        self.assertNotIn("c-wam-lowered-helper", targets)
        self.assertNotIn("c-wam-lowered-helper-interpreted", targets)

    def test_c_lowered_helper_targets_are_registered_separately(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["c-wam-lowered-helper"],
        )

        self.assertEqual(
            targets,
            [
                "c-wam-lowered-helper-interpreted",
                "c-wam-lowered-helper",
            ],
        )
        self.assertEqual(TARGETS["c-wam-lowered-helper"].category, "hybrid-wam-lowered-helper")
        self.assertEqual(
            TARGETS["c-wam-lowered-helper-interpreted"].category,
            "hybrid-wam-lowered-helper",
        )

    def test_scala_targets_are_registered(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["scala-wam"],
        )

        self.assertEqual(
            targets,
            [
                "scala-wam-seeded",
                "scala-wam-seeded-no-kernels",
                "scala-wam-accumulated",
                "scala-wam-accumulated-no-kernels",
            ],
        )
        self.assertEqual(TARGETS["scala-wam-seeded"].category, "hybrid-wam")
        self.assertEqual(TARGETS["scala-wam-seeded-no-kernels"].category, "hybrid-wam")
        self.assertEqual(TARGETS["scala-wam-accumulated"].category, "hybrid-wam")
        self.assertEqual(TARGETS["scala-wam-accumulated-no-kernels"].category, "hybrid-wam")

    def test_scala_artifact_targets_are_registered_separately(self) -> None:
        targets = resolve_targets(
            explicit_targets=None,
            target_set_names=["scala-wam-artifact"],
        )

        self.assertEqual(
            targets,
            [
                "scala-wam-seeded",
                "scala-wam-seeded-artifact",
                "scala-wam-accumulated",
                "scala-wam-accumulated-artifact",
            ],
        )
        self.assertEqual(TARGETS["scala-wam-seeded-artifact"].category, "hybrid-wam")
        self.assertEqual(TARGETS["scala-wam-accumulated-artifact"].category, "hybrid-wam")

    def test_list_targets_includes_clojure_scaffold_set(self) -> None:
        text = list_targets_text()

        self.assertIn("clojure-wam-accumulated\thybrid-wam", text)
        self.assertIn("clojure-wam-accumulated-no-kernels\thybrid-wam", text)
        self.assertIn("clojure-wam-seeded\thybrid-wam", text)
        self.assertIn("clojure-wam-seeded-no-kernels\thybrid-wam", text)
        self.assertIn(
            "clojure-wam\tclojure-wam-accumulated,clojure-wam-seeded,"
            "clojure-wam-seeded-no-kernels,clojure-wam-accumulated-no-kernels",
            text,
        )
        self.assertIn(
            "scala-wam\tscala-wam-seeded,scala-wam-seeded-no-kernels,"
            "scala-wam-accumulated,scala-wam-accumulated-no-kernels",
            text,
        )
        self.assertIn(
            "scala-wam-artifact\tscala-wam-seeded,scala-wam-seeded-artifact,"
            "scala-wam-accumulated,scala-wam-accumulated-artifact",
            text,
        )
        self.assertIn("clojure-wam-scaffold\t", text)

    def test_effective_distance_runner_resolves_seeded_clojure_targets(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "benchmark_effective_distance_matrix.py",
                "--targets",
                "clojure-wam-seeded,prolog-accumulated",
            ]
            args = parse_args()
            self.assertEqual(resolve_requested_targets(args), ["clojure-wam-seeded", "prolog-accumulated"])
        finally:
            sys.argv = original_argv

    def test_lowered_helper_benchmark_target_does_not_require_scale_data_dir(self) -> None:
        with patch("benchmark_effective_distance_matrix.run_command") as run_command_mock:
            run_command_mock.return_value = subprocess.CompletedProcess(
                args=["lowered-helper-benchmark"],
                returncode=0,
                stdout="left\tright\tscore\nrow\tvalue\t1.000000\n",
                stderr="",
            )

            result = benchmark_target(
                ["lowered-helper-benchmark"],
                "25x",
                1,
                "c-wam-lowered-helper",
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.row_count, 1)
        run_command_mock.assert_called_once()
        self.assertEqual(run_command_mock.call_args.args[0], ["lowered-helper-benchmark"])

    def test_kernel_pair_registry_covers_registered_wam_pairs(self) -> None:
        pairs = {(pair.family, pair.mode): pair for pair in KERNEL_TARGET_PAIRS}

        expected = {
            ("rust", "seeded"),
            ("rust", "accumulated"),
            ("rust", "interpreter"),
            ("rust", "lowered"),
            ("go", "accumulated"),
            ("c", "accumulated"),
            ("c", "accumulated-lmdb"),
            ("scala", "seeded"),
            ("scala", "accumulated"),
            ("clojure", "seeded"),
            ("clojure", "accumulated"),
            ("clojure", "seeded-artifact"),
            ("clojure", "accumulated-artifact"),
            ("haskell", "interpreter"),
            ("haskell", "lowered"),
        }
        self.assertEqual(set(pairs), expected)

        for pair in KERNEL_TARGET_PAIRS:
            self.assertIn(pair.kernels_target, TARGETS)
            self.assertIn(pair.no_kernels_target, TARGETS)

    def test_list_kernel_pairs_text_is_tsv(self) -> None:
        text = list_kernel_pairs_text()

        self.assertTrue(text.startswith("family\tmode\tkernels_target\tno_kernels_target\n"))
        self.assertIn(
            "rust\taccumulated\twam-rust-accumulated\twam-rust-accumulated-no-kernels",
            text,
        )
        self.assertIn(
            "rust\tlowered\trust-lowered-ffi\trust-lowered-only",
            text,
        )
        self.assertIn(
            "haskell\tlowered\thaskell-lowered-ffi\thaskell-lowered-only",
            text,
        )
        self.assertIn(
            "scala\taccumulated\tscala-wam-accumulated\tscala-wam-accumulated-no-kernels",
            text,
        )
        self.assertIn(
            "scala\tseeded\tscala-wam-seeded\tscala-wam-seeded-no-kernels",
            text,
        )
        self.assertIn(
            "c\taccumulated-lmdb\tc-wam-accumulated-lmdb\tc-wam-accumulated-no-kernels-lmdb",
            text,
        )
        self.assertIn(
            "clojure\tseeded-artifact\tclojure-wam-seeded-artifact\tclojure-wam-seeded-no-kernels-artifact",
            text,
        )
        self.assertIn(
            "clojure\taccumulated-artifact\tclojure-wam-accumulated-artifact\tclojure-wam-accumulated-no-kernels-artifact",
            text,
        )

    def test_effective_distance_runner_accepts_kernel_pair_listing(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = [
                "benchmark_effective_distance_matrix.py",
                "--list-kernel-pairs",
            ]
            args = parse_args()
            self.assertTrue(args.list_kernel_pairs)
        finally:
            sys.argv = original_argv

    def test_kernel_pair_delta_rows_report_matching_pair(self) -> None:
        rows = kernel_pair_delta_rows(
            "dev",
            [
                RunResult(
                    "wam-rust-accumulated",
                    "dev",
                    [1.0, 1.2, 1.1],
                    "same",
                    42,
                    "",
                ),
                RunResult(
                    "wam-rust-accumulated-no-kernels",
                    "dev",
                    [2.0, 2.2, 2.1],
                    "same",
                    42,
                    "",
                ),
            ],
        )

        self.assertEqual(
            rows,
            [
                "dev\trust\taccumulated\twam-rust-accumulated\t"
                "wam-rust-accumulated-no-kernels\t1.100\t2.100\t1.909\ttrue\ttrue"
            ],
        )

    def test_kernel_pair_delta_rows_skip_incomplete_pair(self) -> None:
        rows = kernel_pair_delta_rows(
            "dev",
            [
                RunResult(
                    "wam-rust-accumulated",
                    "dev",
                    [1.0],
                    "digest",
                    42,
                    "",
                )
            ],
        )

        self.assertEqual(rows, [])

    def test_kernel_pair_delta_rows_skip_non_ok_pair_member(self) -> None:
        rows = kernel_pair_delta_rows(
            "dev",
            [
                RunResult("scala-wam-accumulated", "dev", [1.0], "digest", 42, ""),
                RunResult(
                    "scala-wam-accumulated-no-kernels",
                    "dev",
                    [10.0],
                    "",
                    0,
                    "",
                    status="timeout",
                    message="timed out after 10.000s",
                ),
                RunResult(
                    "c-wam-accumulated",
                    "dev",
                    [0.5],
                    "digest",
                    10,
                    "",
                    status="error",
                    message="exited with status 134",
                ),
            ],
        )

        self.assertEqual(rows, [])

    def test_print_summary_reports_bounded_statuses_without_comparisons(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            print_summary(
                [
                    RunResult("prolog-accumulated", "dev", [0.1], "a", 10, ""),
                    RunResult(
                        "scala-wam-accumulated-no-kernels",
                        "dev",
                        [10.0],
                        "",
                        0,
                        "",
                        status="timeout",
                        message="timed out after 10.000s",
                    ),
                    RunResult(
                        "c-wam-accumulated-no-kernels",
                        "dev",
                        [0.5],
                        "",
                        0,
                        "",
                        status="compile_only",
                        message="generated/built but not executed",
                    ),
                    RunResult(
                        "c-wam-accumulated",
                        "dev",
                        [0.5],
                        "b",
                        19,
                        "",
                        status="error",
                        message="exited with status 134",
                    ),
                ],
                "prolog-accumulated",
            )

        text = output.getvalue()
        self.assertIn("scale\ttarget\tcategory\tstatus\tmedian_s", text)
        self.assertIn("dev\tscala-wam-accumulated-no-kernels\thybrid-wam\ttimeout", text)
        self.assertIn("dev\tc-wam-accumulated-no-kernels\thybrid-wam\tcompile_only", text)
        self.assertIn("dev\tc-wam-accumulated\thybrid-wam\terror", text)
        self.assertNotIn("all_outputs", text)
        self.assertNotIn("speedup_vs_prolog-accumulated", text)

    def test_print_kernel_pair_deltas_emits_one_table_for_all_scales(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            print_kernel_pair_deltas(
                [
                    RunResult("wam-rust-seeded", "dev", [1.0], "a", 10, ""),
                    RunResult("wam-rust-seeded-no-kernels", "dev", [2.0], "a", 10, ""),
                    RunResult("wam-rust-seeded", "10x", [3.0], "b", 20, ""),
                    RunResult("wam-rust-seeded-no-kernels", "10x", [6.0], "c", 20, ""),
                ]
            )

        lines = output.getvalue().strip().splitlines()
        self.assertEqual(
            lines[0],
            "scale\tfamily\tmode\tkernels_target\tno_kernels_target\tkernels_median_s\t"
            "no_kernels_median_s\tkernels_speedup_vs_no_kernels\toutput_match\trows_match",
        )
        self.assertEqual(len(lines), 3)
        self.assertIn("dev\trust\tseeded\twam-rust-seeded\twam-rust-seeded-no-kernels", lines[1])
        self.assertIn("10x\trust\tseeded\twam-rust-seeded\twam-rust-seeded-no-kernels", lines[2])


if __name__ == "__main__":
    unittest.main()
