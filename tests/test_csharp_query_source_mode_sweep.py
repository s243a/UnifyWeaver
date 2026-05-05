#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_csharp_query_source_mode_sweep import (  # noqa: E402
    calibration_failures,
    parse_ratio,
    parse_runner_output,
)


class CSharpQuerySourceModeSweepTests(unittest.TestCase):
    def test_parse_runner_output_summarizes_modes_and_registrations(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query:auto\t0.467\t0.467\t0.467\t42\tsamehash",
                "300\tcsharp-query:preload\t0.360\t0.360\t0.360\t42\tsamehash",
                "300\tcsharp-query:artifact-prebuilt\t0.904\t0.904\t0.904\t42\tsamehash",
                "300\tcsharp-query:auto-metrics\tsource_registration_preloaded_preload_arity2=2 other=ignored",
                "300\tcsharp-query:preload-metrics\tsource_registration_preloaded_preload_arity2=2",
                (
                    "300\tcsharp-query:artifact-prebuilt-metrics\t"
                    "source_registration_binary_artifact_artifact-prebuilt_arity2=2"
                ),
                "300\tcsharp_query_best_source_mode\tpreload",
                "300\tcsharp_query_auto_vs_best_source_mode\t1.30x",
            ]
        )

        summaries = parse_runner_output("category-influence", output)

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.workload, "category-influence")
        self.assertEqual(summary.scale, "300")
        self.assertEqual(summary.best_source_mode, "preload")
        self.assertEqual(summary.auto_vs_best, "1.30x")
        self.assertEqual(summary.output_agreement, "match")
        self.assertEqual(
            summary.median_summary,
            "artifact-prebuilt:0.904,auto:0.467,preload:0.360",
        )
        self.assertEqual(
            summary.source_registration_summary,
            (
                "artifact-prebuilt:binary_artifact_artifact-prebuilt_arity2=2,"
                "auto:preloaded_preload_arity2=2,"
                "preload:preloaded_preload_arity2=2"
            ),
        )

    def test_parse_runner_output_flags_hash_mismatch(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query:auto\t0.200\t0.200\t0.200\t42\thash-a",
                "300\tcsharp-query:preload\t0.180\t0.180\t0.180\t42\thash-b",
                "300\tcsharp_query_best_source_mode\tpreload",
            ]
        )

        summaries = parse_runner_output("dependency-depth", output)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].output_agreement, "MISMATCH")
        self.assertEqual(summaries[0].auto_vs_best, "")

    def test_calibration_failures_detect_slow_auto(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query:auto\t0.300\t0.300\t0.300\t42\tsamehash",
                "300\tcsharp-query:preload\t0.200\t0.200\t0.200\t42\tsamehash",
                "300\tcsharp_query_best_source_mode\tpreload",
                "300\tcsharp_query_auto_vs_best_source_mode\t1.50x",
            ]
        )
        summaries = parse_runner_output("dependency-depth", output)

        failures = calibration_failures(summaries, max_auto_vs_best_ratio=1.25)

        self.assertEqual(
            failures,
            ["dependency-depth/300: auto_vs_best 1.50x exceeds 1.25x"],
        )

    def test_calibration_failures_can_detect_output_mismatch(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query:auto\t0.200\t0.200\t0.200\t42\thash-a",
                "300\tcsharp-query:preload\t0.180\t0.180\t0.180\t42\thash-b",
            ]
        )
        summaries = parse_runner_output("category-influence", output)

        failures = calibration_failures(summaries, fail_on_output_mismatch=True)

        self.assertEqual(
            failures,
            ["category-influence/300: source-mode outputs MISMATCH"],
        )

    def test_parse_ratio_accepts_x_suffix_and_empty_values(self) -> None:
        self.assertEqual(parse_ratio("1.25x"), 1.25)
        self.assertEqual(parse_ratio("1.00"), 1.0)
        self.assertIsNone(parse_ratio(""))
        self.assertIsNone(parse_ratio("not-a-ratio"))


if __name__ == "__main__":
    unittest.main()
