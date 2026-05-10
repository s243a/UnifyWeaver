#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_csharp_query_source_mode_sweep import (  # noqa: E402
    CALIBRATION_ARTIFACT,
    CalibrationArtifactRow,
    CompetingProcess,
    DEFAULT_WORKLOADS,
    SourceModeSummary,
    WORKLOAD_SCRIPTS,
    calibration_failures,
    calibration_rows_from_stability,
    calibration_rows_from_summaries,
    compare_calibration,
    filter_workload_scales,
    load_calibration_artifact,
    parse_args,
    parse_competing_processes,
    parse_mem_available_mib,
    parse_mode_summary,
    parse_ratio,
    parse_runner_output,
    parse_workloads,
    resource_preflight_failures,
    summarize_stability,
    supported_scales_for_workload,
)
from benchmark_common import csharp_query_source_mode_choices  # noqa: E402


class CSharpQuerySourceModeSweepTests(unittest.TestCase):
    def test_default_workloads_cover_all_graph_workloads(self) -> None:
        original_argv = sys.argv
        try:
            sys.argv = ["benchmark_csharp_query_source_mode_sweep.py"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.workloads, DEFAULT_WORKLOADS)
        self.assertEqual(parse_workloads(args.workloads), list(WORKLOAD_SCRIPTS))

    def test_parse_workloads_all_expands_to_all_graph_workloads(self) -> None:
        self.assertEqual(parse_workloads("all"), list(WORKLOAD_SCRIPTS))
        self.assertEqual(parse_workloads(" ALL "), list(WORKLOAD_SCRIPTS))

    def test_parse_workloads_rejects_unknown_workloads(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unknown workload"):
            parse_workloads("category-influence,not-a-workload")

    def test_filter_workload_scales_skips_unsupported_generated_graph_scales(self) -> None:
        self.assertEqual(
            supported_scales_for_workload("dependency-depth"),
            ("300", "1k", "5k", "10k"),
        )
        self.assertEqual(
            filter_workload_scales("dependency-depth", "dev,300"),
            ("300", ["dev"]),
        )

    def test_filter_workload_scales_keeps_dev_for_file_backed_graph_scales(self) -> None:
        self.assertEqual(
            filter_workload_scales("effective-distance", "dev,300"),
            ("dev,300", []),
        )
        self.assertEqual(
            filter_workload_scales("shortest-path", "dev"),
            ("dev", []),
        )

    def test_parse_mem_available_mib(self) -> None:
        self.assertEqual(
            parse_mem_available_mib("MemTotal:       4096000 kB\nMemAvailable:   2097152 kB\n"),
            2048,
        )
        self.assertIsNone(parse_mem_available_mib("MemTotal:       4096000 kB\n"))

    def test_parse_competing_processes_filters_current_pid_and_threshold(self) -> None:
        processes = parse_competing_processes(
            "\n".join(
                [
                    "PID %CPU COMMAND",
                    "100 75.0 swipl -q benchmark.pl",
                    "101 10.0 idle-process",
                    "102 99.0 current-python",
                ]
            ),
            current_pid=102,
            cpu_threshold=50.0,
        )

        self.assertEqual(
            processes,
            [CompetingProcess(pid=100, cpu_percent=75.0, command="swipl -q benchmark.pl")],
        )

    def test_resource_preflight_failures_checks_memory_and_cpu(self) -> None:
        failures = resource_preflight_failures(
            min_free_memory_mib=1024,
            max_competing_cpu_percent=50.0,
            available_memory_mib=512,
            competing_processes=[
                CompetingProcess(pid=100, cpu_percent=75.0, command="swipl -q benchmark.pl")
            ],
        )

        self.assertEqual(
            failures,
            [
                "available memory 512 MiB is below required 1024 MiB",
                "competing process 100 uses 75.0% CPU (threshold 50.0%): swipl -q benchmark.pl",
            ],
        )

    def test_calibration_artifact_covers_registered_graph_workloads(self) -> None:
        rows = load_calibration_artifact(CALIBRATION_ARTIFACT)

        self.assertEqual(
            [(row.workload, row.scale) for row in rows],
            [
                (workload, scale)
                for workload in WORKLOAD_SCRIPTS
                for scale in ("300", "1k", "5k", "10k")
            ],
        )
        self.assertEqual({row.scale for row in rows}, {"300", "1k", "5k", "10k"})

    def test_calibration_artifact_matches_current_auto_policy_boundary(self) -> None:
        rows = load_calibration_artifact(CALIBRATION_ARTIFACT)
        allowed_modes = set(csharp_query_source_mode_choices())

        for row in rows:
            with self.subTest(workload=row.workload):
                self.assertIn(row.observed_best_source_mode, allowed_modes)
                self.assertEqual(row.current_auto_resolved_source_mode, "preload")
                self.assertEqual(row.output_agreement, "match")
                self.assertLessEqual(parse_ratio(row.observed_auto_vs_best) or 0.0, 2.0)
                self.assertIn("auto:preload", row.resolved_source_mode_summary)

    def test_parse_mode_summary(self) -> None:
        self.assertEqual(
            parse_mode_summary("artifact-prebuilt:0.904,auto:0.467,preload:0.360"),
            {
                "artifact-prebuilt": "0.904",
                "auto": "0.467",
                "preload": "0.360",
            },
        )

    def test_compare_calibration_flags_policy_drift_as_critical(self) -> None:
        drift = compare_calibration(
            [
                self._summary(
                    resolved_source_mode_summary="auto:artifact-prebuilt,preload:preload",
                )
            ],
            [self._baseline()],
        )

        self.assertEqual(drift.timing, [])
        self.assertEqual(
            drift.critical,
            [
                (
                    "category-influence/300: auto resolved source mode changed "
                    "from preload to artifact-prebuilt"
                )
            ],
        )

    def test_compare_calibration_flags_output_and_registration_drift_as_critical(self) -> None:
        drift = compare_calibration(
            [
                self._summary(
                    output_agreement="MISMATCH",
                    source_registration_summary="auto:binary_artifact_artifact-prebuilt_arity2=2",
                )
            ],
            [self._baseline()],
        )

        self.assertEqual(
            drift.critical,
            [
                "category-influence/300: fresh source-mode outputs MISMATCH",
                "category-influence/300: source registration shape changed",
            ],
        )

    def test_compare_calibration_keeps_best_mode_and_median_changes_as_timing(self) -> None:
        drift = compare_calibration(
            [
                self._summary(
                    best_source_mode="auto",
                    auto_vs_best="1.00x",
                    median_summary="artifact-prebuilt:0.904,auto:0.100,preload:0.900",
                )
            ],
            [self._baseline()],
            timing_drift_ratio=1.20,
        )

        self.assertEqual(drift.critical, [])
        self.assertEqual(
            drift.timing,
            [
                "category-influence/300: best source mode changed from preload to auto",
                "category-influence/300: auto_vs_best changed from 1.30x to 1.00x (1.30x)",
                "category-influence/300: median[auto] changed from 0.467 to 0.100 (4.67x)",
                "category-influence/300: median[preload] changed from 0.360 to 0.900 (2.50x)",
            ],
        )

    def test_compare_calibration_flags_missing_baseline_for_fresh_rows(self) -> None:
        drift = compare_calibration([self._summary()], [])

        self.assertEqual(drift.timing, [])
        self.assertEqual(
            drift.critical,
            ["category-influence/300: no calibration baseline row"],
        )

    def test_summarize_stability_reports_majority_and_medians(self) -> None:
        stability = summarize_stability(
            [
                [
                    self._summary(
                        best_source_mode="preload",
                        auto_vs_best="1.30x",
                        median_summary="artifact-prebuilt:0.904,auto:0.467,preload:0.360",
                    )
                ],
                [
                    self._summary(
                        best_source_mode="auto",
                        auto_vs_best="1.00x",
                        median_summary="artifact-prebuilt:0.800,auto:0.200,preload:0.400",
                    )
                ],
                [
                    self._summary(
                        best_source_mode="auto",
                        auto_vs_best="1.00x",
                        median_summary="artifact-prebuilt:1.000,auto:0.300,preload:0.500",
                    )
                ],
            ]
        )

        self.assertEqual(len(stability), 1)
        summary = stability[0]
        self.assertEqual(summary.workload, "category-influence")
        self.assertEqual(summary.scale, "300")
        self.assertEqual(summary.runs, 3)
        self.assertEqual(summary.output_agreement, "match")
        self.assertEqual(summary.stable_best_source_mode, "auto")
        self.assertEqual(summary.best_source_mode_counts, "auto:2,preload:1")
        self.assertEqual(summary.stable_auto_resolved_source_mode, "preload")
        self.assertEqual(summary.auto_resolved_source_mode_counts, "preload:3")
        self.assertEqual(summary.auto_vs_best_median, "1.00x")
        self.assertEqual(
            summary.median_summary,
            "artifact-prebuilt:0.904,auto:0.300,preload:0.400",
        )

    def test_summarize_stability_requires_majority_for_stable_best_mode(self) -> None:
        stability = summarize_stability(
            [
                [self._summary(best_source_mode="auto")],
                [self._summary(best_source_mode="preload")],
            ]
        )

        self.assertEqual(stability[0].stable_best_source_mode, "")
        self.assertEqual(stability[0].best_source_mode_counts, "auto:1,preload:1")

    def test_calibration_rows_from_summaries_match_artifact_shape(self) -> None:
        rows = calibration_rows_from_summaries(
            [
                self._summary(
                    resolved_source_mode_summary="artifact-prebuilt:artifact-prebuilt,auto:preload",
                    source_registration_summary="auto:preloaded_preload_arity2=2",
                )
            ]
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.workload, "category-influence")
        self.assertEqual(row.scale, "300")
        self.assertEqual(row.observed_best_source_mode, "preload")
        self.assertEqual(row.current_auto_resolved_source_mode, "preload")
        self.assertEqual(row.observed_auto_vs_best, "1.30x")
        self.assertEqual(row.output_agreement, "match")
        self.assertEqual(
            row.median_summary,
            "artifact-prebuilt:0.904,auto:0.467,preload:0.360",
        )
        self.assertEqual(
            row.resolved_source_mode_summary,
            "artifact-prebuilt:artifact-prebuilt,auto:preload",
        )
        self.assertEqual(row.source_registration_summary, "auto:preloaded_preload_arity2=2")

    def test_calibration_rows_from_stability_use_median_timing_fields(self) -> None:
        run_one = self._summary(
            best_source_mode="preload",
            auto_vs_best="1.30x",
            median_summary="artifact-prebuilt:0.904,auto:0.467,preload:0.360",
            resolved_source_mode_summary="artifact-prebuilt:artifact-prebuilt,auto:preload",
            source_registration_summary="auto:preloaded_preload_arity2=2",
        )
        run_two = self._summary(
            best_source_mode="auto",
            auto_vs_best="1.00x",
            median_summary="artifact-prebuilt:0.800,auto:0.200,preload:0.400",
            resolved_source_mode_summary="artifact-prebuilt:artifact-prebuilt,auto:preload",
            source_registration_summary="auto:preloaded_preload_arity2=2",
        )
        run_three = self._summary(
            best_source_mode="auto",
            auto_vs_best="1.00x",
            median_summary="artifact-prebuilt:1.000,auto:0.300,preload:0.500",
            resolved_source_mode_summary="artifact-prebuilt:artifact-prebuilt,auto:preload",
            source_registration_summary="auto:preloaded_preload_arity2=2",
        )
        sweep_runs = [[run_one], [run_two], [run_three]]
        stability = summarize_stability(sweep_runs)

        rows = calibration_rows_from_stability(stability, sweep_runs)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.observed_best_source_mode, "auto")
        self.assertEqual(row.current_auto_resolved_source_mode, "preload")
        self.assertEqual(row.observed_auto_vs_best, "1.00x")
        self.assertEqual(
            row.median_summary,
            "artifact-prebuilt:0.904,auto:0.300,preload:0.400",
        )
        self.assertEqual(
            row.resolved_source_mode_summary,
            "artifact-prebuilt:artifact-prebuilt,auto:preload",
        )
        self.assertEqual(row.source_registration_summary, "auto:preloaded_preload_arity2=2")

    def test_parse_runner_output_summarizes_modes_and_registrations(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query:auto\t0.467\t0.467\t0.467\t42\tsamehash",
                "300\tcsharp-query:preload\t0.360\t0.360\t0.360\t42\tsamehash",
                "300\tcsharp-query:artifact-prebuilt\t0.904\t0.904\t0.904\t42\tsamehash",
                (
                    "300\tcsharp-query:auto-metrics\t"
                    "source_mode=auto resolved_source_mode=preload "
                    "source_registration_preloaded_preload_arity2=2 other=ignored"
                ),
                (
                    "300\tcsharp-query:preload-metrics\t"
                    "source_mode=preload resolved_source_mode=preload "
                    "source_registration_preloaded_preload_arity2=2"
                ),
                (
                    "300\tcsharp-query:artifact-prebuilt-metrics\t"
                    "source_mode=artifact-prebuilt resolved_source_mode=artifact-prebuilt "
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
            summary.resolved_source_mode_summary,
            "artifact-prebuilt:artifact-prebuilt,auto:preload,preload:preload",
        )
        self.assertEqual(
            summary.source_registration_summary,
            (
                "artifact-prebuilt:binary_artifact_artifact-prebuilt_arity2=2,"
                "auto:preloaded_preload_arity2=2,"
                "preload:preloaded_preload_arity2=2"
            ),
        )

    def test_parse_runner_output_handles_single_auto_mode_labels(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query\t0.200\t0.200\t0.200\t42\tsamehash",
                (
                    "300\tcsharp-query-metrics\t"
                    "source_mode=auto resolved_source_mode=preload "
                    "source_registration_preloaded_preload_arity2=2"
                ),
            ]
        )

        summaries = parse_runner_output("dependency-depth", output, default_source_mode="auto")

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.best_source_mode, "auto")
        self.assertEqual(summary.auto_vs_best, "1.00x")
        self.assertEqual(summary.output_agreement, "match")
        self.assertEqual(summary.median_summary, "auto:0.200")
        self.assertEqual(summary.resolved_source_mode_summary, "auto:preload")
        self.assertEqual(
            summary.source_registration_summary,
            "auto:preloaded_preload_arity2=2",
        )

    def test_parse_runner_output_handles_single_preload_mode_labels(self) -> None:
        output = "\n".join(
            [
                "scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256",
                "300\tcsharp-query\t0.180\t0.180\t0.180\t42\tsamehash",
                (
                    "300\tcsharp-query-metrics\t"
                    "source_mode=preload resolved_source_mode=preload "
                    "source_registration_preloaded_preload_arity2=2"
                ),
            ]
        )

        summaries = parse_runner_output("dependency-depth", output, default_source_mode="preload")

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.best_source_mode, "preload")
        self.assertEqual(summary.auto_vs_best, "")
        self.assertEqual(summary.median_summary, "preload:0.180")
        self.assertEqual(summary.resolved_source_mode_summary, "preload:preload")

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

    def _baseline(self, **overrides: str) -> CalibrationArtifactRow:
        values = {
            "workload": "category-influence",
            "scale": "300",
            "observed_best_source_mode": "preload",
            "current_auto_resolved_source_mode": "preload",
            "observed_auto_vs_best": "1.30x",
            "output_agreement": "match",
            "median_summary": "artifact-prebuilt:0.904,auto:0.467,preload:0.360",
            "resolved_source_mode_summary": "artifact-prebuilt:artifact-prebuilt,auto:preload,preload:preload",
            "source_registration_summary": "auto:preloaded_preload_arity2=2",
        }
        values.update(overrides)
        return CalibrationArtifactRow(**values)

    def _summary(self, **overrides: str) -> SourceModeSummary:
        values = {
            "workload": "category-influence",
            "scale": "300",
            "best_source_mode": "preload",
            "auto_vs_best": "1.30x",
            "output_agreement": "match",
            "median_summary": "artifact-prebuilt:0.904,auto:0.467,preload:0.360",
            "resolved_source_mode_summary": "artifact-prebuilt:artifact-prebuilt,auto:preload,preload:preload",
            "source_registration_summary": "auto:preloaded_preload_arity2=2",
        }
        values.update(overrides)
        return SourceModeSummary(**values)


if __name__ == "__main__":
    unittest.main()
