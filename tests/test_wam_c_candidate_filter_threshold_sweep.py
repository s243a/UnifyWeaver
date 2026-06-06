#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_wam_c_candidate_filter_threshold_sweep import (  # noqa: E402
    OFF_THRESHOLD,
    PROFILES,
    TARGET,
    ThresholdSpec,
    matrix_command,
    parse_matrix_output,
    parse_thresholds,
    render_tsv,
    summarize_rows,
)


class WamCCandidateFilterThresholdSweepTests(unittest.TestCase):
    def test_parse_threshold_aliases(self) -> None:
        thresholds = parse_thresholds("auto,always,off,16,0")

        self.assertEqual(
            [(item.label, item.min_roots) for item in thresholds],
            [
                ("auto", None),
                ("always", 1),
                ("off", OFF_THRESHOLD),
                ("16", 16),
                ("auto", None),
            ],
        )

    def test_matrix_command_uses_profile_and_threshold(self) -> None:
        command = matrix_command(
            "50k_cats",
            PROFILES["low"],
            ThresholdSpec("always", 1),
            repetitions=2,
            run_timeout_seconds=45.0,
            extra_args=["--keep-temp"],
        )

        self.assertEqual(command[0], sys.executable)
        self.assertIn("benchmark_effective_distance_matrix.py", command[1])
        self.assertEqual(command[command.index("--scales") + 1], "50k_cats")
        self.assertEqual(command[command.index("--targets") + 1], TARGET)
        self.assertEqual(command[command.index("--repetitions") + 1], "2")
        self.assertEqual(command[command.index("--run-timeout-seconds") + 1], "45")
        self.assertEqual(command[command.index("--wam-c-article-stride") + 1], "1000")
        self.assertEqual(command[command.index("--wam-c-root-stride") + 1], "100")
        self.assertEqual(command[command.index("--wam-c-candidate-filter-min-roots") + 1], "1")
        self.assertEqual(command[-1], "--keep-temp")

    def test_matrix_command_omits_auto_threshold_override(self) -> None:
        command = matrix_command("50k_cats", PROFILES["high-capped"], ThresholdSpec("auto", None))

        self.assertNotIn("--wam-c-candidate-filter-min-roots", command)
        self.assertEqual(command[command.index("--wam-c-max-results") + 1], "50")

    def test_boundary_profiles_select_expected_root_strides(self) -> None:
        expected = {
            "boundary-250": "16",
            "boundary-500": "8",
            "boundary-800": "5",
        }

        for profile_name, root_stride in expected.items():
            with self.subTest(profile=profile_name):
                command = matrix_command("50k_cats", PROFILES[profile_name], ThresholdSpec("auto", None))
                self.assertEqual(command[command.index("--wam-c-article-stride") + 1], "1000")
                self.assertEqual(command[command.index("--wam-c-root-stride") + 1], root_stride)

    def test_parse_matrix_output_extracts_metrics(self) -> None:
        output = (
            "scale\ttarget\tcategory\tstatus\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256\tmessage\n"
            "50k_cats\tc-wam-accumulated-child-csr\thybrid-wam-child-search\tok\t0.1\t0.1\t0.1\t"
            "1\tabcdef123456\twam_c_effective_setup selected_articles=50 selected_roots=41 "
            "candidate_filter_min_roots=256 setup_ms=1.0; wam_c_effective_runtime queries=2050 "
            "results=1 category_visits=2050 candidate_filter_articles=0 "
            "candidate_schedule_roots=0 query_ms=26.435\n"
        )

        rows = parse_matrix_output(output, "low", ThresholdSpec("auto", None))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].scale, "50k_cats")
        self.assertEqual(rows[0].profile, "low")
        self.assertEqual(rows[0].median_s, 0.1)
        self.assertEqual(rows[0].rows, 1)
        self.assertEqual(rows[0].metrics["selected_roots"], "41")
        self.assertEqual(rows[0].metrics["query_ms"], "26.435")

    def test_summary_compares_rows_to_dense_off_baseline(self) -> None:
        dense_output = (
            "scale\ttarget\tcategory\tstatus\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256\tmessage\n"
            "50k_cats\tc-wam-accumulated-child-csr\thybrid-wam-child-search\tok\t0.1\t0.1\t0.1\t"
            "1\thash1\twam_c_effective_setup selected_articles=50 selected_roots=41; "
            "wam_c_effective_runtime query_ms=25.000 candidate_filter_articles=0 "
            "candidate_schedule_roots=0 category_visits=2050\n"
        )
        sparse_output = (
            "scale\ttarget\tcategory\tstatus\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256\tmessage\n"
            "50k_cats\tc-wam-accumulated-child-csr\thybrid-wam-child-search\tok\t0.2\t0.2\t0.2\t"
            "1\thash1\twam_c_effective_setup selected_articles=50 selected_roots=41; "
            "wam_c_effective_runtime query_ms=100.000 candidate_filter_articles=50 "
            "candidate_schedule_roots=41 category_visits=1\n"
        )
        rows = (
            parse_matrix_output(dense_output, "low", ThresholdSpec("off", OFF_THRESHOLD))
            + parse_matrix_output(sparse_output, "low", ThresholdSpec("always", 1))
        )

        summary = summarize_rows(rows)

        self.assertEqual(summary[0]["policy"], "dense")
        self.assertEqual(summary[0]["hash_agreement"], "match")
        self.assertEqual(summary[0]["query_ms_vs_dense"], "1.00x")
        self.assertEqual(summary[1]["policy"], "sparse")
        self.assertEqual(summary[1]["hash_agreement"], "match")
        self.assertEqual(summary[1]["query_ms_vs_dense"], "4.00x")
        self.assertIn("candidate_schedule_roots", render_tsv(summary))


if __name__ == "__main__":
    unittest.main()
