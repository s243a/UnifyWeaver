#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from review_csharp_query_effective_distance_policy import (  # noqa: E402
    BENCHMARK_SCRIPT,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_SUMMARY_OUTPUT,
    build_review_command,
    main,
    parse_args,
)


class CSharpQueryPolicyReviewWrapperTests(unittest.TestCase):
    def test_default_command_uses_small_guarded_policy_actionable_run(self) -> None:
        command = build_review_command(parse_args([]))

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(BENCHMARK_SCRIPT))
        self.assertEqual(self._option_value(command, "--scales"), "dev")
        self.assertEqual(self._option_value(command, "--relation"), "category_parent")
        self.assertEqual(self._option_value(command, "--lookup-keys"), "4")
        self.assertEqual(self._option_value(command, "--lookup-repetitions"), "1")
        self.assertEqual(self._option_value(command, "--repetitions"), "1")
        self.assertEqual(self._option_value(command, "--artifact-root"), str(DEFAULT_ARTIFACT_ROOT))
        self.assertEqual(self._option_value(command, "--summary-output"), str(DEFAULT_SUMMARY_OUTPUT))
        self.assertEqual(self._option_value(command, "--policy-action-threshold"), "1.1")
        self.assertEqual(self._option_value(command, "--format"), "policy-actionable-markdown")
        self.assertIn("--require-idle", command)
        self.assertIn("--use-scale-lmdb-artifact", command)
        self.assertIn("--fail-on-policy-actions", command)

    def test_offline_summary_input_renders_without_live_benchmark_options(self) -> None:
        summary = ROOT / "output" / "summary.tsv"
        command = build_review_command(parse_args(["--summary-input", str(summary)]))

        self.assertEqual(self._option_value(command, "--summary-input"), str(summary))
        self.assertNotIn("--scales", command)
        self.assertNotIn("--artifact-root", command)
        self.assertNotIn("--summary-output", command)
        self.assertNotIn("--require-idle", command)
        self.assertEqual(self._option_value(command, "--format"), "policy-actionable-markdown")
        self.assertIn("--fail-on-policy-actions", command)

    def test_live_run_options_are_forwarded(self) -> None:
        command = build_review_command(
            parse_args(
                [
                    "--scales",
                    "300,1k",
                    "--relation",
                    "article_category",
                    "--lookup-keys",
                    "8",
                    "--lookup-repetitions",
                    "2",
                    "--repetitions",
                    "3",
                    "--policy-action-threshold",
                    "1.25",
                    "--format",
                    "policy-compare-tsv",
                    "--no-require-idle",
                    "--skip-resource-check",
                    "--skip-missing-scales",
                    "--refresh-artifacts",
                    "--no-use-scale-lmdb-artifact",
                    "--no-fail-on-policy-actions",
                ]
            )
        )

        self.assertEqual(self._option_value(command, "--scales"), "300,1k")
        self.assertEqual(self._option_value(command, "--relation"), "article_category")
        self.assertEqual(self._option_value(command, "--lookup-keys"), "8")
        self.assertEqual(self._option_value(command, "--lookup-repetitions"), "2")
        self.assertEqual(self._option_value(command, "--repetitions"), "3")
        self.assertEqual(self._option_value(command, "--policy-action-threshold"), "1.25")
        self.assertEqual(self._option_value(command, "--format"), "policy-compare-tsv")
        self.assertIn("--skip-resource-check", command)
        self.assertIn("--skip-missing-scales", command)
        self.assertIn("--refresh-artifacts", command)
        self.assertNotIn("--require-idle", command)
        self.assertNotIn("--use-scale-lmdb-artifact", command)
        self.assertNotIn("--fail-on-policy-actions", command)

    def test_summary_input_rejects_custom_summary_output(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--summary-input",
                        "output/summary.tsv",
                        "--summary-output",
                        "output/other.tsv",
                    ]
                )

        self.assertIn("--summary-output cannot be used with --summary-input", stderr.getvalue())

    def test_dry_run_prints_command_without_running(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            result = main(["--dry-run", "--no-require-idle"])

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn(str(BENCHMARK_SCRIPT), output)
        self.assertIn("--summary-output", output)
        self.assertIn("--policy-action-threshold 1.1", output)
        self.assertIn("--format policy-actionable-markdown", output)

    @staticmethod
    def _option_value(command: list[str], option: str) -> str:
        return command[command.index(option) + 1]


if __name__ == "__main__":
    unittest.main()
