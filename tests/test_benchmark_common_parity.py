#!/usr/bin/env python3
"""Unit tests for three_column_parity_vs_reference in benchmark_common."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmark"))

from benchmark_common import (  # noqa: E402
    normalize_three_column_float_rows,
    three_column_parity_vs_reference,
)


def write_tsv(tmp: Path, name: str, contents: str) -> Path:
    path = tmp / name
    path.write_text(contents, encoding="utf-8")
    return path


class ThreeColumnParityTests(unittest.TestCase):
    def test_match_when_normalised_outputs_identical(self) -> None:
        output = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
            "B\tR\t1.999728\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref = write_tsv(tmp, "reference_output.tsv", output)
            normalised = normalize_three_column_float_rows(output, decimals=6)
            self.assertEqual(
                three_column_parity_vs_reference(normalised, ref, decimals=6),
                "match",
            )

    def test_no_ref_when_file_missing(self) -> None:
        normalised = normalize_three_column_float_rows(
            "article\troot\tdeff\nA\tR\t1.0\n", decimals=6
        )
        missing = Path("/tmp/uw-definitely-does-not-exist.tsv")
        self.assertEqual(
            three_column_parity_vs_reference(normalised, missing, decimals=6),
            "no-ref",
        )

    def test_numeric_diff_reports_shared_article_delta(self) -> None:
        ours = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
            "B\tR\t1.999700\n"   # differs from reference below
        )
        ref = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
            "B\tR\t1.999728\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref_path = write_tsv(tmp, "reference_output.tsv", ref)
            normalised = normalize_three_column_float_rows(ours, decimals=6)
            self.assertEqual(
                three_column_parity_vs_reference(normalised, ref_path, decimals=6),
                "diff:1/ours:0/ref:0",
            )

    def test_missing_article_reports_ref_only(self) -> None:
        ours = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
        )
        ref = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
            "B\tR\t1.999728\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref_path = write_tsv(tmp, "reference_output.tsv", ref)
            normalised = normalize_three_column_float_rows(ours, decimals=6)
            self.assertEqual(
                three_column_parity_vs_reference(normalised, ref_path, decimals=6),
                "diff:0/ours:0/ref:1",
            )

    def test_extra_article_reports_ours_only(self) -> None:
        ours = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
            "B\tR\t1.999728\n"
            "C\tR\t2.500000\n"   # only in ours
        )
        ref = (
            "article\troot\tdeff\n"
            "A\tR\t0.993073\n"
            "B\tR\t1.999728\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref_path = write_tsv(tmp, "reference_output.tsv", ref)
            normalised = normalize_three_column_float_rows(ours, decimals=6)
            self.assertEqual(
                three_column_parity_vs_reference(normalised, ref_path, decimals=6),
                "diff:0/ours:1/ref:0",
            )


if __name__ == "__main__":
    unittest.main()
