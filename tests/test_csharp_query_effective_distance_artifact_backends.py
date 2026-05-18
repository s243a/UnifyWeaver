#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_effective_distance_artifact_backends.py"
LIGHTNINGDB_PACKAGE = Path.home() / ".nuget" / "packages" / "lightningdb" / "0.21.0"

SPEC = importlib.util.spec_from_file_location("effective_distance_artifact_backends", SCRIPT)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CSharpQueryEffectiveDistanceArtifactBackendTests(unittest.TestCase):
    def test_large_scale_helpers_recognize_category_scales(self) -> None:
        self.assertFalse(MODULE.is_large_scale("dev"))
        self.assertFalse(MODULE.is_large_scale("10k"))
        self.assertTrue(MODULE.is_large_scale("50k_cats"))
        self.assertTrue(MODULE.is_large_scale("100k_cats"))
        self.assertTrue(MODULE.is_large_scale("500k_cats"))
        self.assertTrue(MODULE.is_large_scale("1m_cats"))
        self.assertTrue(MODULE.is_large_scale("1m"))
        self.assertEqual(MODULE.scale_seed_cap("50k_cats"), 50_000)
        self.assertIsNone(MODULE.scale_seed_cap("100k_cats"))

    def test_parse_mem_available_mib(self) -> None:
        self.assertEqual(
            MODULE.parse_mem_available_mib("MemTotal: 4096000 kB\nMemAvailable: 2097152 kB\n"),
            2048,
        )
        self.assertIsNone(MODULE.parse_mem_available_mib("MemTotal: 4096000 kB\n"))

    def test_unsupported_auto_preparation_scale_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "automatic fixture preparation only supports"):
            MODULE.scale_seed_cap("1m_cats")

    def test_load_summary_tsv_validates_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            path.write_text("scale\trelation\nexample\tcategory_parent\n")

            with self.assertRaisesRegex(ValueError, "missing summary column"):
                MODULE.load_summary_tsv(path)

    def test_write_summary_tsv_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "summary.tsv"
            row = MODULE.SummaryRow(
                {
                    "scale": "dev",
                    "relation": "category_parent",
                    "rows": "6",
                    "distinct_categories": "4",
                    "lookup_keys": "2",
                    "best_lookup_mode": "mmap-array",
                    "best_lookup_col1_mode": "mmap-array",
                    "best_bucket_mode": "mmap-array",
                    "best_bucket_col1_mode": "mmap-array",
                    "best_scan_mode": "preload",
                    "smallest_artifact_mode": "mmap-array",
                    "lookup_ms_by_mode": "mmap-array:1.250",
                    "lookup_col1_ms_by_mode": "mmap-array:1.500",
                    "bucket_ms_by_mode": "mmap-array:2.000",
                    "bucket_col1_ms_by_mode": "mmap-array:2.500",
                    "scan_ms_by_mode": "preload:1.000",
                    "artifact_bytes_by_mode": "mmap-array:4096|preload:0",
                }
            )

            MODULE.write_summary_tsv(path, [row])
            loaded = MODULE.load_summary_tsv(path)

        self.assertEqual([summary.values for summary in loaded], [row.values])

    def test_summary_input_policy_actionable_markdown_uses_existing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            path.write_text(
                "\t".join(MODULE.SUMMARY_HEADERS)
                + "\n"
                + "\t".join(
                    [
                        "enwiki_page_5m",
                        "article_category",
                        "5000000",
                        "1918305",
                        "128",
                        "lmdb",
                        "mmap-array",
                        "mmap-array",
                        "mmap-array",
                        "mmap-array",
                        "mmap-array",
                        "lmdb:3.251|mmap-array:3.288",
                        "lmdb:1219.110|mmap-array:236.382",
                        "delimited-artifact:400.000|mmap-array:350.000",
                        "mmap-array:350.000",
                        "mmap-array:850.000",
                        "lmdb:142443019|mmap-array:80000643",
                    ]
                )
                + "\n"
            )

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--summary-input",
                    str(path),
                    "--format",
                    "policy-actionable-markdown",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("| Policy mode | Policy artifact value | Policy vs best |", result.stdout)
        self.assertIn("| bucket_c0 | ms | delimited-artifact | 400.000 | 1.143x | mmap-array | 350.000 | diff |", result.stdout)
        self.assertNotIn("| lookup_c0 | ms | lmdb | 3.251 | 1.000x | lmdb | 3.251 | match |", result.stdout)

    def test_summary_input_rejects_raw_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            path.write_text("\t".join(MODULE.SUMMARY_HEADERS) + "\n")
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--summary-input",
                    str(path),
                    "--format",
                    "tsv",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--summary-input requires a summary-* or policy-* --format", result.stderr)

    def test_summary_input_rejects_summary_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "summary.tsv"
            output_path = Path(tmp) / "copy.tsv"
            input_path.write_text("\t".join(MODULE.SUMMARY_HEADERS) + "\n")
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--summary-input",
                    str(input_path),
                    "--summary-output",
                    str(output_path),
                    "--format",
                    "summary-tsv",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--summary-output cannot be used with --summary-input", result.stderr)

    def test_skip_missing_scale_errors_when_nothing_remains(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "__definitely_missing_scale__",
                "--skip-missing-scales",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all requested scales were missing", result.stderr)

    def test_dev_scale_reports_all_backends_with_matching_hashes(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
                "--format",
                "tsv",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 11)
        headers = lines[0].split("\t")
        rows = [dict(zip(headers, line.split("\t"))) for line in lines[1:]]

        self.assertEqual(
            [row["mode"] for row in rows],
            [
                "preload",
                "binary-artifact",
                "delimited-artifact",
                "lmdb",
                "mmap-array",
                "policy-configured-lookup-c0",
                "policy-configured-lookup-c1",
                "policy-configured-bucket-c0",
                "policy-configured-bucket-c1",
                "policy-configured-scan",
            ],
        )
        self.assertEqual({row["scale"] for row in rows}, {"dev"})
        self.assertEqual({row["run"] for row in rows}, {"1"})
        self.assertEqual({row["relation"] for row in rows}, {"category_parent"})
        self.assertEqual({row["scan_hash"] for row in rows}, {rows[0]["scan_hash"]})
        self.assertEqual({row["lookup_hash"] for row in rows}, {rows[0]["lookup_hash"]})
        self.assertEqual({row["lookup_col1_hash"] for row in rows}, {rows[0]["lookup_col1_hash"]})
        self.assertEqual({row["bucket_hash"] for row in rows}, {rows[0]["bucket_hash"]})
        self.assertEqual({row["bucket_col1_hash"] for row in rows}, {rows[0]["bucket_col1_hash"]})
        self.assertEqual(rows[-1]["mode"], "policy-configured-scan")

        for row in rows:
            self.assertGreater(int(row["rows"]), 0)
            self.assertGreater(int(row["distinct_categories"]), 0)
            self.assertEqual(row["lookup_keys"], "4")
            for column in ("artifact_bytes", "open_ms", "lookup_ms", "lookup_col1_ms", "bucket_ms", "bucket_col1_ms", "scan_ms", "retained_bytes"):
                float(row[column])

    def test_summary_markdown_reports_best_modes(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
                "--format",
                "summary-markdown",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("| Scale | Relation | Rows | Distinct values | Lookup keys | Best lookup c0 | Best lookup c1 |", result.stdout)
        self.assertIn("| dev |", result.stdout)
        self.assertIn("Smallest artifact", result.stdout)

    def test_summary_output_writes_reusable_summary_artifact(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary" / "effective-distance.tsv"
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--scales",
                    "dev",
                    "--lookup-keys",
                    "4",
                    "--lookup-repetitions",
                    "1",
                    "--repetitions",
                    "1",
                    "--format",
                    "summary-markdown",
                    "--summary-output",
                    str(summary_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            self.assertTrue(summary_path.exists())
            summaries = MODULE.load_summary_tsv(summary_path)

            offline = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--summary-input",
                    str(summary_path),
                    "--format",
                    "policy-actionable-markdown",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].values["scale"], "dev")
        self.assertEqual(offline.returncode, 0, msg=f"stdout:\n{offline.stdout}\nstderr:\n{offline.stderr}")
        self.assertIn("| Policy mode | Policy artifact value | Policy vs best |", offline.stdout)

    def test_policy_actionable_markdown_cli_reports_policy_diffs(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
                "--format",
                "policy-actionable-markdown",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("| Policy mode | Policy artifact value | Policy vs best |", result.stdout)
        self.assertIn("| diff |", result.stdout)
        self.assertNotIn("| match |", result.stdout)

    def test_dev_article_category_relation_reports_all_backends(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--relation",
                "article_category",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
                "--format",
                "tsv",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        lines = result.stdout.strip().splitlines()
        headers = lines[0].split("\t")
        rows = [dict(zip(headers, line.split("\t"))) for line in lines[1:]]
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["relation"] for row in rows}, {"article_category"})
        self.assertTrue(
            {
                "policy-configured-lookup-c0",
                "policy-configured-lookup-c1",
                "policy-configured-bucket-c0",
                "policy-configured-bucket-c1",
                "policy-configured-scan",
            }.issubset({row["mode"] for row in rows})
        )
        self.assertEqual({row["scan_hash"] for row in rows}, {rows[0]["scan_hash"]})
        self.assertEqual({row["lookup_hash"] for row in rows}, {rows[0]["lookup_hash"]})
        self.assertEqual({row["lookup_col1_hash"] for row in rows}, {rows[0]["lookup_col1_hash"]})

    def test_summary_full_markdown_reports_timing_details(self) -> None:
        row = MODULE.SummaryRow(
            {
                "scale": "rust_lmdb_500k",
                "relation": "category_parent",
                "rows": "500000",
                "distinct_categories": "225697",
                "lookup_keys": "64",
                "best_lookup_mode": "mmap-array",
                "best_lookup_col1_mode": "mmap-array",
                "best_bucket_mode": "mmap-array",
                "best_bucket_col1_mode": "mmap-array",
                "best_scan_mode": "preload",
                "smallest_artifact_mode": "mmap-array",
                "lookup_ms_by_mode": "lmdb:2.087|mmap-array:1.250",
                "lookup_col1_ms_by_mode": "lmdb:2.500|mmap-array:1.750",
                "bucket_ms_by_mode": "lmdb:718.217|mmap-array:250.000",
                "bucket_col1_ms_by_mode": "lmdb:701.000|mmap-array:240.000",
                "scan_ms_by_mode": "lmdb:129.326|preload:1.000",
                "artifact_bytes_by_mode": "lmdb:27971980|mmap-array:4000000",
            }
        )
        output = MODULE.render_summary_full_markdown([row])
        self.assertIn("Lookup c0 ms by mode", output)
        self.assertIn("Lookup c1 ms by mode", output)
        self.assertIn("lmdb:2.087|mmap-array:1.250", output)
        self.assertIn("lmdb:2.500|mmap-array:1.750", output)
        self.assertIn("Artifact bytes by mode", output)

    def test_policy_rows_report_best_overall_and_artifact_modes(self) -> None:
        row = MODULE.SummaryRow(
            {
                "scale": "enwiki_page_5m",
                "relation": "article_category",
                "rows": "5000000",
                "distinct_categories": "1918305",
                "lookup_keys": "128",
                "best_lookup_mode": "lmdb",
                "best_lookup_col1_mode": "preload",
                "best_bucket_mode": "delimited-artifact",
                "best_bucket_col1_mode": "mmap-array",
                "best_scan_mode": "preload",
                "smallest_artifact_mode": "mmap-array",
                "lookup_ms_by_mode": "lmdb:3.251|mmap-array:3.288|preload:4.000",
                "lookup_col1_ms_by_mode": "lmdb:1219.110|mmap-array:236.382|preload:20.000",
                "bucket_ms_by_mode": "delimited-artifact:400.000|lmdb:500.000|mmap-array:450.000|preload:600.000",
                "bucket_col1_ms_by_mode": "delimited-artifact:600.000|lmdb:700.000|mmap-array:350.000|preload:500.000",
                "scan_ms_by_mode": "lmdb:900.000|mmap-array:850.000|preload:50.000",
                "artifact_bytes_by_mode": "lmdb:142443019|mmap-array:80000643|preload:0",
            }
        )
        rows = MODULE.policy_rows_from_summaries([row])
        by_shape = {policy.values["access_shape"]: policy.values for policy in rows}

        self.assertEqual(by_shape["lookup_c0"]["best_mode"], "lmdb")
        self.assertEqual(by_shape["lookup_c0"]["best_artifact_mode"], "lmdb")
        self.assertEqual(by_shape["lookup_c1"]["best_mode"], "preload")
        self.assertEqual(by_shape["lookup_c1"]["best_artifact_mode"], "mmap-array")
        self.assertEqual(by_shape["scan"]["best_mode"], "preload")
        self.assertEqual(by_shape["scan"]["best_artifact_mode"], "mmap-array")
        self.assertEqual(by_shape["storage"]["best_mode"], "mmap-array")
        self.assertEqual(by_shape["storage"]["best_value"], "80000643")
        self.assertNotIn("preload:0", by_shape["storage"]["values_by_mode"])

    def test_policy_renderers_include_access_shape_rows(self) -> None:
        rows = [
            MODULE.PolicyRow(
                {
                    "scale": "dev",
                    "relation": "category_parent",
                    "rows": "6",
                    "distinct_categories": "4",
                    "lookup_keys": "2",
                    "access_shape": "lookup_c0",
                    "metric": "ms",
                    "best_mode": "mmap-array",
                    "best_value": "1.250",
                    "best_artifact_mode": "mmap-array",
                    "best_artifact_value": "1.250",
                    "values_by_mode": "lmdb:2.000|mmap-array:1.250|preload:1.500",
                }
            )
        ]

        tsv = MODULE.render_policy_tsv(rows)
        markdown = MODULE.render_policy_markdown(rows)
        self.assertTrue(tsv.startswith("scale\trelation\trows\tdistinct_categories\tlookup_keys\taccess_shape"))
        self.assertIn("\tlookup_c0\tms\tmmap-array\t1.250\tmmap-array\t1.250\t", tsv)
        self.assertIn("| Access shape | Metric | Best mode |", markdown)
        self.assertIn("| dev | category_parent | 6 | lookup_c0 | ms | mmap-array | 1.250 |", markdown)
        self.assertIn("lmdb:2.000<br>mmap-array:1.250<br>preload:1.500", markdown)

    def test_policy_compare_rows_report_runtime_policy_match_and_diff(self) -> None:
        row = MODULE.SummaryRow(
            {
                "scale": "enwiki_page_5m",
                "relation": "article_category",
                "rows": "5000000",
                "distinct_categories": "1918305",
                "lookup_keys": "128",
                "best_lookup_mode": "lmdb",
                "best_lookup_col1_mode": "preload",
                "best_bucket_mode": "mmap-array",
                "best_bucket_col1_mode": "mmap-array",
                "best_scan_mode": "preload",
                "smallest_artifact_mode": "mmap-array",
                "lookup_ms_by_mode": "lmdb:3.251|mmap-array:3.288|preload:4.000",
                "lookup_col1_ms_by_mode": "lmdb:1219.110|mmap-array:236.382|preload:20.000",
                "bucket_ms_by_mode": "delimited-artifact:400.000|lmdb:500.000|mmap-array:350.000|preload:600.000",
                "bucket_col1_ms_by_mode": "delimited-artifact:600.000|lmdb:700.000|mmap-array:350.000|preload:500.000",
                "scan_ms_by_mode": "lmdb:900.000|mmap-array:850.000|preload:50.000",
                "artifact_bytes_by_mode": "lmdb:142443019|mmap-array:80000643|preload:0",
            }
        )

        rows = MODULE.policy_compare_rows_from_summaries([row])
        by_shape = {policy.values["access_shape"]: policy.values for policy in rows}

        self.assertEqual(by_shape["lookup_c0"]["policy_mode"], "lmdb")
        self.assertEqual(by_shape["lookup_c0"]["best_artifact_mode"], "lmdb")
        self.assertEqual(by_shape["lookup_c0"]["policy_artifact_value"], "3.251")
        self.assertEqual(by_shape["lookup_c0"]["policy_vs_best"], "1.000x")
        self.assertEqual(by_shape["lookup_c0"]["status"], "match")
        self.assertEqual(by_shape["bucket_c0"]["policy_mode"], "delimited-artifact")
        self.assertEqual(by_shape["bucket_c0"]["best_artifact_mode"], "mmap-array")
        self.assertEqual(by_shape["bucket_c0"]["policy_artifact_value"], "400.000")
        self.assertEqual(by_shape["bucket_c0"]["policy_vs_best"], "1.143x")
        self.assertEqual(by_shape["bucket_c0"]["status"], "diff")
        self.assertEqual(by_shape["storage"]["policy_mode"], "mmap-array")
        self.assertEqual(by_shape["storage"]["policy_artifact_value"], "80000643")
        self.assertEqual(by_shape["storage"]["policy_vs_best"], "1.000x")
        self.assertEqual(by_shape["storage"]["status"], "match")

    def test_policy_compare_renderers_include_status(self) -> None:
        rows = [
            MODULE.PolicyCompareRow(
                {
                    "scale": "dev",
                    "relation": "category_parent",
                    "rows": "6",
                    "distinct_categories": "4",
                    "lookup_keys": "2",
                    "access_shape": "lookup_c0",
                    "metric": "ms",
                    "policy_mode": "mmap-array",
                    "policy_artifact_value": "2.000",
                    "policy_vs_best": "1.600x",
                    "best_artifact_mode": "lmdb",
                    "best_artifact_value": "1.250",
                    "status": "diff",
                    "values_by_mode": "lmdb:1.250|mmap-array:2.000",
                }
            )
        ]

        tsv = MODULE.render_policy_compare_tsv(rows)
        markdown = MODULE.render_policy_compare_markdown(rows)
        self.assertTrue(tsv.startswith("scale\trelation\trows\tdistinct_categories\tlookup_keys\taccess_shape"))
        self.assertIn("\tlookup_c0\tms\tmmap-array\t2.000\t1.600x\tlmdb\t1.250\tdiff\t", tsv)
        self.assertIn("| Access shape | Metric | Policy mode | Policy artifact value |", markdown)
        self.assertIn("| dev | category_parent | 6 | lookup_c0 | ms | mmap-array | 2.000 | 1.600x | lmdb | 1.250 | diff |", markdown)
        self.assertIn("lmdb:1.250<br>mmap-array:2.000", markdown)

    def test_policy_compare_reports_missing_policy_mode(self) -> None:
        row = MODULE.SummaryRow(
            {
                "scale": "dev",
                "relation": "category_parent",
                "rows": "6",
                "distinct_categories": "4",
                "lookup_keys": "2",
                "best_lookup_mode": "lmdb",
                "best_lookup_col1_mode": "lmdb",
                "best_bucket_mode": "lmdb",
                "best_bucket_col1_mode": "lmdb",
                "best_scan_mode": "lmdb",
                "smallest_artifact_mode": "lmdb",
                "lookup_ms_by_mode": "lmdb:1.250",
                "lookup_col1_ms_by_mode": "lmdb:1.250",
                "bucket_ms_by_mode": "lmdb:1.250",
                "bucket_col1_ms_by_mode": "lmdb:1.250",
                "scan_ms_by_mode": "lmdb:1.250",
                "artifact_bytes_by_mode": "lmdb:1024",
            }
        )

        rows = MODULE.policy_compare_rows_from_summaries([row])
        by_shape = {policy.values["access_shape"]: policy.values for policy in rows}

        self.assertEqual(by_shape["lookup_c0"]["policy_mode"], "mmap-array")
        self.assertEqual(by_shape["lookup_c0"]["policy_artifact_value"], "")
        self.assertEqual(by_shape["lookup_c0"]["policy_vs_best"], "")
        self.assertEqual(by_shape["lookup_c0"]["status"], "missing-policy-mode")

    def test_policy_actionable_rows_filter_matches(self) -> None:
        rows = [
            MODULE.PolicyCompareRow(
                {
                    "scale": "dev",
                    "relation": "category_parent",
                    "rows": "6",
                    "distinct_categories": "4",
                    "lookup_keys": "2",
                    "access_shape": "lookup_c0",
                    "metric": "ms",
                    "policy_mode": "mmap-array",
                    "policy_artifact_value": "1.250",
                    "policy_vs_best": "1.000x",
                    "best_artifact_mode": "mmap-array",
                    "best_artifact_value": "1.250",
                    "status": "match",
                    "values_by_mode": "mmap-array:1.250",
                }
            ),
            MODULE.PolicyCompareRow(
                {
                    "scale": "dev",
                    "relation": "category_parent",
                    "rows": "6",
                    "distinct_categories": "4",
                    "lookup_keys": "2",
                    "access_shape": "bucket_c0",
                    "metric": "ms",
                    "policy_mode": "mmap-array",
                    "policy_artifact_value": "2.000",
                    "policy_vs_best": "1.600x",
                    "best_artifact_mode": "lmdb",
                    "best_artifact_value": "1.250",
                    "status": "diff",
                    "values_by_mode": "lmdb:1.250|mmap-array:2.000",
                }
            ),
        ]

        actionable = MODULE.policy_actionable_rows(rows)
        self.assertEqual([row.values["access_shape"] for row in actionable], ["bucket_c0"])
        self.assertNotIn("lookup_c0", MODULE.render_policy_actionable_tsv(rows))
        self.assertIn("bucket_c0", MODULE.render_policy_actionable_markdown(rows))

    def test_artifact_root_reuses_existing_manifests(self) -> None:
        if shutil.which("dotnet") is None:
            self.skipTest("dotnet is not available")
        if not LIGHTNINGDB_PACKAGE.exists():
            self.skipTest("LightningDB 0.21.0 package is not available in the local NuGet cache")

        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "artifacts"
            command = [
                "python3",
                str(SCRIPT),
                "--scales",
                "dev",
                "--lookup-keys",
                "4",
                "--lookup-repetitions",
                "1",
                "--repetitions",
                "1",
                "--artifact-root",
                str(artifact_root),
                "--format",
                "summary-markdown",
            ]
            first = subprocess.run(
                command + ["--refresh-artifacts"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            self.assertEqual(first.returncode, 0, msg=f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}")
            manifest = artifact_root / "dev" / "category_parent.lmdb.manifest.json"
            self.assertTrue(manifest.exists())
            first_mtime = manifest.stat().st_mtime_ns

            second = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            self.assertEqual(second.returncode, 0, msg=f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}")
            self.assertEqual(manifest.stat().st_mtime_ns, first_mtime)


if __name__ == "__main__":
    unittest.main()
