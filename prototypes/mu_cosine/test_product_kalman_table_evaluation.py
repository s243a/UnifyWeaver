#!/usr/bin/env python3
"""Tests for the Product-Kalman table-to-evaluation runner.

Run: `python3 test_product_kalman_table_evaluation.py`.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman_evaluation import run_product_kalman_holdout_npz
from product_kalman_table_evaluation import (
    default_product_kalman_run_paths,
    default_product_kalman_split_paths,
    main,
    run_product_kalman_table_evaluation,
)


def synthetic_identity_split(n_cal=80, n_eval=40, seed=23):
    rng = np.random.default_rng(seed)
    n = n_cal + n_eval
    target = rng.normal(size=(n, 1))
    error_covariance = np.array([[1.0, 0.4], [0.4, 0.4]])
    errors = rng.multivariate_normal([0.0, 0.0], error_covariance, size=n)
    prior = target - errors[:, :1]
    measurement = target + errors[:, 1:]
    return prior, measurement, target, n_cal


def write_table(path, delimiter=","):
    prior, measurement, target, n_cal = synthetic_identity_split()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "prior", "measurement", "target"], delimiter=delimiter)
        writer.writeheader()
        for i in range(len(target)):
            split = "calibration" if i < n_cal else "evaluation"
            writer.writerow({
                "split": split,
                "id": f"{split}-{i}",
                "prior": f"{prior[i, 0]:.17g}",
                "measurement": f"{measurement[i, 0]:.17g}",
                "target": f"{target[i, 0]:.17g}",
            })


def write_unsplit_unit_table(path, delimiter="	"):
    prior, measurement, target, _ = synthetic_identity_split(n_cal=90, n_eval=30)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "unit", "prior", "measurement", "target"],
            delimiter=delimiter,
        )
        writer.writeheader()
        for i in range(len(target)):
            writer.writerow({
                "id": f"row-{i}",
                "unit": f"unit-{i}",
                "prior": f"{prior[i, 0]:.17g}",
                "measurement": f"{measurement[i, 0]:.17g}",
                "target": f"{target[i, 0]:.17g}",
            })


def test_table_runner_writes_input_and_evaluation_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        input_npz = Path(tmp) / "input.npz"
        input_manifest = Path(tmp) / "input.manifest.json"
        output_json = Path(tmp) / "scores.json"
        output_npz = Path(tmp) / "eval_artifacts.npz"
        output_md = Path(tmp) / "report.md"
        write_table(table)

        run = run_product_kalman_table_evaluation(
            table,
            input_npz,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
            input_manifest=input_manifest,
            output_json=output_json,
            output_npz=output_npz,
            output_md=output_md,
            report_title="Synthetic Table Product-Kalman Report",
            jitter=1e-8,
            bootstrap_nll=50,
            bootstrap_seed=7,
            bootstrap_confidence=0.90,
        )

        assert input_npz.exists()
        assert input_manifest.exists()
        assert output_json.exists()
        assert output_npz.exists()
        assert output_md.exists()
        assert run.input_arrays["calibration_prior_mean"].shape == (80, 1)
        assert run.result.nll_improvement("prior", "product_kalman") > 0.65
        assert run.result.nll_improvement("independent_kalman", "product_kalman") > 0.05

        summary = json.loads(output_json.read_text())
        assert summary["inputs"]["source_table"] == str(table)
        assert summary["inputs"]["input_npz"] == str(input_npz)
        assert summary["inputs"]["input_manifest"] == str(input_manifest)
        assert summary["inputs"]["evaluation_npz"] == str(output_npz)
        assert summary["inputs"]["report_md"] == str(output_md)
        assert summary["score_order"] == ["prior", "measurement", "independent_kalman", "product_kalman"]
        assert "mahalanobis_per_dim" in summary["scores"]["product_kalman"]
        assert "squared_mahalanobis_q95" in summary["scores"]["product_kalman"]
        assert summary["nll_improvement_vs_prior"]["product_kalman"] > 0.65
        boot = summary["nll_improvement_bootstrap_vs_independent_kalman"]["product_kalman"]
        assert boot["n_boot"] == 50
        assert boot["seed"] == 7
        assert boot["confidence"] == 0.90
        assert abs(
            boot["observed_mean_gain"]
            - summary["nll_improvement_vs_independent_kalman"]["product_kalman"]
        ) < 1e-12

        manifest = json.loads(input_manifest.read_text())
        assert manifest["splits"]["calibration_rows"] == 80
        assert manifest["splits"]["evaluation_rows"] == 40
        assert manifest["ids"]["disjoint_and_unique"] is True
        report = output_md.read_text()
        assert report == run.report_text
        assert "# Synthetic Table Product-Kalman Report" in report
        assert "## Scores" in report
        assert "mahalanobis_per_dim" in report
        assert "sq_mahalanobis_q95" in report
        assert "## NLL Improvement Bootstrap Intervals" in report
        assert "source_table_sha256" in report

        with np.load(output_npz, allow_pickle=False) as artifact:
            assert artifact["product_kalman_mean"].shape == (40, 1)
            assert artifact["score_names"].tolist() == summary["score_order"]
            assert artifact["score_mahalanobis_per_dim"].shape == (4,)
            assert artifact["score_squared_mahalanobis_q95"].shape == (4,)
            assert artifact["score_row_nll"].shape == (4, 40)
            assert artifact["score_row_squared_error"].shape == (4, 40)
            assert artifact["score_row_squared_mahalanobis"].shape == (4, 40)


def test_table_runner_output_dir_writes_canonical_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        output_dir = Path(tmp) / "product_kalman_run"
        write_table(table)

        rc = main([
            str(table),
            "--output-dir",
            str(output_dir),
            "--prior-cols",
            "prior",
            "--measurement-cols",
            "measurement",
            "--target-cols",
            "target",
            "--jitter",
            "1e-8",
            "--indent",
            "0",
        ])

        assert rc == 0
        paths = default_product_kalman_run_paths(output_dir)
        for path in paths.values():
            assert path.exists(), path
        summary = json.loads(paths["output_json"].read_text())
        assert summary["inputs"]["input_npz"] == str(paths["input_npz"])
        assert summary["inputs"]["input_manifest"] == str(paths["input_manifest"])
        assert summary["inputs"]["evaluation_npz"] == str(paths["output_npz"])
        assert summary["inputs"]["report_md"] == str(paths["output_md"])
        assert summary["nll_improvement_vs_prior"]["product_kalman"] > 0.65
        assert "# Product-Kalman Holdout Report" in paths["output_md"].read_text()


def test_table_runner_output_dir_can_materialize_split_before_evaluation():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "unsplit.tsv"
        output_dir = Path(tmp) / "product_kalman_run"
        write_unsplit_unit_table(table, delimiter="	")

        rc = main([
            str(table),
            "--output-dir",
            str(output_dir),
            "--split-unit-cols",
            "unit",
            "--evaluation-unit-frac",
            "0.25",
            "--split-seed",
            "5",
            "--prior-cols",
            "prior",
            "--measurement-cols",
            "measurement",
            "--target-cols",
            "target",
            "--jitter",
            "1e-8",
        ])

        assert rc == 0
        paths = default_product_kalman_run_paths(output_dir)
        split_paths = default_product_kalman_split_paths(output_dir, input_table=table)
        for path in list(paths.values()) + list(split_paths.values()):
            assert path.exists(), path
        assert split_paths["split_table"].name == "split_table.tsv"

        summary = json.loads(paths["output_json"].read_text())
        assert summary["inputs"]["source_table"] == str(split_paths["split_table"])
        assert summary["inputs"]["original_table"] == str(table)
        assert summary["inputs"]["split_manifest"] == str(split_paths["split_manifest"])
        assert summary["inputs"]["input_manifest"] == str(paths["input_manifest"])
        assert summary["nll_improvement_vs_prior"]["product_kalman"] > 0.65

        split_manifest = json.loads(split_paths["split_manifest"].read_text())
        assert split_manifest["source_table"]["path"] == str(table)
        assert split_manifest["source_table"]["delimiter"] == "	"
        assert split_manifest["output_table"]["path"] == str(split_paths["split_table"])
        assert split_manifest["split"]["omitted_crossing_rows"] == 0
        assert split_manifest["split"]["disjoint_observed_units"] is True
        assert split_manifest["split"]["evaluation_rows"] == 30
        input_manifest = json.loads(paths["input_manifest"].read_text())
        assert input_manifest["source_table"]["path"] == str(split_paths["split_table"])
        assert input_manifest["source_table"]["delimiter"] == "	"
        report = paths["output_md"].read_text()
        assert "## Split Materialization" in report
        assert "| original_table |" in report
        assert "| split_manifest |" in report
        assert "| omitted_crossing_rows | 0 |" in report


def test_table_runner_output_dir_allows_explicit_path_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        output_dir = Path(tmp) / "product_kalman_run"
        custom_scores = Path(tmp) / "custom_scores.json"
        write_table(table)

        rc = main([
            str(table),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(custom_scores),
            "--prior-cols",
            "prior",
            "--measurement-cols",
            "measurement",
            "--target-cols",
            "target",
            "--jitter",
            "1e-8",
        ])

        assert rc == 0
        paths = default_product_kalman_run_paths(output_dir)
        assert custom_scores.exists()
        assert not paths["output_json"].exists()
        assert paths["input_npz"].exists()
        assert paths["input_manifest"].exists()
        assert paths["output_npz"].exists()
        assert paths["output_md"].exists()
        summary = json.loads(custom_scores.read_text())
        assert summary["inputs"]["report_md"] == str(paths["output_md"])


def test_table_runner_cli_roundtrips_against_npz_evaluator():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.tsv"
        input_npz = Path(tmp) / "input.npz"
        input_manifest = Path(tmp) / "input.manifest.json"
        output_json = Path(tmp) / "scores.json"
        output_md = Path(tmp) / "scores.md"
        write_table(table, delimiter="\t")

        rc = main([
            str(table),
            "--input-npz",
            str(input_npz),
            "--input-manifest",
            str(input_manifest),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--report-title",
            "CLI Table Product-Kalman Report",
            "--prior-cols",
            "prior",
            "--measurement-cols",
            "measurement",
            "--target-cols",
            "target",
            "--jitter",
            "1e-8",
            "--indent",
            "0",
        ])

        assert rc == 0
        assert output_md.exists()
        assert "# CLI Table Product-Kalman Report" in output_md.read_text()
        from_table = json.loads(output_json.read_text())
        from_npz = run_product_kalman_holdout_npz(input_npz, jitter=1e-8)
        assert abs(
            from_table["scores"]["product_kalman"]["mean_nll"]
            - from_npz.score("product_kalman").mean_nll
        ) < 1e-12
        manifest = json.loads(input_manifest.read_text())
        assert manifest["source_table"]["delimiter"] == "\t"
        assert manifest["ids"]["overlap_count"] == 0
        assert from_table["inputs"]["report_md"] == str(output_md)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman table-evaluation tests passed")
