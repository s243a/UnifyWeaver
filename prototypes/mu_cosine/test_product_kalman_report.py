#!/usr/bin/env python3
"""Tests for Product-Kalman Markdown report generation.

Run: `python3 test_product_kalman_report.py`.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman_report import build_product_kalman_markdown_report, main
from product_kalman_table_evaluation import run_product_kalman_table_evaluation


def synthetic_identity_split(n_cal=80, n_eval=40, seed=23):
    rng = np.random.default_rng(seed)
    n = n_cal + n_eval
    target = rng.normal(size=(n, 1))
    error_covariance = np.array([[1.0, 0.4], [0.4, 0.4]])
    errors = rng.multivariate_normal([0.0, 0.0], error_covariance, size=n)
    prior = target - errors[:, :1]
    measurement = target + errors[:, 1:]
    return prior, measurement, target, n_cal


def write_table(path):
    prior, measurement, target, n_cal = synthetic_identity_split()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "prior", "measurement", "target"])
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


def make_artifacts(tmp):
    table = Path(tmp) / "holdout.csv"
    input_npz = Path(tmp) / "input.npz"
    input_manifest = Path(tmp) / "input.manifest.json"
    scores_json = Path(tmp) / "scores.json"
    eval_npz = Path(tmp) / "eval_artifacts.npz"
    write_table(table)
    run_product_kalman_table_evaluation(
        table,
        input_npz,
        prior_cols=["prior"],
        measurement_cols=["measurement"],
        target_cols=["target"],
        input_manifest=input_manifest,
        output_json=scores_json,
        output_npz=eval_npz,
        jitter=1e-8,
    )
    return input_manifest, scores_json


def test_markdown_report_records_scores_and_guardrails():
    with tempfile.TemporaryDirectory() as tmp:
        input_manifest, scores_json = make_artifacts(tmp)
        scores = json.loads(scores_json.read_text())
        manifest = json.loads(input_manifest.read_text())
        report = build_product_kalman_markdown_report(scores, manifest, title="Synthetic Product-Kalman Report")

        assert report.startswith("# Synthetic Product-Kalman Report\n")
        assert "This report is descriptive" in report
        assert "| product_kalman |" in report
        assert "| prior | product_kalman |" in report
        assert "| calibration_rows | 80 |" in report
        assert "| evaluation_rows | 40 |" in report
        assert "| ids_disjoint_and_unique | True |" in report
        assert "| prior | prior |" in report
        assert "does not encode a decision rule" in report
        assert "should be compared against the registered joint-posterior" in report


def test_markdown_report_cli_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        input_manifest, scores_json = make_artifacts(tmp)
        output_md = Path(tmp) / "report.md"
        rc = main([
            str(scores_json),
            "--input-manifest",
            str(input_manifest),
            "--output-md",
            str(output_md),
            "--title",
            "CLI Product-Kalman Report",
        ])
        assert rc == 0
        text = output_md.read_text()
        assert "# CLI Product-Kalman Report" in text
        assert "## Scores" in text
        assert "## NLL Improvements" in text
        assert "source_table_sha256" in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman report tests passed")
