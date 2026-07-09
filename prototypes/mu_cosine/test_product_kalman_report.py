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

from product_kalman_report import (
    add_artifact_bootstrap_intervals,
    build_product_kalman_markdown_report,
    main,
    validate_artifact_score_consistency,
)
from product_kalman_table_evaluation import run_product_kalman_table_evaluation


def assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (OSError, ValueError):
        return
    raise AssertionError(f"{fn.__name__} should have raised")


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


def make_artifacts(tmp, bootstrap_nll=50):
    table = Path(tmp) / "holdout.csv"
    input_npz = Path(tmp) / "input.npz"
    input_manifest = Path(tmp) / "input.manifest.json"
    scores_json = Path(tmp) / "scores.json"
    eval_npz = Path(tmp) / "eval_artifacts.npz"
    write_table(table)
    kwargs = {}
    if bootstrap_nll:
        kwargs.update({
            "bootstrap_nll": bootstrap_nll,
            "bootstrap_seed": 7,
            "bootstrap_confidence": 0.90,
        })
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
        **kwargs,
    )
    return input_manifest, scores_json, eval_npz


def split_manifest_fixture():
    return {
        "source_table": {
            "path": "raw.tsv",
            "sha256": "a" * 64,
            "delimiter": "\t",
            "rows": 120,
        },
        "output_table": {
            "path": "split_table.tsv",
            "sha256": "b" * 64,
            "rows": 120,
        },
        "split": {
            "seed": 5,
            "evaluation_unit_frac": 0.25,
            "unit_columns": ["unit"],
            "sampled_evaluation_unit_count": 30,
            "observed_calibration_unit_count": 90,
            "observed_evaluation_unit_count": 30,
            "observed_unit_overlap_count": 0,
            "disjoint_observed_units": True,
            "calibration_rows": 90,
            "evaluation_rows": 30,
            "omitted_crossing_rows": 0,
        },
    }


def test_markdown_report_records_scores_and_guardrails():
    with tempfile.TemporaryDirectory() as tmp:
        input_manifest, scores_json, _ = make_artifacts(tmp)
        scores = json.loads(scores_json.read_text())
        manifest = json.loads(input_manifest.read_text())
        report = build_product_kalman_markdown_report(scores, manifest, title="Synthetic Product-Kalman Report")

        assert report.startswith("# Synthetic Product-Kalman Report\n")
        assert "This report is descriptive" in report
        assert "| product_kalman |" in report
        assert "mahalanobis_per_dim" in report
        assert "mean_sq_mahalanobis" in report
        assert "sq_mahalanobis_q95" in report
        assert "| prior | product_kalman |" in report
        assert "## NLL Improvement Bootstrap Intervals" in report
        assert "observed_gain" in report
        assert "paired row-resampling" in report
        assert "| calibration_rows | 80 |" in report
        assert "| evaluation_rows | 40 |" in report
        assert "| ids_disjoint_and_unique | True |" in report
        assert "| prior | prior |" in report
        assert "does not encode a decision rule" in report
        assert "should be compared against the registered joint-posterior" in report

def test_markdown_report_records_split_materialization_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        input_manifest, scores_json, _ = make_artifacts(tmp)
        scores = json.loads(scores_json.read_text())
        manifest = json.loads(input_manifest.read_text())
        scores["inputs"]["original_table"] = "raw.tsv"
        scores["inputs"]["split_manifest"] = "split.manifest.json"
        report = build_product_kalman_markdown_report(
            scores,
            manifest,
            split_manifest=split_manifest_fixture(),
            title="Split Product-Kalman Report",
        )

        assert "## Split Materialization" in report
        assert "| original_table | raw.tsv |" in report
        assert "| split_manifest | split.manifest.json |" in report
        assert "| unit_columns | unit |" in report
        assert "| observed_unit_overlap_count | 0 |" in report
        assert "| disjoint_observed_units | True |" in report
        assert "| omitted_crossing_rows | 0 |" in report


def test_posthoc_bootstrap_intervals_can_be_loaded_from_evaluation_npz():
    with tempfile.TemporaryDirectory() as tmp:
        input_manifest, scores_json, eval_npz = make_artifacts(tmp, bootstrap_nll=0)
        scores = json.loads(scores_json.read_text())
        assert "nll_improvement_bootstrap_vs_independent_kalman" not in scores
        enriched = add_artifact_bootstrap_intervals(
            scores,
            evaluation_npz=eval_npz,
            n_boot=50,
            seed=7,
            confidence=0.90,
        )
        artifact_summary = validate_artifact_score_consistency(scores, evaluation_npz=eval_npz)
        assert artifact_summary["score_order"] == scores["score_order"]
        recorded_path_enriched = add_artifact_bootstrap_intervals(
            scores,
            n_boot=50,
            seed=7,
            confidence=0.90,
        )
        assert "nll_improvement_bootstrap_vs_prior" in recorded_path_enriched

        bad_order = dict(scores)
        bad_order["score_order"] = list(reversed(scores["score_order"]))
        assert_raises(add_artifact_bootstrap_intervals, bad_order, evaluation_npz=eval_npz, n_boot=10)
        bad_mean = json.loads(json.dumps(scores))
        bad_mean["scores"]["product_kalman"]["mean_nll"] += 1.0
        assert_raises(add_artifact_bootstrap_intervals, bad_mean, evaluation_npz=eval_npz, n_boot=10)
        boot = enriched["nll_improvement_bootstrap_vs_independent_kalman"]["product_kalman"]
        assert boot["n_boot"] == 50
        assert boot["seed"] == 7
        assert boot["confidence"] == 0.90

        output_md = Path(tmp) / "posthoc.md"
        output_json = Path(tmp) / "posthoc.scores.json"
        rc = main([
            str(scores_json),
            "--input-manifest",
            str(input_manifest),
            "--evaluation-npz",
            str(eval_npz),
            "--bootstrap-nll",
            "50",
            "--bootstrap-seed",
            "7",
            "--bootstrap-confidence",
            "0.90",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ])
        assert rc == 0
        text = output_md.read_text()
        assert "## NLL Improvement Bootstrap Intervals" in text
        assert "| independent_kalman | product_kalman |" in text
        enriched_json = json.loads(output_json.read_text())
        assert enriched_json["nll_improvement_bootstrap_vs_independent_kalman"]["product_kalman"] == boot


def test_markdown_report_cli_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        input_manifest, scores_json, _ = make_artifacts(tmp)
        output_md = Path(tmp) / "report.md"
        split_manifest = Path(tmp) / "split.manifest.json"
        split_manifest.write_text(json.dumps(split_manifest_fixture()), encoding="utf-8")
        rc = main([
            str(scores_json),
            "--input-manifest",
            str(input_manifest),
            "--split-manifest",
            str(split_manifest),
            "--output-md",
            str(output_md),
            "--title",
            "CLI Product-Kalman Report",
        ])
        assert rc == 0
        text = output_md.read_text()
        assert "# CLI Product-Kalman Report" in text
        assert "## Scores" in text
        assert "mahalanobis_per_dim" in text
        assert "sq_mahalanobis_q95" in text
        assert "## NLL Improvements" in text
        assert "## NLL Improvement Bootstrap Intervals" in text
        assert "## Split Materialization" in text
        assert "source_table_sha256" in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman report tests passed")
