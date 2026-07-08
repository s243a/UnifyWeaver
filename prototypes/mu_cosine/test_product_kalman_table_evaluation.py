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
from product_kalman_table_evaluation import main, run_product_kalman_table_evaluation


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


def test_table_runner_writes_input_and_evaluation_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        input_npz = Path(tmp) / "input.npz"
        input_manifest = Path(tmp) / "input.manifest.json"
        output_json = Path(tmp) / "scores.json"
        output_npz = Path(tmp) / "eval_artifacts.npz"
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
            jitter=1e-8,
        )

        assert input_npz.exists()
        assert input_manifest.exists()
        assert output_json.exists()
        assert output_npz.exists()
        assert run.input_arrays["calibration_prior_mean"].shape == (80, 1)
        assert run.result.nll_improvement("prior", "product_kalman") > 0.65
        assert run.result.nll_improvement("independent_kalman", "product_kalman") > 0.05

        summary = json.loads(output_json.read_text())
        assert summary["inputs"]["source_table"] == str(table)
        assert summary["inputs"]["input_npz"] == str(input_npz)
        assert summary["inputs"]["input_manifest"] == str(input_manifest)
        assert summary["inputs"]["evaluation_npz"] == str(output_npz)
        assert summary["score_order"] == ["prior", "measurement", "independent_kalman", "product_kalman"]
        assert summary["nll_improvement_vs_prior"]["product_kalman"] > 0.65

        manifest = json.loads(input_manifest.read_text())
        assert manifest["splits"]["calibration_rows"] == 80
        assert manifest["splits"]["evaluation_rows"] == 40
        assert manifest["ids"]["disjoint_and_unique"] is True

        with np.load(output_npz, allow_pickle=False) as artifact:
            assert artifact["product_kalman_mean"].shape == (40, 1)
            assert artifact["score_names"].tolist() == summary["score_order"]


def test_table_runner_cli_roundtrips_against_npz_evaluator():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.tsv"
        input_npz = Path(tmp) / "input.npz"
        input_manifest = Path(tmp) / "input.manifest.json"
        output_json = Path(tmp) / "scores.json"
        write_table(table, delimiter="\t")

        rc = main([
            str(table),
            "--input-npz",
            str(input_npz),
            "--input-manifest",
            str(input_manifest),
            "--output-json",
            str(output_json),
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
        from_table = json.loads(output_json.read_text())
        from_npz = run_product_kalman_holdout_npz(input_npz, jitter=1e-8)
        assert abs(
            from_table["scores"]["product_kalman"]["mean_nll"]
            - from_npz.score("product_kalman").mean_nll
        ) < 1e-12
        manifest = json.loads(input_manifest.read_text())
        assert manifest["source_table"]["delimiter"] == "\t"
        assert manifest["ids"]["overlap_count"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman table-evaluation tests passed")
