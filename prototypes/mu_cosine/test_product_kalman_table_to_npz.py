#!/usr/bin/env python3
"""Tests for Product-Kalman CSV/TSV table-to-NPZ input builder.

Run: `python3 test_product_kalman_table_to_npz.py`.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from product_kalman_evaluation import run_product_kalman_holdout_npz
from product_kalman_table_to_npz import (
    TABLE_INPUT_MANIFEST_SCHEMA_VERSION,
    build_product_kalman_input_manifest,
    build_product_kalman_npz_from_table,
    main,
    parse_column_list,
    parse_matrix_literal,
    read_product_kalman_table,
    write_product_kalman_manifest,
)


def assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__} should have raised ValueError")


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
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "id", "prior", "measurement", "target", "hop"],
            delimiter=delimiter,
        )
        writer.writeheader()
        for i in range(len(target)):
            split = "calibration" if i < n_cal else "evaluation"
            writer.writerow({
                "split": split,
                "id": f"{split}-{i}",
                "prior": f"{prior[i, 0]:.17g}",
                "measurement": f"{measurement[i, 0]:.17g}",
                "target": f"{target[i, 0]:.17g}",
                "hop": str(1 + i % 3),
            })


def test_column_and_matrix_parsers():
    assert parse_column_list("a,b, c") == ["a", "b", "c"]
    assert_raises(parse_column_list, "a,a")
    np.testing.assert_allclose(parse_matrix_literal("1,0;0,1"), np.eye(2))
    assert parse_matrix_literal("") is None
    assert_raises(parse_matrix_literal, "1,0;2")
    assert_raises(parse_matrix_literal, "1,x")


def test_table_builder_roundtrips_into_evaluator():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        npz = Path(tmp) / "holdout.npz"
        write_table(table)
        arrays = build_product_kalman_npz_from_table(
            table,
            npz,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
            group_cols=["hop"],
        )
        assert arrays["calibration_prior_mean"].shape == (80, 1)
        assert arrays["evaluation_prior_mean"].shape == (40, 1)
        assert arrays["calibration_ids"].shape == (80,)
        assert arrays["calibration_groups"].shape == (80,)
        assert arrays["evaluation_groups"].shape == (40,)
        assert set(arrays["calibration_groups"].tolist()) == {"1", "2", "3"}
        assert npz.exists()

        result = run_product_kalman_holdout_npz(npz, jitter=1e-8)
        assert result.nll_improvement("prior", "product_kalman") > 0.65
        assert result.nll_improvement("independent_kalman", "product_kalman") > 0.05


def test_input_manifest_records_schema_and_id_audit():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        manifest_path = Path(tmp) / "manifest.json"
        write_table(table)
        arrays = read_product_kalman_table(
            table,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
            group_cols=["hop"],
        )
        manifest = build_product_kalman_input_manifest(
            table,
            arrays,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
            group_cols=["hop"],
        )
        assert manifest["schema_version"] == TABLE_INPUT_MANIFEST_SCHEMA_VERSION
        assert len(manifest["source_table"]["sha256"]) == 64
        assert manifest["splits"]["calibration_rows"] == 80
        assert manifest["splits"]["evaluation_rows"] == 40
        assert manifest["columns"]["prior"] == ["prior"]
        assert manifest["dimensions"] == {"state_dim": 1, "prior_dim": 1, "observation_dim": 1}
        assert manifest["ids"]["disjoint_and_unique"] is True
        assert manifest["groups"]["present"] is True
        assert manifest["groups"]["columns"] == ["hop"]
        assert manifest["groups"]["calibration_unique_count"] == 3
        assert manifest["groups"]["calibration_counts"]["1"] == 27
        assert manifest["arrays"]["calibration_prior_mean"]["shape"] == [80, 1]
        assert manifest["arrays"]["calibration_groups"]["shape"] == [80]
        assert manifest["H"]["present"] is False
        write_product_kalman_manifest(manifest_path, manifest)
        loaded = json.loads(manifest_path.read_text())
        assert loaded["source_table"]["sha256"] == manifest["source_table"]["sha256"]


def test_tsv_delimiter_inference_and_cli():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.tsv"
        npz = Path(tmp) / "holdout.npz"
        manifest = Path(tmp) / "holdout.manifest.json"
        write_table(table, delimiter="	")
        rc = main([
            str(table),
            "--output-npz",
            str(npz),
            "--output-manifest",
            str(manifest),
            "--prior-cols",
            "prior",
            "--measurement-cols",
            "measurement",
            "--target-cols",
            "target",
            "--group-cols",
            "hop",
        ])
        assert rc == 0
        with np.load(npz, allow_pickle=False) as data:
            assert data["calibration_prior_mean"].shape == (80, 1)
            assert data["evaluation_ids"].shape == (40,)
            assert data["calibration_groups"].shape == (80,)
            assert data["evaluation_groups"].shape == (40,)
        manifest_data = json.loads(manifest.read_text())
        assert manifest_data["source_table"]["delimiter"] == "	"
        assert manifest_data["ids"]["overlap_count"] == 0
        assert manifest_data["groups"]["columns"] == ["hop"]


def test_nonidentity_H_is_stored_and_shape_checked():
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "holdout.csv"
        with open(table, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "id", "p0", "p1", "m0", "t0", "t1"],
            )
            writer.writeheader()
            for i in range(8):
                split = "calibration" if i < 5 else "evaluation"
                writer.writerow({
                    "split": split,
                    "id": f"row-{i}",
                    "p0": i * 0.1,
                    "p1": i * -0.2,
                    "m0": i * 0.3,
                    "t0": i * 0.1 + 0.01,
                    "t1": i * -0.2 - 0.02,
                })
        arrays = read_product_kalman_table(
            table,
            prior_cols=["p0", "p1"],
            measurement_cols=["m0"],
            target_cols=["t0", "t1"],
            H="1,-0.4",
        )
        np.testing.assert_allclose(arrays["H"], [[1.0, -0.4]])
        manifest = build_product_kalman_input_manifest(
            table,
            arrays,
            prior_cols=["p0", "p1"],
            measurement_cols=["m0"],
            target_cols=["t0", "t1"],
        )
        assert manifest["H"]["present"] is True
        assert manifest["H"]["shape"] == [1, 2]
        assert manifest["H"]["values"] == [[1.0, -0.4]]
        assert_raises(
            read_product_kalman_table,
            table,
            prior_cols=["p0", "p1"],
            measurement_cols=["m0"],
            target_cols=["t0", "t1"],
            H="1,0;0,1",
        )
        assert_raises(
            read_product_kalman_table,
            table,
            prior_cols=["p0", "p1"],
            measurement_cols=["m0"],
            target_cols=["t0", "t1"],
        )


def test_table_builder_rejects_bad_input():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.csv"
        missing.write_text("split,id,prior,target\ncalibration,a,1,1\n", encoding="utf-8")
        assert_raises(
            read_product_kalman_table,
            missing,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
        )

        bad_split = Path(tmp) / "bad_split.csv"
        bad_split.write_text("split,id,prior,measurement,target\ntrain,a,1,1,1\n", encoding="utf-8")
        assert_raises(
            read_product_kalman_table,
            bad_split,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
        )

        bad_value = Path(tmp) / "bad_value.csv"
        bad_value.write_text("split,id,prior,measurement,target\ncalibration,a,nope,1,1\n", encoding="utf-8")
        assert_raises(
            read_product_kalman_table,
            bad_value,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
        )

        bad_group = Path(tmp) / "bad_group.csv"
        bad_group.write_text(
            "split,id,prior,measurement,target,hop\ncalibration,a,1,1,1,\n",
            encoding="utf-8",
        )
        assert_raises(
            read_product_kalman_table,
            bad_group,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
            group_cols=["hop"],
        )

        bad_prior_dim = Path(tmp) / "bad_prior_dim.csv"
        bad_prior_dim.write_text("split,id,p0,measurement,target\ncalibration,a,1,1,1\n", encoding="utf-8")
        assert_raises(
            read_product_kalman_table,
            bad_prior_dim,
            prior_cols=["p0", "missing_prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
        )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman table-to-NPZ tests passed")
