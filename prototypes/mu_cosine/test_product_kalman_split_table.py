#!/usr/bin/env python3
"""Tests for Product-Kalman split-table helper.

Run: `python3 test_product_kalman_split_table.py`.
"""

import csv
import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from product_kalman_split_table import (
    SPLIT_TABLE_MANIFEST_SCHEMA_VERSION,
    main,
    split_product_kalman_table,
)
from product_kalman_table_to_npz import read_product_kalman_table


def assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__} should have raised ValueError")


def read_rows(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_pair_table(path, delimiter=",", include_split=False):
    units = ["a", "b", "c", "d"]
    fields = ["id", "node", "root", "prior", "measurement", "target"]
    if include_split:
        fields.insert(0, "split")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for idx, (node, root) in enumerate(combinations(units, 2)):
            row = {
                "id": f"row-{idx}",
                "node": node,
                "root": root,
                "prior": f"{0.1 * idx:.17g}",
                "measurement": f"{0.2 + 0.1 * idx:.17g}",
                "target": f"{0.15 + 0.1 * idx:.17g}",
            }
            if include_split:
                row["split"] = "old"
            writer.writerow(row)


def write_single_unit_table(path, delimiter="\t"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "unit", "prior", "measurement", "target"],
            delimiter=delimiter,
        )
        writer.writeheader()
        for idx in range(10):
            writer.writerow({
                "id": f"row-{idx}",
                "unit": f"u{idx}",
                "prior": f"{0.05 * idx:.17g}",
                "measurement": f"{0.1 + 0.05 * idx:.17g}",
                "target": f"{0.08 + 0.05 * idx:.17g}",
            })


def split_unit_sets(rows):
    by_split = {"calibration": set(), "evaluation": set()}
    for row in rows:
        by_split[row["split"]].update([row["node"], row["root"]])
    return by_split


def test_split_table_writes_strict_endpoint_disjoint_output_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "pairs.csv"
        out = Path(tmp) / "pairs.split.csv"
        manifest = Path(tmp) / "pairs.split.manifest.json"
        write_pair_table(src)

        result = split_product_kalman_table(
            src,
            out,
            unit_cols=["node", "root"],
            output_manifest=manifest,
            evaluation_unit_frac=0.5,
            seed=3,
        )

        rows = read_rows(out)
        units = split_unit_sets(rows)
        assert units["calibration"].isdisjoint(units["evaluation"])
        assert len(rows) == 2
        assert result.manifest["schema_version"] == SPLIT_TABLE_MANIFEST_SCHEMA_VERSION
        assert result.manifest["split"]["calibration_rows"] == 1
        assert result.manifest["split"]["evaluation_rows"] == 1
        assert result.manifest["split"]["omitted_crossing_rows"] == 4
        assert result.manifest["split"]["disjoint_observed_units"] is True
        assert len(result.manifest["source_table"]["sha256"]) == 64
        assert len(result.manifest["output_table"]["sha256"]) == 64
        loaded = json.loads(manifest.read_text())
        assert loaded["split"]["unit_columns"] == ["node", "root"]

        arrays = read_product_kalman_table(
            out,
            prior_cols=["prior"],
            measurement_cols=["measurement"],
            target_cols=["target"],
        )
        assert arrays["calibration_prior_mean"].shape == (1, 1)
        assert arrays["evaluation_prior_mean"].shape == (1, 1)


def test_split_table_cli_tsv_roundtrip_without_crossing_rows():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "rows.tsv"
        out = Path(tmp) / "rows.split.tsv"
        manifest = Path(tmp) / "rows.split.manifest.json"
        write_single_unit_table(src)

        rc = main([
            str(src),
            "--output-table",
            str(out),
            "--output-manifest",
            str(manifest),
            "--unit-cols",
            "unit",
            "--evaluation-unit-frac",
            "0.3",
            "--seed",
            "7",
        ])

        assert rc == 0
        rows = read_rows(out, delimiter="\t")
        assert len(rows) == 10
        splits = {row["split"] for row in rows}
        assert splits == {"calibration", "evaluation"}
        data = json.loads(manifest.read_text())
        assert data["source_table"]["delimiter"] == "\t"
        assert data["split"]["omitted_crossing_rows"] == 0
        assert data["split"]["sampled_evaluation_unit_count"] == 3


def test_split_table_guards_fail_fast():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "pairs.csv"
        out = Path(tmp) / "pairs.split.csv"
        write_pair_table(src)

        assert_raises(
            split_product_kalman_table,
            src,
            out,
            unit_cols=["missing"],
        )
        assert_raises(
            split_product_kalman_table,
            src,
            out,
            unit_cols=["node", "root"],
            evaluation_unit_frac=1.0,
        )
        assert_raises(
            split_product_kalman_table,
            src,
            out,
            unit_cols=["node", "root"],
            evaluation_unit_frac=0.5,
            min_evaluation_rows=2,
        )

        with_split = Path(tmp) / "already_split.csv"
        write_pair_table(with_split, include_split=True)
        assert_raises(
            split_product_kalman_table,
            with_split,
            out,
            unit_cols=["node", "root"],
        )

        result = split_product_kalman_table(
            with_split,
            out,
            unit_cols=["node", "root"],
            evaluation_unit_frac=0.5,
            overwrite_split=True,
        )
        assert result.manifest["split"]["calibration_rows"] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman split-table tests passed")
