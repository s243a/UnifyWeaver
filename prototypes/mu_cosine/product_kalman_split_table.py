#!/usr/bin/env python3
"""Add leakage-aware calibration/evaluation splits to Product-Kalman CSV/TSV tables.

The Product-Kalman table evaluator intentionally consumes explicit row tables;
this helper handles the split hygiene just before that step. It samples held-out
unit values, writes evaluation rows only when all split-unit columns are held
out, writes calibration rows only when none are held out, and omits boundary
rows that would share a unit across the two splits.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path

try:
    from .product_kalman_table_to_npz import parse_column_list
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman_table_to_npz import parse_column_list


SPLIT_TABLE_MANIFEST_SCHEMA_VERSION = 1


__all__ = [
    "ProductKalmanSplitTable",
    "SPLIT_TABLE_MANIFEST_SCHEMA_VERSION",
    "split_product_kalman_table",
    "write_split_manifest",
]


@dataclass(frozen=True)
class ProductKalmanSplitTable:
    """One split-table materialization."""

    rows: tuple[dict, ...]
    fieldnames: tuple[str, ...]
    manifest: dict


def _infer_delimiter(path, explicit):
    if explicit is not None:
        return explicit
    lower = str(path).lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "\t"
    return ","


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path, delimiter):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        rows = []
        for row_num, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"row {row_num}: too many columns for header")
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return list(reader.fieldnames), rows


def _row_units(row, unit_cols, row_num):
    units = []
    for col in unit_cols:
        value = row.get(col)
        if value is None:
            raise ValueError(f"row {row_num}: missing split-unit column {col!r}")
        value = str(value).strip()
        if not value:
            raise ValueError(f"row {row_num}: empty split-unit value for column {col!r}")
        units.append(value)
    return frozenset(units)


def _sample_evaluation_units(units, frac, seed):
    if not 0.0 < frac < 1.0:
        raise ValueError("evaluation_unit_frac must be between 0 and 1")
    ordered = sorted(units)
    if len(ordered) < 2:
        raise ValueError("at least two unique split-unit values are required")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_eval = int(round(len(ordered) * frac))
    n_eval = max(1, min(len(ordered) - 1, n_eval))
    return frozenset(ordered[:n_eval])


def _split_rows(rows, unit_cols, evaluation_units, split_col, calibration_value, evaluation_value):
    out_rows = []
    cal_units = set()
    eval_units = set()
    omitted = 0
    for offset, row in enumerate(rows, start=2):
        units = _row_units(row, unit_cols, offset)
        if units <= evaluation_units:
            split = evaluation_value
            eval_units.update(units)
        elif units.isdisjoint(evaluation_units):
            split = calibration_value
            cal_units.update(units)
        else:
            omitted += 1
            continue
        out = dict(row)
        out[split_col] = split
        out_rows.append(out)
    return tuple(out_rows), frozenset(cal_units), frozenset(eval_units), omitted


def _write_table(path, fieldnames, rows, delimiter):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_split_manifest(path, manifest):
    """Write a JSON manifest for a split-table materialization."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def _build_manifest(
    input_table,
    output_table,
    delimiter,
    rows_in,
    rows_out,
    unit_cols,
    evaluation_units,
    calibration_units,
    observed_evaluation_units,
    omitted_crossing_rows,
    split_col,
    calibration_value,
    evaluation_value,
    evaluation_unit_frac,
    seed,
):
    overlap = calibration_units & observed_evaluation_units
    split_counts = {calibration_value: 0, evaluation_value: 0}
    for row in rows_out:
        split_counts[row[split_col]] = split_counts.get(row[split_col], 0) + 1
    return {
        "schema_version": SPLIT_TABLE_MANIFEST_SCHEMA_VERSION,
        "source_table": {
            "path": str(input_table),
            "sha256": _sha256_file(input_table),
            "delimiter": delimiter,
            "rows": len(rows_in),
        },
        "output_table": {
            "path": str(output_table),
            "sha256": _sha256_file(output_table),
            "rows": len(rows_out),
        },
        "split": {
            "seed": int(seed),
            "evaluation_unit_frac": float(evaluation_unit_frac),
            "split_column": split_col,
            "calibration_value": calibration_value,
            "evaluation_value": evaluation_value,
            "unit_columns": list(unit_cols),
            "sampled_evaluation_unit_count": len(evaluation_units),
            "sampled_evaluation_units": sorted(evaluation_units),
            "observed_calibration_unit_count": len(calibration_units),
            "observed_evaluation_unit_count": len(observed_evaluation_units),
            "observed_unit_overlap_count": len(overlap),
            "disjoint_observed_units": len(overlap) == 0,
            "calibration_rows": split_counts.get(calibration_value, 0),
            "evaluation_rows": split_counts.get(evaluation_value, 0),
            "omitted_crossing_rows": int(omitted_crossing_rows),
        },
    }


def split_product_kalman_table(
    input_table,
    output_table,
    unit_cols,
    output_manifest=None,
    split_col="split",
    calibration_value="calibration",
    evaluation_value="evaluation",
    evaluation_unit_frac=0.30,
    seed=0,
    delimiter=None,
    overwrite_split=False,
    min_calibration_rows=1,
    min_evaluation_rows=1,
):
    """Write a split-labeled table with disjoint split-unit values.

    Rows whose unit columns straddle sampled evaluation units and calibration
    units are omitted. This makes the emitted table stricter than a random row
    split: any unit value observed in calibration is absent from evaluation.
    """
    unit_cols = parse_column_list(unit_cols) if isinstance(unit_cols, str) else list(unit_cols)
    if not unit_cols:
        raise ValueError("unit_cols must contain at least one column")
    if calibration_value == evaluation_value:
        raise ValueError("calibration and evaluation split labels must differ")
    if min_calibration_rows < 1 or min_evaluation_rows < 1:
        raise ValueError("minimum row guards must be positive")

    delim = _infer_delimiter(input_table, delimiter)
    fieldnames, rows = _read_rows(input_table, delim)
    missing = [col for col in unit_cols if col not in fieldnames]
    if missing:
        raise ValueError(f"{input_table} missing split-unit columns: {', '.join(missing)}")
    if split_col in fieldnames and not overwrite_split:
        raise ValueError(f"{input_table} already has split column {split_col!r}; pass overwrite_split=True")
    out_fieldnames = list(fieldnames)
    if split_col not in out_fieldnames:
        out_fieldnames.insert(0, split_col)

    all_units = set()
    for offset, row in enumerate(rows, start=2):
        all_units.update(_row_units(row, unit_cols, offset))
    evaluation_units = _sample_evaluation_units(all_units, evaluation_unit_frac, seed)
    out_rows, calibration_units, observed_evaluation_units, omitted = _split_rows(
        rows,
        unit_cols,
        evaluation_units,
        split_col,
        calibration_value,
        evaluation_value,
    )
    split_counts = {calibration_value: 0, evaluation_value: 0}
    for row in out_rows:
        split_counts[row[split_col]] = split_counts.get(row[split_col], 0) + 1
    if split_counts.get(calibration_value, 0) < min_calibration_rows:
        raise ValueError("split produced too few calibration rows")
    if split_counts.get(evaluation_value, 0) < min_evaluation_rows:
        raise ValueError("split produced too few evaluation rows")
    if calibration_units & observed_evaluation_units:
        raise ValueError("internal error: emitted split units overlap")

    output_table = Path(output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    _write_table(output_table, out_fieldnames, out_rows, delim)
    manifest = _build_manifest(
        input_table,
        output_table,
        delim,
        rows,
        out_rows,
        unit_cols,
        evaluation_units,
        calibration_units,
        observed_evaluation_units,
        omitted,
        split_col,
        calibration_value,
        evaluation_value,
        evaluation_unit_frac,
        seed,
    )
    if output_manifest is not None:
        output_manifest = Path(output_manifest)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_split_manifest(output_manifest, manifest)
    return ProductKalmanSplitTable(rows=out_rows, fieldnames=tuple(out_fieldnames), manifest=manifest)


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Add disjoint calibration/evaluation split labels to a CSV/TSV table.")
    ap.add_argument("input_table", help="CSV/TSV table to split")
    ap.add_argument("--output-table", required=True, help="write split-labeled table here")
    ap.add_argument("--output-manifest", help="optional JSON manifest describing the split")
    ap.add_argument("--unit-cols", required=True, help="comma-separated columns whose values must not cross splits")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--calibration-value", default="calibration")
    ap.add_argument("--evaluation-value", default="evaluation")
    ap.add_argument("--evaluation-unit-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delimiter", help="input/output delimiter; defaults to tab for .tsv/.tab, comma otherwise")
    ap.add_argument("--overwrite-split", action="store_true")
    ap.add_argument("--min-calibration-rows", type=int, default=1)
    ap.add_argument("--min-evaluation-rows", type=int, default=1)
    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    try:
        split_product_kalman_table(
            args.input_table,
            args.output_table,
            unit_cols=parse_column_list(args.unit_cols),
            output_manifest=args.output_manifest,
            split_col=args.split_col,
            calibration_value=args.calibration_value,
            evaluation_value=args.evaluation_value,
            evaluation_unit_frac=args.evaluation_unit_frac,
            seed=args.seed,
            delimiter=args.delimiter,
            overwrite_split=args.overwrite_split,
            min_calibration_rows=args.min_calibration_rows,
            min_evaluation_rows=args.min_evaluation_rows,
        )
    except ValueError as exc:
        ap.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
