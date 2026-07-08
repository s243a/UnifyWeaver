#!/usr/bin/env python3
"""Build Product-Kalman holdout-evaluation NPZ inputs from CSV/TSV tables.

The evaluator consumes NPZ arrays with separate calibration/evaluation matrices.
This utility gives later corpus runs a reproducible bridge from a row table to
that NPZ format without hand-written Python snippets.
"""

import argparse
import csv
import sys

import numpy as np


__all__ = [
    "build_product_kalman_npz_from_table",
    "parse_column_list",
    "parse_matrix_literal",
    "read_product_kalman_table",
    "write_product_kalman_npz",
]


DEFAULT_SPLIT_VALUES = ("calibration", "evaluation")


def parse_column_list(text):
    """Parse comma-separated column names."""
    cols = [part.strip() for part in str(text).split(",") if part.strip()]
    if not cols:
        raise ValueError("column list must contain at least one column")
    if len(cols) != len(set(cols)):
        raise ValueError(f"column list contains duplicates: {text!r}")
    return cols


def parse_matrix_literal(text):
    """Parse a small semicolon/comma matrix literal such as `1,0;0,1`."""
    if text is None or str(text).strip() == "":
        return None
    rows = []
    for raw_row in str(text).split(";"):
        row = [part.strip() for part in raw_row.split(",") if part.strip()]
        if not row:
            raise ValueError(f"empty H row in {text!r}")
        try:
            rows.append([float(part) for part in row])
        except ValueError as exc:
            raise ValueError(f"H contains nonnumeric value in {text!r}") from exc
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"H rows must have equal length: {text!r}")
    return np.asarray(rows, dtype=float)


def _infer_delimiter(path, explicit):
    if explicit is not None:
        return explicit
    lower = str(path).lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "	"
    return ","


def _required_columns(split_col, id_col, prior_cols, measurement_cols, target_cols):
    cols = [split_col]
    if id_col:
        cols.append(id_col)
    cols.extend(prior_cols)
    cols.extend(measurement_cols)
    cols.extend(target_cols)
    return cols


def _row_vector(row, cols, row_num):
    values = []
    for col in cols:
        value = row.get(col)
        if value is None or str(value).strip() == "":
            raise ValueError(f"row {row_num}: missing numeric value for column {col!r}")
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"row {row_num}: nonnumeric value {value!r} for column {col!r}") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"row {row_num}: nonfinite value {value!r} for column {col!r}")
        values.append(parsed)
    return values


def _as_2d_rows(name, rows, width):
    if not rows:
        raise ValueError(f"{name} split has no rows")
    arr = np.asarray(rows, dtype=float)
    if arr.shape != (len(rows), width):
        raise ValueError(f"{name} array shape {arr.shape} must be ({len(rows)}, {width})")
    return arr


def _normalize_split_value(value):
    return str(value).strip().lower()


def read_product_kalman_table(
    path,
    prior_cols,
    measurement_cols,
    target_cols,
    split_col="split",
    id_col="id",
    calibration_value="calibration",
    evaluation_value="evaluation",
    delimiter=None,
    H=None,
):
    """Read a CSV/TSV table into evaluator-ready arrays.

    Required columns are: split, prior columns, measurement columns, target
    columns, plus an optional ID column. Split labels are compared
    case-insensitively after trimming whitespace.
    """
    prior_cols = parse_column_list(prior_cols) if isinstance(prior_cols, str) else list(prior_cols)
    measurement_cols = parse_column_list(measurement_cols) if isinstance(measurement_cols, str) else list(measurement_cols)
    target_cols = parse_column_list(target_cols) if isinstance(target_cols, str) else list(target_cols)
    calibration_value = _normalize_split_value(calibration_value)
    evaluation_value = _normalize_split_value(evaluation_value)
    if calibration_value == evaluation_value:
        raise ValueError("calibration and evaluation split labels must differ")
    H = parse_matrix_literal(H) if isinstance(H, str) else H

    split_rows = {
        calibration_value: {"prior": [], "measurement": [], "target": [], "ids": []},
        evaluation_value: {"prior": [], "measurement": [], "target": [], "ids": []},
    }
    delim = _infer_delimiter(path, delimiter)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        missing = [col for col in _required_columns(split_col, id_col, prior_cols, measurement_cols, target_cols)
                   if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        for row_num, row in enumerate(reader, start=2):
            label = _normalize_split_value(row.get(split_col, ""))
            if label not in split_rows:
                raise ValueError(
                    f"row {row_num}: split {row.get(split_col)!r} must be {calibration_value!r} or {evaluation_value!r}"
                )
            bucket = split_rows[label]
            bucket["prior"].append(_row_vector(row, prior_cols, row_num))
            bucket["measurement"].append(_row_vector(row, measurement_cols, row_num))
            bucket["target"].append(_row_vector(row, target_cols, row_num))
            if id_col:
                raw_id = row.get(id_col)
                if raw_id is None or str(raw_id).strip() == "":
                    raise ValueError(f"row {row_num}: missing ID value for column {id_col!r}")
                bucket["ids"].append(str(raw_id))

    cal = split_rows[calibration_value]
    ev = split_rows[evaluation_value]
    arrays = {
        "calibration_prior_mean": _as_2d_rows("calibration_prior_mean", cal["prior"], len(prior_cols)),
        "calibration_measurement": _as_2d_rows("calibration_measurement", cal["measurement"], len(measurement_cols)),
        "calibration_target_state": _as_2d_rows("calibration_target_state", cal["target"], len(target_cols)),
        "evaluation_prior_mean": _as_2d_rows("evaluation_prior_mean", ev["prior"], len(prior_cols)),
        "evaluation_measurement": _as_2d_rows("evaluation_measurement", ev["measurement"], len(measurement_cols)),
        "evaluation_target_state": _as_2d_rows("evaluation_target_state", ev["target"], len(target_cols)),
    }
    if id_col:
        arrays["calibration_ids"] = np.asarray(cal["ids"], dtype=str)
        arrays["evaluation_ids"] = np.asarray(ev["ids"], dtype=str)
    if H is not None:
        H = np.asarray(H, dtype=float)
        if H.shape != (len(measurement_cols), len(target_cols)):
            raise ValueError(f"H shape {H.shape} must be ({len(measurement_cols)}, {len(target_cols)})")
        if not np.isfinite(H).all():
            raise ValueError("H must be finite")
        arrays["H"] = H
    return arrays


def write_product_kalman_npz(path, arrays):
    """Write evaluator input arrays to NPZ."""
    np.savez(path, **arrays)


def build_product_kalman_npz_from_table(path, output_npz, **kwargs):
    """Read a table and write evaluator-ready NPZ arrays."""
    arrays = read_product_kalman_table(path, **kwargs)
    write_product_kalman_npz(output_npz, arrays)
    return arrays


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Build Product-Kalman evaluator NPZ input from a CSV/TSV table.")
    ap.add_argument("input_table", help="CSV/TSV table with split, prior, measurement, and target columns")
    ap.add_argument("--output-npz", required=True, help="write evaluator-ready NPZ here")
    ap.add_argument("--prior-cols", required=True, help="comma-separated prior mean columns")
    ap.add_argument("--measurement-cols", required=True, help="comma-separated measurement columns")
    ap.add_argument("--target-cols", required=True, help="comma-separated target-state columns")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--id-col", default="id", help="ID column to carry into NPZ; use '' to omit IDs")
    ap.add_argument("--calibration-value", default="calibration")
    ap.add_argument("--evaluation-value", default="evaluation")
    ap.add_argument("--delimiter", help="input delimiter; defaults to tab for .tsv/.tab, comma otherwise")
    ap.add_argument("--H", help="optional observation matrix literal, e.g. '1,0;0,1'")
    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    try:
        build_product_kalman_npz_from_table(
            args.input_table,
            args.output_npz,
            prior_cols=parse_column_list(args.prior_cols),
            measurement_cols=parse_column_list(args.measurement_cols),
            target_cols=parse_column_list(args.target_cols),
            split_col=args.split_col,
            id_col=args.id_col or None,
            calibration_value=args.calibration_value,
            evaluation_value=args.evaluation_value,
            delimiter=args.delimiter,
            H=args.H,
        )
    except ValueError as exc:
        ap.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
