#!/usr/bin/env python3
"""Build Product-Kalman holdout-evaluation NPZ inputs from CSV/TSV tables.

The evaluator consumes NPZ arrays with separate calibration/evaluation matrices.
This utility gives later corpus runs a reproducible bridge from a row table to
that NPZ format without hand-written Python snippets.
"""

import argparse
import csv
import hashlib
import json

import numpy as np


TABLE_INPUT_MANIFEST_SCHEMA_VERSION = 1


__all__ = [
    "TABLE_INPUT_MANIFEST_SCHEMA_VERSION",
    "build_product_kalman_input_manifest",
    "build_product_kalman_npz_from_table",
    "parse_column_list",
    "parse_matrix_literal",
    "read_product_kalman_table",
    "write_product_kalman_manifest",
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


def _normalize_columns(prior_cols, measurement_cols, target_cols):
    prior_cols = parse_column_list(prior_cols) if isinstance(prior_cols, str) else list(prior_cols)
    measurement_cols = parse_column_list(measurement_cols) if isinstance(measurement_cols, str) else list(measurement_cols)
    target_cols = parse_column_list(target_cols) if isinstance(target_cols, str) else list(target_cols)
    for name, cols in (("prior_cols", prior_cols), ("measurement_cols", measurement_cols), ("target_cols", target_cols)):
        if not cols:
            raise ValueError(f"{name} must contain at least one column")
        if len(cols) != len(set(cols)):
            raise ValueError(f"{name} contains duplicates: {cols!r}")
    return prior_cols, measurement_cols, target_cols


def _validate_schema_dimensions(prior_cols, measurement_cols, target_cols, H):
    state_dim = len(target_cols)
    if len(prior_cols) != state_dim:
        raise ValueError("prior column count must match target-state column count")
    if H is None and len(measurement_cols) != state_dim:
        raise ValueError("H is required when measurement column count differs from target-state column count")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_manifest(arr):
    arr = np.asarray(arr)
    return {"shape": [int(v) for v in arr.shape], "dtype": str(arr.dtype)}


def _duplicate_count(values):
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return len(duplicates)


def _id_manifest(arrays, id_col):
    if not id_col or "calibration_ids" not in arrays or "evaluation_ids" not in arrays:
        return {"present": False, "column": id_col}
    calibration_ids = [str(v) for v in arrays["calibration_ids"].tolist()]
    evaluation_ids = [str(v) for v in arrays["evaluation_ids"].tolist()]
    overlap = set(calibration_ids) & set(evaluation_ids)
    cal_dupes = _duplicate_count(calibration_ids)
    eval_dupes = _duplicate_count(evaluation_ids)
    return {
        "present": True,
        "column": id_col,
        "calibration_count": len(calibration_ids),
        "evaluation_count": len(evaluation_ids),
        "calibration_duplicate_count": cal_dupes,
        "evaluation_duplicate_count": eval_dupes,
        "overlap_count": len(overlap),
        "disjoint_and_unique": len(overlap) == 0 and cal_dupes == 0 and eval_dupes == 0,
    }


def build_product_kalman_input_manifest(
    path,
    arrays,
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
    """Build a JSON-serializable manifest for a table-derived evaluator NPZ."""
    prior_cols, measurement_cols, target_cols = _normalize_columns(prior_cols, measurement_cols, target_cols)
    delim = _infer_delimiter(path, delimiter)
    cal_rows = int(arrays["calibration_target_state"].shape[0])
    eval_rows = int(arrays["evaluation_target_state"].shape[0])
    H = arrays.get("H")
    manifest = {
        "schema_version": TABLE_INPUT_MANIFEST_SCHEMA_VERSION,
        "source_table": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "delimiter": delim,
        },
        "splits": {
            "column": split_col,
            "calibration_value": calibration_value,
            "evaluation_value": evaluation_value,
            "calibration_rows": cal_rows,
            "evaluation_rows": eval_rows,
        },
        "columns": {
            "prior": list(prior_cols),
            "measurement": list(measurement_cols),
            "target_state": list(target_cols),
        },
        "dimensions": {
            "state_dim": len(target_cols),
            "prior_dim": len(prior_cols),
            "observation_dim": len(measurement_cols),
        },
        "ids": _id_manifest(arrays, id_col),
        "arrays": {name: _array_manifest(value) for name, value in sorted(arrays.items())},
        "H": {
            "present": H is not None,
            "shape": None if H is None else [int(v) for v in np.asarray(H).shape],
            "values": None if H is None else np.asarray(H, dtype=float).tolist(),
        },
    }
    return manifest


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
    prior_cols, measurement_cols, target_cols = _normalize_columns(prior_cols, measurement_cols, target_cols)
    calibration_value = _normalize_split_value(calibration_value)
    evaluation_value = _normalize_split_value(evaluation_value)
    if calibration_value == evaluation_value:
        raise ValueError("calibration and evaluation split labels must differ")
    H = parse_matrix_literal(H) if isinstance(H, str) else H
    _validate_schema_dimensions(prior_cols, measurement_cols, target_cols, H)

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


def write_product_kalman_manifest(path, manifest):
    """Write a JSON manifest for a table-derived Product-Kalman NPZ."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def build_product_kalman_npz_from_table(path, output_npz, output_manifest=None, **kwargs):
    """Read a table and write evaluator-ready NPZ arrays."""
    arrays = read_product_kalman_table(path, **kwargs)
    write_product_kalman_npz(output_npz, arrays)
    if output_manifest is not None:
        manifest = build_product_kalman_input_manifest(path, arrays, **kwargs)
        write_product_kalman_manifest(output_manifest, manifest)
    return arrays


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Build Product-Kalman evaluator NPZ input from a CSV/TSV table.")
    ap.add_argument("input_table", help="CSV/TSV table with split, prior, measurement, and target columns")
    ap.add_argument("--output-npz", required=True, help="write evaluator-ready NPZ here")
    ap.add_argument("--output-manifest", help="optional JSON manifest describing the input table and NPZ schema")
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
            output_manifest=args.output_manifest,
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
