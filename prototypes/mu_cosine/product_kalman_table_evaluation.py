#!/usr/bin/env python3
"""Run Product-Kalman holdout evaluation directly from a CSV/TSV row table.

This is a thin orchestration layer over `product_kalman_table_to_npz.py` and
`product_kalman_evaluation.py`. It keeps real-corpus runs reproducible by
emitting the input NPZ, optional input manifest, JSON score summary, and optional
row-level evaluation artifacts from one command.
"""

import argparse
from dataclasses import dataclass
import json
import sys

try:
    from .product_kalman_evaluation import (
        evaluation_to_json_dict,
        run_product_kalman_holdout_npz,
        write_evaluation_npz,
    )
    from .product_kalman_table_to_npz import build_product_kalman_npz_from_table, parse_column_list
except ImportError:  # direct script execution from prototypes/mu_cosine
    from product_kalman_evaluation import (
        evaluation_to_json_dict,
        run_product_kalman_holdout_npz,
        write_evaluation_npz,
    )
    from product_kalman_table_to_npz import build_product_kalman_npz_from_table, parse_column_list


__all__ = [
    "ProductKalmanTableEvaluation",
    "run_product_kalman_table_evaluation",
]


@dataclass(frozen=True)
class ProductKalmanTableEvaluation:
    """One table-to-NPZ-to-score run."""

    input_arrays: dict
    result: object
    summary: dict


def _write_json(path, data, indent=2):
    text = json.dumps(data, indent=None if indent == 0 else indent, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _attach_input_paths(summary, input_table, input_npz, input_manifest=None, output_npz=None):
    enriched = dict(summary)
    enriched["inputs"] = {
        "source_table": str(input_table),
        "input_npz": str(input_npz),
        "input_manifest": None if input_manifest is None else str(input_manifest),
        "evaluation_npz": None if output_npz is None else str(output_npz),
    }
    return enriched


def run_product_kalman_table_evaluation(
    input_table,
    input_npz,
    prior_cols,
    measurement_cols,
    target_cols,
    input_manifest=None,
    output_json=None,
    output_npz=None,
    split_col="split",
    id_col="id",
    calibration_value="calibration",
    evaluation_value="evaluation",
    delimiter=None,
    H=None,
    shrinkage=0.0,
    jitter=1e-9,
    ddof=1,
    shrinkage_target="diagonal",
    indent=2,
):
    """Build evaluator input artifacts from a table and score the held-out split."""
    arrays = build_product_kalman_npz_from_table(
        input_table,
        input_npz,
        output_manifest=input_manifest,
        prior_cols=prior_cols,
        measurement_cols=measurement_cols,
        target_cols=target_cols,
        split_col=split_col,
        id_col=id_col,
        calibration_value=calibration_value,
        evaluation_value=evaluation_value,
        delimiter=delimiter,
        H=H,
    )
    result = run_product_kalman_holdout_npz(
        input_npz,
        shrinkage=shrinkage,
        jitter=jitter,
        ddof=ddof,
        shrinkage_target=shrinkage_target,
    )
    if output_npz:
        write_evaluation_npz(output_npz, result)
    summary = _attach_input_paths(
        evaluation_to_json_dict(result),
        input_table,
        input_npz,
        input_manifest=input_manifest,
        output_npz=output_npz,
    )
    if output_json:
        _write_json(output_json, summary, indent=indent)
    return ProductKalmanTableEvaluation(
        input_arrays=arrays,
        result=result,
        summary=summary,
    )


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Build Product-Kalman input artifacts from a table and run the holdout evaluator.",
    )
    ap.add_argument("input_table", help="CSV/TSV table with split, prior, measurement, and target columns")
    ap.add_argument("--input-npz", required=True, help="write intermediate evaluator input NPZ here")
    ap.add_argument("--input-manifest", help="write optional input-provenance JSON manifest here")
    ap.add_argument("--output-json", help="write evaluation JSON summary here instead of stdout")
    ap.add_argument("--output-npz", help="write row-level prediction/covariance artifacts to this NPZ path")
    ap.add_argument("--prior-cols", required=True, help="comma-separated prior mean columns")
    ap.add_argument("--measurement-cols", required=True, help="comma-separated measurement columns")
    ap.add_argument("--target-cols", required=True, help="comma-separated target-state columns")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--id-col", default="id", help="ID column to carry into NPZ; use '' to omit IDs")
    ap.add_argument("--calibration-value", default="calibration")
    ap.add_argument("--evaluation-value", default="evaluation")
    ap.add_argument("--delimiter", help="input delimiter; defaults to tab for .tsv/.tab, comma otherwise")
    ap.add_argument("--H", help="optional observation matrix literal, e.g. '1,0;0,1'")
    ap.add_argument("--shrinkage", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=1e-9)
    ap.add_argument("--ddof", type=int, default=1)
    ap.add_argument("--shrinkage-target", default="diagonal", choices=("diagonal", "scaled_identity"))
    ap.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact output")
    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    try:
        run = run_product_kalman_table_evaluation(
            args.input_table,
            args.input_npz,
            prior_cols=parse_column_list(args.prior_cols),
            measurement_cols=parse_column_list(args.measurement_cols),
            target_cols=parse_column_list(args.target_cols),
            input_manifest=args.input_manifest,
            output_json=args.output_json,
            output_npz=args.output_npz,
            split_col=args.split_col,
            id_col=args.id_col or None,
            calibration_value=args.calibration_value,
            evaluation_value=args.evaluation_value,
            delimiter=args.delimiter,
            H=args.H,
            shrinkage=args.shrinkage,
            jitter=args.jitter,
            ddof=args.ddof,
            shrinkage_target=args.shrinkage_target,
            indent=args.indent,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if not args.output_json:
        indent = None if args.indent == 0 else args.indent
        sys.stdout.write(json.dumps(run.summary, indent=indent, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
