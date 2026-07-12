#!/usr/bin/env python3
"""Mechanism power audit for the adjacent component-multiplier statistic.

This runner uses known zero mean and identity within-row covariance.  It does
not emulate the end-to-end calibration/KRR/B fitting pipeline and therefore
cannot by itself unlock the real-data pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from adjacent_residual_pilot import component_multiplier_stability


COMPONENT_COUNTS = (28, 40)
COUPLINGS = (0.0, 0.04, 0.10, 0.20)


def _file_record(path):
    path = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": path, "size_bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def simulate_component_contrasts(components, coupling, seed):
    """Generate one anchor-matched 4x4 contrast per independent component."""
    if components < 2:
        raise ValueError("components must be at least two")
    if not 0.0 <= coupling < 1.0:
        raise ValueError("coupling must be in [0,1)")
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(components):
        anchor = rng.standard_normal(4)
        positive_partner = (
            coupling * anchor
            + np.sqrt(1.0 - coupling * coupling) * rng.standard_normal(4)
        )
        first_control = rng.standard_normal(4)
        second_control = rng.standard_normal(4)
        positive = 0.5 * (
            np.outer(anchor, positive_partner)
            + np.outer(positive_partner, anchor)
        )
        controls = 0.25 * (
            np.outer(anchor, first_control)
            + np.outer(first_control, anchor)
            + np.outer(positive_partner, second_control)
            + np.outer(second_control, positive_partner)
        )
        output.append(positive - controls)
    return np.asarray(output)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def run_scenario(components, coupling, args):
    records = []
    for replicate in range(args.replicates):
        seed = args.seed + 100000 * components + 10000 * int(round(coupling * 100)) + replicate
        matrices = simulate_component_contrasts(components, coupling, seed)
        band = component_multiplier_stability(
            matrices,
            draws=args.multiplier_draws,
            seed=seed + 500000,
            confidence=args.confidence,
        )
        primary = float(band.estimate[0])
        pointwise_low = float(band.pointwise_low[0])
        simultaneous_low = float(band.simultaneous_low[0])
        true_matrix = np.eye(4) * coupling
        error = float(np.max(np.abs(np.linalg.eigvalsh(
            np.mean(matrices, axis=0) - true_matrix
        ))))
        records.append({
            "primary_trace_over_4": primary,
            "pointwise_lower_positive": bool(pointwise_low > 0.0),
            "simultaneous_lower_positive": bool(simultaneous_low > 0.0),
            "pointwise_contains_truth": bool(
                band.pointwise_low[0] <= coupling <= band.pointwise_high[0]
            ),
            "spectral_radius_contains_truth": bool(error <= band.spectral_error_radius),
        })
    return {
        "components": int(components),
        "coupling": float(coupling),
        "replicates": int(args.replicates),
        "primary_trace_over_4": _summary([
            row["primary_trace_over_4"] for row in records
        ]),
        "pointwise_positive_rate": float(np.mean([
            row["pointwise_lower_positive"] for row in records
        ])),
        "simultaneous_positive_rate": float(np.mean([
            row["simultaneous_lower_positive"] for row in records
        ])),
        "pointwise_truth_inclusion_rate": float(np.mean([
            row["pointwise_contains_truth"] for row in records
        ])),
        "spectral_truth_inclusion_rate": float(np.mean([
            row["spectral_radius_contains_truth"] for row in records
        ])),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--multiplier-draws", type=int, default=999)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=731000)
    parser.add_argument("--component-counts", type=int, nargs="+", default=list(COMPONENT_COUNTS))
    parser.add_argument("--couplings", type=float, nargs="+", default=list(COUPLINGS))
    parser.add_argument("--out", default="/tmp/adjacent_residual_synthetic.json")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.replicates < 1 or args.multiplier_draws < 1:
        raise ValueError("replicates and multiplier-draws must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if any(value < 2 for value in args.component_counts):
        raise ValueError("every component count must be at least two")
    if any(not 0.0 <= value < 1.0 for value in args.couplings) or 0.0 not in args.couplings:
        raise ValueError("couplings must be in [0,1) and include the block null 0.0")
    started = time.perf_counter()
    scenarios = [
        run_scenario(components, coupling, args)
        for components in args.component_counts
        for coupling in args.couplings
    ]
    by_count = {}
    for components in args.component_counts:
        rows = {row["coupling"]: row for row in scenarios if row["components"] == components}
        powered_couplings = [value for value in args.couplings if value >= 0.10]
        by_count[str(components)] = {
            "block_null_pointwise_false_positive_at_most_10pct": bool(
                rows[0.0]["pointwise_positive_rate"] <= 0.10
            ),
            "pointwise_detection_at_least_80pct_for_all_requested_couplings_ge_0_10": bool(
                powered_couplings
                and all(rows[value]["pointwise_positive_rate"] >= 0.80 for value in powered_couplings)
            ),
        }
    payload = {
        "schema_version": 1,
        "status": "KNOWN-MEAN/KNOWN-B MECHANISM AUDIT; END-TO-END POWER NOT IMPLEMENTED",
        "mechanism_scope": (
            "one equal-weight contrast with two anchor controls per independent component; "
            "no unequal component sizes or fitted calibration/KRR/B nuisance"
        ),
        "design": "DESIGN_adjacent_residual_pilot.md",
        "implementation": {
            "design": _file_record(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "DESIGN_adjacent_residual_pilot.md"
            )),
            "core": _file_record(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "adjacent_residual_pilot.py"
            )),
            "runner": _file_record(os.path.abspath(__file__)),
        },
        "configuration": vars(args),
        "component_counts": list(args.component_counts),
        "couplings": list(args.couplings),
        "scenarios": scenarios,
        "mechanism_gates": by_count,
        "real_data_gate_unlocked": False,
        "reason": (
            "full-procedure smooth-mean/null calibration and repeated-judge identification are absent"
        ),
    }
    path = os.path.abspath(args.out)
    temporary = path + ".tmp"
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    os.replace(temporary, path)
    print(json.dumps({
        "output": path,
        "wall_seconds": time.perf_counter() - started,
        "mechanism_gates": by_count,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
