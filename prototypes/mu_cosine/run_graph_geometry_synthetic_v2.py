#!/usr/bin/env python3
"""Matched-coupling v2 mechanism audit for graph residual geometries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import time

import numpy as np

from graph_geometry import kernel_diagnostics
from run_graph_geometry_synthetic import (
    _content_record,
    _summary,
    calibrate_familywise_threshold,
    candidate_kernels,
    draw_fields,
    mean_nll_per_scalar,
    prepare_gaussian,
    select_with_threshold,
)


RHO_GRID = (0.0, 0.025, 0.05, 0.10, 0.20)
TRUTH_RHOS = (0.10, 0.20)
EQUIVALENCE_CLASS = {
    "closed": "local_spectral",
    "heat": "local_spectral",
    "resolvent": "local_spectral",
    "walk_decay": "walk",
    "deranged_walk": "deranged",
}


def maximum_off_diagonal(kernel):
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or len(kernel) < 2:
        raise ValueError("kernel must be square with at least two rows")
    off = kernel - np.diag(np.diag(kernel))
    maximum = float(np.max(np.abs(off)))
    if maximum <= 0.0:
        raise ValueError("kernel must contain nonzero off-diagonal geometry")
    return maximum


def matched_correlation(kernel, rho):
    rho = float(rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0,1)")
    amplitude = rho / maximum_off_diagonal(kernel) if rho else 0.0
    if amplitude >= 0.95:
        raise ValueError(f"matched amplitude {amplitude:.3f} is outside the frozen <0.95 domain")
    covariance = (1.0 - amplitude) * np.eye(len(kernel)) + amplitude * kernel
    return covariance, amplitude


def prepare_matched_candidates(kernels, rho_grid=RHO_GRID):
    dimension = len(next(iter(kernels.values())))
    prepared = {("block", 0.0): prepare_gaussian(np.eye(dimension))}
    amplitudes = {}
    for family, kernel in kernels.items():
        for rho in rho_grid:
            if rho == 0.0:
                continue
            covariance, amplitude = matched_correlation(kernel, rho)
            prepared[(family, float(rho))] = prepare_gaussian(covariance)
            amplitudes[f"{family}@{rho:g}"] = amplitude
    return prepared, amplitudes


def _best_family_key(scores, family):
    eligible = [key for key in scores if key[0] == family]
    return min(eligible, key=lambda key: (scores[key], key[1]))


def _deranged(kernel):
    permutation = np.roll(np.arange(len(kernel)), 5)
    return kernel[np.ix_(permutation, permutation)]


def _diagnostic_family_prepared(kernel, label):
    return {
        (label, rho): prepare_gaussian(matched_correlation(kernel, rho)[0])
        for rho in RHO_GRID
        if rho > 0.0
    }


def run_scenario(family, truth_kernel, truth_rho, prepared, threshold, args, seed):
    truth_covariance = (
        np.eye(len(truth_kernel))
        if family == "block"
        else matched_correlation(truth_kernel, truth_rho)[0]
    )
    if family == "block":
        comparison_family = None
        comparison_prepared = {}
    elif family == "deranged_walk":
        comparison_family = "walk_decay"
        comparison_prepared = {}
    else:
        comparison_family = "equal_energy_derangement"
        comparison_prepared = _diagnostic_family_prepared(
            _deranged(truth_kernel), comparison_family
        )
    rng = np.random.default_rng(seed)
    records = []
    sensitivity = {str(rho): [] for rho in RHO_GRID}
    for _ in range(args.replicates):
        train = draw_fields(truth_covariance, args.train_fields, rng)
        held = draw_fields(truth_covariance, args.held_fields, rng)
        selected, train_scores = select_with_threshold(train, prepared, threshold)
        held_scores = {
            key: mean_nll_per_scalar(held, candidate) for key, candidate in prepared.items()
        }
        block_nll = held_scores[("block", 0.0)]
        if family == "block":
            comparison_win = selected == ("block", 0.0)
        else:
            truth_key = _best_family_key(train_scores, family)
            if family == "deranged_walk":
                comparison_key = _best_family_key(train_scores, comparison_family)
                comparison_nll = held_scores[comparison_key]
            else:
                comparison_train_scores = {
                    key: mean_nll_per_scalar(train, candidate)
                    for key, candidate in comparison_prepared.items()
                }
                comparison_key = min(
                    comparison_train_scores,
                    key=lambda key: (comparison_train_scores[key], key[1]),
                )
                comparison_nll = mean_nll_per_scalar(
                    held, comparison_prepared[comparison_key]
                )
            comparison_win = held_scores[truth_key] < comparison_nll
            for rho in RHO_GRID:
                candidate = (
                    prepared[("block", 0.0)]
                    if rho == 0.0
                    else prepare_gaussian(matched_correlation(truth_kernel, rho)[0])
                )
                sensitivity[str(rho)].append(
                    block_nll - mean_nll_per_scalar(held, candidate)
                )
        records.append({
            "selected": selected,
            "selected_gain": block_nll - held_scores[selected],
            "exact_family_selected": selected[0] == family,
            "equivalence_class_selected": (
                family == "block" and selected[0] == "block"
            ) or (
                family != "block"
                and EQUIVALENCE_CLASS.get(selected[0]) == EQUIVALENCE_CLASS[family]
            ),
            "truth_beats_deranged_or_base": comparison_win,
        })
    return {
        "truth_family": family,
        "truth_equivalence_class": "block" if family == "block" else EQUIVALENCE_CLASS[family],
        "truth_maximum_coupling": float(truth_rho),
        "truth_path_amplitude": (
            0.0 if family == "block" else matched_correlation(truth_kernel, truth_rho)[1]
        ),
        "replicates": int(args.replicates),
        "nonzero_selection_rate": float(np.mean([row["selected"][0] != "block" for row in records])),
        "exact_family_selection_rate": float(np.mean([
            row["exact_family_selected"] for row in records
        ])),
        "equivalence_class_selection_rate": float(np.mean([
            row["equivalence_class_selected"] for row in records
        ])),
        "truth_beats_deranged_or_base_rate": float(np.mean([
            row["truth_beats_deranged_or_base"] for row in records
        ])),
        "selected_held_nll_gain_per_scalar": _summary([
            row["selected_gain"] for row in records
        ]),
        "selected_counts": {
            f"{selected_family}@rho={rho:g}": int(sum(
                row["selected"] == (selected_family, rho) for row in records
            ))
            for selected_family, rho in sorted(set(row["selected"] for row in records))
        },
        "truth_geometry_rho_sensitivity": {
            rho: _summary(values) for rho, values in sensitivity.items() if values
        },
    }


def run_benchmark(args):
    nodes, kernels = candidate_kernels()
    prepared, amplitudes = prepare_matched_candidates(kernels)
    threshold, null_maxima = calibrate_familywise_threshold(
        prepared,
        train_fields=args.train_fields,
        draws=args.calibration_draws,
        seed=args.seed + 1,
        confidence=args.confidence,
    )
    scenarios = [run_scenario(
        "block", np.eye(len(nodes)), 0.0, prepared, threshold, args, args.seed + 100
    )]
    for family, kernel in kernels.items():
        for offset, rho in enumerate(TRUTH_RHOS):
            scenarios.append(run_scenario(
                family,
                kernel,
                rho,
                prepared,
                threshold,
                args,
                args.seed + 1000 * (list(kernels).index(family) + 1) + offset,
            ))
    block = scenarios[0]
    strong = [
        row for row in scenarios
        if row["truth_maximum_coupling"] == 0.20 and row["truth_family"] != "deranged_walk"
    ]
    gates = {
        "block_null_nonzero_selection_at_most_10pct": block["nonzero_selection_rate"] <= 0.10,
        "rho_0_20_nonderanged_selected_gain_positive": all(
            row["selected_held_nll_gain_per_scalar"]["mean"] > 0.0 for row in strong
        ),
        "rho_0_20_nonderanged_equivalence_selection_at_least_60pct": all(
            row["equivalence_class_selection_rate"] >= 0.60 for row in strong
        ),
        "rho_0_20_nonderanged_beats_derangement_at_least_80pct": all(
            row["truth_beats_deranged_or_base_rate"] >= 0.80 for row in strong
        ),
    }
    upper = np.triu_indices(len(nodes), 1)
    overlap = {
        left: {
            right: float(np.corrcoef(kernels[left][upper], kernels[right][upper])[0, 1])
            for right in kernels
        }
        for left in kernels
    }
    root = os.path.dirname(os.path.abspath(__file__))
    configuration = vars(args).copy()
    configuration.pop("out", None)
    return {
        "schema_version": 1,
        "status": "MATCHED-COUPLING GRAPH-GEOMETRY V2 MECHANISM AUDIT; NO REAL DEPLOYMENT",
        "design": "DESIGN_graph_geometry_synthetic_v2.md",
        "implementation": {
            "design": _content_record(os.path.join(root, "DESIGN_graph_geometry_synthetic_v2.md")),
            "core": _content_record(os.path.join(root, "graph_geometry.py")),
            "v1_dependency": _content_record(os.path.join(root, "run_graph_geometry_synthetic.py")),
            "runner": _content_record(os.path.abspath(__file__)),
        },
        "configuration": configuration,
        "maximum_coupling_grid": list(RHO_GRID),
        "family_specific_path_amplitudes": amplitudes,
        "equivalence_classes": EQUIVALENCE_CLASS,
        "graph": {"nodes": list(nodes), "candidate_overlap": overlap},
        "kernel_diagnostics": {
            name: asdict(kernel_diagnostics(kernel)) for name, kernel in kernels.items()
        },
        "familywise_null": {
            "threshold_nll_gain_per_scalar": threshold,
            "calibration_maximum_gain": _summary(null_maxima),
        },
        "scenarios": scenarios,
        "mechanism_gates": gates,
        "all_mechanism_gates_pass": all(gates.values()),
        "real_covariance_gate_unlocked": False,
        "batching_gate_unlocked": False,
        "qr_deployment_unlocked": False,
        "reason": (
            "matched synthetic coupling with known mean/B cannot identify real repeated-judge covariance"
        ),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--calibration-draws", type=int, default=1000)
    parser.add_argument("--train-fields", type=int, default=48)
    parser.add_argument("--held-fields", type=int, default=64)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=885200)
    parser.add_argument("--out", default="/tmp/graph_geometry_synthetic_v2.json")
    return parser


def _validate_args(args):
    if min(args.replicates, args.calibration_draws, args.train_fields, args.held_fields) < 1:
        raise ValueError("replicate, calibration, and field counts must be positive")
    if args.calibration_draws < 10 or not 0.5 < args.confidence < 1.0:
        raise ValueError("calibration-draws must be at least 10 and confidence in (0.5,1)")


def main():
    args = build_arg_parser().parse_args()
    _validate_args(args)
    started = time.perf_counter()
    payload = run_benchmark(args)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = os.path.abspath(args.out)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    os.replace(temporary, path)
    print(json.dumps({
        "all_mechanism_gates_pass": payload["all_mechanism_gates_pass"],
        "mechanism_gates": payload["mechanism_gates"],
        "output": path,
        "wall_seconds": time.perf_counter() - started,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
