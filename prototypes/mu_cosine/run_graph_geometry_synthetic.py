#!/usr/bin/env python3
"""Mechanism audit for PSD graph geometries with repeated residual fields.

This is a known-zero-mean, known-unit-channel-covariance audit.  It calibrates
the complete candidate/alpha search under a block-null field, selects on train
fields, and scores independent held fields.  It cannot unlock real covariance
or QR deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import time

import numpy as np

from graph_geometry import (
    closed_neighborhood_kernel,
    heat_kernel_reference,
    kernel_diagnostics,
    resolvent_kernel_reference,
    walk_feature_kernel,
)


ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)
TRUTH_ALPHAS = (0.10, 0.20)


def _content_record(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"size_bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def benchmark_graph():
    """Fixed asymmetric-branch topology with local and longer-range ambiguity."""
    nodes = tuple(f"n{index}" for index in range(12))
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (2, 6), (6, 7), (4, 8), (8, 9),
        (1, 10), (10, 3), (5, 11),
    )
    neighbors = {node: set() for node in nodes}
    for left, right in edges:
        a, b = nodes[left], nodes[right]
        neighbors[a].add(b)
        neighbors[b].add(a)
    return nodes, {node: frozenset(values) for node, values in neighbors.items()}


def candidate_kernels():
    nodes, neighbors = benchmark_graph()
    _, closed, _ = closed_neighborhood_kernel(nodes, neighbors)
    _, walk, _ = walk_feature_kernel(nodes, neighbors, (1.0, 0.5, 0.25, 0.125))
    _, heat = heat_kernel_reference(nodes, neighbors, diffusion_time=1.0)
    _, resolvent = resolvent_kernel_reference(nodes, neighbors, scale=1.0)
    permutation = np.roll(np.arange(len(nodes)), 5)
    deranged_walk = walk[np.ix_(permutation, permutation)]
    return nodes, {
        "closed": closed,
        "walk_decay": walk,
        "heat": heat,
        "resolvent": resolvent,
        "deranged_walk": deranged_walk,
    }


def correlation_path(kernel, alpha):
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0,1)")
    kernel = np.asarray(kernel, dtype=float)
    return (1.0 - alpha) * np.eye(len(kernel)) + alpha * kernel


@dataclass(frozen=True)
class PreparedGaussian:
    precision: np.ndarray
    log_determinant: float
    dimension: int


def prepare_gaussian(covariance):
    covariance = np.asarray(covariance, dtype=float)
    sign, logdet = np.linalg.slogdet(covariance)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or sign <= 0:
        raise ValueError("covariance must be positive definite")
    return PreparedGaussian(np.linalg.inv(covariance), float(logdet), len(covariance))


def mean_nll_per_scalar(fields, prepared):
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 2 or fields.shape[1] != prepared.dimension:
        raise ValueError("fields must be [draw, dimension]")
    quadratic = np.einsum("bi,ij,bj->b", fields, prepared.precision, fields)
    return float(np.mean(
        0.5 * (quadratic + prepared.log_determinant + prepared.dimension * np.log(2.0 * np.pi))
        / prepared.dimension
    ))


def draw_fields(covariance, count, rng):
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) <= 0.0 or count < 1:
        raise ValueError("covariance must be positive definite and count positive")
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    return rng.standard_normal((count, len(covariance))) @ root.T


def prepare_candidates(kernels, alphas=ALPHAS):
    output = {("block", 0.0): prepare_gaussian(np.eye(len(next(iter(kernels.values())))))}
    for name, kernel in kernels.items():
        for alpha in alphas:
            if alpha > 0.0:
                output[(name, float(alpha))] = prepare_gaussian(correlation_path(kernel, alpha))
    return output


def best_candidate(fields, prepared):
    scores = {key: mean_nll_per_scalar(fields, value) for key, value in prepared.items()}
    selected = min(scores, key=lambda key: (scores[key], key[0], key[1]))
    block = scores[("block", 0.0)]
    return selected, float(block - scores[selected]), scores


def calibrate_familywise_threshold(prepared, *, train_fields, draws, seed, confidence=0.95):
    if draws < 10 or not 0.5 < confidence < 1.0:
        raise ValueError("draws must be at least 10 and confidence in (0.5,1)")
    rng = np.random.default_rng(seed)
    maxima = []
    identity = np.eye(next(iter(prepared.values())).dimension)
    for _ in range(draws):
        fields = draw_fields(identity, train_fields, rng)
        _selected, gain, _scores = best_candidate(fields, prepared)
        maxima.append(gain)
    return float(np.quantile(maxima, confidence)), np.asarray(maxima)


def select_with_threshold(fields, prepared, threshold):
    selected, gain, scores = best_candidate(fields, prepared)
    if gain <= threshold:
        selected = ("block", 0.0)
    return selected, scores


def _best_family_key(scores, family):
    eligible = [key for key in scores if key[0] == family]
    if not eligible:
        raise ValueError(f"unknown candidate family: {family}")
    return min(eligible, key=lambda key: (scores[key], key[1]))


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def run_scenario(name, truth_kernel, truth_alpha, kernels, prepared, threshold, args, seed):
    truth_covariance = (
        np.eye(len(truth_kernel))
        if name == "block"
        else correlation_path(truth_kernel, truth_alpha)
    )
    rng = np.random.default_rng(seed)
    records = []
    sensitivity = {str(alpha): [] for alpha in ALPHAS}
    comparison_family = "walk_decay" if name == "deranged_walk" else "deranged_walk"
    for _ in range(args.replicates):
        train = draw_fields(truth_covariance, args.train_fields, rng)
        held = draw_fields(truth_covariance, args.held_fields, rng)
        selected, train_scores = select_with_threshold(train, prepared, threshold)
        held_scores = {
            key: mean_nll_per_scalar(held, candidate) for key, candidate in prepared.items()
        }
        block_nll = held_scores[("block", 0.0)]
        selected_gain = block_nll - held_scores[selected]
        if name == "block":
            correct_beats_comparison = selected == ("block", 0.0)
        else:
            correct_key = _best_family_key(train_scores, name)
            comparison_key = _best_family_key(train_scores, comparison_family)
            correct_beats_comparison = held_scores[correct_key] < held_scores[comparison_key]
            for alpha in ALPHAS:
                candidate = (
                    prepared[("block", 0.0)]
                    if alpha == 0.0
                    else prepare_gaussian(correlation_path(truth_kernel, alpha))
                )
                sensitivity[str(alpha)].append(
                    block_nll - mean_nll_per_scalar(held, candidate)
                )
        records.append({
            "selected": selected,
            "selected_gain": selected_gain,
            "correct_family_selected": selected[0] == name,
            "correct_beats_comparison": correct_beats_comparison,
        })
    return {
        "truth_family": name,
        "truth_alpha": float(truth_alpha),
        "replicates": int(args.replicates),
        "nonzero_selection_rate": float(np.mean([row["selected"][0] != "block" for row in records])),
        "correct_family_selection_rate": float(np.mean([
            row["correct_family_selected"] for row in records
        ])),
        "correct_beats_deranged_or_base_rate": float(np.mean([
            row["correct_beats_comparison"] for row in records
        ])),
        "selected_held_nll_gain_per_scalar": _summary([
            row["selected_gain"] for row in records
        ]),
        "selected_counts": {
            f"{family}@{alpha:g}": int(sum(row["selected"] == (family, alpha) for row in records))
            for family, alpha in sorted(set(row["selected"] for row in records))
        },
        "correct_geometry_alpha_sensitivity": {
            alpha: _summary(values) for alpha, values in sensitivity.items() if values
        },
    }


def run_benchmark(args):
    nodes, kernels = candidate_kernels()
    prepared = prepare_candidates(kernels)
    threshold, null_maxima = calibrate_familywise_threshold(
        prepared,
        train_fields=args.train_fields,
        draws=args.calibration_draws,
        seed=args.seed + 1,
        confidence=args.confidence,
    )
    scenarios = [
        run_scenario(
            "block",
            np.eye(len(nodes)),
            0.0,
            kernels,
            prepared,
            threshold,
            args,
            args.seed + 100,
        )
    ]
    for family, kernel in kernels.items():
        for offset, alpha in enumerate(TRUTH_ALPHAS):
            scenarios.append(run_scenario(
                family,
                kernel,
                alpha,
                kernels,
                prepared,
                threshold,
                args,
                args.seed + 1000 * (list(kernels).index(family) + 1) + offset,
            ))
    block = scenarios[0]
    planted = [row for row in scenarios if row["truth_alpha"] >= 0.10]
    mechanism_gates = {
        "block_null_nonzero_selection_at_most_10pct": block["nonzero_selection_rate"] <= 0.10,
        "every_planted_selected_gain_positive": all(
            row["selected_held_nll_gain_per_scalar"]["mean"] > 0.0 for row in planted
        ),
        "every_planted_correct_beats_control_at_least_80pct": all(
            row["correct_beats_deranged_or_base_rate"] >= 0.80 for row in planted
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
        "status": "KNOWN-MEAN/KNOWN-B GRAPH-GEOMETRY MECHANISM AUDIT; NO REAL DEPLOYMENT",
        "design": "DESIGN_graph_geometry_confirmatory.md",
        "implementation": {
            "design": _content_record(os.path.join(root, "DESIGN_graph_geometry_confirmatory.md")),
            "core": _content_record(os.path.join(root, "graph_geometry.py")),
            "runner": _content_record(os.path.abspath(__file__)),
        },
        "configuration": configuration,
        "graph": {"nodes": list(nodes), "candidate_overlap": overlap},
        "kernel_diagnostics": {
            name: asdict(kernel_diagnostics(kernel)) for name, kernel in kernels.items()
        },
        "familywise_null": {
            "threshold_nll_gain_per_scalar": threshold,
            "calibration_maximum_gain": _summary(null_maxima),
        },
        "scenarios": scenarios,
        "mechanism_gates": mechanism_gates,
        "all_mechanism_gates_pass": all(mechanism_gates.values()),
        "real_covariance_gate_unlocked": False,
        "qr_deployment_unlocked": False,
        "reason": (
            "known mean/B and synthetic repeated fields do not identify real judge covariance; "
            "the preregistered repeated-judge campaign is absent"
        ),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--calibration-draws", type=int, default=1000)
    parser.add_argument("--train-fields", type=int, default=12)
    parser.add_argument("--held-fields", type=int, default=64)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=884200)
    parser.add_argument("--out", default="/tmp/graph_geometry_synthetic.json")
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
