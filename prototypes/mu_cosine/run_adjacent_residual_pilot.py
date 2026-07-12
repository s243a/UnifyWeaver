#!/usr/bin/env python3
"""Run the descriptive adjacent-row conditional-residual pilot.

This executable uses whole-endpoint-component OOF fits and anchor-sharing
nonadjacent controls.  Its multiplier bands are conditional stability
summaries, not population confidence intervals or a QR deployment gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adjacent_residual_pilot import (
    adjacency_feature_kernel,
    anchor_matched_contrasts,
    component_balanced_folds,
    component_contrast_estimate,
    component_multiplier_stability,
    marginal_preserving_adjacency_covariance,
    positive_row_pairs,
    principal_whiten_rows,
    within_descendant_derangement,
)
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from fine_tune_channel_heads import CAMPAIGN_E5_100K, load_campaign_datasets, load_expanded
from run_cheap_judge_joint_posterior import load_decision_targets
from run_covariance_sensitivity import fit_context
from run_product_kalman_realdata import DATASETS
from run_structured_residual_covariance import (
    DEFAULT_CAMPAIGN,
    DEFAULT_LUNA,
    configure_artifact_repo,
    file_provenance,
    materialize_corpus,
)
from structured_residual_covariance import gaussian_joint_nll


ROOT = os.path.dirname(os.path.abspath(__file__))
ADJACENCY_ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_payload(path, payload):
    serialized = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = os.path.abspath(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    os.replace(temporary, path)


def _undirected_neighbors(parents):
    neighbors = {}
    for child, values in parents.items():
        neighbors.setdefault(child, set()).update(values)
        for parent in values:
            neighbors.setdefault(parent, set()).add(child)
    return {node: frozenset(values) for node, values in neighbors.items()}


def _matrix_record(matrix):
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return {
        "matrix": matrix,
        "trace_over_4": float(np.trace(matrix) / 4.0),
        "eigenvalues": eigenvalues,
        "spectral_norm": float(np.max(np.abs(eigenvalues))),
    }


def _stability_record(band):
    values = asdict(band)
    names = values.pop("names")
    vector_fields = (
        "estimate",
        "standard_error",
        "pointwise_low",
        "pointwise_high",
        "simultaneous_low",
        "simultaneous_high",
    )
    by_stat = {}
    for index, name in enumerate(names):
        by_stat[name] = {
            field: float(values[field][index]) for field in vector_fields
        }
    for field in vector_fields:
        values.pop(field)
    values["by_stat"] = by_stat
    values["interpretation"] = (
        "conditional Rademacher component-multiplier stability band; not a population CI"
    )
    return values


def _matching_record(records, excluded, positive_count, pairs):
    component_sizes = sorted(Counter(row.component for row in records).values())
    return {
        "positive_pairs_total": int(positive_count),
        "positive_pairs_eligible": int(len(records)),
        "positive_pairs_excluded_no_anchor_control": int(len(excluded)),
        "positive_endpoint_components": int(len({row.component for row in records})),
        "positive_descendants": int(len({pairs[row.positive[0]][0] for row in records})),
        "eligible_positive_pairs_per_component_sorted": component_sizes,
        "controls_total_with_reuse": int(sum(len(row.controls) for row in records)),
        "positive_tag_pair_counts": dict(Counter(
            "/".join(row.positive_tag_pair) for row in records
        )),
        "mean_tag_distance": float(np.mean([row.mean_tag_distance for row in records])),
        "mean_degree_bin_difference": float(np.mean([
            row.mean_degree_bin_difference for row in records
        ])),
        "mean_semantic_distance": float(np.mean([
            row.mean_semantic_distance for row in records
        ])),
        "matching_note": (
            "controls share descendant and one anchor row; exact hop/tag matching has no support"
        ),
    }


def _oof_whitened_rows(
    materialized, assignment, neighbors, positive_components, assignment_seed, args
):
    row_count = len(materialized["pairs"])
    whitened = np.full((row_count, 4), np.nan, dtype=float)
    fold_records = []
    nll_components = []
    all_rows = np.arange(row_count, dtype=int)
    for fold, held in enumerate(assignment.folds):
        train = np.setdiff1d(all_rows, held, assume_unique=True)
        context = fit_context(materialized, train, held, args)
        whitened[held] = principal_whiten_rows(
            context.centered_evaluate,
            context.block_model.independent_covariance,
        )
        held_component_ids = assignment.component_ids[held]
        for component in sorted(set(held_component_ids) & set(positive_components)):
            local = np.flatnonzero(held_component_ids == component)
            component_pairs = [materialized["pairs"][held[index]] for index in local]
            component_residuals = context.centered_evaluate[local]
            kernel = adjacency_feature_kernel(component_pairs, component_pairs, neighbors)
            deranged, permutation = within_descendant_derangement(
                kernel,
                component_pairs,
                seed=900000 + 1000 * assignment_seed + 100 * fold + int(component),
            )
            topology_difference = kernel - deranged
            relative_topology_difference = float(
                np.linalg.norm(topology_difference)
                / max(np.linalg.norm(kernel), np.finfo(float).eps)
            )
            block_covariance = context.block_model.independent_covariance
            block_nll = gaussian_joint_nll(
                component_residuals,
                marginal_preserving_adjacency_covariance(
                    kernel, block_covariance, 0.0
                ),
            ).per_scalar
            curves = []
            for alpha in ADJACENCY_ALPHAS:
                adjacent_nll = gaussian_joint_nll(
                    component_residuals,
                    marginal_preserving_adjacency_covariance(
                        kernel, block_covariance, alpha
                    ),
                ).per_scalar
                deranged_nll = gaussian_joint_nll(
                    component_residuals,
                    marginal_preserving_adjacency_covariance(
                        deranged, block_covariance, alpha
                    ),
                ).per_scalar
                curves.append({
                    "alpha": alpha,
                    "adjacency_nll_per_scalar": float(adjacent_nll),
                    "deranged_nll_per_scalar": float(deranged_nll),
                    "adjacency_gain_vs_block": float(block_nll - adjacent_nll),
                    "deranged_gain_vs_block": float(block_nll - deranged_nll),
                })
            nll_components.append({
                "component": int(component),
                "fold": int(fold),
                "rows": int(len(local)),
                "derangement_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
                "derangement_relative_frobenius_difference": relative_topology_difference,
                "derangement_changed_entry_fraction": float(np.mean(
                    np.abs(topology_difference) > 1e-12
                )),
                "block_nll_per_scalar": float(block_nll),
                "curve": curves,
            })
        fold_records.append({
            **assignment.fold_diagnostics[fold],
            "train_rows": int(len(train)),
            "regional_mean": context.regional.to_dict(),
            "block_covariance": context.block_model.independent_covariance,
            "block_shrinkage": float(args.shrinkage),
            "semantic_rbf_bandwidth": float(context.semantic_length),
            "graph_rbf_bandwidth": float(context.graph_length),
        })
    if not np.isfinite(whitened).all():
        raise AssertionError("every row must receive one finite OOF whitened residual")
    curve = []
    for alpha in ADJACENCY_ALPHAS:
        rows = [
            next(value for value in component["curve"] if value["alpha"] == alpha)
            for component in nll_components
        ]
        curve.append({
            "alpha": alpha,
            "components": len(rows),
            "adjacency_nll_per_scalar_component_macro": float(np.mean([
                value["adjacency_nll_per_scalar"] for value in rows
            ])),
            "deranged_nll_per_scalar_component_macro": float(np.mean([
                value["deranged_nll_per_scalar"] for value in rows
            ])),
            "adjacency_gain_vs_block_component_macro": float(np.mean([
                value["adjacency_gain_vs_block"] for value in rows
            ])),
            "deranged_gain_vs_block_component_macro": float(np.mean([
                value["deranged_gain_vs_block"] for value in rows
            ])),
        })
    return whitened, fold_records, nll_components, curve


def run_assignment(corpus, materialized, neighbors, degrees, assignment_seed, args):
    positives = positive_row_pairs(materialized["pairs"], neighbors)
    assignment = component_balanced_folds(
        materialized["pairs"],
        materialized["tags"],
        positives,
        n_folds=args.folds,
        seed=assignment_seed,
    )
    records, excluded = anchor_matched_contrasts(
        materialized["pairs"],
        materialized["tags"],
        neighbors,
        degrees,
        materialized["semantic"],
        maximum_controls=args.maximum_controls,
    )
    if len({record.component for record in records}) < 2:
        raise ValueError(f"{corpus} has fewer than two eligible positive components")
    positive_components = {record.component for record in records}
    whitened, folds, nll_components, nll_curve = _oof_whitened_rows(
        materialized,
        assignment,
        neighbors,
        positive_components,
        assignment_seed,
        args,
    )
    estimate = component_contrast_estimate(whitened, records)
    stability = component_multiplier_stability(
        estimate.contrast_matrices,
        draws=args.multiplier_draws,
        seed=700000 + assignment_seed,
        confidence=args.stability_confidence,
    )
    fold_primary = []
    for held in assignment.folds:
        held_set = set(map(int, held))
        selected = [row for row in records if row.positive[0] in held_set]
        if selected:
            fold_primary.append(float(
                component_contrast_estimate(whitened, selected).primary_trace
            ))
        else:
            fold_primary.append(None)
    oracle = min(
        nll_curve,
        key=lambda value: (
            value["adjacency_nll_per_scalar_component_macro"], value["alpha"]
        ),
    )
    oracle_deranged = next(
        value for value in nll_curve if value["alpha"] == oracle["alpha"]
    )
    return {
        "corpus": corpus,
        "assignment_seed": int(assignment_seed),
        "scope": "descriptive existing-data adjacency-plus-local-hop predictive contrast",
        "matching": _matching_record(
            records, excluded, len(positives), materialized["pairs"]
        ),
        "component_assignment": {
            "folds": int(args.folds),
            "endpoint_components": int(assignment.component_count),
            "largest_endpoint_component_rows": int(assignment.largest_component),
            "fold_records": folds,
        },
        "primary_trace_over_4": float(estimate.primary_trace),
        "fold_primary_trace_over_4": fold_primary,
        "positive_cross_product": _matrix_record(estimate.positive),
        "anchor_control_cross_product": _matrix_record(estimate.control),
        "component_macro_contrast": _matrix_record(estimate.contrast),
        "edge_weighted_contrast_secondary": _matrix_record(
            estimate.edge_weighted_contrast
        ),
        "nondeployable_held_alpha_grid": {
            "warning": (
                "outer-held descriptive oracle; alpha was not nested-selected and cannot be deployed"
            ),
            "alphas": list(ADJACENCY_ALPHAS),
            "component_records": nll_components,
            "component_macro_curve": nll_curve,
            "adjacency_oracle": oracle,
            "deranged_at_adjacency_oracle_alpha": oracle_deranged,
            "diagnostic_checks": {
                "oracle_gain_positive": bool(
                    oracle["adjacency_gain_vs_block_component_macro"] > 0.0
                ),
                "adjacency_beats_deranged_at_oracle_alpha": bool(
                    oracle["adjacency_nll_per_scalar_component_macro"]
                    < oracle_deranged["deranged_nll_per_scalar_component_macro"]
                ),
            },
        },
        "stability": _stability_record(stability),
        "advance_only_checks": {
            "primary_positive": bool(estimate.primary_trace > 0.0),
            "at_least_4_of_5_fold_directions_positive": bool(
                sum(value is not None and value > 0.0 for value in fold_primary) >= 4
            ),
            "at_least_80pct_leave_one_component_out_positive": bool(
                stability.leave_one_component_out_positive_fraction >= 0.80
            ),
            "conditional_simultaneous_lower_trace_positive": (
                bool(stability.simultaneous_low[0] > 0.0)
                if stability.gate_evaluable else None
            ),
            "stability_gate_evaluable": bool(stability.gate_evaluable),
        },
    }


def aggregate_results(results, expected_assignments):
    by_corpus = {}
    for corpus in ("exploratory", "fresh"):
        rows = [row for row in results if row["corpus"] == corpus]
        values = np.asarray([row["primary_trace_over_4"] for row in rows], dtype=float)
        all_stability_evaluable = bool(
            len(rows) == expected_assignments
            and all(row["stability"]["gate_evaluable"] for row in rows)
        )
        diagnostic_pointwise_positive = int(sum(
            row["stability"]["by_stat"]["trace_over_4"]["pointwise_low"] > 0.0
            for row in rows
        ))
        diagnostic_simultaneous_positive = int(sum(
            row["stability"]["by_stat"]["trace_over_4"]["simultaneous_low"] > 0.0
            for row in rows
        ))
        alpha_curve = []
        for alpha in ADJACENCY_ALPHAS:
            selected = [
                next(
                    value for value in row["nondeployable_held_alpha_grid"][
                        "component_macro_curve"
                    ] if value["alpha"] == alpha
                )
                for row in rows
            ]
            alpha_curve.append({
                "alpha": alpha,
                "adjacency_gain_vs_block_mean": (
                    float(np.mean([
                        value["adjacency_gain_vs_block_component_macro"]
                        for value in selected
                    ])) if selected else None
                ),
                "deranged_gain_vs_block_mean": (
                    float(np.mean([
                        value["deranged_gain_vs_block_component_macro"]
                        for value in selected
                    ])) if selected else None
                ),
            })
        by_corpus[corpus] = {
            "assignments_completed": int(len(rows)),
            "expected_assignments": int(expected_assignments),
            "primary_trace_over_4_mean": float(np.mean(values)) if len(values) else None,
            "primary_trace_over_4_sd": (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) else None
            ),
            "primary_trace_over_4_min": float(np.min(values)) if len(values) else None,
            "primary_trace_over_4_max": float(np.max(values)) if len(values) else None,
            "positive_assignments": int(np.sum(values > 0.0)),
            "diagnostic_primary_pointwise_lower_positive_assignments": (
                diagnostic_pointwise_positive
            ),
            "diagnostic_primary_simultaneous_lower_positive_assignments": (
                diagnostic_simultaneous_positive
            ),
            "ci_like_positive_assignment_counts": (
                {
                    "pointwise": diagnostic_pointwise_positive,
                    "simultaneous": diagnostic_simultaneous_positive,
                } if all_stability_evaluable else None
            ),
            "nondeployable_held_alpha_curve": alpha_curve,
            "all_stability_gates_evaluable": all_stability_evaluable,
            "minimum_leave_one_component_out_positive_fraction": (
                float(min(
                    row["stability"]["leave_one_component_out_positive_fraction"]
                    for row in rows
                )) if rows else None
            ),
        }
    complete = all(
        by_corpus[corpus]["assignments_completed"] == expected_assignments
        for corpus in by_corpus
    )
    return {
        "by_corpus": by_corpus,
        "complete": complete,
        "advance_to_repeated_judge_confirmation": False,
        "qr_covariance_deployment": False,
        "reason": (
            "real pilot cannot pass deployment; full-procedure synthetic power and repeated-judge "
            "confirmation are not supplied by this runner"
        ),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo", default=os.path.abspath(os.path.join(ROOT, "..", "..")))
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--luna", default=DEFAULT_LUNA)
    parser.add_argument("--assignment-seed-start", type=int, default=20)
    parser.add_argument("--assignments", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-controls", type=int, default=3)
    parser.add_argument("--multiplier-draws", type=int, default=9999)
    parser.add_argument("--stability-confidence", type=float, default=0.95)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument(
        "--ridge-grid", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--lmdb-no-lock",
        action="store_true",
        help="read the immutable local fresh graph without LMDB locking",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="/tmp/adjacent_residual_pilot.json")
    return parser


def _validate_args(args):
    if args.assignments < 1 or args.folds != 5:
        raise ValueError("assignments must be positive and this frozen pilot requires exactly five folds")
    if args.maximum_controls < 1 or args.multiplier_draws < 1:
        raise ValueError("maximum-controls and multiplier-draws must be positive")
    if not 0.0 < args.stability_confidence < 1.0:
        raise ValueError("stability-confidence must be in (0,1)")


def main():
    args = build_arg_parser().parse_args()
    _validate_args(args)
    torch.set_num_threads(args.cpu_threads)
    np.random.seed(0)
    artifacts = configure_artifact_repo(args.artifact_repo)
    if args.lmdb_no_lock:
        DATASETS["fresh"]["graph"]["lmdb_no_lock"] = True
    target_by_pair, cur_rel = load_decision_targets(args.campaign)
    luna_pairs, luna_d, luna_s = load_scored_mu_tsv(args.luna)
    luna_by_pair = {pair: (luna_d[i], luna_s[i]) for i, pair in enumerate(luna_pairs)}
    checkpoint = load_expanded(args.ckpt, dev="cpu")
    checkpoint[0].eval()
    datasets = load_campaign_datasets(campaign_scored=args.campaign)
    materialized, graph = {}, {}
    for name, dataset in datasets.items():
        corpus = name.replace("-campaign", "")
        materialized[corpus] = materialize_corpus(
            name, dataset, target_by_pair, luna_by_pair, checkpoint
        )
        parents = dataset["tok"].parents
        graph[corpus] = {
            "neighbors": _undirected_neighbors(parents),
            "degrees": dataset["tok"].deg,
        }
    configuration = vars(args).copy()
    configuration.pop("resume")
    payload = {
        "schema_version": 1,
        "status": "DESCRIPTIVE EXISTING-DATA PILOT; NO POPULATION CI OR QR DEPLOYMENT GATE",
        "design": "DESIGN_adjacent_residual_pilot.md",
        "target_scope": "GPT-5.5 operating-judge fidelity; not independent ground truth",
        "estimand": "same-descendant adjacency plus local-hop proximity predictive contrast",
        "implementation": {
            "design": file_provenance(os.path.join(ROOT, "DESIGN_adjacent_residual_pilot.md")),
            "core": file_provenance(os.path.join(ROOT, "adjacent_residual_pilot.py")),
            "runner": file_provenance(os.path.abspath(__file__)),
            "dependencies": {
                name: file_provenance(os.path.join(ROOT, name))
                for name in (
                    "fine_tune_channel_heads.py",
                    "run_cheap_judge_joint_posterior.py",
                    "run_covariance_sensitivity.py",
                    "run_structured_residual_covariance.py",
                    "structured_residual_covariance.py",
                )
            },
        },
        "inputs": {
            "checkpoint": file_provenance(args.ckpt),
            "campaign": file_provenance(args.campaign),
            "luna": file_provenance(args.luna),
            "artifact_paths": artifacts,
            "e5_caches": {
                "exploratory": file_provenance(CAMPAIGN_E5_100K),
                "fresh": file_provenance(DATASETS["fresh"]["e5_cache"]),
            },
            "campaign_cur_rel_counts": dict(cur_rel),
        },
        "configuration": configuration,
        "results": [],
        "aggregate": None,
    }
    if args.resume and os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as stream:
            previous = json.load(stream)
        for key in (
            "schema_version", "design", "implementation", "inputs", "configuration", "estimand"
        ):
            if previous.get(key) != payload[key]:
                raise ValueError(f"refusing to resume with mismatched {key}")
        payload = previous
    completed = {
        (row["corpus"], int(row["assignment_seed"])) for row in payload["results"]
    }
    if len(completed) != len(payload["results"]):
        raise ValueError("resume output contains duplicate corpus/assignment records")
    started = time.perf_counter()
    for seed in range(args.assignment_seed_start, args.assignment_seed_start + args.assignments):
        for corpus in ("exploratory", "fresh"):
            if (corpus, seed) in completed:
                print(f"skipping completed {corpus} assignment {seed}", flush=True)
                continue
            print(f"\n=== adjacent residual pilot: {corpus}, assignment {seed} ===", flush=True)
            row = run_assignment(
                corpus,
                materialized[corpus],
                graph[corpus]["neighbors"],
                graph[corpus]["degrees"],
                seed,
                args,
            )
            payload["results"].append(row)
            completed.add((corpus, seed))
            payload["aggregate"] = aggregate_results(payload["results"], args.assignments)
            _write_payload(args.out, payload)
            print(
                f"  trace/4={row['primary_trace_over_4']:+.6f}; "
                f"eligible={row['matching']['positive_pairs_eligible']}; "
                f"components={row['stability']['components']}; "
                f"LOCO+={row['stability']['leave_one_component_out_positive_fraction']:.1%}",
                flush=True,
            )
    payload["aggregate"] = aggregate_results(payload["results"], args.assignments)
    _write_payload(args.out, payload)
    print(f"\nwrote {args.out}; wall_seconds={time.perf_counter() - started:.1f}")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
