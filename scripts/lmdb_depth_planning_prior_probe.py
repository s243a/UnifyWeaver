#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Compare depth-conditioned planning priors with recurrence histograms.

The planning prior is a global or bucketed estimate.  It is not a boundary
condition for a concrete node.  This probe checks whether the prior predicts the
shape and storage cost of node-local recurrence histograms well enough to guide
materialization decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distribution_fit_comparison import (  # noqa: E402
    binomial_pmf,
    exact_excess_distribution,
    effective_support_bins,
    gamma_midpoint_pmf,
    l1_error,
    max_cdf_error,
    nfold_convolution,
    size_biased_excess_pmf,
    distribution_moments,
)
from scripts.lmdb_parent_branching_diagnostic import (  # noqa: E402
    LmdbCategoryGraph,
    parse_int_list,
    root_distances,
    select_targets_by_child_depth,
    size_biased_branching,
)
from scripts.lmdb_parent_histogram_benchmark import safe_graph_name  # noqa: E402
from scripts.parent_histogram_recurrence import recurrence_parent_histogram  # noqa: E402


def mean(values):
    values = list(values)
    return 0.0 if not values else statistics.fmean(values)


def mean_present(values):
    return mean(value for value in values if value is not None)


def histogram_storage_bytes(hist):
    return len(hist) * 16


def dense_distribution_bytes(probabilities):
    return len(probabilities) * 8


def pruned_histogram_bytes(probabilities, tail_epsilon):
    return effective_support_bins(probabilities, tail_epsilon) * 16


def parametric_distribution_bytes():
    return 64


def gamma_parameters(mean_value, variance):
    if mean_value <= 0.0 or variance <= 1e-12:
        return {"family": "degenerate", "shape": None, "scale": 0.0}
    return {
        "family": "gamma_midpoint",
        "shape": (mean_value * mean_value) / variance,
        "scale": variance / mean_value,
    }


def planning_prior_for_bucket(parent_degrees, depth, tail_epsilon):
    """Build a depth-conditioned prior from root-reaching parent degrees."""
    depth = max(0, int(depth))
    base = size_biased_excess_pmf(parent_degrees)
    prior = nfold_convolution(base, depth)
    base_mean, base_variance = distribution_moments(base)
    prior_mean, prior_variance = distribution_moments(prior)
    binomial_probability = 0.0 if depth <= 0 else max(0.0, min(1.0, prior_mean / depth))
    binomial_prior = binomial_pmf(depth, binomial_probability)
    gamma_prior, gamma_params = gamma_midpoint_pmf(prior_mean, prior_variance, len(prior))
    gamma_params = dict(gamma_params)
    gamma_params.update(gamma_parameters(prior_mean, prior_variance))
    return {
        "base_distribution": base,
        "prior_distribution": prior,
        "binomial_distribution": binomial_prior,
        "gamma_distribution": gamma_prior,
        "base_mean_excess": base_mean,
        "base_variance_excess": base_variance,
        "prior_mean_excess": prior_mean,
        "prior_variance_excess": prior_variance,
        "prior_support_bins": len(prior),
        "prior_effective_bins": effective_support_bins(prior, tail_epsilon),
        "prior_dense_bytes": dense_distribution_bytes(prior),
        "prior_pruned_histogram_bytes": pruned_histogram_bytes(prior, tail_epsilon),
        "binomial_effective_bins": effective_support_bins(binomial_prior, tail_epsilon),
        "binomial_pruned_histogram_bytes": pruned_histogram_bytes(binomial_prior, tail_epsilon),
        "gamma_effective_bins": effective_support_bins(gamma_prior, tail_epsilon),
        "gamma_pruned_histogram_bytes": pruned_histogram_bytes(gamma_prior, tail_epsilon),
        "parametric_bytes": parametric_distribution_bytes(),
        "binomial_trials": depth,
        "binomial_probability": binomial_probability,
        "gamma_parameters": gamma_params,
    }


def planning_ratios(predicted_bytes, realized_bytes):
    if realized_bytes <= 0:
        return {
            "ratio": None,
            "underpredicts": False,
            "overpredicts": False,
        }
    ratio = predicted_bytes / realized_bytes
    return {
        "ratio": ratio,
        "underpredicts": ratio < 1.0,
        "overpredicts": ratio > 1.0,
    }


def admission_decision(row, prior, comparison, safety_factor):
    if not comparison.get("comparable"):
        return "skip_no_recurrence"
    if row.get("recurrence_capped") or row.get("recurrence_cycle_approximation"):
        return "risky_try_capped_or_approx"
    if comparison["safety_storage_prediction_ratio"] is not None and comparison["safety_storage_prediction_ratio"] <= 2.0:
        return "exact_recurrence_likely_cheap"
    if int(prior["prior_effective_bins"]) <= 32:
        return "exact_recurrence_likely_cheap"
    return "approximation_first"


def compare_prior_to_hist(hist, prior, tail_epsilon, safety_factor=1.0):
    empirical, origin = exact_excess_distribution(hist)
    if not empirical or origin is None:
        return {
            "comparable": False,
            "histogram_L_min": None,
            "histogram_L_max": None,
            "support_bins": 0,
            "effective_bins": 0,
            "l1_error": None,
            "max_cdf_error": None,
            "histogram_bytes": 0,
            "effective_histogram_bytes": 0,
            "empirical_storage_prediction_ratio": None,
            "binomial_storage_prediction_ratio": None,
            "gamma_storage_prediction_ratio": None,
            "parametric_storage_prediction_ratio": None,
            "safety_storage_prediction_ratio": None,
        }
    realized_bytes = histogram_storage_bytes(hist)
    empirical_ratio = planning_ratios(int(prior["prior_pruned_histogram_bytes"]), realized_bytes)
    safety_bytes = int(math.ceil(float(safety_factor) * int(prior["prior_pruned_histogram_bytes"])))
    safety_ratio = planning_ratios(safety_bytes, realized_bytes)
    binomial_ratio = planning_ratios(int(prior["binomial_pruned_histogram_bytes"]), realized_bytes)
    gamma_ratio = planning_ratios(int(prior["gamma_pruned_histogram_bytes"]), realized_bytes)
    parametric_ratio = planning_ratios(int(prior["parametric_bytes"]), realized_bytes)
    return {
        "comparable": True,
        "histogram_L_min": origin,
        "histogram_L_max": origin + len(empirical) - 1,
        "support_bins": len(empirical),
        "effective_bins": effective_support_bins(empirical, tail_epsilon),
        "l1_error": l1_error(empirical, prior["prior_distribution"]),
        "max_cdf_error": max_cdf_error(empirical, prior["prior_distribution"]),
        "binomial_l1_error": l1_error(empirical, prior["binomial_distribution"]),
        "binomial_max_cdf_error": max_cdf_error(empirical, prior["binomial_distribution"]),
        "gamma_l1_error": l1_error(empirical, prior["gamma_distribution"]),
        "gamma_max_cdf_error": max_cdf_error(empirical, prior["gamma_distribution"]),
        "histogram_bytes": realized_bytes,
        "effective_histogram_bytes": pruned_histogram_bytes(empirical, tail_epsilon),
        "empirical_storage_prediction_ratio": empirical_ratio["ratio"],
        "empirical_storage_underpredicts": empirical_ratio["underpredicts"],
        "empirical_storage_overpredicts": empirical_ratio["overpredicts"],
        "safety_factor": float(safety_factor),
        "safety_prediction_bytes": safety_bytes,
        "safety_storage_prediction_ratio": safety_ratio["ratio"],
        "safety_storage_underpredicts": safety_ratio["underpredicts"],
        "safety_storage_overpredicts": safety_ratio["overpredicts"],
        "binomial_storage_prediction_ratio": binomial_ratio["ratio"],
        "binomial_storage_underpredicts": binomial_ratio["underpredicts"],
        "binomial_storage_overpredicts": binomial_ratio["overpredicts"],
        "gamma_storage_prediction_ratio": gamma_ratio["ratio"],
        "gamma_storage_underpredicts": gamma_ratio["underpredicts"],
        "gamma_storage_overpredicts": gamma_ratio["overpredicts"],
        "parametric_storage_prediction_ratio": parametric_ratio["ratio"],
    }


def root_reaching_parent_degree(graph, parents, distances):
    return sum(1 for parent in parents if distances(parent)["L_min"] is not None)


def build_target_rows(args, graph, targets, target_child_depth):
    distance_memo = {}

    def distances(node):
        return root_distances(node, args.root, graph.parents, args.max_parent_depth, distance_memo)

    rows = []
    for target in targets:
        parents = graph.parents(target)
        target_distances = distances(target)
        budget = target_distances["L_max"]
        hist = {}
        rec_stats = None
        if budget is not None and budget <= args.max_prior_depth:
            hist, rec_stats = recurrence_parent_histogram(
                graph.parents,
                target,
                args.root,
                budget,
                args.path_cap,
                args.expansion_cap,
            )
        rows.append({
            "target_node": target,
            "child_sample_depth": target_child_depth[target],
            "L_min": target_distances["L_min"],
            "L_max": target_distances["L_max"],
            "distance_truncated": target_distances["truncated"],
            "cycle_skipped": target_distances["cycle_skipped"],
            "full_parent_degree": len(parents),
            "root_reaching_parent_degree": root_reaching_parent_degree(graph, parents, distances),
            "recurrence_histogram": hist,
            "recurrence_cycle_approximation": False if rec_stats is None else rec_stats.cycle_approximation,
            "recurrence_capped": False if rec_stats is None else rec_stats.path_cap_hit or rec_stats.expansion_cap_hit,
            "recurrence_states_evaluated": None if rec_stats is None else rec_stats.states_evaluated,
        })
    return rows


def bucket_key(row):
    if row["L_max"] is None:
        return "unreachable_or_truncated"
    return int(row["L_max"])


def build_records(args, target_rows, selection_counts):
    records = [{
        "record_type": "depth_planning_prior_selection",
        "graph": args.graph_name,
        "root": args.root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_selection_counts": selection_counts,
        "targets": len(target_rows),
        "tail_epsilon": args.tail_epsilon,
        "max_prior_depth": args.max_prior_depth,
        "safety_factor": args.safety_factor,
    }]
    buckets = {}
    for row in target_rows:
        buckets.setdefault(bucket_key(row), []).append(row)

    priors = {}
    for key, rows in buckets.items():
        if isinstance(key, str) or key > args.max_prior_depth:
            continue
        degrees = [row["root_reaching_parent_degree"] for row in rows if row["root_reaching_parent_degree"] > 0]
        if not degrees:
            continue
        priors[key] = planning_prior_for_bucket(degrees, key, args.tail_epsilon)
        branching = size_biased_branching(degrees)
        prior = priors[key]
        records.append({
            "record_type": "depth_planning_prior_bucket",
            "graph": args.graph_name,
            "root": args.root,
            "L_max_bucket": key,
            "targets": len(rows),
            "root_reaching_parent_degree": branching,
            "base_support_bins": len(prior["base_distribution"]),
            "base_mean_excess": prior["base_mean_excess"],
            "base_variance_excess": prior["base_variance_excess"],
            "prior_support_bins": prior["prior_support_bins"],
            "prior_effective_bins": prior["prior_effective_bins"],
            "prior_mean_excess": prior["prior_mean_excess"],
            "prior_variance_excess": prior["prior_variance_excess"],
            "prior_dense_bytes": prior["prior_dense_bytes"],
            "prior_pruned_histogram_bytes": prior["prior_pruned_histogram_bytes"],
            "binomial_effective_bins": prior["binomial_effective_bins"],
            "binomial_pruned_histogram_bytes": prior["binomial_pruned_histogram_bytes"],
            "gamma_effective_bins": prior["gamma_effective_bins"],
            "gamma_pruned_histogram_bytes": prior["gamma_pruned_histogram_bytes"],
            "parametric_bytes": prior["parametric_bytes"],
            "binomial_trials": prior["binomial_trials"],
            "binomial_probability": prior["binomial_probability"],
            "gamma_parameters": prior["gamma_parameters"],
        })

    for row in target_rows:
        key = bucket_key(row)
        record = {
            "record_type": "depth_planning_prior_target",
            "graph": args.graph_name,
            "root": args.root,
            **{name: value for name, value in row.items() if name != "recurrence_histogram"},
            "has_planning_prior": key in priors,
        }
        if key in priors and row["recurrence_histogram"]:
            comparison = compare_prior_to_hist(
                row["recurrence_histogram"],
                priors[key],
                args.tail_epsilon,
                args.safety_factor,
            )
            record.update(comparison)
            record["admission_decision"] = admission_decision(row, priors[key], comparison, args.safety_factor)
        records.append(record)
    return records


def summarize(records):
    selections = [row for row in records if row["record_type"] == "depth_planning_prior_selection"]
    bucket_rows = [row for row in records if row["record_type"] == "depth_planning_prior_bucket"]
    target_rows = [row for row in records if row["record_type"] == "depth_planning_prior_target"]
    comparable = [row for row in target_rows if row.get("comparable")]
    lines = [
        "# Depth-Conditioned Planning Prior Probe",
        "",
        "Graph: `{}`".format(selections[0]["graph"] if selections else "unknown"),
        "",
        "Root: `{}`".format(selections[0]["root"] if selections else "unknown"),
        "",
        "## Selection",
        "",
        "| child_depth | sampled_frontier_nodes |",
        "|-------------|------------------------|",
    ]
    selection_counts = selections[0]["target_selection_counts"] if selections else {}
    for depth in sorted(int(key) for key in selection_counts):
        lines.append("| {} | {} |".format(depth, selection_counts[depth]))

    lines.extend([
        "",
        "## Prior Buckets",
        "",
        "| L_max | targets | mean_root_p | b_root | mean_excess | empirical_eff_bins | binomial_eff_bins | gamma_eff_bins | binom_p | gamma_shape | empirical_bytes | binomial_bytes | gamma_bytes |",
        "|------:|--------:|------------:|-------:|------------:|-------------------:|------------------:|---------------:|--------:|------------:|----------------:|---------------:|------------:|",
    ])
    for row in sorted(bucket_rows, key=lambda item: item["L_max_bucket"]):
        parent = row["root_reaching_parent_degree"]
        gamma = row["gamma_parameters"]
        lines.append(
            "| {bucket} | {targets} | {mean_p:.3f} | {branching:.6f} | {excess:.6f} | {emp_eff} | {binom_eff} | {gamma_eff} | {binom_p:.6f} | {gamma_shape} | {emp_bytes} | {binom_bytes} | {gamma_bytes} |".format(
                bucket=row["L_max_bucket"],
                targets=row["targets"],
                mean_p=parent["mean_parent_degree"],
                branching=0.0 if parent["size_biased_parent_branching"] is None else parent["size_biased_parent_branching"],
                excess=0.0 if parent["mean_excess"] is None else parent["mean_excess"],
                emp_eff=row["prior_effective_bins"],
                binom_eff=row["binomial_effective_bins"],
                gamma_eff=row["gamma_effective_bins"],
                binom_p=row["binomial_probability"],
                gamma_shape="n/a" if gamma["shape"] is None else "{:.6f}".format(gamma["shape"]),
                emp_bytes=row["prior_pruned_histogram_bytes"],
                binom_bytes=row["binomial_pruned_histogram_bytes"],
                gamma_bytes=row["gamma_pruned_histogram_bytes"],
            )
        )

    lines.extend([
        "",
        "## Planning Calibration",
        "",
        "| L_max | rows | mean_realized_bins | empirical_eff_bins | mean_emp_ratio | mean_safety_ratio | safety_under | mean_binom_ratio | binom_under | mean_gamma_ratio | gamma_under | capped | cycle_approx |",
        "|------:|-----:|-------------------:|-------------------:|---------------:|------------------:|-------------:|-----------------:|------------:|-----------------:|------------:|-------:|-------------:|",
    ])
    by_bucket = {}
    for row in comparable:
        by_bucket.setdefault(row["L_max"], []).append(row)
    for key in sorted(by_bucket):
        rows = by_bucket[key]
        prior = next(bucket for bucket in bucket_rows if bucket["L_max_bucket"] == key)
        lines.append(
            "| {bucket} | {rows} | {bins:.3f} | {pred_bins:.3f} | {emp_ratio:.3f} | {safety_ratio:.3f} | {safety_under} | {binom_ratio:.3f} | {binom_under} | {gamma_ratio:.3f} | {gamma_under} | {capped} | {cycle} |".format(
                bucket=key,
                rows=len(rows),
                bins=mean(row["support_bins"] for row in rows),
                pred_bins=float(prior["prior_effective_bins"]),
                emp_ratio=mean_present(row["empirical_storage_prediction_ratio"] for row in rows),
                safety_ratio=mean_present(row["safety_storage_prediction_ratio"] for row in rows),
                safety_under=sum(1 for row in rows if row["safety_storage_underpredicts"]),
                binom_ratio=mean_present(row["binomial_storage_prediction_ratio"] for row in rows),
                binom_under=sum(1 for row in rows if row["binomial_storage_underpredicts"]),
                gamma_ratio=mean_present(row["gamma_storage_prediction_ratio"] for row in rows),
                gamma_under=sum(1 for row in rows if row["gamma_storage_underpredicts"]),
                capped=sum(1 for row in rows if row["recurrence_capped"]),
                cycle=sum(1 for row in rows if row["recurrence_cycle_approximation"]),
            )
        )

    lines.extend([
        "",
        "## Shape Diagnostics",
        "",
        "| L_max | rows | empirical_l1 | empirical_cdf | binomial_l1 | gamma_l1 |",
        "|------:|-----:|-------------:|--------------:|------------:|---------:|",
    ])
    for key in sorted(by_bucket):
        rows = by_bucket[key]
        lines.append(
            "| {bucket} | {rows} | {emp_l1:.6f} | {emp_cdf:.6f} | {binom_l1:.6f} | {gamma_l1:.6f} |".format(
                bucket=key,
                rows=len(rows),
                emp_l1=mean(row["l1_error"] for row in rows),
                emp_cdf=mean(row["max_cdf_error"] for row in rows),
                binom_l1=mean(row["binomial_l1_error"] for row in rows),
                gamma_l1=mean(row["gamma_l1_error"] for row in rows),
            )
        )

    lines.extend([
        "",
        "## Admission Decisions",
        "",
        "| decision | rows |",
        "|----------|-----:|",
    ])
    decision_counts = {}
    for row in comparable:
        decision_counts[row.get("admission_decision", "unknown")] = decision_counts.get(row.get("admission_decision", "unknown"), 0) + 1
    for decision in sorted(decision_counts):
        lines.append("| {} | {} |".format(decision, decision_counts[decision]))

    skipped = [row for row in target_rows if not row.get("comparable")]
    lines.extend([
        "",
        "## Coverage",
        "",
        "| targets | comparable | skipped | buckets |",
        "|--------:|-----------:|--------:|--------:|",
        "| {} | {} | {} | {} |".format(len(target_rows), len(comparable), len(skipped), len(bucket_rows)),
    ])
    return "\n".join(lines) + "\n"


def run_probe(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        targets, target_child_depth, selection_counts = select_targets_by_child_depth(
            graph,
            args.root,
            parse_int_list(args.target_depths),
            args.children_per_node,
            args.frontier_limit,
            args.targets_per_depth,
            args.seed,
        )
        target_rows = build_target_rows(args, graph, targets, target_child_depth)
        records = build_records(args, target_rows, selection_counts)
        return records, summarize(records)
    finally:
        graph.close()


def write_outputs(records, summary, output_dir, graph_name, write_jsonl=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "{}_depth_planning_prior_summary.json".format(safe_name)
    summary_md = output_dir / "{}_depth_planning_prior_summary.md".format(safe_name)
    summary_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(summary, encoding="utf-8")
    jsonl_path = None
    if write_jsonl:
        jsonl_path = output_dir / "{}_depth_planning_prior.jsonl".format(safe_name)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return summary_json, summary_md, jsonl_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_depth_planning_prior")
    parser.add_argument("--target-depths", default="2,3,4")
    parser.add_argument("--children-per-node", type=int, default=32)
    parser.add_argument("--frontier-limit", type=int, default=300)
    parser.add_argument("--targets-per-depth", type=int, default=8)
    parser.add_argument("--max-parent-depth", type=int, default=24)
    parser.add_argument("--max-prior-depth", type=int, default=8)
    parser.add_argument("--tail-epsilon", type=float, default=0.001)
    parser.add_argument("--safety-factor", type=float, default=1.25, help="Multiplier for empirical prior bytes in admission planning.")
    parser.add_argument("--path-cap", type=int, default=10000)
    parser.add_argument("--expansion-cap", type=int, default=50000)
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--seed", default="depth-planning-prior-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records, summary = run_probe(args)
    summary_json, summary_md, jsonl_path = write_outputs(records, summary, args.output_dir, args.graph_name, args.write_jsonl)
    print(summary, end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    if jsonl_path is not None:
        print("jsonl={}".format(jsonl_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
