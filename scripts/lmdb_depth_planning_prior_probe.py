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
from scripts.lmdb_parent_histogram_benchmark import percentile, safe_graph_name  # noqa: E402
from scripts.parent_histogram_recurrence import recurrence_parent_histogram  # noqa: E402


def mean(values):
    values = list(values)
    return 0.0 if not values else statistics.fmean(values)


def histogram_storage_bytes(hist):
    return len(hist) * 16


def dense_distribution_bytes(probabilities):
    return len(probabilities) * 8


def pruned_histogram_bytes(probabilities, tail_epsilon):
    return effective_support_bins(probabilities, tail_epsilon) * 16


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
    return {
        "base_distribution": base,
        "prior_distribution": prior,
        "base_mean_excess": base_mean,
        "base_variance_excess": base_variance,
        "prior_mean_excess": prior_mean,
        "prior_variance_excess": prior_variance,
        "prior_support_bins": len(prior),
        "prior_effective_bins": effective_support_bins(prior, tail_epsilon),
        "prior_dense_bytes": dense_distribution_bytes(prior),
        "prior_pruned_histogram_bytes": pruned_histogram_bytes(prior, tail_epsilon),
        "binomial_trials": depth,
        "binomial_probability": binomial_probability,
        "gamma_parameters": gamma_parameters(prior_mean, prior_variance),
    }


def compare_prior_to_hist(hist, prior, tail_epsilon):
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
            "storage_prediction_ratio": None,
        }
    predicted_bytes = int(prior["prior_pruned_histogram_bytes"])
    realized_bytes = histogram_storage_bytes(hist)
    return {
        "comparable": True,
        "histogram_L_min": origin,
        "histogram_L_max": origin + len(empirical) - 1,
        "support_bins": len(empirical),
        "effective_bins": effective_support_bins(empirical, tail_epsilon),
        "l1_error": l1_error(empirical, prior["prior_distribution"]),
        "max_cdf_error": max_cdf_error(empirical, prior["prior_distribution"]),
        "histogram_bytes": realized_bytes,
        "effective_histogram_bytes": pruned_histogram_bytes(empirical, tail_epsilon),
        "storage_prediction_ratio": None if realized_bytes == 0 else predicted_bytes / realized_bytes,
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
            record.update(compare_prior_to_hist(row["recurrence_histogram"], priors[key], args.tail_epsilon))
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
        "| L_max | targets | mean_root_p | b_root | mean_excess | prior_bins | prior_eff_bins | binom_p | gamma_shape | gamma_scale | prior_pruned_bytes |",
        "|------:|--------:|------------:|-------:|------------:|-----------:|---------------:|--------:|------------:|------------:|-------------------:|",
    ])
    for row in sorted(bucket_rows, key=lambda item: item["L_max_bucket"]):
        parent = row["root_reaching_parent_degree"]
        gamma = row["gamma_parameters"]
        lines.append(
            "| {bucket} | {targets} | {mean_p:.3f} | {branching:.6f} | {excess:.6f} | {prior_bins} | {prior_eff} | {binom_p:.6f} | {gamma_shape} | {gamma_scale} | {bytes} |".format(
                bucket=row["L_max_bucket"],
                targets=row["targets"],
                mean_p=parent["mean_parent_degree"],
                branching=0.0 if parent["size_biased_parent_branching"] is None else parent["size_biased_parent_branching"],
                excess=0.0 if parent["mean_excess"] is None else parent["mean_excess"],
                prior_bins=row["prior_support_bins"],
                prior_eff=row["prior_effective_bins"],
                binom_p=row["binomial_probability"],
                gamma_shape="n/a" if gamma["shape"] is None else "{:.6f}".format(gamma["shape"]),
                gamma_scale="n/a" if gamma["scale"] is None else "{:.6f}".format(gamma["scale"]),
                bytes=row["prior_pruned_histogram_bytes"],
            )
        )

    lines.extend([
        "",
        "## Prior vs Recurrence Histograms",
        "",
        "| L_max | rows | mean_l1 | p95_l1 | mean_cdf | mean_realized_bins | mean_pred_eff_bins | mean_storage_ratio | capped | cycle_approx |",
        "|------:|-----:|--------:|-------:|---------:|-------------------:|-------------------:|-------------------:|-------:|-------------:|",
    ])
    by_bucket = {}
    for row in comparable:
        by_bucket.setdefault(row["L_max"], []).append(row)
    for key in sorted(by_bucket):
        rows = by_bucket[key]
        prior = next(bucket for bucket in bucket_rows if bucket["L_max_bucket"] == key)
        lines.append(
            "| {bucket} | {rows} | {l1:.6f} | {p95_l1:.6f} | {cdf:.6f} | {bins:.3f} | {pred_bins:.3f} | {ratio:.3f} | {capped} | {cycle} |".format(
                bucket=key,
                rows=len(rows),
                l1=mean(row["l1_error"] for row in rows),
                p95_l1=percentile([row["l1_error"] for row in rows], 95),
                cdf=mean(row["max_cdf_error"] for row in rows),
                bins=mean(row["support_bins"] for row in rows),
                pred_bins=float(prior["prior_effective_bins"]),
                ratio=mean(row["storage_prediction_ratio"] for row in rows if row["storage_prediction_ratio"] is not None),
                capped=sum(1 for row in rows if row["recurrence_capped"]),
                cycle=sum(1 for row in rows if row["recurrence_cycle_approximation"]),
            )
        )

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
