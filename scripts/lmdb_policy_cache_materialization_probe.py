#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Probe exact versus closed-form cache materialization candidates.

Policy residency says a node is useful enough to keep.  It does not say that an
exact boundary histogram is cheap enough to compute.  This probe classifies
policy-resident nodes by suffix budget:

- `exact_histogram`: exact bounded histogram materialized without caps.
- `budget_too_short`: root distance is known but exceeds the suffix budget.
- `closed_form_candidate`: exact materialization hit a cap; store a compact
  closed-form approximation candidate instead, initially a binomial support prior.
- `unresolved`: no exact histogram and no usable support metadata.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distribution_fit_comparison import distribution_moments, exact_excess_distribution
from scripts.lmdb_ancestor_cache_policy_sweep import (
    CacheConfig,
    collect_distances,
    collect_query_inputs,
    sweep_configs,
)
from scripts.lmdb_parent_branching_diagnostic import LmdbCategoryGraph, parse_int_list
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram, percentile, safe_graph_name
from scripts.lmdb_policy_cache_runtime_benchmark import policy_cache_entries


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def binomial_support_prior(entry, hist, boundary_budget):
    if entry.l_min is None:
        return None
    support_min = int(entry.l_min)
    support_max = boundary_budget
    if entry.l_max is not None:
        support_max = min(int(entry.l_max), boundary_budget)
    if support_max < support_min:
        return None
    trials = support_max - support_min

    probability = 0.5
    source = "support_midpoint"
    empirical, origin = exact_excess_distribution(hist)
    if empirical and origin is not None and trials > 0:
        partial_mean, partial_variance = distribution_moments(empirical)
        shifted_mean = (origin - support_min) + partial_mean
        probability = max(0.0, min(1.0, shifted_mean / trials))
        source = "partial_histogram"
    elif trials == 0:
        probability = 0.0

    return {
        "family": "binomial_support_prior",
        "source": source,
        "support_min": support_min,
        "support_max": support_max,
        "trials": trials,
        "probability": probability,
        "storage_scalars": 5,
        "storage_bytes_estimate": 40,
    }


def classify_materialization(entry, hist, stats, boundary_budget):
    if hist and not stats.path_cap_hit and not stats.expansion_cap_hit:
        return "exact_histogram", None
    if entry.l_min is not None and int(entry.l_min) > boundary_budget:
        return "budget_too_short", None
    if stats.path_cap_hit or stats.expansion_cap_hit:
        model = binomial_support_prior(entry, hist, boundary_budget)
        if model is not None:
            return "closed_form_candidate", model
    if hist:
        model = binomial_support_prior(entry, hist, boundary_budget)
        if model is not None:
            return "closed_form_candidate", model
    return "unresolved", None


def probe_entry(parents_func, root, slot, entry, boundary_budget, path_cap, expansion_cap):
    started = time.perf_counter_ns()
    hist, stats = bounded_parent_histogram(
        parents_func,
        entry.node,
        root,
        boundary_budget,
        path_cap,
        expansion_cap,
    )
    elapsed = time.perf_counter_ns() - started
    classification, fallback = classify_materialization(entry, hist, stats, boundary_budget)
    return {
        "record_type": "policy_cache_materialization_probe",
        "slot": slot,
        "node": entry.node,
        "boundary_budget": boundary_budget,
        "classification": classification,
        "fallback_model": fallback,
        "l_min": entry.l_min,
        "l_max": entry.l_max,
        "distance_truncated": entry.truncated,
        "path_count": sum(hist.values()),
        "support_bins": len(hist),
        "nodes_expanded": stats.nodes_expanded,
        "edges_examined": stats.edges_examined,
        "cycle_skips": stats.cycle_skips,
        "path_cap_hit": stats.path_cap_hit,
        "expansion_cap_hit": stats.expansion_cap_hit,
        "histogram_time_ns": elapsed,
    }


def summarize_records(args, target_counts, query_inputs, config, policy_actions, rows):
    by_budget = {}
    for row in rows:
        by_budget.setdefault(row["boundary_budget"], []).append(row)
    budget_rows = []
    for boundary_budget in sorted(by_budget):
        bucket = by_budget[boundary_budget]
        classifications = Counter(row["classification"] for row in bucket)
        fallback_rows = [row for row in bucket if row["fallback_model"]]
        support_widths = [
            row["fallback_model"]["support_max"] - row["fallback_model"]["support_min"]
            for row in fallback_rows
        ]
        budget_rows.append({
            "boundary_budget": boundary_budget,
            "resident_entries": len(bucket),
            "classification_counts": dict(classifications),
            "exact_rate": 0.0 if not bucket else classifications.get("exact_histogram", 0) / len(bucket),
            "closed_form_candidate_rate": 0.0 if not bucket else classifications.get("closed_form_candidate", 0) / len(bucket),
            "budget_too_short_rate": 0.0 if not bucket else classifications.get("budget_too_short", 0) / len(bucket),
            "unresolved_rate": 0.0 if not bucket else classifications.get("unresolved", 0) / len(bucket),
            "path_cap_entries": sum(1 for row in bucket if row["path_cap_hit"]),
            "expansion_cap_entries": sum(1 for row in bucket if row["expansion_cap_hit"]),
            "mean_nodes_expanded": mean(row["nodes_expanded"] for row in bucket),
            "p95_nodes_expanded": percentile([row["nodes_expanded"] for row in bucket], 95),
            "mean_histogram_time_ns": mean(row["histogram_time_ns"] for row in bucket),
            "fallback_models": Counter(row["fallback_model"]["family"] for row in fallback_rows),
            "mean_fallback_support_width": mean(support_widths),
            "p95_fallback_support_width": percentile(support_widths, 95),
        })
    ancestor_nodes = [len(query.space.nodes) for query in query_inputs]
    capped_queries = sum(1 for query in query_inputs if query.space.capped)
    return {
        "record_type": "policy_cache_materialization_summary",
        "graph": args.graph_name,
        "root": args.root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_depths": parse_int_list(args.target_depths),
        "target_selection_counts": target_counts,
        "targets": len(query_inputs),
        "ancestor_capped_queries": capped_queries,
        "ancestor_capped_query_rate": 0.0 if not query_inputs else capped_queries / len(query_inputs),
        "mean_ancestor_nodes": mean(ancestor_nodes),
        "p95_ancestor_nodes": percentile(ancestor_nodes, 95),
        "cache_slots": config.cache_slots,
        "admit_l_min": config.admit_l_min,
        "admit_l_max": config.admit_l_max,
        "path_cap": args.path_cap,
        "expansion_cap": args.expansion_cap,
        "policy_actions": dict(policy_actions),
        "policy_resident_entries": len(set(row["node"] for row in rows)),
        "budget_rows": budget_rows,
    }


def markdown_summary(summaries):
    lines = [
        "# Policy Cache Materialization Probe",
        "",
        "| slots | L_min | L_max | boundary_budget | resident | exact | closed_form | too_short | unresolved | expansion_capped | mean_nodes | mean_fallback_width |",
        "|------:|------:|------:|----------------:|---------:|------:|------------:|----------:|-----------:|-----------------:|-----------:|--------------------:|",
    ]
    for summary in summaries:
        for row in summary["budget_rows"]:
            counts = row["classification_counts"]
            lines.append(
                "| {slots} | {l_min} | {l_max} | {budget} | {resident} | {exact} | {closed} | {short} | {unresolved} | {capped} | {mean_nodes:.1f} | {width:.1f} |".format(
                    slots=summary["cache_slots"],
                    l_min=summary["admit_l_min"],
                    l_max=summary["admit_l_max"],
                    budget=row["boundary_budget"],
                    resident=row["resident_entries"],
                    exact=counts.get("exact_histogram", 0),
                    closed=counts.get("closed_form_candidate", 0),
                    short=counts.get("budget_too_short", 0),
                    unresolved=counts.get("unresolved", 0),
                    capped=row["expansion_cap_entries"],
                    mean_nodes=row["mean_nodes_expanded"],
                    width=row["mean_fallback_support_width"],
                )
            )
    return "\n".join(lines) + "\n"


def run_probe(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        _targets, target_counts, query_inputs = collect_query_inputs(graph, args)
        distances = collect_distances(graph, args, query_inputs)
        summaries = []
        records = []
        for config in sweep_configs(args):
            policy_cache, policy_actions = policy_cache_entries(query_inputs, distances, config)
            rows = []
            for boundary_budget in parse_int_list(args.boundary_budgets):
                for slot, entry in sorted(policy_cache.items()):
                    rows.append(probe_entry(
                        graph.parents,
                        args.root,
                        slot,
                        entry,
                        boundary_budget,
                        args.path_cap,
                        args.expansion_cap,
                    ))
            summary = summarize_records(args, target_counts, query_inputs, config, policy_actions, rows)
            summaries.append(summary)
            records.append(summary)
            if args.write_jsonl:
                records.extend(rows)
        return records, summaries
    finally:
        graph.close()


def write_outputs(records, summaries, output_dir, graph_name, write_jsonl=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "{}_policy_cache_materialization_summary.json".format(safe_name)
    summary_md = output_dir / "{}_policy_cache_materialization_summary.md".format(safe_name)
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_summary(summaries), encoding="utf-8")
    jsonl_path = None
    if write_jsonl:
        jsonl_path = output_dir / "{}_policy_cache_materialization.jsonl".format(safe_name)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return summary_json, summary_md, jsonl_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_policy_cache_materialization")
    parser.add_argument("--target-depths", default="3")
    parser.add_argument("--children-per-node", type=int, default=32)
    parser.add_argument("--frontier-limit", type=int, default=200)
    parser.add_argument("--targets-per-depth", type=int, default=4)
    parser.add_argument("--max-ancestor-nodes", type=int, default=1000)
    parser.add_argument("--max-ancestor-edges", type=int, default=6000)
    parser.add_argument("--root-distance-cap", type=int, default=32)
    parser.add_argument("--cache-slots", type=int, default=128)
    parser.add_argument("--admit-l-min", type=int, default=6)
    parser.add_argument("--admit-l-max", type=int, default=12)
    parser.add_argument("--sweep-cache-slots", default="128")
    parser.add_argument("--sweep-admit-l-min", default=None)
    parser.add_argument("--sweep-admit-l-max", default="8,12")
    parser.add_argument("--boundary-budgets", default="1,2,3,4")
    parser.add_argument("--path-cap", type=int, default=10000)
    parser.add_argument("--expansion-cap", type=int, default=50000)
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--seed", default="policy-cache-materialization-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    records, summaries = run_probe(args)
    summary_json, summary_md, jsonl_path = write_outputs(
        records,
        summaries,
        args.output_dir,
        args.graph_name,
        write_jsonl=args.write_jsonl,
    )
    print(markdown_summary(summaries), end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    if jsonl_path is not None:
        print("jsonl={}".format(jsonl_path))


if __name__ == "__main__":
    main()
