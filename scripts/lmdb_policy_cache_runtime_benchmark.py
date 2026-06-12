#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Measure bounded parent-histogram runtime using policy-selected cache nodes.

The policy sweep tells us which root-near ancestor nodes would remain resident
under a fixed-size cache.  This benchmark uses those resident nodes as the
boundary-cache set, precomputes bounded histograms for them, then compares full
bounded parent search with cached search on the sampled targets.
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

from scripts.lmdb_ancestor_cache_policy_benchmark import (
    admit_candidate,
    cache_priority,
    candidate_from_distance,
    update_cache,
)
from scripts.lmdb_ancestor_cache_policy_sweep import (
    CacheConfig,
    collect_distances,
    collect_query_inputs,
    parse_grid,
    sweep_configs,
)
from scripts.lmdb_parent_boundary_cache_benchmark import (
    cached_parent_histogram,
    comparison_record,
)
from scripts.lmdb_parent_branching_diagnostic import LmdbCategoryGraph, parse_int_list
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram, percentile, safe_graph_name


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def action_rates(action_counts):
    total = sum(action_counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(action_counts.items())}


def policy_cache_entries(query_inputs, distances, config):
    cache = {}
    action_counts = Counter()
    for query in query_inputs:
        candidates = []
        for node in sorted(query.space.nodes):
            distance = distances[node]
            if admit_candidate(distance, config.admit_l_min, config.admit_l_max):
                candidates.append(candidate_from_distance(node, distance))
        for candidate in sorted(candidates, key=cache_priority):
            action_counts[update_cache(cache, candidate, query.space.nodes, config.cache_slots)] += 1
    return cache, action_counts


def build_policy_boundary_cache(parents_func, root, cache_entries, boundary_budget, path_cap, expansion_cap):
    boundary_cache = {}
    rows = []
    for slot, entry in sorted(cache_entries.items()):
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
        cached = bool(hist) and not stats.path_cap_hit and not stats.expansion_cap_hit
        if cached:
            boundary_cache[entry.node] = hist
        rows.append({
            "record_type": "policy_boundary_cache_entry",
            "slot": slot,
            "node": entry.node,
            "l_min": entry.l_min,
            "l_max": entry.l_max,
            "truncated": entry.truncated,
            "cached": cached,
            "histogram": hist,
            "path_count": sum(hist.values()),
            "support_bins": len(hist),
            "nodes_expanded": stats.nodes_expanded,
            "edges_examined": stats.edges_examined,
            "cycle_skips": stats.cycle_skips,
            "path_cap_hit": stats.path_cap_hit,
            "expansion_cap_hit": stats.expansion_cap_hit,
            "histogram_time_ns": elapsed,
        })
    return boundary_cache, rows


def compare_targets(graph, args, query_inputs, boundary_cache):
    rows = []
    for query in query_inputs:
        for budget in parse_int_list(args.budgets):
            full_started = time.perf_counter_ns()
            full_hist, full_stats = bounded_parent_histogram(
                graph.parents,
                query.target,
                args.root,
                budget,
                args.path_cap,
                args.expansion_cap,
            )
            full_time = time.perf_counter_ns() - full_started
            cached_started = time.perf_counter_ns()
            cached_hist, cached_stats = cached_parent_histogram(
                graph.parents,
                query.target,
                args.root,
                budget,
                boundary_cache,
                args.path_cap,
                args.expansion_cap,
            )
            cached_time = time.perf_counter_ns() - cached_started
            rows.append(comparison_record(
                args.graph_name,
                args.root,
                query.target,
                query.child_depth,
                budget,
                full_hist,
                full_stats,
                full_time,
                cached_hist,
                cached_stats,
                cached_time,
            ))
    return rows


def summarize_runtime(args, target_counts, query_inputs, config, policy_actions, cache_rows, comparison_rows):
    by_budget = {}
    for row in comparison_rows:
        by_budget.setdefault(row["budget"], []).append(row)
    budget_rows = []
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        l1 = [float(row["l1_error"]) for row in rows]
        cdf = [float(row["max_cdf_error"]) for row in rows]
        ratios = [float(row["node_expansion_ratio"]) for row in rows]
        hits = [int(row["cache_hits"]) for row in rows]
        full_times = [int(row["full_time_ns"]) for row in rows]
        cached_times = [int(row["cached_time_ns"]) for row in rows]
        time_ratios = [
            0.0 if int(row["full_time_ns"]) == 0 else int(row["cached_time_ns"]) / int(row["full_time_ns"])
            for row in rows
        ]
        budget_rows.append({
            "budget": budget,
            "rows": len(rows),
            "mean_l1_error": mean(l1),
            "p95_l1_error": percentile(l1, 95),
            "max_l1_error": max(l1, default=0.0),
            "mean_max_cdf_error": mean(cdf),
            "mean_node_expansion_ratio": mean(ratios),
            "mean_time_ratio": mean(time_ratios),
            "mean_full_time_ns": mean(full_times),
            "mean_cached_time_ns": mean(cached_times),
            "mean_cache_hits": mean(hits),
            "full_capped": sum(1 for row in rows if row.get("full_path_cap_hit") or row.get("full_expansion_cap_hit")),
            "cached_capped": sum(1 for row in rows if row.get("cached_path_cap_hit") or row.get("cached_expansion_cap_hit")),
        })
    cached_rows = [row for row in cache_rows if row["cached"]]
    ancestor_nodes = [len(query.space.nodes) for query in query_inputs]
    capped_queries = sum(1 for query in query_inputs if query.space.capped)
    return {
        "record_type": "policy_cache_runtime_summary",
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
        "boundary_budget": args.boundary_budget,
        "path_cap": args.path_cap,
        "expansion_cap": args.expansion_cap,
        "policy_actions": dict(policy_actions),
        "policy_action_rates": action_rates(policy_actions),
        "policy_resident_entries": len(cache_rows),
        "materialized_boundary_entries": len(cached_rows),
        "materialized_boundary_rate": 0.0 if not cache_rows else len(cached_rows) / len(cache_rows),
        "mean_boundary_nodes_expanded": mean(row["nodes_expanded"] for row in cache_rows),
        "boundary_capped_entries": sum(1 for row in cache_rows if row["path_cap_hit"] or row["expansion_cap_hit"]),
        "budget_rows": budget_rows,
    }


def markdown_summary(summaries):
    lines = [
        "# Policy Cache Runtime Benchmark",
        "",
        "| slots | L_min | L_max | boundary_budget | targets | capped_cones | resident | materialized | mean_nodes | budget | mean_l1 | mean_node_ratio | mean_time_ratio | mean_cache_hits | full_capped | cached_capped |",
        "|------:|------:|------:|----------------:|--------:|-------------:|---------:|-------------:|-----------:|-------:|--------:|----------------:|----------------:|----------------:|------------:|--------------:|",
    ]
    for summary in summaries:
        for budget in summary["budget_rows"]:
            lines.append(
                "| {slots} | {l_min} | {l_max} | {boundary_budget} | {targets} | {capped} | {resident} | {materialized} | {mean_nodes:.1f} | {budget} | {mean_l1:.6f} | {node_ratio:.3f} | {time_ratio:.3f} | {hits:.3f} | {full_capped} | {cached_capped} |".format(
                    slots=summary["cache_slots"],
                    l_min=summary["admit_l_min"],
                    l_max=summary["admit_l_max"],
                    boundary_budget=summary["boundary_budget"],
                    targets=summary["targets"],
                    capped=summary["ancestor_capped_queries"],
                    resident=summary["policy_resident_entries"],
                    materialized=summary["materialized_boundary_entries"],
                    mean_nodes=summary["mean_ancestor_nodes"],
                    budget=budget["budget"],
                    mean_l1=budget["mean_l1_error"],
                    node_ratio=budget["mean_node_expansion_ratio"],
                    time_ratio=budget["mean_time_ratio"],
                    hits=budget["mean_cache_hits"],
                    full_capped=budget["full_capped"],
                    cached_capped=budget["cached_capped"],
                )
            )
    return "\n".join(lines) + "\n"


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        _targets, target_counts, query_inputs = collect_query_inputs(graph, args)
        distances = collect_distances(graph, args, query_inputs)
        records = []
        summaries = []
        for config in sweep_configs(args):
            policy_cache, policy_actions = policy_cache_entries(query_inputs, distances, config)
            boundary_cache, cache_rows = build_policy_boundary_cache(
                graph.parents,
                args.root,
                policy_cache,
                args.boundary_budget,
                args.path_cap,
                args.expansion_cap,
            )
            comparison_rows = compare_targets(graph, args, query_inputs, boundary_cache)
            summary = summarize_runtime(args, target_counts, query_inputs, config, policy_actions, cache_rows, comparison_rows)
            summaries.append(summary)
            records.append(summary)
            if args.write_jsonl:
                records.extend(cache_rows)
                records.extend(comparison_rows)
        return records, summaries
    finally:
        graph.close()


def write_outputs(records, summaries, output_dir, graph_name, write_jsonl=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "{}_policy_cache_runtime_summary.json".format(safe_name)
    summary_md = output_dir / "{}_policy_cache_runtime_summary.md".format(safe_name)
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_summary(summaries), encoding="utf-8")
    jsonl_path = None
    if write_jsonl:
        jsonl_path = output_dir / "{}_policy_cache_runtime.jsonl".format(safe_name)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return summary_json, summary_md, jsonl_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_policy_cache_runtime")
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
    parser.add_argument("--sweep-cache-slots", default="128,256")
    parser.add_argument("--sweep-admit-l-min", default=None)
    parser.add_argument("--sweep-admit-l-max", default="8,12")
    parser.add_argument("--boundary-budget", type=int, default=8)
    parser.add_argument("--budgets", default="4,6")
    parser.add_argument("--path-cap", type=int, default=10000)
    parser.add_argument("--expansion-cap", type=int, default=50000)
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--seed", default="policy-cache-runtime-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    records, summaries = run_benchmark(args)
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
