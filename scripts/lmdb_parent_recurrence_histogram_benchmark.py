#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Compare DFS path enumeration with parent-histogram recurrence."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distribution_fit_comparison import exact_excess_distribution, l1_error, max_cdf_error
from scripts.lmdb_parent_branching_diagnostic import LmdbCategoryGraph, parse_int_list, select_targets_by_child_depth
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram, percentile, safe_graph_name
from scripts.parent_histogram_recurrence import recurrence_parent_histogram


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def histogram_error(left_hist, right_hist):
    left, _left_origin = exact_excess_distribution(left_hist)
    right, _right_origin = exact_excess_distribution(right_hist)
    if not left and not right:
        return 0.0, 0.0
    if not left or not right:
        return 1.0, 1.0
    return l1_error(left, right), max_cdf_error(left, right)


def comparison_record(args, target, child_depth, budget, dfs_hist, dfs_stats, dfs_time, recurrence_hist, recurrence_stats, recurrence_time):
    l1, cdf = histogram_error(dfs_hist, recurrence_hist)
    dfs_paths = sum(dfs_hist.values())
    recurrence_paths = sum(recurrence_hist.values())
    return {
        "record_type": "parent_recurrence_histogram_comparison",
        "graph": args.graph_name,
        "root": args.root,
        "target_node": target,
        "child_sample_depth": child_depth,
        "budget": budget,
        "dfs_histogram": dfs_hist,
        "recurrence_histogram": recurrence_hist,
        "dfs_path_count": dfs_paths,
        "recurrence_path_count": recurrence_paths,
        "path_count_delta": recurrence_paths - dfs_paths,
        "l1_error": l1,
        "max_cdf_error": cdf,
        "dfs_nodes_expanded": dfs_stats.nodes_expanded,
        "recurrence_states_evaluated": recurrence_stats.states_evaluated,
        "state_expansion_ratio": 0.0 if dfs_stats.nodes_expanded == 0 else recurrence_stats.states_evaluated / dfs_stats.nodes_expanded,
        "dfs_edges_examined": dfs_stats.edges_examined,
        "recurrence_edges_examined": recurrence_stats.edges_examined,
        "dfs_cycle_skips": dfs_stats.cycle_skips,
        "recurrence_cycle_edges": recurrence_stats.cycle_edges,
        "recurrence_cycle_approximation": recurrence_stats.cycle_approximation,
        "dfs_path_cap_hit": dfs_stats.path_cap_hit,
        "dfs_expansion_cap_hit": dfs_stats.expansion_cap_hit,
        "recurrence_path_cap_hit": recurrence_stats.path_cap_hit,
        "recurrence_expansion_cap_hit": recurrence_stats.expansion_cap_hit,
        "dfs_time_ns": dfs_time,
        "recurrence_time_ns": recurrence_time,
        "time_ratio": 0.0 if dfs_time == 0 else recurrence_time / dfs_time,
    }


def summarize(records):
    rows = [row for row in records if row.get("record_type") == "parent_recurrence_histogram_comparison"]
    by_budget = {}
    for row in rows:
        by_budget.setdefault(row["budget"], []).append(row)
    out = [{
        "record_type": "parent_recurrence_histogram_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph": records[0].get("graph") if records else None,
        "budgets": sorted(by_budget),
        "rows": len(rows),
        "budget_rows": [],
    }]
    for budget in sorted(by_budget):
        bucket = by_budget[budget]
        out[0]["budget_rows"].append({
            "budget": budget,
            "rows": len(bucket),
            "exact_match_rows": sum(1 for row in bucket if row["dfs_histogram"] == row["recurrence_histogram"]),
            "cycle_approximation_rows": sum(1 for row in bucket if row["recurrence_cycle_approximation"]),
            "mean_l1_error": mean(row["l1_error"] for row in bucket),
            "p95_l1_error": percentile([row["l1_error"] for row in bucket], 95),
            "mean_max_cdf_error": mean(row["max_cdf_error"] for row in bucket),
            "mean_state_expansion_ratio": mean(row["state_expansion_ratio"] for row in bucket),
            "mean_time_ratio": mean(row["time_ratio"] for row in bucket),
            "dfs_capped_rows": sum(1 for row in bucket if row["dfs_path_cap_hit"] or row["dfs_expansion_cap_hit"]),
            "recurrence_capped_rows": sum(1 for row in bucket if row["recurrence_path_cap_hit"] or row["recurrence_expansion_cap_hit"]),
        })
    return out[0]


def markdown_summary(summary):
    lines = [
        "# Parent Histogram Recurrence Benchmark",
        "",
        "| budget | rows | exact_matches | cycle_approx | mean_l1 | mean_cdf | mean_state_ratio | mean_time_ratio | dfs_capped | recurrence_capped |",
        "|-------:|-----:|--------------:|-------------:|--------:|---------:|-----------------:|----------------:|-----------:|------------------:|",
    ]
    for row in summary["budget_rows"]:
        lines.append(
            "| {budget} | {rows} | {exact} | {cycle} | {l1:.6f} | {cdf:.6f} | {state_ratio:.3f} | {time_ratio:.3f} | {dfs_capped} | {rec_capped} |".format(
                budget=row["budget"],
                rows=row["rows"],
                exact=row["exact_match_rows"],
                cycle=row["cycle_approximation_rows"],
                l1=row["mean_l1_error"],
                cdf=row["mean_max_cdf_error"],
                state_ratio=row["mean_state_expansion_ratio"],
                time_ratio=row["mean_time_ratio"],
                dfs_capped=row["dfs_capped_rows"],
                rec_capped=row["recurrence_capped_rows"],
            )
        )
    return "\n".join(lines) + "\n"


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        budgets = parse_int_list(args.budgets)
        targets, target_child_depth, selection_counts = select_targets_by_child_depth(
            graph,
            args.root,
            parse_int_list(args.target_depths),
            args.children_per_node,
            args.frontier_limit,
            args.targets_per_depth,
            args.seed,
        )
        records = [{
            "record_type": "parent_recurrence_histogram_selection",
            "graph": args.graph_name,
            "root": args.root,
            "target_selection_counts": selection_counts,
            "targets": len(targets),
            "budgets": budgets,
        }]
        for target in targets:
            for budget in budgets:
                dfs_started = time.perf_counter_ns()
                dfs_hist, dfs_stats = bounded_parent_histogram(
                    graph.parents,
                    target,
                    args.root,
                    budget,
                    args.path_cap,
                    args.expansion_cap,
                )
                dfs_time = time.perf_counter_ns() - dfs_started
                rec_started = time.perf_counter_ns()
                recurrence_hist, recurrence_stats = recurrence_parent_histogram(
                    graph.parents,
                    target,
                    args.root,
                    budget,
                    args.path_cap,
                    args.expansion_cap,
                )
                recurrence_time = time.perf_counter_ns() - rec_started
                records.append(comparison_record(
                    args,
                    target,
                    target_child_depth[target],
                    budget,
                    dfs_hist,
                    dfs_stats,
                    dfs_time,
                    recurrence_hist,
                    recurrence_stats,
                    recurrence_time,
                ))
        return records, summarize(records)
    finally:
        graph.close()


def write_outputs(records, summary, output_dir, graph_name, write_jsonl=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "{}_parent_recurrence_summary.json".format(safe_name)
    summary_md = output_dir / "{}_parent_recurrence_summary.md".format(safe_name)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_summary(summary), encoding="utf-8")
    jsonl_path = None
    if write_jsonl:
        jsonl_path = output_dir / "{}_parent_recurrence.jsonl".format(safe_name)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return summary_json, summary_md, jsonl_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_parent_recurrence")
    parser.add_argument("--target-depths", default="3")
    parser.add_argument("--children-per-node", type=int, default=32)
    parser.add_argument("--frontier-limit", type=int, default=200)
    parser.add_argument("--targets-per-depth", type=int, default=4)
    parser.add_argument("--budgets", default="4,6,8")
    parser.add_argument("--path-cap", type=int, default=10000)
    parser.add_argument("--expansion-cap", type=int, default=50000)
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--seed", default="parent-recurrence-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    records, summary = run_benchmark(args)
    summary_json, summary_md, jsonl_path = write_outputs(records, summary, args.output_dir, args.graph_name, args.write_jsonl)
    print(markdown_summary(summary), end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    if jsonl_path is not None:
        print("jsonl={}".format(jsonl_path))


if __name__ == "__main__":
    main()
