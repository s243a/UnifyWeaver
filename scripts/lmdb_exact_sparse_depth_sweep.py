#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Sweep exact sparse parent histograms by child depth on an LMDB graph.

This probe answers a narrower question than the boundary-cache benchmarks:
given root-reachable targets at increasing child-depths, how expensive is it to
keep the parent-path histogram exact and sparse?

For each sampled target and path-length budget it compares:

* full simple-path DFS enumeration, which is exact under the budget; and
* shifted-parent recurrence, which is exact on acyclic cones and reports cycle
  approximation when a node-only recurrence cannot preserve simple-path state.

The summary treats a point cap such as 50 as an upper bound, not a cost paid by
every histogram.  The observed support, tail-pruned support, path mass, DFS
expansions, recurrence states, and state-based break-even hits are reported by
target child-depth and path-length budget.
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

from scripts.distribution_fit_comparison import (  # noqa: E402
    effective_support_bins,
    exact_excess_distribution,
    histogram_bytes,
    tail_pruning_summary,
)
from scripts.lmdb_parent_branching_diagnostic import (  # noqa: E402
    LmdbCategoryGraph,
    parse_int_list,
    root_distances,
    select_targets_by_child_depth,
)
from scripts.lmdb_parent_histogram_benchmark import (  # noqa: E402
    bounded_parent_histogram,
    percentile,
    safe_graph_name,
)
from scripts.parent_histogram_recurrence import recurrence_parent_histogram  # noqa: E402


DEFAULT_TARGET_DEPTHS = "1,2,3,4"
DEFAULT_BUDGETS = "10,20"


def mean(values):
    values = [float(value) for value in values if value is not None]
    return 0.0 if not values else statistics.fmean(values)


def max_or_zero(values):
    values = [value for value in values if value is not None]
    return 0 if not values else max(values)


def safe_ratio(numerator, denominator):
    denominator = float(denominator or 0.0)
    if denominator <= 0.0:
        return None
    return float(numerator or 0.0) / denominator


def histogram_metrics(hist, tail_epsilon, prune_thresholds):
    if not hist:
        return {
            "reachable": False,
            "path_count": 0,
            "L_min": None,
            "L_max": None,
            "support_bins": 0,
            "support_width": None,
            "effective_support_bins": 0,
            "tail_pruning": {},
            "histogram_bytes": 0,
        }
    probabilities, origin = exact_excess_distribution(hist)
    l_min = min(hist)
    l_max = max(hist)
    return {
        "reachable": True,
        "path_count": sum(int(value) for value in hist.values()),
        "L_min": l_min,
        "L_max": l_max,
        "support_bins": len(hist),
        "support_width": l_max - l_min,
        "effective_support_bins": effective_support_bins(probabilities, tail_epsilon),
        "tail_pruning": tail_pruning_summary(probabilities, prune_thresholds, origin or 0),
        "histogram_bytes": histogram_bytes(hist),
    }


def break_even_hits(build_states, uncached_states, eval_bins):
    saved_states = max(0.0, float(uncached_states or 0) - float(eval_bins or 0))
    if saved_states <= 0.0:
        return None
    return float(build_states or 0) / saved_states


def select_root_reachable_targets(graph, root, target_depths, args):
    targets, child_depth_by_node, selection_counts = select_targets_by_child_depth(
        graph,
        root,
        target_depths,
        args.children_per_node,
        args.frontier_limit,
        args.targets_per_depth,
        args.seed,
    )
    distance_memo = {}
    kept = []
    filtered = []
    for target in targets:
        distances = root_distances(target, root, graph.parents, args.max_parent_depth, distance_memo)
        row = {
            "target_node": target,
            "child_sample_depth": child_depth_by_node[target],
            "L_min": distances["L_min"],
            "L_max": distances["L_max"],
            "distance_truncated": distances["truncated"],
            "distance_cycle_skipped": distances["cycle_skipped"],
        }
        if distances["L_min"] is not None and not distances["truncated"]:
            kept.append(row)
        else:
            filtered.append(row)
    return kept, filtered, selection_counts


def comparison_record(args, target_row, budget, dfs_hist, dfs_stats, dfs_time_ns, recurrence_hist, recurrence_stats, recurrence_time_ns):
    dfs_metrics = histogram_metrics(dfs_hist, args.tail_epsilon, args.prune_thresholds)
    recurrence_metrics = histogram_metrics(recurrence_hist, args.tail_epsilon, args.prune_thresholds)
    exact_match = dfs_hist == recurrence_hist
    support_bins = dfs_metrics["support_bins"]
    effective_bins = dfs_metrics["effective_support_bins"]
    return {
        "record_type": "exact_sparse_depth_sweep_row",
        "graph": args.graph_name,
        "root": args.root,
        "target_node": target_row["target_node"],
        "child_sample_depth": target_row["child_sample_depth"],
        "budget": int(budget),
        "target_L_min": target_row["L_min"],
        "target_L_max": target_row["L_max"],
        "target_distance_cycle_skipped": target_row["distance_cycle_skipped"],
        "dfs_histogram": dfs_hist,
        "recurrence_histogram": recurrence_hist,
        "exact_match": exact_match,
        "reachable": dfs_metrics["reachable"],
        "path_count": dfs_metrics["path_count"],
        "L_min": dfs_metrics["L_min"],
        "L_max": dfs_metrics["L_max"],
        "support_bins": support_bins,
        "support_width": dfs_metrics["support_width"],
        "effective_support_bins": effective_bins,
        "tail_pruning": dfs_metrics["tail_pruning"],
        "exact_histogram_bytes": dfs_metrics["histogram_bytes"],
        "recurrence_path_count": recurrence_metrics["path_count"],
        "recurrence_support_bins": recurrence_metrics["support_bins"],
        "dfs_nodes_expanded": dfs_stats.nodes_expanded,
        "dfs_edges_examined": dfs_stats.edges_examined,
        "dfs_cycle_skips": dfs_stats.cycle_skips,
        "dfs_budget_cutoffs": dfs_stats.budget_cutoffs,
        "dfs_path_cap_hit": dfs_stats.path_cap_hit,
        "dfs_expansion_cap_hit": dfs_stats.expansion_cap_hit,
        "recurrence_states_evaluated": recurrence_stats.states_evaluated,
        "recurrence_memo_hits": recurrence_stats.memo_hits,
        "recurrence_edges_examined": recurrence_stats.edges_examined,
        "recurrence_cycle_edges": recurrence_stats.cycle_edges,
        "recurrence_cycle_approximation": recurrence_stats.cycle_approximation,
        "recurrence_path_cap_hit": recurrence_stats.path_cap_hit,
        "recurrence_expansion_cap_hit": recurrence_stats.expansion_cap_hit,
        "state_expansion_ratio": safe_ratio(recurrence_stats.states_evaluated, dfs_stats.nodes_expanded),
        "time_ratio": safe_ratio(recurrence_time_ns, dfs_time_ns),
        "dfs_time_ns": dfs_time_ns,
        "recurrence_time_ns": recurrence_time_ns,
        "exact_sparse_under_point_cap": exact_match and effective_bins <= args.point_cap,
        "hits_to_break_even_states": break_even_hits(
            recurrence_stats.states_evaluated,
            dfs_stats.nodes_expanded,
            max(1, effective_bins),
        ),
    }


def summary_row(key, rows, point_cap):
    depth, budget = key
    reachable = [row for row in rows if row["reachable"]]
    exact_sparse = [
        row for row in reachable
        if row["exact_sparse_under_point_cap"]
        and not row["dfs_path_cap_hit"]
        and not row["dfs_expansion_cap_hit"]
        and not row["recurrence_path_cap_hit"]
        and not row["recurrence_expansion_cap_hit"]
        and not row["recurrence_cycle_approximation"]
    ]
    break_even = [
        row["hits_to_break_even_states"]
        for row in reachable
        if row["hits_to_break_even_states"] is not None
    ]
    return {
        "record_type": "exact_sparse_depth_sweep_summary_bucket",
        "child_sample_depth": depth,
        "budget": budget,
        "rows": len(rows),
        "reachable_rows": len(reachable),
        "exact_match_rows": sum(1 for row in reachable if row["exact_match"]),
        "exact_sparse_under_point_cap_rows": len(exact_sparse),
        "cycle_approximation_rows": sum(1 for row in reachable if row["recurrence_cycle_approximation"]),
        "dfs_capped_rows": sum(1 for row in reachable if row["dfs_path_cap_hit"] or row["dfs_expansion_cap_hit"]),
        "recurrence_capped_rows": sum(1 for row in reachable if row["recurrence_path_cap_hit"] or row["recurrence_expansion_cap_hit"]),
        "mean_path_count": mean(row["path_count"] for row in reachable),
        "p95_path_count": percentile([row["path_count"] for row in reachable], 95),
        "max_path_count": max_or_zero(row["path_count"] for row in reachable),
        "mean_support_bins": mean(row["support_bins"] for row in reachable),
        "p95_support_bins": percentile([row["support_bins"] for row in reachable], 95),
        "max_support_bins": max_or_zero(row["support_bins"] for row in reachable),
        "mean_effective_support_bins": mean(row["effective_support_bins"] for row in reachable),
        "p95_effective_support_bins": percentile([row["effective_support_bins"] for row in reachable], 95),
        "max_effective_support_bins": max_or_zero(row["effective_support_bins"] for row in reachable),
        "point_cap": point_cap,
        "pct_effective_bins_le_point_cap": 0.0 if not reachable else 100.0 * sum(1 for row in reachable if row["effective_support_bins"] <= point_cap) / len(reachable),
        "mean_exact_histogram_bytes": mean(row["exact_histogram_bytes"] for row in reachable),
        "mean_dfs_nodes_expanded": mean(row["dfs_nodes_expanded"] for row in reachable),
        "mean_recurrence_states_evaluated": mean(row["recurrence_states_evaluated"] for row in reachable),
        "mean_state_expansion_ratio": mean(row["state_expansion_ratio"] for row in reachable),
        "mean_time_ratio": mean(row["time_ratio"] for row in reachable),
        "mean_hits_to_break_even_states": mean(break_even),
        "p95_hits_to_break_even_states": percentile(break_even, 95),
    }


def build_summary(records, args):
    rows = [row for row in records if row.get("record_type") == "exact_sparse_depth_sweep_row"]
    by_key = {}
    for row in rows:
        by_key.setdefault((row["child_sample_depth"], row["budget"]), []).append(row)
    buckets = [summary_row(key, by_key[key], args.point_cap) for key in sorted(by_key)]
    selection = next(row for row in records if row.get("record_type") == "exact_sparse_depth_sweep_selection")
    return {
        "record_type": "exact_sparse_depth_sweep_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph": args.graph_name,
        "root": args.root,
        "point_cap": args.point_cap,
        "tail_epsilon": args.tail_epsilon,
        "selection": selection,
        "buckets": buckets,
    }


def markdown_summary(summary):
    selection = summary["selection"]
    lines = [
        "# LMDB Exact Sparse Depth Sweep",
        "",
        "Graph: `{}`".format(summary["graph"]),
        "",
        "Root: `{}`".format(summary["root"]),
        "",
        "Point cap: `{}`".format(summary["point_cap"]),
        "",
        "Tail epsilon: `{}`".format(summary["tail_epsilon"]),
        "",
        "## Selection",
        "",
        "| child_depth | sampled_frontier_nodes | selected_targets | root_reachable_targets | filtered_targets |",
        "|------------:|-----------------------:|-----------------:|-----------------------:|-----------------:|",
    ]
    selected_by_depth = Counter(row["child_sample_depth"] for row in selection["selected_targets"])
    kept_by_depth = Counter(row["child_sample_depth"] for row in selection["root_reachable_targets"])
    filtered_by_depth = Counter(row["child_sample_depth"] for row in selection["filtered_targets"])
    for depth in sorted(set(selection["selection_counts"]) | set(selected_by_depth) | set(kept_by_depth) | set(filtered_by_depth), key=int):
        depth_int = int(depth)
        lines.append("| {depth} | {frontier} | {selected} | {kept} | {filtered} |".format(
            depth=depth_int,
            frontier=selection["selection_counts"].get(str(depth), selection["selection_counts"].get(depth, 0)),
            selected=selected_by_depth.get(depth_int, 0),
            kept=kept_by_depth.get(depth_int, 0),
            filtered=filtered_by_depth.get(depth_int, 0),
        ))
    lines.extend([
        "",
        "## Depth And Budget Buckets",
        "",
        "| child_depth | budget | rows | exact_sparse | exact_matches | mean_paths | max_paths | mean_eff_bins | max_eff_bins | pct_eff_bins_le_cap | mean_dfs_nodes | mean_rec_states | mean_state_ratio | mean_time_ratio | mean_break_even_hits |",
        "|------------:|-------:|-----:|-------------:|--------------:|-----------:|----------:|--------------:|-------------:|--------------------:|---------------:|----------------:|-----------------:|----------------:|---------------------:|",
    ])
    for row in summary["buckets"]:
        lines.append(
            "| {depth} | {budget} | {rows} | {exact_sparse} | {exact} | {mean_paths:.3f} | {max_paths} | {mean_bins:.3f} | {max_bins} | {pct_bins:.3f} | {dfs_nodes:.3f} | {rec_states:.3f} | {state_ratio:.3f} | {time_ratio:.3f} | {break_even:.3f} |".format(
                depth=row["child_sample_depth"],
                budget=row["budget"],
                rows=row["rows"],
                exact_sparse=row["exact_sparse_under_point_cap_rows"],
                exact=row["exact_match_rows"],
                mean_paths=row["mean_path_count"],
                max_paths=row["max_path_count"],
                mean_bins=row["mean_effective_support_bins"],
                max_bins=row["max_effective_support_bins"],
                pct_bins=row["pct_effective_bins_le_point_cap"],
                dfs_nodes=row["mean_dfs_nodes_expanded"],
                rec_states=row["mean_recurrence_states_evaluated"],
                state_ratio=row["mean_state_expansion_ratio"],
                time_ratio=row["mean_time_ratio"],
                break_even=row["mean_hits_to_break_even_states"],
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `exact_sparse` counts reachable, uncapped rows where recurrence matched DFS, no cycle approximation was needed, and the effective support stayed under the point cap.",
        "- `mean_break_even_hits` is state based: recurrence build states divided by saved states per cached hit, using effective bins as the cached evaluation cost.  It is a planning estimate, not a wall-clock guarantee.",
        "- If mean effective bins stay far below the point cap, the cap is not the paid storage cost; exact sparse histograms remain the first representation to consider.",
    ])
    return "\n".join(lines) + "\n"


def run_sweep(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        target_depths = parse_int_list(args.target_depths)
        budgets = parse_int_list(args.budgets)
        root_reachable_targets, filtered_targets, selection_counts = select_root_reachable_targets(
            graph,
            args.root,
            target_depths,
            args,
        )
        records = [{
            "record_type": "exact_sparse_depth_sweep_selection",
            "graph": args.graph_name,
            "root": args.root,
            "target_depths": target_depths,
            "budgets": budgets,
            "selection_counts": selection_counts,
            "selected_targets": root_reachable_targets + filtered_targets,
            "root_reachable_targets": root_reachable_targets,
            "filtered_targets": filtered_targets,
            "children_per_node": args.children_per_node,
            "frontier_limit": args.frontier_limit,
            "targets_per_depth": args.targets_per_depth,
            "max_parent_depth": args.max_parent_depth,
            "path_cap": args.path_cap,
            "expansion_cap": args.expansion_cap,
        }]
        for target_row in root_reachable_targets:
            for budget in budgets:
                started = time.perf_counter_ns()
                dfs_hist, dfs_stats = bounded_parent_histogram(
                    graph.parents,
                    target_row["target_node"],
                    args.root,
                    budget,
                    args.path_cap,
                    args.expansion_cap,
                )
                dfs_time_ns = time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                recurrence_hist, recurrence_stats = recurrence_parent_histogram(
                    graph.parents,
                    target_row["target_node"],
                    args.root,
                    budget,
                    args.path_cap,
                    args.expansion_cap,
                )
                recurrence_time_ns = time.perf_counter_ns() - started
                records.append(comparison_record(
                    args,
                    target_row,
                    budget,
                    dfs_hist,
                    dfs_stats,
                    dfs_time_ns,
                    recurrence_hist,
                    recurrence_stats,
                    recurrence_time_ns,
                ))
        summary = build_summary(records, args)
        return records, summary
    finally:
        graph.close()


def write_outputs(records, summary, output_dir, graph_name, write_jsonl=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "lmdb_exact_sparse_depth_sweep_{}_{}.json".format(safe_name, timestamp)
    summary_md = output_dir / "lmdb_exact_sparse_depth_sweep_{}_{}.md".format(safe_name, timestamp)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_summary(summary), encoding="utf-8")
    jsonl_path = None
    if write_jsonl:
        jsonl_path = output_dir / "lmdb_exact_sparse_depth_sweep_{}_{}.jsonl".format(safe_name, timestamp)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return summary_json, summary_md, jsonl_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", type=int, required=True, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_exact_sparse_depth_sweep", help="Graph label used in output filenames.")
    parser.add_argument("--target-depths", default=DEFAULT_TARGET_DEPTHS, help="Child depths to sample, e.g. `1,2,3,4`.")
    parser.add_argument("--children-per-node", type=int, default=50000, help="Child sample cap per frontier node.")
    parser.add_argument("--frontier-limit", type=int, default=50000, help="Depth frontier sample cap.")
    parser.add_argument("--targets-per-depth", type=int, default=12, help="Targets sampled per requested child depth.")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS, help="Parent path-length budgets, e.g. `10,20`.")
    parser.add_argument("--max-parent-depth", type=int, default=48, help="Root-reachable filtering horizon.")
    parser.add_argument("--path-cap", type=int, default=100000, help="Maximum root-reaching paths before capping one row.")
    parser.add_argument("--expansion-cap", type=int, default=250000, help="Maximum DFS/recurrence states before capping one row.")
    parser.add_argument("--point-cap", type=int, default=50, help="Exact sparse support cap used for classification.")
    parser.add_argument("--tail-epsilon", type=float, default=0.01, help="Allowed CDF tail mass for effective support bins.")
    parser.add_argument("--prune-thresholds", default="0.01,0.001,0.0001", help="Tail pruning thresholds recorded per row.")
    parser.add_argument("--seed", default="simplewiki-exact-sparse-depth-sweep", help="Deterministic sampling seed.")
    parser.add_argument("--write-jsonl", action="store_true", help="Also write full row-level JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"), help="Output directory.")
    args = parser.parse_args(argv)
    args.prune_thresholds = [float(value) for value in args.prune_thresholds.split(",") if value.strip()]
    if args.children_per_node <= 0:
        raise SystemExit("--children-per-node must be positive")
    if args.frontier_limit <= 0:
        raise SystemExit("--frontier-limit must be positive")
    if args.targets_per_depth <= 0:
        raise SystemExit("--targets-per-depth must be positive")
    if args.max_parent_depth <= 0:
        raise SystemExit("--max-parent-depth must be positive")
    if args.point_cap <= 0:
        raise SystemExit("--point-cap must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    records, summary = run_sweep(args)
    summary_json, summary_md, jsonl_path = write_outputs(records, summary, args.output_dir, args.graph_name, args.write_jsonl)
    print(markdown_summary(summary), end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    if jsonl_path is not None:
        print("jsonl={}".format(jsonl_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
