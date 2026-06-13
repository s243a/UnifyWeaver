#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Benchmark bounded parent-path histograms on numeric-keyed category LMDBs.

The unbounded distribution-cache helpers assume a DAG.  Enwiki category graphs
can contain cycles, so this benchmark uses an explicit finite path budget and a
per-path visited set.  The resulting histogram is exact for simple parent paths
up to the requested budget unless a caller-specified expansion or path cap is
hit.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distribution_fit_comparison import (
    DEFAULT_PRUNE_THRESHOLDS,
    append_packed_exact_table,
    append_representation_selection_table,
    cheapest_candidate_within,
    choose_distribution_representation,
    compare_models,
    distribution_moments,
    distribution_skewness,
    effective_support_bins,
    exact_excess_distribution,
    histogram_bytes,
    packed_exact_candidates,
    parametric_candidate_from_model,
    parametric_state_bytes_estimate,
    representation_policy_candidates,
    realized_model_builders,
    tail_pruning_summary,
)
from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    parse_int_list,
    select_targets_by_child_depth,
    size_biased_branching,
)


DEFAULT_BUDGETS = [4, 6, 8, 10, 12]


@dataclass
class HistogramStats:
    nodes_expanded: int = 0
    edges_examined: int = 0
    cycle_skips: int = 0
    budget_cutoffs: int = 0
    path_cap_hit: bool = False
    expansion_cap_hit: bool = False


def parse_float_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def bounded_parent_histogram(parents_func, target, root, budget, path_cap=None, expansion_cap=None):
    """Count simple parent paths from target to root with length <= budget."""
    hist = Counter()
    stats = HistogramStats()

    def dfs(node, remaining, depth, visited):
        if expansion_cap is not None and stats.nodes_expanded >= expansion_cap:
            stats.expansion_cap_hit = True
            return
        stats.nodes_expanded += 1
        if node == root:
            hist[depth] += 1
            if path_cap is not None and sum(hist.values()) >= path_cap:
                stats.path_cap_hit = True
            return
        if remaining <= 0:
            stats.budget_cutoffs += 1
            return
        for parent in parents_func(node):
            stats.edges_examined += 1
            if parent in visited:
                stats.cycle_skips += 1
                continue
            if stats.path_cap_hit or stats.expansion_cap_hit:
                return
            visited.add(parent)
            dfs(parent, remaining - 1, depth + 1, visited)
            visited.remove(parent)
            if stats.path_cap_hit or stats.expansion_cap_hit:
                return

    dfs(target, budget, 0, {target})
    return dict(sorted(hist.items())), stats


def parent_degree_record(graph, root, target, parents, budget):
    root_reaching = 0
    for parent in parents:
        hist, _stats = bounded_parent_histogram(graph.parents, parent, root, max(0, budget - 1), path_cap=1)
        if hist:
            root_reaching += 1
    return len(parents), root_reaching


def target_budget_records(
    graph_name,
    graph,
    root,
    target,
    child_depth,
    budget,
    tail_epsilon,
    prune_thresholds,
    path_cap,
    expansion_cap,
):
    started = time.perf_counter_ns()
    hist, stats = bounded_parent_histogram(graph.parents, target, root, budget, path_cap, expansion_cap)
    elapsed_ns = time.perf_counter_ns() - started
    parents = graph.parents(target)
    full_parent_degree, root_reaching_parent_degree = parent_degree_record(graph, root, target, parents, budget)
    base = {
        "record_type": "lmdb_parent_histogram",
        "graph": graph_name,
        "root": root,
        "target_node": target,
        "child_sample_depth": child_depth,
        "budget": budget,
        "histogram": hist,
        "path_count": sum(hist.values()),
        "nodes_expanded": stats.nodes_expanded,
        "edges_examined": stats.edges_examined,
        "cycle_skips": stats.cycle_skips,
        "budget_cutoffs": stats.budget_cutoffs,
        "path_cap_hit": stats.path_cap_hit,
        "expansion_cap_hit": stats.expansion_cap_hit,
        "histogram_time_ns": elapsed_ns,
        "full_parent_degree": full_parent_degree,
        "root_reaching_parent_degree": root_reaching_parent_degree,
    }
    empirical, origin = exact_excess_distribution(hist)
    if not empirical or origin is None:
        base.update({
            "reachable": False,
            "L_min": None,
            "L_max": None,
            "support_width": None,
            "support_bins": 0,
            "effective_support_bins": 0,
        })
        return [base]

    mean, variance = distribution_moments(empirical)
    base.update({
        "reachable": True,
        "L_min": origin,
        "L_max": origin + len(empirical) - 1,
        "support_width": len(empirical) - 1,
        "support_bins": len(empirical),
        "effective_support_bins": effective_support_bins(empirical, tail_epsilon),
        "tail_pruning": tail_pruning_summary(empirical, prune_thresholds, origin),
        "mean_excess": mean,
        "variance_excess": variance,
        "skewness_excess": distribution_skewness(empirical),
        "exact_histogram_bytes": histogram_bytes(hist),
    })

    records = [base]
    packed_candidates = packed_exact_candidates(empirical, prune_thresholds)
    best_packed_cdf = cheapest_candidate_within(packed_candidates, tail_epsilon)
    for model_record in compare_models(empirical, realized_model_builders()):
        parametric_bytes = parametric_state_bytes_estimate(model_record["fit_params"])
        policy_candidates = representation_policy_candidates(
            base["exact_histogram_bytes"],
            packed_candidates,
            parametric_candidate_from_model(model_record, parametric_bytes),
        )
        prefix_selection = choose_distribution_representation(policy_candidates, tail_epsilon, workload="prefix_mass")
        functional_selection = choose_distribution_representation(policy_candidates, tail_epsilon, workload="arbitrary_functional")
        model = dict(base)
        model.update({
            "record_type": "lmdb_parent_histogram_fit",
            "packed_exact_candidates": packed_candidates,
            "best_packed_exact_cdf": best_packed_cdf,
            "parametric_state_bytes_estimate": parametric_bytes,
            "representation_policy_candidates": policy_candidates,
            "selected_prefix_representation": prefix_selection["selected_representation"],
            "selected_prefix_policy": prefix_selection,
            "selected_functional_representation": functional_selection["selected_representation"],
            "selected_functional_policy": functional_selection,
            "selected_distribution_representation": prefix_selection["selected_representation"],
            "selected_distribution_reason": prefix_selection["selected_reason"],
            **model_record,
        })
        records.append(model)
    return records


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        child_depths = parse_int_list(args.child_depths)
        budgets = parse_int_list(args.budgets)
        prune_thresholds = parse_float_list(args.prune_thresholds)
        targets, target_child_depth, selection_counts = select_targets_by_child_depth(
            graph,
            args.root,
            child_depths,
            args.children_per_node,
            args.frontier_limit,
            args.targets_per_depth,
            args.seed,
        )
        records = [{
            "record_type": "lmdb_parent_histogram_selection",
            "graph": args.graph_name,
            "root": args.root,
            "selection_counts": selection_counts,
            "targets": len(targets),
            "budgets": budgets,
        }]
        for target in targets:
            for budget in budgets:
                records.extend(
                    target_budget_records(
                        args.graph_name,
                        graph,
                        args.root,
                        target,
                        target_child_depth[target],
                        budget,
                        args.tail_epsilon,
                        prune_thresholds,
                        args.path_cap,
                        args.expansion_cap,
                    )
                )
        return records, summarize(records)
    finally:
        graph.close()


def unique_histogram_rows(records):
    return [row for row in records if row.get("record_type") == "lmdb_parent_histogram"]


def summarize(records):
    selection = next((row for row in records if row.get("record_type") == "lmdb_parent_histogram_selection"), {})
    histogram_rows = unique_histogram_rows(records)
    fit_rows = [row for row in records if row.get("record_type") == "lmdb_parent_histogram_fit"]
    reachable = [row for row in histogram_rows if row.get("reachable")]
    capped = [row for row in histogram_rows if row.get("path_cap_hit") or row.get("expansion_cap_hit")]
    lines = [
        "# LMDB Parent Histogram Benchmark",
        "",
        "Graph: `{}`".format(selection.get("graph", "")),
        "",
        "Root: `{}`".format(selection.get("root", "")),
        "",
        "## Selection",
        "",
        "| child_depth | sampled_frontier_nodes |",
        "|-------------|------------------------|",
    ]
    for depth, count in sorted(selection.get("selection_counts", {}).items()):
        lines.append("| {} | {} |".format(depth, count))
    lines.extend([
        "",
        "## Summary",
        "",
        "| target_budget_rows | reachable_rows | capped_rows |",
        "|--------------------|----------------|-------------|",
        "| {} | {} | {} |".format(len(histogram_rows), len(reachable), len(capped)),
        "",
        "## Histogram Cost By Budget",
        "",
        "| budget | rows | reachable | mean_paths | p95_paths | max_paths | mean_bins | p95_bins | max_bins | mean_nodes_expanded | mean_cycle_skips | capped_rows |",
        "|--------|------|-----------|------------|-----------|-----------|-----------|----------|----------|---------------------|------------------|-------------|",
    ])
    by_budget = {}
    for row in histogram_rows:
        by_budget.setdefault(row["budget"], []).append(row)
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        reachable_rows = [row for row in rows if row.get("reachable")]
        path_counts = [int(row["path_count"]) for row in reachable_rows]
        bins = [int(row["support_bins"]) for row in reachable_rows]
        lines.append(
            "| {budget} | {rows} | {reachable} | {mean_paths:.3f} | {p95_paths:.3f} | {max_paths} | {mean_bins:.3f} | {p95_bins:.3f} | {max_bins} | {mean_expanded:.1f} | {mean_cycles:.1f} | {capped} |".format(
                budget=budget,
                rows=len(rows),
                reachable=len(reachable_rows),
                mean_paths=statistics.mean(path_counts) if path_counts else 0.0,
                p95_paths=percentile(path_counts, 95),
                max_paths=max(path_counts, default=0),
                mean_bins=statistics.mean(bins) if bins else 0.0,
                p95_bins=percentile(bins, 95),
                max_bins=max(bins, default=0),
                mean_expanded=statistics.mean(int(row["nodes_expanded"]) for row in rows) if rows else 0.0,
                mean_cycles=statistics.mean(int(row["cycle_skips"]) for row in rows) if rows else 0.0,
                capped=sum(1 for row in rows if row.get("path_cap_hit") or row.get("expansion_cap_hit")),
            )
        )
    if reachable:
        lines.extend([
            "",
            "## Parent Degree Signal For Reachable Rows",
            "",
            "| budget | rows | mean_full_p | b_full | mean_root_p | b_root |",
            "|--------|------|-------------|--------|-------------|--------|",
        ])
        for budget in sorted(by_budget):
            rows = [row for row in by_budget[budget] if row.get("reachable")]
            full = size_biased_branching([row["full_parent_degree"] for row in rows])
            root_reaching = size_biased_branching([row["root_reaching_parent_degree"] for row in rows])
            lines.append(
                "| {budget} | {rows} | {mean_full:.3f} | {b_full} | {mean_root:.3f} | {b_root} |".format(
                    budget=budget,
                    rows=len(rows),
                    mean_full=full["mean_parent_degree"],
                    b_full=format_optional(full["size_biased_parent_branching"]),
                    mean_root=root_reaching["mean_parent_degree"],
                    b_root=format_optional(root_reaching["size_biased_parent_branching"]),
                )
            )
    if fit_rows:
        lines.extend([
            "",
            "## Fit Error By Model",
            "",
            "| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error |",
            "|-------|------|---------|--------|--------|----------------|",
        ])
        by_model = {}
        for row in fit_rows:
            by_model.setdefault(row["model"], []).append(row)
        for model in sorted(by_model):
            rows = by_model[model]
            l1 = [float(row["l1_error"]) for row in rows]
            cdf = [float(row["max_cdf_error"]) for row in rows]
            lines.append(
                "| {model} | {rows} | {mean_l1:.6f} | {p95_l1:.6f} | {max_l1:.6f} | {mean_cdf:.6f} |".format(
                    model=model,
                    rows=len(rows),
                    mean_l1=statistics.mean(l1) if l1 else 0.0,
                    p95_l1=percentile(l1, 95),
                    max_l1=max(l1, default=0.0),
                    mean_cdf=statistics.mean(cdf) if cdf else 0.0,
                )
            )
        lines.extend([
            "",
            "## Packed Exact Candidate Selection",
            "",
            "| representation | rows | mean_bytes | mean_cdf | mean_w1 |",
            "|----------------|------|------------|----------|---------|",
        ])
        append_packed_exact_table(lines, fit_rows)
        lines.extend([
            "",
            "## Representation Policy Selection",
            "",
        ])
        append_representation_selection_table(lines, fit_rows)
        lines.extend([
            "",
            "## Representation Policy By Budget And Depth",
            "",
        ])
        append_budget_depth_policy_table(lines, fit_rows)

    return "\n".join(lines) + "\n"


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def compact_counts(values):
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return ", ".join("{}:{}".format(key, counts[key]) for key in sorted(counts))


def append_budget_depth_policy_table(lines, fit_rows):
    lines.extend([
        "| child_depth | budget | workload | model_rows | parametric_cdf_pass | selected_counts | mean_bins | capped_hist_rows |",
        "|------------:|-------:|----------|-----------:|--------------------:|-----------------|----------:|-----------------:|",
    ])
    by_group = {}
    for row in fit_rows:
        group = (row.get("child_sample_depth"), row.get("budget"))
        by_group.setdefault(group, []).append(row)
    for child_depth, budget in sorted(by_group):
        rows = by_group[(child_depth, budget)]
        unique_histogram_keys = {
            (row.get("target_node"), row.get("budget"))
            for row in rows
        }
        capped_histogram_keys = {
            (row.get("target_node"), row.get("budget"))
            for row in rows
            if row.get("path_cap_hit") or row.get("expansion_cap_hit")
        }
        mean_bins = statistics.mean(float(row.get("support_bins", 0)) for row in rows) if rows else 0.0
        for workload, policy_key in [
            ("prefix_mass", "selected_prefix_policy"),
            ("arbitrary_functional", "selected_functional_policy"),
        ]:
            selected = [
                row.get(policy_key, {}).get("selected_representation") or "none"
                for row in rows
            ]
            parametric_pass = 0
            for row in rows:
                policy = row.get(policy_key, {})
                max_cdf = policy.get("policy_max_cdf_error")
                if max_cdf is not None and float(row.get("max_cdf_error", 1.0)) <= float(max_cdf):
                    parametric_pass += 1
            lines.append(
                "| {child_depth} | {budget} | {workload} | {model_rows} | {param_pass} | {selected_counts} | {mean_bins:.3f} | {capped} |".format(
                    child_depth=child_depth,
                    budget=budget,
                    workload=workload,
                    model_rows=len(rows),
                    param_pass=parametric_pass,
                    selected_counts=compact_counts(selected),
                    mean_bins=mean_bins,
                    capped=len(capped_histogram_keys),
                )
            )
        if not unique_histogram_keys:
            lines.append(
                "| {child_depth} | {budget} | none | 0 | 0 | none:0 | 0.000 | 0 |".format(
                    child_depth=child_depth,
                    budget=budget,
                )
            )


def format_optional(value):
    return "n/a" if value is None else "{:.6f}".format(float(value))


def safe_graph_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("_") or "graph"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / "lmdb_parent_histogram_benchmark_{}_{}.jsonl".format(safe_name, timestamp)
    summary_path = output_dir / "lmdb_parent_histogram_benchmark_summary_{}_{}.md".format(safe_name, timestamp)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", required=True, type=int, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_parent_histogram", help="Graph label used in output filenames.")
    parser.add_argument("--child-depths", default="2,3,4", help="Child-sampling depths to target.")
    parser.add_argument("--children-per-node", type=int, default=128, help="Deterministic sample cap for children per frontier node.")
    parser.add_argument("--frontier-limit", type=int, default=5000, help="Deterministic cap for each sampled child-depth frontier.")
    parser.add_argument("--targets-per-depth", type=int, default=50, help="Deterministic target cap per requested child depth.")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)), help="Comma-separated parent path budgets.")
    parser.add_argument("--tail-epsilon", type=float, default=0.001, help="Tail mass allowed outside effective support.")
    parser.add_argument("--prune-thresholds", default=",".join(map(str, DEFAULT_PRUNE_THRESHOLDS)), help="Comma-separated tail pruning thresholds.")
    parser.add_argument("--path-cap", type=int, default=100000, help="Stop a target-budget row after this many root paths.")
    parser.add_argument("--expansion-cap", type=int, default=250000, help="Stop a target-budget row after this many expanded nodes.")
    parser.add_argument("--seed", default="0", help="Deterministic sampling seed.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    args = parser.parse_args(argv)

    records, summary = run_benchmark(args)
    if args.output_dir:
        jsonl_path, summary_path = write_outputs(records, summary, args.output_dir, args.graph_name)
        print(summary, end="")
        print("jsonl={}".format(jsonl_path))
        print("summary={}".format(summary_path))
    else:
        print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
