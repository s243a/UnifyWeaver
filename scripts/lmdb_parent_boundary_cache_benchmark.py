#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Compare bounded parent-path search with a histogram boundary cache.

The cache mode precomputes histograms for selected lower-depth boundary nodes.
During a later target search, it stops at those boundary nodes and splices the
cached histogram into the current path length.  This is exact on acyclic cones.
On cyclic cones with simple-path semantics it is an approximation, because the
cached suffix histogram does not know the current visited set.  The benchmark
therefore reports histogram error against the full simple-path search.
"""

from __future__ import annotations

import argparse
import json
import math
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

from scripts.distribution_fit_comparison import binomial_pmf, exact_excess_distribution, l1_error, max_cdf_error
from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    parse_int_list,
    root_distances,
    select_targets_by_child_depth,
)
from scripts.lmdb_depth_planning_prior_probe import (
    cache_admission_policy,
    histogram_storage_bytes,
    planning_prior_for_bucket,
)
from scripts.lmdb_parent_histogram_benchmark import (
    HistogramStats,
    bounded_parent_histogram,
    percentile,
    safe_graph_name,
)


DEFAULT_BUDGETS = [6, 8]


@dataclass
class CachedSearchStats:
    nodes_expanded: int = 0
    edges_examined: int = 0
    cycle_skips: int = 0
    budget_cutoffs: int = 0
    cache_hits: int = 0
    histogram_cache_hits: int = 0
    parametric_cache_hits: int = 0
    cache_bins_spliced: int = 0
    histogram_bins_spliced: int = 0
    parametric_bins_spliced: int = 0
    path_cap_hit: bool = False
    expansion_cap_hit: bool = False


def scaled_distribution_histogram(probabilities, origin, total_count):
    """Convert a compact probability vector into deterministic integer mass."""
    if origin is None or total_count <= 0:
        return {}
    weighted = [(index, max(0.0, float(probability)) * int(total_count)) for index, probability in enumerate(probabilities)]
    floors = {origin + index: int(value) for index, value in weighted if int(value) > 0}
    remainder = int(total_count) - sum(floors.values())
    if remainder > 0 and weighted:
        ranked = sorted(
            weighted,
            key=lambda item: (item[1] - int(item[1]), item[1], -item[0]),
            reverse=True,
        )
        for index, _value in ranked[:remainder]:
            floors[origin + index] = floors.get(origin + index, 0) + 1
    return dict(sorted((length, count) for length, count in floors.items() if count > 0))


def estimate_parametric_total_count(row, prior, mass_model, mass_cap=None):
    """Estimate unnormalized mass for a parametric boundary histogram."""
    oracle_count = int(row.get("path_count", 0))
    mass_cap = None if mass_cap is None or int(mass_cap) <= 0 else int(mass_cap)
    capped = False

    if mass_model == "oracle":
        estimated = oracle_count
    elif mass_model == "unit":
        estimated = 1 if oracle_count > 0 else 0
    elif mass_model == "depth-prior":
        lmax = row.get("histogram_L_max")
        depth = 0 if lmax is None else max(0, int(lmax))
        branching_pressure = max(0.0, 1.0 + float(prior.get("base_mean_excess", 0.0)))
        if oracle_count <= 0:
            estimated = 0
        elif branching_pressure <= 1.0 or depth == 0:
            estimated = 1
        else:
            log_estimate = depth * math.log(branching_pressure)
            if mass_cap is not None and log_estimate >= math.log(mass_cap):
                estimated = mass_cap
                capped = True
            else:
                estimated = max(1, int(round(math.exp(log_estimate))))
    else:
        raise ValueError("unknown parametric mass model: {}".format(mass_model))

    if mass_cap is not None and estimated > mass_cap:
        estimated = mass_cap
        capped = True

    return {
        "mass_model": mass_model,
        "estimated_path_count": estimated,
        "oracle_path_count": oracle_count,
        "mass_delta": estimated - oracle_count,
        "mass_ratio": None if oracle_count <= 0 else estimated / oracle_count,
        "mass_capped": capped,
    }


def parametric_shape_distribution(row, prior, shape_model):
    """Return a compact probability vector and origin for a boundary state."""
    origin = row.get("histogram_L_min")
    if origin is None:
        return [], None
    if shape_model == "empirical-prior":
        return prior["prior_distribution"], int(origin)
    if shape_model in {"support-binomial", "support-binomial-midpoint"}:
        lmax = row.get("histogram_L_max")
        width = 0 if lmax is None else max(0, int(lmax) - int(origin))
        if shape_model == "support-binomial-midpoint":
            mean_excess = width / 2.0
        else:
            mean_excess = max(0.0, float(prior.get("prior_mean_excess", 0.0)))
        probability = 0.0 if width <= 0 else max(0.0, min(1.0, mean_excess / width))
        return binomial_pmf(width, probability), int(origin)
    raise ValueError("unknown parametric shape model: {}".format(shape_model))


def add_shifted_hist(out, suffix_hist, prefix_depth, remaining):
    added = 0
    for suffix_length, count in suffix_hist.items():
        if suffix_length <= remaining:
            out[prefix_depth + suffix_length] += count
            added += 1
    return added


def cached_parent_histogram(
    parents_func,
    target,
    root,
    budget,
    boundary_cache,
    path_cap=None,
    expansion_cap=None,
    parametric_boundary_cache=None,
):
    parametric_boundary_cache = parametric_boundary_cache or {}
    hist = Counter()
    stats = CachedSearchStats()

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
        if node in boundary_cache:
            stats.cache_hits += 1
            stats.histogram_cache_hits += 1
            added = add_shifted_hist(hist, boundary_cache[node], depth, remaining)
            stats.cache_bins_spliced += added
            stats.histogram_bins_spliced += added
            if path_cap is not None and sum(hist.values()) >= path_cap:
                stats.path_cap_hit = True
            return
        if node in parametric_boundary_cache:
            stats.cache_hits += 1
            stats.parametric_cache_hits += 1
            added = add_shifted_hist(hist, parametric_boundary_cache[node], depth, remaining)
            stats.cache_bins_spliced += added
            stats.parametric_bins_spliced += added
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


def histogram_distribution_error(full_hist, cached_hist):
    full_dist, _full_origin = exact_excess_distribution(full_hist)
    cached_dist, _cached_origin = exact_excess_distribution(cached_hist)
    if not full_dist and not cached_dist:
        return 0.0, 0.0
    if not full_dist or not cached_dist:
        return 1.0, 1.0
    # The distributions are compared after shifting to their own minimum path
    # length.  A separate L_min mismatch is reported in each row.
    return l1_error(full_dist, cached_dist), max_cdf_error(full_dist, cached_dist)


def root_reaching_parent_degree(graph, root, node, max_parent_depth, distance_memo):
    def distances(candidate):
        return root_distances(candidate, root, graph.parents, max_parent_depth, distance_memo)

    return sum(1 for parent in graph.parents(node) if distances(parent)["L_min"] is not None)


def build_boundary_cache(
    graph,
    root,
    boundary_nodes,
    boundary_budget,
    path_cap,
    expansion_cap,
    admission_policy="baseline",
    safety_factor=1.25,
    max_histogram_bytes=1024,
    parametric_bytes=64,
    parametric_shape_model="empirical-prior",
    parametric_mass_model="oracle",
    parametric_mass_cap=1000000,
    tail_epsilon=0.001,
    max_parent_depth=24,
):
    cache = {}
    parametric_cache = {}
    rows = []
    distance_memo = {}
    for node in boundary_nodes:
        started = time.perf_counter_ns()
        hist, stats = bounded_parent_histogram(graph.parents, node, root, boundary_budget, path_cap, expansion_cap)
        elapsed = time.perf_counter_ns() - started
        rows.append({
            "record_type": "boundary_cache_entry",
            "node": node,
            "cached": False,
            "histogram": hist,
            "path_count": sum(hist.values()),
            "support_bins": len(hist),
            "histogram_L_min": min(hist) if hist else None,
            "histogram_L_max": max(hist) if hist else None,
            "histogram_bytes": histogram_storage_bytes(hist),
            "root_reaching_parent_degree": root_reaching_parent_degree(
                graph,
                root,
                node,
                max_parent_depth,
                distance_memo,
            ),
            "nodes_expanded": stats.nodes_expanded,
            "edges_examined": stats.edges_examined,
            "cycle_skips": stats.cycle_skips,
            "path_cap_hit": stats.path_cap_hit,
            "expansion_cap_hit": stats.expansion_cap_hit,
            "histogram_time_ns": elapsed,
        })

    priors = {}
    if admission_policy == "depth-prior":
        degrees_by_lmax = {}
        for row in rows:
            lmax = row["histogram_L_max"]
            degree = row["root_reaching_parent_degree"]
            if lmax is not None and degree > 0:
                degrees_by_lmax.setdefault(int(lmax), []).append(int(degree))
        for lmax, degrees in degrees_by_lmax.items():
            priors[lmax] = planning_prior_for_bucket(degrees, lmax, tail_epsilon)

    for row in rows:
        hist = row["histogram"]
        lmax = row["histogram_L_max"]
        if admission_policy == "baseline":
            cached = bool(hist) and not row["path_cap_hit"] and not row["expansion_cap_hit"]
            action = "materialize_exact" if cached else "skip_cache"
            reason = "baseline_uncapped_histogram" if cached else "baseline_empty_or_capped"
            policy = {
                "action": action,
                "reason": reason,
                "safety_prediction_bytes": None,
                "observed_or_predicted_bytes": row["histogram_bytes"],
                "max_histogram_bytes": None,
                "parametric_bytes": None,
            }
            predicted_prior_bytes = None
            prior_effective_bins = None
        elif lmax in priors and hist:
            prior = priors[lmax]
            predicted_prior_bytes = int(prior["prior_pruned_histogram_bytes"])
            prior_effective_bins = int(prior["prior_effective_bins"])
            policy = cache_admission_policy(
                predicted_prior_bytes,
                safety_factor,
                max_histogram_bytes,
                recurrence_capped=row["path_cap_hit"] or row["expansion_cap_hit"],
                recurrence_cycle_approximation=row["cycle_skips"] > 0,
                realized_histogram_bytes=row["histogram_bytes"],
                parametric_bytes=parametric_bytes,
            )
            cached = policy["action"] in {"materialize_exact", "materialize_capped"}
        else:
            predicted_prior_bytes = None
            prior_effective_bins = None
            policy = {
                "action": "skip_cache",
                "reason": "no_planning_prior_or_histogram",
                "safety_prediction_bytes": None,
                "observed_or_predicted_bytes": row["histogram_bytes"],
                "max_histogram_bytes": max_histogram_bytes,
                "parametric_bytes": parametric_bytes,
            }
            cached = False

        row["admission_policy"] = admission_policy
        row["predicted_prior_bytes"] = predicted_prior_bytes
        row["prior_effective_bins"] = prior_effective_bins
        row["cache_admission_action"] = policy["action"]
        row["cache_admission_reason"] = policy["reason"]
        row["cache_admission_safety_prediction_bytes"] = policy["safety_prediction_bytes"]
        row["cache_admission_observed_or_predicted_bytes"] = policy["observed_or_predicted_bytes"]
        row["cache_admission_max_histogram_bytes"] = policy["max_histogram_bytes"]
        row["cache_admission_parametric_bytes"] = policy["parametric_bytes"]
        row["cached"] = cached
        row["parametric_cached"] = False
        row["parametric_histogram"] = {}
        row["parametric_path_count"] = 0
        row["parametric_oracle_path_count"] = row["path_count"]
        row["parametric_shape_model"] = parametric_shape_model
        row["parametric_mass_model"] = parametric_mass_model
        row["parametric_mass_delta"] = None
        row["parametric_mass_ratio"] = None
        row["parametric_mass_capped"] = False
        row["parametric_support_bins"] = 0
        row["parametric_support_min"] = None
        row["parametric_support_max"] = None
        if cached:
            cache[row["node"]] = hist
        elif policy["action"] == "use_parametric_prior" and lmax in priors and hist:
            mass_estimate = estimate_parametric_total_count(
                row,
                priors[lmax],
                parametric_mass_model,
                parametric_mass_cap,
            )
            probabilities, origin = parametric_shape_distribution(
                row,
                priors[lmax],
                parametric_shape_model,
            )
            approx_hist = scaled_distribution_histogram(
                probabilities,
                origin,
                mass_estimate["estimated_path_count"],
            )
            if approx_hist:
                parametric_cache[row["node"]] = approx_hist
                row["parametric_cached"] = True
                row["parametric_histogram"] = approx_hist
                row["parametric_path_count"] = sum(approx_hist.values())
                row["parametric_oracle_path_count"] = mass_estimate["oracle_path_count"]
                row["parametric_mass_delta"] = mass_estimate["mass_delta"]
                row["parametric_mass_ratio"] = mass_estimate["mass_ratio"]
                row["parametric_mass_capped"] = mass_estimate["mass_capped"]
                row["parametric_support_bins"] = len(approx_hist)
                row["parametric_support_min"] = min(approx_hist)
                row["parametric_support_max"] = max(approx_hist)
    return cache, parametric_cache, rows


def comparison_record(graph_name, root, target, child_depth, budget, full_hist, full_stats, full_time, cached_hist, cached_stats, cached_time):
    l1, cdf = histogram_distribution_error(full_hist, cached_hist)
    full_paths = sum(full_hist.values())
    cached_paths = sum(cached_hist.values())
    return {
        "record_type": "boundary_cache_comparison",
        "graph": graph_name,
        "root": root,
        "target_node": target,
        "child_sample_depth": child_depth,
        "budget": budget,
        "full_histogram": full_hist,
        "cached_histogram": cached_hist,
        "full_path_count": full_paths,
        "cached_path_count": cached_paths,
        "path_count_delta": cached_paths - full_paths,
        "abs_path_count_delta": abs(cached_paths - full_paths),
        "path_count_relative_error": 0.0 if full_paths == 0 else abs(cached_paths - full_paths) / full_paths,
        "full_L_min": min(full_hist) if full_hist else None,
        "cached_L_min": min(cached_hist) if cached_hist else None,
        "full_L_max": max(full_hist) if full_hist else None,
        "cached_L_max": max(cached_hist) if cached_hist else None,
        "l1_error": l1,
        "max_cdf_error": cdf,
        "full_nodes_expanded": full_stats.nodes_expanded,
        "cached_nodes_expanded": cached_stats.nodes_expanded,
        "node_expansion_ratio": 0.0 if full_stats.nodes_expanded == 0 else cached_stats.nodes_expanded / full_stats.nodes_expanded,
        "full_edges_examined": full_stats.edges_examined,
        "cached_edges_examined": cached_stats.edges_examined,
        "full_cycle_skips": full_stats.cycle_skips,
        "cached_cycle_skips": cached_stats.cycle_skips,
        "cache_hits": cached_stats.cache_hits,
        "histogram_cache_hits": cached_stats.histogram_cache_hits,
        "parametric_cache_hits": cached_stats.parametric_cache_hits,
        "cache_bins_spliced": cached_stats.cache_bins_spliced,
        "histogram_bins_spliced": cached_stats.histogram_bins_spliced,
        "parametric_bins_spliced": cached_stats.parametric_bins_spliced,
        "full_path_cap_hit": full_stats.path_cap_hit,
        "full_expansion_cap_hit": full_stats.expansion_cap_hit,
        "cached_path_cap_hit": cached_stats.path_cap_hit,
        "cached_expansion_cap_hit": cached_stats.expansion_cap_hit,
        "full_time_ns": full_time,
        "cached_time_ns": cached_time,
    }


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        budgets = parse_int_list(args.budgets)
        boundary_depths = parse_int_list(args.boundary_depths)
        target_depths = parse_int_list(args.target_depths)
        boundary_nodes, _boundary_depth_by_node, boundary_counts = select_targets_by_child_depth(
            graph,
            args.root,
            boundary_depths,
            args.children_per_node,
            args.frontier_limit,
            args.boundaries_per_depth,
            args.seed + ":boundary",
        )
        targets, target_depth_by_node, target_counts = select_targets_by_child_depth(
            graph,
            args.root,
            target_depths,
            args.children_per_node,
            args.frontier_limit,
            args.targets_per_depth,
            args.seed + ":target",
        )
        cache, parametric_cache, cache_rows = build_boundary_cache(
            graph,
            args.root,
            boundary_nodes,
            args.boundary_budget,
            args.path_cap,
            args.expansion_cap,
            args.admission_policy,
            args.safety_factor,
            args.max_histogram_bytes,
            args.parametric_bytes,
            args.parametric_shape_model,
            args.parametric_mass_model,
            args.parametric_mass_cap,
            args.tail_epsilon,
            args.max_parent_depth,
        )
        records = [{
            "record_type": "boundary_cache_selection",
            "graph": args.graph_name,
            "root": args.root,
            "boundary_counts": boundary_counts,
            "target_counts": target_counts,
            "boundary_nodes": len(boundary_nodes),
            "cached_boundary_nodes": len(cache),
            "parametric_boundary_nodes": len(parametric_cache),
            "targets": len(targets),
            "budgets": budgets,
            "boundary_budget": args.boundary_budget,
            "admission_policy": args.admission_policy,
            "safety_factor": args.safety_factor,
            "max_histogram_bytes": args.max_histogram_bytes,
            "parametric_bytes": args.parametric_bytes,
            "parametric_shape_model": args.parametric_shape_model,
            "parametric_mass_model": args.parametric_mass_model,
            "parametric_mass_cap": args.parametric_mass_cap,
        }]
        records.extend(cache_rows)
        for target in targets:
            for budget in budgets:
                full_started = time.perf_counter_ns()
                full_hist, full_stats = bounded_parent_histogram(
                    graph.parents,
                    target,
                    args.root,
                    budget,
                    args.path_cap,
                    args.expansion_cap,
                )
                full_time = time.perf_counter_ns() - full_started
                cached_started = time.perf_counter_ns()
                cached_hist, cached_stats = cached_parent_histogram(
                    graph.parents,
                    target,
                    args.root,
                    budget,
                    cache,
                    args.path_cap,
                    args.expansion_cap,
                    parametric_cache,
                )
                cached_time = time.perf_counter_ns() - cached_started
                records.append(
                    comparison_record(
                        args.graph_name,
                        args.root,
                        target,
                        target_depth_by_node[target],
                        budget,
                        full_hist,
                        full_stats,
                        full_time,
                        cached_hist,
                        cached_stats,
                        cached_time,
                    )
                )
        return records, summarize(records)
    finally:
        graph.close()


def summarize(records):
    selection = next((row for row in records if row.get("record_type") == "boundary_cache_selection"), {})
    cache_rows = [row for row in records if row.get("record_type") == "boundary_cache_entry"]
    comparison_rows = [row for row in records if row.get("record_type") == "boundary_cache_comparison"]
    lines = [
        "# LMDB Boundary Cache Benchmark",
        "",
        "Graph: `{}`".format(selection.get("graph", "")),
        "",
        "Root: `{}`".format(selection.get("root", "")),
        "",
        "## Selection",
        "",
        "| role | child_depth | sampled_frontier_nodes |",
        "|------|-------------|------------------------|",
    ]
    for depth, count in sorted(selection.get("boundary_counts", {}).items()):
        lines.append("| boundary | {} | {} |".format(depth, count))
    for depth, count in sorted(selection.get("target_counts", {}).items()):
        lines.append("| target | {} | {} |".format(depth, count))
    lines.extend([
        "",
        "| boundary_nodes | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget |",
        "|----------------|-----------------------|---------------------------|---------|-----------------|",
        "| {} | {} | {} | {} | {} |".format(
            selection.get("boundary_nodes", 0),
            selection.get("cached_boundary_nodes", 0),
            selection.get("parametric_boundary_nodes", 0),
            selection.get("targets", 0),
            selection.get("boundary_budget", 0),
        ),
        "",
        "## Admission Policy",
        "",
        "| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mass_model | parametric_mass_cap |",
        "|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|--------------------:|",
        "| {} | {} | {} | {} | {} | {} | {} |".format(
            selection.get("admission_policy", "baseline"),
            selection.get("safety_factor", "n/a"),
            selection.get("max_histogram_bytes", "n/a"),
            selection.get("parametric_bytes", "n/a"),
            selection.get("parametric_shape_model", "empirical-prior"),
            selection.get("parametric_mass_model", "oracle"),
            selection.get("parametric_mass_cap", "n/a"),
        ),
        "",
        "## Boundary Admission Outcomes",
        "",
        "| action | rows | histogram_cached | parametric_cached |",
        "|--------|-----:|-----------------:|------------------:|",
    ])
    action_counts = {}
    cached_by_action = {}
    parametric_by_action = {}
    for row in cache_rows:
        action = row.get("cache_admission_action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        if row.get("cached"):
            cached_by_action[action] = cached_by_action.get(action, 0) + 1
        if row.get("parametric_cached"):
            parametric_by_action[action] = parametric_by_action.get(action, 0) + 1
    for action in sorted(action_counts):
        lines.append("| {} | {} | {} | {} |".format(
            action,
            action_counts[action],
            cached_by_action.get(action, 0),
            parametric_by_action.get(action, 0),
        ))
    lines.extend([
        "",
        "## Boundary Admission Reasons",
        "",
        "| reason | rows |",
        "|--------|-----:|",
    ])
    reason_counts = {}
    for row in cache_rows:
        reason = row.get("cache_admission_reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason in sorted(reason_counts):
        lines.append("| {} | {} |".format(reason, reason_counts[reason]))
    lines.extend([
        "",
        "## Boundary Cache Build",
        "",
        "| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |",
        "|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|",
    ])
    cached_entries = [row for row in cache_rows if row.get("cached")]
    parametric_entries = [row for row in cache_rows if row.get("parametric_cached")]
    lines.append(
        "| {entries} | {cached} | {parametric} | {mean_paths:.3f} | {mean_bins:.3f} | {mean_parametric_paths:.3f} | {mean_parametric_mass_ratio:.3f} | {mean_parametric_bins:.3f} | {mean_expanded:.1f} | {capped} |".format(
            entries=len(cache_rows),
            cached=len(cached_entries),
            parametric=len(parametric_entries),
            mean_paths=statistics.mean(int(row["path_count"]) for row in cached_entries) if cached_entries else 0.0,
            mean_bins=statistics.mean(int(row["support_bins"]) for row in cached_entries) if cached_entries else 0.0,
            mean_parametric_paths=statistics.mean(int(row["parametric_path_count"]) for row in parametric_entries) if parametric_entries else 0.0,
            mean_parametric_mass_ratio=statistics.mean(float(row["parametric_mass_ratio"]) for row in parametric_entries if row.get("parametric_mass_ratio") is not None) if any(row.get("parametric_mass_ratio") is not None for row in parametric_entries) else 0.0,
            mean_parametric_bins=statistics.mean(int(row["parametric_support_bins"]) for row in parametric_entries) if parametric_entries else 0.0,
            mean_expanded=statistics.mean(int(row["nodes_expanded"]) for row in cache_rows) if cache_rows else 0.0,
            capped=sum(1 for row in cache_rows if row.get("path_cap_hit") or row.get("expansion_cap_hit")),
        )
    )
    lines.extend([
        "",
        "## Full Search Versus Boundary Cache",
        "",
        "| budget | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | full_capped | cached_capped |",
        "|--------|------|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|---------------:|----------------:|-----------------------:|------------------------:|-------------|---------------|",
    ])
    by_budget = {}
    for row in comparison_rows:
        by_budget.setdefault(row["budget"], []).append(row)
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        l1 = [float(row["l1_error"]) for row in rows]
        cdf = [float(row["max_cdf_error"]) for row in rows]
        path_relative = [float(row["path_count_relative_error"]) for row in rows]
        path_delta = [int(row["abs_path_count_delta"]) for row in rows]
        ratios = [float(row["node_expansion_ratio"]) for row in rows]
        histogram_hits = [int(row["histogram_cache_hits"]) for row in rows]
        parametric_hits = [int(row["parametric_cache_hits"]) for row in rows]
        histogram_bins_spliced = [int(row["histogram_bins_spliced"]) for row in rows]
        parametric_bins_spliced = [int(row["parametric_bins_spliced"]) for row in rows]
        lines.append(
            "| {budget} | {rows} | {mean_l1:.6f} | {p95_l1:.6f} | {max_l1:.6f} | {mean_cdf:.6f} | {mean_path_relative:.6f} | {mean_path_delta:.3f} | {mean_ratio:.3f} | {mean_hist_hits:.3f} | {mean_param_hits:.3f} | {mean_hist_bins:.3f} | {mean_param_bins:.3f} | {full_capped} | {cached_capped} |".format(
                budget=budget,
                rows=len(rows),
                mean_l1=statistics.mean(l1) if l1 else 0.0,
                p95_l1=percentile(l1, 95),
                max_l1=max(l1, default=0.0),
                mean_cdf=statistics.mean(cdf) if cdf else 0.0,
                mean_path_relative=statistics.mean(path_relative) if path_relative else 0.0,
                mean_path_delta=statistics.mean(path_delta) if path_delta else 0.0,
                mean_ratio=statistics.mean(ratios) if ratios else 0.0,
                mean_hist_hits=statistics.mean(histogram_hits) if histogram_hits else 0.0,
                mean_param_hits=statistics.mean(parametric_hits) if parametric_hits else 0.0,
                mean_hist_bins=statistics.mean(histogram_bins_spliced) if histogram_bins_spliced else 0.0,
                mean_param_bins=statistics.mean(parametric_bins_spliced) if parametric_bins_spliced else 0.0,
                full_capped=sum(1 for row in rows if row.get("full_path_cap_hit") or row.get("full_expansion_cap_hit")),
                cached_capped=sum(1 for row in rows if row.get("cached_path_cap_hit") or row.get("cached_expansion_cap_hit")),
            )
        )
    return "\n".join(lines) + "\n"


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / "lmdb_parent_boundary_cache_benchmark_{}_{}.jsonl".format(safe_name, timestamp)
    summary_path = output_dir / "lmdb_parent_boundary_cache_benchmark_summary_{}_{}.md".format(safe_name, timestamp)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", required=True, type=int, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_boundary_cache", help="Graph label used in output filenames.")
    parser.add_argument("--boundary-depths", default="2", help="Child depths used as cache boundary candidates.")
    parser.add_argument("--target-depths", default="3", help="Child depths used as target candidates.")
    parser.add_argument("--children-per-node", type=int, default=128, help="Deterministic child sample cap per frontier node.")
    parser.add_argument("--frontier-limit", type=int, default=2000, help="Deterministic cap for each sampled child-depth frontier.")
    parser.add_argument("--boundaries-per-depth", type=int, default=100, help="Boundary cache candidates per requested boundary depth.")
    parser.add_argument("--targets-per-depth", type=int, default=30, help="Targets per requested target depth.")
    parser.add_argument("--boundary-budget", type=int, default=6, help="Path budget used to precompute boundary histograms.")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)), help="Comma-separated target path budgets.")
    parser.add_argument("--admission-policy", choices=["baseline", "depth-prior"], default="baseline", help="Policy used to decide whether measured boundary histograms enter the cache.")
    parser.add_argument("--safety-factor", type=float, default=1.25, help="Multiplier for depth-prior predicted histogram bytes.")
    parser.add_argument("--max-histogram-bytes", type=int, default=1024, help="Maximum bytes allowed for exact or capped boundary histograms under depth-prior admission.")
    parser.add_argument("--parametric-bytes", type=int, default=64, help="Estimated bytes for a parametric prior state.")
    parser.add_argument("--parametric-shape-model", choices=["empirical-prior", "support-binomial", "support-binomial-midpoint"], default="empirical-prior", help="Shape used for parametric boundary states.")
    parser.add_argument("--parametric-mass-model", choices=["oracle", "unit", "depth-prior"], default="oracle", help="Mass used to unnormalize parametric boundary states.")
    parser.add_argument("--parametric-mass-cap", type=int, default=1000000, help="Maximum estimated path mass for non-oracle parametric states; non-positive disables the cap.")
    parser.add_argument("--tail-epsilon", type=float, default=0.001, help="Tail epsilon used when estimating depth-prior effective support.")
    parser.add_argument("--max-parent-depth", type=int, default=24, help="Parent depth cap for root-reaching parent degree signals.")
    parser.add_argument("--path-cap", type=int, default=100000, help="Stop a row after this many root paths.")
    parser.add_argument("--expansion-cap", type=int, default=250000, help="Stop a row after this many expanded nodes.")
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
