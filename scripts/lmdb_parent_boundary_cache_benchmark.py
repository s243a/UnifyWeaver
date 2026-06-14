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

from scripts.distribution_fit_comparison import binomial_pmf, effective_support_bins, exact_excess_distribution, l1_error, max_cdf_error
from scripts.distribution_serialization import decode_distribution_payload, encode_selected_distribution
from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    deterministic_sample,
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
from scripts.parent_histogram_recurrence import recurrence_parent_histogram


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
    cache_payload_bytes_read: int = 0
    cache_decode_ns: int = 0
    cache_probe_ns: int = 0
    cache_splice_ns: int = 0
    cache_path_cap_check_ns: int = 0
    parent_lookup_ns: int = 0
    path_cap_hit: bool = False
    expansion_cap_hit: bool = False


@dataclass
class BoundaryBuildResult:
    histogram: dict
    nodes_expanded: int
    edges_examined: int
    cycle_skips: int
    path_cap_hit: bool
    expansion_cap_hit: bool
    recurrence_cycle_approximation: bool = False
    recurrence_states_evaluated: int | None = None
    recurrence_memo_hits: int | None = None


@dataclass
class SerializedHistogramCacheEntry:
    payload: bytes
    metadata: dict

    def decode_histogram(self):
        started = time.perf_counter_ns()
        probabilities, decoded_meta = decode_distribution_payload(self.payload)
        elapsed = time.perf_counter_ns() - started
        total_count = int(round(float(decoded_meta.get("total_mass", 0.0))))
        hist = scaled_distribution_histogram(probabilities, int(decoded_meta.get("origin", 0)), total_count)
        return hist, elapsed

    def values(self):
        hist, _elapsed = self.decode_histogram()
        return hist.values()

    def items(self):
        hist, _elapsed = self.decode_histogram()
        return hist.items()

    def __getitem__(self, key):
        hist, _elapsed = self.decode_histogram()
        return hist[key]

    def __iter__(self):
        hist, _elapsed = self.decode_histogram()
        return iter(hist)

    def __len__(self):
        hist, _elapsed = self.decode_histogram()
        return len(hist)

    def __eq__(self, other):
        hist, _elapsed = self.decode_histogram()
        return hist == other


def serialized_histogram_cache_entry(hist, representation="packed_sparse_histogram"):
    probabilities, origin = exact_excess_distribution(hist)
    payload, metadata = encode_selected_distribution(probabilities, representation, origin=0 if origin is None else origin, total_mass=sum(hist.values()))
    return SerializedHistogramCacheEntry(payload=payload, metadata=metadata)


def build_boundary_histogram(parents_func, node, root, budget, path_cap, expansion_cap, boundary_builder):
    """Build one boundary histogram by search or shifted parent recurrence."""
    if boundary_builder == "search":
        hist, stats = bounded_parent_histogram(parents_func, node, root, budget, path_cap, expansion_cap)
        return BoundaryBuildResult(
            histogram=hist,
            nodes_expanded=stats.nodes_expanded,
            edges_examined=stats.edges_examined,
            cycle_skips=stats.cycle_skips,
            path_cap_hit=stats.path_cap_hit,
            expansion_cap_hit=stats.expansion_cap_hit,
        )
    if boundary_builder == "recurrence":
        hist, stats = recurrence_parent_histogram(parents_func, node, root, budget, path_cap, expansion_cap)
        return BoundaryBuildResult(
            histogram=hist,
            nodes_expanded=stats.states_evaluated,
            edges_examined=stats.edges_examined,
            cycle_skips=stats.cycle_edges,
            path_cap_hit=stats.path_cap_hit,
            expansion_cap_hit=stats.expansion_cap_hit,
            recurrence_cycle_approximation=stats.cycle_approximation,
            recurrence_states_evaluated=stats.states_evaluated,
            recurrence_memo_hits=stats.memo_hits,
        )
    raise ValueError("unknown boundary builder: {}".format(boundary_builder))


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


def support_binomial_mean(width, prior_mean_excess, mean_model, mean_blend):
    """Choose a bounded binomial mean over a finite support width."""
    width = max(0, int(width))
    prior_mean = max(0.0, float(prior_mean_excess))
    midpoint = width / 2.0
    if mean_model == "prior-clipped":
        mean = prior_mean
    elif mean_model == "midpoint":
        mean = midpoint
    elif mean_model == "blend":
        alpha = max(0.0, min(1.0, float(mean_blend)))
        mean = alpha * prior_mean + (1.0 - alpha) * midpoint
    else:
        raise ValueError("unknown parametric mean model: {}".format(mean_model))
    return max(0.0, min(float(width), mean))


def parametric_support_interval(row, support_source="measured", boundary_budget=None):
    """Select the support interval used by a parametric boundary state."""
    if support_source == "measured":
        lower = row.get("histogram_L_min")
        upper = row.get("histogram_L_max")
    elif support_source == "distance-bounds":
        lower = row.get("distance_L_min")
        upper = row.get("distance_L_max")
        if upper is not None and boundary_budget is not None:
            upper = min(int(upper), int(boundary_budget))
    else:
        raise ValueError("unknown parametric support source: {}".format(support_source))

    if lower is None or upper is None:
        return {
            "support_source": support_source,
            "support_min": None,
            "support_max": None,
            "support_width": None,
            "support_valid": False,
        }

    lower = int(lower)
    upper = int(upper)
    if upper < lower:
        return {
            "support_source": support_source,
            "support_min": lower,
            "support_max": upper,
            "support_width": None,
            "support_valid": False,
        }

    return {
        "support_source": support_source,
        "support_min": lower,
        "support_max": upper,
        "support_width": upper - lower,
        "support_valid": True,
    }


def parametric_shape_distribution(
    row,
    prior,
    shape_model,
    mean_model="prior-clipped",
    mean_blend=0.5,
    support_source="measured",
    boundary_budget=None,
):
    """Return a compact probability vector and origin for a boundary state."""
    support = parametric_support_interval(row, support_source, boundary_budget)
    origin = support["support_min"]
    if not support["support_valid"]:
        return [], None, {}
    if shape_model == "empirical-prior":
        return prior["prior_distribution"], int(origin), {
            "shape_model": shape_model,
            "mean_model": None,
            "support_width": len(prior["prior_distribution"]) - 1,
            "support_source": support_source,
            "support_min": support["support_min"],
            "support_max": support["support_max"],
            "support_bound_width": support["support_width"],
            "mean_excess": prior.get("prior_mean_excess"),
            "probability": None,
        }
    if shape_model in {"support-binomial", "support-binomial-midpoint"}:
        width = int(support["support_width"])
        selected_mean_model = "midpoint" if shape_model == "support-binomial-midpoint" else mean_model
        if shape_model == "support-binomial-midpoint":
            mean_excess = support_binomial_mean(width, prior.get("prior_mean_excess", 0.0), "midpoint", mean_blend)
        else:
            mean_excess = support_binomial_mean(width, prior.get("prior_mean_excess", 0.0), mean_model, mean_blend)
        probability = 0.0 if width <= 0 else max(0.0, min(1.0, mean_excess / width))
        return binomial_pmf(width, probability), int(origin), {
            "shape_model": shape_model,
            "mean_model": selected_mean_model,
            "support_width": width,
            "support_source": support_source,
            "support_min": support["support_min"],
            "support_max": support["support_max"],
            "support_bound_width": support["support_width"],
            "mean_excess": mean_excess,
            "probability": probability,
        }
    raise ValueError("unknown parametric shape model: {}".format(shape_model))


def add_shifted_hist(out, suffix_hist, prefix_depth, remaining):
    added = 0
    for suffix_length, count in suffix_hist.items():
        if suffix_length <= remaining:
            out[prefix_depth + suffix_length] += count
            added += 1
    return added


def cache_entry_histogram(entry):
    if isinstance(entry, SerializedHistogramCacheEntry):
        hist, decode_ns = entry.decode_histogram()
        return hist, int(entry.metadata.get("payload_bytes", len(entry.payload))), decode_ns
    return entry, 0, 0


def cached_parent_histogram(
    parents_func,
    target,
    root,
    budget,
    boundary_cache,
    path_cap=None,
    expansion_cap=None,
    parametric_boundary_cache=None,
    collect_attribution=False,
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
            if path_cap is not None:
                started = time.perf_counter_ns() if collect_attribution else None
                cap_hit = sum(hist.values()) >= path_cap
                if collect_attribution:
                    stats.cache_path_cap_check_ns += time.perf_counter_ns() - started
                if cap_hit:
                    stats.path_cap_hit = True
            return

        started = time.perf_counter_ns() if collect_attribution else None
        histogram_entry = boundary_cache.get(node)
        parametric_entry = None if histogram_entry is not None else parametric_boundary_cache.get(node)
        if collect_attribution:
            stats.cache_probe_ns += time.perf_counter_ns() - started

        if histogram_entry is not None:
            stats.cache_hits += 1
            stats.histogram_cache_hits += 1
            suffix_hist, payload_bytes, decode_ns = cache_entry_histogram(histogram_entry)
            stats.cache_payload_bytes_read += payload_bytes
            stats.cache_decode_ns += decode_ns
            started = time.perf_counter_ns() if collect_attribution else None
            added = add_shifted_hist(hist, suffix_hist, depth, remaining)
            if collect_attribution:
                stats.cache_splice_ns += time.perf_counter_ns() - started
            stats.cache_bins_spliced += added
            stats.histogram_bins_spliced += added
            if path_cap is not None:
                started = time.perf_counter_ns() if collect_attribution else None
                cap_hit = sum(hist.values()) >= path_cap
                if collect_attribution:
                    stats.cache_path_cap_check_ns += time.perf_counter_ns() - started
                if cap_hit:
                    stats.path_cap_hit = True
            return
        if parametric_entry is not None:
            stats.cache_hits += 1
            stats.parametric_cache_hits += 1
            suffix_hist, payload_bytes, decode_ns = cache_entry_histogram(parametric_entry)
            stats.cache_payload_bytes_read += payload_bytes
            stats.cache_decode_ns += decode_ns
            started = time.perf_counter_ns() if collect_attribution else None
            added = add_shifted_hist(hist, suffix_hist, depth, remaining)
            if collect_attribution:
                stats.cache_splice_ns += time.perf_counter_ns() - started
            stats.cache_bins_spliced += added
            stats.parametric_bins_spliced += added
            if path_cap is not None:
                started = time.perf_counter_ns() if collect_attribution else None
                cap_hit = sum(hist.values()) >= path_cap
                if collect_attribution:
                    stats.cache_path_cap_check_ns += time.perf_counter_ns() - started
                if cap_hit:
                    stats.path_cap_hit = True
            return
        if remaining <= 0:
            stats.budget_cutoffs += 1
            return

        if collect_attribution:
            started = time.perf_counter_ns()
            parents = list(parents_func(node))
            stats.parent_lookup_ns += time.perf_counter_ns() - started
        else:
            parents = parents_func(node)
        for parent in parents:
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


def histogram_effective_bins(hist, tail_epsilon):
    probabilities, _origin = exact_excess_distribution(hist)
    if not probabilities:
        return 0
    return effective_support_bins(probabilities, tail_epsilon)


def recurrence_threshold_decision(row, max_recurrence_states, max_effective_bins_after_trim):
    states_limit = max(0, int(max_recurrence_states or 0))
    bins_limit = max(0, int(max_effective_bins_after_trim or 0))
    state_count = row.get("recurrence_states_evaluated")
    effective_bins = int(row.get("effective_support_bins_after_trim", 0))
    states_over = bool(row.get("boundary_builder") == "recurrence" and states_limit > 0 and state_count is not None and int(state_count) > states_limit)
    bins_over = bool(bins_limit > 0 and effective_bins > bins_limit)
    reasons = []
    if states_over:
        reasons.append("recurrence_states_over_limit")
    if bins_over:
        reasons.append("effective_bins_over_limit")
    return {
        "states_over": states_over,
        "bins_over": bins_over,
        "reason": "+".join(reasons) if reasons else None,
    }


def root_reaching_parent_degree(graph, root, node, max_parent_depth, distance_memo):
    def distances(candidate):
        return root_distances(candidate, root, graph.parents, max_parent_depth, distance_memo)

    return sum(1 for parent in graph.parents(node) if distances(parent)["L_min"] is not None)


def collect_target_ancestor_boundaries(
    parents_func,
    root,
    targets,
    boundary_depths,
    max_hops,
    max_parent_depth,
    limit=None,
    seed="target-ancestor-boundaries",
):
    """Collect target ancestors whose root distance matches a boundary depth."""
    wanted_depths = set(int(depth) for depth in boundary_depths)
    if not wanted_depths or max_hops <= 0:
        return []
    distance_memo = {}
    found = []
    found_set = set()

    def distances(node):
        return root_distances(node, root, parents_func, max_parent_depth, distance_memo)

    def visit(node, remaining, visited):
        if remaining <= 0:
            return
        for parent in parents_func(node):
            if parent in visited:
                continue
            parent_distances = distances(parent)
            parent_depth = parent_distances["L_min"]
            if parent_depth in wanted_depths and parent not in found_set:
                found_set.add(parent)
                found.append(parent)
            if parent == root:
                continue
            visited.add(parent)
            visit(parent, remaining - 1, visited)
            visited.remove(parent)

    for target in targets:
        visit(target, int(max_hops), {target})

    return deterministic_sample(
        found,
        None if limit is None or int(limit) <= 0 else int(limit),
        seed,
    )


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
    parametric_mean_model="prior-clipped",
    parametric_mean_blend=0.5,
    parametric_support_source="measured",
    parametric_mass_model="oracle",
    parametric_mass_cap=1000000,
    tail_epsilon=0.001,
    max_parent_depth=24,
    boundary_builder="search",
    max_recurrence_states=0,
    max_effective_bins_after_trim=0,
):
    cache = {}
    parametric_cache = {}
    rows = []
    distance_memo = {}
    for node in boundary_nodes:
        started = time.perf_counter_ns()
        built = build_boundary_histogram(graph.parents, node, root, boundary_budget, path_cap, expansion_cap, boundary_builder)
        elapsed = time.perf_counter_ns() - started
        hist = built.histogram
        distances = root_distances(node, root, graph.parents, max_parent_depth, distance_memo)
        rows.append({
            "record_type": "boundary_cache_entry",
            "node": node,
            "cached": False,
            "histogram": hist,
            "path_count": sum(hist.values()),
            "support_bins": len(hist),
            "effective_support_bins_after_trim": histogram_effective_bins(hist, tail_epsilon),
            "histogram_L_min": min(hist) if hist else None,
            "histogram_L_max": max(hist) if hist else None,
            "histogram_bytes": histogram_storage_bytes(hist),
            "distance_L_min": distances["L_min"],
            "distance_L_max": distances["L_max"],
            "distance_truncated": distances["truncated"],
            "distance_cycle_skipped": distances["cycle_skipped"],
            "root_reaching_parent_degree": root_reaching_parent_degree(
                graph,
                root,
                node,
                max_parent_depth,
                distance_memo,
            ),
            "boundary_builder": boundary_builder,
            "nodes_expanded": built.nodes_expanded,
            "edges_examined": built.edges_examined,
            "cycle_skips": built.cycle_skips,
            "path_cap_hit": built.path_cap_hit,
            "expansion_cap_hit": built.expansion_cap_hit,
            "histogram_time_ns": elapsed,
            "recurrence_cycle_approximation": built.recurrence_cycle_approximation,
            "recurrence_states_evaluated": built.recurrence_states_evaluated,
            "recurrence_memo_hits": built.recurrence_memo_hits,
        })

    priors = {}
    if admission_policy == "depth-prior" or max_recurrence_states or max_effective_bins_after_trim:
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
            if cached and row["recurrence_cycle_approximation"]:
                action = "materialize_capped"
                reason = "baseline_recurrence_cycle_approximation"
            else:
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
                recurrence_cycle_approximation=row["recurrence_cycle_approximation"],
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

        threshold = recurrence_threshold_decision(row, max_recurrence_states, max_effective_bins_after_trim)
        threshold_forces_parametric = bool(threshold["reason"] and hist and lmax in priors)
        parametric_representation_fits = int(parametric_bytes) > 0 and (int(max_histogram_bytes) <= 0 or int(parametric_bytes) <= int(max_histogram_bytes))
        if threshold_forces_parametric and parametric_representation_fits:
            policy = {
                "action": "use_parametric_prior",
                "reason": threshold["reason"],
                "safety_prediction_bytes": policy["safety_prediction_bytes"],
                "observed_or_predicted_bytes": policy["observed_or_predicted_bytes"],
                "max_histogram_bytes": policy["max_histogram_bytes"],
                "parametric_bytes": parametric_bytes,
            }
            cached = False

        row["admission_policy"] = admission_policy
        row["max_recurrence_states"] = max_recurrence_states
        row["max_effective_bins_after_trim"] = max_effective_bins_after_trim
        row["recurrence_states_over_limit"] = threshold["states_over"]
        row["effective_bins_over_limit"] = threshold["bins_over"]
        row["approximation_forced_by_threshold"] = bool(policy["action"] == "use_parametric_prior" and threshold["reason"])
        row["approximation_threshold_reason"] = threshold["reason"]
        row["predicted_prior_bytes"] = predicted_prior_bytes
        row["prior_effective_bins"] = prior_effective_bins
        row["cache_admission_action"] = policy["action"]
        row["cache_admission_reason"] = policy["reason"]
        row["cache_admission_safety_prediction_bytes"] = policy["safety_prediction_bytes"]
        row["cache_admission_observed_or_predicted_bytes"] = policy["observed_or_predicted_bytes"]
        row["cache_admission_max_histogram_bytes"] = policy["max_histogram_bytes"]
        row["cache_admission_parametric_bytes"] = policy["parametric_bytes"]
        row["cached"] = cached
        row["cache_payload_representation"] = None
        row["cache_payload_bytes"] = 0
        row["cache_payload_decoded_max_cdf_error"] = None
        row["cache_payload_decoded_w1_cdf_error"] = None
        row["cache_payload_bin_count"] = 0
        row["parametric_payload_representation"] = None
        row["parametric_payload_bytes"] = 0
        row["parametric_payload_decoded_max_cdf_error"] = None
        row["parametric_payload_decoded_w1_cdf_error"] = None
        row["parametric_payload_bin_count"] = 0
        row["parametric_cached"] = False
        row["parametric_histogram"] = {}
        row["parametric_path_count"] = 0
        row["parametric_oracle_path_count"] = row["path_count"]
        row["parametric_shape_model"] = parametric_shape_model
        row["parametric_mean_model"] = parametric_mean_model
        row["parametric_mean_blend"] = parametric_mean_blend
        row["parametric_shape_mean_excess"] = None
        row["parametric_support_source"] = parametric_support_source
        row["parametric_support_bound_min"] = None
        row["parametric_support_bound_max"] = None
        row["parametric_support_bound_width"] = None
        row["parametric_support_min_delta"] = None
        row["parametric_support_max_delta"] = None
        row["parametric_support_width_delta"] = None
        row["parametric_support_truncated"] = False
        row["parametric_support_cycle_skipped"] = False
        row["parametric_shape_probability"] = None
        row["parametric_mass_model"] = parametric_mass_model
        row["parametric_mass_delta"] = None
        row["parametric_mass_ratio"] = None
        row["parametric_mass_capped"] = False
        row["parametric_support_bins"] = 0
        row["parametric_support_min"] = None
        row["parametric_support_max"] = None
        if cached:
            entry = serialized_histogram_cache_entry(hist)
            cache[row["node"]] = entry
            row["cache_payload_representation"] = entry.metadata["representation"]
            row["cache_payload_bytes"] = entry.metadata["payload_bytes"]
            row["cache_payload_decoded_max_cdf_error"] = entry.metadata["decoded_max_cdf_error"]
            row["cache_payload_decoded_w1_cdf_error"] = entry.metadata["decoded_w1_cdf_error"]
            row["cache_payload_bin_count"] = entry.metadata["bin_count"]
        elif policy["action"] == "use_parametric_prior" and lmax in priors and hist:
            mass_estimate = estimate_parametric_total_count(
                row,
                priors[lmax],
                parametric_mass_model,
                parametric_mass_cap,
            )
            probabilities, origin, shape_params = parametric_shape_distribution(
                row,
                priors[lmax],
                parametric_shape_model,
                parametric_mean_model,
                parametric_mean_blend,
                parametric_support_source,
                boundary_budget,
            )
            if shape_params:
                support_min = shape_params.get("support_min")
                support_max = shape_params.get("support_max")
                support_width = shape_params.get("support_bound_width")
                measured_min = row.get("histogram_L_min")
                measured_max = row.get("histogram_L_max")
                measured_width = None if measured_min is None or measured_max is None else int(measured_max) - int(measured_min)
                row["parametric_support_bound_min"] = support_min
                row["parametric_support_bound_max"] = support_max
                row["parametric_support_bound_width"] = support_width
                row["parametric_support_min_delta"] = None if measured_min is None or support_min is None else int(support_min) - int(measured_min)
                row["parametric_support_max_delta"] = None if measured_max is None or support_max is None else int(support_max) - int(measured_max)
                row["parametric_support_width_delta"] = None if measured_width is None or support_width is None else int(support_width) - int(measured_width)
                if parametric_support_source == "distance-bounds":
                    row["parametric_support_truncated"] = row["distance_truncated"]
                    row["parametric_support_cycle_skipped"] = row["distance_cycle_skipped"]
            approx_hist = scaled_distribution_histogram(
                probabilities,
                origin,
                mass_estimate["estimated_path_count"],
            )
            if approx_hist and shape_params:
                entry = serialized_histogram_cache_entry(approx_hist)
                parametric_cache[row["node"]] = entry
                row["parametric_cached"] = True
                row["parametric_histogram"] = approx_hist
                row["parametric_path_count"] = sum(approx_hist.values())
                row["parametric_oracle_path_count"] = mass_estimate["oracle_path_count"]
                row["parametric_mean_model"] = shape_params["mean_model"]
                row["parametric_shape_mean_excess"] = shape_params["mean_excess"]
                row["parametric_shape_probability"] = shape_params["probability"]
                row["parametric_mass_delta"] = mass_estimate["mass_delta"]
                row["parametric_mass_ratio"] = mass_estimate["mass_ratio"]
                row["parametric_mass_capped"] = mass_estimate["mass_capped"]
                row["parametric_support_bins"] = len(approx_hist)
                row["parametric_support_min"] = min(approx_hist)
                row["parametric_support_max"] = max(approx_hist)
                row["parametric_payload_representation"] = entry.metadata["representation"]
                row["parametric_payload_bytes"] = entry.metadata["payload_bytes"]
                row["parametric_payload_decoded_max_cdf_error"] = entry.metadata["decoded_max_cdf_error"]
                row["parametric_payload_decoded_w1_cdf_error"] = entry.metadata["decoded_w1_cdf_error"]
                row["parametric_payload_bin_count"] = entry.metadata["bin_count"]
    return cache, parametric_cache, rows


def comparison_record(graph_name, root, target, child_depth, budget, full_hist, full_stats, full_time, cached_hist, cached_stats, cached_time, collect_attribution=False):
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
        "cache_payload_bytes_read": cached_stats.cache_payload_bytes_read,
        "cache_decode_ns": cached_stats.cache_decode_ns,
        "cache_probe_ns": cached_stats.cache_probe_ns,
        "cache_splice_ns": cached_stats.cache_splice_ns,
        "cache_path_cap_check_ns": cached_stats.cache_path_cap_check_ns,
        "cached_parent_lookup_ns": cached_stats.parent_lookup_ns,
        "cached_attributed_ns": (
            cached_stats.cache_decode_ns
            + cached_stats.cache_probe_ns
            + cached_stats.cache_splice_ns
            + cached_stats.cache_path_cap_check_ns
            + cached_stats.parent_lookup_ns
        ),
        "collect_attribution": bool(collect_attribution),
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
        selected_boundary_nodes = set(boundary_nodes)
        target_ancestor_boundary_nodes = []
        if args.include_target_ancestor_boundaries:
            target_ancestor_boundary_nodes = collect_target_ancestor_boundaries(
                graph.parents,
                args.root,
                targets,
                boundary_depths,
                max(budgets, default=args.boundary_budget),
                args.max_parent_depth,
                args.target_ancestor_boundary_limit,
                args.seed + ":target-ancestor-boundaries",
            )
            boundary_nodes = sorted(selected_boundary_nodes | set(target_ancestor_boundary_nodes))
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
            args.parametric_mean_model,
            args.parametric_mean_blend,
            args.parametric_support_source,
            args.parametric_mass_model,
            args.parametric_mass_cap,
            args.tail_epsilon,
            args.max_parent_depth,
            args.boundary_builder,
            args.max_recurrence_states,
            args.max_effective_bins_after_trim,
        )
        records = [{
            "record_type": "boundary_cache_selection",
            "graph": args.graph_name,
            "root": args.root,
            "boundary_counts": boundary_counts,
            "target_counts": target_counts,
            "boundary_nodes": len(boundary_nodes),
            "selected_boundary_nodes": len(selected_boundary_nodes),
            "target_ancestor_boundary_nodes_added": len(set(boundary_nodes) - selected_boundary_nodes),
            "include_target_ancestor_boundaries": args.include_target_ancestor_boundaries,
            "cached_boundary_nodes": len(cache),
            "parametric_boundary_nodes": len(parametric_cache),
            "targets": len(targets),
            "budgets": budgets,
            "boundary_budget": args.boundary_budget,
            "admission_policy": args.admission_policy,
            "boundary_builder": args.boundary_builder,
            "max_recurrence_states": args.max_recurrence_states,
            "max_effective_bins_after_trim": args.max_effective_bins_after_trim,
            "safety_factor": args.safety_factor,
            "max_histogram_bytes": args.max_histogram_bytes,
            "parametric_bytes": args.parametric_bytes,
            "parametric_shape_model": args.parametric_shape_model,
            "parametric_mean_model": args.parametric_mean_model,
            "parametric_mean_blend": args.parametric_mean_blend,
            "parametric_support_source": args.parametric_support_source,
            "parametric_mass_model": args.parametric_mass_model,
            "parametric_mass_cap": args.parametric_mass_cap,
            "collect_attribution": args.collect_attribution,
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
                    args.collect_attribution,
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
                        args.collect_attribution,
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
        "| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |",
        "|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|",
        "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            selection.get("boundary_nodes", 0),
            selection.get("selected_boundary_nodes", selection.get("boundary_nodes", 0)),
            selection.get("target_ancestor_boundary_nodes_added", 0),
            selection.get("cached_boundary_nodes", 0),
            selection.get("parametric_boundary_nodes", 0),
            selection.get("targets", 0),
            selection.get("boundary_budget", 0),
            selection.get("boundary_builder", "search"),
        ),
        "",
        "## Admission Policy",
        "",
        "| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |",
        "|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|",
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            selection.get("admission_policy", "baseline"),
            selection.get("safety_factor", "n/a"),
            selection.get("max_histogram_bytes", "n/a"),
            selection.get("parametric_bytes", "n/a"),
            selection.get("parametric_shape_model", "empirical-prior"),
            selection.get("parametric_mean_model", "prior-clipped"),
            selection.get("parametric_mean_blend", "n/a"),
            selection.get("parametric_support_source", "measured"),
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
    support_rows = [row for row in cache_rows if row.get("parametric_support_bound_width") is not None]
    if support_rows:
        lines.extend([
            "",
            "## Parametric Support Bounds",
            "",
            "| support_source | rows | mean_bound_width | mean_width_delta | mean_min_delta | mean_max_delta | truncated | cycle_skipped |",
            "|----------------|-----:|-----------------:|-----------------:|---------------:|---------------:|----------:|--------------:|",
        ])
        support_by_source = {}
        for row in support_rows:
            support_by_source.setdefault(row.get("parametric_support_source", "unknown"), []).append(row)
        for source in sorted(support_by_source):
            rows = support_by_source[source]
            width_delta = [int(row["parametric_support_width_delta"]) for row in rows if row.get("parametric_support_width_delta") is not None]
            min_delta = [int(row["parametric_support_min_delta"]) for row in rows if row.get("parametric_support_min_delta") is not None]
            max_delta = [int(row["parametric_support_max_delta"]) for row in rows if row.get("parametric_support_max_delta") is not None]
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
                source,
                len(rows),
                statistics.mean(int(row["parametric_support_bound_width"]) for row in rows),
                statistics.mean(width_delta) if width_delta else 0.0,
                statistics.mean(min_delta) if min_delta else 0.0,
                statistics.mean(max_delta) if max_delta else 0.0,
                sum(1 for row in rows if row.get("parametric_support_truncated")),
                sum(1 for row in rows if row.get("parametric_support_cycle_skipped")),
            ))
    threshold_rows = [row for row in cache_rows if row.get("max_recurrence_states") or row.get("max_effective_bins_after_trim")]
    if threshold_rows:
        lines.extend([
            "",
            "## Approximation Thresholds",
            "",
            "| max_recurrence_states | max_effective_bins_after_trim | rows | states_over_limit | bins_over_limit | forced_parametric | mean_effective_bins_after_trim |",
            "|----------------------:|------------------------------:|-----:|------------------:|----------------:|------------------:|-------------------------------:|",
        ])
        lines.append("| {} | {} | {} | {} | {} | {} | {:.3f} |".format(
            selection.get("max_recurrence_states", 0),
            selection.get("max_effective_bins_after_trim", 0),
            len(threshold_rows),
            sum(1 for row in threshold_rows if row.get("recurrence_states_over_limit")),
            sum(1 for row in threshold_rows if row.get("effective_bins_over_limit")),
            sum(1 for row in threshold_rows if row.get("approximation_forced_by_threshold")),
            statistics.mean(int(row.get("effective_support_bins_after_trim", 0)) for row in threshold_rows) if threshold_rows else 0.0,
        ))
        threshold_reasons = {}
        for row in threshold_rows:
            reason = row.get("approximation_threshold_reason") or "within_threshold"
            threshold_reasons[reason] = threshold_reasons.get(reason, 0) + 1
        lines.extend(["", "| threshold_reason | rows |", "|------------------|-----:|"])
        for reason in sorted(threshold_reasons):
            lines.append("| {} | {} |".format(reason, threshold_reasons[reason]))
    builder_rows = {}
    for row in cache_rows:
        builder_rows.setdefault(row.get("boundary_builder", "search"), []).append(row)
    if builder_rows:
        lines.extend([
            "",
            "## Boundary Builders",
            "",
            "| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |",
            "|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|",
        ])
        for builder in sorted(builder_rows):
            rows = builder_rows[builder]
            lines.append("| {} | {} | {:.3f} | {:.3f} | {} | {} |".format(
                builder,
                len(rows),
                statistics.mean(int(row["nodes_expanded"]) for row in rows) if rows else 0.0,
                statistics.mean(int(row["edges_examined"]) for row in rows) if rows else 0.0,
                sum(1 for row in rows if row.get("recurrence_cycle_approximation")),
                sum(1 for row in rows if row.get("path_cap_hit") or row.get("expansion_cap_hit")),
            ))
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
    payload_rows = [row for row in cache_rows if row.get("cache_payload_bytes") or row.get("parametric_payload_bytes")]
    if payload_rows:
        histogram_payloads = [int(row["cache_payload_bytes"]) for row in cache_rows if row.get("cache_payload_bytes")]
        parametric_payloads = [int(row["parametric_payload_bytes"]) for row in cache_rows if row.get("parametric_payload_bytes")]
        histogram_cdf = [float(row["cache_payload_decoded_max_cdf_error"]) for row in cache_rows if row.get("cache_payload_decoded_max_cdf_error") is not None]
        parametric_cdf = [float(row["parametric_payload_decoded_max_cdf_error"]) for row in cache_rows if row.get("parametric_payload_decoded_max_cdf_error") is not None]
        lines.extend([
            "",
            "## Boundary Cache Payloads",
            "",
            "| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |",
            "|------|--------:|-------------------:|------------------:|-----------------:|",
            "| histogram | {} | {:.3f} | {} | {:.6f} |".format(
                len(histogram_payloads),
                statistics.mean(histogram_payloads) if histogram_payloads else 0.0,
                max(histogram_payloads, default=0),
                statistics.mean(histogram_cdf) if histogram_cdf else 0.0,
            ),
            "| parametric | {} | {:.3f} | {} | {:.6f} |".format(
                len(parametric_payloads),
                statistics.mean(parametric_payloads) if parametric_payloads else 0.0,
                max(parametric_payloads, default=0),
                statistics.mean(parametric_cdf) if parametric_cdf else 0.0,
            ),
        ])
    lines.extend([
        "",
        "## Full Search Versus Boundary Cache",
        "",
        "| budget | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | full_capped | cached_capped |",
        "|--------|------|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|-------------|---------------|",
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
        payload_bytes_read = [int(row["cache_payload_bytes_read"]) for row in rows]
        decode_ns = [int(row["cache_decode_ns"]) for row in rows]
        splice_ns = [int(row.get("cache_splice_ns", 0)) for row in rows]
        parent_lookup_ns = [int(row.get("cached_parent_lookup_ns", 0)) for row in rows]
        probe_ns = [int(row.get("cache_probe_ns", 0)) for row in rows]
        path_cap_check_ns = [int(row.get("cache_path_cap_check_ns", 0)) for row in rows]
        attributed_ns = [int(row.get("cached_attributed_ns", 0)) for row in rows]
        full_times = [int(row["full_time_ns"]) for row in rows]
        cached_times = [int(row["cached_time_ns"]) for row in rows]
        unattributed_ns = [
            max(0, int(row["cached_time_ns"]) - int(row.get("cached_attributed_ns", 0)))
            for row in rows
        ]
        time_ratios = [
            0.0 if int(row["full_time_ns"]) == 0 else int(row["cached_time_ns"]) / int(row["full_time_ns"])
            for row in rows
        ]
        lines.append(
            "| {budget} | {rows} | {mean_l1:.6f} | {p95_l1:.6f} | {max_l1:.6f} | {mean_cdf:.6f} | {mean_path_relative:.6f} | {mean_path_delta:.3f} | {mean_ratio:.3f} | {mean_time_ratio:.3f} | {mean_full_time:.1f} | {mean_cached_time:.1f} | {mean_hist_hits:.3f} | {mean_param_hits:.3f} | {mean_hist_bins:.3f} | {mean_param_bins:.3f} | {mean_payload_bytes:.3f} | {mean_decode_ns:.1f} | {full_capped} | {cached_capped} |".format(
                budget=budget,
                rows=len(rows),
                mean_l1=statistics.mean(l1) if l1 else 0.0,
                p95_l1=percentile(l1, 95),
                max_l1=max(l1, default=0.0),
                mean_cdf=statistics.mean(cdf) if cdf else 0.0,
                mean_path_relative=statistics.mean(path_relative) if path_relative else 0.0,
                mean_path_delta=statistics.mean(path_delta) if path_delta else 0.0,
                mean_ratio=statistics.mean(ratios) if ratios else 0.0,
                mean_time_ratio=statistics.mean(time_ratios) if time_ratios else 0.0,
                mean_full_time=statistics.mean(full_times) if full_times else 0.0,
                mean_cached_time=statistics.mean(cached_times) if cached_times else 0.0,
                mean_hist_hits=statistics.mean(histogram_hits) if histogram_hits else 0.0,
                mean_param_hits=statistics.mean(parametric_hits) if parametric_hits else 0.0,
                mean_hist_bins=statistics.mean(histogram_bins_spliced) if histogram_bins_spliced else 0.0,
                mean_param_bins=statistics.mean(parametric_bins_spliced) if parametric_bins_spliced else 0.0,
                mean_payload_bytes=statistics.mean(payload_bytes_read) if payload_bytes_read else 0.0,
                mean_decode_ns=statistics.mean(decode_ns) if decode_ns else 0.0,
                mean_splice_ns=statistics.mean(splice_ns) if splice_ns else 0.0,
                mean_parent_lookup_ns=statistics.mean(parent_lookup_ns) if parent_lookup_ns else 0.0,
                mean_probe_ns=statistics.mean(probe_ns) if probe_ns else 0.0,
                mean_path_cap_check_ns=statistics.mean(path_cap_check_ns) if path_cap_check_ns else 0.0,
                mean_attributed_ns=statistics.mean(attributed_ns) if attributed_ns else 0.0,
                mean_unattributed_ns=statistics.mean(unattributed_ns) if unattributed_ns else 0.0,
                full_capped=sum(1 for row in rows if row.get("full_path_cap_hit") or row.get("full_expansion_cap_hit")),
                cached_capped=sum(1 for row in rows if row.get("cached_path_cap_hit") or row.get("cached_expansion_cap_hit")),
            )
        )
    if any(row.get("collect_attribution") for row in comparison_rows):
        lines.extend([
            "",
            "## Cached Runtime Attribution",
            "",
            "These columns attribute only the cached search path. `unattributed` is the remaining cached wall time after decode, splice, cache-probe, path-cap check, and parent lookup timing buckets.",
            "",
            "| budget | rows | mean_cached_time_ns | mean_decode_ns | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_cap_check_ns | mean_attributed_ns | mean_unattributed_ns | decode_share | splice_share | parent_lookup_share |",
            "|--------|-----:|--------------------:|---------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|-------------:|-------------:|--------------------:|",
        ])
        for budget in sorted(by_budget):
            rows = by_budget[budget]
            cached_times = [int(row["cached_time_ns"]) for row in rows]
            decode_ns = [int(row.get("cache_decode_ns", 0)) for row in rows]
            splice_ns = [int(row.get("cache_splice_ns", 0)) for row in rows]
            parent_lookup_ns = [int(row.get("cached_parent_lookup_ns", 0)) for row in rows]
            probe_ns = [int(row.get("cache_probe_ns", 0)) for row in rows]
            path_cap_check_ns = [int(row.get("cache_path_cap_check_ns", 0)) for row in rows]
            attributed_ns = [int(row.get("cached_attributed_ns", 0)) for row in rows]
            unattributed_ns = [
                max(0, int(row["cached_time_ns"]) - int(row.get("cached_attributed_ns", 0)))
                for row in rows
            ]
            mean_cached = statistics.mean(cached_times) if cached_times else 0.0
            mean_decode = statistics.mean(decode_ns) if decode_ns else 0.0
            mean_splice = statistics.mean(splice_ns) if splice_ns else 0.0
            mean_parent_lookup = statistics.mean(parent_lookup_ns) if parent_lookup_ns else 0.0
            lines.append(
                "| {budget} | {rows} | {cached:.1f} | {decode:.1f} | {splice:.1f} | {parent_lookup:.1f} | {probe:.1f} | {path_cap:.1f} | {attributed:.1f} | {unattributed:.1f} | {decode_share:.3f} | {splice_share:.3f} | {parent_lookup_share:.3f} |".format(
                    budget=budget,
                    rows=len(rows),
                    cached=mean_cached,
                    decode=mean_decode,
                    splice=mean_splice,
                    parent_lookup=mean_parent_lookup,
                    probe=statistics.mean(probe_ns) if probe_ns else 0.0,
                    path_cap=statistics.mean(path_cap_check_ns) if path_cap_check_ns else 0.0,
                    attributed=statistics.mean(attributed_ns) if attributed_ns else 0.0,
                    unattributed=statistics.mean(unattributed_ns) if unattributed_ns else 0.0,
                    decode_share=0.0 if mean_cached <= 0.0 else mean_decode / mean_cached,
                    splice_share=0.0 if mean_cached <= 0.0 else mean_splice / mean_cached,
                    parent_lookup_share=0.0 if mean_cached <= 0.0 else mean_parent_lookup / mean_cached,
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
    parser.add_argument("--include-target-ancestor-boundaries", action="store_true", help="Add ancestors of sampled targets whose root distance matches requested boundary depths.")
    parser.add_argument("--target-ancestor-boundary-limit", type=int, default=500, help="Maximum target-ancestor boundary nodes to add; non-positive means no extra cap.")
    parser.add_argument("--boundary-budget", type=int, default=6, help="Path budget used to precompute boundary histograms.")
    parser.add_argument("--boundary-builder", choices=["search", "recurrence"], default="search", help="Method used to build boundary histograms before admission.")
    parser.add_argument("--max-recurrence-states", type=int, default=0, help="If positive, use a parametric boundary state when recurrence state count exceeds this limit.")
    parser.add_argument("--max-effective-bins-after-trim", type=int, default=0, help="If positive, use a parametric boundary state when tail-trimmed effective bins exceed this limit.")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)), help="Comma-separated target path budgets.")
    parser.add_argument("--admission-policy", choices=["baseline", "depth-prior"], default="baseline", help="Policy used to decide whether measured boundary histograms enter the cache.")
    parser.add_argument("--safety-factor", type=float, default=1.25, help="Multiplier for depth-prior predicted histogram bytes.")
    parser.add_argument("--max-histogram-bytes", type=int, default=1024, help="Maximum bytes allowed for exact or capped boundary histograms under depth-prior admission.")
    parser.add_argument("--parametric-bytes", type=int, default=64, help="Estimated bytes for a parametric prior state.")
    parser.add_argument("--parametric-shape-model", choices=["empirical-prior", "support-binomial", "support-binomial-midpoint"], default="empirical-prior", help="Shape used for parametric boundary states.")
    parser.add_argument("--parametric-mean-model", choices=["prior-clipped", "midpoint", "blend"], default="prior-clipped", help="Mean rule for support-binomial parametric shape states.")
    parser.add_argument("--parametric-mean-blend", type=float, default=0.5, help="Prior weight for --parametric-mean-model blend; midpoint gets the remaining weight.")
    parser.add_argument("--parametric-support-source", choices=["measured", "distance-bounds"], default="measured", help="Support interval source for parametric boundary states.")
    parser.add_argument("--parametric-mass-model", choices=["oracle", "unit", "depth-prior"], default="oracle", help="Mass used to unnormalize parametric boundary states.")
    parser.add_argument("--parametric-mass-cap", type=int, default=1000000, help="Maximum estimated path mass for non-oracle parametric states; non-positive disables the cap.")
    parser.add_argument("--tail-epsilon", type=float, default=0.001, help="Tail epsilon used when estimating depth-prior effective support.")
    parser.add_argument("--max-parent-depth", type=int, default=24, help="Parent depth cap for root-reaching parent degree signals.")
    parser.add_argument("--path-cap", type=int, default=100000, help="Stop a row after this many root paths.")
    parser.add_argument("--expansion-cap", type=int, default=250000, help="Stop a row after this many expanded nodes.")
    parser.add_argument("--seed", default="0", help="Deterministic sampling seed.")
    parser.add_argument("--collect-attribution", action="store_true", help="Collect cached-search timing attribution for cache probes, splicing, parent lookups, and path-cap checks.")
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
