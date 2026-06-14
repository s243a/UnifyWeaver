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
from dataclasses import dataclass, field
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
AGGREGATE_VALUE_KERNELS = ["count", "bp-decay", "weighted-power"]
PARENT_FILTERS = ["all", "root-reachable", "root-cone"]


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
    cache_decode_memo_hits: int = 0
    path_count: int = 0
    cache_hit_depth_sum: int = 0
    cache_hit_remaining_budget_sum: int = 0
    cache_hit_suffix_path_count_sum: int = 0
    first_cache_hit_depth: int | None = None
    first_cache_hit_remaining_budget: int | None = None
    max_cache_hit_remaining_budget: int = 0
    cache_hits_by_depth: Counter = field(default_factory=Counter)
    cache_hits_by_remaining_budget: Counter = field(default_factory=Counter)
    cache_probe_ns: int = 0
    cache_splice_ns: int = 0
    cache_path_cap_check_ns: int = 0
    parent_lookup_ns: int = 0
    root_unreachable_parent_skips: int = 0
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


class RootReachabilityParentFilter:
    """Finite-horizon parent filter for root-reaching candidate parents."""

    def __init__(self, parents_func, root):
        self.parents_func = parents_func
        self.root = root
        self.memo = {}
        self.checks = 0
        self.memo_hits = 0
        self.cycle_skips = 0

    def accepts(self, _node, parent, remaining):
        return self.can_reach(parent, remaining)

    def accepts_without_remaining(self, parent, horizon):
        return self.can_reach(parent, horizon)

    def can_reach(self, node, remaining):
        return self._can_reach(node, int(remaining), set())

    def _can_reach(self, node, remaining, visiting):
        self.checks += 1
        if node == self.root:
            return True
        if remaining <= 0:
            return False
        key = (node, int(remaining))
        if key in self.memo:
            self.memo_hits += 1
            return self.memo[key]
        if node in visiting:
            self.cycle_skips += 1
            return False

        visiting.add(node)
        reachable = False
        for parent in self.parents_func(node):
            if self._can_reach(parent, remaining - 1, visiting):
                reachable = True
                break
        visiting.remove(node)
        self.memo[key] = reachable
        return reachable


class RootConeParentFilter:
    """Parent filter backed by a precomputed child-depth cone from the root."""

    def __init__(self, depth_by_node):
        self.depth_by_node = dict(depth_by_node)
        self.checks = 0
        self.depth_misses = 0
        self.remaining_misses = 0

    def accepts(self, _node, parent, remaining):
        self.checks += 1
        depth = self.depth_by_node.get(parent)
        if depth is None:
            self.depth_misses += 1
            return False
        if int(depth) > int(remaining):
            self.remaining_misses += 1
            return False
        return True

    def accepts_without_remaining(self, parent, _horizon):
        self.checks += 1
        if parent not in self.depth_by_node:
            self.depth_misses += 1
            return False
        return True


def normalize_positive_limit(value):
    return None if value is None or int(value) <= 0 else int(value)


def build_root_cone_depths(children_func, root, max_depth, children_per_node, frontier_limit, seed):
    """Build a bounded child-reachable cone from root with minimum depths."""
    max_depth = int(max_depth)
    children_limit = normalize_positive_limit(children_per_node)
    frontier_limit = normalize_positive_limit(frontier_limit)
    depth_by_node = {root: 0}
    frontier = [root]
    counts = {0: 1}

    for depth in range(1, max_depth + 1):
        next_nodes = []
        for node in frontier:
            children = deterministic_sample(
                children_func(node),
                children_limit,
                "{}:root-cone:children:{}:{}".format(seed, depth, node),
            )
            next_nodes.extend(children)
        sampled = deterministic_sample(
            list(dict.fromkeys(next_nodes)),
            frontier_limit,
            "{}:root-cone:frontier:{}".format(seed, depth),
        )
        new_frontier = []
        for child in sampled:
            if child in depth_by_node:
                continue
            depth_by_node[child] = depth
            new_frontier.append(child)
        counts[depth] = len(new_frontier)
        frontier = new_frontier
        if not frontier:
            break

    return depth_by_node, counts


def filtered_parents_func(parents_func, parent_filter, horizon):
    if parent_filter is None:
        return parents_func

    def parents(node):
        return [
            parent
            for parent in parents_func(node)
            if parent_filter.accepts_without_remaining(parent, horizon)
        ]

    return parents


def filtered_bounded_parent_histogram(parents_func, target, root, budget, path_cap=None, expansion_cap=None, parent_filter=None):
    """Count simple root paths while optionally rejecting off-scope parents."""
    if parent_filter is None:
        hist, stats = bounded_parent_histogram(parents_func, target, root, budget, path_cap, expansion_cap)
        stats.root_unreachable_parent_skips = 0
        return hist, stats

    hist = Counter()
    stats = HistogramStats()
    stats.root_unreachable_parent_skips = 0

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
            if not parent_filter.accepts(node, parent, remaining - 1):
                stats.root_unreachable_parent_skips += 1
                continue
            if stats.path_cap_hit or stats.expansion_cap_hit:
                return
            visited.add(parent)
            dfs(parent, remaining - 1, depth + 1, visited)
            visited.remove(parent)
            if stats.path_cap_hit or stats.expansion_cap_hit:
                return

    dfs(target, int(budget), 0, {target})
    return dict(sorted(hist.items())), stats


def build_boundary_histogram(parents_func, node, root, budget, path_cap, expansion_cap, boundary_builder, parent_filter=None):
    """Build one boundary histogram by search or shifted parent recurrence."""
    if boundary_builder == "search":
        hist, stats = filtered_bounded_parent_histogram(parents_func, node, root, budget, path_cap, expansion_cap, parent_filter)
        return BoundaryBuildResult(
            histogram=hist,
            nodes_expanded=stats.nodes_expanded,
            edges_examined=stats.edges_examined,
            cycle_skips=stats.cycle_skips,
            path_cap_hit=stats.path_cap_hit,
            expansion_cap_hit=stats.expansion_cap_hit,
        )
    if boundary_builder == "recurrence":
        recurrence_parents = filtered_parents_func(parents_func, parent_filter, budget)
        hist, stats = recurrence_parent_histogram(recurrence_parents, node, root, budget, path_cap, expansion_cap)
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
    added_bins = 0
    added_paths = 0
    for suffix_length, count in suffix_hist.items():
        if suffix_length <= remaining:
            out[prefix_depth + suffix_length] += count
            added_bins += 1
            added_paths += int(count)
    return added_bins, added_paths


def build_boundary_lookup(boundary_cache, parametric_boundary_cache=None):
    lookup = {node: ("histogram", node, entry) for node, entry in boundary_cache.items()}
    for node, entry in (parametric_boundary_cache or {}).items():
        lookup.setdefault(node, ("parametric", node, entry))
    return lookup


def cache_entry_histogram(entry, decoded_cache_memo=None, memo_key=None):
    if decoded_cache_memo is not None and memo_key is not None and memo_key in decoded_cache_memo:
        hist, _payload_bytes = decoded_cache_memo[memo_key]
        return hist, 0, 0, True
    if isinstance(entry, SerializedHistogramCacheEntry):
        hist, decode_ns = entry.decode_histogram()
        payload_bytes = int(entry.metadata.get("payload_bytes", len(entry.payload)))
    else:
        hist, decode_ns, payload_bytes = entry, 0, 0
    if decoded_cache_memo is not None and memo_key is not None:
        decoded_cache_memo[memo_key] = (hist, payload_bytes)
    return hist, payload_bytes, decode_ns, False


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
    boundary_lookup=None,
    decoded_cache_memo=None,
    parent_filter=None,
):
    if boundary_lookup is None:
        boundary_lookup = build_boundary_lookup(boundary_cache, parametric_boundary_cache)
    hist = Counter()
    stats = CachedSearchStats()

    def dfs(node, remaining, depth, visited):
        if expansion_cap is not None and stats.nodes_expanded >= expansion_cap:
            stats.expansion_cap_hit = True
            return
        stats.nodes_expanded += 1
        if node == root:
            hist[depth] += 1
            stats.path_count += 1
            if path_cap is not None:
                started = time.perf_counter_ns() if collect_attribution else None
                cap_hit = stats.path_count >= path_cap
                if collect_attribution:
                    stats.cache_path_cap_check_ns += time.perf_counter_ns() - started
                if cap_hit:
                    stats.path_cap_hit = True
            return

        started = time.perf_counter_ns() if collect_attribution else None
        boundary_entry = boundary_lookup.get(node)
        if collect_attribution:
            stats.cache_probe_ns += time.perf_counter_ns() - started

        if boundary_entry is not None:
            cache_kind, cache_node, entry = boundary_entry
            stats.cache_hits += 1
            if cache_kind == "histogram":
                stats.histogram_cache_hits += 1
            else:
                stats.parametric_cache_hits += 1
            suffix_hist, payload_bytes, decode_ns, memo_hit = cache_entry_histogram(
                entry,
                decoded_cache_memo,
                (cache_kind, cache_node),
            )
            stats.cache_payload_bytes_read += payload_bytes
            stats.cache_decode_ns += decode_ns
            if memo_hit:
                stats.cache_decode_memo_hits += 1
            started = time.perf_counter_ns() if collect_attribution else None
            added_bins, added_paths = add_shifted_hist(hist, suffix_hist, depth, remaining)
            if collect_attribution:
                stats.cache_splice_ns += time.perf_counter_ns() - started
            stats.path_count += added_paths
            hit_depth = int(depth)
            hit_remaining = int(remaining)
            if stats.first_cache_hit_depth is None:
                stats.first_cache_hit_depth = hit_depth
                stats.first_cache_hit_remaining_budget = hit_remaining
            stats.max_cache_hit_remaining_budget = max(stats.max_cache_hit_remaining_budget, hit_remaining)
            stats.cache_hit_depth_sum += hit_depth
            stats.cache_hit_remaining_budget_sum += hit_remaining
            stats.cache_hit_suffix_path_count_sum += int(added_paths)
            stats.cache_hits_by_depth[hit_depth] += 1
            stats.cache_hits_by_remaining_budget[hit_remaining] += 1
            stats.cache_bins_spliced += added_bins
            if cache_kind == "histogram":
                stats.histogram_bins_spliced += added_bins
            else:
                stats.parametric_bins_spliced += added_bins
            if path_cap is not None:
                started = time.perf_counter_ns() if collect_attribution else None
                cap_hit = stats.path_count >= path_cap
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
            if parent_filter is not None and not parent_filter.accepts(node, parent, remaining - 1):
                stats.root_unreachable_parent_skips += 1
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


def search_stop_reason(path_count_cap_hit, expansion_cap_hit, length_budget_cutoffs):
    reasons = []
    if path_count_cap_hit:
        reasons.append("path_count_cap")
    if expansion_cap_hit:
        reasons.append("expansion_cap")
    if reasons:
        return "+".join(reasons)
    if int(length_budget_cutoffs or 0) > 0:
        return "path_length_budget"
    return "complete"


def histogram_effective_bins(hist, tail_epsilon):
    probabilities, _origin = exact_excess_distribution(hist)
    if not probabilities:
        return 0
    return effective_support_bins(probabilities, tail_epsilon)


def aggregate_path_value(length, kernel="count", branching_factor=None, power=1.0):
    """Evaluate a path-length aggregate kernel."""
    length = int(length)
    kernel = str(kernel or "count").strip().lower().replace("_", "-")
    if kernel == "count":
        return 1.0
    if kernel == "bp-decay":
        if branching_factor is None or float(branching_factor) <= 0.0:
            raise ValueError("bp-decay aggregate requires a positive branching factor")
        return float(branching_factor) ** (-length)
    if kernel == "weighted-power":
        power = 1.0 if power is None else float(power)
        if power < 0.0:
            raise ValueError("weighted-power aggregate requires a non-negative power")
        return (length + 1.0) ** (-power)
    raise ValueError("unknown aggregate kernel: {}".format(kernel))


def histogram_value_sum(hist, kernel="count", branching_factor=None, power=1.0):
    return sum(
        int(count) * aggregate_path_value(length, kernel, branching_factor, power)
        for length, count in hist.items()
    )


def histogram_length_sum(hist):
    return sum(int(length) * int(count) for length, count in hist.items())


def histogram_mean_length(hist):
    path_count = sum(int(count) for count in hist.values())
    return None if path_count <= 0 else histogram_length_sum(hist) / path_count


def aggregate_relative_error(full_value, cached_value):
    full_value = float(full_value)
    cached_value = float(cached_value)
    if full_value == 0.0:
        return 0.0 if cached_value == 0.0 else None
    return abs(cached_value - full_value) / abs(full_value)


def histogram_aggregate_metrics(hist, kernel="count", branching_factor=None, power=1.0):
    path_count = sum(int(count) for count in hist.values())
    length_sum = histogram_length_sum(hist)
    return {
        "path_count": path_count,
        "path_length_sum": length_sum,
        "mean_path_length": None if path_count <= 0 else length_sum / path_count,
        "aggregate_value_sum": histogram_value_sum(hist, kernel, branching_factor, power),
    }


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


def descendant_frontier(graph, start, suffix_depth, children_per_node, frontier_limit, seed):
    frontier = [start]
    for depth in range(1, int(suffix_depth) + 1):
        next_nodes = []
        for node in frontier:
            children = deterministic_sample(
                graph.children(node),
                children_per_node,
                "{}:children:{}:{}".format(seed, depth, node),
            )
            next_nodes.extend(children)
        frontier = deterministic_sample(
            list(dict.fromkeys(next_nodes)),
            frontier_limit,
            "{}:frontier:{}".format(seed, depth),
        )
        if not frontier:
            break
    return frontier


def select_targets_by_boundary_descendants(
    graph,
    boundary_depth_by_node,
    target_depths,
    children_per_node,
    frontier_limit,
    targets_per_depth,
    seed,
):
    """Sample targets below selected boundaries so cache hits are intentional."""
    target_depths = [int(depth) for depth in target_depths]
    candidates_by_depth = {depth: [] for depth in target_depths}
    for boundary in sorted(boundary_depth_by_node):
        boundary_depth = int(boundary_depth_by_node[boundary])
        for target_depth in target_depths:
            suffix_depth = target_depth - boundary_depth
            if suffix_depth <= 0:
                continue
            descendants = descendant_frontier(
                graph,
                boundary,
                suffix_depth,
                children_per_node,
                frontier_limit,
                "{}:boundary:{}:target-depth:{}".format(seed, boundary, target_depth),
            )
            candidates_by_depth.setdefault(target_depth, []).extend(descendants)

    targets = []
    target_child_depth = {}
    selection_counts = {}
    for target_depth in sorted(candidates_by_depth):
        unique_candidates = list(dict.fromkeys(candidates_by_depth[target_depth]))
        selection_counts[target_depth] = len(unique_candidates)
        sampled = deterministic_sample(
            unique_candidates,
            targets_per_depth,
            "{}:targets:{}".format(seed, target_depth),
        )
        for node in sampled:
            if node not in target_child_depth:
                targets.append(node)
                target_child_depth[node] = target_depth
    return targets, target_child_depth, selection_counts


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
    parent_filter=None,
):
    cache = {}
    parametric_cache = {}
    rows = []
    distance_memo = {}
    for node in boundary_nodes:
        started = time.perf_counter_ns()
        built = build_boundary_histogram(
            graph.parents,
            node,
            root,
            boundary_budget,
            path_cap,
            expansion_cap,
            boundary_builder,
            parent_filter,
        )
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


def comparison_record(
    graph_name,
    root,
    target,
    child_depth,
    budget,
    full_hist,
    full_stats,
    full_time,
    cached_hist,
    cached_stats,
    cached_time,
    collect_attribution=False,
    path_count_cap=None,
    expansion_cap=None,
    aggregate_kernel="count",
    aggregate_branching_factor=None,
    aggregate_power=1.0,
):
    l1, cdf = histogram_distribution_error(full_hist, cached_hist)
    full_paths = sum(full_hist.values())
    cached_paths = sum(cached_hist.values())
    full_aggregates = histogram_aggregate_metrics(
        full_hist,
        aggregate_kernel,
        aggregate_branching_factor,
        aggregate_power,
    )
    cached_aggregates = histogram_aggregate_metrics(
        cached_hist,
        aggregate_kernel,
        aggregate_branching_factor,
        aggregate_power,
    )
    aggregate_delta = cached_aggregates["aggregate_value_sum"] - full_aggregates["aggregate_value_sum"]
    path_length_delta = cached_aggregates["path_length_sum"] - full_aggregates["path_length_sum"]
    mean_length_delta = (
        None
        if full_aggregates["mean_path_length"] is None or cached_aggregates["mean_path_length"] is None
        else cached_aggregates["mean_path_length"] - full_aggregates["mean_path_length"]
    )
    cache_hits = int(cached_stats.cache_hits)
    mean_cache_hit_depth = None if cache_hits <= 0 else cached_stats.cache_hit_depth_sum / cache_hits
    mean_cache_hit_remaining_budget = None if cache_hits <= 0 else cached_stats.cache_hit_remaining_budget_sum / cache_hits
    mean_cache_hit_suffix_path_count = None if cache_hits <= 0 else cached_stats.cache_hit_suffix_path_count_sum / cache_hits
    return {
        "record_type": "boundary_cache_comparison",
        "graph": graph_name,
        "root": root,
        "target_node": target,
        "child_sample_depth": child_depth,
        "budget": budget,
        "path_length_budget": budget,
        "path_count_cap": path_count_cap,
        "expansion_cap": expansion_cap,
        "full_histogram": full_hist,
        "cached_histogram": cached_hist,
        "full_path_count": full_paths,
        "cached_path_count": cached_paths,
        "path_count_delta": cached_paths - full_paths,
        "abs_path_count_delta": abs(cached_paths - full_paths),
        "path_count_relative_error": 0.0 if full_paths == 0 else abs(cached_paths - full_paths) / full_paths,
        "aggregate_kernel": aggregate_kernel,
        "aggregate_branching_factor": aggregate_branching_factor,
        "aggregate_power": aggregate_power,
        "full_aggregate_value_sum": full_aggregates["aggregate_value_sum"],
        "cached_aggregate_value_sum": cached_aggregates["aggregate_value_sum"],
        "aggregate_value_delta": aggregate_delta,
        "abs_aggregate_value_delta": abs(aggregate_delta),
        "aggregate_value_relative_error": aggregate_relative_error(
            full_aggregates["aggregate_value_sum"],
            cached_aggregates["aggregate_value_sum"],
        ),
        "full_path_length_sum": full_aggregates["path_length_sum"],
        "cached_path_length_sum": cached_aggregates["path_length_sum"],
        "path_length_sum_delta": path_length_delta,
        "abs_path_length_sum_delta": abs(path_length_delta),
        "full_mean_path_length": full_aggregates["mean_path_length"],
        "cached_mean_path_length": cached_aggregates["mean_path_length"],
        "mean_path_length_delta": mean_length_delta,
        "abs_mean_path_length_delta": None if mean_length_delta is None else abs(mean_length_delta),
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
        "full_root_unreachable_parent_skips": int(getattr(full_stats, "root_unreachable_parent_skips", 0)),
        "cached_root_unreachable_parent_skips": int(cached_stats.root_unreachable_parent_skips),
        "full_length_budget_cutoffs": full_stats.budget_cutoffs,
        "cached_length_budget_cutoffs": cached_stats.budget_cutoffs,
        "cache_hits": cached_stats.cache_hits,
        "cache_hit_depth_sum": cached_stats.cache_hit_depth_sum,
        "cache_hit_remaining_budget_sum": cached_stats.cache_hit_remaining_budget_sum,
        "cache_hit_suffix_path_count_sum": cached_stats.cache_hit_suffix_path_count_sum,
        "mean_cache_hit_depth": mean_cache_hit_depth,
        "mean_cache_hit_remaining_budget": mean_cache_hit_remaining_budget,
        "mean_cache_hit_suffix_path_count": mean_cache_hit_suffix_path_count,
        "first_cache_hit_depth": cached_stats.first_cache_hit_depth,
        "first_cache_hit_remaining_budget": cached_stats.first_cache_hit_remaining_budget,
        "max_cache_hit_remaining_budget": cached_stats.max_cache_hit_remaining_budget,
        "cache_hits_by_depth": dict(sorted(cached_stats.cache_hits_by_depth.items())),
        "cache_hits_by_remaining_budget": dict(sorted(cached_stats.cache_hits_by_remaining_budget.items())),
        "histogram_cache_hits": cached_stats.histogram_cache_hits,
        "parametric_cache_hits": cached_stats.parametric_cache_hits,
        "cache_bins_spliced": cached_stats.cache_bins_spliced,
        "histogram_bins_spliced": cached_stats.histogram_bins_spliced,
        "parametric_bins_spliced": cached_stats.parametric_bins_spliced,
        "cache_payload_bytes_read": cached_stats.cache_payload_bytes_read,
        "cache_decode_ns": cached_stats.cache_decode_ns,
        "cache_decode_memo_hits": cached_stats.cache_decode_memo_hits,
        "cached_path_count_tracked": cached_stats.path_count,
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
        "full_stop_reason": search_stop_reason(full_stats.path_cap_hit, full_stats.expansion_cap_hit, full_stats.budget_cutoffs),
        "cached_stop_reason": search_stop_reason(cached_stats.path_cap_hit, cached_stats.expansion_cap_hit, cached_stats.budget_cutoffs),
        "full_time_ns": full_time,
        "cached_time_ns": cached_time,
    }


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        budgets = parse_int_list(args.budgets)
        boundary_depths = parse_int_list(args.boundary_depths)
        target_depths = parse_int_list(args.target_depths)
        root_cone_depth_by_node = None
        root_cone_counts = {}
        root_cone_depth = int(args.root_cone_depth or 0)
        if args.parent_filter == "root-cone":
            if root_cone_depth <= 0:
                root_cone_depth = max(
                    [int(args.boundary_budget)]
                    + [int(depth) for depth in budgets]
                    + [int(depth) for depth in boundary_depths]
                    + [int(depth) for depth in target_depths]
                )
            root_cone_depth_by_node, root_cone_counts = build_root_cone_depths(
                graph.children,
                args.root,
                root_cone_depth,
                args.root_cone_children_per_node,
                args.root_cone_frontier_limit,
                args.seed,
            )
            parent_filter = RootConeParentFilter(root_cone_depth_by_node)
        elif args.parent_filter == "root-reachable":
            parent_filter = RootReachabilityParentFilter(graph.parents, args.root)
        else:
            parent_filter = None
        ancestor_parents = filtered_parents_func(
            graph.parents,
            parent_filter,
            max([int(args.max_parent_depth)] + [int(depth) for depth in budgets] + [int(args.boundary_budget)]),
        )
        boundary_nodes, boundary_depth_by_node, boundary_counts = select_targets_by_child_depth(
            graph,
            args.root,
            boundary_depths,
            args.children_per_node,
            args.frontier_limit,
            args.boundaries_per_depth,
            args.seed + ":boundary",
        )
        if args.target_selection == "child-depth":
            targets, target_depth_by_node, target_counts = select_targets_by_child_depth(
                graph,
                args.root,
                target_depths,
                args.children_per_node,
                args.frontier_limit,
                args.targets_per_depth,
                args.seed + ":target",
            )
        else:
            targets, target_depth_by_node, target_counts = select_targets_by_boundary_descendants(
                graph,
                boundary_depth_by_node,
                target_depths,
                args.children_per_node,
                args.frontier_limit,
                args.targets_per_depth,
                args.seed + ":boundary-descendant-target",
            )
        selected_boundary_nodes = set(boundary_nodes)
        target_ancestor_boundary_nodes = []
        if args.include_target_ancestor_boundaries:
            target_ancestor_boundary_nodes = collect_target_ancestor_boundaries(
                ancestor_parents,
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
            parent_filter,
        )
        boundary_lookup = build_boundary_lookup(cache, parametric_cache)
        decoded_cache_memo = {}
        records = [{
            "record_type": "boundary_cache_selection",
            "graph": args.graph_name,
            "root": args.root,
            "seed": args.seed,
            "boundary_depths": boundary_depths,
            "target_depths": target_depths,
            "boundary_counts": boundary_counts,
            "target_counts": target_counts,
            "boundary_nodes": len(boundary_nodes),
            "selected_boundary_nodes": len(selected_boundary_nodes),
            "target_ancestor_boundary_nodes_added": len(set(boundary_nodes) - selected_boundary_nodes),
            "include_target_ancestor_boundaries": args.include_target_ancestor_boundaries,
            "target_ancestor_boundary_limit": args.target_ancestor_boundary_limit,
            "cached_boundary_nodes": len(cache),
            "parametric_boundary_nodes": len(parametric_cache),
            "boundary_lookup_nodes": len(boundary_lookup),
            "targets": len(targets),
            "target_selection": args.target_selection,
            "parent_filter": args.parent_filter,
            "root_cone_depth": root_cone_depth if args.parent_filter == "root-cone" else None,
            "root_cone_nodes": None if root_cone_depth_by_node is None else len(root_cone_depth_by_node),
            "root_cone_counts": root_cone_counts,
            "root_cone_children_per_node": args.root_cone_children_per_node,
            "root_cone_frontier_limit": args.root_cone_frontier_limit,
            "children_per_node": args.children_per_node,
            "frontier_limit": args.frontier_limit,
            "boundaries_per_depth": args.boundaries_per_depth,
            "targets_per_depth": args.targets_per_depth,
            "budgets": budgets,
            "boundary_budget": args.boundary_budget,
            "path_count_cap": args.path_cap,
            "expansion_cap": args.expansion_cap,
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
            "aggregate_kernel": args.aggregate_kernel,
            "aggregate_branching_factor": args.aggregate_branching_factor,
            "aggregate_power": args.aggregate_power,
        }]
        records.extend(cache_rows)
        for target in targets:
            for budget in budgets:
                full_started = time.perf_counter_ns()
                full_hist, full_stats = filtered_bounded_parent_histogram(
                    graph.parents,
                    target,
                    args.root,
                    budget,
                    args.path_cap,
                    args.expansion_cap,
                    parent_filter,
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
                    boundary_lookup,
                    decoded_cache_memo,
                    parent_filter,
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
                        args.path_cap,
                        args.expansion_cap,
                        args.aggregate_kernel,
                        args.aggregate_branching_factor,
                        args.aggregate_power,
                    )
                )
        return records, summarize(records)
    finally:
        graph.close()


def _sorted_depth_items(counts):
    return sorted(counts.items(), key=lambda item: int(item[0]))


def _depths_text(counts):
    if not counts:
        return "none"
    return ", ".join(str(depth) for depth, _count in _sorted_depth_items(counts))


def _numeric_values(rows, key):
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]


def _mean_numeric(rows, key):
    values = _numeric_values(rows, key)
    return statistics.mean(values) if values else 0.0


def boundary_cache_generation_notes(selection):
    parent_filter = selection.get("parent_filter", "all")
    boundary_builder = selection.get("boundary_builder", "search")
    lines = [
        "## How This Was Generated",
        "",
    ]
    if selection.get("boundary_depths") is not None and selection.get("target_depths") is not None:
        lines.append(
            "- Boundary candidates were sampled from requested child depth(s) `{}` and target rows from requested child depth(s) `{}` using deterministic sampling. The `Selection` table also reports frontier counts observed while traversing to those depths.".format(
                ", ".join(str(value) for value in selection.get("boundary_depths", [])),
                ", ".join(str(value) for value in selection.get("target_depths", [])),
            )
        )
    else:
        lines.append(
            "- This older selection record stores observed boundary/target frontier counts over child-depths `{}` and `{}`, but not the requested depth arguments separately.".format(
                _depths_text(selection.get("boundary_counts", {})),
                _depths_text(selection.get("target_counts", {})),
            )
        )
    if all(selection.get(key) is not None for key in ("children_per_node", "frontier_limit", "boundaries_per_depth", "targets_per_depth")):
        lines.append(
            "- The child frontier sampler used `children_per_node={}` and `frontier_limit={}`; per-depth sampling limits were `boundaries_per_depth={}` and `targets_per_depth={}`.".format(
                selection.get("children_per_node"),
                selection.get("frontier_limit"),
                selection.get("boundaries_per_depth"),
                selection.get("targets_per_depth"),
            )
        )
    else:
        lines.append(
            "- This selection record predates sampler-limit provenance fields; newer JSONL records include child/frontier/per-depth sampling limits directly."
        )
    lines.extend([
        "- Boundary states used builder `{}` with `boundary_budget={}`. This budget limits how far each cached suffix histogram is built; it is separate from the target-row `path_length_budget` values `{}`.".format(
            boundary_builder,
            selection.get("boundary_budget", "n/a"),
            ",".join(str(value) for value in selection.get("budgets", [])),
        ),
        "- Target rows compare full simple-path parent DFS against cache-aware DFS over the same parent filter `{}`. Both searches reject repeated nodes in a path and use the same path-count and expansion caps.".format(parent_filter),
        "- A cache hit stops the live DFS at a boundary node and splices in the cached suffix histogram for the remaining path budget. Exactness requires compatible root, parent filter, budget, and cycle policy.",
    ])
    if boundary_builder == "recurrence":
        lines.append(
            "- The recurrence builder forms each boundary histogram by shifting parent histograms one hop to the right and summing them; recurrence state counts and cycle approximations are reported in the builder table."
        )
    if selection.get("include_target_ancestor_boundaries"):
        if selection.get("target_ancestor_boundary_limit") is None:
            lines.append(
                "- Target-ancestor boundary inclusion was enabled, so sampled boundary-depth ancestors of each target could be added to the selected boundary set."
            )
        else:
            lines.append(
                "- Target-ancestor boundary inclusion was enabled, so up to `{}` sampled boundary-depth ancestors of each target could be added to the selected boundary set.".format(
                    selection.get("target_ancestor_boundary_limit")
                )
            )
    if parent_filter == "root-cone":
        lines.append(
            "- The root-cone filter was built to child depth `{}` with `{}` nodes. Parent edges outside that cone, or too deep for the remaining parent-hop budget, are rejected and counted as filtered skips.".format(
                selection.get("root_cone_depth", "n/a"),
                selection.get("root_cone_nodes", "n/a"),
            )
        )
    elif parent_filter == "root-reachable":
        lines.append(
            "- The root-reachable filter rejects parent edges that cannot reach the configured root within the memoized reachability test."
        )
    return lines


def boundary_cache_table_guide():
    return [
        "## Table Guide",
        "",
        "- `Selection` reports the sampled boundary and target frontiers, plus the root-cone shape when a root-cone parent filter is active.",
        "- `Admission Policy`, `Boundary Admission Outcomes`, and `Boundary Admission Reasons` explain why candidate boundary states became exact histograms, parametric approximations, or skipped rows.",
        "- `Boundary Builders` and `Boundary Cache Build` report precompute cost and payload size. `mean_nodes_or_states` means DFS nodes for the search builder and recurrence states for the recurrence builder.",
        "- `Full Search Versus Boundary Cache` compares full DFS histograms with boundary-stopped histograms. `mean_l1` and `mean_cdf` measure distribution-shape error; path-count, aggregate, and mean-length deltas measure functional error.",
        "- In the comparison table, `mean_node_ratio` and `mean_time_ratio` are cached/full ratios. Values below `1` mean the cached search expanded fewer nodes or ran faster; values above `1` mean extra work or overhead dominated.",
        "- `Search Termination Diagnostics` tells whether rows are complete. If path-count or expansion caps fire, timing and hit rates describe only the enumerated prefix.",
        "- `Cache Hit Geometry` and `Cached Runtime Attribution` explain where hits occur, how much remaining budget they replace, and where cached-side runtime is spent.",
    ]


def boundary_cache_result_implications(selection, cache_rows, comparison_rows):
    lines = ["## Result Implications", ""]
    if cache_rows:
        exact_entries = sum(1 for row in cache_rows if row.get("cached"))
        parametric_entries = sum(1 for row in cache_rows if row.get("parametric_cached"))
        capped_entries = sum(1 for row in cache_rows if row.get("path_cap_hit") or row.get("expansion_cap_hit"))
        skipped_entries = len(cache_rows) - exact_entries - parametric_entries
        lines.append(
            "- Boundary build produced `{}` candidate rows: `{}` exact histogram entries, `{}` parametric entries, and `{}` rows without a cached payload.".format(
                len(cache_rows),
                exact_entries,
                parametric_entries,
                skipped_entries,
            )
        )
        if capped_entries:
            lines.append(
                "- `{}` boundary rows hit a path-count or expansion cap; those cached suffixes should be treated as partial unless the cap was intentional for an approximation experiment.".format(capped_entries)
            )
    else:
        lines.append("- No boundary cache entry rows were generated, so this report cannot evaluate precompute cost or cache admission behavior.")
    if not comparison_rows:
        lines.append("- No target comparison rows were generated; selection and build sections are still useful, but no search-quality or runtime conclusion should be drawn.")
        return lines

    total_rows = len(comparison_rows)
    complete_rows = sum(
        1
        for row in comparison_rows
        if not row.get("full_path_cap_hit")
        and not row.get("full_expansion_cap_hit")
        and not row.get("cached_path_cap_hit")
        and not row.get("cached_expansion_cap_hit")
    )
    max_l1 = max(_numeric_values(comparison_rows, "l1_error"), default=0.0)
    max_cdf = max(_numeric_values(comparison_rows, "max_cdf_error"), default=0.0)
    max_path_relative = max(_numeric_values(comparison_rows, "path_count_relative_error"), default=0.0)
    max_aggregate_relative = max(_numeric_values(comparison_rows, "aggregate_value_relative_error"), default=0.0)
    mean_hits = _mean_numeric(comparison_rows, "cache_hits")
    mean_hist_hits = _mean_numeric(comparison_rows, "histogram_cache_hits")
    mean_param_hits = _mean_numeric(comparison_rows, "parametric_cache_hits")
    mean_node_ratio = _mean_numeric(comparison_rows, "node_expansion_ratio")
    time_ratios = [
        0.0 if int(row.get("full_time_ns", 0)) == 0 else int(row.get("cached_time_ns", 0)) / int(row.get("full_time_ns", 0))
        for row in comparison_rows
    ]
    mean_time_ratio = statistics.mean(time_ratios) if time_ratios else 0.0
    budgets = sorted({row.get("budget") for row in comparison_rows})
    budget_summaries = []
    for budget in budgets:
        rows = [row for row in comparison_rows if row.get("budget") == budget]
        budget_summaries.append(
            "budget `{}`: `{}/{}` complete rows, max L1 `{:.6f}`, max CDF `{:.6f}`, mean cache hits `{:.3f}`".format(
                budget,
                sum(
                    1
                    for row in rows
                    if not row.get("full_path_cap_hit")
                    and not row.get("full_expansion_cap_hit")
                    and not row.get("cached_path_cap_hit")
                    and not row.get("cached_expansion_cap_hit")
                ),
                len(rows),
                max(_numeric_values(rows, "l1_error"), default=0.0),
                max(_numeric_values(rows, "max_cdf_error"), default=0.0),
                _mean_numeric(rows, "cache_hits"),
            )
        )
    lines.append(
        "- Target comparisons completed `{}/{}` rows without path-count or expansion caps. {}.".format(
            complete_rows,
            total_rows,
            "; ".join(budget_summaries),
        )
    )
    lines.append(
        "- Across all comparison rows, max errors were L1 `{:.6f}`, CDF `{:.6f}`, path-count relative `{:.6f}`, and aggregate relative `{:.6f}`.".format(
            max_l1,
            max_cdf,
            max_path_relative,
            max_aggregate_relative,
        )
    )
    if max_l1 == 0.0 and max_cdf == 0.0 and max_path_relative == 0.0 and max_aggregate_relative == 0.0 and complete_rows == total_rows:
        lines.append("- This sampled scope validates the boundary condition semantically: boundary-stopped evaluation matched full DFS for distribution shape, path mass, selected aggregate value, and mean path length.")
    elif complete_rows < total_rows:
        lines.append("- Because at least one comparison row was capped, cache benefit and error should be read as evidence for the enumerated prefix, not a full-cone proof.")
    else:
        lines.append("- Nonzero error means the cached suffix representation diverged from full simple-path DFS on this sample; inspect cycle skips, parametric entries, and support-bound rows before treating the cache as exact.")
    lines.append(
        "- Mean cache hits per row were `{:.3f}` (`{:.3f}` exact-histogram hits and `{:.3f}` parametric hits). Low hit rates mostly validate semantics; higher target budgets or deeper targets are needed to measure speed benefit.".format(
            mean_hits,
            mean_hist_hits,
            mean_param_hits,
        )
    )
    if mean_node_ratio < 1.0 and mean_time_ratio >= 1.0:
        lines.append(
            "- Cached search expanded fewer nodes on average (`mean_node_ratio={:.3f}`) but was slower in this Python run (`mean_time_ratio={:.3f}`), so payload decode, lookup, and instrumentation overhead still dominate at this scale.".format(
                mean_node_ratio,
                mean_time_ratio,
            )
        )
    elif mean_node_ratio < 1.0:
        lines.append(
            "- Cached search reduced node expansions on average (`mean_node_ratio={:.3f}`); this is the main structural signal for whether precompute is paying off.".format(mean_node_ratio)
        )
    else:
        lines.append(
            "- Cached search did not reduce node expansions on average (`mean_node_ratio={:.3f}`); move boundaries closer to expected target paths or increase target budgets before expecting runtime wins.".format(mean_node_ratio)
        )
    if selection.get("parent_filter") == "root-cone":
        lines.append(
            "- Root-cone filtered-skip counts are part of the result: they show how much off-scope parent branching was removed before evaluating boundary-cache behavior."
        )
    return lines


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
        "Target selection: `{}`".format(selection.get("target_selection", "child-depth")),
        "",
        "Parent filter: `{}`".format(selection.get("parent_filter", "all")),
        "",
        "Path length budgets: `{}`".format(",".join(str(value) for value in selection.get("budgets", []))),
        "",
        "Path count cap: `{}`".format(selection.get("path_count_cap")),
        "",
        "Expansion cap: `{}`".format(selection.get("expansion_cap")),
        "",
        "Aggregate kernel: `{}`".format(selection.get("aggregate_kernel", "count")),
        "",
        "Aggregate branching factor: `{}`".format(selection.get("aggregate_branching_factor")),
        "",
        "Aggregate power: `{}`".format(selection.get("aggregate_power")),
        "",
    ]
    lines.extend(boundary_cache_generation_notes(selection))
    lines.extend([""])
    lines.extend(boundary_cache_table_guide())
    lines.extend([""])
    lines.extend(boundary_cache_result_implications(selection, cache_rows, comparison_rows))
    lines.extend([
        "",
        "## Selection",
        "",
        "| role | child_depth | sampled_frontier_nodes |",
        "|------|-------------|------------------------|",
    ])
    for depth, count in _sorted_depth_items(selection.get("boundary_counts", {})):
        lines.append("| boundary | {} | {} |".format(depth, count))
    for depth, count in _sorted_depth_items(selection.get("target_counts", {})):
        lines.append("| target | {} | {} |".format(depth, count))
    if selection.get("parent_filter") == "root-cone":
        lines.extend([
            "",
            "| root_cone_depth | root_cone_nodes | root_cone_children_per_node | root_cone_frontier_limit |",
            "|----------------:|----------------:|----------------------------:|-------------------------:|",
            "| {} | {} | {} | {} |".format(
                selection.get("root_cone_depth"),
                selection.get("root_cone_nodes"),
                selection.get("root_cone_children_per_node"),
                selection.get("root_cone_frontier_limit"),
            ),
            "",
            "| root_cone_child_depth | nodes |",
            "|----------------------:|------:|",
        ])
        for depth, count in _sorted_depth_items(selection.get("root_cone_counts", {})):
            lines.append("| {} | {} |".format(depth, count))
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
        "Here `path_length_budget` is the maximum parent hops in a path. `path_count_cap` is the maximum number of root-reaching paths enumerated before stopping a row.",
        "",
        "| path_length_budget | rows | path_count_cap | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_aggregate_relative_error | mean_abs_aggregate_delta | mean_abs_mean_length_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits | mean_full_filtered_parent_skips | mean_cached_filtered_parent_skips | full_path_count_cap_hits | full_expansion_cap_hits | cached_path_count_cap_hits | cached_expansion_cap_hits |",
        "|-------------------:|-----:|---------------:|---------|--------|--------|----------|-------------------------------:|--------------------:|------------------------------:|-------------------------:|---------------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|----------------------:|-------------------------------:|---------------------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|",
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
        aggregate_relative = [
            float(row["aggregate_value_relative_error"])
            for row in rows
            if row.get("aggregate_value_relative_error") is not None
        ]
        aggregate_delta = [float(row["abs_aggregate_value_delta"]) for row in rows]
        mean_length_delta = [
            float(row["abs_mean_path_length_delta"])
            for row in rows
            if row.get("abs_mean_path_length_delta") is not None
        ]
        ratios = [float(row["node_expansion_ratio"]) for row in rows]
        histogram_hits = [int(row["histogram_cache_hits"]) for row in rows]
        parametric_hits = [int(row["parametric_cache_hits"]) for row in rows]
        histogram_bins_spliced = [int(row["histogram_bins_spliced"]) for row in rows]
        parametric_bins_spliced = [int(row["parametric_bins_spliced"]) for row in rows]
        payload_bytes_read = [int(row["cache_payload_bytes_read"]) for row in rows]
        decode_ns = [int(row["cache_decode_ns"]) for row in rows]
        decode_memo_hits = [int(row.get("cache_decode_memo_hits", 0)) for row in rows]
        full_filtered_skips = [int(row.get("full_root_unreachable_parent_skips", 0)) for row in rows]
        cached_filtered_skips = [int(row.get("cached_root_unreachable_parent_skips", 0)) for row in rows]
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
            "| {budget} | {rows} | {path_count_cap} | {mean_l1:.6f} | {p95_l1:.6f} | {max_l1:.6f} | {mean_cdf:.6f} | {mean_path_relative:.6f} | {mean_path_delta:.3f} | {mean_aggregate_relative:.6f} | {mean_aggregate_delta:.6f} | {mean_length_delta:.6f} | {mean_ratio:.3f} | {mean_time_ratio:.3f} | {mean_full_time:.1f} | {mean_cached_time:.1f} | {mean_hist_hits:.3f} | {mean_param_hits:.3f} | {mean_hist_bins:.3f} | {mean_param_bins:.3f} | {mean_payload_bytes:.3f} | {mean_decode_ns:.1f} | {mean_decode_memo_hits:.3f} | {mean_full_filtered_skips:.3f} | {mean_cached_filtered_skips:.3f} | {full_path_count_cap_hits} | {full_expansion_cap_hits} | {cached_path_count_cap_hits} | {cached_expansion_cap_hits} |".format(
                budget=budget,
                rows=len(rows),
                path_count_cap="n/a" if rows[0].get("path_count_cap") is None else rows[0].get("path_count_cap"),
                mean_l1=statistics.mean(l1) if l1 else 0.0,
                p95_l1=percentile(l1, 95),
                max_l1=max(l1, default=0.0),
                mean_cdf=statistics.mean(cdf) if cdf else 0.0,
                mean_path_relative=statistics.mean(path_relative) if path_relative else 0.0,
                mean_path_delta=statistics.mean(path_delta) if path_delta else 0.0,
                mean_aggregate_relative=statistics.mean(aggregate_relative) if aggregate_relative else 0.0,
                mean_aggregate_delta=statistics.mean(aggregate_delta) if aggregate_delta else 0.0,
                mean_length_delta=statistics.mean(mean_length_delta) if mean_length_delta else 0.0,
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
                mean_decode_memo_hits=statistics.mean(decode_memo_hits) if decode_memo_hits else 0.0,
                mean_full_filtered_skips=statistics.mean(full_filtered_skips) if full_filtered_skips else 0.0,
                mean_cached_filtered_skips=statistics.mean(cached_filtered_skips) if cached_filtered_skips else 0.0,
                mean_splice_ns=statistics.mean(splice_ns) if splice_ns else 0.0,
                mean_parent_lookup_ns=statistics.mean(parent_lookup_ns) if parent_lookup_ns else 0.0,
                mean_probe_ns=statistics.mean(probe_ns) if probe_ns else 0.0,
                mean_path_count_cap_check_ns=statistics.mean(path_cap_check_ns) if path_cap_check_ns else 0.0,
                mean_attributed_ns=statistics.mean(attributed_ns) if attributed_ns else 0.0,
                mean_unattributed_ns=statistics.mean(unattributed_ns) if unattributed_ns else 0.0,
                full_path_count_cap_hits=sum(1 for row in rows if row.get("full_path_cap_hit")),
                full_expansion_cap_hits=sum(1 for row in rows if row.get("full_expansion_cap_hit")),
                cached_path_count_cap_hits=sum(1 for row in rows if row.get("cached_path_cap_hit")),
                cached_expansion_cap_hits=sum(1 for row in rows if row.get("cached_expansion_cap_hit")),
            )
        )
    if comparison_rows:
        lines.extend([
            "",
            "## Search Termination Diagnostics",
            "",
            "`path_count_cap` is a root-reaching path-count cap. It is distinct from `path_length_budget`, which is the maximum number of parent hops allowed in a path.",
            "",
            "If a path-count or expansion cap fires, timing and cache-hit statistics describe only the enumerated prefix. Treat full-run cache benefit as an extrapolation unless the unvisited path mass is estimated and assumed to have comparable boundary-hit statistics.",
            "",
            "| path_length_budget | rows | path_count_cap | full_stop_reasons | cached_stop_reasons | full_length_budget_cutoff_rows | cached_length_budget_cutoff_rows | full_cycle_skips | cached_cycle_skips |",
            "|-------------------:|-----:|---------------:|-------------------|---------------------|-------------------------------:|---------------------------------:|-----------------:|-------------------:|",
        ])
        for budget in sorted(by_budget):
            rows = by_budget[budget]
            full_reasons = Counter(row.get("full_stop_reason", "unknown") for row in rows)
            cached_reasons = Counter(row.get("cached_stop_reason", "unknown") for row in rows)
            lines.append(
                "| {budget} | {rows} | {path_count_cap} | `{full_reasons}` | `{cached_reasons}` | {full_budget_cutoffs} | {cached_budget_cutoffs} | {full_cycle_skips} | {cached_cycle_skips} |".format(
                    budget=budget,
                    rows=len(rows),
                    path_count_cap="n/a" if rows[0].get("path_count_cap") is None else rows[0].get("path_count_cap"),
                    full_reasons=json.dumps(dict(sorted(full_reasons.items())), sort_keys=True),
                    cached_reasons=json.dumps(dict(sorted(cached_reasons.items())), sort_keys=True),
                    full_budget_cutoffs=sum(1 for row in rows if int(row.get("full_length_budget_cutoffs", 0)) > 0),
                    cached_budget_cutoffs=sum(1 for row in rows if int(row.get("cached_length_budget_cutoffs", 0)) > 0),
                    full_cycle_skips=sum(int(row.get("full_cycle_skips", 0)) for row in rows),
                    cached_cycle_skips=sum(int(row.get("cached_cycle_skips", 0)) for row in rows),
                )
            )
    if any(int(row.get("cache_hits", 0)) > 0 for row in comparison_rows):
        lines.extend([
            "",
            "## Cache Hit Geometry",
            "",
            "| path_length_budget | rows | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | mean_first_remaining_budget | mean_max_remaining_budget | hits_rem_ge_2 | hits_rem_ge_4 | hits_rem_ge_6 | hit_depth_histogram | remaining_budget_histogram |",
            "|-------------------:|-----:|----------------:|---------------:|----------------------:|-----------------------:|----------------------------:|--------------------------:|--------------:|--------------:|--------------:|---------------------|----------------------------|",
        ])
        for budget in sorted(by_budget):
            rows = by_budget[budget]
            total_hits = sum(int(row.get("cache_hits", 0)) for row in rows)
            depth_sum = sum(int(row.get("cache_hit_depth_sum", 0)) for row in rows)
            remaining_sum = sum(int(row.get("cache_hit_remaining_budget_sum", 0)) for row in rows)
            suffix_path_sum = sum(int(row.get("cache_hit_suffix_path_count_sum", 0)) for row in rows)
            hit_depths = Counter()
            hit_remaining = Counter()
            first_remaining = [
                int(row["first_cache_hit_remaining_budget"])
                for row in rows
                if row.get("first_cache_hit_remaining_budget") is not None
            ]
            max_remaining = [
                int(row.get("max_cache_hit_remaining_budget", 0))
                for row in rows
                if int(row.get("cache_hits", 0)) > 0
            ]
            for row in rows:
                for depth, count in row.get("cache_hits_by_depth", {}).items():
                    hit_depths[int(depth)] += int(count)
                for remaining, count in row.get("cache_hits_by_remaining_budget", {}).items():
                    hit_remaining[int(remaining)] += int(count)
            lines.append(
                "| {budget} | {rows} | {mean_hits:.3f} | {mean_depth} | {mean_remaining} | {mean_suffix_paths} | {mean_first_remaining} | {mean_max_remaining} | {hits_ge_2} | {hits_ge_4} | {hits_ge_6} | `{depth_hist}` | `{remaining_hist}` |".format(
                    budget=budget,
                    rows=len(rows),
                    mean_hits=statistics.mean(int(row.get("cache_hits", 0)) for row in rows) if rows else 0.0,
                    mean_depth="n/a" if total_hits <= 0 else "{:.3f}".format(depth_sum / total_hits),
                    mean_remaining="n/a" if total_hits <= 0 else "{:.3f}".format(remaining_sum / total_hits),
                    mean_suffix_paths="n/a" if total_hits <= 0 else "{:.3f}".format(suffix_path_sum / total_hits),
                    mean_first_remaining="n/a" if not first_remaining else "{:.3f}".format(statistics.mean(first_remaining)),
                    mean_max_remaining="n/a" if not max_remaining else "{:.3f}".format(statistics.mean(max_remaining)),
                    hits_ge_2=sum(count for remaining, count in hit_remaining.items() if remaining >= 2),
                    hits_ge_4=sum(count for remaining, count in hit_remaining.items() if remaining >= 4),
                    hits_ge_6=sum(count for remaining, count in hit_remaining.items() if remaining >= 6),
                    depth_hist=json.dumps(dict(sorted(hit_depths.items())), sort_keys=True),
                    remaining_hist=json.dumps(dict(sorted(hit_remaining.items())), sort_keys=True),
                )
            )

    if any(row.get("collect_attribution") for row in comparison_rows):
        lines.extend([
            "",
            "## Cached Runtime Attribution",
            "",
            "These columns attribute only the cached search path. `unattributed` is the remaining cached wall time after decode, splice, cache-probe, path-count-cap check, and parent lookup timing buckets.",
            "",
            "| path_length_budget | rows | mean_cached_time_ns | mean_decode_ns | mean_decode_memo_hits | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_count_cap_check_ns | mean_attributed_ns | mean_unattributed_ns | decode_share | splice_share | parent_lookup_share |",
            "|-------------------:|-----:|--------------------:|---------------:|----------------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|-------------:|-------------:|--------------------:|",
        ])
        for budget in sorted(by_budget):
            rows = by_budget[budget]
            cached_times = [int(row["cached_time_ns"]) for row in rows]
            decode_ns = [int(row.get("cache_decode_ns", 0)) for row in rows]
            decode_memo_hits = [int(row.get("cache_decode_memo_hits", 0)) for row in rows]
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
                "| {budget} | {rows} | {cached:.1f} | {decode:.1f} | {decode_memo_hits:.3f} | {splice:.1f} | {parent_lookup:.1f} | {probe:.1f} | {path_cap:.1f} | {attributed:.1f} | {unattributed:.1f} | {decode_share:.3f} | {splice_share:.3f} | {parent_lookup_share:.3f} |".format(
                    budget=budget,
                    rows=len(rows),
                    cached=mean_cached,
                    decode=mean_decode,
                    decode_memo_hits=statistics.mean(decode_memo_hits) if decode_memo_hits else 0.0,
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
    parser.add_argument("--target-selection", choices=["child-depth", "boundary-descendants"], default="child-depth", help="Target sampling mode. boundary-descendants samples targets below selected boundary nodes so cache hits are intentional.")
    parser.add_argument("--include-target-ancestor-boundaries", action="store_true", help="Add ancestors of sampled targets whose root distance matches requested boundary depths.")
    parser.add_argument("--target-ancestor-boundary-limit", type=int, default=500, help="Maximum target-ancestor boundary nodes to add; non-positive means no extra cap.")
    parser.add_argument("--parent-filter", choices=PARENT_FILTERS, default="all", help="Optional parent-edge filter used for boundary construction and target search.")
    parser.add_argument("--root-cone-depth", type=int, default=0, help="Child-depth horizon for --parent-filter root-cone; non-positive derives a horizon from budgets and sampled depths.")
    parser.add_argument("--root-cone-children-per-node", type=int, default=128, help="Child sample cap while building the root cone for --parent-filter root-cone.")
    parser.add_argument("--root-cone-frontier-limit", type=int, default=2000, help="Frontier cap while building the root cone for --parent-filter root-cone.")
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
    parser.add_argument("--path-cap", type=int, default=100000, help="Stop a row after this many root-reaching paths; this is a path-count cap, not the path-length budget.")
    parser.add_argument("--expansion-cap", type=int, default=250000, help="Stop a row after this many expanded nodes.")
    parser.add_argument("--aggregate-kernel", choices=AGGREGATE_VALUE_KERNELS, default="count", help="Path-value aggregate to compare between full DFS and boundary-stopped histograms.")
    parser.add_argument("--aggregate-branching-factor", type=float, default=2.0, help="Branching factor used by --aggregate-kernel bp-decay.")
    parser.add_argument("--aggregate-power", type=float, default=1.0, help="Power used by --aggregate-kernel weighted-power.")
    parser.add_argument("--seed", default="0", help="Deterministic sampling seed.")
    parser.add_argument("--collect-attribution", action="store_true", help="Collect cached-search timing attribution for cache probes, splicing, parent lookups, and path-count-cap checks.")
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
