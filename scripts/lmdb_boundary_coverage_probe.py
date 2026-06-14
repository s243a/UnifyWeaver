#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Probe how often parent paths reach histogram/distribution boundaries.

The boundary-cache runtime benchmark measures speed and histogram error.  This
probe measures the shape of the path space that the cache can actually cover.
In exact mode it enumerates all simple parent-prefixes until root, boundary, or
the path-length budget is reached, subject only to optional safety caps.  In
sample mode it samples simple random parent walks and uses branch-product
proposal corrections to estimate path-prefix counts; those estimates are
statistical and depend on the random-walk proposal distribution.  A separate
path-value kernel, if any, belongs to the measured functional rather than to the
proposal correction.  Boundary-terminating sampling is the cache-boundary
estimator: after a boundary hit, the remaining suffix can be supplied by the
boundary histogram.  Root-walk sampling is the no-boundary
search-space estimator: it ignores boundaries and walks to root, budget
exhaustion, or dead end.  The default parent filter is root-cone: a bounded
child-reachable cone is precomputed from the root, then a parent edge is
followed only when the parent is in that cone and its root-depth fits within
the remaining path budget.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

from scripts.lmdb_parent_boundary_cache_benchmark import (
    collect_target_ancestor_boundaries,
    select_targets_by_boundary_descendants,
)
from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    deterministic_sample,
    parse_int_list,
    select_targets_by_child_depth,
)
from scripts.lmdb_parent_histogram_benchmark import bounded_parent_histogram, safe_graph_name


DEFAULT_BOUNDARY_DEPTHS = [2]
DEFAULT_TARGET_DEPTHS = [4]


@dataclass
class CoverageStats:
    terminal_prefixes: int = 0
    root_paths: int = 0
    boundary_hits: int = 0
    budget_exhausted_prefixes: int = 0
    dead_end_prefixes: int = 0
    nodes_expanded: int = 0
    edges_examined: int = 0
    cycle_skips: int = 0
    root_unreachable_parent_skips: int = 0
    filtered_dead_end_prefixes: int = 0
    path_count_cap_hit: bool = False
    expansion_cap_hit: bool = False
    boundary_hit_depth_sum: int = 0
    boundary_hit_remaining_budget_sum: int = 0
    boundary_suffix_path_mass_sum: int = 0
    boundary_hits_by_depth: Counter | None = None
    boundary_hits_by_remaining_budget: Counter | None = None
    boundary_hits_by_node: Counter | None = None

    def __post_init__(self):
        if self.boundary_hits_by_depth is None:
            self.boundary_hits_by_depth = Counter()
        if self.boundary_hits_by_remaining_budget is None:
            self.boundary_hits_by_remaining_budget = Counter()
        if self.boundary_hits_by_node is None:
            self.boundary_hits_by_node = Counter()


class RootReachabilityFilter:
    """Memoized finite-horizon predicate for root-reaching parent candidates."""

    def __init__(self, parents_func, root):
        self.parents_func = parents_func
        self.root = root
        self.memo = {}
        self.checks = 0
        self.memo_hits = 0
        self.cycle_skips = 0

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


class RootConeFilter:
    """Constant-time finite-horizon predicate from a precomputed root cone."""

    def __init__(self, depth_by_node):
        self.depth_by_node = dict(depth_by_node)
        self.checks = 0
        self.depth_misses = 0
        self.remaining_misses = 0

    def can_reach(self, node, remaining):
        self.checks += 1
        depth = self.depth_by_node.get(node)
        if depth is None:
            self.depth_misses += 1
            return False
        if int(depth) > int(remaining):
            self.remaining_misses += 1
            return False
        return True


def fraction(numerator, denominator):
    return None if denominator <= 0 else numerator / denominator


def normalize_limit(limit):
    return None if limit is None or int(limit) <= 0 else int(limit)


def build_root_cone(children_func, root, max_depth, children_per_node, frontier_limit, seed):
    """Build a bounded child-reachable cone from root with minimum depths."""
    max_depth = int(max_depth)
    children_limit = normalize_limit(children_per_node)
    frontier_limit = normalize_limit(frontier_limit)
    depth_by_node = {root: 0}
    frontier = [root]
    counts = {0: 1}

    for depth in range(1, max_depth + 1):
        next_nodes = []
        for node in frontier:
            children = deterministic_sample(
                children_func(node),
                children_limit,
                "{}:children:{}:{}".format(seed, depth, node),
            )
            next_nodes.extend(children)

        sampled = deterministic_sample(
            list(dict.fromkeys(next_nodes)),
            frontier_limit,
            "{}:frontier:{}".format(seed, depth),
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


def select_nodes_by_root_cone_depth(depth_by_node, depths, nodes_per_depth, seed):
    """Sample nodes from a precomputed root cone by minimum child depth."""
    requested = [int(depth) for depth in depths]
    buckets = {depth: [] for depth in requested}
    for node, depth in depth_by_node.items():
        if int(depth) in buckets:
            buckets[int(depth)].append(node)

    selected = []
    selected_depth_by_node = {}
    counts = {}
    limit = normalize_limit(nodes_per_depth)
    for depth in requested:
        candidates = sorted(buckets.get(depth, []))
        counts[depth] = len(candidates)
        sampled = deterministic_sample(
            candidates,
            limit,
            "{}:depth:{}".format(seed, depth),
        )
        for node in sampled:
            if node not in selected_depth_by_node:
                selected.append(node)
                selected_depth_by_node[node] = depth
    return selected, selected_depth_by_node, counts


def suffix_path_mass(parents_func, boundary_node, root, remaining, memo):
    key = (boundary_node, int(remaining))
    if key not in memo:
        hist, stats = bounded_parent_histogram(parents_func, boundary_node, root, int(remaining))
        memo[key] = {
            "path_count": sum(hist.values()),
            "support_bins": len(hist),
            "path_count_cap_hit": stats.path_cap_hit,
            "expansion_cap_hit": stats.expansion_cap_hit,
            "cycle_skips": stats.cycle_skips,
        }
    return memo[key]


def exact_boundary_coverage(
    parents_func,
    target,
    root,
    budget,
    boundary_nodes,
    path_count_cap=None,
    expansion_cap=None,
    reachability_filter=None,
    parent_filter_name="all",
    measure_suffix_mass=True,
):
    """Enumerate simple parent prefixes until root, boundary, or budget."""
    budget = int(budget)
    boundary_nodes = set(boundary_nodes)
    path_count_cap = None if path_count_cap is None or int(path_count_cap) <= 0 else int(path_count_cap)
    expansion_cap = None if expansion_cap is None or int(expansion_cap) <= 0 else int(expansion_cap)
    stats = CoverageStats()
    suffix_memo = {}
    suffix_measure_compatible = measure_suffix_mass and reachability_filter is None

    def terminal_reached():
        if path_count_cap is not None and stats.terminal_prefixes >= path_count_cap:
            stats.path_count_cap_hit = True
            return True
        return False

    def dfs(node, remaining, depth, visited):
        if expansion_cap is not None and stats.nodes_expanded >= expansion_cap:
            stats.expansion_cap_hit = True
            return
        stats.nodes_expanded += 1

        if node == root:
            stats.terminal_prefixes += 1
            stats.root_paths += 1
            terminal_reached()
            return

        if depth > 0 and node in boundary_nodes:
            stats.terminal_prefixes += 1
            stats.boundary_hits += 1
            stats.boundary_hit_depth_sum += int(depth)
            stats.boundary_hit_remaining_budget_sum += int(remaining)
            stats.boundary_hits_by_depth[int(depth)] += 1
            stats.boundary_hits_by_remaining_budget[int(remaining)] += 1
            stats.boundary_hits_by_node[node] += 1
            if suffix_measure_compatible:
                suffix = suffix_path_mass(parents_func, node, root, remaining, suffix_memo)
                stats.boundary_suffix_path_mass_sum += int(suffix["path_count"])
            terminal_reached()
            return

        if remaining <= 0:
            stats.terminal_prefixes += 1
            stats.budget_exhausted_prefixes += 1
            terminal_reached()
            return

        parents = list(parents_func(node))
        eligible = []
        filtered_parents = 0
        for parent in parents:
            stats.edges_examined += 1
            if parent in visited:
                stats.cycle_skips += 1
            elif reachability_filter is not None and not reachability_filter(parent, remaining - 1):
                stats.root_unreachable_parent_skips += 1
                filtered_parents += 1
            else:
                eligible.append(parent)
        if not eligible:
            stats.terminal_prefixes += 1
            if filtered_parents > 0:
                stats.filtered_dead_end_prefixes += 1
            else:
                stats.dead_end_prefixes += 1
            terminal_reached()
            return

        for parent in eligible:
            if stats.path_count_cap_hit or stats.expansion_cap_hit:
                return
            visited.add(parent)
            dfs(parent, remaining - 1, depth + 1, visited)
            visited.remove(parent)

    dfs(target, budget, 0, {target})
    return coverage_record(
        mode="exact",
        target=target,
        budget=budget,
        stats=stats,
        samples=None,
        weighted=None,
        elapsed_ns=None,
        parent_filter_name=parent_filter_name,
        measure_suffix_mass=suffix_measure_compatible,
    )


def weighted_mean(values):
    return sum(values) / len(values) if values else 0.0


def standard_error(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def bootstrap_ratio(boundary_weights, terminal_weights, seed, reps=200):
    if not boundary_weights or not terminal_weights or len(boundary_weights) != len(terminal_weights):
        return None, None
    rng = random.Random(seed)
    ratios = []
    n = len(boundary_weights)
    for _ in range(int(reps)):
        boundary_total = 0.0
        terminal_total = 0.0
        for _sample in range(n):
            index = rng.randrange(n)
            boundary_total += boundary_weights[index]
            terminal_total += terminal_weights[index]
        if terminal_total > 0.0:
            ratios.append(boundary_total / terminal_total)
    if not ratios:
        return None, None
    ratios.sort()
    lo = ratios[int(0.025 * (len(ratios) - 1))]
    hi = ratios[int(0.975 * (len(ratios) - 1))]
    return lo, hi


def sample_boundary_coverage(
    parents_func,
    target,
    root,
    budget,
    boundary_nodes,
    samples,
    seed,
    reachability_filter=None,
    parent_filter_name="all",
    measure_suffix_mass=True,
):
    """Sample random parent walks and estimate prefix counts by proposal correction.

    The branch product corrects for the local random-walk proposal.  Boundary
    suffix mass is a value supplied by the cached histogram for the remaining
    budget; metric-specific kernels require matching suffix bases.
    """
    budget = int(budget)
    boundary_nodes = set(boundary_nodes)
    rng = random.Random(str(seed))
    stats = CoverageStats()
    suffix_memo = {}
    suffix_measure_compatible = measure_suffix_mass and reachability_filter is None
    terminal_weights = []
    root_weights = []
    boundary_weights = []
    boundary_spliced_weights = []
    budget_weights = []
    dead_end_weights = []
    suffix_mass_weights = []

    for _sample in range(int(samples)):
        node = target
        remaining = budget
        depth = 0
        visited = {target}
        weight = 1.0

        while True:
            stats.nodes_expanded += 1
            if node == root:
                stats.terminal_prefixes += 1
                stats.root_paths += 1
                terminal_weights.append(weight)
                root_weights.append(weight)
                boundary_weights.append(0.0)
                boundary_spliced_weights.append(0.0)
                budget_weights.append(0.0)
                dead_end_weights.append(0.0)
                suffix_mass_weights.append(0.0)
                break

            if depth > 0 and node in boundary_nodes:
                stats.terminal_prefixes += 1
                stats.boundary_hits += 1
                stats.boundary_hit_depth_sum += int(depth)
                stats.boundary_hit_remaining_budget_sum += int(remaining)
                stats.boundary_hits_by_depth[int(depth)] += 1
                stats.boundary_hits_by_remaining_budget[int(remaining)] += 1
                stats.boundary_hits_by_node[node] += 1
                suffix_mass = 0
                if suffix_measure_compatible:
                    suffix = suffix_path_mass(parents_func, node, root, remaining, suffix_memo)
                    suffix_mass = int(suffix["path_count"])
                    stats.boundary_suffix_path_mass_sum += suffix_mass
                terminal_weights.append(weight)
                root_weights.append(0.0)
                boundary_weights.append(weight)
                boundary_spliced_weights.append(weight * suffix_mass)
                budget_weights.append(0.0)
                dead_end_weights.append(0.0)
                suffix_mass_weights.append(weight * suffix_mass)
                break

            if remaining <= 0:
                stats.terminal_prefixes += 1
                stats.budget_exhausted_prefixes += 1
                terminal_weights.append(weight)
                root_weights.append(0.0)
                boundary_weights.append(0.0)
                boundary_spliced_weights.append(0.0)
                budget_weights.append(weight)
                dead_end_weights.append(0.0)
                suffix_mass_weights.append(0.0)
                break

            parents = list(parents_func(node))
            eligible = []
            filtered_parents = 0
            for parent in parents:
                stats.edges_examined += 1
                if parent in visited:
                    stats.cycle_skips += 1
                elif reachability_filter is not None and not reachability_filter(parent, remaining - 1):
                    stats.root_unreachable_parent_skips += 1
                    filtered_parents += 1
                else:
                    eligible.append(parent)
            if not eligible:
                stats.terminal_prefixes += 1
                if filtered_parents > 0:
                    stats.filtered_dead_end_prefixes += 1
                else:
                    stats.dead_end_prefixes += 1
                terminal_weights.append(weight)
                root_weights.append(0.0)
                boundary_weights.append(0.0)
                boundary_spliced_weights.append(0.0)
                budget_weights.append(0.0)
                dead_end_weights.append(weight)
                suffix_mass_weights.append(0.0)
                break

            weight *= len(eligible)
            parent = eligible[rng.randrange(len(eligible))]
            visited.add(parent)
            node = parent
            remaining -= 1
            depth += 1

    weighted = {
        "estimated_terminal_prefixes": weighted_mean(terminal_weights),
        "estimated_terminal_prefixes_se": standard_error(terminal_weights),
        "estimated_root_paths": weighted_mean(root_weights),
        "estimated_root_paths_se": standard_error(root_weights),
        "estimated_boundary_hit_prefixes": weighted_mean(boundary_weights),
        "estimated_boundary_hit_prefixes_se": standard_error(boundary_weights),
        "estimated_boundary_spliced_root_paths": weighted_mean(boundary_spliced_weights),
        "estimated_boundary_spliced_root_paths_se": standard_error(boundary_spliced_weights),
        "estimated_budget_exhausted_prefixes": weighted_mean(budget_weights),
        "estimated_budget_exhausted_prefixes_se": standard_error(budget_weights),
        "estimated_dead_end_prefixes": weighted_mean(dead_end_weights),
        "estimated_dead_end_prefixes_se": standard_error(dead_end_weights),
        "estimated_boundary_suffix_path_mass": weighted_mean(suffix_mass_weights),
        "estimated_boundary_suffix_path_mass_se": standard_error(suffix_mass_weights),
    }
    if suffix_measure_compatible:
        weighted["estimated_spliced_total_root_paths"] = (
            weighted["estimated_root_paths"] + weighted["estimated_boundary_spliced_root_paths"]
        )
    else:
        weighted["estimated_boundary_spliced_root_paths"] = None
        weighted["estimated_boundary_spliced_root_paths_se"] = None
        weighted["estimated_spliced_total_root_paths"] = None
        weighted["estimated_boundary_suffix_path_mass"] = None
        weighted["estimated_boundary_suffix_path_mass_se"] = None
    total_estimate = weighted["estimated_terminal_prefixes"]
    boundary_estimate = weighted["estimated_boundary_hit_prefixes"]
    weighted["estimated_boundary_hit_fraction"] = None if total_estimate <= 0.0 else boundary_estimate / total_estimate
    ci_low, ci_high = bootstrap_ratio(boundary_weights, terminal_weights, str(seed) + ":bootstrap")
    weighted["estimated_boundary_hit_fraction_ci95_low"] = ci_low
    weighted["estimated_boundary_hit_fraction_ci95_high"] = ci_high

    return coverage_record(
        mode="sample",
        target=target,
        budget=budget,
        stats=stats,
        samples=int(samples),
        weighted=weighted,
        elapsed_ns=None,
        parent_filter_name=parent_filter_name,
        measure_suffix_mass=suffix_measure_compatible,
    )


def sample_root_path_space(
    parents_func,
    target,
    root,
    budget,
    boundary_nodes,
    samples,
    seed,
    reachability_filter=None,
    parent_filter_name="all",
):
    """Sample full simple parent walks and estimate root-path search space.

    For a uniformly sampled simple parent walk, the product of eligible branch
    counts along the walk is the inverse-proposal correction for the size of the
    sampled path space.  Path length is the first concrete mean reported here;
    arbitrary property means need their property-specific numerator derived
    before adding them to this probe.
    """
    budget = int(budget)
    boundary_nodes = set(boundary_nodes)
    rng = random.Random(str(seed))
    stats = CoverageStats()
    terminal_weights = []
    root_weights = []
    boundary_weights = []
    root_boundary_weights = []
    root_length_weights = []
    budget_weights = []
    dead_end_weights = []

    for _sample in range(int(samples)):
        node = target
        remaining = budget
        depth = 0
        visited = {target}
        weight = 1.0
        boundary_seen = False

        while True:
            stats.nodes_expanded += 1
            if depth > 0 and not boundary_seen and node in boundary_nodes:
                boundary_seen = True
                stats.boundary_hits += 1
                stats.boundary_hit_depth_sum += int(depth)
                stats.boundary_hit_remaining_budget_sum += int(remaining)
                stats.boundary_hits_by_depth[int(depth)] += 1
                stats.boundary_hits_by_remaining_budget[int(remaining)] += 1
                stats.boundary_hits_by_node[node] += 1

            if node == root:
                stats.terminal_prefixes += 1
                stats.root_paths += 1
                terminal_weights.append(weight)
                root_weights.append(weight)
                boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_weights.append(weight if boundary_seen else 0.0)
                root_length_weights.append(weight * depth)
                budget_weights.append(0.0)
                dead_end_weights.append(0.0)
                break

            if remaining <= 0:
                stats.terminal_prefixes += 1
                stats.budget_exhausted_prefixes += 1
                terminal_weights.append(weight)
                root_weights.append(0.0)
                boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_weights.append(0.0)
                root_length_weights.append(0.0)
                budget_weights.append(weight)
                dead_end_weights.append(0.0)
                break

            parents = list(parents_func(node))
            eligible = []
            filtered_parents = 0
            for parent in parents:
                stats.edges_examined += 1
                if parent in visited:
                    stats.cycle_skips += 1
                elif reachability_filter is not None and not reachability_filter(parent, remaining - 1):
                    stats.root_unreachable_parent_skips += 1
                    filtered_parents += 1
                else:
                    eligible.append(parent)
            if not eligible:
                stats.terminal_prefixes += 1
                if filtered_parents > 0:
                    stats.filtered_dead_end_prefixes += 1
                else:
                    stats.dead_end_prefixes += 1
                terminal_weights.append(weight)
                root_weights.append(0.0)
                boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_weights.append(0.0)
                root_length_weights.append(0.0)
                budget_weights.append(0.0)
                dead_end_weights.append(weight)
                break

            weight *= len(eligible)
            parent = eligible[rng.randrange(len(eligible))]
            visited.add(parent)
            node = parent
            remaining -= 1
            depth += 1

    weighted = {
        "estimated_terminal_prefixes": weighted_mean(terminal_weights),
        "estimated_terminal_prefixes_se": standard_error(terminal_weights),
        "estimated_root_paths": weighted_mean(root_weights),
        "estimated_root_paths_se": standard_error(root_weights),
        "estimated_boundary_hit_prefixes": weighted_mean(boundary_weights),
        "estimated_boundary_hit_prefixes_se": standard_error(boundary_weights),
        "estimated_root_boundary_hit_paths": weighted_mean(root_boundary_weights),
        "estimated_root_boundary_hit_paths_se": standard_error(root_boundary_weights),
        "estimated_root_path_length_sum": weighted_mean(root_length_weights),
        "estimated_root_path_length_sum_se": standard_error(root_length_weights),
        "estimated_budget_exhausted_prefixes": weighted_mean(budget_weights),
        "estimated_budget_exhausted_prefixes_se": standard_error(budget_weights),
        "estimated_dead_end_prefixes": weighted_mean(dead_end_weights),
        "estimated_dead_end_prefixes_se": standard_error(dead_end_weights),
    }
    total_estimate = weighted["estimated_terminal_prefixes"]
    boundary_estimate = weighted["estimated_boundary_hit_prefixes"]
    root_estimate = weighted["estimated_root_paths"]
    root_boundary_estimate = weighted["estimated_root_boundary_hit_paths"]
    weighted["estimated_boundary_hit_fraction"] = None if total_estimate <= 0.0 else boundary_estimate / total_estimate
    weighted["estimated_root_boundary_hit_fraction"] = None if root_estimate <= 0.0 else root_boundary_estimate / root_estimate
    weighted["estimated_mean_root_path_length"] = (
        None if root_estimate <= 0.0 else weighted["estimated_root_path_length_sum"] / root_estimate
    )
    ci_low, ci_high = bootstrap_ratio(boundary_weights, terminal_weights, str(seed) + ":bootstrap")
    weighted["estimated_boundary_hit_fraction_ci95_low"] = ci_low
    weighted["estimated_boundary_hit_fraction_ci95_high"] = ci_high

    return coverage_record(
        mode="root-sample",
        target=target,
        budget=budget,
        stats=stats,
        samples=int(samples),
        weighted=weighted,
        elapsed_ns=None,
        parent_filter_name=parent_filter_name,
        measure_suffix_mass=False,
    )


def coverage_record(
    mode,
    target,
    budget,
    stats,
    samples=None,
    weighted=None,
    elapsed_ns=None,
    parent_filter_name="all",
    measure_suffix_mass=True,
):
    boundary_hits = int(stats.boundary_hits)
    record = {
        "record_type": "boundary_coverage_target",
        "mode": mode,
        "parent_filter": parent_filter_name,
        "target_node": target,
        "path_length_budget": int(budget),
        "samples": samples,
        "terminal_prefixes": int(stats.terminal_prefixes),
        "root_paths": int(stats.root_paths),
        "boundary_hit_prefixes": boundary_hits,
        "budget_exhausted_prefixes": int(stats.budget_exhausted_prefixes),
        "dead_end_prefixes": int(stats.dead_end_prefixes),
        "filtered_dead_end_prefixes": int(stats.filtered_dead_end_prefixes),
        "boundary_hit_fraction": fraction(boundary_hits, int(stats.terminal_prefixes)),
        "root_path_fraction": fraction(int(stats.root_paths), int(stats.terminal_prefixes)),
        "budget_exhausted_fraction": fraction(int(stats.budget_exhausted_prefixes), int(stats.terminal_prefixes)),
        "nodes_expanded": int(stats.nodes_expanded),
        "edges_examined": int(stats.edges_examined),
        "cycle_skips": int(stats.cycle_skips),
        "root_unreachable_parent_skips": int(stats.root_unreachable_parent_skips),
        "path_count_cap_hit": bool(stats.path_count_cap_hit),
        "expansion_cap_hit": bool(stats.expansion_cap_hit),
        "completed": not stats.path_count_cap_hit and not stats.expansion_cap_hit,
        "mean_boundary_hit_depth": None if boundary_hits <= 0 else stats.boundary_hit_depth_sum / boundary_hits,
        "mean_boundary_hit_remaining_budget": None if boundary_hits <= 0 else stats.boundary_hit_remaining_budget_sum / boundary_hits,
        "boundary_suffix_mass_measured": bool(measure_suffix_mass),
        "mean_boundary_suffix_path_mass": None if not measure_suffix_mass or boundary_hits <= 0 else stats.boundary_suffix_path_mass_sum / boundary_hits,
        "boundary_suffix_path_mass_sum": int(stats.boundary_suffix_path_mass_sum),
        "boundary_hits_by_depth": dict(sorted(stats.boundary_hits_by_depth.items())),
        "boundary_hits_by_remaining_budget": dict(sorted(stats.boundary_hits_by_remaining_budget.items())),
        "boundary_hits_by_node": dict(sorted(stats.boundary_hits_by_node.items())),
        "elapsed_ns": elapsed_ns,
    }
    if weighted:
        record.update(weighted)
    return record


def time_target(callable_fn):
    started = time.perf_counter_ns()
    record = callable_fn()
    record["elapsed_ns"] = time.perf_counter_ns() - started
    return record


def aggregate_rows(rows):
    if not rows:
        return {}
    terminal = sum(int(row["terminal_prefixes"]) for row in rows)
    boundary = sum(int(row["boundary_hit_prefixes"]) for row in rows)
    root = sum(int(row["root_paths"]) for row in rows)
    budget = sum(int(row["budget_exhausted_prefixes"]) for row in rows)
    filtered_dead = sum(int(row.get("filtered_dead_end_prefixes", 0)) for row in rows)
    suffix_mass = sum(int(row["boundary_suffix_path_mass_sum"]) for row in rows)
    suffix_mass_measured = all(bool(row.get("boundary_suffix_mass_measured", True)) for row in rows)
    return {
        "targets": len(rows),
        "terminal_prefixes": terminal,
        "root_paths": root,
        "boundary_hit_prefixes": boundary,
        "budget_exhausted_prefixes": budget,
        "filtered_dead_end_prefixes": filtered_dead,
        "boundary_hit_fraction": fraction(boundary, terminal),
        "root_path_fraction": fraction(root, terminal),
        "budget_exhausted_fraction": fraction(budget, terminal),
        "boundary_suffix_path_mass_sum": suffix_mass,
        "boundary_suffix_mass_measured": suffix_mass_measured,
        "mean_boundary_suffix_path_mass": None if not suffix_mass_measured or boundary <= 0 else suffix_mass / boundary,
        "completed_targets": sum(1 for row in rows if row.get("completed")),
        "path_count_cap_hit_targets": sum(1 for row in rows if row.get("path_count_cap_hit")),
        "expansion_cap_hit_targets": sum(1 for row in rows if row.get("expansion_cap_hit")),
        "cycle_skips": sum(int(row.get("cycle_skips", 0)) for row in rows),
        "root_unreachable_parent_skips": sum(int(row.get("root_unreachable_parent_skips", 0)) for row in rows),
    }


def format_optional(value, digits=3):
    if value is None:
        return "n/a"
    return ("{:." + str(digits) + "f}").format(float(value))


def mean_optional_field(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else statistics.mean(values)


def summarize(records):
    selection = next((row for row in records if row.get("record_type") == "boundary_coverage_selection"), {})
    target_rows = [row for row in records if row.get("record_type") == "boundary_coverage_target"]
    by_mode_budget = {}
    for row in target_rows:
        by_mode_budget.setdefault((row["mode"], row["path_length_budget"]), []).append(row)

    lines = [
        "# LMDB Boundary Coverage Probe",
        "",
        "Graph: `{}`".format(selection.get("graph", "")),
        "",
        "Root: `{}`".format(selection.get("root", "")),
        "",
        "Target selection: `{}`".format(selection.get("target_selection", "")),
        "",
        "Selection source: `{}`".format(selection.get("selection_source", "graph")),
        "",
        "Parent filter: `{}`".format(selection.get("parent_filter", "all")),
        "",
        "Root cone depth: `{}`".format(selection.get("root_cone_depth", "n/a")),
        "",
        "Root cone nodes: `{}`".format(selection.get("root_cone_nodes", 0)),
        "",
        "Boundary suffix mass measured: `{}`".format(selection.get("measure_boundary_suffix_mass", True)),
        "",
        "Boundary nodes: `{}`".format(selection.get("boundary_nodes", 0)),
        "",
        "Targets: `{}`".format(selection.get("targets", 0)),
        "",
        "Path length budgets: `{}`".format(",".join(str(value) for value in selection.get("budgets", []))),
        "",
        "Exact mode enumerates simple parent-prefixes until root, boundary, or the path-length budget. Sample mode uses branch-product weighted random walks; its path-count totals are estimates, not direct counts. With `root-reachable`, parent expansion checks finite-horizon reachability recursively. With `root-cone`, parent expansion uses a precomputed root cone and keeps only parent nodes whose cone depth fits within the remaining path budget.",
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
    if selection.get("root_cone_counts"):
        lines.extend([
            "",
            "## Root Cone",
            "",
            "| child_depth | new_nodes |",
            "|------------:|----------:|",
        ])
        for depth, count in sorted(selection.get("root_cone_counts", {}).items()):
            lines.append("| {} | {} |".format(depth, count))

    lines.extend([
        "",
        "## Coverage Summary",
        "",
        "For sample and root-sample modes, these are observed random-walk outcomes. Use the estimate sections below for branch-product weighted path-space estimates.",
        "",
        "| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips |",
        "|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|----------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|",
    ])
    for (mode, budget), rows in sorted(by_mode_budget.items()):
        aggregate = aggregate_rows(rows)
        lines.append(
            "| {mode} | {budget} | {targets} | {completed} | {terminal} | {root} | {boundary} | {boundary_fraction} | {budget_exhausted} | {filtered_dead} | {suffix_mean} | {path_cap} | {expansion_cap} | {cycle_skips} | {root_unreachable} |".format(
                mode=mode,
                budget=budget,
                targets=aggregate["targets"],
                completed=aggregate["completed_targets"],
                terminal=aggregate["terminal_prefixes"],
                root=aggregate["root_paths"],
                boundary=aggregate["boundary_hit_prefixes"],
                boundary_fraction=format_optional(aggregate["boundary_hit_fraction"], 6),
                budget_exhausted=aggregate["budget_exhausted_prefixes"],
                filtered_dead=aggregate["filtered_dead_end_prefixes"],
                suffix_mean=format_optional(aggregate["mean_boundary_suffix_path_mass"], 3),
                path_cap=aggregate["path_count_cap_hit_targets"],
                expansion_cap=aggregate["expansion_cap_hit_targets"],
                cycle_skips=aggregate["cycle_skips"],
                root_unreachable=aggregate["root_unreachable_parent_skips"],
            )
        )

    sampled = [row for row in target_rows if row.get("mode") == "sample"]
    if sampled:
        lines.extend([
            "",
            "## Boundary Sample Estimates",
            "",
            "Boundary sample mode stops at the first boundary. `estimated_spliced_total_root_paths` adds direct root-path weight to boundary-prefix weight multiplied by the remaining-budget suffix mass.",
            "",
            "| path_length_budget | targets | samples_per_target | mean_estimated_terminal_prefixes | mean_estimated_boundary_hit_fraction | mean_estimated_spliced_total_root_paths | mean_ci95_low | mean_ci95_high |",
            "|-------------------:|--------:|-------------------:|---------------------------------:|------------------------------------:|---------------------------------------:|--------------:|---------------:|",
        ])
        by_budget = {}
        for row in sampled:
            by_budget.setdefault(row["path_length_budget"], []).append(row)
        for budget, rows in sorted(by_budget.items()):
            lines.append(
                "| {budget} | {targets} | {samples} | {terminal} | {fraction} | {spliced} | {lo} | {hi} |".format(
                    budget=budget,
                    targets=len(rows),
                    samples=rows[0].get("samples"),
                    terminal=format_optional(mean_optional_field(rows, "estimated_terminal_prefixes"), 3),
                    fraction=format_optional(mean_optional_field(rows, "estimated_boundary_hit_fraction"), 6),
                    spliced=format_optional(mean_optional_field(rows, "estimated_spliced_total_root_paths"), 3),
                    lo=format_optional(mean_optional_field(rows, "estimated_boundary_hit_fraction_ci95_low"), 6),
                    hi=format_optional(mean_optional_field(rows, "estimated_boundary_hit_fraction_ci95_high"), 6),
                )
            )

    root_sampled = [row for row in target_rows if row.get("mode") == "root-sample"]
    if root_sampled:
        lines.extend([
            "",
            "## Root Path Sample Estimates",
            "",
            "Root-sample mode ignores boundary stopping and walks until root, budget exhaustion, or dead end. `estimated_root_paths` estimates the root-reaching search-space size from the branch-product weight. `estimated_mean_root_path_length` is the corresponding weighted mean path length.",
            "",
            "| path_length_budget | targets | samples_per_target | mean_estimated_root_paths | mean_estimated_mean_root_path_length | mean_estimated_root_boundary_hit_fraction |",
            "|-------------------:|--------:|-------------------:|--------------------------:|-------------------------------------:|------------------------------------------:|",
        ])
        by_budget = {}
        for row in root_sampled:
            by_budget.setdefault(row["path_length_budget"], []).append(row)
        for budget, rows in sorted(by_budget.items()):
            lines.append(
                "| {budget} | {targets} | {samples} | {root_paths} | {mean_len} | {boundary_fraction} |".format(
                    budget=budget,
                    targets=len(rows),
                    samples=rows[0].get("samples"),
                    root_paths=format_optional(mean_optional_field(rows, "estimated_root_paths"), 3),
                    mean_len=format_optional(mean_optional_field(rows, "estimated_mean_root_path_length"), 3),
                    boundary_fraction=format_optional(mean_optional_field(rows, "estimated_root_boundary_hit_fraction"), 6),
                )
            )

    lines.extend([
        "",
        "## Target Rows",
        "",
        "| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips |",
        "|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|",
    ])
    for row in target_rows:
        lines.append(
            "| {mode} | {target} | {budget} | {terminal} | {root} | {boundary} | {fraction} | {budget_exhausted} | {filtered_dead} | {remaining} | {completed} | {cycles} | {root_unreachable} |".format(
                mode=row["mode"],
                target=row["target_node"],
                budget=row["path_length_budget"],
                terminal=row["terminal_prefixes"],
                root=row["root_paths"],
                boundary=row["boundary_hit_prefixes"],
                fraction=format_optional(row["boundary_hit_fraction"], 6),
                budget_exhausted=row["budget_exhausted_prefixes"],
                filtered_dead=row.get("filtered_dead_end_prefixes", 0),
                remaining=format_optional(row["mean_boundary_hit_remaining_budget"], 3),
                completed="yes" if row.get("completed") else "no",
                cycles=row.get("cycle_skips", 0),
                root_unreachable=row.get("root_unreachable_parent_skips", 0),
            )
        )

    return "\n".join(lines) + "\n"


def run_probe(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        boundary_depths = parse_int_list(args.boundary_depths)
        target_depths = parse_int_list(args.target_depths)
        budgets = parse_int_list(args.budgets)
        max_requested_depth = max(boundary_depths + target_depths + budgets, default=0)

        root_cone_depth_by_node = None
        root_cone_counts = {}
        root_cone_depth = int(args.root_cone_depth) if int(args.root_cone_depth) > 0 else max_requested_depth
        if args.parent_filter == "root-cone" or args.selection_source == "root-cone":
            root_cone_depth_by_node, root_cone_counts = build_root_cone(
                graph.children,
                args.root,
                root_cone_depth,
                args.root_cone_children_per_node if args.root_cone_children_per_node is not None else args.children_per_node,
                args.root_cone_frontier_limit if args.root_cone_frontier_limit is not None else args.frontier_limit,
                args.seed + ":root-cone",
            )

        if args.selection_source == "root-cone":
            boundary_nodes, boundary_depth_by_node, boundary_counts = select_nodes_by_root_cone_depth(
                root_cone_depth_by_node,
                boundary_depths,
                args.boundaries_per_depth,
                args.seed + ":root-cone-boundary",
            )
            targets, target_depth_by_node, target_counts = select_nodes_by_root_cone_depth(
                root_cone_depth_by_node,
                target_depths,
                args.targets_per_depth,
                args.seed + ":root-cone-target",
            )
        else:
            boundary_nodes, boundary_depth_by_node, boundary_counts = select_targets_by_child_depth(
                graph,
                args.root,
                boundary_depths,
                args.children_per_node,
                args.frontier_limit,
                args.boundaries_per_depth,
                args.seed + ":boundary",
            )
            if args.target_selection == "boundary-descendants":
                targets, target_depth_by_node, target_counts = select_targets_by_boundary_descendants(
                    graph,
                    boundary_depth_by_node,
                    target_depths,
                    args.children_per_node,
                    args.frontier_limit,
                    args.targets_per_depth,
                    args.seed + ":boundary-descendant-target",
                )
            else:
                targets, target_depth_by_node, target_counts = select_targets_by_child_depth(
                    graph,
                    args.root,
                    target_depths,
                    args.children_per_node,
                    args.frontier_limit,
                    args.targets_per_depth,
                    args.seed + ":target",
                )

        if args.selection_source == "root-cone":
            target_selection_label = "root-cone-child-depth"
        else:
            target_selection_label = args.target_selection

        if args.require_targets_in_root_cone and root_cone_depth_by_node is not None:
            targets = [target for target in targets if target in root_cone_depth_by_node]
            target_depth_by_node = {target: target_depth_by_node[target] for target in targets}

        if args.require_boundaries_in_root_cone and root_cone_depth_by_node is not None:
            boundary_nodes = [node for node in boundary_nodes if node in root_cone_depth_by_node]
            boundary_depth_by_node = {node: boundary_depth_by_node[node] for node in boundary_nodes}

        selected_boundary_nodes = set(boundary_nodes)
        if args.include_target_ancestor_boundaries:
            extra = collect_target_ancestor_boundaries(
                graph.parents,
                args.root,
                targets,
                boundary_depths,
                max(budgets, default=0),
                args.max_parent_depth,
                args.target_ancestor_boundary_limit,
                args.seed + ":target-ancestor-boundaries",
            )
            if args.parent_filter == "root-cone" and root_cone_depth_by_node is not None:
                extra = [node for node in extra if node in root_cone_depth_by_node]
            boundary_nodes = sorted(selected_boundary_nodes | set(extra))

        reachability = None
        if args.parent_filter == "root-reachable":
            reachability = RootReachabilityFilter(graph.parents, args.root)
        elif args.parent_filter == "root-cone":
            reachability = RootConeFilter(root_cone_depth_by_node)
        measure_suffix_mass = not args.skip_boundary_suffix_mass and reachability is None

        records = [{
            "record_type": "boundary_coverage_selection",
            "graph": args.graph_name,
            "root": args.root,
            "parent_filter": args.parent_filter,
            "selection_source": args.selection_source,
            "root_cone_depth": root_cone_depth if root_cone_depth_by_node is not None else None,
            "root_cone_nodes": len(root_cone_depth_by_node or {}),
            "root_cone_counts": root_cone_counts,
            "root_cone_children_per_node": args.root_cone_children_per_node if args.root_cone_children_per_node is not None else args.children_per_node,
            "root_cone_frontier_limit": args.root_cone_frontier_limit if args.root_cone_frontier_limit is not None else args.frontier_limit,
            "boundary_counts": boundary_counts,
            "target_counts": target_counts,
            "boundary_nodes": len(boundary_nodes),
            "selected_boundary_nodes": len(selected_boundary_nodes),
            "target_ancestor_boundary_nodes_added": len(set(boundary_nodes) - selected_boundary_nodes),
            "targets": len(targets),
            "target_selection": target_selection_label,
            "budgets": budgets,
            "mode": args.mode,
            "samples": args.samples,
            "path_count_cap": args.path_count_cap,
            "expansion_cap": args.expansion_cap,
            "measure_boundary_suffix_mass": measure_suffix_mass,
        }]

        boundary_set = set(boundary_nodes)
        for target in targets:
            for budget in budgets:
                if args.mode in {"exact", "both", "all"}:
                    records.append(time_target(lambda target=target, budget=budget: exact_boundary_coverage(
                        graph.parents,
                        target,
                        args.root,
                        budget,
                        boundary_set,
                        args.path_count_cap,
                        args.expansion_cap,
                        None if reachability is None else reachability.can_reach,
                        args.parent_filter,
                        measure_suffix_mass,
                    )))
                    records[-1]["child_sample_depth"] = target_depth_by_node[target]
                    records[-1]["graph"] = args.graph_name
                    records[-1]["root"] = args.root
                if args.mode in {"sample", "both", "all"}:
                    records.append(time_target(lambda target=target, budget=budget: sample_boundary_coverage(
                        graph.parents,
                        target,
                        args.root,
                        budget,
                        boundary_set,
                        args.samples,
                        "{}:{}:{}".format(args.seed, target, budget),
                        None if reachability is None else reachability.can_reach,
                        args.parent_filter,
                        measure_suffix_mass,
                    )))
                    records[-1]["child_sample_depth"] = target_depth_by_node[target]
                    records[-1]["graph"] = args.graph_name
                    records[-1]["root"] = args.root
                if args.mode in {"root-sample", "all"}:
                    records.append(time_target(lambda target=target, budget=budget: sample_root_path_space(
                        graph.parents,
                        target,
                        args.root,
                        budget,
                        boundary_set,
                        args.samples,
                        "{}:{}:{}:root-sample".format(args.seed, target, budget),
                        None if reachability is None else reachability.can_reach,
                        args.parent_filter,
                    )))
                    records[-1]["child_sample_depth"] = target_depth_by_node[target]
                    records[-1]["graph"] = args.graph_name
                    records[-1]["root"] = args.root

        return records, summarize(records)
    finally:
        graph.close()


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / "lmdb_boundary_coverage_probe_{}_{}.jsonl".format(safe_name, timestamp)
    summary_path = output_dir / "lmdb_boundary_coverage_probe_summary_{}_{}.md".format(safe_name, timestamp)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", required=True, type=int, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_boundary_coverage", help="Graph label used in output filenames.")
    parser.add_argument("--mode", choices=["exact", "sample", "root-sample", "both", "all"], default="exact", help="Coverage audit mode. sample stops at boundaries; root-sample walks to root/budget/dead-end; both runs exact+sample; all runs exact+sample+root-sample.")
    parser.add_argument("--parent-filter", choices=["all", "root-reachable", "root-cone"], default="root-cone", help="Parent expansion filter. root-reachable checks recursive finite-horizon reachability; root-cone uses a precomputed bounded child-reachable cone.")
    parser.add_argument("--selection-source", choices=["graph", "root-cone"], default="root-cone", help="Source for boundary and target sampling. root-cone samples directly from the precomputed root cone by child depth.")
    parser.add_argument("--root-cone-depth", type=int, default=0, help="Maximum child depth for the precomputed root cone. Non-positive uses the maximum requested budget/depth.")
    parser.add_argument("--root-cone-children-per-node", type=int, help="Child sample cap per root-cone frontier node. Omit to reuse --children-per-node; non-positive disables the per-node cap.")
    parser.add_argument("--root-cone-frontier-limit", type=int, help="Root-cone frontier cap per depth. Omit to reuse --frontier-limit; non-positive disables the frontier cap.")
    parser.add_argument("--require-targets-in-root-cone", action="store_true", help="Drop selected targets that are not present in the precomputed root cone.")
    parser.add_argument("--require-boundaries-in-root-cone", action="store_true", help="Drop selected boundaries that are not present in the precomputed root cone.")
    parser.add_argument("--skip-boundary-suffix-mass", action="store_true", help="Do not enumerate suffix path histograms after boundary hits. Use this for larger path budgets when only boundary coverage is being measured.")
    parser.add_argument("--boundary-depths", default=",".join(map(str, DEFAULT_BOUNDARY_DEPTHS)), help="Child depths used as boundary candidates.")
    parser.add_argument("--target-depths", default=",".join(map(str, DEFAULT_TARGET_DEPTHS)), help="Child depths used as target candidates.")
    parser.add_argument("--children-per-node", type=int, default=128, help="Deterministic child sample cap per frontier node.")
    parser.add_argument("--frontier-limit", type=int, default=2000, help="Deterministic cap for each sampled child-depth frontier.")
    parser.add_argument("--boundaries-per-depth", type=int, default=100, help="Boundary candidates per requested boundary depth.")
    parser.add_argument("--targets-per-depth", type=int, default=20, help="Targets per requested target depth.")
    parser.add_argument("--target-selection", choices=["child-depth", "boundary-descendants"], default="boundary-descendants", help="Target sampling mode when --selection-source graph is used.")
    parser.add_argument("--include-target-ancestor-boundaries", action="store_true", help="Add target ancestors whose root distance matches requested boundary depths.")
    parser.add_argument("--target-ancestor-boundary-limit", type=int, default=500, help="Maximum target-ancestor boundary nodes to add; non-positive means no extra cap.")
    parser.add_argument("--max-parent-depth", type=int, default=24, help="Parent depth cap for target-ancestor boundary collection.")
    parser.add_argument("--budgets", default="6", help="Comma-separated path-length budgets.")
    parser.add_argument("--path-count-cap", type=int, default=0, help="Exact mode safety cap on terminal prefixes; non-positive disables it.")
    parser.add_argument("--expansion-cap", type=int, default=0, help="Exact mode safety cap on expanded nodes; non-positive disables it.")
    parser.add_argument("--samples", type=int, default=200, help="Sampled mode random walks per target.")
    parser.add_argument("--seed", default="0", help="Deterministic sampling seed.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    args = parser.parse_args(argv)

    records, summary = run_probe(args)
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
