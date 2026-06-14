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
    filtered_bounded_parent_histogram,
    select_targets_by_boundary_descendants,
)
from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    deterministic_sample,
    parse_int_list,
    select_targets_by_child_depth,
)
from scripts.lmdb_parent_histogram_benchmark import safe_graph_name


DEFAULT_BOUNDARY_DEPTHS = [2]
DEFAULT_TARGET_DEPTHS = [4]
PATH_VALUE_KERNELS = ["count", "bp-decay", "weighted-power"]


@dataclass(frozen=True)
class PathValueKernel:
    name: str = "count"
    branching_factor: float | None = None
    branching_factor_source: str | None = None
    power: float | None = None


def normalize_path_value_kernel_name(name):
    return str(name or "count").strip().lower().replace("_", "-")


def make_path_value_kernel(name="count", branching_factor=None, branching_factor_source=None, power=None):
    name = normalize_path_value_kernel_name(name)
    if name not in PATH_VALUE_KERNELS:
        raise ValueError("unknown path-value kernel: {}".format(name))
    if name == "bp-decay":
        if branching_factor is None or float(branching_factor) <= 0.0:
            raise ValueError("bp-decay requires a positive branching factor")
        return PathValueKernel(
            name=name,
            branching_factor=float(branching_factor),
            branching_factor_source=branching_factor_source,
            power=None,
        )
    if name == "weighted-power":
        power = 1.0 if power is None else float(power)
        if power < 0.0:
            raise ValueError("weighted-power requires a non-negative power")
        return PathValueKernel(name=name, branching_factor=None, branching_factor_source=None, power=power)
    return PathValueKernel(name="count", branching_factor=None, branching_factor_source=None, power=None)


def path_value(length, kernel):
    """Evaluate the configured value kernel for a complete root path length."""
    length = int(length)
    if kernel.name == "count":
        return 1.0
    if kernel.name == "bp-decay":
        return float(kernel.branching_factor) ** (-length)
    if kernel.name == "weighted-power":
        return (length + 1.0) ** (-float(kernel.power))
    raise ValueError("unknown path-value kernel: {}".format(kernel.name))


def estimate_parent_branching_factor(parents_func, nodes, parent_accept=None):
    """Estimate b_p = E[p^2] / E[p] from eligible parent counts."""
    seen = set()
    measured = 0
    parent_sum = 0.0
    parent_sq_sum = 0.0
    max_parent_degree = 0
    for node in nodes:
        if node in seen:
            continue
        seen.add(node)
        parents = list(parents_func(node))
        if parent_accept is not None:
            parents = [parent for parent in parents if parent_accept(node, parent)]
        degree = len(parents)
        measured += 1
        parent_sum += degree
        parent_sq_sum += degree * degree
        max_parent_degree = max(max_parent_degree, degree)

    if measured <= 0 or parent_sum <= 0.0:
        return {
            "branching_factor": 1.0,
            "nodes": measured,
            "mean_parent_degree": 0.0,
            "second_parent_degree_moment": 0.0,
            "max_parent_degree": max_parent_degree,
        }

    mean_parent_degree = parent_sum / measured
    second_parent_degree_moment = parent_sq_sum / measured
    return {
        "branching_factor": parent_sq_sum / parent_sum,
        "nodes": measured,
        "mean_parent_degree": mean_parent_degree,
        "second_parent_degree_moment": second_parent_degree_moment,
        "max_parent_degree": max_parent_degree,
    }


def suffix_path_value(suffix, prefix_depth, kernel):
    return sum(
        count * path_value(int(prefix_depth) + int(length), kernel)
        for length, count in suffix["histogram"].items()
    )


def suffix_path_length_sum(suffix, prefix_depth):
    return sum(
        int(count) * (int(prefix_depth) + int(length))
        for length, count in suffix["histogram"].items()
    )


def histogram_path_value_sum(histogram, kernel):
    return sum(
        int(count) * path_value(int(length), kernel)
        for length, count in histogram.items()
    )


def histogram_path_length_sum(histogram):
    return sum(int(length) * int(count) for length, count in histogram.items())


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
    root_path_length_sum: int = 0
    root_path_value_sum: float = 0.0
    boundary_suffix_path_mass_sum: int = 0
    boundary_suffix_path_length_sum: int = 0
    boundary_suffix_path_value_sum: float = 0.0
    boundary_suffix_path_count_cap_hits: int = 0
    boundary_suffix_expansion_cap_hits: int = 0
    boundary_suffix_cycle_skips: int = 0
    boundary_suffix_root_unreachable_parent_skips: int = 0
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

    def accepts(self, _node, parent, remaining):
        return self.can_reach(parent, remaining)

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

    def accepts(self, _node, parent, remaining):
        return self.can_reach(parent, remaining)


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


def suffix_path_mass(
    parents_func,
    boundary_node,
    root,
    remaining,
    memo,
    parent_filter=None,
    path_count_cap=None,
    expansion_cap=None,
):
    key = (boundary_node, int(remaining), id(parent_filter), path_count_cap, expansion_cap)
    if key not in memo:
        hist, stats = filtered_bounded_parent_histogram(
            parents_func,
            boundary_node,
            root,
            int(remaining),
            path_count_cap,
            expansion_cap,
            parent_filter,
        )
        memo[key] = {
            "histogram": hist,
            "path_count": sum(hist.values()),
            "support_bins": len(hist),
            "path_length_sum": sum(int(length) * int(count) for length, count in hist.items()),
            "path_count_cap_hit": stats.path_cap_hit,
            "expansion_cap_hit": stats.expansion_cap_hit,
            "cycle_skips": stats.cycle_skips,
            "root_unreachable_parent_skips": getattr(stats, "root_unreachable_parent_skips", 0),
            "budget_cutoffs": stats.budget_cutoffs,
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
    suffix_parent_filter=None,
    suffix_path_count_cap=None,
    suffix_expansion_cap=None,
    path_value_kernel=None,
):
    """Enumerate simple parent prefixes until root, boundary, or budget."""
    budget = int(budget)
    boundary_nodes = set(boundary_nodes)
    path_count_cap = None if path_count_cap is None or int(path_count_cap) <= 0 else int(path_count_cap)
    expansion_cap = None if expansion_cap is None or int(expansion_cap) <= 0 else int(expansion_cap)
    kernel = path_value_kernel or PathValueKernel()
    stats = CoverageStats()
    suffix_memo = {}
    suffix_measure_compatible = measure_suffix_mass and (
        reachability_filter is None or suffix_parent_filter is not None
    )

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
            stats.root_path_length_sum += int(depth)
            stats.root_path_value_sum += path_value(depth, kernel)
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
                suffix = suffix_path_mass(
                    parents_func,
                    node,
                    root,
                    remaining,
                    suffix_memo,
                    suffix_parent_filter,
                    suffix_path_count_cap,
                    suffix_expansion_cap,
                )
                stats.boundary_suffix_path_mass_sum += int(suffix["path_count"])
                stats.boundary_suffix_path_length_sum += suffix_path_length_sum(suffix, depth)
                stats.boundary_suffix_path_value_sum += suffix_path_value(suffix, depth, kernel)
                stats.boundary_suffix_path_count_cap_hits += 1 if suffix["path_count_cap_hit"] else 0
                stats.boundary_suffix_expansion_cap_hits += 1 if suffix["expansion_cap_hit"] else 0
                stats.boundary_suffix_cycle_skips += int(suffix["cycle_skips"])
                stats.boundary_suffix_root_unreachable_parent_skips += int(suffix["root_unreachable_parent_skips"])
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
        path_value_kernel=kernel,
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
    suffix_parent_filter=None,
    path_value_kernel=None,
):
    """Sample random parent walks and estimate prefix counts by proposal correction.

    The branch product corrects for the local random-walk proposal.  Boundary
    suffix mass is a value supplied by the cached histogram for the remaining
    budget; metric-specific kernels require matching suffix bases.
    """
    budget = int(budget)
    boundary_nodes = set(boundary_nodes)
    kernel = path_value_kernel or PathValueKernel()
    rng = random.Random(str(seed))
    stats = CoverageStats()
    suffix_memo = {}
    suffix_measure_compatible = measure_suffix_mass and (
        reachability_filter is None or suffix_parent_filter is not None
    )
    terminal_weights = []
    root_weights = []
    root_length_weights = []
    root_value_weights = []
    boundary_weights = []
    boundary_spliced_weights = []
    boundary_spliced_length_weights = []
    boundary_spliced_value_weights = []
    budget_weights = []
    dead_end_weights = []
    suffix_mass_weights = []
    suffix_length_weights = []
    suffix_value_weights = []

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
                stats.root_path_length_sum += int(depth)
                stats.root_path_value_sum += path_value(depth, kernel)
                root_value = weight * path_value(depth, kernel)
                terminal_weights.append(weight)
                root_weights.append(weight)
                root_length_weights.append(weight * depth)
                root_value_weights.append(root_value)
                boundary_weights.append(0.0)
                boundary_spliced_weights.append(0.0)
                boundary_spliced_length_weights.append(0.0)
                boundary_spliced_value_weights.append(0.0)
                budget_weights.append(0.0)
                dead_end_weights.append(0.0)
                suffix_mass_weights.append(0.0)
                suffix_length_weights.append(0.0)
                suffix_value_weights.append(0.0)
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
                suffix_length = 0
                suffix_value = 0.0
                if suffix_measure_compatible:
                    suffix = suffix_path_mass(
                        parents_func,
                        node,
                        root,
                        remaining,
                        suffix_memo,
                        suffix_parent_filter,
                    )
                    suffix_mass = int(suffix["path_count"])
                    suffix_length = suffix_path_length_sum(suffix, depth)
                    suffix_value = suffix_path_value(suffix, depth, kernel)
                    stats.boundary_suffix_path_mass_sum += suffix_mass
                    stats.boundary_suffix_path_length_sum += suffix_length
                    stats.boundary_suffix_path_value_sum += suffix_value
                    stats.boundary_suffix_path_count_cap_hits += 1 if suffix["path_count_cap_hit"] else 0
                    stats.boundary_suffix_expansion_cap_hits += 1 if suffix["expansion_cap_hit"] else 0
                    stats.boundary_suffix_cycle_skips += int(suffix["cycle_skips"])
                    stats.boundary_suffix_root_unreachable_parent_skips += int(suffix["root_unreachable_parent_skips"])
                terminal_weights.append(weight)
                root_weights.append(0.0)
                root_length_weights.append(0.0)
                root_value_weights.append(0.0)
                boundary_weights.append(weight)
                boundary_spliced_weights.append(weight * suffix_mass)
                boundary_spliced_length_weights.append(weight * suffix_length)
                boundary_spliced_value_weights.append(weight * suffix_value)
                budget_weights.append(0.0)
                dead_end_weights.append(0.0)
                suffix_mass_weights.append(weight * suffix_mass)
                suffix_length_weights.append(weight * suffix_length)
                suffix_value_weights.append(weight * suffix_value)
                break

            if remaining <= 0:
                stats.terminal_prefixes += 1
                stats.budget_exhausted_prefixes += 1
                terminal_weights.append(weight)
                root_weights.append(0.0)
                root_length_weights.append(0.0)
                root_value_weights.append(0.0)
                boundary_weights.append(0.0)
                boundary_spliced_weights.append(0.0)
                boundary_spliced_length_weights.append(0.0)
                boundary_spliced_value_weights.append(0.0)
                budget_weights.append(weight)
                dead_end_weights.append(0.0)
                suffix_mass_weights.append(0.0)
                suffix_length_weights.append(0.0)
                suffix_value_weights.append(0.0)
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
                root_length_weights.append(0.0)
                root_value_weights.append(0.0)
                boundary_weights.append(0.0)
                boundary_spliced_weights.append(0.0)
                boundary_spliced_length_weights.append(0.0)
                boundary_spliced_value_weights.append(0.0)
                budget_weights.append(0.0)
                dead_end_weights.append(weight)
                suffix_mass_weights.append(0.0)
                suffix_length_weights.append(0.0)
                suffix_value_weights.append(0.0)
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
        "estimated_root_path_length_sum": weighted_mean(root_length_weights),
        "estimated_root_path_length_sum_se": standard_error(root_length_weights),
        "estimated_root_value_sum": weighted_mean(root_value_weights),
        "estimated_root_value_sum_se": standard_error(root_value_weights),
        "estimated_boundary_hit_prefixes": weighted_mean(boundary_weights),
        "estimated_boundary_hit_prefixes_se": standard_error(boundary_weights),
        "estimated_boundary_spliced_root_paths": weighted_mean(boundary_spliced_weights),
        "estimated_boundary_spliced_root_paths_se": standard_error(boundary_spliced_weights),
        "estimated_boundary_spliced_path_length_sum": weighted_mean(boundary_spliced_length_weights),
        "estimated_boundary_spliced_path_length_sum_se": standard_error(boundary_spliced_length_weights),
        "estimated_boundary_spliced_value_sum": weighted_mean(boundary_spliced_value_weights),
        "estimated_boundary_spliced_value_sum_se": standard_error(boundary_spliced_value_weights),
        "estimated_budget_exhausted_prefixes": weighted_mean(budget_weights),
        "estimated_budget_exhausted_prefixes_se": standard_error(budget_weights),
        "estimated_dead_end_prefixes": weighted_mean(dead_end_weights),
        "estimated_dead_end_prefixes_se": standard_error(dead_end_weights),
        "estimated_boundary_suffix_path_mass": weighted_mean(suffix_mass_weights),
        "estimated_boundary_suffix_path_mass_se": standard_error(suffix_mass_weights),
        "estimated_boundary_suffix_path_length_sum": weighted_mean(suffix_length_weights),
        "estimated_boundary_suffix_path_length_sum_se": standard_error(suffix_length_weights),
        "estimated_boundary_suffix_path_value": weighted_mean(suffix_value_weights),
        "estimated_boundary_suffix_path_value_se": standard_error(suffix_value_weights),
    }
    if suffix_measure_compatible:
        weighted["estimated_spliced_total_root_paths"] = (
            weighted["estimated_root_paths"] + weighted["estimated_boundary_spliced_root_paths"]
        )
        weighted["estimated_spliced_total_path_length_sum"] = (
            weighted["estimated_root_path_length_sum"] + weighted["estimated_boundary_spliced_path_length_sum"]
        )
        weighted["estimated_spliced_total_value_sum"] = (
            weighted["estimated_root_value_sum"] + weighted["estimated_boundary_spliced_value_sum"]
        )
        weighted["estimated_spliced_mean_path_length"] = (
            None
            if weighted["estimated_spliced_total_root_paths"] <= 0.0
            else weighted["estimated_spliced_total_path_length_sum"] / weighted["estimated_spliced_total_root_paths"]
        )
    else:
        weighted["estimated_boundary_spliced_root_paths"] = None
        weighted["estimated_boundary_spliced_root_paths_se"] = None
        weighted["estimated_boundary_spliced_path_length_sum"] = None
        weighted["estimated_boundary_spliced_path_length_sum_se"] = None
        weighted["estimated_boundary_spliced_value_sum"] = None
        weighted["estimated_boundary_spliced_value_sum_se"] = None
        weighted["estimated_spliced_total_root_paths"] = None
        weighted["estimated_spliced_total_path_length_sum"] = None
        weighted["estimated_spliced_total_value_sum"] = None
        weighted["estimated_spliced_mean_path_length"] = None
        weighted["estimated_boundary_suffix_path_mass"] = None
        weighted["estimated_boundary_suffix_path_mass_se"] = None
        weighted["estimated_boundary_suffix_path_length_sum"] = None
        weighted["estimated_boundary_suffix_path_length_sum_se"] = None
        weighted["estimated_boundary_suffix_path_value"] = None
        weighted["estimated_boundary_suffix_path_value_se"] = None
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
        path_value_kernel=kernel,
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
    path_value_kernel=None,
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
    kernel = path_value_kernel or PathValueKernel()
    rng = random.Random(str(seed))
    stats = CoverageStats()
    terminal_weights = []
    root_weights = []
    root_value_weights = []
    boundary_weights = []
    root_boundary_weights = []
    root_boundary_value_weights = []
    root_length_weights = []
    root_value_length_weights = []
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
                stats.root_path_length_sum += int(depth)
                stats.root_path_value_sum += path_value(depth, kernel)
                root_value = weight * path_value(depth, kernel)
                terminal_weights.append(weight)
                root_weights.append(weight)
                root_value_weights.append(root_value)
                boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_value_weights.append(root_value if boundary_seen else 0.0)
                root_length_weights.append(weight * depth)
                root_value_length_weights.append(root_value * depth)
                budget_weights.append(0.0)
                dead_end_weights.append(0.0)
                break

            if remaining <= 0:
                stats.terminal_prefixes += 1
                stats.budget_exhausted_prefixes += 1
                terminal_weights.append(weight)
                root_weights.append(0.0)
                root_value_weights.append(0.0)
                boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_weights.append(0.0)
                root_boundary_value_weights.append(0.0)
                root_length_weights.append(0.0)
                root_value_length_weights.append(0.0)
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
                root_value_weights.append(0.0)
                boundary_weights.append(weight if boundary_seen else 0.0)
                root_boundary_weights.append(0.0)
                root_boundary_value_weights.append(0.0)
                root_length_weights.append(0.0)
                root_value_length_weights.append(0.0)
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
        "estimated_root_value_sum": weighted_mean(root_value_weights),
        "estimated_root_value_sum_se": standard_error(root_value_weights),
        "estimated_boundary_hit_prefixes": weighted_mean(boundary_weights),
        "estimated_boundary_hit_prefixes_se": standard_error(boundary_weights),
        "estimated_root_boundary_hit_paths": weighted_mean(root_boundary_weights),
        "estimated_root_boundary_hit_paths_se": standard_error(root_boundary_weights),
        "estimated_root_boundary_hit_value_sum": weighted_mean(root_boundary_value_weights),
        "estimated_root_boundary_hit_value_sum_se": standard_error(root_boundary_value_weights),
        "estimated_root_path_length_sum": weighted_mean(root_length_weights),
        "estimated_root_path_length_sum_se": standard_error(root_length_weights),
        "estimated_root_value_path_length_sum": weighted_mean(root_value_length_weights),
        "estimated_root_value_path_length_sum_se": standard_error(root_value_length_weights),
        "estimated_budget_exhausted_prefixes": weighted_mean(budget_weights),
        "estimated_budget_exhausted_prefixes_se": standard_error(budget_weights),
        "estimated_dead_end_prefixes": weighted_mean(dead_end_weights),
        "estimated_dead_end_prefixes_se": standard_error(dead_end_weights),
    }
    total_estimate = weighted["estimated_terminal_prefixes"]
    boundary_estimate = weighted["estimated_boundary_hit_prefixes"]
    root_estimate = weighted["estimated_root_paths"]
    root_value_estimate = weighted["estimated_root_value_sum"]
    root_boundary_estimate = weighted["estimated_root_boundary_hit_paths"]
    root_boundary_value_estimate = weighted["estimated_root_boundary_hit_value_sum"]
    weighted["estimated_boundary_hit_fraction"] = None if total_estimate <= 0.0 else boundary_estimate / total_estimate
    weighted["estimated_root_boundary_hit_fraction"] = None if root_estimate <= 0.0 else root_boundary_estimate / root_estimate
    weighted["estimated_root_value_boundary_hit_fraction"] = (
        None if root_value_estimate <= 0.0 else root_boundary_value_estimate / root_value_estimate
    )
    weighted["estimated_mean_root_path_length"] = (
        None if root_estimate <= 0.0 else weighted["estimated_root_path_length_sum"] / root_estimate
    )
    weighted["estimated_kernel_mean_root_path_length"] = (
        None if root_value_estimate <= 0.0 else weighted["estimated_root_value_path_length_sum"] / root_value_estimate
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
        path_value_kernel=kernel,
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
    path_value_kernel=None,
):
    boundary_hits = int(stats.boundary_hits)
    kernel = path_value_kernel or PathValueKernel()
    suffix_measured = bool(measure_suffix_mass)
    spliced_total_root_paths = None
    spliced_total_path_length_sum = None
    spliced_total_value_sum = None
    spliced_mean_path_length = None
    if suffix_measured:
        spliced_total_root_paths = int(stats.root_paths) + int(stats.boundary_suffix_path_mass_sum)
        spliced_total_path_length_sum = int(stats.root_path_length_sum) + int(stats.boundary_suffix_path_length_sum)
        spliced_total_value_sum = float(stats.root_path_value_sum) + float(stats.boundary_suffix_path_value_sum)
        if spliced_total_root_paths > 0:
            spliced_mean_path_length = spliced_total_path_length_sum / spliced_total_root_paths
    record = {
        "record_type": "boundary_coverage_target",
        "mode": mode,
        "parent_filter": parent_filter_name,
        "path_value_kernel": kernel.name,
        "path_value_branching_factor": kernel.branching_factor,
        "path_value_branching_factor_source": kernel.branching_factor_source,
        "path_value_power": kernel.power,
        "target_node": target,
        "path_length_budget": int(budget),
        "samples": samples,
        "terminal_prefixes": int(stats.terminal_prefixes),
        "root_paths": int(stats.root_paths),
        "root_path_length_sum": int(stats.root_path_length_sum),
        "root_path_value_sum": float(stats.root_path_value_sum),
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
        "boundary_suffix_mass_measured": suffix_measured,
        "mean_boundary_suffix_path_mass": None if not measure_suffix_mass or boundary_hits <= 0 else stats.boundary_suffix_path_mass_sum / boundary_hits,
        "boundary_suffix_path_mass_sum": int(stats.boundary_suffix_path_mass_sum),
        "boundary_suffix_path_length_sum": int(stats.boundary_suffix_path_length_sum),
        "mean_boundary_suffix_path_value": None if not measure_suffix_mass or boundary_hits <= 0 else stats.boundary_suffix_path_value_sum / boundary_hits,
        "boundary_suffix_path_value_sum": float(stats.boundary_suffix_path_value_sum),
        "boundary_suffix_path_count_cap_hits": int(stats.boundary_suffix_path_count_cap_hits),
        "boundary_suffix_expansion_cap_hits": int(stats.boundary_suffix_expansion_cap_hits),
        "boundary_suffix_cycle_skips": int(stats.boundary_suffix_cycle_skips),
        "boundary_suffix_root_unreachable_parent_skips": int(stats.boundary_suffix_root_unreachable_parent_skips),
        "spliced_total_root_paths": spliced_total_root_paths,
        "spliced_total_path_length_sum": spliced_total_path_length_sum,
        "spliced_total_value_sum": spliced_total_value_sum,
        "spliced_mean_path_length": spliced_mean_path_length,
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


def relative_error(delta, baseline):
    baseline = float(baseline)
    if baseline == 0.0:
        return 0.0 if float(delta) == 0.0 else None
    return abs(float(delta)) / abs(baseline)


def full_exact_splice_validation_record(
    boundary_row,
    parents_func,
    root,
    path_count_cap,
    expansion_cap,
    parent_filter,
    path_value_kernel,
):
    target = boundary_row["target_node"]
    budget = int(boundary_row["path_length_budget"])
    path_count_cap = normalize_limit(path_count_cap)
    expansion_cap = normalize_limit(expansion_cap)
    started = time.perf_counter_ns()
    full_hist, full_stats = filtered_bounded_parent_histogram(
        parents_func,
        target,
        root,
        budget,
        path_count_cap,
        expansion_cap,
        parent_filter,
    )
    elapsed_ns = time.perf_counter_ns() - started
    full_root_paths = sum(int(count) for count in full_hist.values())
    full_path_length_sum = histogram_path_length_sum(full_hist)
    full_value_sum = histogram_path_value_sum(full_hist, path_value_kernel)
    full_mean_path_length = None if full_root_paths <= 0 else full_path_length_sum / full_root_paths
    spliced_root_paths = boundary_row.get("spliced_total_root_paths")
    spliced_path_length_sum = boundary_row.get("spliced_total_path_length_sum")
    spliced_value_sum = boundary_row.get("spliced_total_value_sum")
    spliced_mean_path_length = boundary_row.get("spliced_mean_path_length")
    path_delta = None if spliced_root_paths is None else float(spliced_root_paths) - float(full_root_paths)
    length_delta = None if spliced_path_length_sum is None else float(spliced_path_length_sum) - float(full_path_length_sum)
    value_delta = None if spliced_value_sum is None else float(spliced_value_sum) - float(full_value_sum)
    mean_length_delta = (
        None
        if spliced_mean_path_length is None or full_mean_path_length is None
        else float(spliced_mean_path_length) - float(full_mean_path_length)
    )
    boundary_partial = (
        bool(boundary_row.get("path_count_cap_hit"))
        or bool(boundary_row.get("expansion_cap_hit"))
        or int(boundary_row.get("boundary_suffix_path_count_cap_hits", 0)) > 0
        or int(boundary_row.get("boundary_suffix_expansion_cap_hits", 0)) > 0
    )
    full_partial = bool(full_stats.path_cap_hit) or bool(full_stats.expansion_cap_hit)
    comparable = (
        not boundary_partial
        and not full_partial
        and spliced_root_paths is not None
        and spliced_path_length_sum is not None
        and spliced_value_sum is not None
    )
    return {
        "record_type": "boundary_splice_validation",
        "graph": boundary_row.get("graph"),
        "root": boundary_row.get("root", root),
        "target_node": target,
        "child_sample_depth": boundary_row.get("child_sample_depth"),
        "path_length_budget": budget,
        "parent_filter": boundary_row.get("parent_filter", "all"),
        "path_value_kernel": path_value_kernel.name,
        "path_value_branching_factor": path_value_kernel.branching_factor,
        "path_value_branching_factor_source": path_value_kernel.branching_factor_source,
        "path_value_power": path_value_kernel.power,
        "spliced_total_root_paths": spliced_root_paths,
        "spliced_total_path_length_sum": spliced_path_length_sum,
        "spliced_total_value_sum": spliced_value_sum,
        "spliced_mean_path_length": spliced_mean_path_length,
        "full_root_paths": full_root_paths,
        "full_path_length_sum": full_path_length_sum,
        "full_value_sum": full_value_sum,
        "full_mean_path_length": full_mean_path_length,
        "root_path_delta": path_delta,
        "abs_root_path_delta": None if path_delta is None else abs(path_delta),
        "root_path_relative_error": None if path_delta is None else relative_error(path_delta, full_root_paths),
        "path_length_sum_delta": length_delta,
        "abs_path_length_sum_delta": None if length_delta is None else abs(length_delta),
        "value_sum_delta": value_delta,
        "abs_value_sum_delta": None if value_delta is None else abs(value_delta),
        "value_sum_relative_error": None if value_delta is None else relative_error(value_delta, full_value_sum),
        "mean_path_length_delta": mean_length_delta,
        "abs_mean_path_length_delta": None if mean_length_delta is None else abs(mean_length_delta),
        "comparable": comparable,
        "boundary_partial": boundary_partial,
        "full_partial": full_partial,
        "boundary_path_count_cap_hit": bool(boundary_row.get("path_count_cap_hit")),
        "boundary_expansion_cap_hit": bool(boundary_row.get("expansion_cap_hit")),
        "boundary_suffix_path_count_cap_hits": int(boundary_row.get("boundary_suffix_path_count_cap_hits", 0)),
        "boundary_suffix_expansion_cap_hits": int(boundary_row.get("boundary_suffix_expansion_cap_hits", 0)),
        "full_path_count_cap_hit": bool(full_stats.path_cap_hit),
        "full_expansion_cap_hit": bool(full_stats.expansion_cap_hit),
        "full_nodes_expanded": int(full_stats.nodes_expanded),
        "full_edges_examined": int(full_stats.edges_examined),
        "full_cycle_skips": int(full_stats.cycle_skips),
        "full_root_unreachable_parent_skips": int(getattr(full_stats, "root_unreachable_parent_skips", 0)),
        "full_elapsed_ns": elapsed_ns,
    }


def aggregate_rows(rows):
    if not rows:
        return {}
    terminal = sum(int(row["terminal_prefixes"]) for row in rows)
    boundary = sum(int(row["boundary_hit_prefixes"]) for row in rows)
    root = sum(int(row["root_paths"]) for row in rows)
    root_length_sum = sum(int(row.get("root_path_length_sum", 0)) for row in rows)
    root_value_sum = sum(float(row.get("root_path_value_sum", 0.0)) for row in rows)
    budget = sum(int(row["budget_exhausted_prefixes"]) for row in rows)
    filtered_dead = sum(int(row.get("filtered_dead_end_prefixes", 0)) for row in rows)
    suffix_mass = sum(int(row["boundary_suffix_path_mass_sum"]) for row in rows)
    suffix_length_sum = sum(int(row.get("boundary_suffix_path_length_sum", 0)) for row in rows)
    suffix_value = sum(float(row.get("boundary_suffix_path_value_sum", 0.0)) for row in rows)
    suffix_mass_measured = all(bool(row.get("boundary_suffix_mass_measured", True)) for row in rows)
    spliced_total_root_paths = root + suffix_mass if suffix_mass_measured else None
    spliced_total_path_length_sum = root_length_sum + suffix_length_sum if suffix_mass_measured else None
    spliced_total_value_sum = root_value_sum + suffix_value if suffix_mass_measured else None
    spliced_mean_path_length = (
        None
        if not suffix_mass_measured or not spliced_total_root_paths
        else spliced_total_path_length_sum / spliced_total_root_paths
    )
    return {
        "targets": len(rows),
        "terminal_prefixes": terminal,
        "root_paths": root,
        "root_path_length_sum": root_length_sum,
        "root_path_value_sum": root_value_sum,
        "boundary_hit_prefixes": boundary,
        "budget_exhausted_prefixes": budget,
        "filtered_dead_end_prefixes": filtered_dead,
        "boundary_hit_fraction": fraction(boundary, terminal),
        "root_path_fraction": fraction(root, terminal),
        "budget_exhausted_fraction": fraction(budget, terminal),
        "boundary_suffix_path_mass_sum": suffix_mass,
        "boundary_suffix_path_length_sum": suffix_length_sum,
        "boundary_suffix_path_value_sum": suffix_value,
        "boundary_suffix_mass_measured": suffix_mass_measured,
        "mean_boundary_suffix_path_mass": None if not suffix_mass_measured or boundary <= 0 else suffix_mass / boundary,
        "mean_boundary_suffix_path_value": None if not suffix_mass_measured or boundary <= 0 else suffix_value / boundary,
        "spliced_total_root_paths": spliced_total_root_paths,
        "spliced_total_path_length_sum": spliced_total_path_length_sum,
        "spliced_total_value_sum": spliced_total_value_sum,
        "spliced_mean_path_length": spliced_mean_path_length,
        "completed_targets": sum(1 for row in rows if row.get("completed")),
        "path_count_cap_hit_targets": sum(1 for row in rows if row.get("path_count_cap_hit")),
        "expansion_cap_hit_targets": sum(1 for row in rows if row.get("expansion_cap_hit")),
        "cycle_skips": sum(int(row.get("cycle_skips", 0)) for row in rows),
        "root_unreachable_parent_skips": sum(int(row.get("root_unreachable_parent_skips", 0)) for row in rows),
        "boundary_suffix_path_count_cap_hits": sum(int(row.get("boundary_suffix_path_count_cap_hits", 0)) for row in rows),
        "boundary_suffix_expansion_cap_hits": sum(int(row.get("boundary_suffix_expansion_cap_hits", 0)) for row in rows),
        "boundary_suffix_cycle_skips": sum(int(row.get("boundary_suffix_cycle_skips", 0)) for row in rows),
        "boundary_suffix_root_unreachable_parent_skips": sum(int(row.get("boundary_suffix_root_unreachable_parent_skips", 0)) for row in rows),
    }


def format_optional(value, digits=3):
    if value is None:
        return "n/a"
    return ("{:." + str(digits) + "f}").format(float(value))


def mean_optional_field(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else statistics.mean(values)


def max_optional_field(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else max(values)


def validation_exact_match(row, tolerance=1.0e-9):
    if not row.get("comparable"):
        return False
    return (
        float(row.get("abs_root_path_delta") or 0.0) <= tolerance
        and float(row.get("abs_value_sum_delta") or 0.0) <= tolerance
        and float(row.get("abs_mean_path_length_delta") or 0.0) <= tolerance
    )


def sorted_depth_items(counts):
    return sorted(counts.items(), key=lambda item: int(item[0]))


def depths_text(values):
    if values is None:
        return None
    return ", ".join(str(value) for value in values)


def boundary_coverage_generation_notes(selection):
    mode = selection.get("mode", "exact")
    selection_source = selection.get("selection_source", "graph")
    parent_filter = selection.get("parent_filter", "all")
    lines = [
        "## How This Was Generated",
        "",
    ]
    boundary_depths = depths_text(selection.get("boundary_depths"))
    target_depths = depths_text(selection.get("target_depths"))
    if boundary_depths is None or target_depths is None:
        lines.append(
            "- This older selection record stores observed boundary/target frontier counts over child-depths `{}` and `{}`, but not the requested depth arguments separately.".format(
                depths_text(depth for depth, _count in sorted_depth_items(selection.get("boundary_counts", {}))) or "none",
                depths_text(depth for depth, _count in sorted_depth_items(selection.get("target_counts", {}))) or "none",
            )
        )
    else:
        lines.append(
            "- Boundary candidates were sampled from requested child depth(s) `{}` and target rows from requested child depth(s) `{}` using selection source `{}`.".format(
                boundary_depths,
                target_depths,
                selection_source,
            )
        )
    if all(selection.get(key) is not None for key in ("children_per_node", "frontier_limit", "boundaries_per_depth", "targets_per_depth")):
        if selection_source == "root-cone":
            lines.append(
                "- Boundary and target nodes were sampled from the precomputed root-cone depth buckets with per-depth limits `boundaries_per_depth={}` and `targets_per_depth={}`.".format(
                    selection.get("boundaries_per_depth"),
                    selection.get("targets_per_depth"),
                )
            )
        else:
            lines.append(
                "- The graph child-frontier sampler used `children_per_node={}` and `frontier_limit={}`; per-depth sample limits were `boundaries_per_depth={}` and `targets_per_depth={}`.".format(
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
    lines.append(
        "- Mode `{}` controls the row generator: `exact` enumerates all simple parent prefixes until root, boundary, or budget; `sample` performs branch-product weighted boundary-stopped random walks; `root-sample` samples walks to root without stopping at boundaries.".format(mode)
    )
    lines.append(
        "- Parent filter `{}` is applied during parent expansion. `root-cone` accepts only parents inside the precomputed root cone whose cone depth fits within the remaining path budget; `root-reachable` uses recursive finite-horizon reachability; `all` does no root-scope pruning.".format(parent_filter)
    )
    if selection.get("root_cone_counts"):
        lines.append(
            "- The root cone was built to child depth `{}` with `{}` nodes, `root_cone_children_per_node={}`, and `root_cone_frontier_limit={}`.".format(
                selection.get("root_cone_depth", "n/a"),
                selection.get("root_cone_nodes", 0),
                selection.get("root_cone_children_per_node", "n/a"),
                selection.get("root_cone_frontier_limit", "n/a"),
            )
        )
    if selection.get("include_target_ancestor_boundaries"):
        if selection.get("target_ancestor_boundary_limit") is None:
            lines.append("- Target-ancestor boundary inclusion was enabled, so sampled boundary-depth ancestors of targets could be added to the selected boundary set.")
        else:
            lines.append(
                "- Target-ancestor boundary inclusion was enabled, with `target_ancestor_boundary_limit={}`.".format(
                    selection.get("target_ancestor_boundary_limit")
                )
            )
    lines.append(
        "- Boundary suffix mass measured: `{}`. When this is false, boundary-hit rows measure coverage only; they do not splice cached suffix mass into a total root-path estimate.".format(
            selection.get("measure_boundary_suffix_mass", True)
        )
    )
    if selection.get("parent_filter") != "all" and not selection.get("measure_filtered_boundary_suffix_mass", False):
        lines.append(
            "- Filtered suffix measurement was not requested. Pass `--measure-filtered-boundary-suffix-mass` to materialize boundary suffix histograms under the same parent filter."
        )
    if selection.get("validate_full_exact"):
        lines.append(
            "- Full exact validation was requested. The report includes a separate comparison between boundary-stopped suffix splicing and full filtered DFS to root for exact-mode rows."
        )
    lines.append(
        "- Path value kernel `{}` defines the functional being estimated after path coverage is known. It is separate from the random-walk proposal correction.".format(
            selection.get("path_value_kernel", "count")
        )
    )
    return lines


def boundary_coverage_table_guide():
    return [
        "## Table Guide",
        "",
        "- `Selection` lists observed frontier sizes for boundary and target depths. In newer reports, the requested depths are stated above; the table may include intermediate traversal depths.",
        "- `Root Cone` shows the bounded child-reachable cone used for root-cone filtering. These are not parent-path counts; they are child-depth frontier counts from the root.",
        "- `Coverage Summary` aggregates observed terminal outcomes by mode and path-length budget. In exact mode these are enumerated simple-prefix counts; in sample/root-sample modes they are raw sample outcomes.",
        "- `spliced_total_root_paths`, `spliced_total_value_sum`, and `spliced_mean_path_length` are boundary-aware estimates: direct root terminals plus suffix histogram mass/value from boundary hits.",
        "- `Boundary Sample Estimates` and `Root Path Sample Estimates` contain branch-product weighted estimates. Use those estimate tables, not raw observed sample counts, when reasoning about path-space size.",
        "- `Full Exact Splice Validation`, when present, checks whether boundary-stopped exact search plus suffix histograms reproduces full filtered DFS to root for the same target and budget.",
        "- `Target Rows` is per target and budget. `root_paths` counts direct root terminals reached before a boundary stop; `boundary_hit_prefixes` counts prefixes where the boundary condition would take over.",
        "- `root_unreachable_parent_skips` counts parent edges rejected by the active parent filter. Under `root-cone`, that includes parents outside the cone or too deep for the remaining budget, not only globally unreachable parents.",
        "- `budget_exhausted_prefixes`, `path_count_cap_hit_targets`, and `expansion_cap_hit_targets` identify rows whose result is limited by the path budget or safety caps.",
    ]


def boundary_coverage_result_implications(selection, target_rows):
    lines = ["## Result Implications", ""]
    if not target_rows:
        lines.append("- No target rows were generated, so this report only documents selection shape.")
        return lines
    by_mode_budget = {}
    for row in target_rows:
        by_mode_budget.setdefault((row["mode"], row["path_length_budget"]), []).append(row)
    total_rows = len(target_rows)
    complete_rows = sum(1 for row in target_rows if row.get("completed"))
    cap_rows = total_rows - complete_rows
    zero_root_boundary_rows = [
        row for row in target_rows
        if int(row.get("root_paths", 0)) == 0 and int(row.get("boundary_hit_prefixes", 0)) > 0
    ]
    budget_cutoff_rows = [row for row in target_rows if int(row.get("budget_exhausted_prefixes", 0)) > 0]
    root_skip_total = sum(int(row.get("root_unreachable_parent_skips", 0)) for row in target_rows)
    cycle_skip_total = sum(int(row.get("cycle_skips", 0)) for row in target_rows)
    filtered_dead_total = sum(int(row.get("filtered_dead_end_prefixes", 0)) for row in target_rows)
    lines.append("- Target evaluation completed `{}/{}` rows without path-count or expansion caps.".format(complete_rows, total_rows))
    if cap_rows:
        lines.append("- `{}` rows hit a path-count or expansion cap; read those rows as partial coverage evidence rather than full target-cone counts.".format(cap_rows))
    for key in sorted(by_mode_budget):
        mode, budget = key
        rows = by_mode_budget[key]
        aggregate = aggregate_rows(rows)
        lines.append(
            "- `{}` budget `{}`: `{}` terminal prefixes, `{}` direct root paths, `{}` boundary-hit prefixes, boundary-hit fraction `{}`, `{}` budget-exhausted prefixes, and spliced root mass `{}`.".format(
                mode,
                budget,
                aggregate["terminal_prefixes"],
                aggregate["root_paths"],
                aggregate["boundary_hit_prefixes"],
                format_optional(aggregate["boundary_hit_fraction"], 6),
                aggregate["budget_exhausted_prefixes"],
                format_optional(aggregate["spliced_total_root_paths"], 3),
            )
        )
    if zero_root_boundary_rows:
        lines.append(
            "- `{}` target-budget rows have `root_paths=0` and positive boundary hits. In this report that means enumeration stopped at a boundary before reaching root; it is boundary coverage, not evidence that those targets lack root paths.".format(
                len(zero_root_boundary_rows)
            )
        )
    if not selection.get("measure_boundary_suffix_mass", True):
        lines.append(
            "- Boundary suffix mass was not measured, so boundary hits cannot yet be converted into total root-path mass or budgeted CDF mass from this report alone."
        )
    elif any(row.get("boundary_suffix_path_mass_sum") for row in target_rows):
        lines.append(
            "- Boundary suffix mass was measured, so boundary-hit prefixes are combined with suffix histograms to estimate total root-path mass, aggregate value, and mean path length under the remaining budget."
        )
    if root_skip_total:
        lines.append(
            "- The active parent filter rejected `{}` parent edges. Under `{}`, these skips are part of the scoped experiment definition, not necessarily data errors.".format(
                root_skip_total,
                selection.get("parent_filter", "all"),
            )
        )
    if budget_cutoff_rows:
        lines.append(
            "- `{}` rows exhausted the path-length budget before root or boundary. Larger budgets may change coverage conclusions for those targets.".format(
                len(budget_cutoff_rows)
            )
        )
    if cycle_skip_total:
        lines.append(
            "- Simple-path cycle checks skipped `{}` edges. That keeps rows cycle-free, but it also means cached suffixes must be interpreted with the same cycle policy.".format(cycle_skip_total)
        )
    if filtered_dead_total:
        lines.append(
            "- Filtered dead ends occurred `{}` times; these are prefixes whose remaining parents were removed by the active parent filter.".format(filtered_dead_total)
        )
    if any(row.get("mode") in {"sample", "root-sample"} for row in target_rows):
        lines.append(
            "- Sample-mode rows are statistical estimates. Compare confidence intervals and increase `samples` before using them as performance-planning inputs."
        )
    return lines


def boundary_splice_validation_section(validation_rows):
    if not validation_rows:
        return []
    by_budget = {}
    for row in validation_rows:
        by_budget.setdefault(row["path_length_budget"], []).append(row)
    lines = [
        "## Full Exact Splice Validation",
        "",
        "This section compares boundary-stopped exact search plus suffix splicing against full filtered DFS to root on the same targets and path-length budgets. Comparable rows are uncapped on both sides and have measured suffix mass. For those rows, zero deltas mean the boundary condition reproduced full exact search for path mass, selected value, and mean path length.",
        "",
        "| path_length_budget | rows | comparable_rows | exact_match_rows | max_abs_root_path_delta | max_abs_value_sum_delta | max_abs_mean_path_length_delta | boundary_partial_rows | full_partial_rows | mean_full_nodes_expanded |",
        "|-------------------:|-----:|----------------:|-----------------:|------------------------:|------------------------:|-------------------------------:|----------------------:|------------------:|-------------------------:|",
    ]
    for budget, rows in sorted(by_budget.items()):
        comparable = [row for row in rows if row.get("comparable")]
        lines.append(
            "| {budget} | {rows} | {comparable} | {matches} | {max_path_delta} | {max_value_delta} | {max_mean_delta} | {boundary_partial} | {full_partial} | {mean_full_nodes} |".format(
                budget=budget,
                rows=len(rows),
                comparable=len(comparable),
                matches=sum(1 for row in rows if validation_exact_match(row)),
                max_path_delta=format_optional(max_optional_field(comparable, "abs_root_path_delta"), 3),
                max_value_delta=format_optional(max_optional_field(comparable, "abs_value_sum_delta"), 6),
                max_mean_delta=format_optional(max_optional_field(comparable, "abs_mean_path_length_delta"), 6),
                boundary_partial=sum(1 for row in rows if row.get("boundary_partial")),
                full_partial=sum(1 for row in rows if row.get("full_partial")),
                mean_full_nodes=format_optional(mean_optional_field(rows, "full_nodes_expanded"), 3),
            )
        )
    lines.extend([
        "",
        "| target_node | path_length_budget | comparable | spliced_total_root_paths | full_root_paths | root_path_delta | spliced_total_value_sum | full_value_sum | value_sum_delta | spliced_mean_path_length | full_mean_path_length | mean_path_length_delta | boundary_partial | full_partial |",
        "|------------:|-------------------:|------------|-------------------------:|----------------:|----------------:|------------------------:|---------------:|----------------:|-------------------------:|----------------------:|-----------------------:|------------------|--------------|",
    ])
    for row in validation_rows:
        lines.append(
            "| {target} | {budget} | {comparable} | {spliced_paths} | {full_paths} | {path_delta} | {spliced_value} | {full_value} | {value_delta} | {spliced_mean} | {full_mean} | {mean_delta} | {boundary_partial} | {full_partial} |".format(
                target=row["target_node"],
                budget=row["path_length_budget"],
                comparable="yes" if row.get("comparable") else "no",
                spliced_paths=format_optional(row.get("spliced_total_root_paths"), 3),
                full_paths=row.get("full_root_paths"),
                path_delta=format_optional(row.get("root_path_delta"), 3),
                spliced_value=format_optional(row.get("spliced_total_value_sum"), 6),
                full_value=format_optional(row.get("full_value_sum"), 6),
                value_delta=format_optional(row.get("value_sum_delta"), 6),
                spliced_mean=format_optional(row.get("spliced_mean_path_length"), 3),
                full_mean=format_optional(row.get("full_mean_path_length"), 3),
                mean_delta=format_optional(row.get("mean_path_length_delta"), 6),
                boundary_partial="yes" if row.get("boundary_partial") else "no",
                full_partial="yes" if row.get("full_partial") else "no",
            )
        )
    return lines


def summarize(records):
    selection = next((row for row in records if row.get("record_type") == "boundary_coverage_selection"), {})
    target_rows = [row for row in records if row.get("record_type") == "boundary_coverage_target"]
    validation_rows = [row for row in records if row.get("record_type") == "boundary_splice_validation"]
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
        "Path value kernel: `{}`".format(selection.get("path_value_kernel", "count")),
        "",
        "Path value branching factor: `{}`".format(format_optional(selection.get("path_value_branching_factor"), 6)),
        "",
        "Path value power: `{}`".format(format_optional(selection.get("path_value_power"), 3)),
        "",
        "Boundary nodes: `{}`".format(selection.get("boundary_nodes", 0)),
        "",
        "Targets: `{}`".format(selection.get("targets", 0)),
        "",
        "Path length budgets: `{}`".format(",".join(str(value) for value in selection.get("budgets", []))),
        "",
    ]
    lines.extend(boundary_coverage_generation_notes(selection))
    lines.extend([""])
    lines.extend(boundary_coverage_table_guide())
    lines.extend([""])
    lines.extend(boundary_coverage_result_implications(selection, target_rows))
    lines.extend([
        "",
        "## Selection",
        "",
        "| role | child_depth | sampled_frontier_nodes |",
        "|------|-------------|------------------------|",
    ])
    for depth, count in sorted_depth_items(selection.get("boundary_counts", {})):
        lines.append("| boundary | {} | {} |".format(depth, count))
    for depth, count in sorted_depth_items(selection.get("target_counts", {})):
        lines.append("| target | {} | {} |".format(depth, count))
    if selection.get("root_cone_counts"):
        lines.extend([
            "",
            "## Root Cone",
            "",
            "| child_depth | new_nodes |",
            "|------------:|----------:|",
        ])
        for depth, count in sorted_depth_items(selection.get("root_cone_counts", {})):
            lines.append("| {} | {} |".format(depth, count))

    lines.extend([
        "",
        "## Coverage Summary",
        "",
        "For sample and root-sample modes, these are observed random-walk outcomes. Use the estimate sections below for branch-product weighted path-space estimates.",
        "",
        "| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | spliced_total_root_paths | spliced_total_value_sum | spliced_mean_path_length | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips | boundary_suffix_path_count_cap_hits | boundary_suffix_expansion_cap_hits |",
        "|------|-------------------:|--------:|------------------:|---------------------------:|--------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|---------------------------:|-------------------------------:|-------------------------:|------------------------:|-------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|------------------------------------:|-----------------------------------:|",
    ])
    for (mode, budget), rows in sorted(by_mode_budget.items()):
        aggregate = aggregate_rows(rows)
        lines.append(
            "| {mode} | {budget} | {targets} | {completed} | {terminal} | {root} | {boundary} | {boundary_fraction} | {budget_exhausted} | {filtered_dead} | {suffix_mean} | {spliced_paths} | {spliced_value} | {spliced_mean_length} | {path_cap} | {expansion_cap} | {cycle_skips} | {root_unreachable} | {suffix_path_cap_hits} | {suffix_expansion_cap_hits} |".format(
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
                spliced_paths=format_optional(aggregate["spliced_total_root_paths"], 3),
                spliced_value=format_optional(aggregate["spliced_total_value_sum"], 6),
                spliced_mean_length=format_optional(aggregate["spliced_mean_path_length"], 3),
                path_cap=aggregate["path_count_cap_hit_targets"],
                expansion_cap=aggregate["expansion_cap_hit_targets"],
                cycle_skips=aggregate["cycle_skips"],
                root_unreachable=aggregate["root_unreachable_parent_skips"],
                suffix_path_cap_hits=aggregate["boundary_suffix_path_count_cap_hits"],
                suffix_expansion_cap_hits=aggregate["boundary_suffix_expansion_cap_hits"],
            )
        )

    validation_section = boundary_splice_validation_section(validation_rows)
    if validation_section:
        lines.extend([""])
        lines.extend(validation_section)

    sampled = [row for row in target_rows if row.get("mode") == "sample"]
    if sampled:
        lines.extend([
            "",
            "## Boundary Sample Estimates",
            "",
            "Boundary sample mode stops at the first boundary. `estimated_spliced_total_root_paths` adds direct root-path weight to boundary-prefix weight multiplied by the remaining-budget suffix mass. `estimated_spliced_total_value_sum` uses the selected path-value kernel over the full prefix-plus-suffix length.",
            "",
            "| path_length_budget | targets | samples_per_target | mean_estimated_terminal_prefixes | mean_estimated_boundary_hit_fraction | mean_estimated_spliced_total_root_paths | mean_estimated_spliced_total_value_sum | mean_estimated_spliced_mean_path_length | mean_ci95_low | mean_ci95_high |",
            "|-------------------:|--------:|-------------------:|---------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------:|----------------------------------------:|--------------:|---------------:|",
        ])
        by_budget = {}
        for row in sampled:
            by_budget.setdefault(row["path_length_budget"], []).append(row)
        for budget, rows in sorted(by_budget.items()):
            lines.append(
                "| {budget} | {targets} | {samples} | {terminal} | {fraction} | {spliced} | {value_sum} | {mean_length} | {lo} | {hi} |".format(
                    budget=budget,
                    targets=len(rows),
                    samples=rows[0].get("samples"),
                    terminal=format_optional(mean_optional_field(rows, "estimated_terminal_prefixes"), 3),
                    fraction=format_optional(mean_optional_field(rows, "estimated_boundary_hit_fraction"), 6),
                    spliced=format_optional(mean_optional_field(rows, "estimated_spliced_total_root_paths"), 3),
                    value_sum=format_optional(mean_optional_field(rows, "estimated_spliced_total_value_sum"), 6),
                    mean_length=format_optional(mean_optional_field(rows, "estimated_spliced_mean_path_length"), 3),
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
            "Root-sample mode ignores boundary stopping and walks until root, budget exhaustion, or dead end. `estimated_root_paths` estimates the root-reaching search-space size from the branch-product weight. `estimated_root_value_sum` applies the selected path-value kernel, and `estimated_kernel_mean_root_path_length` is the corresponding value-weighted mean path length.",
            "",
            "| path_length_budget | targets | samples_per_target | mean_estimated_root_paths | mean_estimated_root_value_sum | mean_estimated_mean_root_path_length | mean_estimated_kernel_mean_root_path_length | mean_estimated_root_boundary_hit_fraction | mean_estimated_root_value_boundary_hit_fraction |",
            "|-------------------:|--------:|-------------------:|--------------------------:|------------------------------:|-------------------------------------:|--------------------------------------------:|------------------------------------------:|------------------------------------------------:|",
        ])
        by_budget = {}
        for row in root_sampled:
            by_budget.setdefault(row["path_length_budget"], []).append(row)
        for budget, rows in sorted(by_budget.items()):
            lines.append(
                "| {budget} | {targets} | {samples} | {root_paths} | {value_sum} | {mean_len} | {kernel_mean_len} | {boundary_fraction} | {value_boundary_fraction} |".format(
                    budget=budget,
                    targets=len(rows),
                    samples=rows[0].get("samples"),
                    root_paths=format_optional(mean_optional_field(rows, "estimated_root_paths"), 3),
                    value_sum=format_optional(mean_optional_field(rows, "estimated_root_value_sum"), 6),
                    mean_len=format_optional(mean_optional_field(rows, "estimated_mean_root_path_length"), 3),
                    kernel_mean_len=format_optional(mean_optional_field(rows, "estimated_kernel_mean_root_path_length"), 3),
                    boundary_fraction=format_optional(mean_optional_field(rows, "estimated_root_boundary_hit_fraction"), 6),
                    value_boundary_fraction=format_optional(mean_optional_field(rows, "estimated_root_value_boundary_hit_fraction"), 6),
                )
            )

    lines.extend([
        "",
        "## Target Rows",
        "",
        "Each row is one target under one path-length budget. `root_paths` counts direct root-reaching terminals found before the search stops at a boundary. A row with `root_paths=0` and positive `boundary_hit_prefixes` is boundary-covered; it is not automatically root-unreachable. When boundary suffix mass is disabled, use these rows to judge boundary coverage rather than total root-path mass.",
        "",
        "| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips | spliced_total_root_paths | spliced_total_value_sum | spliced_mean_path_length |",
        "|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|-------------------------:|------------------------:|-------------------------:|",
    ])
    for row in target_rows:
        lines.append(
            "| {mode} | {target} | {budget} | {terminal} | {root} | {boundary} | {fraction} | {budget_exhausted} | {filtered_dead} | {remaining} | {completed} | {cycles} | {root_unreachable} | {spliced_paths} | {spliced_value} | {spliced_mean_length} |".format(
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
                spliced_paths=format_optional(row.get("spliced_total_root_paths"), 3),
                spliced_value=format_optional(row.get("spliced_total_value_sum"), 6),
                spliced_mean_length=format_optional(row.get("spliced_mean_path_length"), 3),
            )
        )

    return "\n".join(lines) + "\n"


def resolve_path_value_kernel(args, graph, root, targets, boundary_nodes, root_cone_depth_by_node):
    name = normalize_path_value_kernel_name(args.path_value_kernel)
    branching_stats = None
    branching_factor = args.path_value_branching_factor
    branching_factor_source = None

    if name == "bp-decay":
        if branching_factor is not None and float(branching_factor) > 0.0:
            branching_factor = float(branching_factor)
            branching_factor_source = "user"
        else:
            root_cone_depth_by_node = root_cone_depth_by_node or {}
            if root_cone_depth_by_node:
                root_cone_nodes = [node for node in root_cone_depth_by_node if node != root]
                root_cone_set = set(root_cone_depth_by_node)
                if args.parent_filter == "root-cone" or args.selection_source == "root-cone":
                    parent_accept = lambda _node, parent: parent in root_cone_set
                    branching_factor_source = "root_cone_eligible_parent_e_p2_over_e_p"
                else:
                    parent_accept = None
                    branching_factor_source = "root_cone_full_parent_e_p2_over_e_p"
                branching_stats = estimate_parent_branching_factor(graph.parents, root_cone_nodes, parent_accept)
            else:
                scope_nodes = list(dict.fromkeys(list(targets) + list(boundary_nodes)))
                branching_factor_source = "selected_nodes_full_parent_e_p2_over_e_p"
                branching_stats = estimate_parent_branching_factor(graph.parents, scope_nodes)
            branching_factor = branching_stats["branching_factor"]

    kernel = make_path_value_kernel(
        name,
        branching_factor=branching_factor,
        branching_factor_source=branching_factor_source,
        power=args.path_value_power,
    )
    return kernel, branching_stats


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
        measure_suffix_mass = (
            not args.skip_boundary_suffix_mass
            and (reachability is None or args.measure_filtered_boundary_suffix_mass or args.validate_full_exact)
        )
        suffix_parent_filter = reachability if measure_suffix_mass and reachability is not None else None
        path_value_kernel, path_value_branching_stats = resolve_path_value_kernel(
            args,
            graph,
            args.root,
            targets,
            boundary_nodes,
            root_cone_depth_by_node,
        )

        records = [{
            "record_type": "boundary_coverage_selection",
            "graph": args.graph_name,
            "root": args.root,
            "seed": args.seed,
            "parent_filter": args.parent_filter,
            "path_value_kernel": path_value_kernel.name,
            "path_value_branching_factor": path_value_kernel.branching_factor,
            "path_value_branching_factor_source": path_value_kernel.branching_factor_source,
            "path_value_power": path_value_kernel.power,
            "path_value_branching_stats": path_value_branching_stats,
            "selection_source": args.selection_source,
            "boundary_depths": boundary_depths,
            "target_depths": target_depths,
            "root_cone_depth": root_cone_depth if root_cone_depth_by_node is not None else None,
            "root_cone_nodes": len(root_cone_depth_by_node or {}),
            "root_cone_counts": root_cone_counts,
            "root_cone_children_per_node": args.root_cone_children_per_node if args.root_cone_children_per_node is not None else args.children_per_node,
            "root_cone_frontier_limit": args.root_cone_frontier_limit if args.root_cone_frontier_limit is not None else args.frontier_limit,
            "children_per_node": args.children_per_node,
            "frontier_limit": args.frontier_limit,
            "boundaries_per_depth": args.boundaries_per_depth,
            "targets_per_depth": args.targets_per_depth,
            "require_targets_in_root_cone": args.require_targets_in_root_cone,
            "require_boundaries_in_root_cone": args.require_boundaries_in_root_cone,
            "boundary_counts": boundary_counts,
            "target_counts": target_counts,
            "boundary_nodes": len(boundary_nodes),
            "selected_boundary_nodes": len(selected_boundary_nodes),
            "target_ancestor_boundary_nodes_added": len(set(boundary_nodes) - selected_boundary_nodes),
            "include_target_ancestor_boundaries": args.include_target_ancestor_boundaries,
            "target_ancestor_boundary_limit": args.target_ancestor_boundary_limit,
            "targets": len(targets),
            "target_selection": target_selection_label,
            "budgets": budgets,
            "mode": args.mode,
            "samples": args.samples,
            "path_count_cap": args.path_count_cap,
            "expansion_cap": args.expansion_cap,
            "measure_boundary_suffix_mass": measure_suffix_mass,
            "measure_filtered_boundary_suffix_mass": args.measure_filtered_boundary_suffix_mass,
            "boundary_suffix_parent_filter": args.parent_filter if suffix_parent_filter is not None else "none",
            "validate_full_exact": args.validate_full_exact,
        }]

        boundary_set = set(boundary_nodes)
        for target in targets:
            for budget in budgets:
                if args.mode in {"exact", "both", "all"}:
                    exact_row = time_target(lambda target=target, budget=budget: exact_boundary_coverage(
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
                        suffix_parent_filter,
                        args.path_count_cap,
                        args.expansion_cap,
                        path_value_kernel,
                    ))
                    exact_row["child_sample_depth"] = target_depth_by_node[target]
                    exact_row["graph"] = args.graph_name
                    exact_row["root"] = args.root
                    records.append(exact_row)
                    if args.validate_full_exact:
                        records.append(full_exact_splice_validation_record(
                            exact_row,
                            graph.parents,
                            args.root,
                            args.path_count_cap,
                            args.expansion_cap,
                            reachability,
                            path_value_kernel,
                        ))
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
                        suffix_parent_filter,
                        path_value_kernel,
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
                        path_value_kernel,
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
    parser.add_argument("--measure-filtered-boundary-suffix-mass", action="store_true", help="When a parent filter is active, also materialize boundary suffix histograms through the same filter so boundary hits can be converted into spliced root-path mass/value estimates.")
    parser.add_argument("--validate-full-exact", action="store_true", help="Run full filtered DFS for exact rows and compare it with boundary-stopped suffix splicing. Implies filtered suffix measurement when a parent filter is active.")
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
    parser.add_argument("--path-value-kernel", choices=PATH_VALUE_KERNELS, default="count", help="Path-value kernel for value-sum estimates: count, bp-decay, or weighted-power.")
    parser.add_argument("--path-value-branching-factor", type=float, help="Branching factor b_p for bp-decay. Omit or pass a non-positive value to estimate E[p^2]/E[p] from the current root-cone or selected-node scope.")
    parser.add_argument("--path-value-power", type=float, default=1.0, help="Power n for weighted-power: g(L) = (L + 1)^(-n).")
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
