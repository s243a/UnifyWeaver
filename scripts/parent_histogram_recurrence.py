#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Parent-path histogram recurrence helpers.

For a parent-only DAG the recurrence is exact:

    H_root[0] = 1
    H_v[d] = sum(H_parent[d - 1] for parent in parents(v))

The helpers use unnormalized histograms: every bin stores path-count mass.  If
callers store normalized parent distributions instead, each shifted parent
distribution must be multiplied by its path count N_p before the sum, and N_v
must be stored separately.

This avoids enumerating every path from a target to root.  In cyclic graphs a
per-node histogram cannot enforce a unique-node visited set, so cycle encounters
are reported through `cycle_approximation`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class RecurrenceHistogramStats:
    states_evaluated: int = 0
    memo_hits: int = 0
    edges_examined: int = 0
    cycle_edges: int = 0
    budget_cutoffs: int = 0
    path_cap_hit: bool = False
    expansion_cap_hit: bool = False
    cycle_approximation: bool = False


def shifted_add(out, parent_hist, remaining, path_cap, stats):
    for length, count in parent_hist.items():
        shifted = int(length) + 1
        if shifted <= remaining:
            out[shifted] += int(count)
            if path_cap is not None and sum(out.values()) >= path_cap:
                stats.path_cap_hit = True
                return


def recurrence_parent_histogram(parents_func, target, root, budget, path_cap=None, expansion_cap=None):
    """Compute a finite-horizon parent histogram by shifted parent sums.

    The returned histogram counts root-reaching walks under the recurrence.  It
    is exact for DAG parent cones.  If a cycle is encountered, the cyclic edge is
    skipped and `cycle_approximation` is set because a node-only recurrence does
    not carry the visited-set state required for exact simple-path semantics.
    """
    stats = RecurrenceHistogramStats()
    memo = {}
    visiting = set()

    def rec(node, remaining):
        key = (node, remaining)
        if key in memo:
            stats.memo_hits += 1
            return memo[key]
        if node == root:
            memo[key] = {0: 1}
            return memo[key]
        if remaining <= 0:
            stats.budget_cutoffs += 1
            memo[key] = {}
            return memo[key]
        if node in visiting:
            stats.cycle_edges += 1
            stats.cycle_approximation = True
            return {}
        if expansion_cap is not None and stats.states_evaluated >= expansion_cap:
            stats.expansion_cap_hit = True
            return {}

        visiting.add(node)
        stats.states_evaluated += 1
        out = Counter()
        for parent in parents_func(node):
            stats.edges_examined += 1
            if parent in visiting:
                stats.cycle_edges += 1
                stats.cycle_approximation = True
                continue
            parent_hist = rec(parent, remaining - 1)
            shifted_add(out, parent_hist, remaining, path_cap, stats)
            if stats.path_cap_hit or stats.expansion_cap_hit:
                break
        visiting.remove(node)
        memo[key] = dict(sorted(out.items()))
        return memo[key]

    return dict(sorted(rec(target, int(budget)).items())), stats
