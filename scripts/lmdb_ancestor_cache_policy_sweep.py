#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Sweep ancestor-cache policy settings with one shared cone collection.

The single-config benchmark can emit a full JSONL trace when explicitly asked.
This sweep driver is intentionally summary-only: it samples targets once,
collects each target's ancestor cone once, computes root-distance summaries once,
then replays cache residency for each cache/admission configuration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_ancestor_cache_policy_benchmark import (
    admit_candidate,
    cache_priority,
    candidate_from_distance,
    collect_ancestor_space,
    update_cache,
)
from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    parse_int_list,
    root_distances,
    select_targets_by_child_depth,
)
from scripts.lmdb_parent_histogram_benchmark import percentile, safe_graph_name


@dataclass
class QueryInput:
    target: int
    child_depth: int
    space: object


@dataclass
class CacheConfig:
    cache_slots: int
    admit_l_min: int
    admit_l_max: int


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def parse_grid(text, default):
    if text is None or str(text).strip() == "":
        return [default]
    return parse_int_list(text)


def sweep_configs(args):
    cache_slots_values = parse_grid(args.sweep_cache_slots, args.cache_slots)
    l_min_values = parse_grid(args.sweep_admit_l_min, args.admit_l_min)
    l_max_values = parse_grid(args.sweep_admit_l_max, args.admit_l_max)
    return [
        CacheConfig(cache_slots=cache_slots, admit_l_min=l_min, admit_l_max=l_max)
        for cache_slots, l_min, l_max in product(cache_slots_values, l_min_values, l_max_values)
    ]


def action_rates(action_counts):
    total = sum(action_counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(action_counts.items())}


def collect_query_inputs(graph, args):
    target_depths = parse_int_list(args.target_depths)
    targets, target_depth_by_node, target_counts = select_targets_by_child_depth(
        graph,
        args.root,
        target_depths,
        args.children_per_node,
        args.frontier_limit,
        args.targets_per_depth,
        args.seed,
    )
    query_inputs = []
    for target in targets:
        query_inputs.append(QueryInput(
            target=target,
            child_depth=target_depth_by_node[target],
            space=collect_ancestor_space(
                graph.parents,
                target,
                args.max_ancestor_nodes,
                args.max_ancestor_edges,
            ),
        ))
    return targets, target_counts, query_inputs


def collect_distances(graph, args, query_inputs):
    distance_memo = {}
    distances = {}
    for query in query_inputs:
        for node in query.space.nodes:
            if node not in distances:
                distances[node] = root_distances(node, args.root, graph.parents, args.root_distance_cap, distance_memo)
    return distances


def simulate_config(args, target_counts, query_inputs, distances, config):
    cache = {}
    action_counts = Counter()
    ancestor_nodes = []
    ancestor_edges = []
    admission_candidates = []
    cache_hits_before = []
    hit_rates = []
    capped_queries = 0
    cycle_edges = 0

    for query in query_inputs:
        space = query.space
        if space.capped:
            capped_queries += 1
        cycle_edges += space.cycle_edges
        ancestor_nodes.append(len(space.nodes))
        ancestor_edges.append(space.edges_seen)
        hits_before = sum(1 for entry in cache.values() if entry.node in space.nodes)
        candidates = []
        for node in sorted(space.nodes):
            distance = distances[node]
            if admit_candidate(distance, config.admit_l_min, config.admit_l_max):
                candidates.append(candidate_from_distance(node, distance))

        for candidate in sorted(candidates, key=cache_priority):
            action_counts[update_cache(cache, candidate, space.nodes, config.cache_slots)] += 1

        admission_candidates.append(len(candidates))
        cache_hits_before.append(hits_before)
        hit_rates.append(0.0 if not candidates else hits_before / len(candidates))

    final_cache_entries = len(cache)
    return {
        "record_type": "ancestor_cache_policy_sweep_summary",
        "graph": args.graph_name,
        "root": args.root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_depths": parse_int_list(args.target_depths),
        "target_selection_counts": target_counts,
        "targets": len(query_inputs),
        "cache_slots": config.cache_slots,
        "admit_l_min": config.admit_l_min,
        "admit_l_max": config.admit_l_max,
        "final_cache_entries": final_cache_entries,
        "cache_occupancy_rate": 0.0 if config.cache_slots == 0 else final_cache_entries / config.cache_slots,
        "capped_queries": capped_queries,
        "capped_query_rate": 0.0 if not query_inputs else capped_queries / len(query_inputs),
        "cycle_edges_seen": cycle_edges,
        "mean_ancestor_nodes": mean(ancestor_nodes),
        "p95_ancestor_nodes": percentile(ancestor_nodes, 95),
        "mean_ancestor_edges": mean(ancestor_edges),
        "p95_ancestor_edges": percentile(ancestor_edges, 95),
        "mean_admission_candidates": mean(admission_candidates),
        "p95_admission_candidates": percentile(admission_candidates, 95),
        "mean_cache_hits_before": mean(cache_hits_before),
        "p95_cache_hits_before": percentile(cache_hits_before, 95),
        "mean_cache_hits_before_per_candidate": mean(hit_rates),
        "cache_actions": dict(action_counts),
        "cache_action_rates": action_rates(action_counts),
    }


def run_sweep(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        _targets, target_counts, query_inputs = collect_query_inputs(graph, args)
        distances = collect_distances(graph, args, query_inputs)
        return [
            simulate_config(args, target_counts, query_inputs, distances, config)
            for config in sweep_configs(args)
        ]
    finally:
        graph.close()


def write_sweep(records, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "{}_ancestor_cache_policy_sweep_summary.json".format(safe_graph_name(graph_name))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_category_graph")
    parser.add_argument("--target-depths", default="3,4")
    parser.add_argument("--children-per-node", type=int, default=64)
    parser.add_argument("--frontier-limit", type=int, default=800)
    parser.add_argument("--targets-per-depth", type=int, default=12)
    parser.add_argument("--max-ancestor-nodes", type=int, default=1500)
    parser.add_argument("--max-ancestor-edges", type=int, default=8000)
    parser.add_argument("--root-distance-cap", type=int, default=32)
    parser.add_argument("--cache-slots", type=int, default=256)
    parser.add_argument("--admit-l-min", type=int, default=6)
    parser.add_argument("--admit-l-max", type=int, default=12)
    parser.add_argument("--sweep-cache-slots", default="128,256,512")
    parser.add_argument("--sweep-admit-l-min", default=None)
    parser.add_argument("--sweep-admit-l-max", default="8,10,12")
    parser.add_argument("--seed", default="ancestor-cache-policy-sweep-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    records = run_sweep(args)
    path = write_sweep(records, args.output_dir, args.graph_name)
    print(json.dumps(records, indent=2, sort_keys=True))
    print("wrote {}".format(path))


if __name__ == "__main__":
    main()
