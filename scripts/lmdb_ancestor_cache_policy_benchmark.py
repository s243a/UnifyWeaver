#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Benchmark fixed-size cache policy for root-anchored ancestor histograms.

This script models the residency decision separately from bounded path search.
Each query first collects the parent-only ancestor search space for a sampled
target.  Cache admission is limited to nodes in that current ancestor cone whose
root-distance summary is near the root.  A fixed-size direct-mapped cache then
uses ancestor-cone membership during collisions: entries outside the current
cone are cheap to replace; entries inside the cone compete by root-distance
priority.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    parse_int_list,
    root_distances,
    select_targets_by_child_depth,
)
from scripts.lmdb_parent_histogram_benchmark import percentile, safe_graph_name


@dataclass
class AncestorSpace:
    nodes: set
    edges_seen: int = 0
    cycle_edges: int = 0
    capped: bool = False


@dataclass
class CacheEntry:
    node: int
    l_min: object = None
    l_max: object = None
    truncated: bool = False
    hits: int = 0
    stores: int = 1


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def collect_ancestor_space(parents_func, target, max_nodes, max_edges):
    """Collect unique nodes reachable by simple parent-only traversal.

    The caps bound diagnostic collection work.  They are not the path-length
    budget used by histogram search.
    """
    nodes = {target}
    stack = [(target, (target,))]
    space = AncestorSpace(nodes=nodes)

    while stack:
        node, path = stack.pop()
        for parent in parents_func(node):
            if space.edges_seen >= max_edges:
                space.capped = True
                continue
            space.edges_seen += 1
            if parent in path:
                space.cycle_edges += 1
                continue
            if parent in nodes:
                continue
            if len(nodes) >= max_nodes:
                space.capped = True
                continue
            nodes.add(parent)
            stack.append((parent, path + (parent,)))
    return space


def slot_for(node, cache_slots):
    return int(node) % int(cache_slots)


def cache_priority(entry):
    missing_l_max = entry.l_max is None
    missing_l_min = entry.l_min is None
    l_max = 10 ** 12 if missing_l_max else int(entry.l_max)
    l_min = 10 ** 12 if missing_l_min else int(entry.l_min)
    return (bool(entry.truncated), missing_l_max, l_max, missing_l_min, l_min, -int(entry.hits), int(entry.node))


def update_cache(cache, candidate, ancestor_nodes, cache_slots):
    slot = slot_for(candidate.node, cache_slots)
    existing = cache.get(slot)
    if existing is None:
        cache[slot] = candidate
        return "insert"
    if existing.node == candidate.node:
        existing.hits += 1
        existing.stores += 1
        return "refresh"
    if existing.node not in ancestor_nodes:
        cache[slot] = candidate
        return "overwrite_outside_cone"
    if cache_priority(candidate) < cache_priority(existing):
        cache[slot] = candidate
        return "overwrite_lower_priority"
    return "keep_existing"


def candidate_from_distance(node, distance):
    return CacheEntry(
        node=int(node),
        l_min=distance["L_min"],
        l_max=distance["L_max"],
        truncated=bool(distance["truncated"]),
    )


def admit_candidate(distance, admit_l_min, admit_l_max):
    if distance["L_min"] is None or distance["L_max"] is None:
        return False
    if int(distance["L_min"]) > admit_l_min:
        return False
    if int(distance["L_max"]) > admit_l_max:
        return False
    return True


def cache_entry_record(graph_name, root, slot, entry):
    record = asdict(entry)
    record.update({
        "record_type": "ancestor_cache_final_entry",
        "graph": graph_name,
        "root": root,
        "slot": slot,
    })
    return record


def query_record(graph_name, root, target, child_depth, space, candidates, cache_hits_before, cache_occupancy, action_counts):
    return {
        "record_type": "ancestor_cache_policy_query",
        "graph": graph_name,
        "root": root,
        "target_node": target,
        "child_sample_depth": child_depth,
        "ancestor_nodes": len(space.nodes),
        "ancestor_edges_seen": space.edges_seen,
        "cycle_edges_seen": space.cycle_edges,
        "ancestor_collection_capped": space.capped,
        "admission_candidates": len(candidates),
        "candidate_l_min_min": min((candidate.l_min for candidate in candidates), default=None),
        "candidate_l_max_max": max((candidate.l_max for candidate in candidates), default=None),
        "cache_hits_before": cache_hits_before,
        "cache_occupancy_after": cache_occupancy,
        "cache_actions": dict(action_counts),
    }


def summarize(records, args, target_counts):
    query_rows = [row for row in records if row["record_type"] == "ancestor_cache_policy_query"]
    action_counts = Counter()
    for row in query_rows:
        action_counts.update(row["cache_actions"])
    ancestor_nodes = [row["ancestor_nodes"] for row in query_rows]
    candidates = [row["admission_candidates"] for row in query_rows]
    hits = [row["cache_hits_before"] for row in query_rows]
    return {
        "record_type": "ancestor_cache_policy_summary",
        "graph": args.graph_name,
        "root": args.root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_depths": parse_int_list(args.target_depths),
        "target_selection_counts": target_counts,
        "targets": len(query_rows),
        "cache_slots": args.cache_slots,
        "admit_l_min": args.admit_l_min,
        "admit_l_max": args.admit_l_max,
        "final_cache_entries": len([row for row in records if row["record_type"] == "ancestor_cache_final_entry"]),
        "capped_queries": sum(1 for row in query_rows if row["ancestor_collection_capped"]),
        "mean_ancestor_nodes": mean(ancestor_nodes),
        "p95_ancestor_nodes": percentile(ancestor_nodes, 95),
        "mean_admission_candidates": mean(candidates),
        "p95_admission_candidates": percentile(candidates, 95),
        "mean_cache_hits_before": mean(hits),
        "p95_cache_hits_before": percentile(hits, 95),
        "cache_actions": dict(action_counts),
    }


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
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

        records = [{
            "record_type": "ancestor_cache_policy_selection",
            "graph": args.graph_name,
            "root": args.root,
            "target_depths": target_depths,
            "target_selection_counts": target_counts,
            "targets": len(targets),
            "children_per_node": args.children_per_node,
            "frontier_limit": args.frontier_limit,
            "targets_per_depth": args.targets_per_depth,
            "max_ancestor_nodes": args.max_ancestor_nodes,
            "max_ancestor_edges": args.max_ancestor_edges,
            "root_distance_cap": args.root_distance_cap,
            "cache_slots": args.cache_slots,
            "admit_l_min": args.admit_l_min,
            "admit_l_max": args.admit_l_max,
            "seed": args.seed,
        }]

        cache = {}
        distance_memo = {}
        for target in targets:
            space = collect_ancestor_space(
                graph.parents,
                target,
                args.max_ancestor_nodes,
                args.max_ancestor_edges,
            )
            cache_hits_before = sum(1 for entry in cache.values() if entry.node in space.nodes)
            candidates = []
            for node in sorted(space.nodes):
                distance = root_distances(node, args.root, graph.parents, args.root_distance_cap, distance_memo)
                if admit_candidate(distance, args.admit_l_min, args.admit_l_max):
                    candidates.append(candidate_from_distance(node, distance))

            action_counts = Counter()
            for candidate in sorted(candidates, key=cache_priority):
                action_counts[update_cache(cache, candidate, space.nodes, args.cache_slots)] += 1

            records.append(query_record(
                args.graph_name,
                args.root,
                target,
                target_depth_by_node[target],
                space,
                candidates,
                cache_hits_before,
                len(cache),
                action_counts,
            ))

        for slot in sorted(cache):
            records.append(cache_entry_record(args.graph_name, args.root, slot, cache[slot]))

        records.append(summarize(records, args, target_counts))
        return records
    finally:
        graph.close()


def write_records(records, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "{}_ancestor_cache_policy".format(safe_graph_name(graph_name))
    jsonl_path = output_dir / "{}.jsonl".format(stem)
    summary_path = output_dir / "{}_summary.json".format(stem)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = records[-1]
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return jsonl_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_category_graph")
    parser.add_argument("--target-depths", default="3,4")
    parser.add_argument("--children-per-node", type=int, default=128)
    parser.add_argument("--frontier-limit", type=int, default=2000)
    parser.add_argument("--targets-per-depth", type=int, default=40)
    parser.add_argument("--max-ancestor-nodes", type=int, default=5000)
    parser.add_argument("--max-ancestor-edges", type=int, default=25000)
    parser.add_argument("--root-distance-cap", type=int, default=48)
    parser.add_argument("--cache-slots", type=int, default=512)
    parser.add_argument("--admit-l-min", type=int, default=6)
    parser.add_argument("--admit-l-max", type=int, default=12)
    parser.add_argument("--seed", default="ancestor-cache-policy-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    records = run_benchmark(args)
    jsonl_path, summary_path = write_records(records, args.output_dir, args.graph_name)
    print(json.dumps(records[-1], indent=2, sort_keys=True))
    print("wrote {}".format(jsonl_path))
    print("wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
