#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Diagnose full parent branching for numeric-keyed category LMDB exports.

Sampled TSVs only contain parents retained by the sampling procedure.  This
script samples targets from an LMDB child index, but measures parent degree from
the full LMDB parent index and buckets the signal by maximum parent-only
distance to the chosen root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DEPTHS = [1, 2, 3, 4]


def parse_int_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo = int(lo_text)
            hi = int(hi_text)
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(part))
    return values


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def size_biased_branching(degrees):
    degrees = [int(degree) for degree in degrees if degree is not None and degree > 0]
    if not degrees:
        return {
            "nodes": 0,
            "mean_parent_degree": 0.0,
            "second_parent_degree_moment": 0.0,
            "size_biased_parent_branching": None,
            "mean_excess": None,
            "max_parent_degree": 0,
            "p95_parent_degree": 0.0,
        }
    mean = sum(degrees) / len(degrees)
    second = sum(degree * degree for degree in degrees) / len(degrees)
    branching = None if mean == 0.0 else second / mean
    return {
        "nodes": len(degrees),
        "mean_parent_degree": mean,
        "second_parent_degree_moment": second,
        "size_biased_parent_branching": branching,
        "mean_excess": None if branching is None else branching - 1.0,
        "max_parent_degree": max(degrees),
        "p95_parent_degree": percentile(degrees, 95),
    }


def deterministic_sample(values, limit, seed_text):
    values = list(dict.fromkeys(values))
    if limit is None or len(values) <= limit:
        return values
    digest = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=16).hexdigest()
    rng = random.Random(digest)
    return sorted(rng.sample(values, limit))


def encode_key(value):
    return struct.pack("<i", int(value))


def decode_value(value):
    return struct.unpack("<i", value)[0]


def iter_dups(txn, db, key):
    cursor = txn.cursor(db=db)
    if not cursor.set_key(encode_key(key)):
        return []
    return [decode_value(value) for value in cursor.iternext_dup()]


class LmdbCategoryGraph:
    def __init__(self, lmdb_dir):
        try:
            import lmdb
        except ImportError as exc:
            raise SystemExit("python-lmdb is required for LMDB diagnostics") from exc

        self.env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, max_dbs=32)
        self.txn = self.env.begin()
        self.category_parent = self.env.open_db(b"category_parent", txn=self.txn)
        self.category_child = self.env.open_db(b"category_child", txn=self.txn)
        self._parents = {}
        self._children = {}

    def close(self):
        self.txn.abort()
        self.env.close()

    def parents(self, node):
        if node not in self._parents:
            self._parents[node] = iter_dups(self.txn, self.category_parent, node)
        return self._parents[node]

    def children(self, node):
        if node not in self._children:
            self._children[node] = iter_dups(self.txn, self.category_child, node)
        return self._children[node]


def select_targets_by_child_depth(graph, root, depths, children_per_node, frontier_limit, targets_per_depth, seed):
    max_depth = max(depths) if depths else 0
    depth_nodes = {0: [root]}
    frontier = [root]
    for depth in range(1, max_depth + 1):
        next_nodes = []
        for node in frontier:
            children = deterministic_sample(
                graph.children(node),
                children_per_node,
                "{}:children:{}:{}".format(seed, depth, node),
            )
            next_nodes.extend(children)
        next_nodes = deterministic_sample(
            list(dict.fromkeys(next_nodes)),
            frontier_limit,
            "{}:frontier:{}".format(seed, depth),
        )
        depth_nodes[depth] = next_nodes
        frontier = next_nodes

    targets = []
    target_child_depth = {}
    for depth in depths:
        sampled = deterministic_sample(
            depth_nodes.get(depth, []),
            targets_per_depth,
            "{}:targets:{}".format(seed, depth),
        )
        for node in sampled:
            if node not in target_child_depth:
                targets.append(node)
                target_child_depth[node] = depth
    return targets, target_child_depth, {depth: len(nodes) for depth, nodes in depth_nodes.items()}


def root_distances(node, root, parents_func, max_parent_depth, memo=None, visiting=None):
    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()
    key = (node, max_parent_depth)
    if key in memo:
        return memo[key]
    if node == root:
        memo[key] = {"L_min": 0, "L_max": 0, "truncated": False, "cycle_skipped": False}
        return memo[key]
    if max_parent_depth <= 0:
        return {"L_min": None, "L_max": None, "truncated": True, "cycle_skipped": False}
    if node in visiting:
        return {"L_min": None, "L_max": None, "truncated": False, "cycle_skipped": True}

    visiting.add(node)
    mins = []
    maxes = []
    truncated = False
    cycle_skipped = False
    for parent in parents_func(node):
        result = root_distances(parent, root, parents_func, max_parent_depth - 1, memo, visiting)
        truncated = truncated or bool(result["truncated"])
        cycle_skipped = cycle_skipped or bool(result["cycle_skipped"])
        if result["L_min"] is not None:
            mins.append(int(result["L_min"]) + 1)
        if result["L_max"] is not None:
            maxes.append(int(result["L_max"]) + 1)
    visiting.remove(node)
    memo[key] = {
        "L_min": min(mins) if mins else None,
        "L_max": max(maxes) if maxes else None,
        "truncated": truncated,
        "cycle_skipped": cycle_skipped,
    }
    return memo[key]


def target_record(graph_name, root, target, child_depth, parents, distances, parent_distances):
    root_reaching_parent_degree = 0
    for parent in parents:
        if parent_distances(parent)["L_min"] is not None:
            root_reaching_parent_degree += 1
    return {
        "record_type": "lmdb_parent_branching_target",
        "graph": graph_name,
        "root": root,
        "target_node": target,
        "child_sample_depth": child_depth,
        "L_min": distances["L_min"],
        "L_max": distances["L_max"],
        "distance_truncated": distances["truncated"],
        "cycle_skipped": distances["cycle_skipped"],
        "full_parent_degree": len(parents),
        "root_reaching_parent_degree": root_reaching_parent_degree,
    }


def bucket_records(records):
    buckets = {}
    for record in records:
        if record["record_type"] != "lmdb_parent_branching_target":
            continue
        key = "unreachable_or_truncated" if record.get("L_max") is None else int(record["L_max"])
        buckets.setdefault(key, []).append(record)

    out = []
    for key in sorted(buckets, key=lambda value: (isinstance(value, str), value)):
        rows = buckets[key]
        out.append({
            "record_type": "lmdb_parent_branching_bucket",
            "L_max_bucket": key,
            "targets": len(rows),
            "full_parent_degree": size_biased_branching([row["full_parent_degree"] for row in rows]),
            "root_reaching_parent_degree": size_biased_branching([row["root_reaching_parent_degree"] for row in rows]),
            "truncated_targets": sum(1 for row in rows if row["distance_truncated"]),
            "cycle_skipped_targets": sum(1 for row in rows if row["cycle_skipped"]),
        })
    return out


def format_none(value):
    return "n/a" if value is None else str(value)


def format_optional(value):
    return "n/a" if value is None else "{:.6f}".format(float(value))


def summarize(records, graph_name, root, selection_counts):
    target_rows = [row for row in records if row["record_type"] == "lmdb_parent_branching_target"]
    bucket_rows = [row for row in records if row["record_type"] == "lmdb_parent_branching_bucket"]
    reachable_rows = [row for row in target_rows if row["L_max"] is not None]
    lines = [
        "# LMDB Parent Branching Diagnostic",
        "",
        "Graph: `{}`".format(graph_name),
        "",
        "Root: `{}`".format(root),
        "",
        "## Selection",
        "",
        "| child_depth | sampled_frontier_nodes |",
        "|-------------|------------------------|",
    ]
    for depth in sorted(selection_counts):
        lines.append("| {} | {} |".format(depth, selection_counts[depth]))
    lines.extend([
        "",
        "## Target Summary",
        "",
        "| targets | root_reachable | truncated | cycle_skipped |",
        "|---------|----------------|-----------|---------------|",
        "| {} | {} | {} | {} |".format(
            len(target_rows),
            len(reachable_rows),
            sum(1 for row in target_rows if row["distance_truncated"]),
            sum(1 for row in target_rows if row["cycle_skipped"]),
        ),
        "",
        "## Parent Degree By Maximum Parent Distance To Root",
        "",
        "| L_max | targets | mean_full_p | p95_full_p | max_full_p | b_full | mean_excess_full | mean_root_p | p95_root_p | max_root_p | b_root | mean_excess_root |",
        "|-------|---------|-------------|------------|------------|--------|------------------|-------------|------------|------------|--------|------------------|",
    ])
    for row in bucket_rows:
        full = row["full_parent_degree"]
        reachable = row["root_reaching_parent_degree"]
        lines.append(
            "| {bucket} | {targets} | {mean_full:.3f} | {p95_full:.3f} | {max_full} | {b_full} | {excess_full} | {mean_root:.3f} | {p95_root:.3f} | {max_root} | {b_root} | {excess_root} |".format(
                bucket=row["L_max_bucket"],
                targets=row["targets"],
                mean_full=full["mean_parent_degree"],
                p95_full=full["p95_parent_degree"],
                max_full=full["max_parent_degree"],
                b_full=format_optional(full["size_biased_parent_branching"]),
                excess_full=format_optional(full["mean_excess"]),
                mean_root=reachable["mean_parent_degree"],
                p95_root=reachable["p95_parent_degree"],
                max_root=reachable["max_parent_degree"],
                b_root=format_optional(reachable["size_biased_parent_branching"]),
                excess_root=format_optional(reachable["mean_excess"]),
            )
        )
    lines.extend([
        "",
        "## Highest Parent-Degree Sampled Targets",
        "",
        "| target | child_depth | L_min | L_max | full_parent_degree | root_reaching_parent_degree |",
        "|--------|-------------|-------|-------|--------------------|-----------------------------|",
    ])
    for row in sorted(target_rows, key=lambda item: (item["full_parent_degree"], item["root_reaching_parent_degree"]), reverse=True)[:20]:
        lines.append(
            "| {target} | {child_depth} | {lmin} | {lmax} | {full} | {root_p} |".format(
                target=row["target_node"],
                child_depth=row["child_sample_depth"],
                lmin=format_none(row["L_min"]),
                lmax=format_none(row["L_max"]),
                full=row["full_parent_degree"],
                root_p=row["root_reaching_parent_degree"],
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_graph = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in graph_name).strip("_") or "graph"
    jsonl_path = output_dir / "lmdb_parent_branching_diagnostic_{}_{}.jsonl".format(safe_graph, timestamp)
    summary_path = output_dir / "lmdb_parent_branching_diagnostic_summary_{}_{}.md".format(safe_graph, timestamp)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def run_lmdb_diagnostic(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        depths = parse_int_list(args.child_depths)
        targets, child_depths, selection_counts = select_targets_by_child_depth(
            graph,
            args.root,
            depths,
            args.children_per_node,
            args.frontier_limit,
            args.targets_per_depth,
            args.seed,
        )
        distance_memo = {}

        def distances(node):
            return root_distances(node, args.root, graph.parents, args.max_parent_depth, distance_memo)

        records = []
        for target in targets:
            records.append(
                target_record(
                    args.graph_name,
                    args.root,
                    target,
                    child_depths[target],
                    graph.parents(target),
                    distances(target),
                    distances,
                )
            )
        records.extend(bucket_records(records))
        summary = summarize(records, args.graph_name, args.root, selection_counts)
        return records, summary
    finally:
        graph.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", required=True, type=int, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_parent_branching", help="Graph label used in output filenames.")
    parser.add_argument("--child-depths", default=",".join(map(str, DEFAULT_DEPTHS)), help="Child-sampling depths to target.")
    parser.add_argument("--children-per-node", type=int, default=64, help="Deterministic sample cap for children per frontier node.")
    parser.add_argument("--frontier-limit", type=int, default=5000, help="Deterministic cap for each sampled child-depth frontier.")
    parser.add_argument("--targets-per-depth", type=int, default=200, help="Deterministic target cap per requested child depth.")
    parser.add_argument("--max-parent-depth", type=int, default=48, help="Maximum parent hops to search when computing L_min/L_max.")
    parser.add_argument("--seed", default="0", help="Deterministic sampling seed.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    args = parser.parse_args(argv)

    if args.children_per_node <= 0:
        raise SystemExit("--children-per-node must be positive")
    if args.frontier_limit <= 0:
        raise SystemExit("--frontier-limit must be positive")
    if args.targets_per_depth <= 0:
        raise SystemExit("--targets-per-depth must be positive")
    if args.max_parent_depth <= 0:
        raise SystemExit("--max-parent-depth must be positive")

    records, summary = run_lmdb_diagnostic(args)
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
