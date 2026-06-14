#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Profile parent branching inside a root-conditioned category subgraph.

This is a preprocessing/profile tool.  It walks the child graph from a chosen
root, retains the reached nodes, then compares raw parent degree against parent
degree after filtering parents to the retained common-root subgraph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_parent_branching_diagnostic import (
    LmdbCategoryGraph,
    size_biased_branching,
)
from scripts.lmdb_parent_histogram_benchmark import safe_graph_name


def mean(values):
    values = list(values)
    return 0.0 if not values else sum(values) / len(values)


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((float(pct) / 100.0) * (len(values) - 1))))
    return float(values[index])


def collect_root_conditioned_nodes(children_func, root, max_child_depth=0, max_nodes=0):
    """Return ``node -> min child-depth from root`` for the retained subgraph."""
    max_child_depth = int(max_child_depth or 0)
    max_nodes = int(max_nodes or 0)
    depth_by_node = {root: 0}
    queue = deque([root])
    child_edges_examined = 0
    truncated_by_depth = False
    truncated_by_nodes = False

    while queue:
        node = queue.popleft()
        depth = depth_by_node[node]
        if max_child_depth > 0 and depth >= max_child_depth:
            truncated_by_depth = True
            continue
        for child in children_func(node):
            child_edges_examined += 1
            if child in depth_by_node:
                continue
            if max_nodes > 0 and len(depth_by_node) >= max_nodes:
                truncated_by_nodes = True
                continue
            depth_by_node[child] = depth + 1
            queue.append(child)

    return {
        "depth_by_node": depth_by_node,
        "child_edges_examined": child_edges_examined,
        "truncated_by_depth": truncated_by_depth,
        "truncated_by_nodes": truncated_by_nodes,
    }


def parent_degree_record(node, depth, parents, retained_nodes):
    full_parent_degree = len(parents)
    root_conditioned_parent_degree = sum(1 for parent in parents if parent in retained_nodes)
    return {
        "record_type": "root_conditioned_branching_node",
        "node": node,
        "child_depth": int(depth),
        "full_parent_degree": full_parent_degree,
        "root_conditioned_parent_degree": root_conditioned_parent_degree,
        "outside_root_parent_degree": full_parent_degree - root_conditioned_parent_degree,
    }


def summarize_degrees(degrees):
    stats = size_biased_branching(degrees)
    stats.update({
        "zero_parent_nodes": sum(1 for degree in degrees if int(degree) <= 0),
        "p50_parent_degree": percentile(degrees, 50),
        "p99_parent_degree": percentile(degrees, 99),
    })
    return stats


def profile_rows(graph, root, graph_name, max_child_depth=0, max_nodes=0):
    started = time.perf_counter_ns()
    collection = collect_root_conditioned_nodes(
        graph.children,
        root,
        max_child_depth=max_child_depth,
        max_nodes=max_nodes,
    )
    collected_ns = time.perf_counter_ns() - started
    depth_by_node = collection["depth_by_node"]
    retained_nodes = set(depth_by_node)

    degree_started = time.perf_counter_ns()
    node_rows = [
        parent_degree_record(node, depth, graph.parents(node), retained_nodes)
        for node, depth in sorted(depth_by_node.items(), key=lambda item: (item[1], item[0]))
    ]
    degree_ns = time.perf_counter_ns() - degree_started

    rows = [{
        "record_type": "root_conditioned_branching_selection",
        "graph": graph_name,
        "root": root,
        "max_child_depth": int(max_child_depth or 0),
        "max_nodes": int(max_nodes or 0),
        "retained_nodes": len(retained_nodes),
        "max_observed_child_depth": max(depth_by_node.values()) if depth_by_node else None,
        "child_edges_examined": collection["child_edges_examined"],
        "truncated_by_depth": collection["truncated_by_depth"],
        "truncated_by_nodes": collection["truncated_by_nodes"],
        "collect_elapsed_ns": collected_ns,
        "degree_elapsed_ns": degree_ns,
        "total_elapsed_ns": collected_ns + degree_ns,
    }]
    rows.append(summary_row("root_conditioned_branching_overall", node_rows, graph_name, root, None))
    for depth in sorted(set(depth_by_node.values())):
        bucket = [row for row in node_rows if row["child_depth"] == depth]
        rows.append(summary_row("root_conditioned_branching_depth_bucket", bucket, graph_name, root, depth))
    rows.extend(node_rows)
    return rows


def summary_row(record_type, rows, graph_name, root, depth):
    full_degrees = [row["full_parent_degree"] for row in rows]
    conditioned_degrees = [row["root_conditioned_parent_degree"] for row in rows]
    outside_degrees = [row["outside_root_parent_degree"] for row in rows]
    out = {
        "record_type": record_type,
        "graph": graph_name,
        "root": root,
        "nodes": len(rows),
        "full_parent_degree": summarize_degrees(full_degrees),
        "root_conditioned_parent_degree": summarize_degrees(conditioned_degrees),
        "outside_root_parent_degree": summarize_degrees(outside_degrees),
        "mean_outside_parent_fraction": mean(
            0.0 if row["full_parent_degree"] <= 0 else row["outside_root_parent_degree"] / row["full_parent_degree"]
            for row in rows
        ),
    }
    if depth is not None:
        out["child_depth"] = int(depth)
    return out


def format_optional(value):
    return "n/a" if value is None else "{:.6f}".format(float(value))


def summarize_report(rows):
    selection = next(row for row in rows if row["record_type"] == "root_conditioned_branching_selection")
    overall = next(row for row in rows if row["record_type"] == "root_conditioned_branching_overall")
    buckets = [row for row in rows if row["record_type"] == "root_conditioned_branching_depth_bucket"]
    lines = [
        "# Root-Conditioned Parent Branching Profile",
        "",
        "Graph: `{}`".format(selection["graph"]),
        "",
        "Root: `{}`".format(selection["root"]),
        "",
        "## Selection",
        "",
        "| retained_nodes | max_observed_depth | child_edges_examined | truncated_by_depth | truncated_by_nodes | total_elapsed_ms |",
        "|---------------:|-------------------:|---------------------:|--------------------|--------------------|-----------------:|",
        "| {nodes} | {depth} | {edges} | {depth_trunc} | {node_trunc} | {ms:.3f} |".format(
            nodes=selection["retained_nodes"],
            depth=selection["max_observed_child_depth"],
            edges=selection["child_edges_examined"],
            depth_trunc="yes" if selection["truncated_by_depth"] else "no",
            node_trunc="yes" if selection["truncated_by_nodes"] else "no",
            ms=selection["total_elapsed_ns"] / 1_000_000.0,
        ),
        "",
        "## Overall Moments",
        "",
        "| degree_scope | nodes | mean_p | p95_p | p99_p | max_p | E[p^2]/E[p] | mean_excess | zero_parent_nodes |",
        "|--------------|------:|-------:|------:|------:|------:|-------------:|------------:|------------------:|",
    ]
    for label, key in [
        ("raw_full_graph", "full_parent_degree"),
        ("root_conditioned", "root_conditioned_parent_degree"),
        ("outside_root", "outside_root_parent_degree"),
    ]:
        stats = overall[key]
        lines.append(
            "| {label} | {nodes} | {mean:.3f} | {p95:.3f} | {p99:.3f} | {maxp} | {b} | {excess} | {zeros} |".format(
                label=label,
                nodes=stats["nodes"],
                mean=stats["mean_parent_degree"],
                p95=stats["p95_parent_degree"],
                p99=stats["p99_parent_degree"],
                maxp=stats["max_parent_degree"],
                b=format_optional(stats["size_biased_parent_branching"]),
                excess=format_optional(stats["mean_excess"]),
                zeros=stats["zero_parent_nodes"],
            )
        )
    lines.extend([
        "",
        "## Depth Buckets",
        "",
        "| child_depth | nodes | raw_b | root_conditioned_b | mean_raw_p | mean_root_p | mean_outside_fraction |",
        "|------------:|------:|------:|-------------------:|-----------:|------------:|----------------------:|",
    ])
    for row in buckets:
        full = row["full_parent_degree"]
        conditioned = row["root_conditioned_parent_degree"]
        lines.append(
            "| {depth} | {nodes} | {raw_b} | {root_b} | {raw_mean:.3f} | {root_mean:.3f} | {outside:.3f} |".format(
                depth=row["child_depth"],
                nodes=row["nodes"],
                raw_b=format_optional(full["size_biased_parent_branching"]),
                root_b=format_optional(conditioned["size_biased_parent_branching"]),
                raw_mean=full["mean_parent_degree"],
                root_mean=conditioned["mean_parent_degree"],
                outside=row["mean_outside_parent_fraction"],
            )
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `raw_full_graph` counts all parents of retained nodes.",
        "- `root_conditioned` counts only parents that are also retained descendants of the chosen root.",
        "- This profile is preprocessing evidence for the estimator; the query path should consume the resulting prior instead of discovering the root-conditioned subgraph online.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(rows, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / "lmdb_root_conditioned_branching_profile_{}_{}.jsonl".format(safe_name, timestamp)
    summary_path = output_dir / "lmdb_root_conditioned_branching_profile_summary_{}_{}.md".format(safe_name, timestamp)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path.write_text(summarize_report(rows), encoding="utf-8")
    return jsonl_path, summary_path


def run_profile(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        return profile_rows(
            graph,
            args.root,
            args.graph_name,
            max_child_depth=args.max_child_depth,
            max_nodes=args.max_nodes,
        )
    finally:
        graph.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", required=True, type=int, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_root_conditioned_branching", help="Graph label used in output filenames.")
    parser.add_argument("--max-child-depth", type=int, default=0, help="Maximum child depth to retain; 0 means no depth limit.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Maximum retained nodes; 0 means no node limit.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    args = parser.parse_args(argv)

    if args.max_child_depth < 0:
        raise SystemExit("--max-child-depth must be non-negative")
    if args.max_nodes < 0:
        raise SystemExit("--max-nodes must be non-negative")

    rows = run_profile(args)
    report = summarize_report(rows)
    if args.output_dir:
        jsonl_path, summary_path = write_outputs(rows, args.output_dir, args.graph_name)
        print(report, end="")
        print("jsonl={}".format(jsonl_path))
        print("summary={}".format(summary_path))
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
