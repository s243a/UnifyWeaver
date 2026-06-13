#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Benchmark one parent-payload recurrence layer against recursive recurrence."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distribution_serialization import encode_selected_distribution
from scripts.lmdb_parent_branching_diagnostic import LmdbCategoryGraph, parse_int_list, select_targets_by_child_depth
from scripts.lmdb_parent_histogram_benchmark import percentile, safe_graph_name
from scripts.lmdb_parent_recurrence_histogram_benchmark import histogram_error
from scripts.parent_histogram_recurrence import (
    histogram_distribution,
    recurrence_parent_histogram,
    serialize_shifted_parent_payload_histogram,
)


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def serialize_histogram_payload(hist, representation="packed_sparse_histogram", cdf_bits=16):
    probabilities, origin, total_count = histogram_distribution(hist)
    return encode_selected_distribution(
        probabilities,
        representation,
        origin=origin,
        total_mass=total_count,
        cdf_bits=cdf_bits,
    )


def build_parent_payloads(
    parents_func,
    root,
    parent_nodes,
    parent_budget,
    path_cap,
    expansion_cap,
    representation="packed_sparse_histogram",
):
    payloads = {}
    rows = []
    for node in sorted(parent_nodes):
        started = time.perf_counter_ns()
        hist, stats = recurrence_parent_histogram(parents_func, node, root, parent_budget, path_cap, expansion_cap)
        recurrence_time_ns = time.perf_counter_ns() - started
        payload, metadata = serialize_histogram_payload(hist, representation)
        payloads[node] = payload
        rows.append({
            "record_type": "payload_recurrence_parent_payload",
            "node": node,
            "parent_budget": parent_budget,
            "histogram": hist,
            "path_count": sum(hist.values()),
            "support_bins": len(hist),
            "payload_representation": metadata["representation"],
            "payload_bytes": metadata["payload_bytes"],
            "payload_decoded_max_cdf_error": metadata["decoded_max_cdf_error"],
            "payload_decoded_w1_cdf_error": metadata["decoded_w1_cdf_error"],
            "recurrence_states_evaluated": stats.states_evaluated,
            "recurrence_edges_examined": stats.edges_examined,
            "recurrence_cycle_approximation": stats.cycle_approximation,
            "recurrence_time_ns": recurrence_time_ns,
        })
    return payloads, rows


def payload_layer_records(
    parents_func,
    root,
    child_nodes,
    child_depth_by_node,
    parent_payloads,
    budgets,
    path_cap,
    expansion_cap,
    representation="packed_sparse_histogram",
):
    rows = []
    for child in child_nodes:
        direct_parents = list(parents_func(child))
        available_parent_nodes = [parent for parent in direct_parents if parent in parent_payloads]
        missing_parent_count = len(direct_parents) - len(available_parent_nodes)
        available_payloads = [parent_payloads[parent] for parent in available_parent_nodes]
        for budget in budgets:
            rec_started = time.perf_counter_ns()
            recurrence_hist, recurrence_stats = recurrence_parent_histogram(
                parents_func,
                child,
                root,
                budget,
                path_cap,
                expansion_cap,
            )
            recurrence_time_ns = time.perf_counter_ns() - rec_started

            payload_started = time.perf_counter_ns()
            child_payload, child_metadata, payload_hist, payload_stats = serialize_shifted_parent_payload_histogram(
                available_payloads,
                representation=representation,
                remaining=budget,
                path_cap=path_cap,
            )
            payload_time_ns = time.perf_counter_ns() - payload_started

            l1, cdf, w1 = histogram_error(recurrence_hist, payload_hist)
            rows.append({
                "record_type": "payload_recurrence_layer_comparison",
                "target_node": child,
                "child_sample_depth": child_depth_by_node[child],
                "budget": budget,
                "direct_parent_count": len(direct_parents),
                "parent_payloads_available": len(available_payloads),
                "missing_parent_payloads": missing_parent_count,
                "recurrence_histogram": recurrence_hist,
                "payload_histogram": payload_hist,
                "recurrence_path_count": sum(recurrence_hist.values()),
                "payload_path_count": sum(payload_hist.values()),
                "path_count_delta": sum(payload_hist.values()) - sum(recurrence_hist.values()),
                "l1_error": l1,
                "max_cdf_error": cdf,
                "w1_cdf_error": w1,
                "exact_match": recurrence_hist == payload_hist,
                "recurrence_states_evaluated": recurrence_stats.states_evaluated,
                "recurrence_edges_examined": recurrence_stats.edges_examined,
                "recurrence_cycle_approximation": recurrence_stats.cycle_approximation,
                "recurrence_path_cap_hit": recurrence_stats.path_cap_hit,
                "recurrence_expansion_cap_hit": recurrence_stats.expansion_cap_hit,
                "payloads_decoded": payload_stats.payloads_decoded,
                "payload_bytes_read": payload_stats.payload_bytes_read,
                "payload_decode_ns": payload_stats.decode_ns,
                "payload_decoded_bins": payload_stats.decoded_bins,
                "payload_output_bins": payload_stats.output_bins,
                "payload_output_path_count": payload_stats.output_path_count,
                "payload_output_bytes": payload_stats.output_payload_bytes,
                "payload_output_representation": child_metadata["representation"],
                "payload_path_cap_hit": payload_stats.path_cap_hit,
                "recurrence_time_ns": recurrence_time_ns,
                "payload_time_ns": payload_time_ns,
                "time_ratio": 0.0 if recurrence_time_ns == 0 else payload_time_ns / recurrence_time_ns,
                "child_payload_bytes": len(child_payload),
            })
    return rows


def summarize(records):
    selection = next((row for row in records if row.get("record_type") == "payload_recurrence_layer_selection"), {})
    parent_rows = [row for row in records if row.get("record_type") == "payload_recurrence_parent_payload"]
    comparison_rows = [row for row in records if row.get("record_type") == "payload_recurrence_layer_comparison"]
    by_budget = {}
    for row in comparison_rows:
        by_budget.setdefault(row["budget"], []).append(row)
    return {
        "record_type": "payload_recurrence_layer_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph": selection.get("graph"),
        "root": selection.get("root"),
        "parent_depths": selection.get("parent_depths", []),
        "child_depths": selection.get("child_depths", []),
        "parent_nodes": len(parent_rows),
        "child_rows": len(comparison_rows),
        "mean_parent_payload_bytes": mean(row["payload_bytes"] for row in parent_rows),
        "mean_parent_payload_bins": mean(row["support_bins"] for row in parent_rows),
        "budget_rows": [
            {
                "budget": budget,
                "rows": len(rows),
                "exact_match_rows": sum(1 for row in rows if row["exact_match"]),
                "missing_parent_rows": sum(1 for row in rows if row["missing_parent_payloads"]),
                "mean_missing_parent_payloads": mean(row["missing_parent_payloads"] for row in rows),
                "mean_parent_payloads_available": mean(row["parent_payloads_available"] for row in rows),
                "mean_l1_error": mean(row["l1_error"] for row in rows),
                "p95_l1_error": percentile([row["l1_error"] for row in rows], 95),
                "mean_max_cdf_error": mean(row["max_cdf_error"] for row in rows),
                "mean_w1_cdf_error": mean(row["w1_cdf_error"] for row in rows),
                "mean_payload_bytes_read": mean(row["payload_bytes_read"] for row in rows),
                "mean_payload_decode_ns": mean(row["payload_decode_ns"] for row in rows),
                "mean_payload_decoded_bins": mean(row["payload_decoded_bins"] for row in rows),
                "mean_payload_output_bins": mean(row["payload_output_bins"] for row in rows),
                "mean_payload_output_bytes": mean(row["payload_output_bytes"] for row in rows),
                "mean_recurrence_time_ns": mean(row["recurrence_time_ns"] for row in rows),
                "mean_payload_time_ns": mean(row["payload_time_ns"] for row in rows),
                "mean_time_ratio": mean(row["time_ratio"] for row in rows),
                "recurrence_capped_rows": sum(1 for row in rows if row["recurrence_path_cap_hit"] or row["recurrence_expansion_cap_hit"]),
                "payload_capped_rows": sum(1 for row in rows if row["payload_path_cap_hit"]),
            }
            for budget, rows in sorted(by_budget.items())
        ],
    }


def markdown_summary(summary):
    lines = [
        "# Payload Recurrence Layer Benchmark",
        "",
        "| budget | rows | exact_matches | missing_parent_rows | mean_missing | mean_parents_used | mean_l1 | mean_cdf | mean_payload_read | mean_decode_ns | mean_output_bins | mean_output_bytes | mean_time_ratio | recurrence_capped | payload_capped |",
        "|-------:|-----:|--------------:|--------------------:|-------------:|------------------:|--------:|---------:|------------------:|---------------:|-----------------:|------------------:|----------------:|------------------:|---------------:|",
    ]
    for row in summary["budget_rows"]:
        lines.append(
            "| {budget} | {rows} | {exact} | {missing_rows} | {missing:.3f} | {parents:.3f} | {l1:.6f} | {cdf:.6f} | {payload_read:.3f} | {decode_ns:.1f} | {out_bins:.3f} | {out_bytes:.3f} | {time_ratio:.3f} | {rec_capped} | {payload_capped} |".format(
                budget=row["budget"],
                rows=row["rows"],
                exact=row["exact_match_rows"],
                missing_rows=row["missing_parent_rows"],
                missing=row["mean_missing_parent_payloads"],
                parents=row["mean_parent_payloads_available"],
                l1=row["mean_l1_error"],
                cdf=row["mean_max_cdf_error"],
                payload_read=row["mean_payload_bytes_read"],
                decode_ns=row["mean_payload_decode_ns"],
                out_bins=row["mean_payload_output_bins"],
                out_bytes=row["mean_payload_output_bytes"],
                time_ratio=row["mean_time_ratio"],
                rec_capped=row["recurrence_capped_rows"],
                payload_capped=row["payload_capped_rows"],
            )
        )
    return "\n".join(lines) + "\n"


def run_benchmark(args):
    graph = LmdbCategoryGraph(args.lmdb_dir)
    try:
        budgets = parse_int_list(args.budgets)
        parent_budget = max(0, int(args.parent_budget if args.parent_budget is not None else max(budgets, default=0) - 1))
        parent_nodes, _parent_depth_by_node, parent_counts = select_targets_by_child_depth(
            graph,
            args.root,
            parse_int_list(args.parent_depths),
            args.children_per_node,
            args.frontier_limit,
            args.parents_per_depth,
            args.seed + ":parents",
        )
        child_nodes, child_depth_by_node, child_counts = select_targets_by_child_depth(
            graph,
            args.root,
            parse_int_list(args.child_depths),
            args.children_per_node,
            args.frontier_limit,
            args.children_per_depth,
            args.seed + ":children",
        )
        selected_parent_nodes = set(parent_nodes)
        child_direct_parent_nodes = {
            parent
            for child in child_nodes
            for parent in graph.parents(child)
        }
        parent_nodes = sorted(selected_parent_nodes | child_direct_parent_nodes) if args.include_child_direct_parents else sorted(selected_parent_nodes)
        records = [{
            "record_type": "payload_recurrence_layer_selection",
            "graph": args.graph_name,
            "root": args.root,
            "parent_depths": parse_int_list(args.parent_depths),
            "child_depths": parse_int_list(args.child_depths),
            "parent_selection_counts": parent_counts,
            "child_selection_counts": child_counts,
            "parents": len(parent_nodes),
            "children": len(child_nodes),
            "parent_budget": parent_budget,
            "include_child_direct_parents": args.include_child_direct_parents,
            "selected_parent_nodes": len(selected_parent_nodes),
            "child_direct_parent_nodes_added": len(set(parent_nodes) - selected_parent_nodes),
            "budgets": budgets,
            "representation": args.representation,
        }]
        parent_payloads, parent_rows = build_parent_payloads(
            graph.parents,
            args.root,
            parent_nodes,
            parent_budget,
            args.path_cap,
            args.expansion_cap,
            args.representation,
        )
        records.extend(parent_rows)
        records.extend(payload_layer_records(
            graph.parents,
            args.root,
            child_nodes,
            child_depth_by_node,
            parent_payloads,
            budgets,
            args.path_cap,
            args.expansion_cap,
            args.representation,
        ))
        return records, summarize(records)
    finally:
        graph.close()


def write_outputs(records, summary, output_dir, graph_name, write_jsonl=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "{}_payload_recurrence_layer_summary.json".format(safe_name)
    summary_md = output_dir / "{}_payload_recurrence_layer_summary.md".format(safe_name)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_summary(summary), encoding="utf-8")
    jsonl_path = None
    if write_jsonl:
        jsonl_path = output_dir / "{}_payload_recurrence_layer.jsonl".format(safe_name)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return summary_json, summary_md, jsonl_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", type=Path, required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--graph-name", default="lmdb_payload_recurrence_layer")
    parser.add_argument("--parent-depths", default="0,1")
    parser.add_argument("--child-depths", default="2")
    parser.add_argument("--children-per-node", type=int, default=32)
    parser.add_argument("--frontier-limit", type=int, default=200)
    parser.add_argument("--parents-per-depth", type=int, default=64)
    parser.add_argument("--children-per-depth", type=int, default=16)
    parser.add_argument("--parent-budget", type=int, default=None)
    parser.add_argument("--budgets", default="3,4")
    parser.add_argument("--no-include-child-direct-parents", dest="include_child_direct_parents", action="store_false", help="Use only sampled parent-depth nodes; by default direct parents of sampled children are added to the payload set.")
    parser.add_argument("--representation", choices=["packed_sparse_histogram", "quantized_cdf_table"], default="packed_sparse_histogram")
    parser.add_argument("--path-cap", type=int, default=10000)
    parser.add_argument("--expansion-cap", type=int, default=50000)
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--seed", default="payload-recurrence-layer-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    records, summary = run_benchmark(args)
    summary_json, summary_md, jsonl_path = write_outputs(records, summary, args.output_dir, args.graph_name, args.write_jsonl)
    print(markdown_summary(summary), end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    if jsonl_path is not None:
        print("jsonl={}".format(jsonl_path))


if __name__ == "__main__":
    main()
