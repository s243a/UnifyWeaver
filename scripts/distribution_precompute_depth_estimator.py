#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Estimate amortized precompute depth for root-anchored distributions.

This is a planning tool, not a correctness oracle.  It combines a depth prior
for expected query hits with a simple measured-unit cost model:

    expected_hits * saved_suffix_work_per_hit
        > build + fit + storage + amortized decode

The parent-branching prior uses b = E[p^2] / E[p].  If query traffic is spread
across each layer, per-node hits fall by about 1 / b per depth step.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_parent_histogram_benchmark import safe_graph_name


def mean(values):
    values = list(values)
    return 0.0 if not values else statistics.fmean(values)


def parse_depth_float_map(text):
    """Parse depth:value pairs such as '0:1,1:1.5,2:4'."""
    out = {}
    if not text:
        return out
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("expected depth:value pair, got {!r}".format(part))
        depth_text, value_text = part.split(":", 1)
        out[int(depth_text.strip())] = float(value_text.strip())
    return out


def branching_at_depth(depth, default_branching, depth_branching):
    if depth in depth_branching:
        return max(1.0, float(depth_branching[depth]))
    return max(1.0, float(default_branching))


def cumulative_branching(depth, default_branching, depth_branching):
    product = 1.0
    for step in range(1, int(depth) + 1):
        product *= branching_at_depth(step, default_branching, depth_branching)
    return product


def estimated_path_states(depth, default_branching, depth_branching, minimum=1.0):
    if depth <= 0:
        return float(minimum)
    return max(float(minimum), cumulative_branching(depth, default_branching, depth_branching))


@dataclass(frozen=True)
class Representation:
    name: str
    points: int
    bytes_per_point: float
    fixed_bytes: float
    fit_cost_per_point: float

    @property
    def bytes_estimate(self):
        return self.fixed_bytes + self.points * self.bytes_per_point


def default_representations(depth, args):
    exact_bins = min(max(1, int(depth) + 1), max(1, int(args.exact_max_bins)))
    return [
        Representation(
            "exact_sparse_histogram",
            exact_bins,
            args.exact_bytes_per_bin,
            0.0,
            0.0,
        ),
        Representation(
            "sampled_{}_point_distribution".format(args.sample_points),
            max(1, int(args.sample_points)),
            args.sample_bytes_per_point,
            0.0,
            args.sample_fit_cost_per_point,
        ),
        Representation(
            "parametric_closed_form",
            max(1, int(args.parametric_points)),
            0.0,
            args.parametric_bytes,
            args.parametric_fit_cost,
        ),
    ]


def estimate_row(depth, representation, args, depth_branching, query_reach_prob):
    cumulative_b = cumulative_branching(depth, args.branching_factor, depth_branching)
    expected_hits = args.expected_queries * query_reach_prob / max(cumulative_b, 1.0)
    path_states = estimated_path_states(depth, args.branching_factor, depth_branching, args.min_path_states)

    uncached_suffix_cost = args.uncached_base_cost + path_states * args.uncached_cost_per_state
    cached_suffix_eval_cost = (
        args.cached_eval_base_cost
        + representation.points * args.cached_eval_cost_per_point
    )
    saved_per_hit = max(0.0, uncached_suffix_cost - cached_suffix_eval_cost)

    build_cost = args.build_base_cost + path_states * args.build_cost_per_state
    fit_cost = representation.points * representation.fit_cost_per_point
    storage_cost = representation.bytes_estimate * args.storage_cost_per_byte
    amortized_decode_cost = representation.bytes_estimate * args.decode_cost_per_byte
    one_time_cost = build_cost + fit_cost + storage_cost + amortized_decode_cost
    hits_to_break_even = math.inf if saved_per_hit <= 0.0 else one_time_cost / saved_per_hit
    net_value = expected_hits * saved_per_hit - one_time_cost

    return {
        "record_type": "distribution_precompute_depth_estimate",
        "depth": int(depth),
        "representation": representation.name,
        "points": int(representation.points),
        "bytes_estimate": representation.bytes_estimate,
        "branching_factor": branching_at_depth(depth, args.branching_factor, depth_branching),
        "cumulative_branching": cumulative_b,
        "query_reach_probability": query_reach_prob,
        "expected_hits": expected_hits,
        "expected_path_states": path_states,
        "uncached_suffix_cost": uncached_suffix_cost,
        "cached_suffix_eval_cost": cached_suffix_eval_cost,
        "saved_per_hit": saved_per_hit,
        "build_cost": build_cost,
        "fit_cost": fit_cost,
        "storage_cost": storage_cost,
        "amortized_decode_cost": amortized_decode_cost,
        "one_time_cost": one_time_cost,
        "hits_to_break_even": None if math.isinf(hits_to_break_even) else hits_to_break_even,
        "expected_net_value": net_value,
        "precompute_pays": net_value > 0.0,
    }


def build_records(args):
    depth_branching = parse_depth_float_map(args.depth_branching)
    query_reach = parse_depth_float_map(args.query_reach_probability)
    records = [{
        "record_type": "distribution_precompute_depth_estimator_selection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph": args.graph_name,
        "expected_queries": args.expected_queries,
        "branching_factor": args.branching_factor,
        "depth_branching": depth_branching,
        "query_reach_probability": query_reach,
        "max_depth": args.max_depth,
        "cost_model": {
            "uncached_base_cost": args.uncached_base_cost,
            "uncached_cost_per_state": args.uncached_cost_per_state,
            "cached_eval_base_cost": args.cached_eval_base_cost,
            "cached_eval_cost_per_point": args.cached_eval_cost_per_point,
            "build_base_cost": args.build_base_cost,
            "build_cost_per_state": args.build_cost_per_state,
            "storage_cost_per_byte": args.storage_cost_per_byte,
            "decode_cost_per_byte": args.decode_cost_per_byte,
            "sample_fit_cost_per_point": args.sample_fit_cost_per_point,
            "parametric_fit_cost": args.parametric_fit_cost,
        },
    }]
    for depth in range(0, int(args.max_depth) + 1):
        reach_probability = query_reach.get(depth, args.default_query_reach_probability)
        for representation in default_representations(depth, args):
            records.append(estimate_row(depth, representation, args, depth_branching, reach_probability))
    return records


def summarize(records):
    selection = next(row for row in records if row["record_type"] == "distribution_precompute_depth_estimator_selection")
    estimate_rows = [row for row in records if row["record_type"] == "distribution_precompute_depth_estimate"]
    by_depth = {}
    for row in estimate_rows:
        by_depth.setdefault(row["depth"], []).append(row)

    lines = [
        "# Distribution Precompute Depth Estimator",
        "",
        "Graph: `{}`".format(selection["graph"]),
        "",
        "Expected queries: `{:.3f}`".format(selection["expected_queries"]),
        "",
        "Default parent branching prior `b = E[p^2] / E[p]`: `{:.6f}`".format(selection["branching_factor"]),
        "",
        "## Depth Recommendation",
        "",
        "| depth | expected_hits | path_states | best_representation | hits_to_break_even | net_value | pays |",
        "|------:|--------------:|------------:|--------------------|-------------------:|----------:|------|",
    ]
    recommendation_rows = []
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        best = max(rows, key=lambda item: item["expected_net_value"])
        recommendation_rows.append(best)
        lines.append(
            "| {depth} | {hits:.3f} | {states:.3f} | {rep} | {breakeven} | {net:.3f} | {pays} |".format(
                depth=depth,
                hits=best["expected_hits"],
                states=best["expected_path_states"],
                rep=best["representation"],
                breakeven="n/a" if best["hits_to_break_even"] is None else "{:.3f}".format(best["hits_to_break_even"]),
                net=best["expected_net_value"],
                pays="yes" if best["precompute_pays"] else "no",
            )
        )

    lines.extend([
        "",
        "## Representation Detail",
        "",
        "| depth | representation | points | bytes | expected_hits | saved_per_hit | one_time_cost | hits_to_break_even | net_value | pays |",
        "|------:|----------------|-------:|------:|--------------:|--------------:|--------------:|-------------------:|----------:|------|",
    ])
    for row in estimate_rows:
        lines.append(
            "| {depth} | {rep} | {points} | {bytes:.1f} | {hits:.3f} | {saved:.3f} | {cost:.3f} | {breakeven} | {net:.3f} | {pays} |".format(
                depth=row["depth"],
                rep=row["representation"],
                points=row["points"],
                bytes=row["bytes_estimate"],
                hits=row["expected_hits"],
                saved=row["saved_per_hit"],
                cost=row["one_time_cost"],
                breakeven="n/a" if row["hits_to_break_even"] is None else "{:.3f}".format(row["hits_to_break_even"]),
                net=row["expected_net_value"],
                pays="yes" if row["precompute_pays"] else "no",
            )
        )

    paying_depths = [row["depth"] for row in recommendation_rows if row["precompute_pays"]]
    lines.extend([
        "",
        "## Summary",
        "",
        "- deepest recommended depth: `{}`".format(max(paying_depths) if paying_depths else "none"),
        "- mean best net value: `{:.3f}`".format(mean(row["expected_net_value"] for row in recommendation_rows)),
        "- note: point count is a representation cost input, not the break-even hit count.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    summary_json = output_dir / "{}_distribution_precompute_depth_estimator_summary.json".format(safe_name)
    summary_md = output_dir / "{}_distribution_precompute_depth_estimator_summary.md".format(safe_name)
    summary_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(summary, encoding="utf-8")
    return summary_json, summary_md


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-name", default="distribution_precompute_depth_estimator")
    parser.add_argument("--expected-queries", type=float, default=1000.0)
    parser.add_argument("--branching-factor", type=float, default=4.0, help="Default b = E[p^2] / E[p].")
    parser.add_argument("--depth-branching", default="", help="Optional depth-specific b values, e.g. 1:1,2:1.67,3:4.25.")
    parser.add_argument("--query-reach-probability", default="", help="Optional depth-specific query reach probabilities.")
    parser.add_argument("--default-query-reach-probability", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-path-states", type=float, default=1.0)
    parser.add_argument("--uncached-base-cost", type=float, default=0.0)
    parser.add_argument("--uncached-cost-per-state", type=float, default=1.0)
    parser.add_argument("--cached-eval-base-cost", type=float, default=1.0)
    parser.add_argument("--cached-eval-cost-per-point", type=float, default=0.02)
    parser.add_argument("--build-base-cost", type=float, default=0.0)
    parser.add_argument("--build-cost-per-state", type=float, default=1.0)
    parser.add_argument("--storage-cost-per-byte", type=float, default=0.01)
    parser.add_argument("--decode-cost-per-byte", type=float, default=0.02)
    parser.add_argument("--exact-max-bins", type=int, default=50)
    parser.add_argument("--exact-bytes-per-bin", type=float, default=16.0)
    parser.add_argument("--sample-points", type=int, default=50)
    parser.add_argument("--sample-bytes-per-point", type=float, default=8.0)
    parser.add_argument("--sample-fit-cost-per-point", type=float, default=1.0)
    parser.add_argument("--parametric-points", type=int, default=4)
    parser.add_argument("--parametric-bytes", type=float, default=64.0)
    parser.add_argument("--parametric-fit-cost", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = build_records(args)
    summary = summarize(records)
    summary_json, summary_md = write_outputs(records, summary, args.output_dir, args.graph_name)
    print(summary, end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
