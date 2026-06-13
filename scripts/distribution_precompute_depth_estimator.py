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
import re
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


def positive_mean(values):
    return mean(value for value in values if value is not None and value > 0.0)


def safe_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return numerator / denominator


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
    return cumulative_branching_between(0, depth, default_branching, depth_branching)


def cumulative_branching_between(start_depth, end_depth, default_branching, depth_branching):
    product = 1.0
    for step in range(int(start_depth) + 1, int(end_depth) + 1):
        product *= branching_at_depth(step, default_branching, depth_branching)
    return product


def estimated_path_states(depth, default_branching, depth_branching, minimum=1.0):
    if depth <= 0:
        return float(minimum)
    return max(float(minimum), cumulative_branching(depth, default_branching, depth_branching))


def estimated_suffix_states(boundary_depth, target_depth, default_branching, depth_branching, minimum=1.0):
    if int(target_depth) <= int(boundary_depth):
        return float(minimum)
    return max(float(minimum), cumulative_branching_between(boundary_depth, target_depth, default_branching, depth_branching))


def load_payload_recurrence_summaries(paths):
    summaries = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    return summaries


def load_jsonl_records(paths):
    records = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def parse_boundary_depth_from_graph(graph):
    match = re.search(r"_b([0-9]+)(?:_|$)", str(graph))
    return None if match is None else int(match.group(1))


def max_int_key(mapping):
    if not mapping:
        return None
    return max(int(key) for key in mapping)


def validation_measurements_from_records(records):
    """Aggregate boundary-cache benchmark JSONL rows by boundary depth.

    Zero-hit rows are treated as validation-shape warnings, not economic
    evidence.  In the intended ancestor-tree search, a target that shares the
    root and recurses only through ancestors should hit a selected ancestor
    boundary.  Measured saved-per-hit calibration therefore uses positive-hit
    rows only and marks all-zero depths unusable for validation capping.
    """
    selections_by_graph = {}
    comparisons_by_graph = {}
    for record in records:
        graph = record.get("graph")
        if record.get("record_type") == "boundary_cache_selection":
            selections_by_graph[graph] = record
        elif record.get("record_type") == "boundary_cache_comparison":
            comparisons_by_graph.setdefault(graph, []).append(record)

    measurements = {}
    for graph, rows in sorted(comparisons_by_graph.items()):
        selection = selections_by_graph.get(graph, {})
        boundary_depth = max_int_key(selection.get("boundary_counts")) if selection else None
        if boundary_depth is None:
            boundary_depth = parse_boundary_depth_from_graph(graph)
        if boundary_depth is None:
            continue

        target_depth = max_int_key(selection.get("target_counts")) if selection else None
        positive_rows = [
            row for row in rows
            if float(row.get("cache_hits", row.get("histogram_cache_hits", 0.0))) > 0.0
        ]
        calibration_rows = positive_rows
        calibration_usable = bool(calibration_rows)
        full_times = [float(row.get("full_time_ns", 0.0)) for row in calibration_rows]
        cached_times = [float(row.get("cached_time_ns", 0.0)) for row in calibration_rows]
        cache_hits = [float(row.get("cache_hits", row.get("histogram_cache_hits", 0.0))) for row in calibration_rows]
        all_cache_hits = [float(row.get("cache_hits", row.get("histogram_cache_hits", 0.0))) for row in rows]
        mean_full_time = mean(full_times)
        mean_cached_time = mean(cached_times)
        mean_cache_hits = mean(cache_hits)
        raw_saved_per_hit = None
        if mean_cache_hits > 0.0:
            raw_saved_per_hit = (mean_full_time - mean_cached_time) / mean_cache_hits
        clipped_saved_per_hit = max(0.0, raw_saved_per_hit or 0.0)
        time_ratios = [
            ratio for ratio in (
                safe_ratio(row.get("cached_time_ns", 0.0), row.get("full_time_ns", 0.0))
                for row in calibration_rows
            )
            if ratio is not None
        ]
        all_time_ratios = [
            ratio for ratio in (
                safe_ratio(row.get("cached_time_ns", 0.0), row.get("full_time_ns", 0.0))
                for row in rows
            )
            if ratio is not None
        ]

        measurements[int(boundary_depth)] = {
            "record_type": "distribution_precompute_validation_measurement",
            "graph": graph,
            "boundary_depth": int(boundary_depth),
            "target_depth": target_depth,
            "rows": len(rows),
            "boundary_nodes": selection.get("boundary_nodes"),
            "cached_boundary_nodes": selection.get("cached_boundary_nodes"),
            "selected_boundary_nodes": selection.get("selected_boundary_nodes"),
            "target_ancestor_boundary_nodes_added": selection.get("target_ancestor_boundary_nodes_added"),
            "mean_full_time_ns": mean_full_time,
            "mean_cached_time_ns": mean_cached_time,
            "mean_time_ratio": mean(time_ratios),
            "all_rows_mean_time_ratio": mean(all_time_ratios),
            "mean_full_nodes_expanded": mean(row.get("full_nodes_expanded", 0.0) for row in calibration_rows),
            "mean_cached_nodes_expanded": mean(row.get("cached_nodes_expanded", 0.0) for row in calibration_rows),
            "mean_cache_hits": mean_cache_hits,
            "all_rows_mean_cache_hits": mean(all_cache_hits),
            "positive_cache_hit_rows": len(positive_rows),
            "zero_cache_hit_rows": len(rows) - len(positive_rows),
            "validation_usable_for_cap": calibration_usable,
            "mean_histogram_cache_hits": mean(row.get("histogram_cache_hits", 0.0) for row in calibration_rows),
            "mean_cache_bins_spliced": mean(row.get("cache_bins_spliced", 0.0) for row in calibration_rows),
            "mean_payload_bytes_read": mean(row.get("cache_payload_bytes_read", 0.0) for row in calibration_rows),
            "mean_decode_ns": mean(row.get("cache_decode_ns", 0.0) for row in calibration_rows),
            "full_capped_rows": sum(1 for row in calibration_rows if row.get("full_expansion_cap_hit")),
            "cached_capped_rows": sum(1 for row in calibration_rows if row.get("cached_expansion_cap_hit")),
            "measured_pays": calibration_usable and mean_cached_time < mean_full_time,
            "measured_saved_per_hit_ns": raw_saved_per_hit,
            "clipped_saved_per_hit_ns": clipped_saved_per_hit,
        }
    return measurements


def validation_measurements_from_jsonl(paths):
    if not paths:
        return {}
    return validation_measurements_from_records(load_jsonl_records(paths))


def calibration_from_payload_recurrence_summaries(paths, args):
    """Derive cost constants from payload recurrence summary JSON files."""
    summaries = load_payload_recurrence_summaries(paths)
    recurrence_costs = []
    cached_eval_costs = []
    decode_costs = []
    build_costs = []
    reuse_ratios = []
    source_rows = 0
    depth_branching = parse_depth_float_map(args.depth_branching)

    for summary in summaries:
        parent_depths = summary.get("parent_depths", [])
        child_depths = summary.get("child_depths", [])
        parent_depth = max(parent_depths) if parent_depths else 0
        child_depth = max(child_depths) if child_depths else parent_depth
        parent_states = estimated_path_states(parent_depth, args.branching_factor, depth_branching, args.min_path_states)
        child_states = estimated_path_states(child_depth, args.branching_factor, depth_branching, args.min_path_states)
        # Parent payload build timing is not persisted in these summaries, so
        # use child recurrence timing as the first build-cost proxy.
        build_states = max(parent_states, 1.0)

        for row in summary.get("budget_rows", []):
            source_rows += 1
            recurrence_ratio = safe_ratio(row.get("mean_recurrence_time_ns", 0.0), child_states)
            if recurrence_ratio is not None:
                recurrence_costs.append(recurrence_ratio)
                build_costs.append(safe_ratio(row.get("mean_recurrence_time_ns", 0.0), build_states))

            if row.get("mean_payload_bytes_read", 0.0) > 0.0 and row.get("mean_payload_decode_ns", 0.0) > 0.0:
                decode_ratio = safe_ratio(row["mean_payload_decode_ns"], row["mean_payload_bytes_read"])
                if decode_ratio is not None:
                    decode_costs.append(decode_ratio)

            payload_bins = max(1.0, float(row.get("mean_payload_output_bins", 0.0)))
            payload_time = float(row.get("mean_payload_time_ns", 0.0))
            decode_time = float(row.get("mean_payload_decode_ns", 0.0))
            decode_free_payload_time = max(0.0, payload_time - decode_time)
            cached_ratio = safe_ratio(decode_free_payload_time, payload_bins)
            if cached_ratio is not None:
                cached_eval_costs.append(cached_ratio)

            total_refs = row.get("total_parent_payload_references")
            unique_refs = row.get("unique_parent_payloads_referenced")
            if total_refs and unique_refs:
                reuse_ratios.append(safe_ratio(total_refs, unique_refs))

    return {
        "source_paths": [str(path) for path in paths],
        "source_summaries": len(summaries),
        "source_budget_rows": source_rows,
        "uncached_cost_per_state": positive_mean(recurrence_costs),
        "build_cost_per_state": positive_mean(build_costs),
        "cached_eval_cost_per_point": positive_mean(cached_eval_costs),
        "decode_cost_per_byte": positive_mean(decode_costs),
        "mean_parent_reference_reuse": positive_mean(reuse_ratios),
    }


def apply_calibration(args):
    if not args.calibration_summary:
        return None
    calibration = calibration_from_payload_recurrence_summaries(args.calibration_summary, args)
    if calibration["uncached_cost_per_state"] > 0.0:
        args.uncached_cost_per_state = calibration["uncached_cost_per_state"]
    if calibration["build_cost_per_state"] > 0.0:
        args.build_cost_per_state = calibration["build_cost_per_state"]
    if calibration["cached_eval_cost_per_point"] > 0.0:
        args.cached_eval_cost_per_point = calibration["cached_eval_cost_per_point"]
    if calibration["decode_cost_per_byte"] > 0.0:
        args.decode_cost_per_byte = calibration["decode_cost_per_byte"]
    return calibration


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
    sampled_points = min(max(1, int(args.sample_points)), exact_bins)
    return [
        Representation(
            "exact_sparse_histogram",
            exact_bins,
            args.exact_bytes_per_bin,
            0.0,
            0.0,
        ),
        Representation(
            "sampled_up_to_{}_point_distribution".format(args.sample_points),
            sampled_points,
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


def estimate_row(boundary_depth, target_depth, representation, args, depth_branching, query_reach_prob, validation_measurement=None):
    cumulative_b = cumulative_branching(boundary_depth, args.branching_factor, depth_branching)
    expected_hits = args.expected_queries * query_reach_prob / max(cumulative_b, 1.0)
    build_states = estimated_path_states(boundary_depth, args.branching_factor, depth_branching, args.min_path_states)
    suffix_hops = max(0, int(target_depth) - int(boundary_depth))
    suffix_states = estimated_suffix_states(boundary_depth, target_depth, args.branching_factor, depth_branching, args.min_path_states)

    uncapped_suffix_work = suffix_states
    cap_limited_suffix_work = uncapped_suffix_work
    cap_ceiling = None
    if args.cap_mode == "measured":
        cap_ceiling = max(0.0, float(args.estimated_full_work))
    elif args.cap_mode == "path":
        cap_ceiling = max(0.0, float(args.path_cap))
    elif args.cap_mode == "expansion":
        cap_ceiling = max(0.0, float(args.expansion_cap))
    elif args.cap_mode not in {"uncapped", "validation"}:
        raise ValueError("unknown cap mode: {}".format(args.cap_mode))
    if cap_ceiling is not None and cap_ceiling > 0.0:
        cap_limited_suffix_work = min(uncapped_suffix_work, cap_ceiling)

    uncached_suffix_cost = args.uncached_base_cost + cap_limited_suffix_work * args.uncached_cost_per_state
    per_hit_decode_cost = 0.0 if args.decode_memoized else representation.bytes_estimate * args.decode_cost_per_byte
    splice_cost = representation.points * args.splice_cost_per_point
    cached_suffix_eval_cost = (
        args.cached_eval_base_cost
        + representation.points * args.cached_eval_cost_per_point
        + per_hit_decode_cost
        + splice_cost
    )
    saved_per_hit = max(0.0, uncached_suffix_cost - cached_suffix_eval_cost)
    validation_effective_suffix_states = None

    if (
        args.cap_mode == "validation"
        and validation_measurement is not None
        and validation_measurement.get("validation_usable_for_cap")
    ):
        saved_per_hit = float(validation_measurement["clipped_saved_per_hit_ns"])
        validation_effective_suffix_states = safe_ratio(saved_per_hit, args.uncached_cost_per_state)
        uncached_suffix_cost = cached_suffix_eval_cost + saved_per_hit

    build_cost = args.build_base_cost + build_states * args.build_cost_per_state
    fit_cost = representation.points * representation.fit_cost_per_point
    storage_cost = representation.bytes_estimate * args.storage_cost_per_byte
    amortized_decode_cost = representation.bytes_estimate * args.decode_cost_per_byte
    one_time_cost = build_cost + fit_cost + storage_cost + amortized_decode_cost
    hits_to_break_even = math.inf if saved_per_hit <= 0.0 else one_time_cost / saved_per_hit
    net_value = expected_hits * saved_per_hit - one_time_cost

    row = {
        "record_type": "distribution_precompute_depth_estimate",
        "cap_mode": args.cap_mode,
        "boundary_depth": int(boundary_depth),
        "target_depth": int(target_depth),
        "suffix_hops": suffix_hops,
        "representation": representation.name,
        "points": int(representation.points),
        "bytes_estimate": representation.bytes_estimate,
        "branching_factor": branching_at_depth(boundary_depth, args.branching_factor, depth_branching),
        "cumulative_branching": cumulative_b,
        "query_reach_probability": query_reach_prob,
        "expected_hits": expected_hits,
        "expected_build_states": build_states,
        "expected_suffix_states": suffix_states,
        "cap_mode": args.cap_mode,
        "cap_limited_suffix_states": cap_limited_suffix_work,
        "cap_ceiling": cap_ceiling,
        "uncached_suffix_cost": uncached_suffix_cost,
        "cached_suffix_eval_cost": cached_suffix_eval_cost,
        "per_hit_decode_cost": per_hit_decode_cost,
        "splice_cost": splice_cost,
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
    if validation_measurement is not None:
        row.update({
            "validation_rows": validation_measurement["rows"],
            "validation_mean_time_ratio": validation_measurement["mean_time_ratio"],
            "validation_mean_full_time_ns": validation_measurement["mean_full_time_ns"],
            "validation_mean_cached_time_ns": validation_measurement["mean_cached_time_ns"],
            "validation_mean_cache_hits": validation_measurement["mean_cache_hits"],
            "validation_all_rows_mean_cache_hits": validation_measurement["all_rows_mean_cache_hits"],
            "validation_positive_cache_hit_rows": validation_measurement["positive_cache_hit_rows"],
            "validation_zero_cache_hit_rows": validation_measurement["zero_cache_hit_rows"],
            "validation_usable_for_cap": validation_measurement["validation_usable_for_cap"],
            "validation_mean_payload_bytes_read": validation_measurement["mean_payload_bytes_read"],
            "validation_mean_decode_ns": validation_measurement["mean_decode_ns"],
            "validation_measured_pays": validation_measurement["measured_pays"],
            "validation_measured_saved_per_hit_ns": validation_measurement["measured_saved_per_hit_ns"],
            "validation_clipped_saved_per_hit_ns": validation_measurement["clipped_saved_per_hit_ns"],
            "validation_effective_suffix_states": validation_effective_suffix_states,
            "validation_prediction_matches_measured": None if not validation_measurement["validation_usable_for_cap"] else (net_value > 0.0) == validation_measurement["measured_pays"],
        })
    return row


def build_records(args, calibration=None, validation_measurements=None):
    depth_branching = parse_depth_float_map(args.depth_branching)
    query_reach = parse_depth_float_map(args.query_reach_probability)
    validation_measurements = validation_measurements or {}
    target_depth = args.max_depth if args.target_depth is None else int(args.target_depth)
    records = [{
        "record_type": "distribution_precompute_depth_estimator_selection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph": args.graph_name,
        "expected_queries": args.expected_queries,
        "branching_factor": args.branching_factor,
        "depth_branching": depth_branching,
        "query_reach_probability": query_reach,
        "max_depth": args.max_depth,
        "target_depth": target_depth,
        "calibration": calibration,
        "validation_measurements": [
            validation_measurements[depth]
            for depth in sorted(validation_measurements)
        ],
        "cost_model": {
            "cap_mode": args.cap_mode,
            "path_cap": args.path_cap,
            "expansion_cap": args.expansion_cap,
            "estimated_full_work": args.estimated_full_work,
            "uncached_base_cost": args.uncached_base_cost,
            "uncached_cost_per_state": args.uncached_cost_per_state,
            "cached_eval_base_cost": args.cached_eval_base_cost,
            "cached_eval_cost_per_point": args.cached_eval_cost_per_point,
            "splice_cost_per_point": args.splice_cost_per_point,
            "decode_memoized": args.decode_memoized,
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
        validation_measurement = validation_measurements.get(depth)
        for representation in default_representations(depth, args):
            records.append(estimate_row(depth, target_depth, representation, args, depth_branching, reach_probability, validation_measurement))
    return records


def summarize(records):
    selection = next(row for row in records if row["record_type"] == "distribution_precompute_depth_estimator_selection")
    estimate_rows = [row for row in records if row["record_type"] == "distribution_precompute_depth_estimate"]
    by_depth = {}
    for row in estimate_rows:
        by_depth.setdefault(row["boundary_depth"], []).append(row)

    lines = [
        "# Distribution Precompute Depth Estimator",
        "",
        "Graph: `{}`".format(selection["graph"]),
        "",
        "Expected queries: `{:.3f}`".format(selection["expected_queries"]),
        "",
        "Default parent branching prior `b = E[p^2] / E[p]`: `{:.6f}`".format(selection["branching_factor"]),
        "",
        "Target depth: `{}`".format(selection["target_depth"]),
        "",
        "Cap mode: `{}`".format(selection["cost_model"]["cap_mode"]),
        "",
    ]
    if selection.get("calibration"):
        calibration = selection["calibration"]
        lines.extend([
            "## Calibration",
            "",
            "| source_summaries | budget_rows | uncached_cost_per_state | build_cost_per_state | cached_eval_cost_per_point | decode_cost_per_byte | mean_parent_reference_reuse |",
            "|-----------------:|------------:|------------------------:|---------------------:|---------------------------:|---------------------:|----------------------------:|",
            "| {sources} | {rows} | {uncached:.3f} | {build:.3f} | {cached:.3f} | {decode:.3f} | {reuse:.3f} |".format(
                sources=calibration["source_summaries"],
                rows=calibration["source_budget_rows"],
                uncached=calibration["uncached_cost_per_state"],
                build=calibration["build_cost_per_state"],
                cached=calibration["cached_eval_cost_per_point"],
                decode=calibration["decode_cost_per_byte"],
                reuse=calibration["mean_parent_reference_reuse"],
            ),
            "",
        ])
    if selection.get("validation_measurements"):
        lines.extend([
            "## Validation Measurements",
            "",
            "| boundary_depth | rows | positive_hit_rows | zero_hit_rows | mean_time_ratio | mean_cache_hits | measured_saved_per_hit_ns | clipped_saved_per_hit_ns | usable_for_cap | measured_pays |",
            "|---------------:|-----:|------------------:|--------------:|----------------:|----------------:|--------------------------:|-------------------------:|----------------|---------------|",
        ])
        for measurement in selection["validation_measurements"]:
            raw_saved = measurement["measured_saved_per_hit_ns"]
            lines.append(
                "| {depth} | {rows} | {positive_rows} | {zero_rows} | {ratio} | {hits:.3f} | {raw} | {clipped:.3f} | {usable} | {pays} |".format(
                    depth=measurement["boundary_depth"],
                    rows=measurement["rows"],
                    positive_rows=measurement["positive_cache_hit_rows"],
                    zero_rows=measurement["zero_cache_hit_rows"],
                    ratio="n/a" if not measurement["validation_usable_for_cap"] else "{:.3f}".format(measurement["mean_time_ratio"]),
                    hits=measurement["mean_cache_hits"],
                    raw="n/a" if raw_saved is None else "{:.3f}".format(raw_saved),
                    clipped=measurement["clipped_saved_per_hit_ns"],
                    usable="yes" if measurement["validation_usable_for_cap"] else "no",
                    pays="yes" if measurement["measured_pays"] else "no",
                )
            )
        lines.append("")
    lines.extend([
        "## Depth Recommendation",
        "",
        "| boundary_depth | suffix_hops | expected_hits | suffix_states | cap_limited_suffix_states | build_states | best_representation | validation_time_ratio | hits_to_break_even | net_value | pays |",
        "|---------------:|------------:|--------------:|--------------:|--------------------------:|-------------:|--------------------|----------------------:|-------------------:|----------:|------|",
    ])
    recommendation_rows = []
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        best = max(rows, key=lambda item: item["expected_net_value"])
        recommendation_rows.append(best)
        lines.append(
            "| {depth} | {suffix_hops} | {hits:.3f} | {suffix_states:.3f} | {cap_suffix_states:.3f} | {build_states:.3f} | {rep} | {ratio} | {breakeven} | {net:.3f} | {pays} |".format(
                depth=depth,
                suffix_hops=best["suffix_hops"],
                hits=best["expected_hits"],
                suffix_states=best["expected_suffix_states"],
                cap_suffix_states=best["cap_limited_suffix_states"],
                build_states=best["expected_build_states"],
                rep=best["representation"],
                ratio=(
                    "n/a"
                    if "validation_mean_time_ratio" not in best or not best.get("validation_usable_for_cap", True)
                    else "{:.3f}".format(best["validation_mean_time_ratio"])
                ),
                breakeven="n/a" if best["hits_to_break_even"] is None else "{:.3f}".format(best["hits_to_break_even"]),
                net=best["expected_net_value"],
                pays="yes" if best["precompute_pays"] else "no",
            )
        )

    lines.extend([
        "",
        "## Representation Detail",
        "",
        "| boundary_depth | suffix_hops | representation | points | bytes | expected_hits | suffix_states | cap_limited_suffix_states | saved_per_hit | validation_time_ratio | per_hit_decode | splice_cost | one_time_cost | hits_to_break_even | net_value | pays |",
        "|---------------:|------------:|----------------|-------:|------:|--------------:|--------------:|--------------------------:|--------------:|----------------------:|---------------:|------------:|--------------:|-------------------:|----------:|------|",
    ])
    for row in estimate_rows:
        lines.append(
            "| {depth} | {suffix_hops} | {rep} | {points} | {bytes:.1f} | {hits:.3f} | {suffix_states:.3f} | {cap_suffix_states:.3f} | {saved:.3f} | {ratio} | {decode:.3f} | {splice:.3f} | {cost:.3f} | {breakeven} | {net:.3f} | {pays} |".format(
                depth=row["boundary_depth"],
                suffix_hops=row["suffix_hops"],
                rep=row["representation"],
                points=row["points"],
                bytes=row["bytes_estimate"],
                hits=row["expected_hits"],
                suffix_states=row["expected_suffix_states"],
                cap_suffix_states=row["cap_limited_suffix_states"],
                saved=row["saved_per_hit"],
                ratio=(
                    "n/a"
                    if "validation_mean_time_ratio" not in row or not row.get("validation_usable_for_cap", True)
                    else "{:.3f}".format(row["validation_mean_time_ratio"])
                ),
                decode=row["per_hit_decode_cost"],
                splice=row["splice_cost"],
                cost=row["one_time_cost"],
                breakeven="n/a" if row["hits_to_break_even"] is None else "{:.3f}".format(row["hits_to_break_even"]),
                net=row["expected_net_value"],
                pays="yes" if row["precompute_pays"] else "no",
            )
        )

    paying_depths = [row["boundary_depth"] for row in recommendation_rows if row["precompute_pays"]]
    lines.extend([
        "",
        "## Summary",
        "",
        "- deepest recommended boundary depth: `{}`".format(max(paying_depths) if paying_depths else "none"),
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
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum boundary depth to evaluate.")
    parser.add_argument("--target-depth", type=int, default=None, help="Depth of the query target. Defaults to --max-depth.")
    parser.add_argument("--min-path-states", type=float, default=1.0)
    parser.add_argument("--cap-mode", choices=["uncapped", "path", "expansion", "measured", "validation"], default="uncapped", help="Ceiling or validation mode applied to skipped suffix work.")
    parser.add_argument("--path-cap", type=float, default=0.0, help="Path cap used when --cap-mode=path; non-positive disables the ceiling.")
    parser.add_argument("--expansion-cap", type=float, default=0.0, help="Expansion cap used when --cap-mode=expansion; non-positive disables the ceiling.")
    parser.add_argument("--estimated-full-work", type=float, default=0.0, help="Measured full-search work ceiling used when --cap-mode=measured.")
    parser.add_argument("--uncached-base-cost", type=float, default=0.0)
    parser.add_argument("--uncached-cost-per-state", type=float, default=1.0)
    parser.add_argument("--cached-eval-base-cost", type=float, default=1.0)
    parser.add_argument("--cached-eval-cost-per-point", type=float, default=0.02)
    parser.add_argument("--splice-cost-per-point", type=float, default=0.0)
    parser.add_argument("--decode-memoized", action="store_true", help="Do not charge distribution decode cost per query hit.")
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
    parser.add_argument("--calibration-summary", action="append", type=Path, default=[], help="Payload recurrence layer summary JSON used to derive timing costs. Can be repeated.")
    parser.add_argument("--validation-jsonl", action="append", type=Path, default=[], help="Boundary-cache validation JSONL used to derive per-depth measured saved-per-hit. Can be repeated.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    calibration = apply_calibration(args)
    validation_measurements = validation_measurements_from_jsonl(args.validation_jsonl)
    records = build_records(args, calibration, validation_measurements)
    summary = summarize(records)
    summary_json, summary_md = write_outputs(records, summary, args.output_dir, args.graph_name)
    print(summary, end="")
    print("summary_json={}".format(summary_json))
    print("summary_md={}".format(summary_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
