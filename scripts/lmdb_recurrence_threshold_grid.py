#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Calibrate recurrence approximation thresholds over LMDB categories."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_parent_boundary_cache_benchmark import run_benchmark  # noqa: E402
from scripts.lmdb_parent_branching_diagnostic import parse_int_list  # noqa: E402
from scripts.lmdb_parent_histogram_benchmark import safe_graph_name  # noqa: E402


def parse_float_list(text):
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_name_list(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def mean(values):
    values = list(values)
    return 0.0 if not values else statistics.fmean(values)


def max_or_zero(values):
    values = list(values)
    return max(values) if values else 0.0


def grid_cases(args):
    cases = []
    state_limits = parse_int_list(args.max_recurrence_states_grid)
    bin_limits = parse_int_list(args.max_effective_bins_grid)
    mean_models = parse_name_list(args.mean_models)
    blend_values = parse_float_list(args.blend_values)
    for state_limit in state_limits:
        for bin_limit in bin_limits:
            for mean_model in mean_models:
                if mean_model == "blend":
                    for alpha in blend_values:
                        cases.append({
                            "max_recurrence_states": state_limit,
                            "max_effective_bins_after_trim": bin_limit,
                            "mean_model": mean_model,
                            "mean_blend": alpha,
                        })
                else:
                    cases.append({
                        "max_recurrence_states": state_limit,
                        "max_effective_bins_after_trim": bin_limit,
                        "mean_model": mean_model,
                        "mean_blend": args.parametric_mean_blend,
                    })
    return cases


def case_graph_name(base_name, case):
    alpha = ""
    if case["mean_model"] == "blend":
        alpha = "_a{:03d}".format(int(round(float(case["mean_blend"]) * 1000)))
    return "{}_s{}_b{}_{}{}".format(
        base_name,
        case["max_recurrence_states"],
        case["max_effective_bins_after_trim"],
        case["mean_model"].replace("-", "_"),
        alpha,
    )


def benchmark_args_for_case(args, case):
    values = copy.copy(vars(args))
    values["graph_name"] = case_graph_name(args.graph_name, case)
    values["boundary_builder"] = "recurrence"
    values["max_recurrence_states"] = case["max_recurrence_states"]
    values["max_effective_bins_after_trim"] = case["max_effective_bins_after_trim"]
    values["parametric_shape_model"] = "support-binomial"
    values["parametric_mean_model"] = case["mean_model"]
    values["parametric_mean_blend"] = case["mean_blend"]
    values["output_dir"] = None
    return SimpleNamespace(**values)


def summarize_case(case, records):
    selection = next(row for row in records if row.get("record_type") == "boundary_cache_selection")
    cache_rows = [row for row in records if row.get("record_type") == "boundary_cache_entry"]
    comparisons = [row for row in records if row.get("record_type") == "boundary_cache_comparison"]
    parametric_rows = [row for row in cache_rows if row.get("parametric_cached")]
    forced_rows = [row for row in cache_rows if row.get("approximation_forced_by_threshold")]
    states_over = [row for row in cache_rows if row.get("recurrence_states_over_limit")]
    bins_over = [row for row in cache_rows if row.get("effective_bins_over_limit")]
    threshold_disagree = [
        row for row in cache_rows
        if bool(row.get("recurrence_states_over_limit")) != bool(row.get("effective_bins_over_limit"))
    ]
    out = {
        "record_type": "recurrence_threshold_grid_case",
        "graph": selection["graph"],
        "root": selection["root"],
        **case,
        "boundary_nodes": selection["boundary_nodes"],
        "histogram_cached": selection["cached_boundary_nodes"],
        "parametric_cached": selection["parametric_boundary_nodes"],
        "forced_parametric": len(forced_rows),
        "states_over_limit": len(states_over),
        "bins_over_limit": len(bins_over),
        "threshold_disagree": len(threshold_disagree),
        "mean_recurrence_states": mean(
            int(row["recurrence_states_evaluated"])
            for row in cache_rows
            if row.get("recurrence_states_evaluated") is not None
        ),
        "mean_effective_bins_after_trim": mean(int(row.get("effective_support_bins_after_trim", 0)) for row in cache_rows),
        "mean_forced_recurrence_states": mean(
            int(row["recurrence_states_evaluated"])
            for row in forced_rows
            if row.get("recurrence_states_evaluated") is not None
        ),
        "mean_parametric_bins": mean(int(row["parametric_support_bins"]) for row in parametric_rows),
        "mean_parametric_mass_ratio": mean(
            float(row["parametric_mass_ratio"])
            for row in parametric_rows
            if row.get("parametric_mass_ratio") is not None
        ),
    }
    by_budget = {}
    for row in comparisons:
        by_budget.setdefault(int(row["budget"]), []).append(row)
    for budget, rows in sorted(by_budget.items()):
        prefix = "budget_{}".format(budget)
        out["{}_rows".format(prefix)] = len(rows)
        out["{}_mean_l1".format(prefix)] = mean(float(row["l1_error"]) for row in rows)
        out["{}_max_l1".format(prefix)] = max_or_zero(float(row["l1_error"]) for row in rows)
        out["{}_mean_cdf".format(prefix)] = mean(float(row["max_cdf_error"]) for row in rows)
        out["{}_mean_path_count_relative_error".format(prefix)] = mean(float(row["path_count_relative_error"]) for row in rows)
        out["{}_mean_abs_path_delta".format(prefix)] = mean(int(row["abs_path_count_delta"]) for row in rows)
        out["{}_mean_node_ratio".format(prefix)] = mean(float(row["node_expansion_ratio"]) for row in rows)
        out["{}_mean_param_hits".format(prefix)] = mean(int(row["parametric_cache_hits"]) for row in rows)
        out["{}_mean_param_bins_spliced".format(prefix)] = mean(int(row["parametric_bins_spliced"]) for row in rows)
    return out


def run_grid(args):
    records = [{
        "record_type": "recurrence_threshold_grid_selection",
        "graph": args.graph_name,
        "root": args.root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_recurrence_states_grid": parse_int_list(args.max_recurrence_states_grid),
        "max_effective_bins_grid": parse_int_list(args.max_effective_bins_grid),
        "mean_models": parse_name_list(args.mean_models),
        "blend_values": parse_float_list(args.blend_values),
        "boundary_depths": args.boundary_depths,
        "target_depths": args.target_depths,
        "tail_epsilon": args.tail_epsilon,
    }]
    for case in grid_cases(args):
        case_args = benchmark_args_for_case(args, case)
        case_records, _summary = run_benchmark(case_args)
        records.append(summarize_case(case, case_records))
    return records


def budgets_in_cases(cases):
    return sorted({
        int(key.split("_")[1])
        for row in cases
        for key in row
        if key.startswith("budget_") and key.endswith("_mean_l1")
    })


def summarize(records):
    cases = [row for row in records if row.get("record_type") == "recurrence_threshold_grid_case"]
    budgets = budgets_in_cases(cases)
    lines = [
        "# Recurrence Approximation Threshold Grid",
        "",
        "Graph: `{}`".format(records[0].get("graph", "unknown") if records else "unknown"),
        "",
        "Root: `{}`".format(records[0].get("root", "unknown") if records else "unknown"),
        "",
        "## Threshold Cases",
        "",
        "| states | bins | mean_model | alpha | boundary_nodes | histogram_cached | parametric_cached | forced | states_over | bins_over | disagree | mean_states | mean_bins | forced_mean_states |",
        "|-------:|-----:|------------|------:|---------------:|-----------------:|------------------:|-------:|------------:|----------:|---------:|------------:|----------:|-------------------:|",
    ]
    for row in cases:
        alpha = "n/a" if row["mean_model"] != "blend" else "{:.3f}".format(float(row["mean_blend"]))
        lines.append(
            "| {states} | {bins} | {mean_model} | {alpha} | {boundary_nodes} | {histogram_cached} | {parametric_cached} | {forced} | {states_over} | {bins_over} | {disagree} | {mean_states:.3f} | {mean_bins:.3f} | {forced_states:.3f} |".format(
                states=row["max_recurrence_states"],
                bins=row["max_effective_bins_after_trim"],
                mean_model=row["mean_model"],
                alpha=alpha,
                boundary_nodes=row["boundary_nodes"],
                histogram_cached=row["histogram_cached"],
                parametric_cached=row["parametric_cached"],
                forced=row["forced_parametric"],
                states_over=row["states_over_limit"],
                bins_over=row["bins_over_limit"],
                disagree=row["threshold_disagree"],
                mean_states=row["mean_recurrence_states"],
                mean_bins=row["mean_effective_bins_after_trim"],
                forced_states=row["mean_forced_recurrence_states"],
            )
        )
    for budget in budgets:
        prefix = "budget_{}".format(budget)
        rows = [row for row in cases if "{}_mean_l1".format(prefix) in row]
        rows.sort(key=lambda row: (
            row["{}_mean_path_count_relative_error".format(prefix)],
            row["{}_mean_l1".format(prefix)],
            -row["forced_parametric"],
            row["max_recurrence_states"],
            row["max_effective_bins_after_trim"],
        ))
        lines.extend([
            "",
            "## Budget {}".format(budget),
            "",
            "| states | bins | mean_model | alpha | forced | parametric_cached | mean_l1 | max_l1 | mean_cdf | mean_path_rel | mean_abs_delta | mean_node_ratio | mean_param_hits | mean_param_bins_spliced |",
            "|-------:|-----:|------------|------:|-------:|------------------:|--------:|-------:|---------:|--------------:|---------------:|----------------:|----------------:|------------------------:|",
        ])
        for row in rows:
            alpha = "n/a" if row["mean_model"] != "blend" else "{:.3f}".format(float(row["mean_blend"]))
            lines.append(
                "| {states} | {bins} | {mean_model} | {alpha} | {forced} | {parametric} | {l1:.6f} | {max_l1:.6f} | {cdf:.6f} | {path:.6f} | {delta:.3f} | {node_ratio:.3f} | {hits:.3f} | {bins_spliced:.3f} |".format(
                    states=row["max_recurrence_states"],
                    bins=row["max_effective_bins_after_trim"],
                    mean_model=row["mean_model"],
                    alpha=alpha,
                    forced=row["forced_parametric"],
                    parametric=row["parametric_cached"],
                    l1=row["{}_mean_l1".format(prefix)],
                    max_l1=row["{}_max_l1".format(prefix)],
                    cdf=row["{}_mean_cdf".format(prefix)],
                    path=row["{}_mean_path_count_relative_error".format(prefix)],
                    delta=row["{}_mean_abs_path_delta".format(prefix)],
                    node_ratio=row["{}_mean_node_ratio".format(prefix)],
                    hits=row["{}_mean_param_hits".format(prefix)],
                    bins_spliced=row["{}_mean_param_bins_spliced".format(prefix)],
                )
            )
    return "\n".join(lines) + "\n"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    json_path = output_dir / "{}_recurrence_threshold_grid.json".format(safe_name)
    summary_path = output_dir / "{}_recurrence_threshold_grid.md".format(safe_name)
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    return json_path, summary_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path)
    parser.add_argument("--root", required=True, type=int)
    parser.add_argument("--graph-name", default="recurrence_threshold_grid")
    parser.add_argument("--max-recurrence-states-grid", default="25,50,100,200")
    parser.add_argument("--max-effective-bins-grid", default="3,4,6,8")
    parser.add_argument("--mean-models", default="prior-clipped,midpoint,blend")
    parser.add_argument("--blend-values", default="0.25,0.50,0.75")
    parser.add_argument("--boundary-depths", default="1")
    parser.add_argument("--target-depths", default="2")
    parser.add_argument("--children-per-node", type=int, default=64)
    parser.add_argument("--frontier-limit", type=int, default=600)
    parser.add_argument("--boundaries-per-depth", type=int, default=24)
    parser.add_argument("--targets-per-depth", type=int, default=8)
    parser.add_argument("--boundary-budget", type=int, default=6)
    parser.add_argument("--budgets", default="6,8")
    parser.add_argument("--admission-policy", choices=["baseline", "depth-prior"], default="baseline")
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--max-histogram-bytes", type=int, default=1024)
    parser.add_argument("--parametric-bytes", type=int, default=64)
    parser.add_argument("--parametric-mean-blend", type=float, default=0.5)
    parser.add_argument("--parametric-support-source", choices=["measured", "distance-bounds"], default="measured")
    parser.add_argument("--parametric-mass-model", choices=["oracle", "unit", "depth-prior"], default="oracle")
    parser.add_argument("--parametric-mass-cap", type=int, default=100000)
    parser.add_argument("--tail-epsilon", type=float, default=0.001)
    parser.add_argument("--max-parent-depth", type=int, default=24)
    parser.add_argument("--path-cap", type=int, default=50000)
    parser.add_argument("--expansion-cap", type=int, default=100000)
    parser.add_argument("--seed", default="recurrence-threshold-grid-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = run_grid(args)
    summary = summarize(records)
    json_path, summary_path = write_outputs(records, summary, args.output_dir, args.graph_name)
    print(summary, end="")
    print("json={}".format(json_path))
    print("summary={}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
