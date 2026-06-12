#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Run a small support-binomial mean calibration grid over LMDB categories."""

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
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def parse_name_list(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def grid_cases(args):
    cases = []
    boundary_depths = parse_int_list(args.boundary_depth_grid)
    target_depths = parse_int_list(args.target_depth_grid)
    mean_models = parse_name_list(args.mean_models)
    blend_values = parse_float_list(args.blend_values)
    for boundary_depth in boundary_depths:
        for target_depth in target_depths:
            for mean_model in mean_models:
                if mean_model == "blend":
                    for alpha in blend_values:
                        cases.append({
                            "boundary_depth": boundary_depth,
                            "target_depth": target_depth,
                            "mean_model": mean_model,
                            "mean_blend": alpha,
                        })
                else:
                    cases.append({
                        "boundary_depth": boundary_depth,
                        "target_depth": target_depth,
                        "mean_model": mean_model,
                        "mean_blend": args.parametric_mean_blend,
                    })
    return cases


def case_graph_name(base_name, case):
    if case["mean_model"] == "blend":
        suffix = "bd{}_td{}_blend{:03d}".format(
            case["boundary_depth"],
            case["target_depth"],
            int(round(float(case["mean_blend"]) * 1000)),
        )
    else:
        suffix = "bd{}_td{}_{}".format(
            case["boundary_depth"],
            case["target_depth"],
            case["mean_model"].replace("-", "_"),
        )
    return "{}_{}".format(base_name, suffix)


def benchmark_args_for_case(args, case):
    values = copy.copy(vars(args))
    values["graph_name"] = case_graph_name(args.graph_name, case)
    values["boundary_depths"] = str(case["boundary_depth"])
    values["target_depths"] = str(case["target_depth"])
    values["parametric_shape_model"] = "support-binomial"
    values["parametric_mean_model"] = case["mean_model"]
    values["parametric_mean_blend"] = case["mean_blend"]
    values["output_dir"] = None
    return SimpleNamespace(**values)


def mean(values):
    values = list(values)
    return 0.0 if not values else statistics.fmean(values)


def summarize_case(case, records):
    selection = next(row for row in records if row.get("record_type") == "boundary_cache_selection")
    cache_rows = [row for row in records if row.get("record_type") == "boundary_cache_entry"]
    comparisons = [row for row in records if row.get("record_type") == "boundary_cache_comparison"]
    parametric_rows = [row for row in cache_rows if row.get("parametric_cached")]
    out = {
        "record_type": "support_binomial_mean_grid_case",
        "graph": selection["graph"],
        "root": selection["root"],
        **case,
        "boundary_nodes": selection["boundary_nodes"],
        "histogram_cached": selection["cached_boundary_nodes"],
        "parametric_cached": selection["parametric_boundary_nodes"],
        "targets": selection["targets"],
        "mean_parametric_mass_ratio": mean(
            float(row["parametric_mass_ratio"])
            for row in parametric_rows
            if row.get("parametric_mass_ratio") is not None
        ),
        "mean_parametric_bins": mean(int(row["parametric_support_bins"]) for row in parametric_rows),
        "mean_shape_probability": mean(
            float(row["parametric_shape_probability"])
            for row in parametric_rows
            if row.get("parametric_shape_probability") is not None
        ),
    }
    by_budget = {}
    for row in comparisons:
        by_budget.setdefault(int(row["budget"]), []).append(row)
    for budget, rows in sorted(by_budget.items()):
        prefix = "budget_{}".format(budget)
        out["{}_rows".format(prefix)] = len(rows)
        out["{}_mean_l1".format(prefix)] = mean(float(row["l1_error"]) for row in rows)
        out["{}_mean_cdf".format(prefix)] = mean(float(row["max_cdf_error"]) for row in rows)
        out["{}_mean_path_count_relative_error".format(prefix)] = mean(float(row["path_count_relative_error"]) for row in rows)
        out["{}_mean_abs_path_delta".format(prefix)] = mean(int(row["abs_path_count_delta"]) for row in rows)
        out["{}_mean_param_hits".format(prefix)] = mean(int(row["parametric_cache_hits"]) for row in rows)
        out["{}_mean_param_bins_spliced".format(prefix)] = mean(int(row["parametric_bins_spliced"]) for row in rows)
    return out


def run_grid(args):
    records = [{
        "record_type": "support_binomial_mean_grid_selection",
        "graph": args.graph_name,
        "root": args.root,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary_depth_grid": parse_int_list(args.boundary_depth_grid),
        "target_depth_grid": parse_int_list(args.target_depth_grid),
        "mean_models": parse_name_list(args.mean_models),
        "blend_values": parse_float_list(args.blend_values),
        "parametric_mass_model": args.parametric_mass_model,
    }]
    for case in grid_cases(args):
        case_args = benchmark_args_for_case(args, case)
        case_records, _summary = run_benchmark(case_args)
        records.append(summarize_case(case, case_records))
    return records


def summarize(records):
    cases = [row for row in records if row.get("record_type") == "support_binomial_mean_grid_case"]
    budgets = sorted({
        int(key.split("_")[1])
        for row in cases
        for key in row
        if key.startswith("budget_") and key.endswith("_mean_l1")
    })
    lines = [
        "# Support-Binomial Mean Calibration Grid",
        "",
        "Graph: `{}`".format(records[0].get("graph", "unknown") if records else "unknown"),
        "",
        "Root: `{}`".format(records[0].get("root", "unknown") if records else "unknown"),
        "",
        "## Cases",
        "",
        "| boundary_depth | target_depth | mean_model | alpha | boundary_nodes | parametric_cached | mean_p | mean_parametric_bins |",
        "|---------------:|-------------:|------------|------:|---------------:|------------------:|-------:|---------------------:|",
    ]
    for row in cases:
        alpha = "n/a" if row["mean_model"] != "blend" else "{:.3f}".format(float(row["mean_blend"]))
        lines.append(
            "| {boundary_depth} | {target_depth} | {mean_model} | {alpha} | {boundary_nodes} | {parametric_cached} | {mean_p:.6f} | {mean_bins:.3f} |".format(
                boundary_depth=row["boundary_depth"],
                target_depth=row["target_depth"],
                mean_model=row["mean_model"],
                alpha=alpha,
                boundary_nodes=row["boundary_nodes"],
                parametric_cached=row["parametric_cached"],
                mean_p=row["mean_shape_probability"],
                mean_bins=row["mean_parametric_bins"],
            )
        )
    for budget in budgets:
        prefix = "budget_{}".format(budget)
        lines.extend([
            "",
            "## Budget {}".format(budget),
            "",
            "| boundary_depth | target_depth | mean_model | alpha | mean_l1 | mean_cdf | mean_path_rel | mean_abs_delta | mean_param_hits | mean_param_bins_spliced |",
            "|---------------:|-------------:|------------|------:|--------:|---------:|--------------:|---------------:|----------------:|------------------------:|",
        ])
        rows = [row for row in cases if "{}_mean_l1".format(prefix) in row]
        rows.sort(key=lambda row: (
            row["boundary_depth"],
            row["target_depth"],
            row["{}_mean_path_count_relative_error".format(prefix)],
            row["{}_mean_l1".format(prefix)],
        ))
        for row in rows:
            alpha = "n/a" if row["mean_model"] != "blend" else "{:.3f}".format(float(row["mean_blend"]))
            lines.append(
                "| {boundary_depth} | {target_depth} | {mean_model} | {alpha} | {l1:.6f} | {cdf:.6f} | {path:.6f} | {delta:.3f} | {hits:.3f} | {bins:.3f} |".format(
                    boundary_depth=row["boundary_depth"],
                    target_depth=row["target_depth"],
                    mean_model=row["mean_model"],
                    alpha=alpha,
                    l1=row["{}_mean_l1".format(prefix)],
                    cdf=row["{}_mean_cdf".format(prefix)],
                    path=row["{}_mean_path_count_relative_error".format(prefix)],
                    delta=row["{}_mean_abs_path_delta".format(prefix)],
                    hits=row["{}_mean_param_hits".format(prefix)],
                    bins=row["{}_mean_param_bins_spliced".format(prefix)],
                )
            )
    return "\n".join(lines) + "\n"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_graph_name(graph_name)
    json_path = output_dir / "{}_support_binomial_mean_grid.json".format(safe_name)
    summary_path = output_dir / "{}_support_binomial_mean_grid.md".format(safe_name)
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    return json_path, summary_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path)
    parser.add_argument("--root", required=True, type=int)
    parser.add_argument("--graph-name", default="support_binomial_mean_grid")
    parser.add_argument("--boundary-depth-grid", default="1,2")
    parser.add_argument("--target-depth-grid", default="3,4")
    parser.add_argument("--mean-models", default="midpoint,blend")
    parser.add_argument("--blend-values", default="0.0,0.02,0.05,0.10")
    parser.add_argument("--children-per-node", type=int, default=64)
    parser.add_argument("--frontier-limit", type=int, default=600)
    parser.add_argument("--boundaries-per-depth", type=int, default=24)
    parser.add_argument("--targets-per-depth", type=int, default=8)
    parser.add_argument("--boundary-budget", type=int, default=6)
    parser.add_argument("--budgets", default="6,8")
    parser.add_argument("--admission-policy", choices=["baseline", "depth-prior"], default="depth-prior")
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--max-histogram-bytes", type=int, default=64)
    parser.add_argument("--parametric-bytes", type=int, default=64)
    parser.add_argument("--parametric-mean-blend", type=float, default=0.5)
    parser.add_argument("--parametric-mass-model", choices=["oracle", "unit", "depth-prior"], default="oracle")
    parser.add_argument("--parametric-mass-cap", type=int, default=100000)
    parser.add_argument("--tail-epsilon", type=float, default=0.001)
    parser.add_argument("--max-parent-depth", type=int, default=24)
    parser.add_argument("--path-cap", type=int, default=50000)
    parser.add_argument("--expansion-cap", type=int, default=100000)
    parser.add_argument("--seed", default="support-binomial-mean-grid-v1")
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
