#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Sweep boundary-probe path-value kernels over one fixed target selection.

The boundary probe exposes several path-value kernels but runs only one kernel
at a time.  This wrapper runs the same LMDB/root/selection settings for a small
kernel grid, tags each emitted record with the variant, and writes a compact
cross-kernel summary.  It is intended for calibration reports, not for heavy
benchmark sweeps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_boundary_coverage_probe import run_probe
from scripts.lmdb_parent_histogram_benchmark import safe_graph_name


DEFAULT_VARIANTS = "count,bp-decay:auto,bp-decay:explicit,weighted-power:1,weighted-power:2"


def parse_kernel_variants(text, explicit_branching_factor):
    variants = []
    for raw_part in str(text).split(","):
        raw_part = raw_part.strip()
        if not raw_part:
            continue
        parts = raw_part.split(":", 1)
        kernel = parts[0].strip().replace("_", "-")
        parameter = parts[1].strip() if len(parts) > 1 else None
        if kernel == "count":
            variants.append({
                "label": "count",
                "path_value_kernel": "count",
                "path_value_branching_factor": None,
                "path_value_power": 1.0,
            })
        elif kernel == "bp-decay":
            if parameter in (None, "", "auto"):
                label = "bp_decay_auto"
                branching_factor = None
            elif parameter == "explicit":
                label = "bp_decay_explicit_{}".format(format_slug(explicit_branching_factor))
                branching_factor = explicit_branching_factor
            else:
                label = "bp_decay_{}".format(format_slug(float(parameter)))
                branching_factor = float(parameter)
            variants.append({
                "label": label,
                "path_value_kernel": "bp-decay",
                "path_value_branching_factor": branching_factor,
                "path_value_power": 1.0,
            })
        elif kernel == "weighted-power":
            power = 1.0 if parameter in (None, "") else float(parameter)
            variants.append({
                "label": "weighted_power_{}".format(format_slug(power)),
                "path_value_kernel": "weighted-power",
                "path_value_branching_factor": None,
                "path_value_power": power,
            })
        else:
            raise SystemExit("unknown kernel variant: {}".format(raw_part))
    if not variants:
        raise SystemExit("--kernel-variants produced no variants")
    return variants


def format_slug(value):
    return str(value).replace(".", "p").replace("-", "m")


def format_optional(value, digits=6):
    if value is None:
        return "n/a"
    return ("{:." + str(digits) + "f}").format(float(value))


def mean_optional(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else statistics.mean(values)


def sum_optional(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else sum(values)


def probe_args_for_variant(args, variant):
    return Namespace(
        lmdb_dir=args.lmdb_dir,
        root=args.root,
        graph_name="{}_{}".format(args.graph_name, variant["label"]),
        mode=args.mode,
        parent_filter=args.parent_filter,
        selection_source=args.selection_source,
        root_cone_depth=args.root_cone_depth,
        root_cone_children_per_node=args.root_cone_children_per_node,
        root_cone_frontier_limit=args.root_cone_frontier_limit,
        require_targets_in_root_cone=args.require_targets_in_root_cone,
        require_boundaries_in_root_cone=args.require_boundaries_in_root_cone,
        skip_boundary_suffix_mass=args.skip_boundary_suffix_mass,
        boundary_depths=args.boundary_depths,
        target_depths=args.target_depths,
        children_per_node=args.children_per_node,
        frontier_limit=args.frontier_limit,
        boundaries_per_depth=args.boundaries_per_depth,
        targets_per_depth=args.targets_per_depth,
        target_selection=args.target_selection,
        include_target_ancestor_boundaries=args.include_target_ancestor_boundaries,
        target_ancestor_boundary_limit=args.target_ancestor_boundary_limit,
        max_parent_depth=args.max_parent_depth,
        budgets=args.budgets,
        path_count_cap=args.path_count_cap,
        expansion_cap=args.expansion_cap,
        samples=args.samples,
        seed=args.seed,
        path_value_kernel=variant["path_value_kernel"],
        path_value_branching_factor=variant["path_value_branching_factor"],
        path_value_power=variant["path_value_power"],
    )


def variant_selection(records):
    return next(row for row in records if row.get("record_type") == "boundary_coverage_selection")


def target_rows(records):
    return [row for row in records if row.get("record_type") == "boundary_coverage_target"]


def variant_summary_record(label, records, elapsed_ns):
    selection = variant_selection(records)
    return {
        "record_type": "path_value_kernel_sweep_variant",
        "variant": label,
        "path_value_kernel": selection.get("path_value_kernel"),
        "path_value_branching_factor": selection.get("path_value_branching_factor"),
        "path_value_branching_factor_source": selection.get("path_value_branching_factor_source"),
        "path_value_power": selection.get("path_value_power"),
        "path_value_branching_stats": selection.get("path_value_branching_stats"),
        "targets": selection.get("targets"),
        "boundary_nodes": selection.get("boundary_nodes"),
        "budgets": selection.get("budgets"),
        "samples": selection.get("samples"),
        "elapsed_ns": int(elapsed_ns),
    }


def grouped_estimate_rows(all_records):
    groups = {}
    for row in all_records:
        if row.get("record_type") != "boundary_coverage_target":
            continue
        key = (row.get("kernel_variant"), row.get("mode"), row.get("path_length_budget"))
        groups.setdefault(key, []).append(row)

    out = []
    for (variant, mode, budget), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        out.append({
            "variant": variant,
            "mode": mode,
            "budget": budget,
            "targets": len(rows),
            "completed_targets": sum(1 for row in rows if row.get("completed")),
            "observed_terminal_prefixes": sum(int(row.get("terminal_prefixes", 0)) for row in rows),
            "observed_root_paths": sum(int(row.get("root_paths", 0)) for row in rows),
            "observed_boundary_hit_prefixes": sum(int(row.get("boundary_hit_prefixes", 0)) for row in rows),
            "mean_estimated_root_paths": mean_optional(rows, "estimated_root_paths"),
            "mean_estimated_root_value_sum": mean_optional(rows, "estimated_root_value_sum"),
            "mean_estimated_kernel_mean_root_path_length": mean_optional(rows, "estimated_kernel_mean_root_path_length"),
            "mean_estimated_spliced_total_root_paths": mean_optional(rows, "estimated_spliced_total_root_paths"),
            "mean_estimated_spliced_total_value_sum": mean_optional(rows, "estimated_spliced_total_value_sum"),
            "mean_estimated_boundary_hit_fraction": mean_optional(rows, "estimated_boundary_hit_fraction"),
            "mean_estimated_root_boundary_hit_fraction": mean_optional(rows, "estimated_root_boundary_hit_fraction"),
            "mean_estimated_root_value_boundary_hit_fraction": mean_optional(rows, "estimated_root_value_boundary_hit_fraction"),
            "elapsed_ms": None if sum_optional(rows, "elapsed_ns") is None else sum_optional(rows, "elapsed_ns") / 1_000_000.0,
        })
    return out


def summarize_sweep(records, args):
    variant_rows = [row for row in records if row.get("record_type") == "path_value_kernel_sweep_variant"]
    estimate_rows = grouped_estimate_rows(records)
    lines = [
        "# LMDB Path-Value Kernel Sweep",
        "",
        "Graph: `{}`".format(args.graph_name),
        "",
        "Root: `{}`".format(args.root),
        "",
        "LMDB: `{}`".format(args.lmdb_dir),
        "",
        "Mode: `{}`".format(args.mode),
        "",
        "Parent filter: `{}`".format(args.parent_filter),
        "",
        "Selection source: `{}`".format(args.selection_source),
        "",
        "Budgets: `{}`".format(args.budgets),
        "",
        "Samples per target: `{}`".format(args.samples),
        "",
        "This report reruns one fixed target-selection recipe for several path-value kernels. Path-count estimates remain the branch-product estimates from the boundary probe; value-sum estimates apply the selected kernel to root-reaching paths or boundary suffix histograms.",
        "",
        "## Variants",
        "",
        "| variant | kernel | b_p | b_p_source | power | targets | boundary_nodes | elapsed_ms |",
        "|---------|--------|----:|------------|------:|--------:|---------------:|-----------:|",
    ]
    for row in variant_rows:
        lines.append(
            "| {variant} | {kernel} | {bp} | {source} | {power} | {targets} | {boundaries} | {elapsed} |".format(
                variant=row["variant"],
                kernel=row["path_value_kernel"],
                bp=format_optional(row.get("path_value_branching_factor"), 6),
                source=row.get("path_value_branching_factor_source") or "n/a",
                power=format_optional(row.get("path_value_power"), 3),
                targets=row.get("targets"),
                boundaries=row.get("boundary_nodes"),
                elapsed=format_optional(row.get("elapsed_ns", 0) / 1_000_000.0, 3),
            )
        )

    lines.extend([
        "",
        "## Estimate Comparison",
        "",
        "| variant | mode | budget | targets | observed_root_paths | observed_boundary_hits | mean_est_root_paths | mean_est_root_value_sum | mean_kernel_mean_length | mean_spliced_paths | mean_spliced_value_sum | mean_boundary_hit_fraction | elapsed_ms |",
        "|---------|------|-------:|--------:|--------------------:|-----------------------:|--------------------:|------------------------:|------------------------:|-------------------:|-----------------------:|---------------------------:|-----------:|",
    ])
    for row in estimate_rows:
        lines.append(
            "| {variant} | {mode} | {budget} | {targets} | {root} | {boundary} | {est_root} | {value_sum} | {mean_len} | {spliced_paths} | {spliced_value} | {boundary_fraction} | {elapsed} |".format(
                variant=row["variant"],
                mode=row["mode"],
                budget=row["budget"],
                targets=row["targets"],
                root=row["observed_root_paths"],
                boundary=row["observed_boundary_hit_prefixes"],
                est_root=format_optional(row["mean_estimated_root_paths"], 3),
                value_sum=format_optional(row["mean_estimated_root_value_sum"], 6),
                mean_len=format_optional(row["mean_estimated_kernel_mean_root_path_length"], 3),
                spliced_paths=format_optional(row["mean_estimated_spliced_total_root_paths"], 3),
                spliced_value=format_optional(row["mean_estimated_spliced_total_value_sum"], 6),
                boundary_fraction=format_optional(row["mean_estimated_boundary_hit_fraction"], 6),
                elapsed=format_optional(row["elapsed_ms"], 3),
            )
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `count` is the control: its value sum equals the path-count quantity.",
        "- `bp-decay:auto` estimates `b_p = E[p^2] / E[p]` from the selected root-cone scope when available.",
        "- `bp-decay:explicit` uses the value passed by `--explicit-branching-factor`.",
        "- `weighted-power:n` computes `(L + 1)^(-n)` and therefore needs prefix-length-aware suffix evaluation.",
        "",
    ])
    return "\n".join(lines)


def run_sweep(args):
    variants = parse_kernel_variants(args.kernel_variants, args.explicit_branching_factor)
    records = []
    for variant in variants:
        probe_args = probe_args_for_variant(args, variant)
        started = time.perf_counter_ns()
        variant_records, _summary = run_probe(probe_args)
        elapsed_ns = time.perf_counter_ns() - started
        for record in variant_records:
            record["kernel_variant"] = variant["label"]
        records.extend(variant_records)
        records.append(variant_summary_record(variant["label"], variant_records, elapsed_ns))
    return records, summarize_sweep(records, args)


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / "lmdb_path_value_kernel_sweep_{}_{}.jsonl".format(safe_name, timestamp)
    summary_path = output_dir / "lmdb_path_value_kernel_sweep_summary_{}_{}.md".format(safe_name, timestamp)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True, type=Path, help="Numeric-keyed category LMDB directory.")
    parser.add_argument("--root", required=True, type=int, help="Numeric root id.")
    parser.add_argument("--graph-name", default="lmdb_path_value_kernel_sweep", help="Graph label used in output filenames.")
    parser.add_argument("--kernel-variants", default=DEFAULT_VARIANTS, help="Comma-separated variants: count, bp-decay:auto, bp-decay:explicit, bp-decay:<b>, weighted-power:<n>.")
    parser.add_argument("--explicit-branching-factor", type=float, default=2.0, help="b_p value used by the bp-decay:explicit variant.")
    parser.add_argument("--mode", choices=["exact", "sample", "root-sample", "both", "all"], default="all", help="Boundary probe mode to run per variant.")
    parser.add_argument("--parent-filter", choices=["all", "root-reachable", "root-cone"], default="all", help="Parent expansion filter passed to the boundary probe.")
    parser.add_argument("--selection-source", choices=["graph", "root-cone"], default="root-cone", help="Selection source passed to the boundary probe.")
    parser.add_argument("--root-cone-depth", type=int, default=3, help="Maximum child depth for the precomputed root cone.")
    parser.add_argument("--root-cone-children-per-node", type=int, help="Child sample cap per root-cone frontier node.")
    parser.add_argument("--root-cone-frontier-limit", type=int, help="Root-cone frontier cap per depth.")
    parser.add_argument("--require-targets-in-root-cone", action="store_true", help="Drop selected targets outside the precomputed root cone.")
    parser.add_argument("--require-boundaries-in-root-cone", action="store_true", help="Drop selected boundaries outside the precomputed root cone.")
    parser.add_argument("--skip-boundary-suffix-mass", action="store_true", help="Disable suffix histogram enumeration after boundary hits.")
    parser.add_argument("--boundary-depths", default="1", help="Child depths used as boundary candidates.")
    parser.add_argument("--target-depths", default="2", help="Child depths used as target candidates.")
    parser.add_argument("--children-per-node", type=int, default=64, help="Deterministic child sample cap per frontier node.")
    parser.add_argument("--frontier-limit", type=int, default=500, help="Deterministic cap for each sampled child-depth frontier.")
    parser.add_argument("--boundaries-per-depth", type=int, default=0, help="Boundary candidates per depth; non-positive keeps all candidates.")
    parser.add_argument("--targets-per-depth", type=int, default=2, help="Targets per requested target depth.")
    parser.add_argument("--target-selection", choices=["child-depth", "boundary-descendants"], default="boundary-descendants", help="Target sampling mode when --selection-source graph is used.")
    parser.add_argument("--include-target-ancestor-boundaries", action="store_true", help="Add target ancestors whose root distance matches requested boundary depths.")
    parser.add_argument("--target-ancestor-boundary-limit", type=int, default=500, help="Maximum target-ancestor boundary nodes to add; non-positive means no extra cap.")
    parser.add_argument("--max-parent-depth", type=int, default=24, help="Parent depth cap for target-ancestor boundary collection.")
    parser.add_argument("--budgets", default="4", help="Comma-separated path-length budgets.")
    parser.add_argument("--path-count-cap", type=int, default=0, help="Exact mode safety cap on terminal prefixes; non-positive disables it.")
    parser.add_argument("--expansion-cap", type=int, default=0, help="Exact mode safety cap on expanded nodes; non-positive disables it.")
    parser.add_argument("--samples", type=int, default=50, help="Sampled mode random walks per target.")
    parser.add_argument("--seed", default="path-value-kernel-sweep", help="Deterministic sampling seed.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    args = parser.parse_args(argv)

    records, summary = run_sweep(args)
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
