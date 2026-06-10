#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Benchmark scalar parent-path support bounds for cache planning."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distribution_cache_support import (  # noqa: E402
    FIXTURES,
    ROOT,
    exact_histogram,
    load_parent_edges_tsv,
    mass_cdf,
    reachable_nodes_by_parent_distance,
    support_bounds,
)


SIMPLEWIKI_ROOT = "Category:Articles"
DEFAULT_BUDGETS = [2, 4, 6]


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
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


def parse_targets(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def load_targets_file(path: Path) -> list[str]:
    targets = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                targets.append(line)
    return targets


def safe_graph_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "graph"


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def select_file_targets(parents, root, explicit_targets, targets_file, max_target_depth, target_limit):
    if explicit_targets:
        targets = parse_targets(explicit_targets)
    elif targets_file:
        targets = load_targets_file(targets_file)
    else:
        targets = reachable_nodes_by_parent_distance(parents, root, max_target_depth)
    if target_limit is not None:
        targets = targets[:target_limit]
    if not targets:
        raise SystemExit("no support-bounds targets selected")
    return targets


def support_width(l_min, l_max):
    if l_min is None or l_max is None:
        return None
    return l_max - l_min


def target_bounds_record(graph_name, graph_label, root, target, hist, l_min, l_max, parent_degree, exact_error=None):
    hist_min = min(hist) if hist else None
    hist_max = max(hist) if hist else None
    width = support_width(l_min, l_max)
    hist_width = support_width(hist_min, hist_max)
    return {
        "record_type": "target_bounds",
        "graph": graph_name,
        "fixture": graph_label,
        "root": root,
        "target_node": target,
        "L_min": l_min,
        "L_max": l_max,
        "support_width": width,
        "hist_support_min": hist_min,
        "parent_degree": parent_degree,
        "hist_support_max": hist_max,
        "hist_support_width": hist_width,
        "path_count": sum(hist.values()),
        "support_bins": len(hist),
        "bounds_match_histogram": exact_error is None and (l_min, l_max) == (hist_min, hist_max),
        "exact_histogram_error": exact_error,
    }


def budget_signal_record(graph_name, graph_label, root, target, budget, hist, l_min, l_max, narrow_width, wide_width):
    width = support_width(l_min, l_max)
    zero_by_min = l_min is None or l_min > budget
    fully_covered_by_max = l_max is not None and l_max <= budget
    return {
        "record_type": "budget_signal",
        "graph": graph_name,
        "fixture": graph_label,
        "root": root,
        "target_node": target,
        "B_search": budget,
        "L_min": l_min,
        "L_max": l_max,
        "support_width": width,
        "zero_by_min_budget": zero_by_min,
        "fully_covered_by_max_budget": fully_covered_by_max,
        "partial_by_bounds": not zero_by_min and not fully_covered_by_max,
        "narrow_support": width is not None and width <= narrow_width,
        "wide_support": width is not None and width >= wide_width,
        "bounded_mass": mass_cdf(hist, budget),
        "total_mass": sum(hist.values()),
    }


def run_graph_support_bounds(
    graph_name,
    graph_label,
    parents,
    targets,
    budgets,
    root,
    narrow_width=2,
    wide_width=10,
):
    records = []
    min_memo = {}
    max_memo = {}
    hist_memo = {}
    for target in targets:
        exact_error = None
        hist = {}
        try:
            hist = exact_histogram(target, parents, root, hist_memo)
            l_min, l_max = support_bounds(target, parents, root, min_memo, max_memo)
        except ValueError as exc:
            l_min, l_max = None, None
            exact_error = str(exc)
        parent_degree = len(parents.get(target, []))

        records.append(target_bounds_record(graph_name, graph_label, root, target, hist, l_min, l_max, parent_degree, exact_error))
        for budget in budgets:
            records.append(
                budget_signal_record(
                    graph_name,
                    graph_label,
                    root,
                    target,
                    budget,
                    hist,
                    l_min,
                    l_max,
                    narrow_width,
                    wide_width,
                )
            )
    return records


def run_fixture_support_bounds(fixture_name, fixture, budgets, narrow_width=2, wide_width=10):
    return run_graph_support_bounds(
        "tiny_fixture",
        fixture_name,
        fixture["parents"],
        fixture["targets"],
        budgets,
        ROOT,
        narrow_width,
        wide_width,
    )


def parent_branching_moments(rows):
    degrees = [r["parent_degree"] for r in rows if r.get("parent_degree") is not None]
    if not degrees:
        return {
            "nodes": 0,
            "nonzero_parent_nodes": 0,
            "mean_parent_degree": 0.0,
            "second_parent_degree_moment": 0.0,
            "size_biased_parent_branching": 0.0,
            "max_parent_degree": 0,
        }
    mean = statistics.mean(degrees)
    second = statistics.mean([degree * degree for degree in degrees])
    return {
        "nodes": len(degrees),
        "nonzero_parent_nodes": sum(1 for degree in degrees if degree > 0),
        "mean_parent_degree": mean,
        "second_parent_degree_moment": second,
        "size_biased_parent_branching": 0.0 if mean == 0 else second / mean,
        "max_parent_degree": max(degrees),
    }


def summarize(records):
    target_records = [r for r in records if r["record_type"] == "target_bounds"]
    signal_records = [r for r in records if r["record_type"] == "budget_signal"]
    validated = [r for r in target_records if r["exact_histogram_error"] is None]
    widths = [r["support_width"] for r in validated if r["support_width"] is not None]
    path_counts = [r["path_count"] for r in validated]
    failures = [r for r in validated if not r["bounds_match_histogram"]]
    global_parent_moments = parent_branching_moments(validated)

    lines = [
        "# Parent Support Bounds Benchmark Summary",
        "",
        "## Target Bounds",
        "",
        "| targets | validated | unreachable | exact_errors | bounds_failures | mean_width | p95_width | max_width | mean_path_count |",
        "|---------|-----------|-------------|--------------|-----------------|------------|-----------|-----------|-----------------|",
        "| {targets} | {validated} | {unreachable} | {errors} | {failures} | {mean_width:.3f} | {p95_width:.3f} | {max_width} | {mean_paths:.3f} |".format(
            targets=len(target_records),
            validated=len(validated),
            unreachable=sum(1 for r in validated if r["L_min"] is None),
            errors=sum(1 for r in target_records if r["exact_histogram_error"] is not None),
            failures=len(failures),
            mean_width=statistics.mean(widths) if widths else 0.0,
            p95_width=percentile(widths, 95),
            max_width=max(widths, default=0),
            mean_paths=statistics.mean(path_counts) if path_counts else 0.0,
        ),
        "",
        "## Parent Branching Moments",
        "",
        "`E[p^2]/E[p]` is the size-biased parent branching signal over the selected reachable nodes, where `p` is the node's parent degree in the benchmark graph.",
        "",
        "| nodes | nonzero_parent_nodes | E[p] | E[p^2] | E[p^2]/E[p] | max_p |",
        "|-------|----------------------|------|--------|-------------|-------|",
        "| {nodes} | {nonzero} | {mean:.6f} | {second:.6f} | {size_biased:.6f} | {max_p} |".format(
            nodes=global_parent_moments["nodes"],
            nonzero=global_parent_moments["nonzero_parent_nodes"],
            mean=global_parent_moments["mean_parent_degree"],
            second=global_parent_moments["second_parent_degree_moment"],
            size_biased=global_parent_moments["size_biased_parent_branching"],
            max_p=global_parent_moments["max_parent_degree"],
        ),
        "",
        "## Budget Signals",
        "",
        "| B_search | targets | zero_by_min | fully_covered_by_max | partial_by_bounds | narrow_support | wide_support |",
        "|----------|---------|-------------|----------------------|-------------------|----------------|--------------|",
    ]

    by_budget = {}
    for record in signal_records:
        by_budget.setdefault(record["B_search"], []).append(record)
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        lines.append(
            "| {budget} | {targets} | {zero} | {full} | {partial} | {narrow} | {wide} |".format(
                budget=budget,
                targets=len(rows),
                zero=sum(1 for r in rows if r["zero_by_min_budget"]),
                full=sum(1 for r in rows if r["fully_covered_by_max_budget"]),
                partial=sum(1 for r in rows if r["partial_by_bounds"]),
                narrow=sum(1 for r in rows if r["narrow_support"]),
                wide=sum(1 for r in rows if r["wide_support"]),
            )
        )

    bucket_rows = {}
    for record in validated:
        if record["L_min"] is None:
            continue
        bucket_rows.setdefault(record["L_min"], []).append(record)
    lines.extend([
        "",
        "## Root-Distance Buckets",
        "",
        "| L_min | targets | mean_width | max_width | mean_path_count | E[p] | E[p^2]/E[p] | max_p |",
        "|-------|---------|------------|-----------|-----------------|------|-------------|-------|",
    ])
    for l_min in sorted(bucket_rows):
        rows = bucket_rows[l_min]
        row_widths = [r["support_width"] for r in rows if r["support_width"] is not None]
        row_paths = [r["path_count"] for r in rows]
        row_moments = parent_branching_moments(rows)
        lines.append(
            "| {l_min} | {targets} | {mean_width:.3f} | {max_width} | {mean_paths:.3f} | {mean_parent:.6f} | {size_biased:.6f} | {max_p} |".format(
                l_min=l_min,
                targets=len(rows),
                mean_width=statistics.mean(row_widths) if row_widths else 0.0,
                max_width=max(row_widths, default=0),
                mean_paths=statistics.mean(row_paths) if row_paths else 0.0,
                mean_parent=row_moments["mean_parent_degree"],
                size_biased=row_moments["size_biased_parent_branching"],
                max_p=row_moments["max_parent_degree"],
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / f"distribution_support_bounds_{safe_name}_{timestamp}.jsonl"
    summary_path = output_dir / f"distribution_support_bounds_summary_{safe_name}_{timestamp}.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="all", help="Comma-separated fixture names or all. Ignored when --edge-file is set.")
    parser.add_argument("--edge-file", type=Path, help="Optional TSV edge list with child<TAB>parent rows.")
    parser.add_argument("--graph-name", help="Graph label used in records and output filenames.")
    parser.add_argument("--root", help="Root node. Defaults to R for fixtures and Category:Articles for edge files.")
    parser.add_argument("--targets", help="Comma-separated target nodes for edge-file mode.")
    parser.add_argument("--targets-file", type=Path, help="Optional newline-delimited target list for edge-file mode.")
    parser.add_argument("--max-target-depth", type=int, help="Select reachable edge-file targets within this parent distance from root.")
    parser.add_argument("--target-limit", type=int, help="Limit selected edge-file targets after sorting/filtering.")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--narrow-width", type=int, default=2, help="Support width considered cheap for exact histogram planning.")
    parser.add_argument("--wide-width", type=int, default=10, help="Support width considered wide enough to prefer a boundary or fit.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if bounds differ from exact histogram support.")
    args = parser.parse_args(argv)

    budgets = parse_int_list(args.budgets)
    records = []

    if args.edge_file:
        root = args.root or SIMPLEWIKI_ROOT
        parents = load_parent_edges_tsv(args.edge_file)
        graph_name = args.graph_name or args.edge_file.stem
        targets = select_file_targets(
            parents,
            root,
            args.targets,
            args.targets_file,
            args.max_target_depth,
            args.target_limit,
        )
        records.extend(
            run_graph_support_bounds(
                graph_name,
                graph_name,
                parents,
                targets,
                budgets,
                root,
                args.narrow_width,
                args.wide_width,
            )
        )
    else:
        root = args.root or ROOT
        if root != ROOT:
            raise SystemExit("--root is only supported with --edge-file; fixtures use root R")
        fixture_names = sorted(FIXTURES) if args.fixtures == "all" else [name.strip() for name in args.fixtures.split(",")]
        for fixture_name in fixture_names:
            if fixture_name not in FIXTURES:
                raise SystemExit(f"unknown fixture: {fixture_name}")
            records.extend(
                run_fixture_support_bounds(
                    fixture_name,
                    FIXTURES[fixture_name],
                    budgets,
                    args.narrow_width,
                    args.wide_width,
                )
            )
        graph_name = args.graph_name or "tiny_fixture"

    summary = summarize(records)
    print(summary, end="")
    if args.output_dir:
        jsonl_path, summary_path = write_outputs(records, summary, args.output_dir, graph_name)
        print(f"\nwrote {jsonl_path}")
        print(f"wrote {summary_path}")

    failures = [
        r
        for r in records
        if r.get("record_type") == "target_bounds"
        and (r.get("exact_histogram_error") is not None or not r.get("bounds_match_histogram", True))
    ]
    if failures and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
