#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Compare histogram materialization regimes across LMDB graph probes.

The report joins two evidence sources:

* root-conditioned parent-branching profiles, which estimate the prior search
  growth seen inside a common-root cone; and
* boundary-cache benchmark rows, which show the observed support width, path
  mass, recurrence cost, and cache-hit shape for materialized histograms.

This keeps the "50 point" policy as a storage/representation cap.  The report
then checks what the measured histograms actually cost, because low-branching
cones can stay exact and sparse far below that cap.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lmdb_parent_histogram_benchmark import safe_graph_name


def mean(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else statistics.mean(values)


def max_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else max(values)


def percentile(values, pct):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((float(pct) / 100.0) * (len(values) - 1))))
    return values[index]


def ratio(numerator, denominator):
    denominator = float(denominator or 0.0)
    if denominator == 0.0:
        return None
    return float(numerator or 0.0) / denominator


def pct(predicate, rows):
    rows = list(rows)
    if not rows:
        return None
    return 100.0 * sum(1 for row in rows if predicate(row)) / len(rows)


def format_value(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "{:.{}f}".format(value, digits)


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise ValueError("glob matched no files: {}".format(pattern))
        paths.extend(Path(match) for match in matches)
    return paths


def load_jsonl_records(paths):
    records = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError("{}:{}: invalid JSONL row: {}".format(path, line_number, exc)) from exc
                row.setdefault("_source_path", str(path))
                records.append(row)
    return records


def parse_labeled_paths(values):
    """Parse repeated LABEL=PATH arguments into label -> [Path]."""
    grouped = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("expected LABEL=PATH, got {!r}".format(value))
        label, pattern = value.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label or not pattern:
            raise ValueError("expected non-empty LABEL=PATH, got {!r}".format(value))
        grouped.setdefault(label, []).extend(expand_paths([pattern]))
    return grouped


def degree_summary(row, scope):
    stats = row.get(scope, {})
    return {
        "nodes": stats.get("nodes"),
        "mean_parent_degree": stats.get("mean_parent_degree"),
        "size_biased_parent_branching": stats.get("size_biased_parent_branching"),
        "mean_excess": stats.get("mean_excess"),
        "max_parent_degree": stats.get("max_parent_degree"),
        "p95_parent_degree": stats.get("p95_parent_degree"),
        "p99_parent_degree": stats.get("p99_parent_degree"),
        "zero_parent_nodes": stats.get("zero_parent_nodes"),
    }


def summarize_profile(label, paths, records):
    selection = next((row for row in records if row.get("record_type") == "root_conditioned_branching_selection"), {})
    overall = next((row for row in records if row.get("record_type") == "root_conditioned_branching_overall"), None)
    if overall is None:
        raise ValueError("{}: no root_conditioned_branching_overall row".format(label))
    depth_rows = [
        row for row in records
        if row.get("record_type") == "root_conditioned_branching_depth_bucket"
    ]
    depth_rows = sorted(depth_rows, key=lambda row: int(row.get("child_depth", 0)))
    return {
        "label": label,
        "source_paths": [str(path) for path in paths],
        "graph": selection.get("graph", overall.get("graph")),
        "root": selection.get("root", overall.get("root")),
        "max_child_depth": selection.get("max_child_depth"),
        "max_nodes": selection.get("max_nodes"),
        "retained_nodes": selection.get("retained_nodes", overall.get("nodes")),
        "max_observed_child_depth": selection.get("max_observed_child_depth"),
        "truncated_by_depth": selection.get("truncated_by_depth"),
        "truncated_by_nodes": selection.get("truncated_by_nodes"),
        "full_parent_degree": degree_summary(overall, "full_parent_degree"),
        "root_conditioned_parent_degree": degree_summary(overall, "root_conditioned_parent_degree"),
        "outside_root_parent_degree": degree_summary(overall, "outside_root_parent_degree"),
        "depth_rows": [
            {
                "child_depth": row.get("child_depth"),
                "nodes": row.get("nodes"),
                "full_parent_degree": degree_summary(row, "full_parent_degree"),
                "root_conditioned_parent_degree": degree_summary(row, "root_conditioned_parent_degree"),
                "mean_outside_parent_fraction": row.get("mean_outside_parent_fraction"),
            }
            for row in depth_rows
        ],
    }


def histogram_regime(max_effective_bins, mean_paths, point_cap):
    if max_effective_bins is None:
        return "no_histogram_rows"
    if max_effective_bins <= point_cap and (mean_paths is None or mean_paths <= point_cap):
        return "exact_sparse_low_mass"
    if max_effective_bins <= point_cap:
        return "exact_sparse_high_mass"
    return "compression_pressure"


def summarize_cache(label, paths, records, point_cap):
    selections = [row for row in records if row.get("record_type") == "boundary_cache_selection"]
    entries = [row for row in records if row.get("record_type") == "boundary_cache_entry"]
    comparisons = [row for row in records if row.get("record_type") == "boundary_cache_comparison"]
    selection = selections[0] if selections else {}
    nonempty_entries = [row for row in entries if int(row.get("path_count") or 0) > 0]
    exact_rows = [row for row in entries if row.get("cached")]
    parametric_rows = [row for row in entries if row.get("parametric_cached")]
    materialized_rows = exact_rows or nonempty_entries
    support_bins = [int(row.get("support_bins") or 0) for row in materialized_rows]
    effective_bins = [int(row.get("effective_support_bins_after_trim") or 0) for row in materialized_rows]
    path_counts = [int(row.get("path_count") or 0) for row in materialized_rows]
    recurrence_states = [
        row.get("recurrence_states_evaluated", row.get("nodes_expanded"))
        for row in materialized_rows
        if row.get("recurrence_states_evaluated", row.get("nodes_expanded")) is not None
    ]
    payload_bytes = [
        row.get("cache_payload_bytes", row.get("histogram_bytes"))
        for row in exact_rows
        if row.get("cache_payload_bytes", row.get("histogram_bytes")) is not None
    ]
    time_ratios = []
    for row in comparisons:
        time_ratio = ratio(row.get("cached_time_ns"), row.get("full_time_ns"))
        if time_ratio is not None:
            time_ratios.append(time_ratio)
    cache_hits = [int(row.get("cache_hits") or 0) for row in comparisons]
    histogram_hits = [int(row.get("histogram_cache_hits") or 0) for row in comparisons]
    parametric_hits = [int(row.get("parametric_cache_hits") or 0) for row in comparisons]
    full_paths = [int(row.get("full_path_count") or 0) for row in comparisons]
    cached_paths = [int(row.get("cached_path_count") or 0) for row in comparisons]
    l1_errors = [float(row.get("l1_error") or 0.0) for row in comparisons]
    cdf_errors = [float(row.get("max_cdf_error") or 0.0) for row in comparisons]
    max_effective = max_or_none(effective_bins)
    mean_path_count = mean(path_counts)
    return {
        "label": label,
        "source_paths": [str(path) for path in paths],
        "graph": selection.get("graph"),
        "root": selection.get("root"),
        "boundary_counts": selection.get("boundary_counts", {}),
        "target_counts": selection.get("target_counts", {}),
        "boundary_nodes": selection.get("boundary_nodes"),
        "selected_boundary_nodes": selection.get("selected_boundary_nodes"),
        "target_ancestor_boundary_nodes_added": selection.get("target_ancestor_boundary_nodes_added"),
        "cached_boundary_nodes": selection.get("cached_boundary_nodes"),
        "parametric_boundary_nodes": selection.get("parametric_boundary_nodes"),
        "targets": selection.get("targets"),
        "budgets": selection.get("budgets", []),
        "boundary_budget": selection.get("boundary_budget"),
        "boundary_builder": selection.get("boundary_builder"),
        "admission_policy": selection.get("admission_policy"),
        "entries": len(entries),
        "nonempty_entries": len(nonempty_entries),
        "exact_histogram_entries": len(exact_rows),
        "parametric_entries": len(parametric_rows),
        "mean_support_bins": mean(support_bins),
        "p95_support_bins": percentile(support_bins, 95),
        "max_support_bins": max_or_none(support_bins),
        "mean_effective_support_bins": mean(effective_bins),
        "p95_effective_support_bins": percentile(effective_bins, 95),
        "max_effective_support_bins": max_effective,
        "pct_support_bins_le_point_cap": pct(lambda row: int(row.get("support_bins") or 0) <= point_cap, materialized_rows),
        "pct_effective_bins_le_point_cap": pct(lambda row: int(row.get("effective_support_bins_after_trim") or 0) <= point_cap, materialized_rows),
        "mean_path_count": mean_path_count,
        "p95_path_count": percentile(path_counts, 95),
        "max_path_count": max_or_none(path_counts),
        "mean_recurrence_states": mean(recurrence_states),
        "p95_recurrence_states": percentile(recurrence_states, 95),
        "max_recurrence_states": max_or_none(recurrence_states),
        "mean_payload_bytes": mean(payload_bytes),
        "max_payload_bytes": max_or_none(payload_bytes),
        "comparison_rows": len(comparisons),
        "mean_cache_hits": mean(cache_hits),
        "mean_histogram_cache_hits": mean(histogram_hits),
        "mean_parametric_cache_hits": mean(parametric_hits),
        "positive_cache_hit_rows": sum(1 for value in cache_hits if value > 0),
        "mean_full_path_count": mean(full_paths),
        "mean_cached_path_count": mean(cached_paths),
        "mean_l1_error": mean(l1_errors),
        "max_l1_error": max_or_none(l1_errors),
        "mean_max_cdf_error": mean(cdf_errors),
        "max_cdf_error": max_or_none(cdf_errors),
        "mean_time_ratio": mean(time_ratios),
        "p95_time_ratio": percentile(time_ratios, 95),
        "measured_cache_faster": None if not time_ratios else mean(time_ratios) < 1.0,
        "histogram_regime": histogram_regime(max_effective, mean_path_count, point_cap),
    }


def summarize_inputs(profile_specs, cache_specs, point_cap):
    profiles = [
        summarize_profile(label, paths, load_jsonl_records(paths))
        for label, paths in sorted(profile_specs.items())
    ]
    caches = [
        summarize_cache(label, paths, load_jsonl_records(paths), point_cap)
        for label, paths in sorted(cache_specs.items())
    ]
    return {
        "point_cap": point_cap,
        "profiles": profiles,
        "caches": caches,
    }


def interpretation_lines(report):
    lines = [
        "- The point cap is a representation upper bound.  A dataset whose measured effective support is far below the cap can stay in exact histogram form without paying for all points.",
    ]
    for cache in report["caches"]:
        label = cache["label"]
        regime = cache["histogram_regime"]
        mean_bins = cache["mean_effective_support_bins"]
        max_bins = cache["max_effective_support_bins"]
        mean_paths = cache["mean_path_count"]
        if regime == "exact_sparse_low_mass":
            lines.append(
                "- `{}` is exact-sparse and low-mass in this probe: mean effective bins {}, max effective bins {}, mean path mass {}.".format(
                    label,
                    format_value(mean_bins),
                    format_value(max_bins),
                    format_value(mean_paths),
                )
            )
        elif regime == "exact_sparse_high_mass":
            lines.append(
                "- `{}` has compact support but larger path mass: mean effective bins {}, max effective bins {}, mean path mass {}.  This is the regime where recurrence materialization can replace many enumerated paths with a small stored state.".format(
                    label,
                    format_value(mean_bins),
                    format_value(max_bins),
                    format_value(mean_paths),
                )
            )
        elif regime == "compression_pressure":
            lines.append(
                "- `{}` exceeds the point cap in this probe, so admission should consider tail trimming or closed-form compression before storing exact histograms.".format(label)
            )
    for profile in report["profiles"]:
        root_stats = profile["root_conditioned_parent_degree"]
        full_stats = profile["full_parent_degree"]
        root_b = root_stats.get("size_biased_parent_branching")
        full_b = full_stats.get("size_biased_parent_branching")
        if profile.get("truncated_by_depth") or profile.get("truncated_by_nodes"):
            lines.append(
                "- `{}` is a constrained profile, so use it as smoke evidence rather than a full-root-cone characterization.".format(
                    profile["label"],
                )
            )
        if root_b is not None and full_b is not None and root_b < full_b:
            lines.append(
                "- `{}` has lower root-conditioned branching than raw parent branching ({} vs {}).  Ancestor-cone planning should prefer the root-conditioned prior when estimating materialization depth.".format(
                    profile["label"],
                    format_value(root_b),
                    format_value(full_b),
                )
            )
    return lines


def markdown_report(report, graph_name):
    lines = [
        "# LMDB Materialization Regime Comparison",
        "",
        "Graph: `{}`".format(graph_name),
        "",
        "Point cap: `{}`".format(report["point_cap"]),
        "",
        "## Branching Profiles",
        "",
        "| dataset | graph | root | retained_nodes | max_observed_depth | truncated | root_conditioned_b | raw_b | mean_root_p | mean_raw_p | max_root_p | max_raw_p |",
        "|---------|-------|-----:|---------------:|-------------------:|-----------|-------------------:|------:|------------:|-----------:|-----------:|----------:|",
    ]
    for profile in report["profiles"]:
        root_stats = profile["root_conditioned_parent_degree"]
        full_stats = profile["full_parent_degree"]
        truncated = bool(profile.get("truncated_by_depth")) or bool(profile.get("truncated_by_nodes"))
        lines.append(
            "| {label} | `{graph}` | {root} | {nodes} | {depth} | {truncated} | {root_b} | {raw_b} | {root_mean} | {raw_mean} | {root_max} | {raw_max} |".format(
                label=profile["label"],
                graph=profile.get("graph"),
                root=profile.get("root"),
                nodes=format_value(profile.get("retained_nodes"), 0),
                depth=format_value(profile.get("max_observed_child_depth"), 0),
                truncated="yes" if truncated else "no",
                root_b=format_value(root_stats.get("size_biased_parent_branching")),
                raw_b=format_value(full_stats.get("size_biased_parent_branching")),
                root_mean=format_value(root_stats.get("mean_parent_degree")),
                raw_mean=format_value(full_stats.get("mean_parent_degree")),
                root_max=format_value(root_stats.get("max_parent_degree"), 0),
                raw_max=format_value(full_stats.get("max_parent_degree"), 0),
            )
        )
    lines.extend([
        "",
        "## Depth Buckets",
        "",
        "| dataset | child_depth | nodes | root_conditioned_b | raw_b | mean_root_p | mean_raw_p | mean_outside_parent_fraction |",
        "|---------|------------:|------:|-------------------:|------:|------------:|-----------:|-----------------------------:|",
    ])
    for profile in report["profiles"]:
        for row in profile["depth_rows"]:
            root_stats = row["root_conditioned_parent_degree"]
            full_stats = row["full_parent_degree"]
            lines.append(
                "| {label} | {depth} | {nodes} | {root_b} | {raw_b} | {root_mean} | {raw_mean} | {outside} |".format(
                    label=profile["label"],
                    depth=row.get("child_depth"),
                    nodes=format_value(row.get("nodes"), 0),
                    root_b=format_value(root_stats.get("size_biased_parent_branching")),
                    raw_b=format_value(full_stats.get("size_biased_parent_branching")),
                    root_mean=format_value(root_stats.get("mean_parent_degree")),
                    raw_mean=format_value(full_stats.get("mean_parent_degree")),
                    outside=format_value(row.get("mean_outside_parent_fraction")),
                )
            )
    lines.extend([
        "",
        "## Boundary Histograms",
        "",
        "| dataset | graph | entries | exact | parametric | mean_eff_bins | max_eff_bins | pct_eff_bins_le_cap | mean_paths | max_paths | mean_states | max_states | mean_payload_bytes | regime |",
        "|---------|-------|--------:|------:|-----------:|--------------:|-------------:|--------------------:|-----------:|----------:|------------:|-----------:|-------------------:|--------|",
    ])
    for cache in report["caches"]:
        lines.append(
            "| {label} | `{graph}` | {entries} | {exact} | {parametric} | {mean_bins} | {max_bins} | {pct_bins} | {mean_paths} | {max_paths} | {mean_states} | {max_states} | {payload} | `{regime}` |".format(
                label=cache["label"],
                graph=cache.get("graph"),
                entries=cache["entries"],
                exact=cache["exact_histogram_entries"],
                parametric=cache["parametric_entries"],
                mean_bins=format_value(cache["mean_effective_support_bins"]),
                max_bins=format_value(cache["max_effective_support_bins"], 0),
                pct_bins=format_value(cache["pct_effective_bins_le_point_cap"]),
                mean_paths=format_value(cache["mean_path_count"]),
                max_paths=format_value(cache["max_path_count"], 0),
                mean_states=format_value(cache["mean_recurrence_states"]),
                max_states=format_value(cache["max_recurrence_states"], 0),
                payload=format_value(cache["mean_payload_bytes"]),
                regime=cache["histogram_regime"],
            )
        )
    lines.extend([
        "",
        "## Cache Search Shape",
        "",
        "| dataset | targets | budgets | comparison_rows | mean_cache_hits | positive_hit_rows | mean_full_paths | mean_time_ratio | measured_cache_faster | mean_l1 | mean_cdf |",
        "|---------|--------:|---------|----------------:|----------------:|------------------:|----------------:|----------------:|-----------------------|--------:|---------:|",
    ])
    for cache in report["caches"]:
        lines.append(
            "| {label} | {targets} | `{budgets}` | {rows} | {hits} | {hit_rows} | {full_paths} | {time_ratio} | {faster} | {l1} | {cdf} |".format(
                label=cache["label"],
                targets=format_value(cache.get("targets"), 0),
                budgets=",".join(str(value) for value in cache.get("budgets") or []),
                rows=cache["comparison_rows"],
                hits=format_value(cache["mean_cache_hits"]),
                hit_rows=cache["positive_cache_hit_rows"],
                full_paths=format_value(cache["mean_full_path_count"]),
                time_ratio=format_value(cache["mean_time_ratio"]),
                faster=format_value(cache["measured_cache_faster"]),
                l1=format_value(cache["mean_l1_error"], 6),
                cdf=format_value(cache["mean_max_cdf_error"], 6),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    lines.extend(interpretation_lines(report))
    lines.extend([
        "",
        "## Source Files",
        "",
    ])
    for profile in report["profiles"]:
        for path in profile["source_paths"]:
            lines.append("- profile `{}`: `{}`".format(profile["label"], path))
    for cache in report["caches"]:
        for path in cache["source_paths"]:
            lines.append("- cache `{}`: `{}`".format(cache["label"], path))
    return "\n".join(lines) + "\n"


def write_outputs(report, markdown, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    json_path = output_dir / "lmdb_materialization_regime_comparison_{}_{}.json".format(safe_name, timestamp)
    markdown_path = output_dir / "lmdb_materialization_regime_comparison_{}_{}.md".format(safe_name, timestamp)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-name", default="lmdb_materialization_regime_comparison", help="Graph label used in output filenames.")
    parser.add_argument("--profile", action="append", default=[], metavar="LABEL=PATH", help="Root-conditioned profile JSONL. May be repeated.")
    parser.add_argument("--cache", action="append", default=[], metavar="LABEL=PATH", help="Boundary-cache benchmark JSONL. May be repeated.")
    parser.add_argument("--point-cap", type=int, default=50, help="Maximum points/bins for exact histogram representation before compression pressure.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory for JSON and markdown.")
    args = parser.parse_args(argv)
    if args.point_cap <= 0:
        raise SystemExit("--point-cap must be positive")
    if not args.profile and not args.cache:
        raise SystemExit("at least one --profile or --cache input is required")

    profile_specs = parse_labeled_paths(args.profile)
    cache_specs = parse_labeled_paths(args.cache)
    report = summarize_inputs(profile_specs, cache_specs, args.point_cap)
    markdown = markdown_report(report, args.graph_name)
    if args.output_dir:
        json_path, markdown_path = write_outputs(report, markdown, args.output_dir, args.graph_name)
        print(markdown, end="")
        print("json={}".format(json_path))
        print("summary={}".format(markdown_path))
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
