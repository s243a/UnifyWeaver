#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Benchmark exact parent-only distribution cache cutoffs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distribution_cache_support import (  # noqa: E402
    EXPONENT,
    FIXTURES,
    ROOT,
    SearchStats,
    build_cache,
    cache_bytes,
    cached_histogram_search,
    entropy,
    first_moment_cdf,
    full_exact_search,
    histogram_bytes,
    load_parent_edges_tsv,
    mass_cdf,
    reachable_nodes_by_parent_distance,
    support_sizes,
    weighted_power_cdf,
)


SIMPLEWIKI_ROOT = "Category:Articles"
DEFAULT_DEPTHS = [0, 1, 2, 3]
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


def timed_call(fn, *args, **kwargs):
    started = time.perf_counter_ns()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return result, elapsed_ms


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def safe_graph_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "graph"


def histogram_record(hist, budget):
    mass = mass_cdf(hist, budget)
    return {
        "result_histogram": hist,
        "result_mass": mass,
        "result_first_moment": first_moment_cdf(hist, budget),
        "result_weighted_power": weighted_power_cdf(hist, budget, EXPONENT),
        "result_entropy_if_available": entropy(hist, budget),
    }


def query_record(graph_name, graph_label, root, precompute_depth, budget, target, mode, hist, runtime_ms, stats, exact_hist):
    exact_mass = mass_cdf(exact_hist, budget)
    result_mass = mass_cdf(hist, budget)
    absolute_error = abs(result_mass - exact_mass)
    relative_error = 0.0 if exact_mass == 0 else absolute_error / exact_mass
    return {
        "record_type": "query",
        "graph": graph_name,
        "fixture": graph_label,
        "root": root,
        "target_node": target,
        "D_pre": precompute_depth,
        "B_search": budget,
        "mode": mode,
        "runtime_ms": runtime_ms,
        "nodes_expanded": stats.nodes_expanded,
        "edges_examined": stats.edges_examined,
        "paths_enumerated_or_aggregated": result_mass,
        "cache_hits": stats.cache_hits,
        "first_cache_hit_depth_from_target": min(stats.hit_depths_from_target, default=None),
        "remaining_budget_at_hit": list(stats.hit_remaining_budgets),
        "histogram_bins_scanned": stats.histogram_bins_scanned,
        "cumulative_basis_lookups": stats.cumulative_basis_lookups,
        "exact_result_reference": exact_hist,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "histogram_exact_match": hist == exact_hist,
        **histogram_record(hist, budget),
    }


def cache_record(graph_name, graph_label, root, precompute_depth, cache, build_runtime_ms):
    sizes = support_sizes(cache)
    total_mass = sum(sum(hist.values()) for hist in cache.values())
    raw_hist_bytes = sum(histogram_bytes(hist) for hist in cache.values())
    return {
        "record_type": "cache_build",
        "graph": graph_name,
        "fixture": graph_label,
        "root": root,
        "D_pre": precompute_depth,
        "eligible_nodes": len(cache),
        "cached_nodes": len(cache),
        "cache_bytes_raw_histogram": raw_hist_bytes,
        "cache_bytes_cumulative_bases": 0,
        "cache_bytes_total_estimate": cache_bytes(cache),
        "build_runtime_ms": build_runtime_ms,
        "max_support_size": max(sizes, default=0),
        "mean_support_size": statistics.mean(sizes) if sizes else 0.0,
        "p95_support_size": percentile(sizes, 95),
        "total_distribution_mass": total_mass,
    }


def run_graph_benchmark(graph_name, graph_label, parents, targets, depths, budgets, root):
    records = []
    for precompute_depth in depths:
        cache, build_runtime_ms = timed_call(build_cache, parents, precompute_depth, root=root)
        records.append(cache_record(graph_name, graph_label, root, precompute_depth, cache, build_runtime_ms))
        for budget in budgets:
            for target in targets:
                full_stats = SearchStats()
                full_hist, full_runtime_ms = timed_call(full_exact_search, target, parents, budget, root=root, stats=full_stats)
                records.append(
                    query_record(
                        graph_name,
                        graph_label,
                        root,
                        precompute_depth,
                        budget,
                        target,
                        "full_exact",
                        full_hist,
                        full_runtime_ms,
                        full_stats,
                        full_hist,
                    )
                )

                cached_stats = SearchStats()
                cached_hist, cached_runtime_ms = timed_call(
                    cached_histogram_search,
                    target,
                    parents,
                    budget,
                    cache,
                    root=root,
                    stats=cached_stats,
                )
                records.append(
                    query_record(
                        graph_name,
                        graph_label,
                        root,
                        precompute_depth,
                        budget,
                        target,
                        "cached_histogram",
                        cached_hist,
                        cached_runtime_ms,
                        cached_stats,
                        full_hist,
                    )
                )
    return records


def run_fixture_benchmark(fixture_name, fixture, depths, budgets):
    return run_graph_benchmark(
        "tiny_fixture",
        fixture_name,
        fixture["parents"],
        fixture["targets"],
        depths,
        budgets,
        ROOT,
    )


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
        raise SystemExit("no benchmark targets selected")
    return targets


def summarize(records):
    query_records = [r for r in records if r["record_type"] == "query"]
    cache_records = [r for r in records if r["record_type"] == "cache_build"]
    cache_by_key = {(r["fixture"], r["D_pre"]): r for r in cache_records}
    groups = {}
    for record in query_records:
        key = (record["fixture"], record["D_pre"], record["B_search"])
        groups.setdefault(key, []).append(record)

    lines = [
        "# Distribution Cache Benchmark Summary",
        "",
        "| fixture | D_pre | B_search | cache_nodes | cache_bytes | full_ms | cached_ms | speedup | hit_rate | exact_failures |",
        "|---------|-------|----------|-------------|-------------|---------|-----------|---------|----------|----------------|",
    ]
    for key in sorted(groups):
        fixture, precompute_depth, budget = key
        rows = groups[key]
        full_rows = [r for r in rows if r["mode"] == "full_exact"]
        cached_rows = [r for r in rows if r["mode"] == "cached_histogram"]
        full_ms = sum(r["runtime_ms"] for r in full_rows)
        cached_ms = sum(r["runtime_ms"] for r in cached_rows)
        speedup = 0.0 if cached_ms == 0 else full_ms / cached_ms
        hit_rate = 0.0 if not cached_rows else sum(1 for r in cached_rows if r["cache_hits"] > 0) / len(cached_rows)
        failures = sum(1 for r in cached_rows if not r["histogram_exact_match"])
        cache = cache_by_key[(fixture, precompute_depth)]
        lines.append(
            "| {fixture} | {depth} | {budget} | {nodes} | {bytes} | {full:.6f} | {cached:.6f} | {speedup:.3f} | {hit_rate:.3f} | {failures} |".format(
                fixture=fixture,
                depth=precompute_depth,
                budget=budget,
                nodes=cache["cached_nodes"],
                bytes=cache["cache_bytes_total_estimate"],
                full=full_ms,
                cached=cached_ms,
                speedup=speedup,
                hit_rate=hit_rate,
                failures=failures,
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / f"distribution_cache_benchmark_{safe_name}_{timestamp}.jsonl"
    summary_path = output_dir / f"distribution_cache_summary_{safe_name}_{timestamp}.md"
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
    parser.add_argument("--precompute-depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if cached histograms differ.")
    args = parser.parse_args(argv)

    depths = parse_int_list(args.precompute_depths)
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
        records.extend(run_graph_benchmark(graph_name, graph_name, parents, targets, depths, budgets, root))
    else:
        root = args.root or ROOT
        if root != ROOT:
            raise SystemExit("--root is only supported with --edge-file; fixtures use root R")
        fixture_names = sorted(FIXTURES) if args.fixtures == "all" else [name.strip() for name in args.fixtures.split(",")]
        graph_name = args.graph_name or "tiny_fixture"
        for fixture_name in fixture_names:
            if fixture_name not in FIXTURES:
                raise SystemExit(f"unknown fixture: {fixture_name}")
            records.extend(run_fixture_benchmark(fixture_name, FIXTURES[fixture_name], depths, budgets))

    summary = summarize(records)
    print(summary, end="")
    if args.output_dir:
        jsonl_path, summary_path = write_outputs(records, summary, args.output_dir, graph_name)
        print(f"\nwrote {jsonl_path}")
        print(f"wrote {summary_path}")

    failures = [r for r in records if r.get("record_type") == "query" and not r.get("histogram_exact_match", True)]
    if failures and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
