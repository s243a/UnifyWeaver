#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Benchmark exact parent-only distribution cache cutoffs on tiny fixtures."""

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

from tools.distribution_cache_support import (  # noqa: E402
    EXPONENT,
    FIXTURES,
    SearchStats,
    build_cache,
    cache_bytes,
    cached_histogram_search,
    entropy,
    first_moment_cdf,
    full_exact_search,
    histogram_bytes,
    mass_cdf,
    support_sizes,
    weighted_power_cdf,
)


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


def histogram_record(hist, budget):
    mass = mass_cdf(hist, budget)
    return {
        "result_histogram": hist,
        "result_mass": mass,
        "result_first_moment": first_moment_cdf(hist, budget),
        "result_weighted_power": weighted_power_cdf(hist, budget, EXPONENT),
        "result_entropy_if_available": entropy(hist, budget),
    }


def query_record(fixture_name, precompute_depth, budget, target, mode, hist, runtime_ms, stats, exact_hist):
    exact_mass = mass_cdf(exact_hist, budget)
    result_mass = mass_cdf(hist, budget)
    absolute_error = abs(result_mass - exact_mass)
    relative_error = 0.0 if exact_mass == 0 else absolute_error / exact_mass
    return {
        "record_type": "query",
        "graph": "tiny_fixture",
        "fixture": fixture_name,
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


def cache_record(fixture_name, precompute_depth, cache, build_runtime_ms):
    sizes = support_sizes(cache)
    total_mass = sum(sum(hist.values()) for hist in cache.values())
    raw_hist_bytes = sum(histogram_bytes(hist) for hist in cache.values())
    return {
        "record_type": "cache_build",
        "graph": "tiny_fixture",
        "fixture": fixture_name,
        "root": "R",
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


def run_fixture_benchmark(fixture_name, fixture, depths, budgets):
    parents = fixture["parents"]
    targets = fixture["targets"]
    records = []
    for precompute_depth in depths:
        cache, build_runtime_ms = timed_call(build_cache, parents, precompute_depth)
        records.append(cache_record(fixture_name, precompute_depth, cache, build_runtime_ms))
        for budget in budgets:
            for target in targets:
                full_stats = SearchStats()
                full_hist, full_runtime_ms = timed_call(full_exact_search, target, parents, budget, stats=full_stats)
                records.append(
                    query_record(
                        fixture_name,
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
                    stats=cached_stats,
                )
                records.append(
                    query_record(
                        fixture_name,
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
    jsonl_path = output_dir / f"distribution_cache_benchmark_{graph_name}_{timestamp}.jsonl"
    summary_path = output_dir / f"distribution_cache_summary_{graph_name}_{timestamp}.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="all", help="Comma-separated fixture names or 'all'.")
    parser.add_argument("--precompute-depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if cached histograms differ.")
    args = parser.parse_args(argv)

    fixture_names = sorted(FIXTURES) if args.fixtures == "all" else [name.strip() for name in args.fixtures.split(",")]
    depths = parse_int_list(args.precompute_depths)
    budgets = parse_int_list(args.budgets)

    records = []
    for fixture_name in fixture_names:
        if fixture_name not in FIXTURES:
            raise SystemExit(f"unknown fixture: {fixture_name}")
        records.extend(run_fixture_benchmark(fixture_name, FIXTURES[fixture_name], depths, budgets))

    summary = summarize(records)
    print(summary, end="")
    if args.output_dir:
        jsonl_path, summary_path = write_outputs(records, summary, args.output_dir, "tiny_fixture")
        print(f"\nwrote {jsonl_path}")
        print(f"wrote {summary_path}")

    failures = [r for r in records if r.get("record_type") == "query" and not r.get("histogram_exact_match", True)]
    if failures and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
