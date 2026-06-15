#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_wam_rust_cached_scaling_sweep.sh — cached-vs-lazy scaling sweep for the
# WAM-Rust graph-search bench. Builds the cached and lazy crates ONCE (they are
# fixture-independent; cache capacity is overridden at runtime), then for each
# benchmark fixture ingests an lmdb_resident database and measures query time,
# cache hit rate, and inner LMDB seek time. Prints one row per fixture.
#
# Single-threaded (WAM_THREADS=1) for deterministic attribution and a clean
# cache-vs-no-cache comparison (the cache benefit is lookup avoidance, not
# parallelism). Query time is the min over a few reps (least noisy).
#
# Usage:
#   examples/benchmark/run_wam_rust_cached_scaling_sweep.sh [fixture ...]
#     default fixtures: 300 1k 5k 10k (dirs under data/benchmark/)
#
# Requires: swipl, cargo, python3 with the `lmdb` package.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
export LC_ALL="${LC_ALL:-C.UTF-8}" LANG="${LANG:-C.UTF-8}"
REPS="${REPS:-3}"
CAP="${WAM_CACHE_CAPACITY:-400000}"
FIXTURES=("$@")
[ ${#FIXTURES[@]} -eq 0 ] && FIXTURES=(300 1k 5k 10k)

echo "== building cached + lazy crates (once) =="
declare -A BIN
for mode in cached lazy; do
    out="$WORK/crate_$mode"
    swipl -q -s "$REPO_ROOT/examples/benchmark/generate_wam_rust_matrix_benchmark.pl" -- \
        "$REPO_ROOT/examples/benchmark/effective_distance.pl" \
        "$out" accumulated functions kernels_on cursor auto "$mode" 25227 >/dev/null 2>&1
    ( cd "$out" && cargo build --release >/dev/null 2>&1 )
    BIN[$mode]="$out/target/release/bench"
done

getv() { grep -oP "$2=\K[0-9.]+" "$1" | head -1; }
min_query() { # <binary> <fixture-dir> -> min query_ms over REPS; leaves last run in $LASTLOG
    local best=99999999
    for _ in $(seq 1 "$REPS"); do
        UW_WAM_CACHE_ATTRIBUTION=1 WAM_THREADS=1 WAM_CACHE_CAPACITY="$CAP" \
            "$1" "$2" >/dev/null 2>"$LASTLOG"
        local q; q=$(getv "$LASTLOG" query_ms); [ -z "$q" ] && q=99999999
        awk "BEGIN{exit !($q<$best)}" && best=$q
    done
    echo "$best"
}

printf "\n%-6s %8s | %-9s %-8s | %-9s | %-7s %-10s %-10s | %s\n" \
    fixture edges cached_ms hit_rate lazy_ms speedup lookups misses correct
for fx in "${FIXTURES[@]}"; do
    src="$REPO_ROOT/data/benchmark/$fx"
    [ -f "$src/category_parent.tsv" ] || { echo "skip $fx (no fixture)"; continue; }
    edges=$(( $(wc -l < "$src/category_parent.tsv") - 1 ))
    fixdir="$WORK/fix_$fx"; mkdir -p "$fixdir"
    python3 "$REPO_ROOT/examples/benchmark/ingest_resident_lmdb_fixture.py" \
        "$src" "$fixdir/lmdb_resident" >/dev/null 2>&1
    cp "$src/article_category.tsv" "$fixdir/"

    LASTLOG="$WORK/c_$fx.log"; cms=$(min_query "${BIN[cached]}" "$fixdir")
    hit=$(getv "$LASTLOG" cache_attr_hit_rate)
    look=$(getv "$LASTLOG" cache_attr_lookups)
    miss=$(getv "$LASTLOG" cache_attr_misses)
    UW_WAM_CACHE_ATTRIBUTION=0 WAM_THREADS=1 "${BIN[cached]}" "$fixdir" >"$WORK/co_$fx.tsv" 2>/dev/null

    LASTLOG="$WORK/l_$fx.log"; lms=$(min_query "${BIN[lazy]}" "$fixdir")
    WAM_THREADS=1 "${BIN[lazy]}" "$fixdir" >"$WORK/lo_$fx.tsv" 2>/dev/null

    speedup=$(awk "BEGIN{ if ($cms>0) printf \"%.2fx\", $lms/$cms; else print \"n/a\" }")
    # Correctness invariant: caching must not change the answer.
    if diff -q <(sort "$WORK/co_$fx.tsv") <(sort "$WORK/lo_$fx.tsv") >/dev/null; then ck="c==l"; else ck="c!=l!"; fi

    printf "%-6s %8s | %-9s %-8s | %-9s | %-7s %-10s %-10s | %s\n" \
        "$fx" "$edges" "$cms" "$hit" "$lms" "$speedup" "$look" "$miss" "$ck"
done
echo
echo "work dir: $WORK"
