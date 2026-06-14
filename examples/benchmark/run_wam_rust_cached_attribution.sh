#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_wam_rust_cached_attribution.sh — end-to-end cached-side runtime
# attribution for the WAM-Rust graph-search bench (PR #3120 Rust analog).
#
# Ingests a graph fixture into an lmdb_resident database, generates and builds
# both a `cached` crate and a `lazy` control crate, runs them with
# UW_WAM_CACHE_ATTRIBUTION=1, and prints the cache_attr_* report for each.
#
# Usage:
#   examples/benchmark/run_wam_rust_cached_attribution.sh <fixture_src_dir> [work_dir]
#
#   <fixture_src_dir>  dir with category_parent.tsv + article_category.tsv
#                      + root_categories.tsv (e.g. data/benchmark/10x)
#   [work_dir]         scratch dir for the lmdb fixture + generated crates
#                      (default: a fresh mktemp dir)
#
# Requires: swipl, cargo, python3 with the `lmdb` package.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE_SRC="${1:?usage: run_wam_rust_cached_attribution.sh <fixture_src_dir> [work_dir]}"
WORK_DIR="${2:-$(mktemp -d)}"
export LC_ALL="${LC_ALL:-C.UTF-8}" LANG="${LANG:-C.UTF-8}"

FIXTURE_SRC="$(cd "$FIXTURE_SRC" && pwd)"
EDGES=$(($(wc -l < "$FIXTURE_SRC/category_parent.tsv") - 1))
BENCH="$WORK_DIR/fixture"
mkdir -p "$BENCH"

echo "== ingest $FIXTURE_SRC ($EDGES edges) -> lmdb_resident =="
python3 "$REPO_ROOT/examples/benchmark/ingest_resident_lmdb_fixture.py" \
    "$FIXTURE_SRC" "$BENCH/lmdb_resident"
cp "$FIXTURE_SRC/article_category.tsv" "$BENCH/"   # root_ids.txt written by ingester

for mode in cached lazy; do
    OUT="$WORK_DIR/bench_$mode"
    echo "== generate + build ($mode) =="
    rm -rf "$OUT"
    swipl -q -s "$REPO_ROOT/examples/benchmark/generate_wam_rust_matrix_benchmark.pl" -- \
        "$REPO_ROOT/examples/benchmark/effective_distance.pl" \
        "$OUT" accumulated functions kernels_on cursor auto "$mode" "$EDGES" >/dev/null 2>&1
    ( cd "$OUT" && cargo build --release >/dev/null 2>&1 )

    echo "== run ($mode) with attribution =="
    UW_WAM_CACHE_ATTRIBUTION=1 "$OUT/target/release/bench" "$BENCH" \
        >"$WORK_DIR/out_$mode.tsv" 2>"$WORK_DIR/err_$mode.txt"
    grep -E 'total_ms|cache_attr_' "$WORK_DIR/err_$mode.txt" | sed 's/^/  /'
done

REF="$FIXTURE_SRC/reference_output.tsv"
if [[ -f "$REF" ]]; then
    if diff -q <(sort "$WORK_DIR/out_cached.tsv") <(sort "$REF") >/dev/null; then
        echo "== correctness: cached output EXACT MATCH vs reference_output.tsv =="
    else
        echo "== correctness: cached output DIFFERS from reference_output.tsv ==" >&2
        exit 1
    fi
fi

echo "work dir: $WORK_DIR"
