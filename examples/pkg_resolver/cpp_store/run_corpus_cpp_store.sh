#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_corpus_cpp_store.sh -- drive the P3 contract corpus (dump_store_data.pl,
# the 51-scenario cases.jsonl) through the C++ WAM store-backed build (D43
# indexed seek stores) and compare every result to the SWI store oracle
# (store_diff_runner.pl over the same cases). C++ lane of
# examples/pkg_resolver/rust_store/run_corpus_rust_store.sh.
#
#   bash examples/pkg_resolver/cpp_store/run_corpus_cpp_store.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.corpus_out"
BIN="$HERE/uw_resolve_wam_cpp_store/cpp/diff_uwresolve_store"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi

STORE_DIR="$STORE" UW_STORE_BACKEND="$BACKEND" bash "$HERE/build.sh"

echo "== SWI store oracle (corpus) =="
STORE_DIR="$STORE" swipl -q -g main -t halt "$HERE/../store_diff_runner.pl" \
  < "$STORE/cases.jsonl" > "$OUT/swi.jsonl"

echo "== cpp_store (corpus) =="
"$BIN" < "$STORE/cases.jsonl" > "$OUT/cpp_store.jsonl"

echo "== comparing (C++ store vs SWI store oracle) =="
node "$HERE/../compare_jsonl.mjs" "$STORE/cases.jsonl" "$OUT/swi.jsonl" "$OUT/cpp_store.jsonl"
