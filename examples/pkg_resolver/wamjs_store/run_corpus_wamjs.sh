#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
#   bash examples/pkg_resolver/wamjs_store/run_corpus_wamjs.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/wamjs_store/run_corpus_wamjs.sh
#
# lmdb arm: loud error if the npm package is missing (never silently
# runs the indexed binary). NODE_PATH comes from ensure_lmdb.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.corpus_out"
WAM="$HERE"
mkdir -p "$OUT"

cd "$ROOT"

if [[ "$BACKEND" == "lmdb" ]]; then
  # shellcheck source=../store/ensure_lmdb.sh
  source examples/pkg_resolver/store/ensure_lmdb.sh
  uw_require_lmdb
  WAM="${WAM_OUT:-$HERE/../store/.out/corpus_wam_lmdb}"
  mkdir -p "$WAM"
  if [[ ! -f "$WAM/js/generated_program.js" ]] ||
     ! grep -q 'kind: "lmdb"' "$WAM/js/generated_program.js" ||
     ! grep -q 'configure_lmdb_materialisation' "$WAM/js/wam_runtime.js" 2>/dev/null; then
    WAM_OUT="$WAM" UW_STORE_BACKEND=lmdb STORE_DIR="$STORE" \
      bash "$HERE/build.sh"
    cp "$HERE/resolver_store.mjs" "$WAM/resolver_store.mjs"
    cp "$HERE/run_corpus.mjs" "$WAM/run_corpus.mjs"
  fi
elif [[ ! -f "$HERE/js/generated_program.js" ]] ||
     ! grep -q 'configure_lmdb_materialisation' "$HERE/js/wam_runtime.js" 2>/dev/null; then
  bash "$HERE/build.sh"
fi

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi
node "$WAM/run_corpus.mjs" "$STORE/cases.jsonl" "$OUT/wamjs.jsonl"
