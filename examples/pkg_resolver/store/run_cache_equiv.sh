#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_cache_equiv.sh -- lazy / cached / eager equivalence (same results,
# same order). Fleet lmdb_materialisation tiers; UW_FACT_CACHE alias not
# needed here (the harness calls configure_lmdb_materialisation).
#
#   bash examples/pkg_resolver/store/run_cache_equiv.sh indexed
#   bash examples/pkg_resolver/store/run_cache_equiv.sh lmdb
#
# indexed is ungated. lmdb requires the npm package (loud error if
# requested and missing).

set -euo pipefail

BACKEND="${1:-indexed}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STORE="${STORE_DIR:-$HERE/.out/corpus}"
cd "$ROOT"

case "$BACKEND" in
  indexed)
    WAM="$ROOT/examples/pkg_resolver/wamjs_store"
    if [[ ! -f "$WAM/js/generated_program.js" ]] ||
       ! grep -q 'configure_lmdb_materialisation' "$WAM/js/wam_runtime.js"; then
      UW_STORE_BACKEND=indexed STORE_DIR="$STORE" bash "$WAM/build.sh"
    fi
    ;;
  lmdb)
    # shellcheck source=ensure_lmdb.sh
    source "$HERE/ensure_lmdb.sh"
    uw_require_lmdb
    WAM="${WAM_OUT:-$HERE/.out/corpus_wam_lmdb}"
    mkdir -p "$WAM"
    if [[ ! -f "$WAM/js/generated_program.js" ]] ||
       ! grep -q 'kind: "lmdb"' "$WAM/js/generated_program.js" ||
       ! grep -q 'configure_lmdb_materialisation' "$WAM/js/wam_runtime.js" 2>/dev/null; then
      WAM_OUT="$WAM" UW_STORE_BACKEND=lmdb STORE_DIR="$STORE" bash \
        "$ROOT/examples/pkg_resolver/wamjs_store/build.sh"
      cp "$ROOT/examples/pkg_resolver/wamjs_store/resolver_store.mjs" "$WAM/resolver_store.mjs"
    fi
    ;;
  *)
    echo "usage: run_cache_equiv.sh indexed|lmdb" >&2
    exit 2
    ;;
esac

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi

echo "== cache equivalence backend=$BACKEND =="
node "$HERE/test_cache_equiv.mjs" "$WAM" "$STORE/cases.jsonl"
echo "cache_equiv $BACKEND ok"
