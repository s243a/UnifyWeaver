#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- dump corpus catalogs to P/2 JSONL, index with D43, compile
# resolver_store.pl against those stores.
#
#   bash examples/pkg_resolver/wamjs_store/build.sh
#   STORE_DIR=path bash examples/pkg_resolver/wamjs_store/build.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/wamjs_store/build.sh
#
# Default backend is indexed(Prefix). lmdb(Dir) is opt-in: missing npm
# `lmdb` is a loud error, never a silent swap to indexed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../resolver_store.pl"
OUT="${WAM_OUT:-$HERE}"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"

mkdir -p "$STORE" "$OUT"
cd "$ROOT"
swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"

case "$BACKEND" in
  indexed)
    bash examples/pkg_resolver/store/build_stores.sh "$STORE"
    ;;
  lmdb)
    # shellcheck source=../store/ensure_lmdb.sh
    source examples/pkg_resolver/store/ensure_lmdb.sh
    uw_require_lmdb
    bash examples/pkg_resolver/store/build_lmdb_stores.sh "$STORE"
    ;;
  *)
    echo "wamjs_store/build.sh: unknown UW_STORE_BACKEND=$BACKEND (indexed|lmdb)" >&2
    exit 2
    ;;
esac

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT" "$STORE" "$BACKEND"

node --check "$OUT/js/generated_program.js"
node --check "$OUT/js/wam_runtime.js"
echo "wamjs_store/build.sh: backend=$BACKEND node --check clean -> $OUT/js/generated_program.js"
