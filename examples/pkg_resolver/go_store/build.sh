#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- dump corpus catalogs to P/2 JSONL, index with D43, compile
# resolver.pl + resolver_store.pl through wam_go against those stores, then
# `go build` the JSON shim. Go lane of examples/pkg_resolver/wamjs_store/build.sh.
#
#   bash examples/pkg_resolver/go_store/build.sh
#   STORE_DIR=path bash examples/pkg_resolver/go_store/build.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/go_store/build.sh
#
# Default backend is indexed(Prefix). lmdb(Dir) is opt-in and fails LOUDLY
# on the Go lane (no compatible reader / no repo dependency), never a silent
# swap to indexed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../resolver_store.pl"
OUT="${WAM_OUT:-$HERE}"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"

mkdir -p "$STORE" "$OUT"
cd "$ROOT"

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi

case "$BACKEND" in
  indexed)
    if [[ ! -f "$STORE/pkg.data" ]]; then
      bash examples/pkg_resolver/store/build_stores.sh "$STORE"
    fi
    ;;
  lmdb)
    # shellcheck source=../store/ensure_lmdb.sh
    source examples/pkg_resolver/store/ensure_lmdb.sh
    uw_require_lmdb
    if [[ ! -d "$STORE/lmdb/pkg" ]]; then
      bash examples/pkg_resolver/store/build_lmdb_stores.sh "$STORE"
    fi
    ;;
  *)
    echo "go_store/build.sh: unknown UW_STORE_BACKEND=$BACKEND (indexed|lmdb)" >&2
    exit 2
    ;;
esac

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT" "$STORE" "$BACKEND"

cd "$OUT"
go build -o uwresolvestore ./cmd/uwresolvestore
echo "go_store/build.sh: backend=$BACKEND go build clean -> $OUT/uwresolvestore"
