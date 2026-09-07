#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- dump corpus catalogs to P/2 JSONL, index with D43, compile
# resolver.pl + resolver_store.pl through wam_rust against those stores, then
# cargo build the JSON shim. Rust lane of examples/pkg_resolver/go_store/build.sh.
#
#   bash examples/pkg_resolver/rust_store/build.sh
#   STORE_DIR=path bash examples/pkg_resolver/rust_store/build.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/rust_store/build.sh
#
# Default backend is indexed(Prefix). lmdb(Dir) is opt-in and fails LOUDLY on
# the Rust lane (no compatible reader / no repo dependency), never a silent
# swap to indexed.
#
# The generated crate under uw_resolve_wam_store/ bakes the absolute store path
# into setup_foreign_predicates, so it is a per-build artifact (gitignored);
# re-running this script regenerates it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../resolver_store.pl"
PROJ="$HERE/uw_resolve_wam_store"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

mkdir -p "$STORE"
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
    echo "rust_store/build.sh: unknown UW_STORE_BACKEND=$BACKEND (indexed|lmdb)" >&2
    exit 2
    ;;
esac

echo "== compiling resolver.pl + resolver_store.pl through wam_rust ($BACKEND) =="
swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$PROJ" "$STORE" "$BACKEND" >/dev/null

# The shim is NOT generated: it is the hand-written edge (JSON <-> WAM terms).
# Copy it into the crate as a second binary target.
mkdir -p "$PROJ/src/bin/uw_resolve_store"
cp "$HERE/shim/main.rs" "$PROJ/src/bin/uw_resolve_store/main.rs"
cp "$HERE/shim/json.rs" "$PROJ/src/bin/uw_resolve_store/json.rs"

echo "== cargo build --release =="
( cd "$PROJ" && cargo build --release --bin uw_resolve_store 2>&1 | tail -3 )
echo "rust_store/build.sh: backend=$BACKEND binary -> $PROJ/target/release/uw_resolve_store"
