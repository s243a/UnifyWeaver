#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_differential_rust_store.sh -- >=500 seeded cases against the 5k catalog
# (store/gen_scale_catalog.mjs, seed 0xc0ffee01); SWI store adapter vs the Rust
# WAM store-backed build, 0 divergences on all 10 queries. SWI reads the same
# JSONL store dump (per D48). Rust lane of
# examples/pkg_resolver/go_store/run_differential_go_store.sh.
#
#   bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
#
# Default backend is indexed. lmdb is opt-in and fails loudly on the Rust lane.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/../store/.out/scale"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.diff_out"
BIN="$HERE/uw_resolve_wam_store/target/release/uw_resolve_store"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

mkdir -p "$SCALE" "$OUT"
cd "$ROOT"

echo "== generating 5k catalog + cases =="
node "$HERE/../store/gen_scale_catalog.mjs" "$SCALE"
if [[ ! -f "$SCALE/pkg.jsonl" ]]; then
  node "$HERE/../store/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
fi

case "$BACKEND" in
  indexed)
    if [[ ! -f "$SCALE/pkg.data" ]]; then
      bash "$HERE/../store/build_stores.sh" "$SCALE"
    fi
    ;;
  lmdb)
    # shellcheck source=../store/ensure_lmdb.sh
    source "$HERE/../store/ensure_lmdb.sh"
    uw_require_lmdb
    if [[ ! -d "$SCALE/lmdb/pkg" ]]; then
      bash "$HERE/../store/build_lmdb_stores.sh" "$SCALE"
    fi
    ;;
  *)
    echo "unknown UW_STORE_BACKEND=$BACKEND (indexed|lmdb)" >&2
    exit 2
    ;;
esac

echo "== compiling rust_store against $BACKEND scale stores =="
STORE_DIR="$SCALE" UW_STORE_BACKEND="$BACKEND" bash "$HERE/build.sh"

echo "== SWI store oracle =="
START_SWI=$(date +%s%N)
STORE_DIR="$SCALE" swipl -q -g main -t halt "$HERE/../store_diff_runner.pl" \
  < "$SCALE/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== rust_store =="
START_RS=$(date +%s%N)
"$BIN" < "$SCALE/cases.jsonl" > "$OUT/rust.jsonl"
END_RS=$(date +%s%N)

python3 - <<PY
o0, o1, r0, r1 = $START_SWI, $END_SWI, $START_RS, $END_RS
print("timing: swi {:.3f}s  rust_store {:.3f}s  backend=$BACKEND".format((o1-o0)/1e9, (r1-r0)/1e9))
PY

echo "== comparing =="
node "$HERE/../compare_jsonl.mjs" "$SCALE/cases.jsonl" "$OUT/swi.jsonl" "$OUT/rust.jsonl"
