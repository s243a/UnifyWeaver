#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_differential_cpp_store.sh -- >=500 seeded cases against the 5k catalog
# (store/gen_scale_catalog.mjs, seed 0xc0ffee01); SWI store adapter vs the C++
# WAM store-backed build, 0 divergences on all 10 queries. SWI reads the same
# JSONL store dump (per D48). C++ lane of
# examples/pkg_resolver/rust_store/run_differential_rust_store.sh.
#
#   bash examples/pkg_resolver/cpp_store/run_differential_cpp_store.sh
#
# Default backend is indexed. lmdb is opt-in and fails loudly on the C++ lane.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/../store/.out/scale"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.diff_out"
BIN="$HERE/uw_resolve_wam_cpp_store/cpp/diff_uwresolve_store"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

# The C++ lmdb reader links vanilla system liblmdb, which rejects the default
# lmdb-js Symas-fork format. Opt into the from-source v1-compatible module so
# the store this script builds is readable by the binary build.sh compiles.
if [[ "$BACKEND" == "lmdb" ]]; then export UW_LMDB_DATA_V1=1; fi

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

echo "== compiling cpp_store against $BACKEND scale stores =="
STORE_DIR="$SCALE" UW_STORE_BACKEND="$BACKEND" bash "$HERE/build.sh"

echo "== SWI store oracle =="
START_SWI=$(date +%s%N)
STORE_DIR="$SCALE" swipl -q -g main -t halt "$HERE/../store_diff_runner.pl" \
  < "$SCALE/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== cpp_store =="
START_CPP=$(date +%s%N)
"$BIN" < "$SCALE/cases.jsonl" > "$OUT/cpp_store.jsonl"
END_CPP=$(date +%s%N)

python3 - <<PY
o0, o1, c0, c1 = $START_SWI, $END_SWI, $START_CPP, $END_CPP
print("timing: swi {:.3f}s  cpp_store {:.3f}s  backend=$BACKEND".format((o1-o0)/1e9, (c1-c0)/1e9))
PY

echo "== comparing =="
node "$HERE/../compare_jsonl.mjs" "$SCALE/cases.jsonl" "$OUT/swi.jsonl" "$OUT/cpp_store.jsonl"
