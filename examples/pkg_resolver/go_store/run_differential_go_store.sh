#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_differential_go_store.sh -- >=500 seeded cases against the 5k catalog
# (store/gen_scale_catalog.mjs, seed 0xc0ffee01); SWI store adapter vs the
# Go WAM store-backed build, 0 divergences on all 10 queries. SWI reads the
# same JSONL store dump (per D48).
#
#   bash examples/pkg_resolver/go_store/run_differential_go_store.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/go_store/run_differential_go_store.sh
#
# Default backend is indexed. lmdb is opt-in and fails loudly on the Go lane.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/../store/.out/scale"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.diff_out"
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

echo "== compiling go_store against $BACKEND scale stores =="
STORE_DIR="$SCALE" UW_STORE_BACKEND="$BACKEND" bash "$HERE/build.sh"

echo "== SWI store oracle =="
START_SWI=$(date +%s%N)
STORE_DIR="$SCALE" swipl -q -g main -t halt "$HERE/../store_diff_runner.pl" \
  < "$SCALE/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== go_store =="
START_GO=$(date +%s%N)
"$HERE/uwresolvestore" < "$SCALE/cases.jsonl" > "$OUT/go.jsonl"
END_GO=$(date +%s%N)

python3 - <<PY
o0, o1, g0, g1 = $START_SWI, $END_SWI, $START_GO, $END_GO
print("timing: swi {:.3f}s  go_store {:.3f}s  backend=$BACKEND".format((o1-o0)/1e9, (g1-g0)/1e9))
PY

echo "== comparing =="
node "$HERE/../compare_jsonl.mjs" "$SCALE/cases.jsonl" "$OUT/swi.jsonl" "$OUT/go.jsonl"
