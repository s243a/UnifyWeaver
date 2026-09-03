#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_store_differential.sh -- ≥500 seeded envs against the 5k catalog;
# SWI store adapter vs wamjs_store, 0 divergences.
#
#   bash examples/pkg_resolver/run_store_differential.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/run_store_differential.sh
#   UW_FACT_CACHE=1 bash examples/pkg_resolver/run_store_differential.sh
#   UW_LMDB_MATERIALISATION=lazy bash examples/pkg_resolver/run_store_differential.sh
#
# Materialisation default is `cached` (fleet). UW_FACT_CACHE=0|1 is the
# measurement alias for lazy|cached.
#
# Default backend is indexed. lmdb is opt-in and fails loudly if the
# npm package is missing (never silently uses indexed).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SCALE="$HERE/store/.out/scale"
BACKEND="${UW_STORE_BACKEND:-indexed}"
if [[ "$BACKEND" == "lmdb" ]]; then
  WAM="$SCALE/wamjs_lmdb"
else
  WAM="$SCALE/wamjs"
fi
OUT="$HERE/store/.out/scale_diff_${BACKEND}"
mkdir -p "$SCALE" "$WAM" "$OUT"

cd "$ROOT"

echo "== generating 5k catalog + 500 cases =="
node "$HERE/store/gen_scale_catalog.mjs" "$SCALE"
if [[ ! -f "$SCALE/pkg.jsonl" ]]; then
  node "$HERE/store/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
fi

case "$BACKEND" in
  indexed)
    if [[ ! -f "$SCALE/pkg.data" ]]; then
      bash "$HERE/store/build_stores.sh" "$SCALE"
    fi
    ;;
  lmdb)
    # shellcheck source=store/ensure_lmdb.sh
    source "$HERE/store/ensure_lmdb.sh"
    uw_require_lmdb
    if [[ ! -d "$SCALE/lmdb/pkg" ]]; then
      bash "$HERE/store/build_lmdb_stores.sh" "$SCALE"
    fi
    ;;
  *)
    echo "unknown UW_STORE_BACKEND=$BACKEND (indexed|lmdb)" >&2
    exit 2
    ;;
esac

echo "== compiling wamjs_store against $BACKEND stores =="
NEED_BUILD=0
if [[ ! -f "$WAM/js/generated_program.js" ]]; then
  NEED_BUILD=1
elif [[ "$BACKEND" == "lmdb" ]] && ! grep -q 'kind: "lmdb"' "$WAM/js/generated_program.js"; then
  NEED_BUILD=1
elif [[ "$BACKEND" == "indexed" ]] && ! grep -q 'kind: "indexed"' "$WAM/js/generated_program.js"; then
  NEED_BUILD=1
elif ! grep -q 'configure_lmdb_materialisation' "$WAM/js/wam_runtime.js" 2>/dev/null; then
  NEED_BUILD=1
fi
if [[ "$NEED_BUILD" -eq 1 ]]; then
  UW_STORE_BACKEND="$BACKEND" swipl -q -g main -t halt "$HERE/wamjs_store/build.pl" -- \
    "$HERE/resolver_store.pl" "$WAM" "$SCALE" "$BACKEND"
  node --check "$WAM/js/generated_program.js"
fi
cp "$HERE/wamjs_store/resolver_store.mjs" "$WAM/resolver_store.mjs"
cp "$HERE/wamjs_store/diff_runner_wamjs.mjs" "$WAM/diff_runner_wamjs.mjs"

echo "== SWI store oracle =="
START_SWI=$(date +%s%N)
STORE_DIR="$SCALE" swipl -q -g main -t halt "$HERE/store_diff_runner.pl" \
  < "$SCALE/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

MAT_LABEL="${UW_LMDB_MATERIALISATION:-}"
if [[ -z "$MAT_LABEL" ]]; then
  case "${UW_FACT_CACHE:-}" in
    0|off|false|no) MAT_LABEL=lazy ;;
    1|on|true|yes) MAT_LABEL=cached ;;
    *) MAT_LABEL="cached (default)" ;;
  esac
fi
echo "== wamjs_store backend=$BACKEND materialisation=$MAT_LABEL =="
START_WAM=$(date +%s%N)
node "$WAM/diff_runner_wamjs.mjs" < "$SCALE/cases.jsonl" > "$OUT/wamjs.jsonl"
END_WAM=$(date +%s%N)

python3 - <<PY
o0, o1, w0, w1 = $START_SWI, $END_SWI, $START_WAM, $END_WAM
print("timing: swi {:.3f}s  wamjs {:.3f}s  backend={} materialisation={}".format(
    (o1-o0)/1e9, (w1-w0)/1e9, "$BACKEND", "$MAT_LABEL"))
PY

echo "== comparing =="
node "$HERE/compare_jsonl.mjs" "$SCALE/cases.jsonl" "$OUT/swi.jsonl" "$OUT/wamjs.jsonl"
