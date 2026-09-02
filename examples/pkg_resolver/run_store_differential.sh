#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_store_differential.sh -- ≥500 seeded envs against the 5k catalog;
# SWI store adapter vs wamjs_store, 0 divergences.
#
#   bash examples/pkg_resolver/run_store_differential.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SCALE="$HERE/store/.out/scale"
WAM="$SCALE/wamjs"
OUT="$HERE/store/.out/scale_diff"
mkdir -p "$SCALE" "$WAM" "$OUT"

cd "$ROOT"

echo "== generating 5k catalog + 500 cases =="
node "$HERE/store/gen_scale_catalog.mjs" "$SCALE"
if [[ ! -f "$SCALE/pkg.data" ]]; then
  node "$HERE/store/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
  bash "$HERE/store/build_stores.sh" "$SCALE"
fi

echo "== compiling wamjs_store against scale indexes =="
if [[ ! -f "$WAM/js/generated_program.js" ]]; then
  swipl -q -g main -t halt "$HERE/wamjs_store/build.pl" -- \
    "$HERE/resolver_store.pl" "$WAM" "$SCALE"
  node --check "$WAM/js/generated_program.js"
fi
cp "$HERE/wamjs_store/resolver_store.mjs" "$WAM/resolver_store.mjs"
cp "$HERE/wamjs_store/diff_runner_wamjs.mjs" "$WAM/diff_runner_wamjs.mjs"

echo "== SWI store oracle =="
START_SWI=$(date +%s%N)
STORE_DIR="$SCALE" swipl -q -g main -t halt "$HERE/store_diff_runner.pl" \
  < "$SCALE/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== wamjs_store =="
START_WAM=$(date +%s%N)
node "$WAM/diff_runner_wamjs.mjs" < "$SCALE/cases.jsonl" > "$OUT/wamjs.jsonl"
END_WAM=$(date +%s%N)

python3 - <<PY
o0, o1, w0, w1 = $START_SWI, $END_SWI, $START_WAM, $END_WAM
print("timing: swi {:.3f}s  wamjs {:.3f}s".format((o1-o0)/1e9, (w1-w0)/1e9))
PY

echo "== comparing =="
node "$HERE/compare_jsonl.mjs" "$SCALE/cases.jsonl" "$OUT/swi.jsonl" "$OUT/wamjs.jsonl"
