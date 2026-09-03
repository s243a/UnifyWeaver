#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_measure_2x2.sh -- {indexed, lmdb} × {cache off, cache on} on the
# 5k catalog (seed 0xc0ffee01): one bound resolve_layered + 100× repeat
# in one process + 500-case store differential wall.
#
#   bash examples/pkg_resolver/store/run_measure_2x2.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/.out/scale"
cd "$ROOT"

# shellcheck source=ensure_lmdb.sh
source "$HERE/ensure_lmdb.sh"

echo "== catalog =="
node "$HERE/gen_scale_catalog.mjs" "$SCALE"
if [[ ! -f "$SCALE/pkg.jsonl" ]]; then
  node "$HERE/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
fi
if [[ ! -f "$SCALE/pkg.data" ]]; then
  bash "$HERE/build_stores.sh" "$SCALE"
fi

compile_backend() {
  local backend="$1"
  local wam="$2"
  mkdir -p "$wam"
  UW_STORE_BACKEND="$backend" swipl -q -g main -t halt \
    examples/pkg_resolver/wamjs_store/build.pl -- \
    examples/pkg_resolver/resolver_store.pl "$wam" "$SCALE" "$backend"
  cp examples/pkg_resolver/wamjs_store/resolver_store.mjs "$wam/resolver_store.mjs"
}

echo "== compile indexed =="
compile_backend indexed "$SCALE/wamjs"

LMDB_RAN=0
if uw_ensure_lmdb; then
  echo "== compile lmdb =="
  bash "$HERE/build_lmdb_stores.sh" "$SCALE"
  compile_backend lmdb "$SCALE/wamjs_lmdb"
  LMDB_RAN=1
else
  echo "lmdb arm skipped (npm package not loadable after ensure)"
fi

probe_one() {
  local backend="$1"
  local cache="$2"
  local wam="$3"
  local label="$4"
  echo "== probe $label =="
  UW_FACT_CACHE="$cache" UW_FACT_IO_STATS=1 \
    node "$HERE/run_probe.mjs" "$wam" "$SCALE/probe.json" --repeat 1 \
    > "$SCALE/probe_${label}.log" 2> "$SCALE/probe_${label}.err" || true
  cat "$SCALE/probe_${label}.log"
  echo "--- stderr ---"
  cat "$SCALE/probe_${label}.err"
}

repeat_one() {
  local cache="$2"
  local wam="$3"
  local label="$4"
  echo "== repeat100 $label =="
  UW_FACT_CACHE="$cache" UW_FACT_IO_STATS=1 \
    node "$HERE/run_probe.mjs" "$wam" "$SCALE/probe.json" --repeat 100 \
    > "$SCALE/repeat_${label}.log" 2> "$SCALE/repeat_${label}.err" || true
  cat "$SCALE/repeat_${label}.log"
  echo "--- stderr ---"
  cat "$SCALE/repeat_${label}.err"
}

diff_one() {
  local backend="$1"
  local cache="$2"
  echo "== differential $backend cache=$cache =="
  UW_STORE_BACKEND="$backend" UW_FACT_CACHE="$cache" \
    bash examples/pkg_resolver/run_store_differential.sh
}

probe_one indexed 0 "$SCALE/wamjs" indexed_off
probe_one indexed 1 "$SCALE/wamjs" indexed_on
repeat_one indexed 0 "$SCALE/wamjs" indexed_off
repeat_one indexed 1 "$SCALE/wamjs" indexed_on

if [[ "$LMDB_RAN" -eq 1 ]]; then
  probe_one lmdb 0 "$SCALE/wamjs_lmdb" lmdb_off
  probe_one lmdb 1 "$SCALE/wamjs_lmdb" lmdb_on
  repeat_one lmdb 0 "$SCALE/wamjs_lmdb" lmdb_off
  repeat_one lmdb 1 "$SCALE/wamjs_lmdb" lmdb_on
fi

echo "== 500-case differentials =="
diff_one indexed 0
diff_one indexed 1
if [[ "$LMDB_RAN" -eq 1 ]]; then
  diff_one lmdb 0
  diff_one lmdb 1
fi

echo "measure_2x2: lmdb_ran=$LMDB_RAN"
