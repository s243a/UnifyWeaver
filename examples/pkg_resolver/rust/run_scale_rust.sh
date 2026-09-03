#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_scale_rust.sh -- B3: one resolve_layered on the scale catalog
# (examples/pkg_resolver/store/gen_scale_catalog.mjs, seed 0xc0ffee01),
# load time and resolve time reported separately.
#
#   bash examples/pkg_resolver/rust/run_scale_rust.sh [maxPackages ...]
#
# With no arguments it sweeps 100 250 500 1000 2500 5000 so the scaling curve
# is visible; the 5000 point is the SWI reference size used by
# examples/pkg_resolver/run_scale_demo.sh. Peak RSS is reported per point --
# the WAM interpreter deep-copies register contents into every choice point,
# so memory, not time, is the first thing that gives out.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
SCALE="$HERE/.scale"
BIN="$HERE/uw_resolve_wam/target/release/uw_resolve"

if [ ! -x "$BIN" ]; then
  echo "run_scale_rust.sh: $BIN missing -- run build.sh first" >&2
  exit 2
fi

mkdir -p "$SCALE"
if [ ! -f "$SCALE/rich.jsonl" ]; then
  node "$PKG/store/gen_scale_catalog.mjs" "$SCALE"
fi

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
  SIZES=(10 20 30 40 60 80 5000)
fi

# The WAM interpreter deep-copies every register value into every choice
# point, and A1 holds the WHOLE catalog term, so memory grows as
# (#choice points x catalog size). Cap address space and wall time so an
# oversized point fails fast instead of thrashing the machine.
MEM_KB="${UW_SCALE_MEM_KB:-4194304}"   # 4 GiB
SECS="${UW_SCALE_TIMEOUT:-120}"

for n in "${SIZES[@]}"; do
  node "$HERE/scale_to_case.mjs" "$SCALE" "$n" > "$SCALE/case_$n.json"
  pkgs=$(node -e "const c=require('fs').readFileSync('$SCALE/case_$n.json','utf8');const o=JSON.parse(c);console.log(o.catalog.packages.length+' '+o.catalog.depends.length)")
  echo "== packages/depends: $pkgs (cap $n) =="
  ( ulimit -v "$MEM_KB"; exec timeout "$SECS" "$BIN" --bench ) \
    < "$SCALE/case_$n.json" > "$SCALE/out_$n.json" 2> "$SCALE/err_$n.txt"
  status=$?
  grep -E '^(load_ms|resolve_ms|selection_size)' "$SCALE/err_$n.txt" || true
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status) -- ${SECS}s / $((MEM_KB / 1024)) MiB budget exhausted"
  fi
done
