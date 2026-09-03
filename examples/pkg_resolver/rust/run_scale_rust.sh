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
# With no arguments it sweeps the documented ladder (40 50 60 100 150 250 500
# 1000) plus the full 5000-package catalog, so the scaling curve is visible;
# the 5000 point is the SWI reference size used by
# examples/pkg_resolver/run_scale_demo.sh.
#
# Since Value gained structural sharing (D52) both curves are LINEAR in
# catalog size and the 5000 point fits in ~680 MB; the address-space cap
# below is a guard rail, not the binding constraint it used to be.

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
  SIZES=(40 50 60 100 150 250 500 1000 5000)
fi

# Cap address space and wall time so a regression (or an oversized point)
# fails fast instead of thrashing the machine. Before structural sharing the
# 5000-package point needed 8.5 GB and was OOM-killed here; it now peaks at
# ~680 MB, so a point that hits this cap means something regressed.
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
