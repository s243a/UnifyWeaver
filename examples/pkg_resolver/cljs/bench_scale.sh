#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# bench_scale.sh -- B3: one resolve_layered on the 5k-package scale catalog,
# ClojureScript leg, with the SWI term leg alongside for reference.
#
# The catalog is the repository's own seeded 5k generator
# (store/gen_scale_catalog.mjs, seed 0xc0ffee01) converted to the JSON catalog
# document the ClojureScript edge reads. The request and the frozen base are
# the generator's own probe.json, which is also what store/scale_demo.pl uses,
# so the two legs answer the same question and their answers are compared.
#
#   bash examples/pkg_resolver/cljs/bench_scale.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$PKG/../.." && pwd)"
SCALE="$HERE/.scale"
mkdir -p "$SCALE"
cd "$ROOT"

if [[ ! -f "$SCALE/rich.jsonl" ]]; then
  node "$PKG/store/gen_scale_catalog.mjs" "$SCALE"
fi
if [[ ! -f "$SCALE/catalog.json" ]]; then
  node "$HERE/scale_to_catalog.mjs" "$SCALE"
fi

echo "== B3: ClojureScript (nbb) =="
nbb --classpath "$HERE" "$HERE/bench_scale.cljs" "$SCALE" | tee "$SCALE/bench_cljs.log"

echo
echo "== B3: SWI term leg (reference; same catalog, same probe) =="
# scale_demo.pl reports a term leg AND a P/2-store leg; only the term leg is
# the B3 reference, and the store leg needs stores this benchmark does not
# build (run ../run_scale_demo.sh for that one), so its lines are filtered out.
swipl -q -g scale_demo -t halt "$PKG/store/scale_demo.pl" -- "$SCALE" 2>/dev/null \
  | grep -E '^swi_term_(load_s|resolve_s|total_s|result)' \
  | tee "$SCALE/bench_swi.log" || true
