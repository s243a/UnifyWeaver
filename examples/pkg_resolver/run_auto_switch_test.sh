#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_auto_switch_test.sh -- prove the size-gated auto-router (resolve_auto.mjs)
# produces BYTE-IDENTICAL results to the pure term lane on both sides of the
# threshold, and that it routes large catalogs to the store lane and small ones
# to the term lane.
#
# Requires both lanes built and the store binary compiled against the 5k SCALE
# store (examples/pkg_resolver/store/.out/scale):
#   bash examples/pkg_resolver/rust/build.sh
#   bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh   # bakes 5k
#
#   bash examples/pkg_resolver/run_auto_switch_test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

TERMBIN="$HERE/rust/uw_resolve_wam/target/release/uw_resolve"
STOREBIN="$HERE/rust_store/uw_resolve_wam_store/target/release/uw_resolve_store"
SCALE="$HERE/rust/.scale"
STORESCALE="$HERE/store/.out/scale"
OUT="$HERE/.auto_out"
DRIVER="$HERE/resolve_auto.mjs"
mkdir -p "$OUT" "$SCALE"

for b in "$TERMBIN" "$STOREBIN"; do
  if [ ! -x "$b" ]; then echo "MISSING binary: $b -- build first" >&2; exit 2; fi
done

# --- fixtures -------------------------------------------------------------
# LARGE: several queries over the SAME full 5k catalog (matches the baked store).
if [ ! -f "$SCALE/rich.jsonl" ]; then
  node "$HERE/store/gen_scale_catalog.mjs" "$SCALE"
fi
node "$HERE/rust/scale_to_case.mjs" "$SCALE" 5000 > "$OUT/base5k.json"

node - "$OUT/base5k.json" "$OUT/large.jsonl" <<'NODE'
import { readFileSync, writeFileSync } from "node:fs";
const base = JSON.parse(readFileSync(process.argv[2], "utf8"));
const cat = base.catalog;
// A handful of the 10 queries, all over the identical 5k catalog. `args` shapes
// match the term shim: resolve*/list of requests; dependents/orphans/single.
const reqs = base.args && base.args.length ? base.args : ["p0"];
const one = reqs[0];
const cases = [
  { id: "L_resolve_layered", catalog: cat, query: "resolve_layered", args: reqs },
  { id: "L_resolve",         catalog: cat, query: "resolve",         args: reqs },
  { id: "L_dependents",      catalog: cat, query: "dependents",      args: one },
  { id: "L_dependents_inst", catalog: cat, query: "dependents_installed", args: one },
  { id: "L_removal_orphans", catalog: cat, query: "removal_orphans", args: one },
  { id: "L_freeze_audit",    catalog: cat, query: "freeze_audit",    args: [] }
];
writeFileSync(process.argv[3], cases.map((c) => JSON.stringify(c)).join("\n") + "\n");
NODE

# SMALL: a self-contained tiny catalog (well below threshold -> term lane).
node - "$OUT/small.jsonl" <<'NODE'
import { writeFileSync } from "node:fs";
const cat = {
  packages: [["a",[1,0,0]],["b",[1,0,0]],["c",[1,0,0]]],
  depends: [["a",[1,0,0],"b","any"],["b",[1,0,0],"c","any"]],
  conflicts: [], base: [], installed: [], requested: [], layers: [], excluded: [], aliases: []
};
const cases = [
  { id: "S_resolve",    catalog: cat, query: "resolve",    args: ["a"] },
  { id: "S_dependents", catalog: cat, query: "dependents", args: "c" }
];
writeFileSync(process.argv[2], cases.map((c) => JSON.stringify(c)).join("\n") + "\n");
NODE

fail=0
check() { # name file-a file-b
  if cmp -s "$2" "$3"; then echo "  OK   $1 (byte-identical)"; else
    echo "  FAIL $1 -- differs:"; diff "$2" "$3" | head -8; fail=1; fi
}

echo "== LARGE (5k catalog, >= threshold) =="
# term oracle
"$TERMBIN" < "$OUT/large.jsonl" > "$OUT/large.term.jsonl"
# driver auto (default threshold 500; 5k catalog -> store)
node "$DRIVER" --store-dir "$STORESCALE" --explain < "$OUT/large.jsonl" \
  > "$OUT/large.auto.jsonl" 2> "$OUT/large.auto.err"
# driver forced store
node "$DRIVER" --backend indexed --store-dir "$STORESCALE" < "$OUT/large.jsonl" \
  > "$OUT/large.store.jsonl"
# driver forced term
node "$DRIVER" --backend term < "$OUT/large.jsonl" > "$OUT/large.forceterm.jsonl"
check "auto(->store)  == term oracle" "$OUT/large.term.jsonl" "$OUT/large.auto.jsonl"
check "force-indexed  == term oracle" "$OUT/large.term.jsonl" "$OUT/large.store.jsonl"
check "force-term     == term oracle" "$OUT/large.term.jsonl" "$OUT/large.forceterm.jsonl"
if grep -q -- "-> indexed" "$OUT/large.auto.err"; then
  echo "  OK   auto routed the 5k catalog to the indexed store"
else
  echo "  FAIL auto did NOT route to store:"; cat "$OUT/large.auto.err"; fail=1
fi

echo "== SMALL (tiny catalog, < threshold) =="
"$TERMBIN" < "$OUT/small.jsonl" > "$OUT/small.term.jsonl"
node "$DRIVER" --store-dir "$STORESCALE" --explain < "$OUT/small.jsonl" \
  > "$OUT/small.auto.jsonl" 2> "$OUT/small.auto.err"
check "auto(->term)   == term oracle" "$OUT/small.term.jsonl" "$OUT/small.auto.jsonl"
if grep -q -- "-> term" "$OUT/small.auto.err"; then
  echo "  OK   auto kept the tiny catalog on the term lane"
else
  echo "  FAIL auto did NOT route small to term:"; cat "$OUT/small.auto.err"; fail=1
fi

echo "== MIXED (small + large in one stream) =="
cat "$OUT/small.jsonl" "$OUT/large.jsonl" > "$OUT/mixed.jsonl"
"$TERMBIN" < "$OUT/mixed.jsonl" > "$OUT/mixed.term.jsonl"
node "$DRIVER" --store-dir "$STORESCALE" < "$OUT/mixed.jsonl" > "$OUT/mixed.auto.jsonl"
check "mixed auto     == term oracle (order preserved)" "$OUT/mixed.term.jsonl" "$OUT/mixed.auto.jsonl"

echo "== fallback: store unavailable -> auto demotes to term =="
node "$DRIVER" --store-dir "$SCALE/does_not_exist" --explain < "$OUT/large.jsonl" \
  > "$OUT/large.fallback.jsonl" 2> "$OUT/large.fallback.err"
check "auto w/o store == term oracle (fallback)" "$OUT/large.term.jsonl" "$OUT/large.fallback.jsonl"

if [ "$fail" -eq 0 ]; then
  echo "AUTO-SWITCH TEST: PASS (byte-identical across the switch, both directions)"
else
  echo "AUTO-SWITCH TEST: FAIL"; exit 1
fi
