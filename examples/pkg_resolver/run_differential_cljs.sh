#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_differential_cljs.sh -- run_differential.sh with the ClojureScript-WAM
# leg swapped in for the JS-WAM one. The SAME seeded generator (gen_catalogs.mjs,
# same seed), the SAME SWI oracle, the SAME comparator: >=2400 catalogs through
# SWI and the CLJS build; exit non-zero on any divergence of any query.
#
# The original run_differential.sh is untouched; this is a sibling, so the two
# legs can be run and timed independently.
#
#   bash examples/pkg_resolver/run_differential_cljs.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/cljs/.diff_out"
mkdir -p "$OUT"

cd "$ROOT"

echo "== generating seeded catalogs =="
node "$HERE/gen_catalogs.mjs" > "$OUT/cases.jsonl"
echo "cases: $(wc -l < "$OUT/cases.jsonl") -> $OUT/cases.jsonl"

echo "== SWI oracle =="
START_SWI=$(date +%s%N)
swipl -q -g main -t halt "$HERE/diff_runner.pl" < "$OUT/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== ClojureScript (nbb) build =="
START_CLJS=$(date +%s%N)
nbb --classpath "$HERE/cljs" "$HERE/cljs/diff_runner_cljs.cljs" \
    < "$OUT/cases.jsonl" > "$OUT/cljs.jsonl"
END_CLJS=$(date +%s%N)

python3 - <<PY
o0, o1, c0, c1 = $START_SWI, $END_SWI, $START_CLJS, $END_CLJS
print("timing: swi {:.3f}s  cljs {:.3f}s".format((o1-o0)/1e9, (c1-c0)/1e9))
PY

echo "== comparing =="
node "$HERE/compare_jsonl.mjs" "$OUT/cases.jsonl" "$OUT/swi.jsonl" "$OUT/cljs.jsonl"
