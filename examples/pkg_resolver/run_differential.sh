#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_differential.sh -- ≥2000 seeded catalogs through SWI and the wamjs
# build; exit non-zero on any divergence of the four queries.
#
#   bash examples/pkg_resolver/run_differential.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/.diff_out"
mkdir -p "$OUT"

cd "$ROOT"

echo "== generating seeded catalogs =="
node "$HERE/gen_catalogs.mjs" > "$OUT/cases.jsonl"
echo "cases: $(wc -l < "$OUT/cases.jsonl") -> $OUT/cases.jsonl"

echo "== SWI oracle =="
START_SWI=$(date +%s%N)
swipl -q -g main -t halt "$HERE/diff_runner.pl" < "$OUT/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== wamjs build =="
START_WAM=$(date +%s%N)
node "$HERE/wamjs/diff_runner_wamjs.mjs" < "$OUT/cases.jsonl" > "$OUT/wamjs.jsonl"
END_WAM=$(date +%s%N)

python3 - <<PY
o0, o1, w0, w1 = $START_SWI, $END_SWI, $START_WAM, $END_WAM
print("timing: swi {:.3f}s  wamjs {:.3f}s".format((o1-o0)/1e9, (w1-w0)/1e9))
PY

echo "== comparing =="
node "$HERE/compare_jsonl.mjs" "$OUT/cases.jsonl" "$OUT/swi.jsonl" "$OUT/wamjs.jsonl"
