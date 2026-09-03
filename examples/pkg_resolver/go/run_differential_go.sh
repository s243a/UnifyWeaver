#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_differential_go.sh -- the SAME seeded generator as the JS
# differential (gen_catalogs.mjs, seed 0xa5b6c7d8); SWI vs Go, 0
# divergences on all 10 queries.
#
#   bash examples/pkg_resolver/go/run_differential_go.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/.diff_out"
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -x "$HERE/uwresolve" ]]; then
  bash "$HERE/build.sh"
fi

echo "== generating seeded catalogs =="
node "$HERE/../gen_catalogs.mjs" > "$OUT/cases.jsonl"
echo "cases: $(wc -l < "$OUT/cases.jsonl") -> $OUT/cases.jsonl"

echo "== SWI oracle =="
START_SWI=$(date +%s%N)
swipl -q -g main -t halt "$HERE/../diff_runner.pl" < "$OUT/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== go build =="
START_GO=$(date +%s%N)
"$HERE/uwresolve" < "$OUT/cases.jsonl" > "$OUT/go.jsonl"
END_GO=$(date +%s%N)

python3 - <<PY
o0, o1, g0, g1 = $START_SWI, $END_SWI, $START_GO, $END_GO
print("timing: swi {:.3f}s  go {:.3f}s".format((o1-o0)/1e9, (g1-g0)/1e9))
PY

echo "== comparing =="
node "$HERE/../compare_jsonl.mjs" "$OUT/cases.jsonl" "$OUT/swi.jsonl" "$OUT/go.jsonl"
