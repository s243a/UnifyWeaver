#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_differential_cpp.sh -- differential driver for C++ WAM pkg_resolver
#
#   bash examples/pkg_resolver/cpp/run_differential_cpp.sh [case_count]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/.diff_out"
DIFF_BIN="$HERE/diff/cpp/diff_uwresolve"
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -x "$DIFF_BIN" ]]; then
  (cd "$HERE" && bash build_diff.sh)
fi

COUNT="${1:-}"

echo "== generating seeded catalogs =="
if [[ -n "$COUNT" ]]; then
  set +o pipefail
  node "$HERE/../gen_catalogs.mjs" | head -n "$COUNT" > "$OUT/cases.jsonl"
  set -o pipefail
else
  node "$HERE/../gen_catalogs.mjs" > "$OUT/cases.jsonl"
fi
echo "cases: $(wc -l < "$OUT/cases.jsonl") -> $OUT/cases.jsonl"

echo "== SWI oracle =="
START_SWI=$(date +%s%N)
swipl -q -g main -t halt "$HERE/../diff_runner.pl" < "$OUT/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== cpp build =="
START_CPP=$(date +%s%N)
"$DIFF_BIN" < "$OUT/cases.jsonl" > "$OUT/cpp.jsonl"
END_CPP=$(date +%s%N)

python3 - <<PY
o0, o1, c0, c1 = $START_SWI, $END_SWI, $START_CPP, $END_CPP
print("timing: swi {:.3f}s  cpp {:.3f}s".format((o1-o0)/1e9, (c1-c0)/1e9))
PY

echo "== comparing =="
node "$HERE/../compare_jsonl.mjs" "$OUT/cases.jsonl" "$OUT/swi.jsonl" "$OUT/cpp.jsonl"
