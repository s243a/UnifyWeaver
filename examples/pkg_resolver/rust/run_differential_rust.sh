#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_differential_rust.sh -- the SAME seeded generator the wamjs differential
# uses (examples/pkg_resolver/gen_catalogs.mjs, 2400 cases), SWI leg vs Rust
# leg. Exits non-zero on any divergence of any of the ten queries.
#
#   bash examples/pkg_resolver/rust/run_differential_rust.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$PKG/../.." && pwd)"
OUT="$HERE/.diff_out"
BIN="$HERE/uw_resolve_wam/target/release/uw_resolve"
mkdir -p "$OUT"

if [ ! -x "$BIN" ]; then
  echo "run_differential_rust.sh: $BIN missing -- run build.sh first" >&2
  exit 2
fi

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

cd "$ROOT"

echo "== generating seeded catalogs =="
node "$PKG/gen_catalogs.mjs" > "$OUT/cases.jsonl"
echo "cases: $(wc -l < "$OUT/cases.jsonl") -> $OUT/cases.jsonl"

echo "== SWI oracle =="
START_SWI=$(date +%s%N)
swipl -q -g main -t halt "$PKG/diff_runner.pl" < "$OUT/cases.jsonl" > "$OUT/swi.jsonl"
END_SWI=$(date +%s%N)

echo "== wam_rust build =="
START_RS=$(date +%s%N)
"$BIN" < "$OUT/cases.jsonl" > "$OUT/rust.jsonl"
END_RS=$(date +%s%N)

python3 - <<PY
o0, o1, r0, r1 = $START_SWI, $END_SWI, $START_RS, $END_RS
print("timing: swi {:.3f}s  wam_rust {:.3f}s".format((o1-o0)/1e9, (r1-r0)/1e9))
PY

echo "== comparing =="
node "$PKG/compare_jsonl.mjs" "$OUT/cases.jsonl" "$OUT/swi.jsonl" "$OUT/rust.jsonl"
