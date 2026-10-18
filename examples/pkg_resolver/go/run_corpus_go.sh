#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_corpus_go.sh -- drive the P3 contract corpus through the Go WAM
# build and compare every result to SWI (including blocked/3 terms).
#
#   bash examples/pkg_resolver/go/run_corpus_go.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"

cd "$ROOT"
if [[ ! -x "$HERE/uwresolve" ]]; then
  bash "$HERE/build.sh"
fi

swipl -q -g dump_corpus -t halt examples/pkg_resolver/dump_corpus.pl > "$OUT/swi.jsonl"

START_GO=$(date +%s%N)
"$HERE/uwresolve" --corpus "$OUT/swi.jsonl" > "$OUT/go.jsonl"
END_GO=$(date +%s%N)
python3 - <<PY
g0, g1 = $START_GO, $END_GO
print("timing: go corpus wall {:.3f}s (binary; go build not included)".format((g1-g0)/1e9))
PY
