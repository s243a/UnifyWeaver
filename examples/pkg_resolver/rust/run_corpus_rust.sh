#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_corpus_rust.sh -- drive the P0.5 contract corpus through the Rust-WAM
# build and compare every result to SWI (selections AND explanation terms).
#
#   bash examples/pkg_resolver/rust/run_corpus_rust.sh
#
# Prints the B1 benchmark number: wall time of one full corpus run in a
# single process (the binary loads the shared WAM program once).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$PKG/../.." && pwd)"
OUT="$HERE/.corpus_out"
BIN="$HERE/uw_resolve_wam/target/release/uw_resolve"
mkdir -p "$OUT"

if [ ! -x "$BIN" ]; then
  echo "run_corpus_rust.sh: $BIN missing -- run build.sh first" >&2
  exit 2
fi

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

cd "$ROOT"
swipl -q -g dump_corpus -t halt examples/pkg_resolver/dump_corpus.pl > "$OUT/swi.jsonl"

START=$(date +%s%N)
"$BIN" < "$OUT/swi.jsonl" > "$OUT/rust.jsonl"
END=$(date +%s%N)

node "$HERE/compare_corpus.mjs" "$OUT/swi.jsonl" "$OUT/rust.jsonl"
echo "B1: corpus wall time $(( (END - START) / 1000000 )) ms ($(wc -l < "$OUT/swi.jsonl") scenarios, single process)"
