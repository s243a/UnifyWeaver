#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_corpus_cljs.sh -- drive the P3 contract corpus through the
# ClojureScript-WAM build and compare every result to SWI (blocked/3, audit/2
# and the upgrade verdicts included).
#
#   bash examples/pkg_resolver/cljs/run_corpus_cljs.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"

cd "$ROOT"
swipl -q -g dump_corpus -t halt examples/pkg_resolver/dump_corpus.pl > "$OUT/swi.jsonl"
echo "corpus cases: $(wc -l < "$OUT/swi.jsonl")"

START=$(date +%s%N)
nbb --classpath "$HERE" "$HERE/run_corpus.cljs" "$OUT/swi.jsonl" "$OUT/cljs.jsonl"
END=$(date +%s%N)
python3 -c "print('B1 corpus wall time: {:.3f}s'.format(($END-$START)/1e9))"
