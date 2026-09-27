#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_corpus_wamjs.sh -- drive the P0 contract corpus through the JS-WAM
# build and compare every result to SWI (including blocked/3 terms).
#
#   bash examples/pkg_resolver/wamjs/run_corpus_wamjs.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"

cd "$ROOT"
swipl -q -g dump_corpus -t halt examples/pkg_resolver/dump_corpus.pl > "$OUT/swi.jsonl"
node "$HERE/run_corpus.mjs" "$OUT/swi.jsonl" "$OUT/wamjs.jsonl"
