#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
#   bash examples/pkg_resolver/wamjs_store/run_corpus_wamjs.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"

cd "$ROOT"
if [[ ! -f "$HERE/js/generated_program.js" ]]; then
  bash "$HERE/build.sh"
fi
# dump_store_data writes cases.jsonl with term-catalog expected results
if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi
node "$HERE/run_corpus.mjs" "$STORE/cases.jsonl" "$OUT/wamjs.jsonl"
