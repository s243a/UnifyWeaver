#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- dump corpus catalogs to P/2 JSONL, index with D43, compile
# resolver_store.pl against those stores.
#
#   bash examples/pkg_resolver/wamjs_store/build.sh
#   STORE_DIR=path bash examples/pkg_resolver/wamjs_store/build.sh   # override

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../resolver_store.pl"
OUT="$HERE"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"

mkdir -p "$STORE"
cd "$ROOT"
swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
bash examples/pkg_resolver/store/build_stores.sh "$STORE"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT" "$STORE"

node --check "$OUT/js/generated_program.js"
node --check "$OUT/js/wam_runtime.js"
echo "wamjs_store/build.sh: node --check clean -> $OUT/js/generated_program.js"
