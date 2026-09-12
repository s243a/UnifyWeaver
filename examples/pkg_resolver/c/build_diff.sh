#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# build_diff.sh -- compile resolver.pl to diff_uwresolve (WAM-C term differential).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
GEN="$HERE/diff/generated"
BIN="$HERE/diff/diff_uwresolve"
mkdir -p "$HERE/diff"
cd "$ROOT"

echo "== swipl build_diff.pl =="
swipl -q -g main -t halt "$HERE/build_diff.pl" -- "$HERE/../resolver.pl" "$GEN"

echo "== gcc diff_uwresolve =="
read -r -a EXTRA_CFLAGS <<< "${CFLAGS:-}"
gcc -std=c11 -Wall -Wextra -O0 "${EXTRA_CFLAGS[@]}" \
  -I"$GEN" -I"$HERE" \
  -o "$BIN" \
  "$HERE/diff_main.c" \
  "$HERE/json.c" \
  "$HERE/term_heap.c" \
  "$HERE/term_build.c" \
  "$HERE/term_to_json.c" \
  "$GEN/lib.c" \
  "$GEN/wam_runtime.c" \
  -lm

echo "build_diff.sh: gcc clean -> $BIN"
