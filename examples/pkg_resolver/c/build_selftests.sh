#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Build and run M2 term-layer selftests (actual WAM heap).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RT="$HERE/diff/runtime"
OUT="$HERE/diff/selftests"
mkdir -p "$RT" "$OUT"
cd "$ROOT"

echo "== gen wam runtime =="
swipl -q -g main -t halt "$HERE/gen_runtime.pl" -- "$RT"

read -r -a EXTRA_CFLAGS <<< "${CFLAGS:-}"
CC=(gcc -std=c11 -Wall -Wextra -O0 "${EXTRA_CFLAGS[@]}" -I"$RT" -I"$HERE")
COMMON=(
  "$HERE/json.c"
  "$HERE/term_heap.c"
  "$HERE/term_build.c"
  "$HERE/term_to_json.c"
  "$HERE/term_render.c"
  "$RT/wam_runtime.c"
  -lm
)

echo "== term_build_selftest =="
"${CC[@]}" -o "$OUT/term_build_selftest" \
  "$HERE/term_build_selftest.c" "${COMMON[@]}"
"$OUT/term_build_selftest"

echo "== term_to_json_actual_heap_selftest =="
"${CC[@]}" -o "$OUT/term_to_json_actual_heap_selftest" \
  "$HERE/term_to_json_actual_heap_selftest.c" "${COMMON[@]}"
"$OUT/term_to_json_actual_heap_selftest"

echo "== runtime_safety_selftest =="
"${CC[@]}" -o "$OUT/runtime_safety_selftest" \
  "$HERE/runtime_safety_selftest.c" "${COMMON[@]}" \
  -Wl,--wrap=realloc -Wl,--wrap=calloc -Wl,--wrap=malloc
"$OUT/runtime_safety_selftest"

echo "build_selftests.sh: all selftests passed"
