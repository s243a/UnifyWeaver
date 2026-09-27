#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- compile examples/pkg_resolver/resolver.pl through wam_c
# (no_kernels(true), lowered helpers off) into generated/, then gcc the
# smoke driver. Fail closed if the generator omitted a predicate.
#
#   bash examples/pkg_resolver/c/build.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../resolver.pl"
GEN="$HERE/generated"

mkdir -p "$GEN"

echo "== compiling $SRC through wam_c =="
set +e
swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$GEN" >"$GEN/build.pl.log" 2>&1
SWIPL_RC=$?
set -e
cat "$GEN/build.pl.log"
echo "orig_exit=$SWIPL_RC"
if [ "$SWIPL_RC" -ne 0 ]; then
  echo "build.sh: generator failed (orig_exit=$SWIPL_RC)" >&2
  exit "$SWIPL_RC"
fi

if [ ! -f "$GEN/lib.c" ] || [ ! -f "$GEN/wam_runtime.c" ] || [ ! -f "$GEN/wam_runtime.h" ] || [ ! -f "$GEN/setup_all.inc" ]; then
  echo "build.sh: generated sources missing under $GEN" >&2
  exit 1
fi

echo "== gcc smoke_runner =="
set +e
gcc -O0 -std=c11 -I"$GEN" -o "$GEN/smoke_runner" \
  "$HERE/driver.c" "$GEN/lib.c" "$GEN/wam_runtime.c" -lm \
  >"$GEN/gcc.log" 2>&1
GCC_RC=$?
set -e
cat "$GEN/gcc.log"
echo "orig_exit=$GCC_RC"
if [ "$GCC_RC" -ne 0 ]; then
  echo "build.sh: gcc failed (orig_exit=$GCC_RC)" >&2
  exit "$GCC_RC"
fi

echo "build.sh: gcc clean -> $GEN/smoke_runner"
