#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- compile examples/pkg_resolver/resolver.pl (+ the contract-corpus
# data from test_resolver.pl + the shared driver) through the C++ WAM target
# (wam_cpp_target.pl, interpreter emit mode), then g++ the emitted project
# into the `uwresolve` corpus driver binary.
#
#   bash examples/pkg_resolver/cpp/build.sh
#
# Memory note (WSL / low-RAM): interpreter emit mode keeps generated_program.cpp
# a flat instruction array, so -O0 compiles peak ~1.4 GB and take <1 min here.
# Do NOT switch to emit_mode(functions)/mixed(...) on a low-RAM box without
# staging -- lowered per-predicate C++ is what has OOM'd cpp_e2e historically.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP="$HERE/cpp"

cd "$HERE"
# 1. Transpile resolver.pl + corpus data + driver.pl -> $HERE/cpp/*.cpp,*.h
swipl -q -g main -t halt build.pl -- .

# 2. Compile.  wam_runtime.cpp is cached by content hash under /tmp so repeat
#    builds only recompile the (test-specific) generated_program.cpp.
cd "$CPP"
CACHE=/tmp/uw_cpp_runtime_cache
mkdir -p "$CACHE"
RT_HASH="$(sha256sum wam_runtime.cpp | cut -d' ' -f1)"
RT_OBJ="$CACHE/wam_runtime_${RT_HASH}.o"
if [[ ! -f "$RT_OBJ" ]]; then
  g++ -std=c++17 -O0 -c -o "$RT_OBJ.tmp.$$" wam_runtime.cpp
  mv -f "$RT_OBJ.tmp.$$" "$RT_OBJ"
fi
g++ -std=c++17 -O0 -c -o generated_program.o generated_program.cpp
g++ -std=c++17 -O0 -o uwresolve "$RT_OBJ" generated_program.o main.cpp
echo "build.sh: g++ clean -> $CPP/uwresolve"
