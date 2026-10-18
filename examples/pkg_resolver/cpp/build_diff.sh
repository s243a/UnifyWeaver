#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build_diff.sh -- compile examples/pkg_resolver/resolver.pl through the
# C++ WAM target into the `diff_uwresolve` differential driver binary.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. Transpile resolver.pl -> $HERE/diff/cpp/*.cpp,*.h
swipl -q -g main -t halt build_diff.pl -- .

# 2. Compile and link.
cd "$HERE/diff/cpp"
CACHE=/tmp/uw_cpp_runtime_cache
mkdir -p "$CACHE"
RT_HASH="$(sha256sum wam_runtime.cpp | cut -d' ' -f1)"
RT_OBJ="$CACHE/wam_runtime_${RT_HASH}.o"
if [[ ! -f "$RT_OBJ" ]]; then
  g++ -std=c++17 -O0 -I. -I../.. -c -o "$RT_OBJ.tmp.$$" wam_runtime.cpp
  mv -f "$RT_OBJ.tmp.$$" "$RT_OBJ"
fi

g++ -std=c++17 -O0 -I. -I../.. -c -o generated_program.o generated_program.cpp
g++ -std=c++17 -O0 -I. -I../.. -c -o json.o              ../../json.cpp
g++ -std=c++17 -O0 -I. -I../.. -c -o term_build.o        ../../term_build.cpp
g++ -std=c++17 -O0 -I. -I../.. -c -o term_to_json.o      ../../term_to_json.cpp
g++ -std=c++17 -O0 -I. -I../.. -c -o diff_main.o         ../../diff_main.cpp

g++ -std=c++17 -O0 -o diff_uwresolve \
    "$RT_OBJ" generated_program.o json.o term_build.o term_to_json.o diff_main.o

echo "build_diff.sh: g++ clean -> $HERE/diff/cpp/diff_uwresolve"
