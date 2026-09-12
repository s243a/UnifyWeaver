#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- dump corpus catalogs to P/2 JSONL, index with D43, compile
# resolver.pl + resolver_store.pl through wam_cpp against those stores, then
# g++ compile+link the JSON diff driver. C++ lane of
# examples/pkg_resolver/rust_store/build.sh.
#
#   bash examples/pkg_resolver/cpp_store/build.sh
#   STORE_DIR=path bash examples/pkg_resolver/cpp_store/build.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/cpp_store/build.sh
#
# Default backend is indexed(Prefix). lmdb(Dir) is opt-in and fails LOUDLY on
# the C++ lane (Stage-2 lazy+cached reader not built yet), never a silent swap
# to indexed.
#
# The generated project under uw_resolve_wam_cpp_store/ bakes the store path
# into register_seek_fact_source, so it is a per-build artifact (gitignored);
# re-running this script regenerates it. Unlike cpp/ (two binaries: a text
# corpus driver and a JSON diff driver) the store lane has ONE binary -- the
# JSON diff driver -- so the diff-driver g++ step is folded in here rather than
# living in a separate build_diff.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/../resolver_store.pl"
PROJ="$HERE/uw_resolve_wam_cpp_store"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

mkdir -p "$STORE"
cd "$ROOT"

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi

case "$BACKEND" in
  indexed)
    if [[ ! -f "$STORE/pkg.data" ]]; then
      bash examples/pkg_resolver/store/build_stores.sh "$STORE"
    fi
    ;;
  lmdb)
    # shellcheck source=../store/ensure_lmdb.sh
    source examples/pkg_resolver/store/ensure_lmdb.sh
    uw_require_lmdb
    if [[ ! -d "$STORE/lmdb/pkg" ]]; then
      bash examples/pkg_resolver/store/build_lmdb_stores.sh "$STORE"
    fi
    ;;
  *)
    echo "cpp_store/build.sh: unknown UW_STORE_BACKEND=$BACKEND (indexed|lmdb)" >&2
    exit 2
    ;;
esac

echo "== compiling resolver.pl + resolver_store.pl through wam_cpp ($BACKEND) =="
swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$PROJ" "$STORE" "$BACKEND" >/dev/null

echo "== g++ compile+link diff driver =="
CPP="$PROJ/cpp"
cd "$CPP"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2}"
# generated WAM runtime + program (wam_runtime.h lives here -> -I.)
$CXX $CXXFLAGS -c -o wam_runtime.o wam_runtime.cpp
$CXX $CXXFLAGS -c -o generated_program.o generated_program.cpp
# hand-written edge (JSON <-> WAM terms), reused verbatim from cpp/ plus the
# store-only env_build + the store diff driver -- all under $HERE.
$CXX $CXXFLAGS -I. -I"$HERE" -c -o json.o             "$HERE/json.cpp"
$CXX $CXXFLAGS -I. -I"$HERE" -c -o term_build.o       "$HERE/term_build.cpp"
$CXX $CXXFLAGS -I. -I"$HERE" -c -o term_to_json.o     "$HERE/term_to_json.cpp"
$CXX $CXXFLAGS -I. -I"$HERE" -c -o env_build.o        "$HERE/env_build.cpp"
$CXX $CXXFLAGS -I. -I"$HERE" -c -o diff_main_store.o  "$HERE/diff_main_store.cpp"
$CXX $CXXFLAGS -o diff_uwresolve_store \
    wam_runtime.o generated_program.o json.o term_build.o term_to_json.o \
    env_build.o diff_main_store.o
echo "cpp_store/build.sh: backend=$BACKEND binary -> $CPP/diff_uwresolve_store"
