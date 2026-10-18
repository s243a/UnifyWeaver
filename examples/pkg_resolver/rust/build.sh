#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# build.sh -- compile examples/pkg_resolver/resolver.pl through wam_rust into
# uw_resolve_wam/ next to this script, drop the thin term<->JSON shim in as a
# second binary, and cargo build --release.
#
#   bash examples/pkg_resolver/rust/build.sh
#
# The generated crate is committed; re-running this script regenerates it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../resolver.pl"
PROJ="$HERE/uw_resolve_wam"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

echo "== compiling $SRC through wam_rust =="
( cd "$HERE" && swipl -q -g main -t halt build.pl -- "$SRC" "$PROJ" ) >/dev/null

# The shim is NOT generated: it is a hand-written edge (JSON <-> WAM terms and
# nothing else). Copy it into the crate as a second binary target; cargo
# auto-discovers src/bin/<name>/main.rs.
mkdir -p "$PROJ/src/bin/uw_resolve"
cp "$HERE/shim/main.rs" "$PROJ/src/bin/uw_resolve/main.rs"
cp "$HERE/shim/json.rs" "$PROJ/src/bin/uw_resolve/json.rs"

echo "== cargo build --release =="
BUILD_START=$(date +%s%N)
( cd "$PROJ" && cargo build --release --bin uw_resolve 2>&1 | tail -3 )
BUILD_END=$(date +%s%N)
echo "build.sh: compile time $(( (BUILD_END - BUILD_START) / 1000000 )) ms"
echo "build.sh: binary -> $PROJ/target/release/uw_resolve"
