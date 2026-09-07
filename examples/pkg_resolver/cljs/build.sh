#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# build.sh -- compile examples/pkg_resolver/resolver.pl through the
# ClojureScript target's WAM lane into generated/resolver/{core,runtime}.cljs,
# then nbb-load the result (the `node --check` analogue).
#
#   bash examples/pkg_resolver/cljs/build.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../resolver.pl"
OUT="$HERE/generated/resolver"

mkdir -p "$OUT"

# The WAM-Clojure templates carry UTF-8; SWI must read and write them as such.
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT"

# write_wam_clojurescript_files/3 stages the JVM project it derives from in
# .wamclj_tmp; only the two .cljs files are the artifact.
rm -rf "$OUT/.wamclj_tmp"

nbb --classpath "$HERE" "$HERE/load_check.cljs"
