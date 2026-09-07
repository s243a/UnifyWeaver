#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build.sh -- compile examples/pkg_resolver/resolver.pl through wam_javascript
# (emit_mode(mixed)) into js/ next to this script.
#
#   bash examples/pkg_resolver/wamjs/build.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../resolver.pl"
OUT="$HERE"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT"

node --check "$OUT/js/generated_program.js"
node --check "$OUT/js/wam_runtime.js"
echo "build.sh: node --check clean -> $OUT/js/generated_program.js"
