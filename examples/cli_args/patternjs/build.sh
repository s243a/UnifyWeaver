#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# build.sh -- ONE command that turns examples/cli_args/cli_args.pl into a
# JavaScript module node can run, through UnifyWeaver's PATTERN lane
# (typescript_target's structural / general clause lowering, then
# vanilla_js_target's type strip).
#
#   bash examples/cli_args/patternjs/build.sh
#
# Output: cliArgs.generated.mjs next to this script -- 41 function declarations,
# every predicate `parse_args/2` transitively calls. Nothing here is hand-written
# JavaScript; the only hand-written file in this directory that touches the
# parser is cliArgs.mjs, and it does term<->JS conversion at the edge and nothing
# else.
#
# The build is a pure compile: it never RUNS cli_args.pl.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../cli_args.pl"
OUT="$HERE/cliArgs.generated.mjs"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT"

node --check "$OUT"
echo "build.sh: node --check clean -> $OUT"
