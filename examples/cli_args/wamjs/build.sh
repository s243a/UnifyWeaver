#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# build.sh -- ONE command that turns examples/cli_args/cli_args.pl into a
# JavaScript WAM project node can run (mixed emit mode: lower what the
# Tier-2 emitter accepts, interpret the rest).
#
#   bash examples/cli_args/wamjs/build.sh
#
# Output: js/generated_program.js + js/wam_runtime.js next to this script.
# The only hand-written file that touches parser behaviour is cliArgs.mjs,
# and it does term<->JS conversion at the edge and nothing else.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../cli_args.pl"
OUT="$HERE"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT"

node --check "$OUT/js/generated_program.js"
node --check "$OUT/js/wam_runtime.js"
echo "build.sh: node --check clean -> $OUT/js/generated_program.js"
