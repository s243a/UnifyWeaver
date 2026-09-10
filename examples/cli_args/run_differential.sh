#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_differential.sh -- the differential harness for the Prolog CLI parser.
#
# Generates a seeded, reproducible sample of argv-lines (the 17-test corpus, a
# hand-written quirk sweep, and 5000 pseudorandom lines of length 2..7 drawn
# from the task's token alphabet), feeds the identical file to the JavaScript
# ORACLE and to the Prolog reference implementation, and compares the two
# result streams semantically.
#
#   bash examples/cli_args/run_differential.sh
#
# Exits non-zero on any divergence. Artifacts land in ./.diff_out/ next to this
# script (cases.txt, oracle.jsonl, prolog.jsonl) so a failure can be inspected.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/.diff_out"
mkdir -p "$OUT"

echo "== generating seeded cases =="
node "$HERE/gen_cases.mjs" > "$OUT/cases.txt"
echo "cases: $(wc -l < "$OUT/cases.txt") lines -> $OUT/cases.txt"

echo "== running the JavaScript oracle (peerhailer parseArgs) =="
node "$HERE/diff_runner.mjs" < "$OUT/cases.txt" > "$OUT/oracle.jsonl"

echo "== running the Prolog reference implementation =="
swipl -q -g main -t halt "$HERE/diff_runner.pl" < "$OUT/cases.txt" > "$OUT/prolog.jsonl"

echo "== comparing =="
node "$HERE/compare_jsonl.mjs" "$OUT/cases.txt" "$OUT/oracle.jsonl" "$OUT/prolog.jsonl"
