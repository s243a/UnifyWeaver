#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_differential_patternjs.sh -- step A4 for the PATTERN lane.
#
# Identical to examples/cli_args/run_differential.sh except for which parser is
# put opposite the oracle: the same generator, the same seed, the same case file,
# the same comparator. The Prolog reference is replaced by the TRANSPILED parser
# (patternjs/cliArgs.generated.mjs, reached through the edge shim).
#
#   bash examples/cli_args/patternjs/run_differential_patternjs.sh
#
# Exits non-zero on any divergence. Artifacts land in ./.diff_out/ next to this
# script (cases.txt, oracle.jsonl, patternjs.jsonl).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UP="$HERE/.."
OUT="$HERE/.diff_out"
mkdir -p "$OUT"

echo "== generating seeded cases (the harness's own generator, unchanged) =="
node "$UP/gen_cases.mjs" > "$OUT/cases.txt"
echo "cases: $(wc -l < "$OUT/cases.txt") lines -> $OUT/cases.txt"

echo "== running the JavaScript oracle (peerhailer parseArgs) =="
node "$UP/diff_runner.mjs" < "$OUT/cases.txt" > "$OUT/oracle.jsonl"

echo "== running the TRANSPILED parser (UnifyWeaver pattern lane) =="
node "$HERE/diff_runner_patternjs.mjs" < "$OUT/cases.txt" > "$OUT/patternjs.jsonl"

echo "== comparing =="
node "$UP/compare_jsonl.mjs" "$OUT/cases.txt" "$OUT/oracle.jsonl" "$OUT/patternjs.jsonl"
