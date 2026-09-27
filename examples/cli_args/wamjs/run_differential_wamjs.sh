#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_differential_wamjs.sh -- step A2 for the JS-WAM lane.
#
# Identical to examples/cli_args/run_differential.sh except for which parser is
# put opposite the oracle: the same generator, the same seed, the same case file,
# the same comparator. The Prolog reference is replaced by the JS-WAM compiled
# parser (wamjs/js/generated_program.js, reached through the edge shim).
#
#   bash examples/cli_args/wamjs/run_differential_wamjs.sh
#
# Exits non-zero on any divergence. Artifacts land in ./.diff_out/ next to this
# script (cases.txt, oracle.jsonl, wamjs.jsonl).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UP="$HERE/.."
OUT="$HERE/.diff_out"
mkdir -p "$OUT"

echo "== generating seeded cases (the harness's own generator, unchanged) =="
node "$UP/gen_cases.mjs" > "$OUT/cases.txt"
echo "cases: $(wc -l < "$OUT/cases.txt") lines -> $OUT/cases.txt"

echo "== running the JavaScript oracle (peerhailer parseArgs) =="
START_ORACLE=$(date +%s%N)
node "$UP/diff_runner.mjs" < "$OUT/cases.txt" > "$OUT/oracle.jsonl"
END_ORACLE=$(date +%s%N)

echo "== running the JS-WAM compiled parser =="
START_WAM=$(date +%s%N)
node "$HERE/diff_runner_wamjs.mjs" < "$OUT/cases.txt" > "$OUT/wamjs.jsonl"
END_WAM=$(date +%s%N)

awk -v o0="$START_ORACLE" -v o1="$END_ORACLE" -v w0="$START_WAM" -v w1="$END_WAM" 'BEGIN {
  printf "timing: oracle %.3fs  wamjs %.3fs\n", (o1-o0)/1e9, (w1-w0)/1e9
}'

echo "== comparing =="
node "$UP/compare_jsonl.mjs" "$OUT/cases.txt" "$OUT/oracle.jsonl" "$OUT/wamjs.jsonl"
