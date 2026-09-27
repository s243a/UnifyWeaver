#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_differential_cljs.sh -- the differential gate for the CLOJURESCRIPT lane.
#
# Identical to examples/cli_args/run_differential.sh except for which parser is
# put opposite the oracle: the same generator, the same seed, the same case
# file, the same comparator. The Prolog reference is replaced by the TRANSPILED
# ClojureScript parser (cljs/generated/cli_args.cljs) running under nbb.
#
#   bash examples/cli_args/cljs/run_differential_cljs.sh
#
# Exits non-zero on any divergence. Artifacts land in ./.diff_out/ next to this
# script (cases.txt, oracle.jsonl, cljs.jsonl).
#
# ONE nbb process handles all 5067 lines: the runner reads the whole case file
# from stdin and writes one JSON line per case. Spawning a process per line
# would multiply nbb's startup cost by 5067.

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

echo "== running the TRANSPILED parser (UnifyWeaver CLJS lane, under nbb) =="
nbb --classpath "$HERE" "$HERE/diff_runner_cljs.cljs" < "$OUT/cases.txt" > "$OUT/cljs.jsonl"

echo "== comparing =="
node "$UP/compare_jsonl.mjs" "$OUT/cases.txt" "$OUT/oracle.jsonl" "$OUT/cljs.jsonl"
