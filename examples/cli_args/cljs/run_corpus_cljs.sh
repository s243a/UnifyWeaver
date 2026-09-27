#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# run_corpus_cljs.sh -- the CONTRACT CORPUS gate for the ClojureScript lane.
#
#   bash examples/cli_args/cljs/run_corpus_cljs.sh
#
# WHAT THIS IS, AND WHAT IT IS NOT.
#
# The patternjs lane runs peerhailer's own corpus by changing ONE line of the
# vendored test file -- its `import` -- and letting `node --test` do the rest.
# That is the strongest possible gate, and it is IMPOSSIBLE here: the corpus is
# an ESM module driven by node's test runner, and the transpiled parser is a
# ClojureScript namespace living in a different runtime (nbb). There is no
# import to swap. Saying so plainly is part of the deliverable.
#
# The honest equivalent, which is what this script does:
#
#   1. extract the argv-lines the 17 test() blocks exercise, from the corpus
#      SOURCE (oracle/cliArgs.test.mjs) rather than from a hand-copied list --
#      extract_corpus.mjs, which fails loudly if the corpus stops having 17
#      blocks or 25 argv-lines;
#   2. run every one of them through peerhailer's ORACLE parser and through the
#      TRANSPILED ClojureScript parser, using the harness's own stdin/JSONL
#      protocol;
#   3. compare the two, per contract point, including the exact CliError MESSAGE
#      -- the thing the corpus's own `assert.throws(..., /regex/)` checks and the
#      thing a class-only comparison would let slide.
#
# What this gate does NOT prove that the import swap would: that the corpus's
# assertions themselves hold. It proves the transpiled parser is
# INDISTINGUISHABLE from the parser those assertions are written against, on
# every line they exercise -- which is the same conclusion by transitivity, since
# `node --test` on the unmodified corpus is run below as step 0 to show the
# oracle passes it.
#
# Exits non-zero if any contract point disagrees.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UP="$HERE/.."
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"
# Scoping file only, copied from oracle/: the repository root package.json has no
# "type": "module", so an .mjs corpus copy needs this to resolve as ESM.
cp "$UP/oracle/package.json" "$OUT/package.json"

echo "== step 0: the vendored corpus passes against the oracle it is written for =="
# The vendored oracle/cliArgs.test.mjs keeps peerhailer's ORIGINAL import path
# (`../src/cliArgs.js`), which does not resolve inside oracle/ -- the same reason
# patternjs/cliArgs.patternjs.test.mjs is a copy with its import changed. So the
# corpus is copied here with ONLY that import repointed at the vendored oracle;
# nothing else is touched, and it is still peerhailer's parser being tested.
sed 's#"\.\./src/cliArgs\.js"#"../../oracle/cliArgs.js"#' \
    "$UP/oracle/cliArgs.test.mjs" > "$OUT/cliArgs.oracle.test.mjs"
node --test "$OUT/cliArgs.oracle.test.mjs" 2>&1 | tail -9

echo
echo "== step 1: extracting the corpus argv-lines from oracle/cliArgs.test.mjs =="
node "$HERE/extract_corpus.mjs"        > "$OUT/corpus.txt"
node "$HERE/extract_corpus.mjs" --map  > "$OUT/corpus_map.tsv"
echo "17 contract points -> $(wc -l < "$OUT/corpus.txt") argv-lines"

echo
echo "== step 2: running the JavaScript oracle (peerhailer parseArgs) =="
node "$UP/diff_runner.mjs" < "$OUT/corpus.txt" > "$OUT/oracle.jsonl"

echo "== step 3: running the TRANSPILED parser (UnifyWeaver CLJS lane, under nbb) =="
nbb --classpath "$HERE" "$HERE/diff_runner_cljs.cljs" < "$OUT/corpus.txt" > "$OUT/cljs.jsonl"

echo
echo "== step 4: comparing, per contract point, error messages included =="
node "$HERE/compare_corpus.mjs" "$OUT/corpus_map.tsv" "$OUT/oracle.jsonl" "$OUT/cljs.jsonl"
