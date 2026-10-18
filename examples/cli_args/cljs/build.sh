#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
#
# build.sh -- ONE command that turns examples/cli_args/cli_args.pl into a
# ClojureScript namespace nbb can run, through UnifyWeaver's PATTERN lane
# (clojure_target's A3 whole-program lowering, then clojurescript_target's
# JVM->JS interop rewrite).
#
#   bash examples/cli_args/cljs/build.sh
#
# Output: generated/cli_args.cljs -- 40 defn forms, one per rule predicate that
# `parse_args/2` transitively calls, plus the A3 runtime block. The three
# ground-fact CONSTANT TABLES (global_options/1, default_registry/1,
# js_object_prototype_keys/1) are inlined at their call sites and are therefore
# not module members; 40 + 3 = the program's 43 predicates.
#
# The path is `generated/cli_args.cljs` because that is where nbb's classpath
# resolver looks for the namespace `generated.cli-args` (Clojure munges `-` to
# `_` in file names). Every runner here passes `--classpath` pointing at this
# directory.
#
# Nothing in this directory is hand-written ClojureScript except the two
# runners, and neither carries any parse logic -- see README.md.
#
# The build is a pure compile: it never RUNS cli_args.pl.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../cli_args.pl"
OUT="$HERE/generated/cli_args.cljs"

mkdir -p "$HERE/generated"

swipl -q -g main -t halt "$HERE/build.pl" -- "$SRC" "$OUT"

# The nbb analogue of `node --check`: nbb reads, analyses and evaluates the whole
# namespace. It catches an unbalanced form, an unresolvable symbol (the failure
# mode a forward reference without `(declare ...)` produces) and a misplaced
# `recur` -- all at load time, before any argv is parsed.
nbb --classpath "$HERE" -e '(require (quote [generated.cli-args :as ca])) (assert (fn? ca/parse-args-2)) (println "load: ok")'
echo "build.sh: nbb loads the namespace clean -> $OUT"
