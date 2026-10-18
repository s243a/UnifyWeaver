#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_corpus_cpp.sh -- drive the 51-scenario P0.5/P3 contract corpus through
# the C++ WAM build and diff every serialised answer against SWI-Prolog (the
# oracle).  Both legs run the SAME driver.pl (emit_all/0 + ser/2), so a
# per-line difference is a genuine C++-WAM-vs-SWI divergence on frozen
# resolver.pl -- never a formatting artefact.
#
#   bash examples/pkg_resolver/cpp/run_corpus_cpp.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"

if [[ ! -x "$HERE/cpp/uwresolve" ]]; then
  bash "$HERE/build.sh"
fi

# SWI oracle: identical driver, run natively.
cd "$PKG"
swipl -q -t halt \
  -g "use_module(resolver), use_module(test_resolver), consult('cpp/driver'), emit_all" \
  2>/dev/null > "$OUT/swi.txt"

# C++ WAM leg.  The generated main shim prints a trailing "true"/"false"
# after the queried goal; emit_all/0 always succeeds, so drop that last line.
"$HERE/cpp/uwresolve" 'emit_all/0' 2>/dev/null | sed '$d' > "$OUT/cpp.txt"

SWI_N=$(wc -l < "$OUT/swi.txt")
CPP_N=$(wc -l < "$OUT/cpp.txt")
PASS=$(diff <(sort "$OUT/swi.txt") <(sort "$OUT/cpp.txt") \
        | grep -c '^<' || true)
MATCH=$(( SWI_N - PASS ))

echo "corpus: SWI emitted $SWI_N cases, C++ emitted $CPP_N cases"
echo "corpus: $MATCH/$SWI_N scenarios match SWI"
if diff -q "$OUT/swi.txt" "$OUT/cpp.txt" >/dev/null; then
  echo "corpus: PASS 51/51"
else
  echo "corpus: divergences (SWI '<' vs C++ '>'):"
  diff "$OUT/swi.txt" "$OUT/cpp.txt" || true
fi
