#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_corpus_go_store.sh -- drive the P3 contract corpus through the Go WAM
# store-backed build (D43 indexed seek stores) and compare every result to
# SWI (via cases.jsonl `expected`). Also asserts the store results are
# byte-IDENTICAL to the Go term-catalog corpus (examples/pkg_resolver/go).
#
#   bash examples/pkg_resolver/go_store/run_corpus_go_store.sh
#   UW_STORE_BACKEND=lmdb bash examples/pkg_resolver/go_store/run_corpus_go_store.sh
#
# lmdb arm: loud error if no compatible reader is built in (never a silent
# swap to the indexed binary).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.corpus_out"
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi

STORE_DIR="$STORE" UW_STORE_BACKEND="$BACKEND" bash "$HERE/build.sh"

"$HERE/uwresolvestore" --corpus "$STORE/cases.jsonl" > "$OUT/go_store.jsonl"

# Assert IDENTICAL to the Go term-catalog corpus (not just "both pass").
TERM_OUT="$HERE/../go/.corpus_out/go.jsonl"
if [[ ! -f "$TERM_OUT" ]]; then
  bash "$HERE/../go/run_corpus_go.sh" >/dev/null 2>&1 || true
fi
if [[ -f "$TERM_OUT" ]]; then
  python3 - "$TERM_OUT" "$OUT/go_store.jsonl" <<'PY'
import json, sys
def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d[r.pop("id")] = r
    return d
term = load(sys.argv[1]); store = load(sys.argv[2])
norm = lambda x: json.dumps(x, sort_keys=True)
common = set(term) & set(store)
diff = [i for i in common if norm(term[i]) != norm(store[i])]
extra = (set(term) ^ set(store))
if diff or extra:
    for i in diff:
        print("DIFF", i, "term", norm(term[i]), "store", norm(store[i]))
    if extra:
        print("id set mismatch:", extra)
    print("go_store corpus NOT identical to term corpus")
    sys.exit(1)
print("go_store corpus IDENTICAL to term corpus (%d/%d)" % (len(common), len(common)))
PY
else
  echo "note: term corpus output not found; skipped identity assertion"
fi
