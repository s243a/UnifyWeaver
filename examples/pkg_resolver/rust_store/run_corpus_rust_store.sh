#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_corpus_rust_store.sh -- drive the P3 contract corpus through the Rust WAM
# store-backed build (D43 indexed seek stores) and compare every result to SWI
# (via cases.jsonl `expected`). Also asserts the store results are IDENTICAL to
# the Rust term-catalog corpus (examples/pkg_resolver/rust). Rust lane of
# examples/pkg_resolver/go_store/run_corpus_go_store.sh.
#
#   bash examples/pkg_resolver/rust_store/run_corpus_rust_store.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STORE="${STORE_DIR:-$HERE/../store/.out/corpus}"
BACKEND="${UW_STORE_BACKEND:-indexed}"
OUT="$HERE/.corpus_out"
BIN="$HERE/uw_resolve_wam_store/target/release/uw_resolve_store"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -f "$STORE/cases.jsonl" ]]; then
  swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$STORE"
fi

STORE_DIR="$STORE" UW_STORE_BACKEND="$BACKEND" bash "$HERE/build.sh"

"$BIN" --corpus < "$STORE/cases.jsonl" > "$OUT/rust_store.jsonl"

# Assert IDENTICAL to the Rust term-catalog corpus (not just "both pass").
TERM_OUT="$HERE/../rust/.corpus_out/rust.jsonl"
if [[ ! -f "$TERM_OUT" ]]; then
  bash "$HERE/../rust/run_corpus_rust.sh" >/dev/null 2>&1 || true
fi
if [[ -f "$TERM_OUT" ]]; then
  python3 - "$TERM_OUT" "$OUT/rust_store.jsonl" <<'PY'
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
    print("rust_store corpus NOT identical to term corpus")
    sys.exit(1)
print("rust_store corpus IDENTICAL to term corpus (%d/%d)" % (len(common), len(common)))
PY
else
  echo "note: term corpus output not found; skipped identity assertion"
fi
