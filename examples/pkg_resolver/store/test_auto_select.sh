#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# test_auto_select.sh -- unit test for the D43 auto store-backend policy
# (uw_resolve_store_backend in ensure_lmdb.sh). Proves the SIZE rule and the
# rows_per_key gate via the UW_STORE_AVAIL_RAM_BYTES override, independent of any
# real store or real memory. It STUBS uw_lmdb_cpp_usable so the test can never
# trigger a real `npm install --build-from-source` or a compiler probe -- the
# lmdb-usability path is exercised separately by the real build. Answers are
# unaffected: a policy that only picks a backend cannot change rows.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=ensure_lmdb.sh
source "$HERE/ensure_lmdb.sh"

# Stub the lmdb-usability probe (default: usable). TEST_LMDB_USABLE=1 -> usable.
uw_lmdb_cpp_usable() { [ "${TEST_LMDB_USABLE:-1}" = "1" ]; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# A store dir whose only sizeable file is a 4096-byte P/2 JSONL (pre-build path,
# rows_per_key unknown -> not gated).
mkdir -p "$TMP/s"
head -c 4096 /dev/zero | tr '\0' 'x' > "$TMP/s/pkg.jsonl"
: > "$TMP/s/cases.jsonl"   # must be ignored by the sizer (it is the query set)
SIZE="$(uw_store_size_bytes "$TMP/s")"

fail=0
check() { if [ "$2" = "$3" ]; then echo "  PASS  $1 (=$3)"; else echo "  FAIL  $1: expected $2 got $3"; fail=1; fi; }

echo "== auto policy (store_size=${SIZE}B, FACTOR=${UW_STORE_LMDB_RAM_FACTOR:-2}, MIN_RPK=${UW_STORE_LMDB_MIN_ROWS_PER_KEY:-2}) =="
check "sizer ignores cases.jsonl, counts pkg.jsonl" 4096 "$SIZE"

# Below threshold -> indexed (RAM huge).
check "store <= 2x RAM -> indexed" indexed \
  "$(UW_STORE_AVAIL_RAM_BYTES=1000000000 uw_resolve_store_backend auto "$TMP/s" 2>/dev/null)"

# Above threshold, rpk unknown (no .idx), lmdb usable -> lmdb.
check "store > 2x RAM, rpk unknown, usable -> lmdb" lmdb \
  "$(UW_STORE_AVAIL_RAM_BYTES=1 TEST_LMDB_USABLE=1 uw_resolve_store_backend auto "$TMP/s" 2>/dev/null)"

# Above threshold, lmdb NOT usable -> indexed (loud fallback).
check "store > 2x RAM but lmdb unusable -> indexed (fallback)" indexed \
  "$(UW_STORE_AVAIL_RAM_BYTES=1 TEST_LMDB_USABLE=0 uw_resolve_store_backend auto "$TMP/s" 2>/dev/null)"

# Explicit modes pass through.
check "explicit indexed passes through" indexed "$(uw_resolve_store_backend indexed "$TMP/s" 2>/dev/null)"
check "explicit lmdb passes through"     lmdb    "$(uw_resolve_store_backend lmdb "$TMP/s" 2>/dev/null)"

# Invalid FACTOR falls back to 2 (and still resolves).
check "invalid FACTOR -> defaults, resolves" indexed \
  "$(UW_STORE_LMDB_RAM_FACTOR=abc UW_STORE_AVAIL_RAM_BYTES=1000000000 uw_resolve_store_backend auto "$TMP/s" 2>/dev/null)"

# rows_per_key gate (needs a real UWIX index; skip if the indexer is unavailable).
INDEX="$ROOT/scripts/js_wam/uw_fact_index.js"
if command -v node >/dev/null 2>&1 && [ -f "$INDEX" ]; then
  mkdir -p "$TMP/rpk1" "$TMP/rpk3"
  printf '["k1","v0"]\n["k2","v0"]\n' > "$TMP/rpk1/pkg.jsonl"
  node "$INDEX" build "$TMP/rpk1/pkg.jsonl" "$TMP/rpk1/pkg" >/dev/null 2>&1
  printf '["k1","a"]\n["k1","b"]\n["k1","c"]\n["k2","a"]\n["k2","b"]\n["k2","c"]\n' > "$TMP/rpk3/pkg.jsonl"
  node "$INDEX" build "$TMP/rpk3/pkg.jsonl" "$TMP/rpk3/pkg" >/dev/null 2>&1
  echo "  info  rpk1 rows_per_key=$(uw_store_rows_per_key "$TMP/rpk1")  rpk3 rows_per_key=$(uw_store_rows_per_key "$TMP/rpk3")"
  check "store > 2x RAM, rows_per_key=1 -> indexed (gate)" indexed \
    "$(UW_STORE_AVAIL_RAM_BYTES=1 TEST_LMDB_USABLE=1 uw_resolve_store_backend auto "$TMP/rpk1" 2>/dev/null)"
  check "store > 2x RAM, rows_per_key=3 -> lmdb" lmdb \
    "$(UW_STORE_AVAIL_RAM_BYTES=1 TEST_LMDB_USABLE=1 uw_resolve_store_backend auto "$TMP/rpk3" 2>/dev/null)"
else
  echo "  SKIP  rows_per_key gate (node / uw_fact_index.js unavailable)"
fi

echo "== $([ $fail -eq 0 ] && echo ALL PASS || echo FAILURES) =="
exit $fail
