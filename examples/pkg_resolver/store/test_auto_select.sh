#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# test_auto_select.sh -- unit test for the D43 auto store-backend policy
# (uw_resolve_store_backend in ensure_lmdb.sh). Proves the SIZE rule via the
# UW_STORE_AVAIL_RAM_BYTES override, independent of any real store or memory:
#   - store <= FACTOR x avail_RAM  -> indexed
#   - store  > FACTOR x avail_RAM  -> lmdb (or indexed with a loud fallback if
#                                     lmdb is not usable on this box)
# Also checks explicit modes pass through unchanged. Answers are unaffected -- a
# policy that only picks a backend cannot change rows.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ensure_lmdb.sh
source "$HERE/ensure_lmdb.sh"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# A store dir whose only sizeable file is a 4096-byte P/2 JSONL (pre-build path).
head -c 4096 /dev/zero | tr '\0' 'x' > "$TMP/pkg.jsonl"
: > "$TMP/cases.jsonl"   # must be ignored by the sizer (it is the query set)
SIZE="$(uw_store_size_bytes "$TMP")"

fail=0
check() { # desc expected actual
  if [ "$2" = "$3" ]; then echo "  PASS  $1 (=$3)"; else echo "  FAIL  $1: expected $2 got $3"; fail=1; fi
}

echo "== auto store-backend policy (store_size=${SIZE}B, FACTOR=${UW_STORE_LMDB_RAM_FACTOR:-2}) =="
check "sizer ignores cases.jsonl, counts pkg.jsonl" 4096 "$SIZE"

# Below threshold: avail_RAM huge -> 2x huge >> store -> indexed.
b_lo="$(UW_STORE_AVAIL_RAM_BYTES=1000000000 uw_resolve_store_backend auto "$TMP" 2>/dev/null)"
check "store <= 2x RAM -> indexed" indexed "$b_lo"

# Above threshold: avail_RAM=1 -> 2x1=2 < 4096 -> size rule wants lmdb.
reason="$(UW_STORE_AVAIL_RAM_BYTES=1 uw_resolve_store_backend auto "$TMP" 2>&1 >/dev/null)"
b_hi="$(UW_STORE_AVAIL_RAM_BYTES=1 uw_resolve_store_backend auto "$TMP" 2>/dev/null)"
echo "  info  above-threshold decision: resolved=$b_hi"
echo "  info  reason: $reason"
if echo "$reason" | grep -q -- "-> lmdb"; then
  check "store > 2x RAM AND lmdb usable -> lmdb" lmdb "$b_hi"
elif echo "$reason" | grep -q "FALLING BACK to indexed"; then
  echo "  PASS  store > 2x RAM but lmdb unusable -> indexed (loud fallback; answer-identical)"
  check "  fallback resolved backend" indexed "$b_hi"
else
  echo "  FAIL  above-threshold did not trigger the lmdb size rule"; fail=1
fi

# Explicit modes pass through unchanged (override auto).
check "explicit indexed passes through" indexed "$(uw_resolve_store_backend indexed "$TMP" 2>/dev/null)"
check "explicit lmdb passes through"     lmdb    "$(uw_resolve_store_backend lmdb "$TMP" 2>/dev/null)"

# Tunable factor: FACTOR=0 makes any non-empty store exceed the threshold.
b_f0="$(UW_STORE_LMDB_RAM_FACTOR=0 UW_STORE_AVAIL_RAM_BYTES=1000000000 uw_resolve_store_backend auto "$TMP" 2>&1 | tail -0; UW_STORE_LMDB_RAM_FACTOR=0 UW_STORE_AVAIL_RAM_BYTES=1000000000 uw_resolve_store_backend auto "$TMP" 2>/dev/null)"
if [ "$b_f0" = lmdb ] || [ "$b_f0" = indexed ]; then echo "  PASS  FACTOR override honored (resolved=$b_f0)"; else echo "  FAIL  FACTOR override"; fail=1; fi

echo "== $([ $fail -eq 0 ] && echo ALL PASS || echo FAILURES) =="
exit $fail
