#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# test_explain_cmd.sh -- self-contained check for `abi_cli.pl explain <bin> <so> <rel>`.
# Builds the same tiny synthetic two-soname store as test_report_cmd.sh (no
# dpkg/gcc/readelf needed): liba is compatible at 1.0, libb's symbol first appears
# at 2.0 so it is below_floor at 1.0. Asserts on distinctive substrings.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORE="$(mktemp -d)"
trap 'rm -rf "$STORE"' EXIT

cat > "$STORE/symprov.jsonl" <<'JSONL'
["liba.so.1|foo@LIBA_1", ["since", "1.0", "1.0", "default"]]
["libb.so.1|bar@LIBB_1", ["since", "2.0", "2.0", "default"]]
JSONL
cat > "$STORE/symreq.jsonl" <<'JSONL'
["mybin|foo@LIBA_1", ["liba.so.1", "GLOBAL"]]
["mybin|bar@LIBB_1", ["libb.so.1", "GLOBAL"]]
JSONL
cat > "$STORE/needed.jsonl" <<'JSONL'
["mybin", "liba.so.1"]
["mybin", "libb.so.1"]
JSONL
cat > "$STORE/evidence.jsonl" <<'JSONL'
["provides|liba.so.1", ["symbols", "1.0", "curated", "test-fixture"]]
["provides|libb.so.1", ["symbols", "2.0", "curated", "test-fixture"]]
["requires|mybin", ["readelf", "complete", "ok"]]
JSONL
cat > "$STORE/releases.jsonl" <<'JSONL'
["liba.so.1", "1.0"]
["liba.so.1", "1.1"]
["libb.so.1", "2.0"]
JSONL
: > "$STORE/replaces.jsonl"

OUT1="$(swipl -q -g main -t halt "$HERE/abi_cli.pl" -- "$STORE" explain mybin libb.so.1 1.0)"
echo "$OUT1" | grep -q "explain mybin libb.so.1 1.0: incompatible" \
    || { echo "FAIL: explain header (incompatible) missing:"; echo "$OUT1"; exit 1; }
echo "$OUT1" | grep -q "first appears in release 2.0 (below_floor)" \
    || { echo "FAIL: explain below_floor reason line missing:"; echo "$OUT1"; exit 1; }
echo "PASS: explain surfaces the below_floor reason"

OUT2="$(swipl -q -g main -t halt "$HERE/abi_cli.pl" -- "$STORE" explain mybin liba.so.1 1.0)"
echo "$OUT2" | grep -q "explain mybin liba.so.1 1.0: compatible" \
    || { echo "FAIL: explain header (compatible) missing:"; echo "$OUT2"; exit 1; }
[ "$(echo "$OUT2" | grep -c '^  ')" -eq 0 ] \
    || { echo "FAIL: compatible explain should have no reason lines:"; echo "$OUT2"; exit 1; }
echo "PASS: explain on a compatible verdict prints only the header"

echo "== 2 passed, 0 failed =="
