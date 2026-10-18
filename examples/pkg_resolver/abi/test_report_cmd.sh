#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# test_report_cmd.sh -- self-contained check for `abi_cli.pl report <binary>`.
# Builds a tiny synthetic two-soname store (no dpkg/gcc/readelf needed: two
# hand-written since() rows, one with a two-release axis so the range spans
# compatible(curated) + compatible(extrapolated)) and diffs the command's
# stdout against the known-good text below. Independent of run_abi_verify.sh:
# report_one/2 only calls already-verified abi_resolve.pl predicates
# (abi_floor/3, release_axis/2, abi_range/3), so this exercises the CLI
# wiring (arg parsing, soname ordering, formatting), not the resolver model.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORE="$(mktemp -d)"
trap 'rm -rf "$STORE"' EXIT

cat > "$STORE/symprov.jsonl" <<'EOF'
["liba.so.1|foo@LIBA_1", ["since", "1.0", "1.0", "default"]]
["libb.so.1|bar@LIBB_1", ["since", "2.0", "2.0", "default"]]
EOF
cat > "$STORE/symreq.jsonl" <<'EOF'
["mybin|foo@LIBA_1", ["liba.so.1", "GLOBAL"]]
["mybin|bar@LIBB_1", ["libb.so.1", "GLOBAL"]]
EOF
cat > "$STORE/needed.jsonl" <<'EOF'
["mybin", "liba.so.1"]
["mybin", "libb.so.1"]
EOF
cat > "$STORE/evidence.jsonl" <<'EOF'
["provides|liba.so.1", ["symbols", "1.0", "curated", "test-fixture"]]
["provides|libb.so.1", ["symbols", "2.0", "curated", "test-fixture"]]
["requires|mybin", ["readelf", "complete", "ok"]]
EOF
cat > "$STORE/releases.jsonl" <<'EOF'
["liba.so.1", "1.0"]
["liba.so.1", "1.1"]
["libb.so.1", "2.0"]
EOF
: > "$STORE/replaces.jsonl"

EXPECTED="$STORE/expected.txt"
cat > "$EXPECTED" <<'EOF'
liba.so.1:
  floor: 1.0
  axis:  [1.0,1.1]
  range: range('1.0','1.1',['1.0'-compatible(curated),'1.1'-compatible(extrapolated)])
libb.so.1:
  floor: 2.0
  axis:  [2.0]
  range: range('2.0','2.0',['2.0'-compatible(curated)])
EOF

ACTUAL="$STORE/actual.txt"
swipl -q -g main -t halt "$HERE/abi_cli.pl" -- "$STORE" report mybin > "$ACTUAL"
diff -u "$EXPECTED" "$ACTUAL" || { echo "FAIL: report mybin output mismatch" >&2; exit 1; }
echo "PASS: report mybin matches known-good output"

OUT2="$(swipl -q -g main -t halt "$HERE/abi_cli.pl" -- "$STORE" report nosuchbin)"
[ "$OUT2" = "report nosuchbin: no NEEDED sonames evidenced" ] || { echo "FAIL: empty-report line: $OUT2" >&2; exit 1; }
echo "PASS: report on a binary with no NEEDED evidence"

echo "== 2 passed, 0 failed =="
