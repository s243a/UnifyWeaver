#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# lmdb_smoke.sh — D43 policy: missing-package path is ungated; the real
# lmdb(Dir) arm is gated on `require('lmdb')`.
#
#   bash examples/pkg_resolver/store/lmdb_smoke.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

echo "== lmdb missing-package (ungated) =="
# Reuse the D43 fixture path: compiling lmdb(...) without the npm package
# must refuse loudly. The fact-sources suite already covers this; we echo
# the same error string from the codec so the policy sentence is pinned.
node -e '
const path = require("path");
const c = require("./scripts/js_wam/uw_fact_codec.js");
const msg = c.lmdbMissingError("store_pkg/2", "/tmp/uw-no-such-lmdb");
if (!/npm install lmdb/.test(msg) || !/not used as a fallback/.test(msg)) {
  process.stderr.write("lmdb missing error text drifted\n");
  process.exit(1);
}
process.stdout.write("lmdb_missing_error_ok\n");
'

echo "== lmdb store arm (gated) =="
if node -e "require('lmdb')" 2>/dev/null; then
  CORPUS="$HERE/.out/corpus"
  LMDIR="$HERE/.out/lmdb_pkg"
  mkdir -p "$LMDIR"
  node scripts/js_wam/uw_fact_lmdb.js build "$CORPUS/pkg.jsonl" "$LMDIR"
  echo "lmdb_arm_ran ok -> $LMDIR"
else
  echo "lmdb_arm_skipped (lmdb npm package not loadable)"
fi
