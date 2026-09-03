#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# lmdb_smoke.sh — D43 policy: missing-package path is ungated; the real
# lmdb(Dir) arm is gated on `require('lmdb')` (via /tmp-prefixed install).
#
#   bash examples/pkg_resolver/store/lmdb_smoke.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
# shellcheck source=ensure_lmdb.sh
source "$HERE/ensure_lmdb.sh"

echo "== lmdb missing-package (ungated) =="
# Reuse the D43 fixture path: compiling lmdb(...) without the npm package
# must refuse loudly. The fact-sources suite already covers this; we echo
# the same error string from the codec so the policy sentence is pinned.
node -e '
const c = require("./scripts/js_wam/uw_fact_codec.js");
const msg = c.lmdbMissingError("store_pkg/2", "/tmp/uw-no-such-lmdb");
if (!/npm install lmdb/.test(msg) || !/not used as a fallback/.test(msg)) {
  process.stderr.write("lmdb missing error text drifted\n");
  process.exit(1);
}
process.stdout.write("lmdb_missing_error_ok\n");
'

echo "== runtime UW_LMDB_FORCE_MISSING (ungated) =="
# The generated runtime throws the same string without installing lmdb.
node -e '
const fs = require("fs");
const path = require("path");
const rt = path.join("examples/pkg_resolver/wamjs_store/js/wam_runtime.js");
if (!fs.existsSync(rt)) {
  process.stdout.write("lmdb_force_missing_skipped (runtime not built yet)\n");
  process.exit(0);
}
process.env.UW_LMDB_FORCE_MISSING = "1";
const { Runtime } = require(path.resolve(rt));
try {
  Runtime.lmdb_missing_error("store_pkg/2", "/tmp/uw-no-such-lmdb");
  const err = Runtime.lmdb_missing_error("store_pkg/2", "/tmp/uw-no-such-lmdb");
  if (!/npm install lmdb/.test(err) || !/not used as a fallback/.test(err)) {
    process.stderr.write("runtime missing-error text drifted\n");
    process.exit(1);
  }
  process.stdout.write("lmdb_force_missing_ok\n");
} catch (e) {
  process.stderr.write(String(e) + "\n");
  process.exit(1);
}
'

echo "== lmdb store arm (gated) =="
if uw_ensure_lmdb; then
  CORPUS="$HERE/.out/corpus"
  mkdir -p "$CORPUS"
  if [[ ! -f "$CORPUS/pkg.jsonl" ]]; then
    swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- "$CORPUS"
  fi
  bash "$HERE/build_lmdb_stores.sh" "$CORPUS"
  echo "lmdb_arm_ran ok -> $CORPUS/lmdb/{pkg,dep,conflict,revdep}"
else
  echo "lmdb_arm_skipped (lmdb npm package not loadable)"
fi
