#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# Index P/2 JSONL dumps as LMDB environments (D43 backend A).
#
#   bash examples/pkg_resolver/store/build_lmdb_stores.sh DIR
#
# Expects DIR/{pkg,dep,conflict,revdep}.jsonl
# Writes DIR/lmdb/{pkg,dep,conflict,revdep}/ (one LMDB env per store).
# Loud error if the `lmdb` npm package is missing; never falls back
# to indexed(...).

set -euo pipefail

DIR="${1:?usage: build_lmdb_stores.sh DIR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck source=ensure_lmdb.sh
source "$ROOT/examples/pkg_resolver/store/ensure_lmdb.sh"
uw_require_lmdb

LMDB_JS="$ROOT/scripts/js_wam/uw_fact_lmdb.js"
PARENT="$DIR/lmdb"
mkdir -p "$PARENT"
node "$LMDB_JS" build-all "$DIR" "$PARENT"
echo "build_lmdb_stores.sh: wrote $PARENT/{pkg,dep,conflict,revdep}"
