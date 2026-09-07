#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# ensure_lmdb.sh -- D43 policy: opt-in `lmdb` npm package via a /tmp-prefixed
# install. NEVER a repo package.json dependency. NEVER a silent fallback
# to indexed(...).
#
#   source examples/pkg_resolver/store/ensure_lmdb.sh
#   uw_ensure_lmdb            # try install; return 1 if still missing
#   uw_require_lmdb           # loud exit 1 if missing
#
# Sets NODE_PATH to $UW_LMDB_PREFIX/node_modules on success.

UW_LMDB_PREFIX="${UW_LMDB_PREFIX:-/tmp/uw-lmdb-pkg}"

uw_lmdb_missing_msg() {
  local pred="${1:-store_pkg/2}"
  local path="${2:-$UW_LMDB_PREFIX}"
  node -e '
const c = require("./scripts/js_wam/uw_fact_codec.js");
process.stderr.write(c.lmdbMissingError(process.argv[1], process.argv[2]) + "\n");
' "$pred" "$path"
}

uw_lmdb_loadable() {
  NODE_PATH="$UW_LMDB_PREFIX/node_modules${NODE_PATH:+:$NODE_PATH}" \
    node -e "require('lmdb')" >/dev/null 2>&1
}

uw_ensure_lmdb() {
  if uw_lmdb_loadable; then
    export NODE_PATH="$UW_LMDB_PREFIX/node_modules${NODE_PATH:+:$NODE_PATH}"
    return 0
  fi
  mkdir -p "$UW_LMDB_PREFIX"
  if ! npm install --prefix "$UW_LMDB_PREFIX" lmdb >/tmp/uw-lmdb-npm.log 2>&1; then
    uw_lmdb_missing_msg "build" "$UW_LMDB_PREFIX"
    echo "uw_ensure_lmdb: npm install --prefix $UW_LMDB_PREFIX lmdb failed (see /tmp/uw-lmdb-npm.log)" >&2
    return 1
  fi
  if ! uw_lmdb_loadable; then
    uw_lmdb_missing_msg "build" "$UW_LMDB_PREFIX"
    return 1
  fi
  export NODE_PATH="$UW_LMDB_PREFIX/node_modules${NODE_PATH:+:$NODE_PATH}"
  return 0
}

uw_require_lmdb() {
  if uw_ensure_lmdb; then
    return 0
  fi
  echo "indexed(...) is a different format and is not used as a fallback." >&2
  exit 1
}
