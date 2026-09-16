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

# UW_LMDB_DATA_V1=1 is an ADDITIVE opt-in for native readers that link vanilla
# system liblmdb (e.g. the C++ cpp_store lazy LMDB backend): the default `lmdb`
# npm prebuilt is a Symas fork whose on-disk page format vanilla liblmdb rejects
# with MDB_INVALID. When set, install into a SEPARATE prefix and force a
# from-source rebuild with LMDB_DATA_V1=true (OpenLDAP-0.9.29 lineage, vanilla-
# compatible). Every existing caller (unset) keeps the fast prebuilt path.
if [ "${UW_LMDB_DATA_V1:-}" = "1" ]; then
  UW_LMDB_PREFIX="${UW_LMDB_PREFIX:-/tmp/uw-lmdb-pkg-v1}"
else
  UW_LMDB_PREFIX="${UW_LMDB_PREFIX:-/tmp/uw-lmdb-pkg}"
fi

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
  if [ "${UW_LMDB_DATA_V1:-}" = "1" ]; then
    # Force a from-source rebuild against the vanilla-compatible v1 data format.
    # Plain `npm install` (even with LMDB_DATA_V1 set) reuses the cached Symas
    # prebuilt and ignores the env var -- --build-from-source is required.
    if ! LMDB_DATA_V1=true npm install --prefix "$UW_LMDB_PREFIX" --build-from-source lmdb >/tmp/uw-lmdb-npm.log 2>&1; then
      uw_lmdb_missing_msg "build" "$UW_LMDB_PREFIX"
      echo "uw_ensure_lmdb: LMDB_DATA_V1 from-source install into $UW_LMDB_PREFIX failed (see /tmp/uw-lmdb-npm.log)" >&2
      return 1
    fi
  elif ! npm install --prefix "$UW_LMDB_PREFIX" lmdb >/tmp/uw-lmdb-npm.log 2>&1; then
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

# ---------------------------------------------------------------------------
# D43 auto store-backend selection (POLICY layer -- never changes ANSWERS; both
# backends return identical rows). Documented in
# examples/pkg_resolver/abi/bench/BACKEND_SELECTION.md.
#
#   choose LMDB iff  store_size_bytes > FACTOR * available_RAM_bytes  AND lmdb
#   is usable; else INDEXED. FACTOR (UW_STORE_LMDB_RAM_FACTOR, default 2) is a
#   conservative headroom over the ~1x-RAM crossover onset. Size-only for now
#   (a future refinement folds in rows_per_key -- see the doc).
# ---------------------------------------------------------------------------

# Sum the store bytes under DIR: the BUILT indexed store (*.data + *.idx) when
# present, else the source P/2 JSONL (*.jsonl) as a pre-build estimate.
uw_store_size_bytes() {  # DIR -> bytes
  local dir="${1:?usage: uw_store_size_bytes DIR}" total=0 f had_built=0
  shopt -s nullglob
  for f in "$dir"/*.data "$dir"/*.idx; do
    had_built=1; total=$(( total + $(stat -c%s "$f" 2>/dev/null || echo 0) ))
  done
  if [ "$had_built" -eq 0 ]; then
    for f in "$dir"/*.jsonl; do
      case "$f" in *cases.jsonl) continue;; esac  # cases.jsonl is the query set, not the store
      total=$(( total + $(stat -c%s "$f" 2>/dev/null || echo 0) ))
    done
  fi
  shopt -u nullglob
  echo "$total"
}

# Available RAM in bytes: UW_STORE_AVAIL_RAM_BYTES override (WSL2 MemAvailable
# balloons, so an override matters for testing/reproducibility) else
# /proc/meminfo MemAvailable.
uw_available_ram_bytes() {
  if [ -n "${UW_STORE_AVAIL_RAM_BYTES:-}" ]; then echo "$UW_STORE_AVAIL_RAM_BYTES"; return 0; fi
  local kb
  kb=$(awk '/^MemAvailable:/{print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)
  echo $(( kb * 1024 ))
}

# Resolve a requested backend (auto|indexed|lmdb) to a CONCRETE one.
# Echoes indexed|lmdb on stdout; all diagnostics go to stderr so callers can
# capture the choice with $(...). auto that wants lmdb but finds it unusable
# WARNS and falls back to indexed (answer-identical, so safe).
uw_resolve_store_backend() {  # MODE DIR -> echoes indexed|lmdb
  local mode="${1:?usage: uw_resolve_store_backend MODE DIR}" dir="${2:?}"
  case "$mode" in
    indexed|lmdb) echo "$mode"; return 0 ;;
    auto) : ;;
    *) echo "uw_resolve_store_backend: unknown backend '$mode' (auto|indexed|lmdb)" >&2; return 2 ;;
  esac
  local factor="${UW_STORE_LMDB_RAM_FACTOR:-2}"
  local store ram threshold
  store=$(uw_store_size_bytes "$dir")
  ram=$(uw_available_ram_bytes)
  threshold=$(( ram * factor ))
  if [ "$store" -gt "$threshold" ]; then
    if uw_ensure_lmdb >/dev/null 2>&1; then
      echo "uw_store auto: store=${store}B > ${factor}x avail_RAM(${ram}B)=${threshold}B -> lmdb" >&2
      echo lmdb; return 0
    fi
    echo "uw_store auto: store=${store}B > ${factor}x avail_RAM(${ram}B)=${threshold}B would pick lmdb, but lmdb is NOT usable (uw_ensure_lmdb failed / MDB_INVALID) -- FALLING BACK to indexed (answer-identical)" >&2
    echo indexed; return 0
  fi
  echo "uw_store auto: store=${store}B <= ${factor}x avail_RAM(${ram}B)=${threshold}B -> indexed" >&2
  echo indexed; return 0
}
