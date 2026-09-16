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
#   choose LMDB iff  store_size_bytes > FACTOR * available_RAM_bytes
#              AND   rows_per_key >= MIN_RPK   (lmdb buys ~nothing at ~1 row/key)
#              AND   lmdb is usable for the C++ lane (system liblmdb links, and a
#                    built lmdb store, if present, opens)
#   else INDEXED. FACTOR (UW_STORE_LMDB_RAM_FACTOR, default 2) is conservative
#   headroom over the ~1x-RAM crossover onset; MIN_RPK
#   (UW_STORE_LMDB_MIN_ROWS_PER_KEY, default 2) reflects that the asymptotic lmdb
#   win is ~= rows_per_key, so a 1-row/key store (e.g. ABI symprov, 1.03) never
#   benefits. NOTE: stat/od are GNU/coreutils (Linux); fine for this lane.
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

# Read a little-endian u32 at byte OFFSET of FILE (assembled from 4 bytes, so it
# is correct on any host endianness). Echoes the value or "" on short read.
uw_le_u32_at() {  # FILE OFFSET -> u32 | ""
  local f="$1" off="$2"; local -a b
  # shellcheck disable=SC2207
  b=($(od -An -tu1 -j"$off" -N4 "$f" 2>/dev/null))
  [ "${#b[@]}" -eq 4 ] || { echo ""; return 0; }
  echo $(( b[0] + b[1] * 256 + b[2] * 65536 + b[3] * 16777216 ))
}

# Aggregate rows_per_key across the store's UWIX indexes (cheap: read the header
# of each *.idx -- n_keys at byte 8, n_records at byte 20, both u32 LE). Echoes a
# 2-dp float, or "" when no index exists yet (pre-build -> caller treats as
# unknown and does not gate on it).
uw_store_rows_per_key() {  # DIR -> float | ""
  local dir="${1:?usage: uw_store_rows_per_key DIR}" recs=0 keys=0 f had=0 nk nr
  shopt -s nullglob
  for f in "$dir"/*.idx; do
    # UWIX header integers are LITTLE-ENDIAN (runtime uses seek_le_u32). od -tu4
    # would read NATIVE-endian -> byte-swapped garbage on a BE host, so assemble
    # each u32 explicitly LE from four -tu1 bytes.
    nk=$(uw_le_u32_at "$f" 8)
    nr=$(uw_le_u32_at "$f" 20)
    if [ -n "$nk" ] && [ -n "$nr" ]; then keys=$(( keys + nk )); recs=$(( recs + nr )); had=1; fi
  done
  shopt -u nullglob
  if [ "$had" -eq 1 ] && [ "$keys" -gt 0 ]; then
    awk "BEGIN{printf \"%.2f\", $recs/$keys}"
  else
    echo ""
  fi
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

# Is lmdb actually usable for the C++ store lane? Checks what the C++ build/read
# needs -- NOT just that the npm module loads:
#   1. uw_ensure_lmdb (the v1-format npm module, used to BUILD the store),
#   2. system liblmdb links (<lmdb.h> + -llmdb) -- the C++ reader needs it,
#   3. best-effort: an already-built lmdb store under DIR/lmdb/* OPENS with
#      vanilla liblmdb (catches MDB_INVALID from a wrong page format).
# Returns 0 if usable, non-zero otherwise. All output suppressed.
uw_lmdb_cpp_usable() {  # [DIR]
  local dir="${1:-}" cxx="${CXX:-g++}"
  uw_ensure_lmdb >/dev/null 2>&1 || return 1
  # Genuine LINK test: the source calls a real symbol (mdb_version) so the linker
  # must resolve liblmdb. The stdin source `-` MUST come BEFORE `-llmdb`: linkers
  # resolve libraries in argument order, so `-llmdb -` leaves the object's mdb_*
  # references undefined and the link ALWAYS fails -- which would make this probe
  # dead code (function returns "usable" on every host).
  printf '#include <lmdb.h>\nint main(){int a,b,c;(void)mdb_version(&a,&b,&c);return 0;}\n' \
    | "$cxx" -x c++ -std=c++17 -O0 - -llmdb -o /dev/null >/dev/null 2>&1 || return 1
  # best-effort smoke-open of a built store (first lmdb sub-env under DIR/lmdb)
  if [ -n "$dir" ]; then
    local envdir=""
    shopt -s nullglob
    local d; for d in "$dir"/lmdb/*/ "$dir"/lmdb_v1/*/; do [ -f "$d/data.mdb" ] && { envdir="${d%/}"; break; }; done
    shopt -u nullglob
    if [ -n "$envdir" ]; then
      local probe; probe="$(mktemp -d)/mdbprobe"
      if printf '#include <lmdb.h>\nint main(int c,char**v){MDB_env*e;if(mdb_env_create(&e))return 1;int rc=mdb_env_open(e,v[1],MDB_RDONLY|MDB_NOTLS,0664);int ok=(rc==0);if(ok){MDB_txn*t;MDB_dbi d;if(mdb_txn_begin(e,0,MDB_RDONLY,&t)==0){if(mdb_dbi_open(t,0,0,&d)!=0)ok=0;mdb_txn_abort(t);}else ok=0;}mdb_env_close(e);return ok?0:2;}\n' \
           | "$cxx" -x c++ -std=c++17 -O0 - -llmdb -o "$probe" >/dev/null 2>&1; then
        "$probe" "$envdir" >/dev/null 2>&1 || { rm -rf "$(dirname "$probe")"; return 1; }
      fi
      rm -rf "$(dirname "$probe")"
    fi
  fi
  return 0
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
  local min_rpk="${UW_STORE_LMDB_MIN_ROWS_PER_KEY:-2}"
  case "$factor" in ''|*[!0-9]*) echo "uw_store auto: UW_STORE_LMDB_RAM_FACTOR='$factor' is not a non-negative integer -> using 2" >&2; factor=2 ;; esac
  # Validate min_rpk too: it is interpolated into an awk program below, so a
  # non-numeric value would be awk-syntax injection / a silent gate skip.
  case "$min_rpk" in ''|*[!0-9.]*|*.*.*) echo "uw_store auto: UW_STORE_LMDB_MIN_ROWS_PER_KEY='$min_rpk' is not numeric -> using 2" >&2; min_rpk=2 ;; esac
  local store ram threshold rpk
  store=$(uw_store_size_bytes "$dir")
  ram=$(uw_available_ram_bytes)
  threshold=$(( ram * factor ))
  if [ "$store" -le "$threshold" ]; then
    echo "uw_store auto: store=${store}B <= ${factor}x avail_RAM(${ram}B)=${threshold}B -> indexed" >&2
    echo indexed; return 0
  fi
  # Above the size threshold: gate on rows_per_key (lmdb's asymptotic win ~= rpk).
  rpk=$(uw_store_rows_per_key "$dir")
  if [ -n "$rpk" ] && awk "BEGIN{exit !($rpk < $min_rpk)}"; then
    echo "uw_store auto: store=${store}B > ${factor}x avail_RAM(${ram}B)=${threshold}B BUT rows_per_key=${rpk} < ${min_rpk} -> indexed (lmdb buys ~nothing at ~1 row/key)" >&2
    echo indexed; return 0
  fi
  # Size + rows_per_key (or unknown rpk) favor lmdb; require it to be usable.
  if uw_lmdb_cpp_usable "$dir"; then
    echo "uw_store auto: store=${store}B > ${factor}x avail_RAM(${ram}B)=${threshold}B, rows_per_key=${rpk:-unknown} -> lmdb" >&2
    echo lmdb; return 0
  fi
  echo "uw_store auto: store=${store}B > ${factor}x avail_RAM(${ram}B)=${threshold}B, rows_per_key=${rpk:-unknown} would pick lmdb, but lmdb is NOT usable for the C++ lane (missing liblmdb / npm module / MDB_INVALID store) -- FALLING BACK to indexed (answer-identical)" >&2
  echo indexed; return 0
}
