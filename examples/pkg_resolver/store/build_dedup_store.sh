#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# build_dedup_store.sh -- build a memory-efficient MULTI-SNAPSHOT store: one
# shared package POOL (deps/conflicts/provides/revdeps stored ONCE, keyed
# PoolId|Name) plus N tiny per-snapshot MEMBERSHIP tables (store_pkg, keyed
# SnapId|Name). Unchanged packages are shared across snapshots, so total size
# is ~O(unique versions + deltas), not O(snapshots x packages).
#
#   bash examples/pkg_resolver/store/build_dedup_store.sh DIR [BASE N ...gen flags]
#
# Writes:
#   DIR/pool.rich.jsonl, DIR/membership/snap-<t>.jsonl   (generator output)
#   DIR/pool/{dep,conflict,revdep,provide}.{jsonl,data,idx}   (shared pool, indexed)
#   DIR/snap-<t>/pkg.{jsonl,data,idx}                         (per-snapshot membership, indexed)
# Query the result with resolver_store_snapshot.pl (env_snap(SnapId, PoolId,...)).

set -euo pipefail

DIR="${1:?usage: build_dedup_store.sh DIR [--base=N --n=N ...]}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
INDEX="$ROOT/scripts/js_wam/uw_fact_index.js"
export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"

mkdir -p "$DIR"
echo "== generating snapshots =="
node "$HERE/gen_snapshots.mjs" --out="$DIR" "$@"

echo "== compiling shared pool -> P2 (keyed PoolId|Name) =="
node "$HERE/rich_to_p2.mjs" "$DIR/pool.rich.jsonl" "$DIR/pool"
# pool/pkg.jsonl (the pool's package rows) is not part of the dedup store -- the
# per-snapshot membership below is the store_pkg table. Index only the pooled
# dep/conflict/revdep/provide tables.
for t in dep conflict revdep provide; do
  node "$INDEX" build "$DIR/pool/$t.jsonl" "$DIR/pool/$t" >/dev/null
done

echo "== building + indexing per-snapshot membership (store_pkg) =="
n=0
for mf in "$DIR"/membership/snap-*.jsonl; do
  t="$(basename "$mf" .jsonl)"; t="${t#snap-}"
  sd="$DIR/snap-$t"
  node "$HERE/build_membership_p2.mjs" "$mf" "$sd" "snap$t" >/dev/null
  node "$INDEX" build "$sd/pkg.jsonl" "$sd/pkg" >/dev/null
  n=$((n+1))
done

pool_bytes=$(cat "$DIR"/pool/*.data "$DIR"/pool/*.idx 2>/dev/null | wc -c)
memb_bytes=$(cat "$DIR"/snap-*/*.data "$DIR"/snap-*/*.idx 2>/dev/null | wc -c)
echo "build_dedup_store: $n snapshots; pool=$pool_bytes B + membership=$memb_bytes B = $((pool_bytes+memb_bytes)) B under $DIR"
echo "  query: SnapId=snap<t>  PoolId=s5k-snap  (resolver_store_snapshot.pl / store_diff_runner_snap.pl)"
