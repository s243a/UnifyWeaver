#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# interval_diff_check.sh -- prove interval-mode resolves == per-snapshot
# store_pkg replay resolves, byte-for-byte, across several snapshots.
#
#   bash examples/pkg_resolver/store/interval_diff_check.sh [DIR] [BASE] [N] [t...]
#
# Builds (if missing) a small multi-snapshot dedup store, folds intervals, then
# for each snapshot t: generates ~80 cases from t's replayed membership and diffs
#   REPLAY  (store_pkg path : STORE_DIR=rep<t> UW_POOL_DIR=pool UW_SNAP_ID=snap<t>)
#   INTERVAL(intervals path : UW_INTERVAL_MODE=1 UW_INTERVALS_FILE=ivl UW_SNAP_ID=t)
# Both use PoolId=s5k-snap and the SAME cases file. Any nonempty diff is a bug.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DIR="${1:-$HERE/.out/ivlcheck}"
BASE="${2:-300}"
N="${3:-20}"
shift $(( $# > 3 ? 3 : $# )) || true
TS=("$@")
[[ ${#TS[@]} -eq 0 ]] && TS=(0 5 12 19)
POOL_ID="s5k-snap"

cd "$ROOT"
if [[ ! -f "$DIR/ivl/intervals.jsonl" ]]; then
  echo "== building dedup store ($BASE base, $N snapshots) under $DIR =="
  bash "$HERE/build_dedup_store.sh" "$DIR" --base="$BASE" --n="$N"
  node "$HERE/materialize_membership.mjs" --git="$DIR/membership-git" --intervals --out="$DIR/ivl"
fi

RUNNER="$ROOT/examples/pkg_resolver/store_diff_runner_snap.pl"
fail=0
for t in "${TS[@]}"; do
  rep="$DIR/rep$t"
  node "$HERE/materialize_membership.mjs" --git="$DIR/membership-git" \
       --snapshot="$t" --snapid="snap$t" --out="$rep" 2>/dev/null
  cases="$DIR/cases-$t.jsonl"
  node "$HERE/gen_snap_cases.mjs" "$rep/pkg.jsonl" "snap$t" > "$cases"

  # REPLAY (store_pkg path)
  STORE_DIR="$rep" UW_POOL_DIR="$DIR/pool" UW_SNAP_ID="snap$t" UW_POOL_ID="$POOL_ID" \
    swipl -q -g main -t halt "$RUNNER" < "$cases" > "$DIR/replay-$t.jsonl"
  # INTERVAL (validity-interval path); T carried as the numeric snapshot index
  UW_INTERVAL_MODE=1 UW_INTERVALS_FILE="$DIR/ivl/intervals.jsonl" \
    UW_POOL_DIR="$DIR/pool" UW_SNAP_ID="$t" UW_POOL_ID="$POOL_ID" \
    swipl -q -g main -t halt "$RUNNER" < "$cases" > "$DIR/interval-$t.jsonl"

  if diff "$DIR/replay-$t.jsonl" "$DIR/interval-$t.jsonl" > "$DIR/diff-$t.txt"; then
    lines="$(wc -l < "$DIR/replay-$t.jsonl")"
    echo "t=$t: 0 divergences ($lines cases)"
  else
    echo "t=$t: DIVERGENCE"
    head "$DIR/diff-$t.txt"
    fail=1
  fi
done

if [[ "$fail" -eq 0 ]]; then
  echo "ALL SNAPSHOTS: 0 divergences (interval-mode == store_pkg replay)"
else
  echo "FAILURES present"; exit 1
fi
