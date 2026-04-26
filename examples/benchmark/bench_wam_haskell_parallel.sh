#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Alternating-order parallel-scaling benchmark for the WAM Haskell
# enwiki/dupsort path.  Avoids the cache-progression artifact that
# sequential N1/N2/N4 blocks suffer from: with limited RAM and a
# warm-up phase, the first config sees cold cache and every later
# config sees a progressively warmer one, producing fake speedup
# numbers that exceed the physical core count.
#
# This harness:
#   1. Runs one untimed warmup pass to mmap-fault the LMDB
#   2. Alternates -N1 / -N4 across iterations so cache state hits
#      both configs equally
#   3. Captures +RTS -s GC and spark stats per run
#
# Usage:
#   ./bench_wam_haskell_parallel.sh <binary> <lmdb-project-dir> [iters]
#
# Example:
#   ./bench_wam_haskell_parallel.sh \
#       /tmp/enwiki-wam-parallel/dist-newstyle/build/.../wam-haskell-enwiki \
#       data/benchmark/enwiki_cats/lmdb_proj \
#       8

set -euo pipefail

BIN="${1:-}"
LMDB_DIR="${2:-}"
ITERS="${3:-8}"

if [[ -z "$BIN" || -z "$LMDB_DIR" ]]; then
    echo "usage: $0 <binary> <lmdb-project-dir> [iters]" >&2
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not executable: $BIN" >&2
    exit 1
fi

if [[ ! -d "$LMDB_DIR/lmdb" ]]; then
    echo "error: missing $LMDB_DIR/lmdb" >&2
    exit 1
fi

LMDB_FILE="$LMDB_DIR/lmdb/data.mdb"

echo "=== environment ==="
free -h | head -3
echo "LMDB size: $(du -m "$LMDB_FILE" | cut -f1) MiB"
echo

echo "=== warmup (untimed) ==="
"$BIN" +RTS -N1 -RTS "$LMDB_DIR" > /dev/null 2>&1 || true
echo "done"
echo

echo "=== alternating -N1 / -N4 (-A32M nursery) ==="
# A 32MB nursery dramatically reduces parallel GC pressure on this
# workload.  With the default 1MB nursery, -N4 spends ~60% of its
# time in stop-the-world Gen0 GC, masking any parallel speedup.
# With -A32M the program barely GCs at all, exposing real scaling.
for ((i=1; i<=ITERS; i++)); do
    if (( i % 2 )); then N=1; else N=4; fi
    echo "--- iter$i -N$N -A32M ---"
    "$BIN" +RTS -N$N -A32M -s -RTS "$LMDB_DIR" 2>&1 | \
        grep -E "query_ms|tuple_count|MUT time|GC time|SPARKS|Productivity|work balance" | \
        sed 's/^/  /'
done
