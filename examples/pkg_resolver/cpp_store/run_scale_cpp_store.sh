#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_scale_cpp_store.sh -- the memory×scale CROSSOVER harness for the C++ WAM
# store lane. Runs the SAME workload (the 5k differential cases) through both
# backends under a sweep of memory caps, capturing wall-time + the D43
# bytes-read counters + (lmdb only) L1/L2 cache hit/miss, and prints a table.
#
#   bash examples/pkg_resolver/cpp_store/run_scale_cpp_store.sh
#   UW_MEM_CAPS="512M 128M 64M" bash .../run_scale_cpp_store.sh
#   UW_L2_CAPS="0 256 4096" bash .../run_scale_cpp_store.sh   # in-proc L2 axis
#
# WHY memory is a swept axis (per docs/design/CACHE_COST_MODEL_PHILOSOPHY.md's
# f_hot = min(1, R_free/W_working)): the `indexed` backend has NO application
# cache -- it does positioned ifstream reads and leans entirely on the OS page
# cache, so it should degrade as the memory cap tightens; the `lmdb` backend's
# bounded L1+L2 keep a hot working set resident, so it should degrade less.
# The interesting result is the CROSSOVER, not either column alone.
#
# NOTE: the shipped 5k store is ~1 MB, so it fits in almost any cap and the
# crossover is muted -- this harness is the MECHANISM; a dramatic crossover
# needs a much larger catalog (see the large-catalog TODO in README.md). It
# uses `systemd-run --scope -p MemoryMax -p MemorySwapMax=0` (cgroup v2) when
# available, else runs uncapped and says so.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/../store/.out/scale"
CASES="$SCALE/cases.jsonl"
BIN="$HERE/uw_resolve_wam_cpp_store/cpp/diff_uwresolve_store"
BACKENDS="${UW_BACKENDS:-indexed lmdb}"
MEM_CAPS="${UW_MEM_CAPS:-max 256M 64M}"
L2_CAPS="${UW_L2_CAPS:-}"

export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"
cd "$ROOT"

# Ensure the scale cases exist.
if [[ ! -f "$CASES" ]]; then
  node "$HERE/../store/gen_scale_catalog.mjs" "$SCALE"
  [[ -f "$SCALE/pkg.jsonl" ]] || node "$HERE/../store/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
fi

have_systemd_run=0
if command -v systemd-run >/dev/null 2>&1 && systemd-run --scope --quiet true >/dev/null 2>&1; then
  have_systemd_run=1
else
  echo "note: systemd-run --scope unavailable here; running UNCAPPED (memory axis is a no-op)." >&2
fi

# run_capped CAP CMD... : run CMD under a MemoryMax cap (no swap) if possible.
run_capped() {
  local cap="$1"; shift
  if [[ "$cap" == "max" || "$have_systemd_run" -eq 0 ]]; then
    "$@"
  else
    systemd-run --scope --quiet -p MemoryMax="$cap" -p MemorySwapMax=0 "$@"
  fi
}

# Build each backend once (idempotent; regenerates the per-store binary).
declare -A BIN_FOR
for be in $BACKENDS; do
  echo "== building $be backend ==" >&2
  STORE_DIR="$SCALE" UW_STORE_BACKEND="$be" bash "$HERE/build.sh" >/dev/null
  BIN_FOR[$be]="$HERE/uw_resolve_wam_cpp_store_$be"
  cp "$BIN" "${BIN_FOR[$be]}"
done

printf '%-8s %-8s %-10s %10s %14s %12s %10s %10s %10s\n' \
  backend memcap l2cap wall_ms io_bytes io_reads l1_hits l2_hits misses

l2_list="${L2_CAPS:-default}"
for be in $BACKENDS; do
  for cap in $MEM_CAPS; do
    for l2 in $l2_list; do
      attr=$(mktemp); tstart=$(date +%s%N)
      env_l2=(); [[ "$l2" != "default" ]] && env_l2=(UW_WAM_LMDB_L2_CAP="$l2")
      run_capped "$cap" env UW_WAM_CACHE_ATTRIBUTION=1 "${env_l2[@]}" \
        "${BIN_FOR[$be]}" < "$CASES" > /dev/null 2>"$attr" || true
      wall_ms=$(( ($(date +%s%N) - tstart) / 1000000 ))
      line=$(grep -o 'fact_io_bytes=[0-9]* fact_io_reads=[0-9]* l1_hits=[0-9]* l2_hits=[0-9]* misses=[0-9]*' "$attr" | head -1)
      iob=$(sed -n 's/.*fact_io_bytes=\([0-9]*\).*/\1/p' <<<"$line"); iob=${iob:-0}
      ior=$(sed -n 's/.*fact_io_reads=\([0-9]*\).*/\1/p' <<<"$line"); ior=${ior:-0}
      h1=$(sed -n 's/.*l1_hits=\([0-9]*\).*/\1/p' <<<"$line"); h1=${h1:-0}
      h2=$(sed -n 's/.*l2_hits=\([0-9]*\).*/\1/p' <<<"$line"); h2=${h2:-0}
      ms=$(sed -n 's/.*misses=\([0-9]*\).*/\1/p' <<<"$line"); ms=${ms:-0}
      printf '%-8s %-8s %-10s %10s %14s %12s %10s %10s %10s\n' \
        "$be" "$cap" "$l2" "$wall_ms" "$iob" "$ior" "$h1" "$h2" "$ms"
      rm -f "$attr"
    done
  done
done
