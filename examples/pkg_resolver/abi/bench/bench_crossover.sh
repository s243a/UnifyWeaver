#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# bench_crossover.sh -- one-shot reproducible entry point for the lmdb-vs-indexed
# store-backend CROSSOVER benchmark on the UnifyWeaver ABI symbol store.
#
# It drives the C++ WAM SeekFactSource read path DIRECTLY (indexed = on-disk
# UWFI/UWIX positioned seeks relying on the OS page cache; lmdb = the C++ lazy
# reader with the L1 direct-mapped + L2 FIFO app caches). NOT the JS/wamjs lmdb
# backend. See RESULTS.md for the finding.
#
# Pipeline:
#   1. ensure store artifacts exist under .out/bench (packed P/2 JSONL, indexed
#      idx/, and a system-liblmdb-compatible lmdb_v1/). Rebuilds lmdb_v1 from the
#      packed JSONL if absent (needs the from-source v1 lmdb via ensure_lmdb.sh).
#   2. generate the query workloads (skewed / uniform / unique) from real keys.
#   3. codegen + g++ the two lookup binaries (bench_indexed, bench_lmdb).
#   4. sweep backend x workload x R x {warm,cold} with min-of-N wall time and the
#      deterministic I/O + cache-hit stats; write results.jsonl + a markdown table.
#
# Memory pressure: this host (WSL2, 10GB) offers NO hard memory-cap mechanism to
# an unprivileged user (systemd-run --user has no bus; system scope needs
# interactive auth; cgroup v2 is not delegated; no root for drop_caches). The
# cold variant instead evicts the store from the page cache per-run via
# posix_fadvise(DONTNEED) (UW_BENCH_EVICT=1) -- the root-free "store not
# resident" proxy. Because the store is far smaller than RAM, cold ~= warm here;
# see RESULTS.md for why a disk-bound crossover is unreachable on this box.
#
# Env knobs:
#   N_REPEAT  (default 5)   min-of-N wall repeats per cell
#   R_LIST    (default "1 3 5 10")  in-process repeat counts
#   WORKLOADS (default "skewed uniform unique")
#   L2_CAP    (default 65536)  UW_WAM_LMDB_L2_CAP for the lmdb backend
#   WL_N / WL_HOT / WL_MISS / WL_SEED  workload generator params

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="$HERE/.out/bench"          # abi/.out/bench (matches the pre-built store dir)
ABIOUT="$ROOT/examples/pkg_resolver/abi/.out/bench"
# The pre-built store lives under examples/pkg_resolver/abi/.out/bench; HERE is
# examples/pkg_resolver/abi/bench, so its sibling .out is the abi one.
OUT="$ABIOUT"

export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"
N_REPEAT="${N_REPEAT:-5}"
R_LIST="${R_LIST:-1 3 5 10}"
WORKLOADS="${WORKLOADS:-skewed uniform unique}"
L2_CAP="${L2_CAP:-65536}"

PACKED="$OUT/symprov.p2.jsonl"
IDX="$OUT/idx/symprov"
LMDB="$OUT/lmdb_v1/symprov"

echo "== bench_crossover: root=$ROOT out=$OUT =="

[ -f "$PACKED" ] || { echo "missing packed store $PACKED. Rebuild recipe is in the task/README." >&2; exit 1; }
[ -f "$IDX.data" ] && [ -f "$IDX.idx" ] || { echo "missing indexed store $IDX.{data,idx}" >&2; exit 1; }

# --- v1-compatible lmdb store (system liblmdb reads this; the pre-built lmdb/ is
# the lmdb-js Symas fork format that vanilla liblmdb rejects with MDB_INVALID) --
if [ ! -f "$LMDB/data.mdb" ]; then
  echo "== building v1-compatible lmdb store (from-source lmdb) =="
  export UW_LMDB_DATA_V1=1
  # shellcheck source=../store/ensure_lmdb.sh
  source "$ROOT/examples/pkg_resolver/store/ensure_lmdb.sh"
  uw_require_lmdb
  mkdir -p "$OUT/lmdb_v1"
  node "$ROOT/scripts/js_wam/uw_fact_lmdb.js" build "$PACKED" "$LMDB"
fi

# --- workloads ---
if [ ! -f "$OUT/wl.skewed.keys" ]; then
  echo "== generating workloads =="
  node "$HERE/gen_workload.mjs" "$PACKED" "$OUT/wl" \
    "${WL_N:-50000}" "${WL_HOT:-0.80}" "${WL_MISS:-0.15}" "${WL_SEED:-1234567}" >/dev/null
fi
if [ ! -f "$OUT/wl.unique.keys" ]; then
  node -e '
const fs=require("fs");const rl=require("readline").createInterface({input:fs.createReadStream(process.argv[1])});
const keys=[];rl.on("line",l=>{const t=l.trim();if(!t)return;try{keys.push(JSON.parse(t)[0])}catch{}});
rl.on("close",()=>{let s=987654321>>>0;const rnd=()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};
for(let i=keys.length-1;i>0;i--){const j=Math.floor(rnd()*(i+1));[keys[i],keys[j]]=[keys[j],keys[i]];}
fs.writeFileSync(process.argv[2],keys.join("\n")+"\n");});
' "$PACKED" "$OUT/wl.unique.keys"
fi

# --- codegen + compile the two lookup binaries ---
echo "== codegen + g++ (indexed | lmdb) =="
swipl -q -g main -t halt "$HERE/build.pl" -- "$OUT/proj_indexed" "$IDX" indexed >/dev/null 2>&1
swipl -q -g main -t halt "$HERE/build.pl" -- "$OUT/proj_lmdb"   "$LMDB" lmdb   >/dev/null 2>&1
CXX="${CXX:-g++}"; CXXFLAGS="${CXXFLAGS:--std=c++17 -O2}"
$CXX $CXXFLAGS -I"$OUT/proj_indexed/cpp" -o "$OUT/bench_indexed" "$HERE/bench_main.cpp"
$CXX $CXXFLAGS -I"$OUT/proj_lmdb/cpp"    -o "$OUT/bench_lmdb"    "$HERE/bench_main.cpp" -llmdb

# --- sweep ---
RESULTS="$OUT/results.jsonl"
[ "${APPEND:-0}" = "1" ] || : > "$RESULTS"
export UW_WAM_LMDB_L2_CAP="$L2_CAP"

run_cell() {  # backend binary storepath workload R evict
  local backend="$1" bin="$2" store="$3" wl="$4" R="$5" evict="$6"
  local keys="$OUT/wl.$wl.keys"
  local samples=""
  for i in $(seq 1 "$N_REPEAT"); do
    samples+="$(UW_BENCH_EVICT="$evict" "$bin" "$backend" "$store" "$keys" "$R")"$'\n'
  done
  # Aggregate: min/max/median wall over the N samples; deterministic stats from
  # the first sample (they are identical across repeats and warm/cold).
  local agg
  agg="$(printf '%s' "$samples" | node -e '
let raw="";process.stdin.on("data",d=>raw+=d).on("end",()=>{
  const rows=raw.split("\n").filter(x=>x.trim()).map(JSON.parse);
  const walls=rows.map(r=>r.wall_ms).sort((a,b)=>a-b);
  const o=rows[0]; o.workload=process.argv[1];
  o.min_wall_ms=walls[0]; o.max_wall_ms=walls[walls.length-1];
  o.median_wall_ms=walls[Math.floor(walls.length/2)]; o.n_repeat=walls.length;
  delete o.wall_ms;
  console.log(JSON.stringify(o));
});' "$wl")"
  echo "$agg" >> "$RESULTS"
  printf '%s' "$agg" | node -e 'let r="";process.stdin.on("data",d=>r+=d).on("end",()=>{const o=JSON.parse(r);
console.log(`  ${o.workload.padEnd(8)} ${o.kind.padEnd(8)} R=${String(o.R).padEnd(3)} evict=${o.evict}  reads=${String(o.fact_io_reads).padStart(9)} bytes=${String(o.fact_io_bytes).padStart(10)} l1=${String(o.l1_hits).padStart(7)} l2=${String(o.l2_hits).padStart(7)} miss=${String(o.cache_misses).padStart(7)}  min_wall=${o.min_wall_ms.toFixed(1)}ms spread=[${o.min_wall_ms.toFixed(1)},${o.max_wall_ms.toFixed(1)}]`);});'
}

for wl in $WORKLOADS; do
  for R in $R_LIST; do
    for ev in 0 1; do
      run_cell indexed "$OUT/bench_indexed" "$IDX"  "$wl" "$R" "$ev"
      run_cell lmdb    "$OUT/bench_lmdb"    "$LMDB" "$wl" "$R" "$ev"
    done
  done
done

echo "== wrote $RESULTS =="
echo "store sizes:"
du -h "$IDX.data" "$IDX.idx" "$LMDB/data.mdb" 2>/dev/null | sed 's/^/  /'
