#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# cost_model.sh -- measure the primitives for the backend-selection cost model
# and print the K(rows_per_key) crossover table. NO memory-cap needed: the
# disk-bound seek cost is measured per-page with posix_fadvise(DONTNEED); the
# aggregate disk-bound regime is then MODELED (extrapolated), not stress-tested.
# See RESULTS.md "Backend-selection cost model" for the write-up + honesty notes.
#
# Prereqs (built by bench_crossover.sh): $OUT/{bench_indexed,bench_lmdb},
# $OUT/idx/symprov.{data,idx}, $OUT/lmdb_v1/symprov. Run bench_crossover.sh first.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="$ROOT/examples/pkg_resolver/abi/.out/bench"
export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"
export NODE_PATH="${NODE_PATH:-/tmp/uw-lmdb-pkg-v1/node_modules}" UW_LMDB_DATA_V1=1
export UW_WAM_FACT_L2_CAP="${UW_WAM_FACT_L2_CAP:-65536}"

gcc -O2 -o "$OUT/cost_model_probe" "$HERE/cost_model_probe.c"

echo "== t_seek / t_mem (single 4KB page, cold via fadvise vs warm) =="
PROBE="$("$OUT/cost_model_probe" "$OUT/idx/symprov.data" 3000)"; echo "$PROBE"
TSEEK=$(echo "$PROBE" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s).t_seek_cold_ns.median))')
TMEM=$(echo "$PROBE" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s).t_mem_warm_ns.median))')

echo "== scatter: indexed distinct .data pages per key (= cold reads/miss) =="
echo "-- ABI symprov store (~1 row/key):"
node "$HERE/cost_model.mjs" scatter "$OUT/idx/symprov" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const o=JSON.parse(s);console.log(`   rows/key=${o.rows_per_key} indexed_cold_reads/miss=${o.indexed_cold_reads_per_miss} lmdb~1`)})'
mkdir -p "$OUT/mr/idx"
for M in 2 4 8 16; do
  node "$HERE/cost_model.mjs" synth "$OUT/mr/mr$M.p2.jsonl" 20000 "$M" interleaved 2>/dev/null
  node "$ROOT/scripts/js_wam/uw_fact_index.js" build "$OUT/mr/mr$M.p2.jsonl" "$OUT/mr/idx/mr$M" >/dev/null 2>&1
  node "$HERE/cost_model.mjs" scatter "$OUT/mr/idx/mr$M" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const o=JSON.parse(s);console.log(`-- interleaved M=${o.rows_per_key}: indexed_cold_reads/miss=${o.indexed_cold_reads_per_miss} vs lmdb~1`)})'
done
echo "-- grouped (key-sorted .data) M=8: indexed clusters too:"
node "$HERE/cost_model.mjs" synth "$OUT/mr/mr8g.p2.jsonl" 20000 8 grouped 2>/dev/null
node "$ROOT/scripts/js_wam/uw_fact_index.js" build "$OUT/mr/mr8g.p2.jsonl" "$OUT/mr/idx/mr8g" >/dev/null 2>&1
node "$HERE/cost_model.mjs" scatter "$OUT/mr/idx/mr8g" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const o=JSON.parse(s);console.log(`   grouped M=${o.rows_per_key}: indexed_cold_reads/miss=${o.indexed_cold_reads_per_miss} (clustered ~1)`)})'

echo "== t_hit (pure cache-hit) & t_miss_resident (warm zero-reuse), per backend =="
head -100 "$OUT/wl.unique.keys" | awk '{for(i=0;i<500;i++)print}' > "$OUT/wl.hot.keys"
hit_of() { UW_BENCH_EVICT=0 "$1" "$2" "$3" "$OUT/wl.hot.keys" 1 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const o=JSON.parse(s);console.log((o.wall_ms*1e6/o.lookups).toFixed(3))})'; }
miss_of() { UW_BENCH_EVICT=0 "$1" "$2" "$3" "$OUT/wl.unique.keys" 1 | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const o=JSON.parse(s);console.log((o.wall_ms*1e6/o.lookups).toFixed(3))})'; }
echo "   (single-shot; ns per lookup)"
echo "   indexed t_hit=$(hit_of "$OUT/bench_indexed" indexed "$OUT/idx/symprov")ns  t_miss_resident=$(miss_of "$OUT/bench_indexed" indexed "$OUT/idx/symprov")ns"
echo "   lmdb    t_hit=$(hit_of "$OUT/bench_lmdb" lmdb "$OUT/lmdb_v1/symprov")ns  t_miss_resident=$(miss_of "$OUT/bench_lmdb" lmdb "$OUT/lmdb_v1/symprov")ns"

echo "== K(rows_per_key): store/RAM ratio where lmdb reaches a speedup (hit_rate=0.9) =="
node "$HERE/cost_model.mjs" K "{\"t_seek_ns\":$TSEEK,\"t_mem_ns\":$TMEM,\"t_hit_ns\":220,\"hit_rate\":0.9,\"rows_per_key_list\":[1,2,4,8,16],\"targets\":[1.2,1.5,2.0]}" \
  | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const o=JSON.parse(s);console.log(`   t_seek=${(o.primitives.t_seek_ns/1000).toFixed(1)}us t_mem=${(o.primitives.t_mem_ns/1000).toFixed(3)}us ratio=${(o.primitives.t_seek_ns/o.primitives.t_mem_ns).toFixed(0)}x`);for(const r of o.table)console.log(`   rows/key=${String(r.rows_per_key).padStart(2)}: speedup(store>>RAM)=${r.ratio_at_r_inf}x  K@1.2x=${r.K_for_speedup["x1.2"]} K@1.5x=${r.K_for_speedup["x1.5"]} K@2x=${r.K_for_speedup["x2"]}`)})'
echo "(K = store/RAM ratio; null = that speedup never reached. K~=1 means the benefit appears as soon as store exceeds RAM; magnitude = rows_per_key.)"
