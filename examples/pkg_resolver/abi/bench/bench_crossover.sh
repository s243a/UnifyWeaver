#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# bench_crossover.sh -- reproducible entry point for the lmdb-vs-indexed store
# backend "fair fight" on the UnifyWeaver ABI store, at TWO scales:
#   symbol scale  -- the 256k-row ABI symprov/2 store (idx ~42MB, lmdb ~128MB)
#   package scale -- the ~5k-key gen_scale_catalog pkg/2 store (idx ~320KB)
#
# It drives the C++ WAM SeekFactSource read path DIRECTLY:
#   indexed  = on-disk UWFI/UWIX; the OPTIMIZED path slurps the whole .idx into
#              RAM once and binary-searches it in memory (+1 .data read/record).
#   lmdb     = the C++ lazy reader with L1 direct-mapped + L2 FIFO row caches.
# NOT the JS/wamjs lmdb backend. See RESULTS.md for the finding.
#
# Binaries (all use the real SeekFactSource; store path via argv, so one binary
# per gate serves any store/scale):
#   bench_indexed      optimized indexed (current runtime template)
#   bench_lmdb         lmdb (unchanged path)
#   bench_indexed_old  BUILD_OLD=1 only: pre-optimization indexed, built from the
#                      committed (HEAD) runtime template via a save/restore swap,
#                      for the OLD-vs-OPTIMIZED comparison in RESULTS.md.
#
# Memory pressure: no hard cap is available unprivileged on this WSL2 host
# (systemd-run --user: no bus; system scope: interactive auth; cgroup v2: not
# delegated; no root). The cold variant evicts the store per-run via
# posix_fadvise(DONTNEED) (UW_BENCH_EVICT=1); stores << RAM so cold ~= warm.
#
# Env knobs: N_REPEAT (3), R_LIST ("1 5 10"), WORKLOADS ("skewed uniform unique"),
#   L2_CAP (65536), WL_N/WL_HOT/WL_MISS/WL_SEED, SCALES ("symbol package"),
#   BUILD_OLD (0).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="$ROOT/examples/pkg_resolver/abi/.out/bench"      # symbol scale store + binaries
SCALE_DIR="$ROOT/examples/pkg_resolver/store/.out/scale"  # package scale store
PB="$OUT/pkg"                                          # package workloads + lmdb
TPL="$ROOT/templates/targets/cpp_wam/runtime.h.mustache"

export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"
N_REPEAT="${N_REPEAT:-3}"; R_LIST="${R_LIST:-1 5 10}"
WORKLOADS="${WORKLOADS:-skewed uniform unique}"
L2_CAP="${L2_CAP:-65536}"; SCALES="${SCALES:-symbol package}"
BUILD_OLD="${BUILD_OLD:-0}"
export UW_WAM_LMDB_L2_CAP="$L2_CAP"
cd "$ROOT"

mkdir -p "$OUT/idx" "$OUT/sym" "$PB"

# ---------- symbol-scale store ----------
if [ ! -f "$OUT/symprov.p2.jsonl" ]; then
  echo "== ingest ABI symbols -> pack (#-join) =="
  node "$ROOT/examples/pkg_resolver/abi/ingest_symbols.mjs" symbols-dir /var/lib/dpkg/info --out "$OUT/sym"
  node -e '
const fs=require("fs");const rl=require("readline").createInterface({input:fs.createReadStream(process.argv[1])});
const out=fs.createWriteStream(process.argv[2]);
rl.on("line",l=>{const t=l.trim();if(!t)return;const a=JSON.parse(t);const v=Array.isArray(a[1])?a[1].join("#"):a[1];out.write(JSON.stringify([a[0],v])+"\n");});
rl.on("close",()=>out.end());' "$OUT/sym/symprov.jsonl" "$OUT/symprov.p2.jsonl"
fi
[ -f "$OUT/idx/symprov.data" ] || node "$ROOT/scripts/js_wam/uw_fact_index.js" build "$OUT/symprov.p2.jsonl" "$OUT/idx/symprov"
if [ ! -f "$OUT/lmdb_v1/symprov/data.mdb" ]; then
  export UW_LMDB_DATA_V1=1
  # shellcheck source=../store/ensure_lmdb.sh
  source "$ROOT/examples/pkg_resolver/store/ensure_lmdb.sh"; uw_require_lmdb
  mkdir -p "$OUT/lmdb_v1"
  node "$ROOT/scripts/js_wam/uw_fact_lmdb.js" build "$OUT/symprov.p2.jsonl" "$OUT/lmdb_v1/symprov"
fi

# ---------- package-scale store ----------
if [ ! -f "$SCALE_DIR/pkg.data" ]; then
  echo "== build 5k package-scale store =="
  mkdir -p "$SCALE_DIR"
  node "$ROOT/examples/pkg_resolver/store/gen_scale_catalog.mjs" "$SCALE_DIR"
  [ -f "$SCALE_DIR/pkg.jsonl" ] || node "$ROOT/examples/pkg_resolver/store/rich_to_p2.mjs" "$SCALE_DIR/rich.jsonl" "$SCALE_DIR"
  bash "$ROOT/examples/pkg_resolver/store/build_stores.sh" "$SCALE_DIR"
fi
if [ ! -f "$PB/lmdb_pkg/data.mdb" ]; then
  export UW_LMDB_DATA_V1=1
  source "$ROOT/examples/pkg_resolver/store/ensure_lmdb.sh"; uw_require_lmdb
  node "$ROOT/scripts/js_wam/uw_fact_lmdb.js" build "$SCALE_DIR/pkg.jsonl" "$PB/lmdb_pkg"
fi

# ---------- workloads ----------
gen_unique() {  # jsonl out
  node -e '
const fs=require("fs");const rl=require("readline").createInterface({input:fs.createReadStream(process.argv[1])});
const seen=new Set();const keys=[];rl.on("line",l=>{const t=l.trim();if(!t)return;try{const k=JSON.parse(t)[0];if(!seen.has(k)){seen.add(k);keys.push(k);}}catch{}});
rl.on("close",()=>{let s=987654321>>>0;const rnd=()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};
for(let i=keys.length-1;i>0;i--){const j=Math.floor(rnd()*(i+1));[keys[i],keys[j]]=[keys[j],keys[i]];}
fs.writeFileSync(process.argv[2],keys.join("\n")+"\n");});' "$1" "$2"
}
if [ ! -f "$OUT/wl.skewed.keys" ]; then
  node "$HERE/gen_workload.mjs" "$OUT/symprov.p2.jsonl" "$OUT/wl" "${WL_N:-50000}" "${WL_HOT:-0.80}" "${WL_MISS:-0.15}" "${WL_SEED:-1234567}" >/dev/null
  gen_unique "$OUT/symprov.p2.jsonl" "$OUT/wl.unique.keys"
fi
if [ ! -f "$PB/wl.skewed.keys" ]; then
  node "$HERE/gen_workload.mjs" "$SCALE_DIR/pkg.jsonl" "$PB/wl" "${WL_N:-50000}" "${WL_HOT:-0.80}" "${WL_MISS:-0.15}" "${WL_SEED:-1234567}" >/dev/null
  gen_unique "$SCALE_DIR/pkg.jsonl" "$PB/wl.unique.keys"
fi

# ---------- binaries ----------
CXX="${CXX:-g++}"; CXXFLAGS="${CXXFLAGS:--std=c++17 -O2}"
echo "== codegen + g++ (optimized indexed | lmdb) =="
swipl -q -g main -t halt "$HERE/build.pl" -- "$OUT/proj_indexed" "$OUT/idx/symprov" indexed >/dev/null 2>&1
swipl -q -g main -t halt "$HERE/build.pl" -- "$OUT/proj_lmdb"   "$OUT/lmdb_v1/symprov" lmdb >/dev/null 2>&1
$CXX $CXXFLAGS -I"$OUT/proj_indexed/cpp" -o "$OUT/bench_indexed" "$HERE/bench_main.cpp"
$CXX $CXXFLAGS -I"$OUT/proj_lmdb/cpp"    -o "$OUT/bench_lmdb"    "$HERE/bench_main.cpp" -llmdb
if [ "$BUILD_OLD" = "1" ]; then
  echo "== codegen + g++ (OLD indexed from HEAD template) =="
  cp "$TPL" "$OUT/.runtime.h.CURRENT"
  restore_tpl() { cp "$OUT/.runtime.h.CURRENT" "$TPL"; }
  trap restore_tpl EXIT
  git -C "$ROOT" show HEAD:templates/targets/cpp_wam/runtime.h.mustache > "$TPL"
  swipl -q -g main -t halt "$HERE/build.pl" -- "$OUT/proj_indexed_old" "$OUT/idx/symprov" indexed >/dev/null 2>&1
  $CXX $CXXFLAGS -I"$OUT/proj_indexed_old/cpp" -o "$OUT/bench_indexed_old" "$HERE/bench_main.cpp"
  restore_tpl; trap - EXIT
fi

# ---------- sweep ----------
run_cell() {  # results label backend bin store wldir wl R evict
  local results="$1" label="$2" backend="$3" bin="$4" store="$5" wldir="$6" wl="$7" R="$8" ev="$9"
  local keys="$wldir/wl.$wl.keys" samples=""
  for _ in $(seq 1 "$N_REPEAT"); do
    samples+="$(UW_BENCH_EVICT="$ev" "$bin" "$backend" "$store" "$keys" "$R")"$'\n'
  done
  printf '%s' "$samples" | node -e '
let raw="";process.stdin.on("data",d=>raw+=d).on("end",()=>{
  const rows=raw.split("\n").filter(x=>x.trim()).map(JSON.parse);
  const w=rows.map(r=>r.wall_ms).sort((a,b)=>a-b); const o=rows[0];
  o.scale=process.argv[1]; o.label=process.argv[2]; o.workload=process.argv[3];
  o.min_wall_ms=w[0]; o.max_wall_ms=w[w.length-1]; o.n_repeat=w.length; delete o.wall_ms;
  console.log(JSON.stringify(o));});' "$SCALE" "$label" "$wl" >> "$results"
  tail -1 "$results" | node -e 'let r="";process.stdin.on("data",d=>r+=d).on("end",()=>{const o=JSON.parse(r);
console.log(`  ${o.scale.padEnd(7)} ${o.workload.padEnd(8)} ${o.label.padEnd(13)} R=${String(o.R).padEnd(3)} ev=${o.evict}  reads=${String(o.fact_io_reads).padStart(9)} l1=${String(o.l1_hits).padStart(7)} l2=${String(o.l2_hits).padStart(7)} miss=${String(o.cache_misses).padStart(7)} rows=${String(o.rows_found).padStart(7)}  min_wall=${o.min_wall_ms.toFixed(1)}ms [${o.min_wall_ms.toFixed(0)}-${o.max_wall_ms.toFixed(0)}]`);});'
}

for SCALE in $SCALES; do
  if [ "$SCALE" = "symbol" ]; then IDX="$OUT/idx/symprov"; LMDB="$OUT/lmdb_v1/symprov"; WLDIR="$OUT";
  else IDX="$SCALE_DIR/pkg"; LMDB="$PB/lmdb_pkg"; WLDIR="$PB"; fi
  RES="$OUT/results.$SCALE.jsonl"; : > "$RES"
  echo "== sweep: $SCALE scale =="
  for wl in $WORKLOADS; do for R in $R_LIST; do for ev in 0 1; do
    [ "$BUILD_OLD" = "1" ] && run_cell "$RES" indexed-old indexed "$OUT/bench_indexed_old" "$IDX" "$WLDIR" "$wl" "$R" "$ev"
    run_cell "$RES" indexed-opt indexed "$OUT/bench_indexed"     "$IDX"  "$WLDIR" "$wl" "$R" "$ev"
    run_cell "$RES" lmdb        lmdb    "$OUT/bench_lmdb"        "$LMDB" "$WLDIR" "$wl" "$R" "$ev"
  done; done; done
  echo "== wrote $RES =="
done
echo "store sizes:"; du -h "$OUT/idx/symprov.data" "$OUT/idx/symprov.idx" "$OUT/lmdb_v1/symprov/data.mdb" \
  "$SCALE_DIR/pkg.data" "$SCALE_DIR/pkg.idx" "$PB/lmdb_pkg/data.mdb" 2>/dev/null | sed 's/^/  /'
