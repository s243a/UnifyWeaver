#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_scale_demo.sh -- 5k catalog: bytes-read proof + term-vs-store timings.
#
#   bash examples/pkg_resolver/run_scale_demo.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SCALE="$HERE/store/.out/scale"
WAM="$SCALE/wamjs"
mkdir -p "$SCALE" "$WAM"
cd "$ROOT"

if [[ ! -f "$SCALE/pkg.data" ]]; then
  node "$HERE/store/gen_scale_catalog.mjs" "$SCALE"
  node "$HERE/store/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
  bash "$HERE/store/build_stores.sh" "$SCALE"
fi
if [[ ! -f "$WAM/js/generated_program.js" ]] ||
   ! grep -q 'configure_lmdb_materialisation' "$WAM/js/wam_runtime.js" 2>/dev/null; then
  swipl -q -g main -t halt "$HERE/wamjs_store/build.pl" -- \
    "$HERE/resolver_store.pl" "$WAM" "$SCALE"
  cp "$HERE/wamjs_store/resolver_store.mjs" "$WAM/resolver_store.mjs"
fi

echo "== store sizes =="
python3 - <<PY
import os
d = "$SCALE"
total = 0
for name in ("pkg", "dep", "conflict", "revdep"):
    for suf in (".data", ".idx"):
        p = os.path.join(d, name + suf)
        if os.path.isfile(p):
            n = os.path.getsize(p)
            total += n
            print(f"  {name}{suf}: {n} bytes")
print(f"  total_store_bytes: {total}")
open(os.path.join(d, "store_size.txt"), "w").write(str(total) + "\n")
PY

echo "== WAM bound resolve_layered (UW_FACT_IO_STATS, lazy) =="
# D48 bytes-read proof: bytes ∝ query. Force the fleet `lazy` tier
# (UW_FACT_CACHE=0 alias). Default is `cached`; eager would scan the store.
UW_LMDB_MATERIALISATION=lazy UW_FACT_CACHE=0 UW_FACT_IO_STATS=1 \
  node "$HERE/store/run_probe.mjs" "$WAM" "$SCALE/probe.json" \
  2> "$SCALE/probe_wamjs.err" | tee "$SCALE/probe_wamjs.log"
echo "--- fact_io stderr ---"
tee -a "$SCALE/probe_wamjs.log" < "$SCALE/probe_wamjs.err"

echo "== SWI term vs store timings (load included) =="
STORE_DIR="$SCALE" swipl -q -g scale_demo -t halt "$HERE/store/scale_demo.pl" -- "$SCALE" \
  | tee "$SCALE/timing_swi.log"
