#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_scale_go_store.sh -- B3 payoff: one bound resolve_layered on the 5k
# catalog (store/gen_scale_catalog.mjs, seed 0xc0ffee01) through the Go store
# path. Reports the D43 bytes-read counter vs total store size AND wall time,
# to compare against Go's term-catalog B3 (examples/pkg_resolver/go).
#
#   bash examples/pkg_resolver/go_store/run_scale_go_store.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/../store/.out/scale"
mkdir -p "$SCALE"
cd "$ROOT"

if [[ ! -f "$SCALE/rich.jsonl" ]]; then
  node "$HERE/../store/gen_scale_catalog.mjs" "$SCALE"
fi
if [[ ! -f "$SCALE/pkg.jsonl" ]]; then
  node "$HERE/../store/rich_to_p2.mjs" "$SCALE/rich.jsonl" "$SCALE"
fi
if [[ ! -f "$SCALE/pkg.data" ]]; then
  bash "$HERE/../store/build_stores.sh" "$SCALE"
fi

STORE_DIR="$SCALE" bash "$HERE/build.sh"

"$HERE/uwresolvestore" --scale-probe "$SCALE"
