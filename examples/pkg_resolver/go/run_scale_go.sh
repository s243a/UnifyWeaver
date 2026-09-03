#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# run_scale_go.sh -- B3: load + resolve_layered on the 5k catalog
# (store/gen_scale_catalog.mjs, seed 0xc0ffee01). Catalogs/requests
# enter as JSON; the compiled WAM does the resolve. SWI reference
# numbers come from examples/pkg_resolver/run_scale_demo.sh.
#
#   bash examples/pkg_resolver/go/run_scale_go.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCALE="$HERE/.scale_out"
mkdir -p "$SCALE"
cd "$ROOT"

if [[ ! -x "$HERE/uwresolve" ]]; then
  bash "$HERE/build.sh"
fi

if [[ ! -f "$SCALE/rich.jsonl" ]]; then
  node "$HERE/../store/gen_scale_catalog.mjs" "$SCALE"
fi

"$HERE/uwresolve" --scale-probe "$SCALE"
