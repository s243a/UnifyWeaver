#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# Index P/2 JSONL dumps with D43 uw_fact_index.js.
#
#   bash examples/pkg_resolver/store/build_stores.sh DIR
#
# Expects DIR/{pkg,dep,conflict,revdep}.jsonl
# Writes DIR/{pkg,dep,conflict,revdep}.data + .idx

set -euo pipefail

DIR="${1:?usage: build_stores.sh DIR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INDEX="$ROOT/scripts/js_wam/uw_fact_index.js"

for name in pkg dep conflict revdep; do
  src="$DIR/${name}.jsonl"
  prefix="$DIR/${name}"
  if [[ ! -f "$src" ]]; then
    : > "$src"
  fi
  node "$INDEX" build "$src" "$prefix"
done
