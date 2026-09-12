#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Focused regressions for the two WAM-C gaps that produced the 106 false
# failures: compare/3 (indexed catalogs) and predsort/3 (Debian versions).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/diff/.reg_out"
DIFF_BIN="$HERE/diff/diff_uwresolve"
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -x "$DIFF_BIN" ]]; then
  bash "$HERE/build_diff.sh"
fi

python3 - "$OUT/cases.jsonl" <<'PY'
import json, sys
path = sys.argv[1]
pkgs = [["p%d" % i, [0, 1 if i == 0 else 0, 0]] for i in range(64)]
index_case = {
    "id": "reg_index_resolve",
    "catalog": {
        "packages": pkgs,
        "depends": [],
        "conflicts": [],
        "base": [],
        "installed": [],
        "requested": ["p0"],
    },
    "query": "resolve",
    "args": ["p0"],
}
layer_case = dict(index_case)
layer_case["id"] = "reg_index_layer_closure"
layer_case["query"] = "layer_closure"
layer_case["args"] = "p0"
deb_lo = {"deb": [0, [["", 1]], [["", 1]]]}
deb_hi = {"deb": [0, [["", 2]], []]}
deb_case = {
    "id": "reg_deb_predsort",
    "catalog": {
        "packages": [["d0", deb_lo], ["d0", deb_hi]],
        "depends": [],
        "conflicts": [],
        "base": [],
        "installed": [],
        "requested": ["d0"],
    },
    "query": "resolve",
    "args": ["d0"],
}
with open(path, "w") as f:
    for row in (index_case, layer_case, deb_case):
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
PY

echo "== focused regressions: $(wc -l < "$OUT/cases.jsonl") cases =="
swipl -q -g main -t halt "$HERE/../diff_runner.pl" < "$OUT/cases.jsonl" > "$OUT/swi.jsonl"
"$DIFF_BIN" < "$OUT/cases.jsonl" > "$OUT/c.jsonl"
node "$HERE/../compare_jsonl.mjs" "$OUT/cases.jsonl" "$OUT/swi.jsonl" "$OUT/c.jsonl"

python3 "$HERE/driver_selftest.py"
