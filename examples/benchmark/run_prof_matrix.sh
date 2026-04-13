#!/usr/bin/env bash
set -euo pipefail

# run_prof_matrix.sh — Build and profile all 4 Haskell WAM configurations
#
# Usage:
#   ./run_prof_matrix.sh <facts.pl> [output-base-dir]
#
# Steps:
#   1. Generate all 4 configs via gen_prof_matrix.pl
#   2. Build each with cabal (profiling enabled)
#   3. Run each with +RTS -p to produce .prof files
#   4. Print summary of top cost centres per config

FACTS="${1:?Usage: $0 <facts.pl> [output-base-dir]}"
BASE="${2:-/tmp/wam-prof-matrix}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=("A-pure-interp" "B-interp-ffi" "C-lowered-only" "D-lowered-ffi")

echo "=== Step 1: Generate all configurations ==="
swipl -q -s "$SCRIPT_DIR/gen_prof_matrix.pl" -- "$FACTS" "$BASE"

echo ""
echo "=== Step 2: Build all configurations ==="
for cfg in "${CONFIGS[@]}"; do
    dir="$BASE/$cfg"
    echo "--- Building $cfg ---"
    (cd "$dir" && cabal build 2>&1 | tail -3)
done

echo ""
echo "=== Step 3: Profile all configurations ==="
for cfg in "${CONFIGS[@]}"; do
    dir="$BASE/$cfg"
    # Find the built executable
    exe=$(cd "$dir" && cabal list-bin "wam-prof-$cfg" 2>/dev/null || true)
    if [ -z "$exe" ]; then
        echo "WARN: Could not find executable for $cfg, skipping"
        continue
    fi
    echo "--- Profiling $cfg ---"
    (cd "$dir" && "$exe" "$FACTS" +RTS -p -RTS 2>&1 | grep -E 'query_ms|total_ms|seeds')
    echo ""
done

echo ""
echo "=== Step 4: Profile summaries ==="
for cfg in "${CONFIGS[@]}"; do
    dir="$BASE/$cfg"
    prof_file=$(ls "$dir"/*.prof 2>/dev/null | head -1 || true)
    if [ -z "$prof_file" ]; then
        echo "--- $cfg: no .prof file found ---"
        continue
    fi
    echo "--- $cfg (top 15 cost centres) ---"
    head -30 "$prof_file"
    echo ""
done

echo "Profile files are at:"
for cfg in "${CONFIGS[@]}"; do
    ls "$BASE/$cfg"/*.prof 2>/dev/null || echo "  $cfg: none"
done
