#!/bin/bash
# Build WASM module from LLVM IR
# Usage: ./build.sh [input.ll] [output_name]

set -e

INPUT="${1:-family_graph.ll}"
OUTPUT="${2:-family_graph}"

echo "=== Building WASM module ==="
echo "Input: $INPUT"
echo "Output: ${OUTPUT}.wasm"

# Check dependencies
command -v llc >/dev/null 2>&1 || { echo "Error: llc not found (install llvm)"; exit 1; }
command -v wasm-ld >/dev/null 2>&1 || { echo "Error: wasm-ld not found (install lld)"; exit 1; }

# Step 1: Fix Prolog escaping (double % -> single %)
echo "[1/3] Fixing escapes..."
sed -i 's/%%/%/g' "$INPUT"

# Step 2: Compile to WASM object
echo "[2/3] Compiling to WASM object..."
llc -march=wasm32 -filetype=obj "$INPUT" -o "${OUTPUT}.o"

# Step 3: Link to WASM module
echo "[3/3] Linking WASM module..."
wasm-ld --no-entry --export-all -o "${OUTPUT}.wasm" "${OUTPUT}.o"

# Verify
echo ""
echo "=== Build complete ==="
ls -la "${OUTPUT}.wasm"
echo ""
wasm-objdump -x "${OUTPUT}.wasm" 2>/dev/null | head -30 || true

echo ""
echo "Use with: GraphWasm.load('${OUTPUT}.wasm')"
