#!/bin/bash
# Build WASM module from Prolog
set -e

cd "$(dirname "$0")"

echo "=== Building WASM module from Prolog ==="

# Generate WASM-targeted LLVM IR
echo "Generating WASM LLVM IR..."
pushd ../.. > /dev/null
swipl -g "use_module('src/unifyweaver/targets/llvm_target'), \
  compile_wasm_module([func(sum, 2, tail_recursion), func(factorial, 1, factorial)], \
  [module_name(prolog_wasm)], Code), write(Code)" -t halt > examples/wasm/prolog_wasm.ll
popd > /dev/null

# Fix Prolog escaping
echo "Fixing escaping..."
sed -i 's/%%/%/g' prolog_wasm.ll

# Compile to WASM object
echo "Compiling to WASM..."
llc -march=wasm32 -filetype=obj prolog_wasm.ll -o prolog_wasm.o

# Link to WASM module
echo "Linking WASM module..."
# Try wasm-ld, fall back to wasm-ld-10
if command -v wasm-ld &> /dev/null; then
    wasm-ld --no-entry --export-all prolog_wasm.o -o prolog_wasm.wasm
else
    wasm-ld-10 --no-entry --export-all prolog_wasm.o -o prolog_wasm.wasm
fi

echo ""
echo "=== WASM module created ==="
file prolog_wasm.wasm
ls -la prolog_wasm.wasm

echo ""
echo "=== Running Node.js test ==="
node test.js
