#!/bin/bash
# Build LLVM FFI integration examples
set -e

cd "$(dirname "$0")"

echo "=== Building LLVM-compiled Prolog shared library ==="

# Generate LLVM IR from Prolog
echo "Generating LLVM IR..."
pushd ../.. > /dev/null
swipl -g "use_module('src/unifyweaver/targets/llvm_target'), \
  compile_shared_library_llvm([func(sum, 2, tail_recursion), func(factorial, 1, factorial)], \
  [library_name(prolog_math)], Code), write(Code)" -t halt > examples/llvm-ffi/prolog_math.ll
popd > /dev/null

# Fix Prolog escaping
echo "Fixing escaping..."
sed -i 's/%%/%/g' prolog_math.ll

# Compile to shared library
echo "Compiling to shared library..."
llc -filetype=obj -relocation-model=pic prolog_math.ll -o prolog_math.o
clang -shared prolog_math.o -o libprolog_math.so

# Verify exports
echo "Verifying exports:"
nm -D libprolog_math.so | grep " T "

echo ""
echo "=== Building Go test ==="
cd go
CGO_ENABLED=1 go build -o llvm-ffi-test
cd ..

echo ""
echo "=== Building Rust test ==="
cd rust
cargo build --release 2>&1 || echo "Rust build may need LD_LIBRARY_PATH set"
cd ..

echo ""
echo "=== Running Go test ==="
cd go
LD_LIBRARY_PATH=.. ./llvm-ffi-test
cd ..

echo ""
echo "=== Running Rust test ==="
cd rust
LD_LIBRARY_PATH=.. cargo run --release 2>&1 || LD_LIBRARY_PATH=.. ./target/release/llvm-ffi-test
cd ..

echo ""
echo "=== All integration tests complete ==="
