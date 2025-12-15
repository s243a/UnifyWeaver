# LLVM FFI Integration Examples

Demonstrates calling LLVM-compiled Prolog predicates from Go and Rust.

## Prerequisites

- LLVM toolchain (`llc`, `clang`)
- Go 1.21+
- Rust (cargo)

## Building the Shared Library

First, generate and build the shared library:

```bash
# From the examples/llvm-ffi directory
./build.sh
```

Or manually:

```bash
# Generate LLVM IR (from project root)
swipl -g "use_module('src/unifyweaver/targets/llvm_target'), \
  compile_shared_library_llvm([func(sum, 2, tail_recursion), func(factorial, 1, factorial)], \
  [library_name(prolog_math)], Code), write(Code)" -t halt > prolog_math.ll

# Fix escaping, compile to shared library
sed -i 's/%%/%/g' prolog_math.ll
llc -filetype=obj -relocation-model=pic prolog_math.ll -o prolog_math.o
clang -shared prolog_math.o -o libprolog_math.so
```

## Go Test

```bash
cd go
CGO_ENABLED=1 go build
LD_LIBRARY_PATH=.. ./llvm-ffi-test
```

Expected output:
```
[PASS] Sum(10) = 55
[PASS] Sum(100) = 5050
[PASS] Factorial(5) = 120
[PASS] Factorial(10) = 3628800
INTEGRATION TEST PASSED
```

## Rust Test

```bash
cd rust
cargo build
LD_LIBRARY_PATH=.. cargo run
```

Expected output:
```
[PASS] sum(10) = 55
[PASS] sum(100) = 5050
[PASS] factorial(5) = 120
[PASS] factorial(10) = 3628800
INTEGRATION TEST PASSED
```

## Files

```
examples/llvm-ffi/
├── prolog_math.h        # C header for FFI
├── libprolog_math.so    # Compiled shared library
├── build.sh             # Build script
├── go/
│   ├── main.go          # Go cgo integration test
│   └── go.mod
└── rust/
    ├── src/main.rs      # Rust FFI integration test
    ├── Cargo.toml
    └── build.rs
```
