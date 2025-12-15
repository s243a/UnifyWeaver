# feat: Add LLVM Phase 3 Go/Rust integration tests

## Summary

This PR completes the LLVM multi-target integration by providing working examples that compile and run LLVM-compiled Prolog predicates from Go and Rust.

## Integration Test Results

| Language | Tests | Results |
|----------|-------|---------|
| **Go (cgo)** | 4/4 ✅ | Sum(10)=55, Sum(100)=5050, Factorial(5)=120, Factorial(10)=3628800 |
| **Rust (FFI)** | 4/4 ✅ | sum(10)=55, sum(100)=5050, factorial(5)=120, factorial(10)=3628800 |

## New Files

```
examples/llvm-ffi/
├── README.md              # Usage instructions
├── build.sh               # Complete build script
├── prolog_math.h          # C header for FFI
├── libprolog_math.so      # Compiled shared library
├── prolog_math.ll         # Generated LLVM IR
├── go/
│   ├── main.go            # Go cgo integration test
│   ├── go.mod
│   └── llvm-ffi-test      # Compiled Go binary
└── rust/
    ├── src/main.rs        # Rust FFI integration test
    ├── Cargo.toml
    ├── build.rs           # Link configuration
    └── target/release/llvm-ffi-test
```

## Usage

```bash
cd examples/llvm-ffi
./build.sh  # Generates lib, builds Go/Rust, runs tests
```

Or manually:

```bash
# Go
cd go && CGO_ENABLED=1 go build
LD_LIBRARY_PATH=.. ./llvm-ffi-test

# Rust
cd rust && cargo build --release
LD_LIBRARY_PATH=.. ./target/release/llvm-ffi-test
```

## Architecture

```
┌─────────────────────────────────────────┐
│        UnifyWeaver (Prolog)             │
│  compile_shared_library_llvm/3          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
           prolog_math.ll (LLVM IR)
                  │
                  ▼
           libprolog_math.so (C ABI)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   go/main.go          rust/main.rs
   (cgo FFI)            (Rust FFI)
   Sum(10)=55          sum(10)=55
```

## LLVM Target Complete Summary

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Core LLVM IR generation | ✅ PR #344 |
| 2 | C ABI shared library | ✅ PR #345 |
| 3 | Go/Rust integration | ✅ This PR |

## Dependencies

- LLVM (`llc`, `clang`)
- Go 1.21+
- Rust (cargo)

## Related

- PR #344: LLVM Phase 1 (core target)
- PR #345: LLVM Phase 2 (C ABI)
