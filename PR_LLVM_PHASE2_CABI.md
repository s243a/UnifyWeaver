# feat: Add LLVM Phase 2 C ABI integration for Go/Rust FFI

## Summary

This PR extends the LLVM target with C ABI integration, enabling LLVM-compiled Prolog predicates to be called from Go (via cgo) and Rust (via FFI).

## New Features

### Shared Library Generation

```prolog
compile_shared_library_llvm(
    [func(sum, 2, tail_recursion),
     func(factorial, 1, factorial)],
    [library_name(prolog_math)],
    Code).
```

### FFI Generators

| Predicate | Output |
|-----------|--------|
| `generate_c_header/2` | `prolog_math.h` with `int64_t` declarations |
| `generate_cgo_bindings/2` | Go package with `C.function()` calls |
| `generate_rust_ffi/2` | Rust `extern "C"` module |
| `build_shared_library/3` | Shell build commands |

### Function Types Supported

- `tail_recursion` - musttail optimized
- `linear_recursion` - memoized
- `factorial` - simple recursion

## Integration Test

```bash
# Build shared library
llc -filetype=obj -relocation-model=pic prolog_math.ll -o prolog_math.o
clang -shared prolog_math.o -o libprolog_math.so

# Verify exports
nm -D libprolog_math.so
#   T factorial
#   T sum
```

## Testing

- **10/10 tests pass** (6 Phase 1 + 4 Phase 2)

## Documentation

### Main Docs
- `docs/LLVM_TARGET.md` - Added Phase 2 API section

### Education Book
- `book-07-cross-target-glue/17_llvm_ffi.md` - New chapter (219 lines)
  - Go cgo integration
  - Rust FFI integration
  - Build commands

## Files Changed

```
src/unifyweaver/targets/llvm_target.pl  [MODIFIED] +220 lines
tests/test_llvm_target.pl                [MODIFIED] +50 lines
docs/LLVM_TARGET.md                      [MODIFIED] +46 lines
build/libprolog_math.so                  [NEW] integration test artifact
build/prolog_math.ll                     [NEW] generated LLVM IR
```

## Usage Example

### Go

```go
// #cgo LDFLAGS: -L. -lprolog_math
// #include "prolog_math.h"
import "C"

func Sum(n int64) int64 {
    return int64(C.sum(C.int64_t(n)))
}
```

### Rust

```rust
extern "C" {
    fn sum(n: i64) -> i64;
}

pub fn sum(n: i64) -> i64 {
    unsafe { ffi::sum(n) }
}
```

## Related

- Extends Phase 1 LLVM target (PR #344)
- Complements existing cross-target glue (subprocess, HTTP)
