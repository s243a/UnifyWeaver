# feat: Add WebAssembly target for browser and Node.js

## Summary

This PR adds WebAssembly compilation support to UnifyWeaver's LLVM target, enabling Prolog predicates to run in browsers, Node.js, and edge computing platforms.

## Integration Test Results ✅

```
[PASS] sum(10) = 55
[PASS] sum(100) = 5050
[PASS] factorial(5) = 120
[PASS] factorial(10) = 3628800
```

## New Features

### WASM Compilation

```prolog
compile_wasm_module(
    [func(sum, 2, tail_recursion), func(factorial, 1, factorial)],
    [module_name(prolog_wasm)],
    Code).
```

### JavaScript & TypeScript Bindings

| Predicate | Output |
|-----------|--------|
| `generate_js_bindings/2` | Browser + Node.js loaders |
| `generate_ts_bindings/2` | Typed interfaces |
| `build_wasm_module/3` | Shell build commands |

## Unit Tests: 13/13 Pass

- Phase 1: 6 (core LLVM)
- Phase 2: 4 (C ABI)
- **Phase 4: 3 (WASM)** ✨

## Files

### Implementation
```
src/unifyweaver/targets/llvm_target.pl  [MODIFIED] +200 lines
tests/test_llvm_target.pl               [MODIFIED] +40 lines
docs/proposals/wasm_target_design.md    [NEW] design doc
```

### Examples
```
examples/wasm/
├── prolog_wasm.wasm  # Compiled module (376 bytes!)
├── test.js           # Node.js integration test
├── build.sh          # Build script
└── README.md         # Usage guide
```

### Education Book
```
education/other-books/book-wasm-target/
├── README.md
├── 01_introduction.md
├── 02_compilation.md
└── 03_javascript.md
```

## Usage

```bash
# Build WASM module
cd examples/wasm && ./build.sh

# Run test
node test.js
```

## Browser Usage

```javascript
const { instance } = await WebAssembly.instantiate(bytes);
console.log(instance.exports.sum(10)); // 55
```

## Dependencies

```bash
sudo apt install lld  # For wasm-ld
```

## Architecture

```
Prolog → LLVM IR (wasm32) → .wasm (376 bytes)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          Node.js          Browser       Cloudflare
```

## Related

- PR #344: LLVM Phase 1 (core target)
- PR #345: LLVM Phase 2 (C ABI)
- PR #346: LLVM Phase 3 (Go/Rust)
