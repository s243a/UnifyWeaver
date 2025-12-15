# feat: Add LLVM IR target with musttail optimization and multi-target integration

## Summary

This PR adds a new LLVM IR compilation target to UnifyWeaver, enabling direct compilation of Prolog predicates to native code with guaranteed tail call optimization.

## New Features

### LLVM Target (`llvm_target.pl`)
- **6 compilation predicates** for different patterns
- **Guaranteed tail call optimization** via `musttail` instruction
- **C ABI export** for FFI with Go/Rust/C
- **Cross-platform** - compiles to any LLVM target (x86, ARM, RISC-V, WASM)

### Recursion Patterns

| Pattern | API | LLVM Feature |
|---------|-----|--------------|
| Facts | `compile_facts_to_llvm/3` | Global arrays |
| Tail Recursion | `compile_tail_recursion_llvm/3` | `musttail call` |
| Linear Recursion | `compile_linear_recursion_llvm/3` | Static memo table |
| Mutual Recursion | `compile_mutual_recursion_llvm/3` | Cross-function `musttail` |
| Transitive Closure | `compile_transitive_closure_llvm/3` | BFS worklist |

### Bindings (`llvm_bindings.pl`)
- Arithmetic: `add`, `sub`, `mul`, `sdiv`, `srem`
- Comparisons: `icmp eq/ne/slt/sle/sgt/sge`
- Control: `br`, `ret`, `musttail call`

## Testing

- **6/6 unit tests pass**
- **2 integration tests pass**:
  - `sum.ll` - tail recursion compiles and runs
  - `reachable.ll` - BFS transitive closure works

## Documentation

### Main Docs
- `docs/LLVM_TARGET.md`
- `docs/proposals/llvm_target_design.md` (3-phase roadmap)

### Education Book
- `book-llvm-target/README.md`
- `book-llvm-target/01_introduction.md`
- `book-llvm-target/02_integration.md` - C, Go, Rust FFI
- `book-llvm-target/03_recursive_queries.md`

## Files Changed

```
src/unifyweaver/targets/llvm_target.pl       [NEW] 500 lines
src/unifyweaver/bindings/llvm_bindings.pl    [NEW] 145 lines
tests/test_llvm_target.pl                    [NEW] 95 lines
docs/LLVM_TARGET.md                          [NEW]
docs/proposals/llvm_target_design.md         [NEW]
build/sum.ll, sum.o, sum                     [NEW] integration test
build/reachable.ll, reachable.o, reachable   [NEW] integration test
```

## Usage

```bash
# Generate LLVM IR
swipl -g "compile_tail_recursion_llvm(sum/2, [], Code), write(Code)" > sum.ll

# Fix escaping and compile
sed -i 's/%%/%/g' sum.ll
llc -filetype=obj sum.ll -o sum.o
clang sum.o -o sum
./sum
```

## Dependencies

- LLVM toolchain (`llc`, `clang`)

## Future Work (Phase 2-3)

- C ABI shared libraries for multi-target glue
- Go integration via cgo
- Rust integration via FFI
- Shared memory between LLVM and other targets
