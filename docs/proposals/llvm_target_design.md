# LLVM Target + Multi-Target Integration Design

## Overview

This proposal adds an LLVM IR compilation target to UnifyWeaver, enabling:
- **Guaranteed tail call optimization** via `musttail`
- **Cross-platform native code** (x86, ARM, RISC-V, WebAssembly)
- **C ABI compatibility** for linking with Go, Rust, C
- **Shared memory glue** between heterogeneous targets

## Motivation

### Current Limitations

| Target | Issue |
|--------|-------|
| GNU Prolog | Tail call optimization not guaranteed |
| SWI Prolog | Interpreted only |
| Go/Rust | No native tail call optimization |

### LLVM Solution

LLVM IR provides:
- `musttail` instruction for guaranteed tail call elimination
- Target-independent IR compiled to any architecture
- Standard C calling convention for FFI

---

## Architecture

### Phase 1: Pure LLVM Target

```
Prolog Predicate
      ↓
llvm_target.pl
      ↓
   .ll file (LLVM IR)
      ↓
   llc (LLVM compiler)
      ↓
   Native executable
```

### Phase 2: C ABI Integration

```
Prolog Predicate → LLVM IR → .so/.dll
                                ↓
                        C/Go/Rust programs
```

### Phase 3: Multi-Target Glue

```
┌─────────────────────────────────────┐
│           UnifyWeaver               │
├──────────┬──────────┬───────────────┤
│ llvm_target │ go_target │ rust_target │
│     ↓       │     ↓     │     ↓       │
│ libmath.so  │  main.go  │  main.rs    │
│     ↑       │     ↓     │     ↓       │
│     └───────┴─ cgo/FFI ─┴─────┘      │
└─────────────────────────────────────┘
```

---

## API Design

### Module: `llvm_target`

```prolog
:- module(llvm_target, [
    compile_predicate_to_llvm/3,       % +Pred/Arity, +Options, -LLVMCode
    compile_facts_to_llvm/3,           % +Pred, +Arity, -LLVMCode
    compile_tail_recursion_llvm/3,     % Uses musttail
    compile_linear_recursion_llvm/3,   % With memoization
    compile_mutual_recursion_llvm/3,   % Cross-function calls
    write_llvm_program/2,              % Write .ll file
    init_llvm_target/0
]).
```

### Options

```prolog
compile_predicate_to_llvm(factorial/1, [
    export(true),              % Generate extern "C" wrapper
    target_triple('x86_64-linux-gnu'),
    optimization(2)            % -O2
], Code).
```

---

## Generated Code Examples

### Tail Recursion

```prolog
sum(0, Acc, Acc).
sum(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum(N1, Acc1, Result).
```

**Generated LLVM IR:**
```llvm
define i64 @sum(i64 %n, i64 %acc) {
entry:
  %cmp = icmp sle i64 %n, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret i64 %acc

recurse:
  %n1 = sub i64 %n, 1
  %acc1 = add i64 %acc, %n
  %result = musttail call i64 @sum(i64 %n1, i64 %acc1)
  ret i64 %result
}
```

### Facts Export

```prolog
person(john, 25).
person(jane, 30).
```

**Generated LLVM IR:**
```llvm
@str.john = private constant [5 x i8] c"john\00"
@str.jane = private constant [5 x i8] c"jane\00"

@person_count = constant i64 2
@person_data = constant [2 x { i8*, i64 }] [
  { i8* @str.john, i64 25 },
  { i8* @str.jane, i64 30 }
]
```

---

## Integration Examples

### Go + LLVM (cgo)

```go
// #cgo LDFLAGS: -L. -lprolog_math
// #include "prolog_math.h"
import "C"

func main() {
    result := C.factorial(10)  // Calls LLVM-compiled code
    fmt.Println(result)
}
```

### Rust + LLVM (FFI)

```rust
extern "C" {
    fn factorial(n: i64) -> i64;
}

fn main() {
    let result = unsafe { factorial(10) };
    println!("{}", result);
}
```

---

## Dependencies

- LLVM toolchain (`llc`, `clang`)
- Optional: `lld` for linking

## Related Work

- [prolog_dialects.pl](../src/unifyweaver/targets/prolog_dialects.pl) - GNU Prolog dialect
- [go_target.pl](../src/unifyweaver/targets/go_target.pl) - Go compilation
- [rust_target.pl](../src/unifyweaver/targets/rust_target.pl) - Rust compilation

## Status

- [ ] Phase 1: Pure LLVM Target
- [ ] Phase 2: C ABI Integration
- [ ] Phase 3: Multi-Target Glue
