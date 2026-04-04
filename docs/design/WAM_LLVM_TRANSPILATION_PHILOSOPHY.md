# WAM-to-LLVM IR Transpilation: Philosophy

## The Problem

UnifyWeaver's LLVM target handles arithmetic, guards, facts, tail
recursion (`musttail`), linear recursion with memoization, mutual
recursion, and transitive closure natively. It generates typed LLVM IR
operating directly on `i64`, `double`, and `i1` — no boxing, no
indirection. But some Prolog predicates resist native lowering: genuine
unification with nested structures, non-deterministic choice points,
mutual recursion with backtracking. Today, when the LLVM target
encounters such predicates, compilation falls back to a stub that
returns its first argument unchanged.

The WAM target (PRs #1086–#1152) provides the missing layer. The next
step — as with Rust (PR #1153) and Go — is connecting these, using WAM
as a fallback for predicates that resist native lowering, producing an
LLVM IR module that mixes natively-lowered functions with WAM-compiled
ones.

## Design Principle: Transpile, Don't Rewrite

Rather than manually writing a WAM runtime in LLVM IR, UnifyWeaver
should **transpile its own WAM runtime** (`wam_runtime.pl`) to LLVM IR
using its compilation infrastructure. This achieves:

- **Single source of truth** — improvements to the Prolog WAM runtime
  automatically propagate to the LLVM version
- **Dog-fooding** — proves the compiler handles its own code
- **Correctness by construction** — the transpiled runtime inherits the
  tested semantics of the Prolog version (28 tests, CI-verified)

This is the same principle as the Rust and Go designs. The LLVM
implementation differs substantially in representation (no native
enums, structs are flat, dispatch is via `switch` on integer tags)
but shares the same compilation pipeline.

## The Compilation Spectrum

```
Most Idiomatic                              Most Coverage
─────────────────────────────────────────────────────────
Templates    Native Lowering    WAM-Compiled    Interpreted
(Mustache)   (clause_body)      (transpiled)    (runtime)
```

For a single LLVM IR output module:

```
factorial/2    → native lowering → musttail i64 @factorial(i64, i64)
grandparent/2  → WAM-compiled    → void @grandparent(%WamState*)
step_wam/3     → native lowering → switch i32 %tag { ... }
eval_arith/4   → native lowering → %Value @eval_arith(%Value)
backtrack/2    → WAM-compiled    → i1 @backtrack(%WamState*)
```

## LLVM vs Rust vs Go: The Key Difference

Rust and Go are high-level targets with built-in sum types (enum/
interface) and collection types (HashMap/map, Vec/slice). LLVM IR is
a **low-level typed SSA representation** — there are no enums, no
hash maps, no dynamic dispatch. Every abstraction must be lowered
to explicit memory layout and integer operations.

This is both a challenge and an advantage:

**Challenge:** We must define explicit memory layouts for `Value`
(tagged union), `WamState` (struct with pointer fields), `Instruction`
(tagged union), and the heap/trail/stack (pointer-based arrays). Hash
maps (used for registers and labels) must use a concrete implementation
— either a simple array-based map for small register sets, or link
against a C runtime hash map.

**Advantage:** LLVM IR is the closest to the metal of any UnifyWeaver
target. The generated code runs through LLVM's full optimization
pipeline (`opt`, `llc`) producing native machine code. This means:

- WAM-compiled predicates get **native-quality code** after LLVM's
  passes — instruction selection, register allocation, loop
  optimization — far beyond what an interpreter achieves
- The `musttail` guarantee already in the LLVM target extends to WAM:
  the main `run_loop` can be a tail-calling trampoline with guaranteed
  O(1) stack usage
- Cross-platform: the same `.ll` file compiles to x86, ARM, RISC-V,
  or WebAssembly via different `--target-triple` flags

## Two-Tier Value Representation

The existing LLVM target operates on unboxed `i64`/`double`/`i1` with
no runtime overhead. The WAM runtime needs a polymorphic `Value` type.
Rather than forcing all predicates into the boxed representation, the
hybrid approach uses **two tiers**:

1. **Unboxed tier** (existing): Natively-lowered predicates continue
   to use typed `i64`, `double`, `i1` parameters and returns. Zero
   overhead, full optimization.

2. **Boxed tier** (new): WAM-compiled predicates use a tagged-union
   `%Value` type. The tag indicates the variant; the payload is a
   union of the possible contents.

The interop bridge between tiers handles boxing/unboxing:

```llvm
; Unboxing: Value → i64 (for native calls from WAM)
define i64 @unbox_integer(%Value %v) {
  %payload = extractvalue %Value %v, 1    ; get the i64 payload
  ret i64 %payload
}

; Boxing: i64 → Value (for WAM calls from native)
define %Value @box_integer(i64 %n) {
  %v = insertvalue %Value { i32 1, i64 0 }, i64 %n, 1
  ret %Value %v
}
```

## Tagged Union Layout

LLVM IR's `insertvalue`/`extractvalue` and integer-width unions make
tagged unions natural:

```llvm
; Tag constants
; 0 = Atom, 1 = Integer, 2 = Float, 3 = Compound, 4 = List,
; 5 = Ref, 6 = Unbound, 7 = Bool

; Value is { tag: i32, payload: i64 }
; For pointer payloads (Atom name, Compound args, List elements),
; the i64 holds a pointer cast to integer.
%Value = type { i32, i64 }
```

This is a NaN-boxing-adjacent approach: the tag discriminates, and
the 64-bit payload holds either an integer, a double (bitcast), or a
pointer (inttoptr). This avoids heap allocation for scalar values.

For compound types (atoms with names, compounds with args, lists with
elements), the pointer payloads reference heap-allocated structures:

```llvm
%Compound = type { i8*, i32, %Value* }  ; functor, arity, args array
%List     = type { i32, %Value* }       ; length, elements array
```

## Instruction Dispatch: `switch` on Integer Tags

Where Rust uses `match` on enum variants and Go uses type switch,
LLVM IR uses `switch` on an integer tag — this is the natural and
efficient lowering:

```llvm
; Instruction tag constants
; 0 = GetConstant, 1 = GetVariable, 2 = GetValue, ...

define i1 @step(%WamState* %vm, %Instruction* %instr) {
entry:
  %tag_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 0
  %tag = load i32, i32* %tag_ptr
  switch i32 %tag, label %default [
    i32 0, label %get_constant
    i32 1, label %get_variable
    i32 2, label %get_value
    ; ...
  ]

get_constant:
  ; extract fields, perform unification
  ; ...
  ret i1 true

get_variable:
  ; copy register
  ; ...
  ret i1 true

default:
  ret i1 false
}
```

LLVM's `switch` instruction compiles to jump tables or binary search
depending on density — the optimizer handles this automatically.

## Register Representation: Array vs Hash Map

The WAM uses named registers (`A1`, `A2`, `X1`, `X2`, ...). In Rust
and Go, these are stored in a `HashMap`/`map[string]Value`. In LLVM
IR, we have two options:

**Option A: Fixed-size register array** (preferred for LLVM)
```llvm
%WamState = type {
  i32,                  ; pc
  [32 x %Value],       ; regs (A1..A16, X1..X16 — fixed slots)
  %Value*,             ; stack pointer
  i32,                 ; stack size
  %Value*,             ; heap pointer
  i32,                 ; heap size
  %TrailEntry*,        ; trail pointer
  i32,                 ; trail size
  i32,                 ; cp (continuation pointer)
  %ChoicePoint*,       ; choice point stack
  i32,                 ; choice point count
  %Instruction*,       ; code array
  i32,                 ; code length
  i32*,                ; labels (label index → PC mapping, parallel array)
  i32                  ; label count
}
```

Register names map to array indices at compile time:
`A1 → 0, A2 → 1, ..., X1 → 16, X2 → 17, ...`

This avoids the need for a hash map entirely — register access is a
single `getelementptr` + `load`/`store`. This is a major simplification
over the Rust/Go implementations and produces better code.

**Option B: Linked to C runtime hash map** (fallback for dynamic regs)

If a predicate needs more than 32 registers (unlikely for WAM), fall
back to linking against a C hash map implementation. But for typical
WAM code, fixed-size arrays suffice.

## Memory Management Strategy

LLVM IR has no garbage collector by default. Three options:

1. **Arena allocation** (preferred): Allocate a large arena at WAM
   startup. Heap cells, trail entries, and stack frames are bump-
   allocated from the arena. On backtracking, the arena pointer rewinds
   — this is exactly how Prolog implementations work (the WAM heap
   *is* an arena). No free calls needed during forward execution.

2. **Link against malloc/free**: Use C runtime allocation. Simple but
   requires careful lifetime tracking. Suitable for the initial
   implementation.

3. **LLVM GC intrinsics**: Use `@llvm.gcroot` and a shadow stack for
   GC-managed memory. Overkill for the WAM — the WAM's own trail-based
   memory management is already a form of region-based allocation.

Arena allocation aligns perfectly with WAM semantics: the heap grows
forward during execution and rewinds on backtracking. Trail entries
record what to undo. This maps directly to bump allocation + pointer
reset.

## Templates vs Native Lowering: Both, Applied Recursively

As with Rust and Go, UnifyWeaver applies both templates and native
lowering at every nesting level:

1. **Mustache templates** define the LLVM IR module structure: type
   definitions, global declarations, external function declarations
2. **Native lowering** compiles `step_wam/3` clauses to `switch` cases,
   `eval_arith/4` to recursive expression evaluation
3. **Nested control flow** recursively dispatches back through the same
   pipeline via `compile_expression/6`

## Interop Between Native and WAM-Compiled Predicates

The interop bridge handles the two-tier value system:

- **Native → WAM**: The caller boxes arguments into `%Value`, constructs
  a `%WamState`, sets argument registers, and calls the WAM entry point.
  The WAM executor runs until `proceed` or failure.
- **WAM → Native**: When WAM encounters a `builtin_call` to a natively-
  lowered predicate, it unboxes arguments from registers, calls the
  native function (typed `i64`/`double`), boxes the result, and stores
  it back in the register array.

```llvm
; Native calls WAM-compiled predicate:
define i1 @query_ancestor(i8* %a, i8* %b) {
  %vm = call %WamState* @wam_state_new(...)
  %val_a = call %Value @box_atom(i8* %a)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %val_a)
  %val_b = call %Value @box_atom(i8* %b)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %val_b)
  %result = call i1 @wam_run(%WamState* %vm)
  ret i1 %result
}

; WAM calls native predicate (in builtin_call dispatch):
; ... unbox A1 to i64, call @factorial(i64), box result to A2
```

## Run Loop as Tail-Calling Trampoline

The existing LLVM target already uses `musttail` for guaranteed tail
call elimination. The WAM `run_loop` is a natural fit:

```llvm
define i1 @run_loop(%WamState* %vm) {
entry:
  %pc = ; load vm->pc
  %instr = ; load vm->code[pc]
  %ok = call i1 @step(%WamState* %vm, %Instruction* %instr)
  br i1 %ok, label %continue, label %backtrack_or_fail

continue:
  %halted = ; check if proceed was reached
  br i1 %halted, label %success, label %next

next:
  %result = musttail call i1 @run_loop(%WamState* %vm)
  ret i1 %result

backtrack_or_fail:
  %can_bt = call i1 @backtrack(%WamState* %vm)
  br i1 %can_bt, label %next, label %failure

success:
  ret i1 true

failure:
  ret i1 false
}
```

With `musttail`, this runs in constant stack space regardless of how
many instructions the WAM program contains — matching the behavior of
a hand-written `while` loop but in pure SSA form.

## What Success Looks Like

A Prolog module with mixed complexity:

```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

unify_complex(f(X, g(Y)), f(a, g(b))) :- X = a, Y = b.
```

Compiles to a single `.ll` module where:

- `factorial` is natively lowered (`musttail` on unboxed `i64`)
- `ancestor` might be natively lowered (TC pattern) or WAM-compiled
  (if the pattern detector doesn't recognize it)
- `unify_complex` is WAM-compiled (deep structure unification) using
  the tagged-union `%Value` representation
- The WAM runtime was itself transpiled from `wam_runtime.pl`
- After `opt -O2` and `llc`, all three produce efficient native code

All three functions interoperate seamlessly within the same binary,
with the boxing/unboxing bridge handling the type-level boundary
between the unboxed native tier and the boxed WAM tier.
