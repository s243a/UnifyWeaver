# WAM-to-ILAsm Transpilation: Philosophy

## The Problem

UnifyWeaver's ILAsm target handles arithmetic, guards, facts, all five
recursion patterns (tail, linear, tree, mutual, general), transitive
closure, and if-then-else natively via the compile_expression framework
and multifile recursion dispatch. But some Prolog predicates resist
native lowering: genuine unification with nested structures, non-
deterministic choice points, mutual recursion with backtracking. Today,
when the ILAsm target encounters such predicates, compilation falls
back to a `throw` instruction.

The WAM target (PRs #1086–#1152) provides the missing layer. The next
step — as with Rust (PR #1153), Go, and LLVM (PR #1200) — is
connecting these, using WAM as a fallback for predicates that resist
native lowering, producing a CIL assembly module that mixes natively-
lowered methods with WAM-compiled ones.

## Design Principle: Transpile, Don't Rewrite

Rather than manually writing a WAM runtime in CIL, UnifyWeaver should
**transpile its own WAM runtime** (`wam_runtime.pl`) to CIL using its
compilation infrastructure. This achieves:

- **Single source of truth** — improvements to the Prolog WAM runtime
  automatically propagate to the CIL version
- **Dog-fooding** — proves the compiler handles its own code
- **Correctness by construction** — the transpiled runtime inherits the
  tested semantics of the Prolog version (28 tests, CI-verified)

This is the same principle as the Rust, Go, and LLVM designs. The CIL
implementation benefits from the CLR's garbage collector and native
`.tail` prefix support, simplifying several aspects that require manual
management in other targets.

## The Compilation Spectrum

```
Most Idiomatic                              Most Coverage
─────────────────────────────────────────────────────────
Templates    Native Lowering    WAM-Compiled    Interpreted
(Mustache)   (clause_body)      (transpiled)    (runtime)
```

For a single CIL output assembly:

```
factorial/2    → native lowering → .method static int64 factorial(int64)
grandparent/2  → WAM-compiled    → .method static bool grandparent(WamState)
step_wam/3     → native lowering → switch (N0, N1, ...) { ... }
eval_arith/4   → native lowering → .method static int64 eval_arith(Value)
backtrack/2    → WAM-compiled    → .method static bool backtrack(WamState)
```

## CIL vs Rust vs Go vs LLVM: The Key Differences

CIL (Common Intermediate Language) is a **stack-based managed bytecode**
that runs on the .NET CLR. This gives it unique properties among
UnifyWeaver's WAM hybrid targets:

### Advantage: Garbage Collection

The CLR provides automatic garbage collection. This eliminates the
entire memory management layer that other targets must implement:

| Target | Memory Strategy |
|--------|----------------|
| Rust | Ownership/borrowing (manual lifetime management) |
| Go | GC (built into runtime) |
| LLVM | Arena allocation (manual bump + rewind) |
| LLVM WASM | Bump allocator (no malloc, manual rewind) |
| **CIL** | **CLR GC (automatic, generational, compacting)** |

For the WAM runtime, this means:
- Registers stored in a `Dictionary<string, Value>` or `object[]` — GC
  handles deallocation
- Choice point state cloned via `new ChoicePoint(regs.Clone())` — GC
  handles old copies
- Trail entries are `List<TrailEntry>` — no manual arena management
- Backtracking simply restores references; GC reclaims orphaned objects

### Advantage: Native `.tail` Prefix

CIL has a `.tail` prefix for call instructions that guarantees tail
call optimization. This is unique among the targets:

| Target | Tail Call Strategy |
|--------|-------------------|
| Rust | while loop (no TCO guarantee) |
| Go | for loop (Go compiler doesn't guarantee TCO) |
| LLVM | `musttail` (guaranteed by LLVM) |
| LLVM WASM | Iterative loop (WASM has no TCO) |
| **CIL** | **`.tail call` (native CLR support)** |

The WAM `run_loop` can use `.tail call` for constant-stack execution:

```il
run_loop:
    ldarg.0                          // push WamState
    call bool PrologGenerated::step(class WamState)
    brtrue continue
    // backtrack or fail...
continue:
    ldarg.0
    .tail
    call bool PrologGenerated::run_loop(class WamState)
    ret
```

### Advantage: Rich Type System

CIL supports classes, interfaces, generics, and boxing/unboxing
natively. The `Value` type can be a proper class hierarchy:

```il
.class public abstract auto ansi Value extends [mscorlib]System.Object { }
.class public auto ansi AtomValue extends Value {
    .field public string Name
}
.class public auto ansi IntegerValue extends Value {
    .field public int64 Val
}
```

This gives us proper virtual dispatch, type checking via `isinst`,
and pattern matching via type tests — more natural than the tagged
union approach needed for LLVM.

### Stack-Based Execution Model

CIL is stack-based (push/pop evaluation stack), unlike LLVM's SSA
registers. This affects code generation:

| LLVM | CIL |
|------|-----|
| `%val = load %Value, %Value* %ptr` | `ldloc val` (push onto stack) |
| `store %Value %val, %Value* %ptr` | `stloc val` (pop from stack) |
| `%tag = extractvalue %Value %v, 0` | `ldfld int32 Value::tag` |
| `switch i32 %tag, ...` | `switch (N0, N1, ...)` |
| `call i1 @step(...)` | `call bool step(...)` |

The stack model is actually simpler for code generation — no SSA
variable naming, no phi nodes, no dominance requirements.

## Value Representation: Class Hierarchy vs Tagged Union

Two options for the WAM `Value` type in CIL:

**Option A: Class hierarchy** (preferred)
```il
.class public abstract Value extends [mscorlib]System.Object { }
.class public AtomValue extends Value { .field public string Name }
.class public IntegerValue extends Value { .field public int64 Val }
.class public FloatValue extends Value { .field public float64 Val }
.class public CompoundValue extends Value {
    .field public string Functor
    .field public class Value[] Args
}
.class public ListValue extends Value { .field public class Value[] Elements }
.class public RefValue extends Value { .field public int32 Addr }
.class public UnboundValue extends Value { .field public string Name }
```

**Option B: Tagged valuetype** (like LLVM)
```il
.class public sequential Value extends [mscorlib]System.ValueType {
    .field public int32 Tag
    .field public int64 Payload
}
```

**Option A is preferred** because:
- CIL's `isinst` instruction provides zero-cost type checks
- Virtual dispatch handles polymorphic operations naturally
- `string` fields for atoms avoid the interning table needed in LLVM
- GC handles all allocation — no manual layout management
- It's idiomatic .NET; interop with C#/F# code is natural

## Instruction Dispatch: CIL `switch`

CIL has a native `switch` instruction that takes the top-of-stack
integer and branches to one of N labels:

```il
// step_wam dispatch
ldarg.1                              // push instruction
ldfld int32 Instruction::Tag
switch (L_get_constant, L_get_variable, L_get_value,
        L_get_structure, L_get_list, L_unify_variable, ...)
br L_default

L_get_constant:
    // ... handle get_constant
    ret
L_get_variable:
    // ... handle get_variable
    ret
```

CIL `switch` compiles to an efficient jump table on the CLR JIT —
the same optimization that LLVM's `switch` gets, but expressed more
compactly.

## Interop Between Native and WAM-Compiled Methods

- **Native → WAM**: The caller constructs a `WamState`, sets argument
  registers, and calls `run_loop`. The WAM executor runs until
  `Proceed` or failure.
- **WAM → Native**: When WAM encounters a `BuiltinCall` instruction
  for a natively-lowered predicate, it reads arguments from registers,
  calls the native CIL method, and stores results back.

```il
// Native calls WAM-compiled predicate:
.method public static bool query_ancestor(string a, string b) cil managed {
    newobj instance void WamState::.ctor(...)
    dup
    ldstr "A1"
    ldarg.0
    newobj instance void AtomValue::.ctor(string)
    callvirt instance void WamState::SetReg(string, class Value)
    // ... set A2 ...
    call bool PrologGenerated::run_loop(class WamState)
    ret
}
```

## What Success Looks Like

A Prolog module with mixed complexity:

```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

unify_complex(f(X, g(Y)), f(a, g(b))) :- X = a, Y = b.
```

Compiles to a single CIL assembly where:

- `factorial` is natively lowered (iterative loop)
- `ancestor` might be natively lowered (TC pattern) or WAM-compiled
  (if the pattern detector doesn't recognize it)
- `unify_complex` is WAM-compiled (deep structure unification)
- The WAM runtime was itself transpiled from `wam_runtime.pl`

All three methods interoperate seamlessly within the same .NET
assembly, running on the CLR with full garbage collection support.
