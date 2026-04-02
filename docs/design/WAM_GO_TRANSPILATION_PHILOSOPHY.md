# WAM-to-Go Transpilation: Philosophy

## The Problem

UnifyWeaver's Go target handles arithmetic, guards, facts, recursive
patterns, transitive closure, and if-then-else natively. It also has
substantial pipeline, aggregation, and database modes. But some Prolog
predicates resist native lowering: genuine unification with nested
structures, non-deterministic choice points, mutual recursion with
backtracking. Today, when the Go target encounters such predicates,
compilation either falls back to a comment or fails silently.

The WAM target (Phase 1, PRs #1086–#1152) provides the missing layer.
The next step — as with Rust (PR #1153) — is connecting these, using
WAM as a fallback for predicates that resist native lowering, producing
a Go module that mixes natively-lowered functions with WAM-compiled ones.

## Design Principle: Transpile, Don't Rewrite

Rather than manually writing a WAM runtime in Go, UnifyWeaver should
**transpile its own WAM runtime** (`wam_runtime.pl`) to Go using its
compilation infrastructure. This achieves:

- **Single source of truth** — improvements to the Prolog WAM runtime
  automatically propagate to the Go version
- **Dog-fooding** — proves the compiler handles its own code
- **Correctness by construction** — the transpiled runtime inherits the
  tested semantics of the Prolog version

This is the same principle as the Rust design (PR #1153). The Go
implementation differs in idiom but shares the same compilation pipeline.

## The Compilation Spectrum

```
Most Idiomatic                              Most Coverage
─────────────────────────────────────────────────────────
Templates    Native Lowering    WAM-Compiled    Interpreted
(Mustache)   (clause_body)      (transpiled)    (runtime)
```

For a single Go output package:

```
factorial/2    → native lowering → func factorial(n int) int { ... }
grandparent/2  → WAM-compiled    → func grandparent(vm *WamState, ...)
step_wam/3     → native lowering → switch instr.(type) { case *GetConstant: ... }
eval_arith/4   → native lowering → func evalArith(expr Value) float64
backtrack/2    → WAM-compiled    → func backtrack(state *WamState) bool
```

## Go vs Rust: Key Design Differences

The WAM-to-Rust design (PR #1153) uses Rust's `enum` + `match` for
instruction dispatch. Go requires a different approach:

### Instruction Dispatch

**Rust:** Single enum with exhaustive match
```rust
match instr {
    Instruction::GetConstant(c, ai) => { ... }
    Instruction::GetVariable(xn, ai) => { ... }
}
```

**Go:** Interface type + type switch
```go
switch i := instr.(type) {
case *GetConstant:
    // i.C, i.Ai available
case *GetVariable:
    // i.Xn, i.Ai available
}
```

Go's type switch on interface values is the natural analog of Rust's
pattern matching on enums. Both provide exhaustive dispatch over a
closed set of variants.

### Memory Model

**Rust:** Ownership + borrowing, zero-copy where possible
```rust
fn step(&mut self, instr: &Instruction) -> bool
```

**Go:** Garbage collected, pointer semantics
```go
func (vm *WamState) Step(instr Instruction) bool
```

Go's GC simplifies the WAM heap — no manual deallocation needed. The
trail/backtracking mechanism is simpler because Go doesn't need to
reason about lifetimes of trail entries.

### Value Type

**Rust:** `enum Value { Atom(String), Integer(i64), ... }` with pattern matching

**Go:** Interface-based with type assertions
```go
type Value interface{ valueTag() }
type Atom struct{ Name string }
type Integer struct{ Val int64 }
type Compound struct{ Functor string; Args []Value }
type Ref struct{ Addr int }
type Unbound struct{ Name string }
```

Go's `interface{}` (or `any`) could also work, but a sealed interface
with a marker method provides type safety without generics.

### Concurrency: A Go Advantage

Go has first-class concurrency via goroutines and channels. This opens
a design path not available in single-threaded Rust:

- **Parallel choice exploration**: Each choice point spawns a goroutine
  that explores one alternative. Solutions are sent back on a channel.
- **Bounded parallelism**: A worker pool limits concurrent explorations.
- **First-solution optimization**: Cancel remaining goroutines once the
  first solution is found.

This is optional and orthogonal to the core transpilation — the initial
implementation should use sequential backtracking (matching Rust). But
Go's goroutine model makes parallel search a natural extension.

### Order-Independent Parallelism

Not all parallelism requires WAM choice points. Predicates that are
**order-independent** — where clause order doesn't affect the result
set — can parallelize at the goal level without backtracking machinery:

```prolog
:- order_independent(node_score/2).
node_score(X, S) :- feature_a(X, A), feature_b(X, B), S is A + B.
```

When a predicate is declared `order_independent` (or proven so by
static analysis), its body goals can execute concurrently:

```go
func nodeScore(x Value) Value {
    var a, b Value
    var wg sync.WaitGroup
    wg.Add(2)
    go func() { defer wg.Done(); a = featureA(x) }()
    go func() { defer wg.Done(); b = featureB(x) }()
    wg.Wait()
    return &Integer{a.(*Integer).Val + b.(*Integer).Val}
}
```

**When parallelism is safe:**

- **Declared**: User annotates with `:- order_independent(pred/arity)`
  or `:- parallel_safe(pred/arity)`. This is a promise that clause
  order doesn't matter and goals have no observable side effects on
  each other.

- **Proven**: Static analysis detects independence:
  - All body goals are pure (no I/O, no assert/retract)
  - No shared mutable state between goals
  - Goals bind disjoint variable sets
  - Commutative operations (e.g. set union, bag collection)

**Granularity levels:**

1. **Goal-level**: Independent body goals in a single clause run
   concurrently (as above). Safe when goals bind disjoint variables.

2. **Clause-level**: Multiple clauses of the same predicate run
   concurrently, collecting all solutions. Safe when clause order
   doesn't affect the result set (true for all `findall`-like usage).

3. **Predicate-level**: Independent predicate calls in a pipeline
   run concurrently. Safe when predicates don't share mutable state.

**Interaction with WAM:**

Order-independent predicates don't need WAM at all — they can be
natively lowered with goroutine parallelism. This means parallelism
applies even to predicates that are too simple for WAM but complex
enough to benefit from concurrent execution (e.g. multi-feature
scoring, independent constraint checks, parallel data lookups).

The key insight: **parallelism and backtracking are orthogonal**.
Goroutine parallelism applies to deterministic predicates that happen
to have independent subgoals. WAM backtracking applies to non-
deterministic predicates with choice points. Go can do both.

## Templates vs Native Lowering: Both, Applied Recursively

As with Rust, UnifyWeaver applies both templates and native lowering
at every nesting level:

1. **Mustache templates** define the Go package structure: `go.mod`,
   `value.go`, `state.go`, `instructions.go`, `runtime.go`
2. **Native lowering** compiles `step_wam/3` clauses to `switch` cases,
   `eval_arith/4` to recursive expression evaluation
3. **Nested control flow** recursively dispatches back through the same
   pipeline via `compile_expression/6`

## Interop Between Native and WAM-Compiled Predicates

The same calling convention as Rust, adapted to Go idioms:

- **Native → WAM**: The caller constructs a `WamState`, sets argument
  registers, and calls `vm.Run()`. The WAM executor runs until
  `Proceed` or failure.
- **WAM → Native**: When WAM encounters a `BuiltinCall` instruction
  for a natively-lowered predicate, it reads arguments from registers,
  calls the native Go function, and stores results back.

```go
// Native calls WAM-compiled predicate:
func queryAncestor(a, b string) bool {
    vm := NewWamState(ancestorCode, ancestorLabels)
    vm.SetReg("A1", &Atom{a})
    vm.SetReg("A2", &Atom{b})
    return vm.Run()
}

// WAM calls native predicate:
case *BuiltinCall:
    switch i.Op {
    case "factorial/2":
        n := vm.GetRegInt("A1")
        result := factorial(n)  // call native func
        vm.SetReg("A2", &Integer{result})
        vm.PC++
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

Compiles to a single Go package where:

- `factorial` is natively lowered (recursive arithmetic)
- `ancestor` might be natively lowered (transitive closure) or
  WAM-compiled (if the pattern detector doesn't recognize it)
- `unify_complex` is WAM-compiled (deep structure unification)
- The WAM runtime was itself transpiled from `wam_runtime.pl`

All three functions interoperate seamlessly within the same Go binary.
