# WAM-to-Rust Transpilation: Philosophy

## The Problem

UnifyWeaver's Rust target handles arithmetic, guards, facts, recursive
patterns, and if-then-else natively. But some Prolog predicates resist
native lowering: genuine unification with nested structures, non-
deterministic choice points, mutual recursion with backtracking. Today,
when the Rust target encounters such predicates, compilation simply
fails.

The WAM target (Phase 1, PRs #1086–#1152) provides the missing layer:
a symbolic compiler and runtime that preserves full Prolog semantics.
The next step is connecting these — using WAM as a fallback for
predicates that resist native lowering, producing a Rust module that
mixes natively-lowered functions with WAM-compiled ones.

## Design Principle: Transpile, Don't Rewrite

Rather than manually writing a WAM runtime in Rust, UnifyWeaver should
**transpile its own WAM runtime** (`wam_runtime.pl`) to Rust using its
compilation infrastructure. This achieves:

- **Single source of truth** — improvements to the Prolog WAM runtime
  automatically propagate to the Rust version
- **Dog-fooding** — proves the compiler handles its own code
- **Correctness by construction** — the transpiled runtime inherits the
  tested semantics of the Prolog version (28 tests, CI-verified)

## The Compilation Spectrum for WAM Predicates

UnifyWeaver's philosophy is a spectrum from maximum idiom to maximum
coverage:

```
Most Idiomatic                              Most Coverage
─────────────────────────────────────────────────────────
Templates    Native Lowering    WAM-Compiled    Interpreted
(Mustache)   (clause_body)      (transpiled)    (runtime)
```

For a single Rust output module, this means:

```
factorial/2    → native lowering → fn factorial(n: i64) -> i64 { ... }
grandparent/2  → WAM-compiled    → fn grandparent(vm: &mut WamState, ...)
step_wam/3     → native lowering → match instr { GetConstant(..) => ... }
eval_arith/4   → native lowering → fn eval_arith(expr: &Value) -> f64
backtrack/2    → WAM-compiled    → fn backtrack(state: &mut WamState) -> bool
```

The key insight: **the WAM runtime itself is a Prolog program**, and
most of its predicates (arithmetic, type checks, register helpers) are
simple enough for native lowering. Only the most complex predicates
(if any) would need the WAM fallback — a form of recursive self-
application.

## Templates vs Native Lowering: Both, Applied Recursively

UnifyWeaver's architecture does not choose between templates and native
lowering — it applies both recursively at every nesting level:

1. **Template** provides the idiomatic Rust skeleton (struct definitions,
   enum declarations, trait implementations, module layout)
2. **Native lowering** fills in the function bodies via clause body
   analysis and expression translation
3. **Nested control flow** recursively dispatches back through the same
   pipeline via `compile_expression/6`

For the WAM runtime transpilation:

- **Mustache templates** define the Rust crate structure:
  `Cargo.toml`, `lib.rs`, `value.rs`, `state.rs`, `instructions.rs`
- **Native lowering** compiles `step_wam/3` clauses to `match` arms,
  `eval_arith/4` to recursive expression evaluation, `is_unbound_var/1`
  to a simple `matches!()` check
- **`shared_logic` pattern** (from agent-loop) could define WAM
  instruction semantics once, expand to Rust/WAT/JVM differently

## AST Analysis Drives the Pipeline

UnifyWeaver's `clause_body_analysis.pl` provides the structural analysis
that makes this possible:

- **`classify_goal/3`** determines whether a goal is a guard, output, or
  control flow — this drives how `step_wam` clauses get lowered
- **`translate_expr/3`** normalizes Prolog expressions to a target-
  independent AST (`op(+, var(x), literal(2))`) that Rust can consume
- **`compile_expression/6`** is the recursive entry point where templates
  and native lowering compose at every level

The `step_wam/3` predicate is a multi-clause dispatch on compound first
arguments — exactly the pattern that `switch_on_structure` indexing
handles. The Rust target's `match` expression is the natural lowering
for this pattern.

## Interop Between Native and WAM-Compiled Predicates

A critical requirement: natively-lowered Rust functions must be callable
from WAM-compiled code and vice versa. The calling convention:

- **Native → WAM**: The caller constructs a `WamState`, sets argument
  registers, and calls the WAM entry point. The WAM executor runs until
  `proceed` or failure.
- **WAM → Native**: When WAM encounters a `builtin_call` or `call` to a
  natively-lowered predicate, it reads arguments from registers, calls
  the native Rust function, and stores results back.

This is the same pattern as the existing `builtin_call` mechanism in the
WAM runtime — builtins like `is/2`, `member/2`, `>/2` already bridge
between WAM execution and native Prolog evaluation. The Rust version
extends this to natively-lowered user predicates.

## What Success Looks Like

A Prolog module with mixed complexity:

```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

unify_complex(f(X, g(Y)), f(a, g(b))) :- X = a, Y = b.
```

Compiles to a single Rust crate where:

- `factorial` is natively lowered (recursive arithmetic)
- `ancestor` might be natively lowered (transitive closure pattern) or
  WAM-compiled (if the pattern detector doesn't recognize it)
- `unify_complex` is WAM-compiled (deep structure unification)
- The WAM runtime support code was itself transpiled from `wam_runtime.pl`

All three functions interoperate seamlessly within the same Rust binary.
