# Haskell WAM Target: Philosophy

## Why Haskell?

Haskell's type system and immutable-by-default data model make it
uniquely suited as a WAM target for three reasons:

1. **Correctness by construction.** The WAM's complex state transitions
   (register files, choice points, trail, bindings) are type-checked at
   compile time. Incorrect state access is a type error, not a runtime
   crash.

2. **Parallelism without data races.** Immutable `WamContext` (code
   array, label map, foreign facts) is shared read-only across threads.
   Immutable `WamState` snapshots can be forked to parallel branches
   without synchronization. This is not possible in mutable-state
   targets like Rust or C without explicit locking.

3. **GHC's optimizer.** `-O2` with `BangPatterns` and strict fields
   produces code competitive with handwritten C for tight loops. The
   generated WAM interpreter benefits from GHC's inlining, constructor
   specialization, and worker-wrapper transformation.

## The optimization pipeline

UnifyWeaver's performance architecture is a multi-stage pipeline where
each stage amplifies the next:

```
C# Query Engine          Rust/Haskell/LLVM WAM Targets
  (demand analysis,        (compiled execution,
   selectivity,      →      FFI kernels,
   pruning)                  lowered functions,
       ↓                     parallelization)
Prolog Target
  (optimized Prolog:
   seeded accumulation,
   branch pruning)
```

The C# engine discovers optimization opportunities (which branches to
prune, which aggregations to precompute, which predicates are order-
independent). The Prolog target materializes these as optimized Prolog
predicates. The WAM targets compile those optimized predicates to
efficient native code.

This separation matters: the WAM targets don't need to rediscover
optimization opportunities — they receive already-optimized Prolog and
compile it faithfully. The intelligence lives in the C# analysis; the
WAM targets provide generality.

## Immutability as a strategic asset

The decision to keep `WamState` immutable is deliberate, not a
limitation:

- **Parallelism.** Immutable state enables `parMap` over seed queries
  (each seed gets the same `WamContext` + independent `WamState`). It
  also enables intra-query parallelism: forking at choice points by
  passing the same state snapshot to both branches.

- **Backtracking for free.** Choice points are immutable snapshots of
  the state at the branch point. Restoring on backtrack is just
  switching pointers — no undo log needed (the trail is only needed
  for variable bindings, not for register/stack restoration).

- **Debugging and replay.** Every `WamState` is a complete, consistent
  snapshot. You can serialize it, compare it, or replay from any point.

The cost of immutability — record copying on every step — is real
(~48% of allocation). But it's a constant factor, not an algorithmic
issue. The FFI path (which handles the hot predicates natively) already
beats SWI-Prolog, so the interpreter overhead matters less than the
architectural flexibility.

## When to introduce mutability

Mutability (via the ST monad) should only be introduced when:

1. The **C# demand analysis** can identify non-forking sections —
   sequential execution paths where no parallel branch will ever need
   the intermediate state.

2. The **freeze/thaw cost** at fork points is amortized by sufficient
   branch work (microseconds of overhead vs milliseconds of parallel
   gain).

3. The **State monad** has been adopted as a code organization pattern
   first — making the transition to ST a mechanical change (`State s a`
   → `ST s a`) rather than an architectural one.

Until these conditions are met, immutable state is the right default.

## Relationship to other targets

- **Rust WAM:** Fastest (126ms at 300 scale). Uses mutable vectors for
  registers. Not parallelizable without explicit Arc/Mutex.
- **LLVM WAM:** Lowest-level, most optimization potential. Not yet
  benchmarked with optimized Prolog.
- **Go hybrid WAM:** Has parallelization instructions. Reference design
  for the Haskell parallelization work.
- **WAT:** WebAssembly target. Structured block semantics, no choice
  point parallelism.

The Haskell target's niche is **correct, parallelizable, optimized
execution** — not raw single-threaded speed (that's Rust/LLVM's
domain).

## References

- `docs/design/WAM_HASKELL_PERF_IMPLEMENTATION_PLAN.md` — optimization
  history and profiling data
- `docs/design/WAM_HASKELL_PERF_PHILOSOPHY.md` — earlier perf philosophy
  (focused on closing the SWI gap, predates parallelization vision)
- `docs/proposals/WAM_IF_THEN_ELSE_COMPILATION.md` — cross-target WAM
  compiler enhancement
- `docs/proposals/WAM_FOREIGN_DISPATCH_RETURN_TYPE.md` — CallForeign
  compile-time dispatch (resolved)
