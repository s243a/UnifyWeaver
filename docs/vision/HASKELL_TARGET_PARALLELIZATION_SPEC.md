# Haskell WAM Target: Parallelization Specification

## Overview

This document specifies the parallelization model for the Haskell WAM
target. The model is phased, starting with the simplest form (seed-level
parallelism) and progressing to intra-query parallelism as the C# demand
analysis matures.

## 1. Seed-Level Parallelism

### 1.1 Model

The effective-distance benchmark (and similar workloads) queries multiple
independent seeds:

```haskell
-- Current: sequential
seedResults <- mapM querySeed seedCats

-- Parallel: no code change needed in the WAM runtime
seedResults <- mapConcurrently querySeed seedCats
-- or: parMap rdeepseq querySeed seedCats
```

Each seed query receives:
- **Shared (immutable):** `WamContext` (code array, labels, foreign
  facts, foreign config, lowered predicates)
- **Independent (per-seed):** `WamState` (fresh `emptyState` with
  seed-specific register setup)

No synchronization needed. No shared mutable state.

### 1.2 Requirements

- GHC `-threaded` runtime flag
- `+RTS -N` to enable multi-core
- `Control.Parallel.Strategies` or `Control.Concurrent.Async`
- `NFData` instance for `Value` and `WamState` (for `rdeepseq`)

### 1.3 Expected gain

Near-linear speedup with core count for seed-dominated workloads.
At 300 scale (386 seeds), 4 cores should give ~3.5x speedup (limited
by seed work variance and GC contention).

## 2. Intra-Query Parallelism

### 2.1 Model

Within a single query, parallelism arises at disjunction points:

```prolog
category_ancestor(Cat, Root, Hops, Visited) :-
    category_parent(Cat, Mid),     % multiple solutions
    \+ member(Mid, Visited),
    category_ancestor(Mid, Root, H1, [Mid|Visited]),
    Hops is H1 + 1.
```

Each `Mid` solution spawns an independent recursive subquery. With
immutable state, these can run in parallel:

```haskell
-- At a TryMeElse choice point:
let snapshot = currentState  -- immutable, free to share
    branch1 = run ctx (snapshot { wsPC = clause1PC })
    branch2 = run ctx (snapshot { wsPC = clause2PC })
in branch1 `par` branch2 `pseq` merge branch1 branch2
```

### 2.2 Fork conditions

Not all choice points should be parallelized. Forking has overhead
(~microseconds for thread creation + GC coordination). A fork is
profitable when:

1. **The predicate is annotated or proven pure** (no side effects,
   no assert/retract, no I/O).
2. **The estimated branch work exceeds the fork threshold.**
   The C# query engine's demand analysis provides selectivity
   estimates and recursion depth bounds that can feed this decision.
3. **The branch count is small** (2-8 branches, not 6000 fact clauses).

### 2.3 Instruction design

The Go hybrid WAM target has parallelization instructions. The Haskell
equivalent would be:

```
ParTryMeElse Label    -- like TryMeElse but marks the CP as forkable
ParRetryMeElse Label  -- like RetryMeElse for parallel branches
ParTrustMe            -- last parallel branch
```

At runtime, `ParTryMeElse` checks a runtime flag or threshold:
- If forking is enabled and work estimate exceeds threshold: fork
- Otherwise: fall back to sequential TryMeElse semantics

### 2.4 Merge semantics

The merge strategy depends on the query context:

| Context | Merge strategy |
|---|---|
| `aggregate_all(sum, ...)` | Sum partial results from each branch |
| `aggregate_all(count, ...)` | Sum counts |
| `findall(X, ...)` | Concatenate solution lists |
| Bare disjunction | First success (race), or all solutions |
| `\+` (negation) | Any branch succeeds → negation fails |

The merge strategy is determined at compile time from the enclosing
aggregate/findall context. Bare disjunctions default to "all solutions"
(standard Prolog semantics).

## 3. Proving Order Independence

### 3.1 Sources of order dependence

A predicate is order-dependent if:
- It uses `assert`/`retract` (modifies the clause database)
- It performs I/O (`write`, `read`, `format`)
- It uses `!` (cut) in a way that depends on evaluation order
- It accesses global mutable state

### 3.2 Analysis pipeline

The C# query engine has infrastructure for:
- **Mode analysis:** which arguments are input (+) vs output (-)
- **Selectivity estimation:** how many solutions a predicate produces
- **Goal reordering:** proving that goals can be rearranged

These analyses can be extended to produce a **purity certificate** for
each predicate: a compile-time proof that the predicate is safe to
evaluate in parallel.

### 3.3 Annotations

For predicates that can't be automatically proven pure, the user can
declare parallelizability:

```prolog
:- parallel(category_ancestor/4).
:- order_independent(power_sum_bound/3).
```

These annotations are consumed by the WAM compiler to emit
`ParTryMeElse` instead of `TryMeElse`.

## 4. Mutable-State Sections

### 4.1 The hybrid approach

When demand analysis identifies a non-forking section (guaranteed
sequential execution), the WAM can use mutable state (ST monad) for
that section:

```haskell
-- Sequential section: mutable for speed
runSequential :: WamContext -> WamState -> ST s WamState
runSequential ctx s = do
  pcRef <- newSTRef (wsPC s)
  regsRef <- newSTRef (wsRegs s)
  -- ... fast in-place updates ...
  
-- Fork point: freeze back to immutable
snapshot <- freezeState pcRef regsRef ...
let branch1 = runST (thawAndRun ctx snapshot goals1)
    branch2 = runST (thawAndRun ctx snapshot goals2)
in ...
```

### 4.2 Prerequisites

This requires:
1. C# demand analysis mature enough to identify non-forking sections
2. State monad adopted as code organization pattern (eases ST transition)
3. Profiling showing that allocation is the bottleneck for the specific
   workload (not all workloads are allocation-bound)

## References

- Go hybrid WAM parallelization instructions — reference design
- C# query engine demand analysis — purity proofs, selectivity
- `docs/design/WAM_HASKELL_PERF_IMPLEMENTATION_PLAN.md` §8.4 — profiling
  data showing 48.6% allocation in step
- `GHC.Conc`, `Control.Parallel.Strategies` — GHC parallelism primitives
