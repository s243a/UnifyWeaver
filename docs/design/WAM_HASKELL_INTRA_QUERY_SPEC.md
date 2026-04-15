# Haskell WAM Intra-Query Parallelism: Specification

> **Scope:** This specifies the WAM instructions, runtime mechanisms, and
> semantic contracts for forking parallel branches at choice points
> within a single query. Fork/merge happens *inside* a `run`-loop
> invocation, in contrast to seed-level parallelism (PR #1377) which
> forks at the level of `parMap` over independent `WamState` snapshots.
>
> See `WAM_HASKELL_INTRA_QUERY_PHILOSOPHY.md` for motivation, and
> `WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md` for a phased work
> breakdown.

## 1. Conceptual model

A **forkable choice point** is a choice point that:

1. Has multiple branches whose execution doesn't share runtime state
   (no cross-branch trail leakage, no shared mutable structures).
2. Is in a context where the merge semantics are statically known
   (we know whether to sum, count, collect, race, or accumulate).
3. Has enough estimated work per branch to justify spark overhead.

When all three hold, the runtime forks the alternatives across cores.
Branches run independently using their own `WamState` snapshot of the
shared `WamContext`. When all branches complete, results are combined
according to the enclosing aggregate.

## 2. New WAM instructions

Add three instructions parallel to the existing choice-point family:

```haskell
data Instruction
  = ...
  | ParTryMeElse !Label    -- like TryMeElse but marks CP as forkable
  | ParRetryMeElse !Label  -- continuation in the parallel CP chain
  | ParTrustMe             -- last branch in the parallel CP chain
  ...
```

Pre-resolved variants (analogous to `TryMeElsePc` etc.) follow the
same naming convention: `ParTryMeElsePc`, `ParRetryMeElsePc`.

### Why three instructions, not one

The existing `TryMeElse` / `RetryMeElse` / `TrustMe` triplet establishes
the linked structure of a choice-point chain. Parallel emission needs
the same structure so the WAM compiler can recognize "this is a
forkable group of N alternatives" and the runtime can collect them
all before forking.

## 3. Generator support: purity annotations

A predicate is annotated parallelizable in one of two ways:

### 3.1 Explicit user annotation

```prolog
:- parallel(category_ancestor/4).
:- order_independent(power_sum_bound/3).
```

`parallel/1` declares the predicate's clauses safe to fork. The WAM
compiler emits `ParTryMeElse` for the multi-clause indexing that would
otherwise be `TryMeElse`.

`order_independent/1` is a weaker assertion: solutions can be returned
in any order, but the user is asserting (not proving) safety. It still
emits parallel instructions; difference is documentation/intent.

### 3.2 Compiler-derived purity (future)

If the C# query engine's demand analysis produces a *purity certificate*
for a predicate, the WAM compiler can emit `ParTryMeElse` automatically.
This is a future hook, not a Phase 4 deliverable.

### 3.3 Global kill-switch: `intra_query_parallel(false)` (future TODO)

A future addition to `write_wam_haskell_project/3`'s options list —
not needed for the Phase 4.1 MVP because the annotation is already
opt-in (unannotated predicates get sequential `TryMeElse` as before,
so the default *is* "off" predicate-by-predicate). The option would:

- Ignore every `:- parallel/1` / `:- order_independent/1` annotation.
- Ignore every C# purity certificate (see §3.2).
- Emit sequential `TryMeElse` / `RetryMeElse` / `TrustMe` for all
  multi-clause predicates regardless.

Useful once Par* emission exists, for:

- **Debugging.** Confirm a test failure is or isn't caused by the
  parallel path without touching sources.
- **Regression benchmarking.** Compare Par* vs sequential on the same
  generated project.
- **Host-portability.** Target a GHC build or runtime where spark
  overhead is intolerable (e.g. very small process, tight RTS budget).

Why it's *not* in the Phase 4.1 MVP: the annotation being opt-in is
already a stronger guarantee than a runtime flag. A predicate without
`:- parallel/1` can't, even in principle, be affected by Phase 4.1's
changes — there's no Par* instruction to gate. Add the option alongside
the first round of Par* emission (likely Phase 4.1 or 4.2), guarded by
a single `option(intra_query_parallel(true), Options, true)` check
around the emit call. Cost is trivial; we leave it out of the MVP only
to avoid shipping dead code.

A purity certificate guarantees:
- No `assert`/`retract` in the predicate or transitively
- No I/O
- No global mutable state access
- No cuts that escape the predicate's scope (`!` truncating to a
  caller's choice point is unsafe; `!` truncating to a clause-local
  choice point is fine)

## 4. Runtime fork mechanism

### 4.1 What gets snapshotted

A choice point already snapshots:
- `cpRegs`, `cpStack`, `cpBindings`, `cpHeap*`, `cpTrailLen`, `cpCutBar`

For parallel forking, we need each branch to have an *independent* copy
of the trail. The trail is currently a list shared across branches;
forking would require either:

1. **Per-branch trail copy.** Each parallel branch starts with the
   prefix trail (up to `cpTrailLen`) and accumulates its own bindings.
   Cost: O(trail length) per fork.
2. **Trail journaling.** Each branch records its trail entries in a
   thread-local buffer; the merge step interprets them. Cost: lower
   per-fork, but more complex.

**MVP picks option 1.** Trail copy is O(n) but n is typically small
(< 100 entries at choice points). Journaling can be a future
optimization if profiling demands it.

### 4.2 Spark primitives

```haskell
import Control.Parallel (par, pseq)
import Control.Parallel.Strategies (parMap, rdeepseq)

-- At a parallel choice point with branches [b1, b2, ..., bN]:
let snapshot = currentState
    branches = [run ctx (snapshot { wsPC = b }) | b <- branchPcs]
    parallelResults = parMap rdeepseq id branches
in mergeResults parallelResults
```

We use `parMap rdeepseq` over branch results — same primitive as
seed-level parallelism, just inside the run loop. The `rdeepseq`
ensures each branch's `Maybe WamState` is fully evaluated before the
merge step runs.

### 4.3 Work-estimation threshold

A branch is forked if:

```
estimated_work(branch) > FORK_THRESHOLD
```

For Phase 4 MVP, `FORK_THRESHOLD` is a runtime constant (e.g., "branches
must run for at least 100 microseconds"). Future iterations can derive
the estimate from C# demand analysis (which knows selectivity and
recursion depth bounds).

If estimate is below threshold, fall back to sequential `TryMeElse`
semantics — the parallel instructions degrade gracefully.

## 5. Merge strategies

The merge function depends on the enclosing aggregate context, known
at compile time:

|Aggregate context|Merge strategy|
|---|---|
|`aggregate_all(sum, ...)`|Sum partial sums from each branch|
|`aggregate_all(count, ...)`|Sum per-branch counts|
|`aggregate_all(bag, ...)`|Concatenate per-branch lists (order arbitrary)|
|`aggregate_all(set, ...)`|Union per-branch sets (dedup)|
|`findall(X, ..., L)`|Concatenate per-branch lists|
|`setof/bagof`|Concatenate, then sort/dedup at the boundary|
|Bare disjunction in body|First success (race), or all if non-deterministic|
|`\+ (negation)`|Any branch succeeds → negation fails (race-to-cancel)|

The compiler determines which strategy applies by walking the surrounding
aggregate frame from the choice point's position. This information is
already available in `wsAggAccum` / `cpAggFrame`.

### 5.1 Merge implementations

For sum/count: a simple fold over partial results.

For findall/bag: list concatenation. Order is non-deterministic across
parallel branches but Prolog's findall doesn't guarantee order.

For race: cancel still-running branches once one succeeds. This requires
`async`-based primitives (`Control.Concurrent.Async`) rather than plain
`par`; budget for this in Phase 4.2.

### 5.2 Negation as cancellation

`\+ Goal` succeeds iff `Goal` has no solutions. Under parallelism, all
branches of `Goal` must fail before `\+` succeeds. If any branch
succeeds, the rest can be cancelled — this is a race-to-cancel pattern.

Implementing cancellation cleanly in the `Strategies` model is
awkward; using `async`'s `race` is more natural. This is one of the
more involved pieces of Phase 4.

## 6. Cut interaction

Cut (`!/0`) under parallelism is the trickiest semantic question.

### 6.1 Within-branch cut

If `!` fires inside a parallel branch and the cut is local to that
branch (the cut barrier is within the branch's frame), behavior is
unchanged — the branch trims its own choice points and continues.

### 6.2 Cross-branch cut

If `!` fires inside a parallel branch and the cut barrier is *outside*
the branch (i.e., the cut would normally truncate the parallel choice
point itself), the semantics get complex. Standard interpretations:

- **Cancel siblings.** The cut commits to this branch; sibling branches
  are abandoned.
- **No-op for already-running branches.** Don't cancel; let them
  complete but discard their results.

The vision spec (§2.4) doesn't pin this down. **MVP picks "no
cross-branch cut"**: the parallelizable annotation must guarantee that
no cut in the predicate body crosses the parallel CP barrier. This is
checkable at compile time (cut barrier analysis is standard WAM stuff).

### 6.3 Cut in disjunction

```prolog
parallel_p(X) :- (clause_a(X) ; clause_b(X) ; clause_c(X)), !, post(X).
```

Here the `!` after the disjunction commits to the first solution. Under
parallelism, "first solution" doesn't have a stable meaning — so we
either:

1. Disallow this combination (compile-time error).
2. Use race semantics — first parallel branch to succeed wins, others
   cancelled.

MVP: option 1 (disallow). Race semantics can be added later if needed.

## 7. Trail discipline

The trail records variable bindings for backtracking. Across parallel
branches:

- Each branch sees a snapshot of bindings up to `cpBindings`.
- New bindings made by branch B go into B's local trail extension.
- On branch completion, B's trail is dropped (since the surrounding
  aggregate consumes B's *result*, not its bindings).

The key invariant: **no shared mutable bindings table**. Each branch's
`wsBindings` is its own `IM.IntMap Value` — pointer-shared at the
common prefix, diverging at branch-local additions.

This is exactly the "immutability is a strategic asset" property from
the philosophy doc.

## 8. Type-level changes

```haskell
-- WamTypes additions

data ChoicePoint = ChoicePoint
  { ...
  , cpForkable :: !Bool   -- emitted by ParTryMeElse, false otherwise
  , cpForkInfo :: !(Maybe ForkContext)
  }

data ForkContext = ForkContext
  { fcMergeStrategy :: !MergeStrategy  -- determined at compile time
  , fcWorkEstimate  :: !(Maybe Double) -- microseconds, if known
  }

data MergeStrategy
  = MergeSumInt | MergeSumDouble
  | MergeCount
  | MergeBag | MergeSet
  | MergeFindall
  | MergeRace
  | MergeNegation
```

### Why a separate `ForkContext`

A choice point can be forkable but not yet forked (e.g., if work
estimate is below threshold). Carrying the merge strategy in the CP
itself lets us defer the fork decision to runtime without losing the
context information.

## 9. What stays the same

- Sequential semantics are unchanged. `ParTryMeElse` falls back to
  `TryMeElse` semantics when work-estimate is below threshold or
  parallelism is disabled at runtime (`+RTS -N1`).
- The lowered-emitter path is unaffected. Lowered functions don't go
  through choice points; they're either fully native or call back into
  the WAM run loop for unsupported operations.
- The FFI kernels are unaffected. They run as native Haskell functions
  with no choice points to parallelize.

## 10. What's out of scope for Phase 4

- **Speculative execution.** We don't run branches that might be
  cancelled by an unfired cut.
- **Work-stealing scheduler.** GHC's spark pool already does this; we
  rely on it rather than rolling our own.
- **NUMA affinity.** Out of scope. Native Linux + `+RTS -qa` may help
  but is orthogonal.
- **Cross-target consistency.** Other WAM targets (Rust, LLVM, WAT)
  may eventually need their own parallelism stories. This spec is
  Haskell-only.

## 11. Open questions for implementation

1. **How do we annotate aggregate context in compiled code?** Currently
   `aggregate_all` is wrapped via `begin_aggregate`/`end_aggregate`
   instructions. The merge strategy might need to be carried in
   `BeginAggregate` so the choice points emitted within know which
   merge to use.

2. **Should the work-estimation threshold be per-predicate or global?**
   Global is simpler (one knob). Per-predicate is more accurate but
   needs C# integration.

3. **What happens if a parallel branch throws an exception?** Haskell
   `par` doesn't have exception semantics in the spark framework.
   Branches that error out should cause the whole query to fail,
   matching sequential semantics. `async` gives this for free if we
   use it.

These are flagged for resolution during Phase 4 implementation, not
in this spec.

## Related documents

- `docs/design/WAM_HASKELL_INTRA_QUERY_PHILOSOPHY.md` — motivation
- `docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md` — phasing
- `docs/vision/HASKELL_TARGET_PARALLELIZATION_SPEC.md` §2 — vision-level reference
- `docs/proposals/WAM_MULTI_OUTPUT_KERNELS.md` — sibling proposal pattern
