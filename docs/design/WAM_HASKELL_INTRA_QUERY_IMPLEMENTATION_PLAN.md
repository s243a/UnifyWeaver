# Haskell WAM Intra-Query Parallelism: Implementation Plan

> **Phased work breakdown** for implementing intra-query parallelism per
> the spec in `WAM_HASKELL_INTRA_QUERY_SPEC.md`. Each phase produces
> something testable and is committable as its own PR. Skip ahead by
> phase when picking up this work.

## Overall shape

Six phases, ordered to minimize integration risk:

|Phase|Scope|Risk|Demonstrable outcome|
|---|---|---|---|
|4.0|Benchmark workload that *needs* intra-query parallelism|Low|Reproducible test case showing seed-level parallelism doesn't help|
|4.1|Instructions + compiler emission|Low|`ParTryMeElse` emitted for annotated predicates, sequential semantics preserved|
|4.2|Runtime fork on `ParTryMeElse` (sum/count merges only)|Med|First measurable speedup on Phase 4.0 benchmark|
|4.3|Findall/bag/set merge strategies|Med|Multi-merge support, more workloads benefit|
|4.4|Negation as race-to-cancel|High|Requires async primitives, careful exception handling|
|4.5|Work-estimation threshold + degradation|Low|Cheap branches no longer fork wastefully|
|4.6|C# purity certificate consumption (optional)|Med|Auto-detection of safe-to-parallelize predicates|

Each phase below details: what changes, where, how it's tested,
what's deferred.

---

## Phase 4.0: Build a benchmark that needs intra-query parallelism

Before changing any runtime code, we need a workload where intra-query
parallelism is the only available speedup vector. Without one, we can't
measure whether our work helped.

### Workload requirements

- Few sources (1-10), so seed-level parallelism gives ≤10 sparks.
- Deep recursion or rich branching per source.
- Pure (no I/O, no side effects), with a known correct answer.
- Reasonably realistic — not a synthetic benchmark with no analog.

### Candidate: graph reachability with pruning

A query like:

```prolog
:- parallel(reach/3).

reach(Source, Target, Path) :-
    edge(Source, Target),
    Path = [Source, Target].
reach(Source, Target, [Source|Rest]) :-
    edge(Source, Mid),
    reach(Mid, Target, Rest).
```

Run with `Source = a, Target = z` on a moderately branching graph.
One seed (Source), but many parallel exploration paths through the
recursion. Pure, deterministic answer (the set of all reachable paths).

### Deliverables

- New `examples/benchmark/intra_query_seed.pl` — workload definition
- New `examples/benchmark/generate_intra_query_benchmark.pl` — wraps
  the workload, generates a project that disables FFI (we need the
  WAM interpreter to be in the hot path for parallelism to help).
- Sequential timing baseline at small/medium scale.

### Validation

Confirms: with FFI disabled and seed-level parallelism enabled,
`+RTS -N4` does NOT speed up this workload (1 spark = 1 core's worth of
work). This is the gap intra-query parallelism would fill.

---

## Phase 4.1: Instruction emission

Add the new instructions and have the WAM compiler emit them for
annotated predicates. **No runtime behavior change yet** — they're
treated as their sequential equivalents at runtime.

### Changes

- `wam_target.pl`: add `par_try_me_else`, `par_retry_me_else`,
  `par_trust_me` to the instruction enum.
- WAM clause-compilation: when a predicate is annotated `:- parallel(P/N)`,
  emit `par_try_me_else` instead of `try_me_else` for the multi-clause
  indexing.
- `wam_haskell_target.pl` (`generate_wam_types_hs`): add
  `ParTryMeElse !String`, `ParRetryMeElse !String`, `ParTrustMe`,
  and the `Pc` variants to the `Instruction` data type.
- Step function: dispatch `ParTryMeElse` to the same handler as
  `TryMeElse` for now (no fork yet). Same for the others.
- `resolveCallInstrs`: pre-resolve labels in the `Par*` instructions
  the same way as the non-Par variants.

### Tests

- `test_wam_haskell_target.pl`:
  - Generated code for an annotated predicate contains
    `ParTryMeElse` (not `TryMeElse`).
  - Generated code with `+RTS -N1` produces identical results to the
    non-annotated baseline (semantic equivalence under serial execution).

### Deferred

- Actual forking. That's Phase 4.2.

---

## Phase 4.2: Runtime fork on ParTryMeElse (sum/count merges)

Wire the runtime fork mechanism. Start with the simplest merge
strategies — sum and count — because they're commutative and
associative, no ordering concerns.

### Changes

- `WamTypes`: add `cpForkable`, `cpForkInfo`, `MergeStrategy`,
  `ForkContext`.
- Step function: when `ParTryMeElse` is encountered AND the surrounding
  aggregate is `sum` or `count`, build the branches list, snapshot the
  state, fork via `parMap rdeepseq`.
- Merge: fold partial results.
- `BeginAggregate` instruction needs to carry the `MergeStrategy` so
  the inner choice points can find it. Add a field.

### Tests

- Sum benchmark from Phase 4.0: `+RTS -N4` gives measurable speedup vs
  `-N1`.
- Result comparison: parallel sum equals sequential sum (modulo
  floating-point associativity for Double).
- Existing tests (effective_distance) unchanged — they use FFI, no
  `Par` instructions emitted.

### Deferred

- Findall/bag/set merges (Phase 4.3).
- Race/negation (Phase 4.4).
- Work-estimation threshold (Phase 4.5) — every parallel CP forks for now.

---

## Phase 4.3: Findall/bag/set merge strategies

Extend the merge step to support list-collecting aggregates.

### Changes

- `MergeFindall`: concatenate per-branch result lists. Order is
  non-deterministic across forks but Prolog's findall doesn't
  guarantee order.
- `MergeBag` / `MergeSet`: same as findall, with optional
  deduplication for set.
- Aggregate-frame interaction: each parallel branch builds its own
  `wsAggAccum`; the merge concatenates them.

### Tests

- Findall benchmark: parallel findall returns the same set of
  solutions as sequential (after sorting).
- Set/bag benchmarks: similar correctness comparison.

### Deferred

- `setof`/`bagof` — these care about ordering and grouping. Non-trivial
  semantics under parallelism. Punt.

---

## Phase 4.4: Negation as race-to-cancel

`\+ Goal` succeeds iff `Goal` has no solutions. Under parallelism,
all branches of `Goal` must fail. If any succeeds, the rest can be
cancelled.

### Why this is high-risk

- Plain `par` has no cancellation. Need `Control.Concurrent.Async`.
- Switching from `Strategies` to `async` for parallel branches is a
  bigger refactor than it sounds — different exception handling,
  different scheduling, different interop with GHC's spark pool.
- Branch cancellation must clean up trail entries cleanly. Half-baked
  bindings could corrupt the parent state.

### Approach

- Replace `parMap rdeepseq` with `Async.race` / `Async.cancel` for
  branches under negation.
- Wrap branch execution in `withAsync` so cleanup is guaranteed even
  if the branch fails.
- Test extensively: this is the most likely source of subtle bugs.

### Tests

- `\+ p(X)` over a parallelizable `p/1` returns the same answer as
  sequential.
- Cancellation actually fires (instrument with a counter; verify
  branches don't all run to completion when one succeeds).

### Deferred

- Cross-branch cut (the spec defers this entirely; not implementing
  in Phase 4 at all).

---

## Phase 4.5: Work-estimation threshold

Right now, every `ParTryMeElse` forks, regardless of branch size.
Tiny branches lose money on spark overhead. Add a threshold.

### Changes

- Default `FORK_THRESHOLD` constant (e.g., 100 microseconds) in
  WamRuntime.
- `ForkContext` carries an estimated work value (default = `Nothing` =
  always fork).
- At the fork point, if `fcWorkEstimate < FORK_THRESHOLD`, fall back
  to sequential `TryMeElse` semantics.

### How estimates are derived

For Phase 4.5 MVP: estimates come from a static analysis pass that
counts instructions per branch. Crude but local — no C# integration.

### Tests

- Tiny-branch benchmark: forking is suppressed, performance equals
  sequential.
- Large-branch benchmark: forking still happens, performance matches
  Phase 4.2 numbers.

### Deferred

- C#-driven estimates (Phase 4.6).

---

## Phase 4.6: C# purity certificate consumption (optional)

If the C# query engine produces purity certificates as part of its
analysis pipeline, the WAM compiler can emit `Par*` instructions
automatically — no user annotation required.

### Why this is "optional"

It depends on the C# pipeline maturing in a particular direction. If
it doesn't, manual `:- parallel/1` annotation is fine. The spec is
already designed to support both modes.

### Changes

- Define a serialization format for purity certificates (probably JSON
  or Prolog terms).
- WAM compiler reads certificates and treats certified predicates as
  if they had `:- parallel/N` annotation.
- Optionally, certificates also include selectivity / depth bounds for
  better work estimates.

### Tests

- Parsing: malformed certificates are rejected.
- End-to-end: a predicate with a certificate gets parallel emission
  even without explicit annotation.

### Deferred

- Anything that requires changes to the C# engine itself.

---

## Cross-cutting concerns

### Test infrastructure

Each phase needs:
- A Prolog test predicate exercising the new behavior
- A Haskell test verifying generated-code shape (look for
  `ParTryMeElse` in the output, etc.)
- A benchmark with sequential vs parallel timing comparison
- Output equivalence test (parallel result == sequential result, modulo
  floating-point associativity)

Existing test files to extend:
- `tests/test_wam_haskell_target.pl` — code generation tests
- `tests/test_wam_haskell_lowered_phase1.pl` etc. — pattern reusable
  for `intra_query_phase4` series

### Documentation

Each phase merge should update:
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` with new numbers
- `docs/vision/HASKELL_TARGET_ROADMAP.md` Phase 4 status (in-progress
  sub-phases)

### Backward compatibility

All existing tests + benchmarks should pass through every phase
without modification. The `Par*` instructions are *additions*, not
replacements. Predicates without `:- parallel/N` annotation continue
to use the sequential `TryMeElse` family.

---

## When to start

Start Phase 4.0 (build a benchmark that needs this) when one of:

1. A user reports that a non-FFI workload is slow and would benefit.
2. Rust target adds intra-query parallelism (don't lose the moat).
3. C# purity certificates become available.

Until then: keep this design shelved. The vision is documented and
the path is clear; we're not blocked, just waiting for the right
trigger to spend the implementation budget.

## Related documents

- `docs/design/WAM_HASKELL_INTRA_QUERY_PHILOSOPHY.md` — motivation
- `docs/design/WAM_HASKELL_INTRA_QUERY_SPEC.md` — mechanism
- `docs/vision/HASKELL_TARGET_PARALLELIZATION_SPEC.md` §2 — vision-level reference
- `docs/vision/HASKELL_TARGET_ROADMAP.md` Phase 4 — current status (this Phase 4)
