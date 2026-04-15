# Haskell WAM Intra-Query Parallelism: Philosophy

> **Vision context:** This is the design-level companion to
> `docs/vision/HASKELL_TARGET_PARALLELIZATION_SPEC.md` §2. The vision
> doc establishes *what* we're aiming for; this doc explains *why now*,
> *when it pays off*, and *what we're willing to spend* to get it.

## What we already have

Seed-level parallelism (Phase 1, PR #1377) gave us a 3-3.5x speedup
across all tested scales. It exploits the fact that the
effective-distance benchmark queries 386 independent seed categories,
each with its own `WamState`, sharing only an immutable `WamContext`.

That's *coarse-grained* parallelism: one Haskell spark per query.
Each query then runs entirely sequentially.

## What intra-query parallelism would add

When a single query has internal parallelism — multiple choice point
branches that don't depend on each other — we currently explore them
serially via backtracking. Intra-query parallelism would fork those
branches across cores.

Concretely, for a recursive predicate like:

```prolog
ancestor(Cat, Mid, Hops, Visited) :-
    parent(Cat, Mid),                  % multiple solutions
    \+ member(Mid, Visited),
    ancestor(Mid, Root, H1, [Mid|Visited]),
    Hops is H1 + 1.
```

Each `Mid` solution spawns an independent recursive subquery. With N
parents, we could explore N branches concurrently rather than
sequentially.

## Why this isn't the right next step for our current benchmark

**The effective-distance benchmark wouldn't measurably benefit.**
Three reasons, in order of importance:

1. **The hot predicate (`category_ancestor/4`) is already FFI-handled.**
   It runs as a tight Haskell function, not through the WAM
   interpreter. There are no choice points to parallelize — the
   recursion is just a `concatMap` over the parents list inside
   `nativeKernel_category_ancestor`.

2. **Seed-level parallelism already saturates the available cores.**
   At 4 cores with 386 seeds, each core has ~96 seeds to chew through.
   Adding intra-seed forks would just oversubscribe.

3. **The remaining bottleneck is GC pressure, not sequential work.**
   The 10k-scale profile showed parallel GC work imbalance (70% utility
   on 4 cores). More forks would worsen this, not help.

So why write a design doc for it at all?

## Why intra-query parallelism is still strategically important

Three workload classes where it would matter:

### 1. Few-seed, deep-recursion queries

Imagine a query like "find all transitively-related papers from this
one citation, weighted by topic similarity." One seed (the citation),
deep recursion, lots of branching at each step. Seed-level parallelism
gives 1 spark — all cores idle except one. Intra-query parallelism
would spread the recursion across cores.

This is the **long-tail** of UnifyWeaver workloads — fewer seeds, more
work per seed. Our current benchmarks happen to be the opposite shape
(many seeds, shallow per-seed work).

### 2. Predicates that *aren't* FFI-handled

The kernel registry handles ~7 specific patterns. Anything outside
that — user-defined predicates, specialized analyses, novel graph
structures — runs through the WAM interpreter. For those, intra-query
parallelism is the only parallelism available short of writing a new
kernel.

This is the **generality** argument: the FFI fast path will always
cover a curated set of common patterns; the interpreter is the fallback
for everything else, and we'd like the fallback to scale too.

### 3. Validation of Haskell's parallelization advantage

The Rust WAM target uses mutable vectors that can't be trivially
forked across cores without explicit `Arc`/`Mutex`. Haskell's
immutable `WamState` makes intra-query forks free at the type-system
level — we just call `par`/`pseq` on alternative branches.

This is the **architectural** argument from the vision doc: we chose
immutability specifically to enable this kind of parallelism. Phase 1
demonstrated it for seed-level; Phase 4 would demonstrate it
*structurally*, where Rust would need significant refactoring to keep
up.

## What it costs

### Direct costs

- **Spark management overhead** (~microseconds per fork). For tiny
  branches, this dominates the actual work — we need a work-estimation
  threshold to skip parallelization for cheap branches.
- **Choice point reification.** Currently, choice points are stored as
  a stack on each `WamState`. To fork at a choice point, both branches
  need their own snapshot. With immutable state this is "free" (pointer
  copies), but the trail must be carefully tracked across branches so
  bindings made in one branch don't leak to the other.
- **Merge complexity.** When branches finish, their results must be
  combined according to the enclosing aggregate context (`sum`,
  `count`, `findall`, bare disjunction). Each merge strategy needs its
  own implementation.

### Indirect costs

- **Purity proofs.** A predicate is safe to parallelize only if it has
  no side effects (no `assert`/`retract`, no I/O, no global state,
  cuts only as scoped). The C# query engine has analyses that can
  produce these proofs; integrating them adds pipeline complexity.
- **Cut interaction.** `!/0` semantics under parallelism are subtle.
  If a cut fires in one branch, do already-running parallel branches
  get cancelled? The standard answer is "yes" but implementing
  cancellation in Haskell with `par` requires care (you'd want
  `async`-based primitives).
- **Test surface.** Every parallelizable predicate needs both
  sequential and parallel correctness tests. Comparing outputs across
  modes is essential — parallelism shouldn't change semantics, only
  scheduling.

## Why we'd do it anyway

The cost-benefit analysis above is workload-specific. For the
effective-distance benchmark, intra-query parallelism is a clear loss.
For graph algorithms with deep recursion or rich branching, it's the
only path to multi-core scaling.

The deciding factor is **what UnifyWeaver wants to be**:

- If it's an effective-distance benchmark optimizer: stop. We're done.
- If it's a general-purpose Datalog/Prolog compiler that wants to
  scale across workload shapes: intra-query parallelism is on the
  critical path.

The vision doc sides with the latter. This design doc series exists
to make Phase 4 implementable when that need becomes concrete.

## Non-goals

- **Auto-parallelize everything.** Forking has overhead. We need
  explicit annotations or proven purity, plus work-estimation
  thresholds.
- **Match Erlang/Cilk semantics.** Those languages are designed
  around fine-grained concurrency; we're bolting it onto a backtracking
  abstract machine. Our model is "fork at choice points, merge at
  aggregate boundaries" — much narrower than general task parallelism.
- **Replace seed-level parallelism.** Both can coexist. Seed-level is
  always faster when applicable (no merge overhead). Intra-query is
  what kicks in when seed-level can't.

## When to revisit

Implement intra-query parallelism when one of:

1. **A real workload benefits** — someone tries UnifyWeaver on a
   long-tail query and the pure interpreter is too slow.
2. **Parallelism becomes competitive ground** — Rust target adds
   intra-query parallelism (would be hard for them; that's our
   architectural moat to widen).
3. **Tooling reaches readiness** — C# query engine produces purity
   certificates we can consume; demand analysis is mature enough to
   inform the work-estimation threshold.

Until then: this design exists, the vision is consistent, and we're
not caught flat-footed when one of those triggers fires.

## Related documents

- `docs/vision/HASKELL_TARGET_PARALLELIZATION_SPEC.md` §2 — vision-level scope
- `docs/vision/HASKELL_TARGET_PHILOSOPHY.md` — broader strategic framing
- `docs/design/WAM_HASKELL_INTRA_QUERY_SPEC.md` — concrete mechanism (next)
- `docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md` — phased work breakdown
