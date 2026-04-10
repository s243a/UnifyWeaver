# WAM Haskell Performance: Philosophy

## Context

The Haskell WAM target was created as a fallback path for Prolog predicates
that resist native lowering. Once it was producing correct output (213/213
tuples, 272/272 line matches against the SWI-Prolog reference at the
effective-distance benchmark), the question shifted from "is it correct" to
"how close can it get to native SWI-Prolog?"

The Rust hybrid WAM had already proven that a WAM compiled to a strict,
pre-resolved instruction stream could *beat* native SWI-Prolog when the
foreign function interface (FFI) handled the recursive `category_ancestor/4`
search. The Haskell WAM, by contrast, started at ~12 seconds for a workload
that SWI-Prolog handles in ~318 ms — a ~37x gap.

Closing that gap is the subject of this branch
(`feat/wam-haskell-perf-profile`). This document explains *why* we picked the
optimizations we did, why we are *not* taking certain other paths that look
attractive on the surface, and how the remaining work fits together.

## Core principle: persistent data structures, not mutation

The Rust WAM was forced to clone the registers, bindings, and stack on every
choice point because Rust's ownership model doesn't give you cheap structural
sharing. The Haskell WAM started with the opposite stance: every WAM field
that backtracks is a *persistent* data structure (`Data.Map`, then
`Data.HashMap.Strict`, then `Data.IntMap.Strict`), so a choice point snapshot
is a constant number of pointer copies — the unmodified subtrees are shared
between the live state and every CP that holds a reference to them.

This is the single architectural reason Haskell can match Rust on
backtracking-heavy workloads at all. **Every optimization we keep must
preserve this property**, and every optimization we reject is rejected
primarily because it would break it.

## Why we are *not* using IORef (or STRef, or MutVar)

The most obvious "next step" for someone coming from a procedural background
is: stop allocating new `WamState` records on every step, mutate one in
place. Wrap the registers in an `IORef (IntMap Value)`, the trail in an
`IORef [TrailEntry]`, and so on. The whole loop runs in `IO`. Each step is
"cheaper" because nothing is allocated.

We rejected this for four reasons:

1. **It destroys the choice-point property.** A snapshot of an `IORef` is
   not O(1) — it is "save the *current* contents", which means reading them
   out and reinstalling them on backtrack. The persistent-data-structure
   trick where the live state and the saved CP literally share subtrees
   stops working. Backtracking gets *slower*, not faster.

2. **It pollutes everything with `IO`.** The `step` function becomes
   `WamState -> IO (Maybe ())`. Every helper that touches state has to live
   in `IO`. Pure helpers that GHC currently inlines and rearranges freely
   become opaque effectful blocks. The optimizer's job gets *harder*.

3. **The marginal wins are smaller than they look.** The thing IORef
   eliminates — record allocation — is exactly the thing GHC's strictness
   analyzer and worker/wrapper transform are good at compressing. A strict
   record with bang-patterned fields and `-O2` typically becomes register
   spills, not heap allocations. GHC will *not* lift mutation out of an
   IORef the same way.

4. **It is the wrong abstraction for the wrong layer.** The right place to
   eliminate "boxed Haskell record overhead" is to skip the WAM and emit
   straight-line Haskell from the Prolog source. That is a separate path
   (the native `haskell_target.pl`) and we are explicitly *not* trying to
   merge them.

The user-facing summary: IORef would turn the Haskell WAM into a worse
version of the Rust WAM, with all of Rust's clone-on-backtrack pain *and*
none of Haskell's structural-sharing wins.

## Why we *are* doing the WamState split next

`WamState` currently has 17 fields. Most of them change on every step
(`wsPC`, `wsRegs`, `wsBindings`, `wsTrail`, `wsTrailLen`, `wsBuilder`,
`wsVarCounter`). A handful of them never change after initialization
(`wsCode`, `wsLabels`, `wsForeignFacts`, `wsForeignConfig`).

When the step function emits `s { wsPC = pc', wsRegs = r' }`, GHC has to
allocate a fresh 17-field record because record-update syntax allocates a
new constructor with all fields copied. Eleven of those copies are pointless
— the cold fields are the same pointer they always were.

**Splitting** the cold fields out into a separate `WamContext` argument
that is threaded but never modified eliminates that copy on every step. The
record we *do* allocate becomes smaller, the constructor allocation is
cheaper, and the GHC worker/wrapper transform has fewer fields to track.
This is the same trick GHC's own RTS uses for `StgRegTable` vs heap.

The split is a refactor, not a redesign. It preserves persistence,
preserves purity, preserves the "step is `WamState -> Maybe WamState`"
shape (the only difference is that step now takes a `WamContext` as well).
It cannot regress correctness; it can only fail to be faster than expected.

We are doing it *first*, before the FFI work, because once the per-step
record allocation cost is gone the FFI wins become measurable. As long as
the per-step overhead is masking everything, an FFI improvement of "saves
500 ms in `nativeCategoryAncestor`" is invisible against the noise of
"every step allocates a 17-field record".

## Why FFI optimization is the *second* priority, not the first

The current FFI path (`nativeCategoryAncestor`) already beats the pure WAM
path on most runs by 10–25%. But it is doing more work than it has to:

- It allocates a fresh `[String]` visited list at each recursive call.
- It rebuilds the result list across recursive frames.
- It re-walks `wsForeignFacts` lookups that could be hoisted.
- It returns Hops as `[Int]` and lets the WAM enumerate them via
  `HopsRetry`, which is correct but means each Hops value pays for one
  full step round-trip back into the WAM dispatch.

Optimizing those is straightforward — *if* the WAM-side overhead has been
removed first. Otherwise the WAM dispatch noise hides the FFI improvements
in benchmark variance.

This is also why the user's intuition ("we should do the split first
because once we eliminate other overhead the FFI wins will be more clear")
is correct: profile-guided optimization is only useful when the profile
isn't dominated by something orthogonal.

## Why "compile to Haskell" is a separate path, not a continuation of this one

A natural temptation, having pushed the WAM-as-interpreter path, is to
try compiling each Prolog predicate directly to a Haskell function: skip
the instruction array, skip the dispatch loop, just emit
`categoryAncestor :: String -> String -> Int -> [String] -> [(Int, [String])]`.

This is a real and valuable target. It already exists, in fact:
`haskell_target.pl` is the native lowering path. The WAM target
(`wam_haskell_target.pl`) only takes over when native lowering fails — for
predicates with complex backtracking, cut interactions, meta-call, or other
features that don't have an obvious closed-form lowering.

We are explicitly *not* merging these. The reasons:

1. **They serve different use cases.** Native lowering is for predicates
   we can statically reason about. WAM-via-Haskell is for predicates we
   *can't*. Trying to make the WAM path do native-lowering work means
   re-deriving the entire native lowering analysis inside the WAM compiler.

2. **Each path's wins are different.** Native lowering wins by emitting
   straight-line code GHC can optimize as if a human wrote it. WAM-via-
   Haskell wins by being a faithful, predictable execution model that
   handles every Prolog feature correctly. Mixing the two means giving
   up the strengths of both.

3. **The "compile WAM to Haskell functions" approach is itself a separate
   exercise** — a third path, distinct from both. It would take WAM
   instructions and emit Haskell, which is a smaller specialization
   problem than full native lowering but a larger one than what we have
   now. It is worth doing eventually. It is not what this branch is about.

For the current branch, we keep the WAM-as-interpreter shape and push as
hard as we can on it. The compile-WAM-to-Haskell experiment can fork off
later as its own design doc.

## Why interning works (and where it stops working)

Several of the optimizations on this branch intern strings as `Int`s:
register names (`A1` → `1`, `X1` → `101`, `Y1` → `201`), variable names
(`_V42` → integer from `wsVarCounter`), and predicate names via
`CallResolved` (label string → target PC).

These work because the interned things are *closed* at compile time —
the WAM compiler knows every register name and every label name when it
emits the instruction list, so it can substitute the integer at generation
time and the runtime never has to hash a string.

The key reason this is safe: **we are interning bookkeeping names, not data
values**. The user's data (category names like `"Nuclear_physics"`) still
flows through `Map.HashMap String [String]` and gets hashed exactly once
per lookup, the way it should. Cycle detection (`elem` on a small visited
list, formerly `Set.member` on a `Set String`) still uses string equality
on data, the way it should.

The line we will not cross: interning *user data*. That would require a
global symbol table, would make string identity a global property, would
prevent on-the-fly fact loading, and would not actually help performance
because data hashing only happens at the boundary (loading facts,
key lookups), not in the inner loop.

## What "good enough" looks like

The success criterion is *not* "match SWI-Prolog at every scale". SWI-Prolog
has 30 years of internal optimization — argument indexing, mode analysis,
clause-level JIT-like first-argument indexing, an in-process garbage
collector tuned for backtracking. We will not catch up.

The success criterion is:

1. **The Haskell WAM is competitive enough for fallback use.** When a
   predicate can't go through native lowering, the user pays a small
   constant factor to use the WAM path, not a 30x penalty.

2. **The pure-WAM path is within 3x of native SWI-Prolog** at the
   effective-distance benchmark scale we use for regression testing.
   Currently we're at ~10x and dropping.

3. **The WAM+FFI path is within 2x** for the same workload, ideally faster
   than the WAM path by enough to justify keeping the FFI code.

4. **The optimization story is honest**: every claimed speedup is from a
   measured benchmark, not from a guess about what should be faster, and
   every optimization reflects the profiler's actual hot spots rather
   than micro-optimization theater.

## What we are *not* trying to do

- We are not going to add `unsafePerformIO` to the step function.
- We are not going to add `unsafe` array fetches anywhere except the
  one place (`unsafeFetchInstr`) where the invariant ("PC is always in
  range or zero, and zero is handled") is locally checkable.
- We are not going to introduce `IORef`, `STRef`, `MutableArray`, or any
  other mutable cell into the WAM state.
- We are not going to fork the codebase into "fast WAM" and "correct WAM"
  variants. There is one WAM, it is correct, and the optimizations are
  things that make it faster *without* changing its semantics.
- We are not going to gate optimizations on a profile flag. Either the
  optimization is a clear win on the benchmark and gets merged, or it
  isn't and gets reverted.

## Summary

| Idea | Status | Reason |
|---|---|---|
| Persistent maps for registers/bindings | Done (commits 0daa9d65, 221f828f, 803ed711) | The whole point of using Haskell |
| Compile-time arity for `PutStructure` | Done (501da4fc) | Profiler showed 8.3% allocation in arity parsing |
| Pre-resolve `Call` to `CallResolved` | Done (cec17ca3) | Profiler showed predicate-name hashing |
| Cached lengths (trail/heap/CPs) | Done (f757b881, bdae58cf) | `length` on linked list is O(n) |
| `INLINE` on hot helpers | Done (d0e639de) | Cheap, lets GHC specialize |
| WamState hot/cold split | **Next** | Per-step record allocation is the next biggest cost |
| FFI path optimization | After split | Wins are visible only after WAM overhead is gone |
| `IORef`-based mutation | **Rejected** | Destroys structural sharing |
| Compile WAM-to-Haskell-functions | Separate exercise | Different path, different design |
| Match SWI-Prolog perfectly | Not a goal | 30 years of head start; "competitive fallback" is the bar |
