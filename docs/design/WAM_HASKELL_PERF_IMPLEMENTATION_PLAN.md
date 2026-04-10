# WAM Haskell Performance: Implementation Plan

This document records the optimization history of the
`feat/wam-haskell-perf-profile` branch and the ordered next steps.

For the *why*, see `WAM_HASKELL_PERF_PHILOSOPHY.md`. For the *what*, see
`WAM_HASKELL_PERF_SPECIFICATION.md`.

## 0. Starting point

Before this branch, the Haskell WAM target produced correct output (213/213
tuples, 272/272 line matches against the SWI-Prolog reference) but ran the
effective-distance benchmark in ~11.7 s wall time at 300 scale (depth=10,
386 seeds, 11172 paths).

For comparison at the same scale:
- Native SWI-Prolog: ~318 ms
- Rust WAM (with FFI): faster than native SWI-Prolog
- Rust WAM (pure interpreter): much slower than native SWI-Prolog

The gap to native SWI-Prolog was ~37x. The goal of this branch is to
close that gap as far as is practical without changing the architectural
shape of the WAM target.

## 1. Optimization history (already on the branch)

Each commit was profile-guided: run `+RTS -p -RTS`, find the hot spot,
attack it, re-profile, repeat.

### 1.1 `0daa9d65` — Data.HashMap.Strict for register/binding maps

**Hot spot:** ~17 GB allocated in 10 s of wall time. The profiler showed
the cost was in `Data.Map.Strict.insert` rebuilding tree paths.

**Change:** Added a `use_hashmap` option (default `true`) to
`write_wam_haskell_project`. The post-processing pass `apply_hashmap_rewrite`
rewrites `Data.Map.Strict` → `Data.HashMap.Strict` and `Map.Map` →
`Map.HashMap` in the generated Haskell. Cabal file conditionally adds
`unordered-containers` and `hashable`. `Value` got a `Hashable` instance
via `DeriveGeneric`.

**Result:** 11,742 ms → 8,615 ms (27% faster).

### 1.2 `501da4fc` — Pre-parse PutStructure arity at compile time

**Hot spot:** `step.arity` was 5.9% time / 8.3% allocation. The runtime
was parsing `"name/N"` on every `PutStructure` invocation via reverse +
break + reads.

**Change:** Added an `Int` arity field to the `PutStructure` ADT. The
WAM compiler already knows the arity when it emits the instruction, so
it embeds the integer directly. Runtime dropped `parseFunctorArity`.

**Result:** 8,615 ms → 7,320 ms (cumulative 38% vs Map baseline).

### 1.3 `221f828f` — IntMap for registers, eliminating string hashing

**Hot spot:** `hashWithSalt1` at 11.2% of runtime, mostly from register
name lookups (`Map.lookup "A1"`, `"X5"`, `"Y2"`, etc.).

**Change:** Register names became `Int`s under a fixed encoding:
- `A1`–`A99`: 1–99
- `X1`–`X99`: 101–199
- `Y1`–`Y99`: 201–299

The compile-time helper `reg_name_to_int/2` parses register names once
when emitting the Haskell instruction list. The runtime uses
`Data.IntMap.Strict` for `wsRegs`. `getReg`/`putReg` detect the Y bank
via `rid >= 200`.

**Result:** Significant drop in `hashWithSalt1` cost (visible in next
profile pass).

### 1.4 `803ed711` — Intern variables as Ints, IntMap for bindings

**Hot spot:** Even after register interning, `hashWithSalt1` was still
~10% of runtime. Variable names like `"_V42"` were the remaining string
hash work.

**Change:** `Unbound String` became `Unbound !Int`, drawing IDs from
`wsVarCounter`. `wsBindings` and `cpBindings` became `IntMap Value`.
`TrailEntry` became `TrailEntry !Int !(Maybe Value)`. The
`__binding__`++name string was retired. `BuiltinState` (FactRetry,
HopsRetry) variables became `Int`.

**Note:** Data-value hashing (`wsForeignFacts`, `wsLabels`) and cycle
detection (`Set String` for visited categories) were unchanged. Only
WAM-internal bookkeeping names were interned.

**Result:** `hashWithSalt1` mostly eliminated for inner-loop work.

### 1.5 `cec17ca3` — Pre-resolve Call instructions at project load

**Hot spot:** `Call` dispatch was hashing predicate name strings on
every invocation (`Map.lookup pred wsLabels`).

**Change:** Added `CallResolved !Int !Int` (target PC, arity). At
project load, `resolveCallInstrs` walks the code list and replaces every
`Call` whose target is a known label with `CallResolved`. Foreign
predicates (e.g., `category_ancestor/4` → `executeForeign`) and indexed
facts (`category_parent/2` → `callIndexedFact2`) keep the original `Call`
because they need runtime dispatch.

**Result:** `hashWithSalt1` dropped from 9.2% to 7.4% of runtime.
Benchmark: ~5,500 ms → ~4,000–5,100 ms.

### 1.6 `0d9695a9` — List-based visited + unsafe instruction fetch

Two small wins from the same profile pass:

**Hot spot 1:** `executeForeign.visitedStrs` was 4.7% — building a
`Set String` per recursive call into `nativeCategoryAncestor`.
**Change:** Use a plain `[String]` for visited. With `max_depth ≤ 10`,
`elem` on a 10-element list beats `Set.fromList + Set.member` overhead.

**Hot spot 2:** `fetchInstr` had a `Maybe` wrapper that allocated 3.8%
of total time and 10.5% of allocation.
**Change:** `unsafeFetchInstr` uses `code ! pc` directly. The run loop
already handles `PC = 0` as halt, and a well-formed WAM program never
jumps out of bounds.

**Result:** Profile total time dropped 14.5 s → 12.9 s. Benchmark:
~3,000–4,200 ms.

### 1.7 `f757b881` — Cache wsTrailLen/wsHeapLen, eliminate length calls

**Hot spot:** Multiple step paths called `length(wsTrail)` and
`length(wsHeap)` for choice point creation, which is O(n) on linked lists.
Trails grew to 100s of entries during deep recursion.

**Change:** Added `wsTrailLen` and `wsHeapLen` fields. Step functions
that cons to `wsTrail` increment the length; backtrack restores from
the CP; aggregate frame finalization recomputes correctly.

**Bonus fix:** `finalizeAggregate` had a latent bug where
`take (cpTrailLen cp) (wsTrail s)` was keeping the *newest* entries
instead of restoring to the snapshot. Fixed to use `drop (wsTrailLen s -
cpTrailLen cp) (wsTrail s)`. Hadn't caused observable issues because
aggregate finalization is the end of the WAM run, but worth fixing.

**Result:** Variance tightened. Benchmark: 3,100–4,100 ms.

### 1.8 `bdae58cf` — Cache wsCPsLen, fix addToBuilder O(n²) append

Two more list-length wins:

**Hot spot 1:** `Allocate` set `cutBar = length(wsCPs)` on every clause
entry. With deep recursion this added up.
**Change:** `wsCPsLen` field cached the length. Updated by `TryMeElse`,
`BeginAggregate`, `callIndexedFact2`, `executeForeign` (increment),
`TrustMe`, backtrack pop, cut, finalizeAggregate (decrement / set).
`!/0` cut also tracks via `wsCPsLen = wsCutBar`.

**Hot spot 2:** `addToBuilder` used `args ++ [val]`, O(n) per call,
O(n²) total for an arity-N structure.
**Change:** Use cons (`val : args`) and reverse on finalize.

**Result:** Benchmark: 3,000–4,100 ms (slightly tighter range).

### 1.9 `d0e639de` — INLINE pragmas on getReg, putReg, derefVar

**Hot spot:** Three helpers called many times per WAM step, no INLINE
pragma so GHC was conservative about call-site specialization.

**Change:** Added `{-# INLINE getReg #-}`, `{-# INLINE putReg #-}`,
`{-# INLINE derefVar #-}`.

**Result:** Small wins (~5% high variance), free.

### 1.10 Cumulative result

| Stage | Wall time at 300 scale (depth=10) |
|---|---|
| Pre-branch (Map baseline) | ~11,742 ms |
| HashMap | ~8,615 ms |
| HashMap + arity | ~7,320 ms |
| IntMap registers | ~6,000 ms |
| IntMap bindings + variable interning | ~5,000 ms |
| CallResolved | ~4,500 ms |
| List-visited + unsafeFetch | ~3,500 ms |
| Cached lengths + INLINE | ~3,100–4,100 ms (variance ~3,400 ms typical) |
| **Native SWI-Prolog reference** | **~318 ms** |

WAM-only path: ~3.4 s typical. WAM+FFI path: ~3.4 s typical (FFI is
slightly faster but within noise; the WAM-side overhead is now the
dominant cost, which is the next thing to attack).

Gap to native SWI-Prolog: ~37x → ~10x. Not at the "competitive" goal
yet, but closing.

## 2. Next: WamState hot/cold split

See `WAM_HASKELL_PERF_SPECIFICATION.md` §2 for the data-shape spec. This
section is the step-by-step plan.

### 2.1 Phase 1: Add WamContext type, update WamTypes.hs emission

- [ ] Add `WamContext` data declaration to the generator's
      `WamTypes.hs` template.
- [ ] Move `wsCode`, `wsLabels`, `wsForeignFacts`, `wsForeignConfig`
      out of `WamState` and into `WamContext`.
- [ ] Update `emptyState` to take a `WamContext` and produce a
      `WamState` with the cold fields gone.
- [ ] Add `mkContext :: ... -> WamContext` for the benchmark driver.

**Verify:** `cabal build` of a generated project still succeeds (it will
not yet because `step` isn't updated, that's the next phase — but
incremental compile should at least show the new types are well-formed).

### 2.2 Phase 2: Thread context through step

- [ ] Change `step :: WamState -> Maybe WamState` to
      `step :: WamContext -> WamState -> Maybe WamState`.
- [ ] Add `!ctx !s` bang patterns.
- [ ] In every `case` branch that currently reads `wsCode s`, `wsLabels s`,
      `wsForeignFacts s`, or `wsForeignConfig s`, change to read from
      `ctx`.
- [ ] Update `executeForeign` signature to take the context.
- [ ] Update `callIndexedFact2` signature to take the context.

**Verify:** `cabal build` succeeds. Type errors here are localized
because the field accessors are differently typed.

### 2.3 Phase 3: Update run loop and Main.hs

- [ ] Change `runLoop :: WamState -> Maybe WamState` to
      `runLoop :: WamContext -> WamState -> Maybe WamState`.
- [ ] Update `backtrack` if any of its subcases now need the context
      (probably none, but verify).
- [ ] In `Main.hs`, build `ctx` once before the query loop, pass it
      into every `runLoop` call.

**Verify:** `cabal run` produces the same 213 tuples. Same SHA256 of
output if you have it.

### 2.4 Phase 4: Benchmark and profile

- [ ] Run the benchmark 5 times, record min/median/max.
- [ ] Compare against the pre-split baseline (latest commit on
      `feat/wam-haskell-perf-profile`).
- [ ] Run with `+RTS -p -RTS` and look at the new top hot spots.
- [ ] If a clear new hot spot appears, decide whether to attack it
      before moving on to FFI work, or to defer.

**Acceptance criteria:**
- 213/213 tuples produced (no regression in correctness).
- Wall time at 300 scale (depth=10) is ≤ pre-split median.
- New profile shows the cold-field record-update cost has been
  eliminated (not "reduced" — it should be gone).

### 2.5 Risks

| Risk | Mitigation |
|---|---|
| Lazy `ctx` reads (no bang) → silent no-op | Audit every `step`/`runLoop` for `!ctx` |
| Forgetting a generator-side reference to a moved field | `cabal build` will catch it |
| Unexpected cabal config issue with cached builds | `cabal clean` between baseline and split runs |
| The split is a no-op because the per-step cost was already small | Acceptable; move on to FFI work |

## 3. After the split: FFI optimization

See `WAM_HASKELL_PERF_SPECIFICATION.md` §3 for the data-shape changes.

### 3.1 Steps (in order)

- [ ] Convert `nativeCategoryAncestor` to a difference-list (`[Int] ->
      [Int]`) accumulator. Target: eliminate the `baseHits ++ recHits`
      append.
- [ ] Profile to confirm the difference-list version is faster (or
      neutral). If neutral, keep it because it reads more cleanly.
- [ ] Audit the WAM round-trip cost via `HopsRetry`. If profile shows
      the round-trip is dominant, evaluate the direct-aggregate path
      (`nativeCategoryAncestorSum`) as a separate change.
- [ ] If the direct-aggregate path is taken, gate it on the aggregate
      shape: only apply when the WAM is summing `H ** NegN` for some
      `NegN`, fall back to the general path otherwise.

### 3.2 Acceptance criteria

- 213/213 tuples (no regression).
- WAM+FFI path is measurably faster than the pure-WAM path (it
  currently isn't — they're within noise).
- Profile no longer shows the FFI path as a hot spot.

## 4. Deferred: compile WAM-to-Haskell-functions

See `WAM_HASKELL_PERF_SPECIFICATION.md` §4 for the concept.

This is its own branch and its own design doc. The expected order is:

1. **Land this branch** (perf optimizations + design docs). PR pending.
2. **Land the WamState split** (separate branch off main).
3. **Land the FFI optimization** (separate branch off main).
4. **Open a discussion / write a proposal** for the compile-to-Haskell
   path. This deserves its own `WAM_HASKELL_FUNCTIONS_*.md` triple of
   docs because the design space is genuinely different.

## 5. Test strategy

Throughout all phases, the regression test is the same: the
effective-distance benchmark at 300 scale (depth=10, 386 seeds, 11172
paths) must produce 213 tuples that match the SWI-Prolog reference
output exactly.

The benchmark driver is generated by the `wam_haskell_target.pl` test
helpers. The reference output is the one used in the existing
`feat/benchmark-semantic-distance-at-scale` test path.

Regression test command (from the generated project root):
```
cabal run -- 300 10
```

Output should match `examples/benchmark/effective_distance/reference_300.tsv`
(or whatever the canonical reference is on `main`).

## 6. Commit message conventions

The branch convention so far is:
```
perf(wam-haskell): <one-line summary>

<2-3 paragraphs of context: hot spot, change, rationale>

Profile data and benchmark numbers if relevant.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Continue this for the WamState split and FFI work.

## 7. Out of scope

Anything in `WAM_HASKELL_PERF_SPECIFICATION.md` §5 ("Out of scope for
this branch") applies here too. Notably:

- No `IORef`/`STRef`/mutable arrays.
- No `unsafePerformIO`.
- No new instruction set.
- No changes to native lowering or to the Rust target.
