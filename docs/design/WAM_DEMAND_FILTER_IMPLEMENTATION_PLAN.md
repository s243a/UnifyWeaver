# Demand Filter: Implementation Plan

For *why*, see `WAM_DEMAND_FILTER_PHILOSOPHY.md`. For *what*, see
`WAM_DEMAND_FILTER_SPECIFICATION.md`. This document records the
ordered rollout.

This plan composes with the LMDB-resident interning rollout — see:

- `WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`
- `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`
- `WAM_LMDB_RESIDENT_INTERNING_IMPLEMENTATION_PLAN.md`

The two work streams interleave because the demand BFS walks edge
sub-dbs defined in the LMDB-resident plan. Phase 2 of the LMDB plan
is renamed Phase 2 here for symmetry.

## 0. Starting point

Current state on `main` after PR #1905:

- `data/benchmark/100k_cats/intmap/src/Main.hs` lines ~144-201
  build `demandSet` via plain reverse BFS over `parentsIndexInterned`,
  then derive `filteredSeedCats` via `IS.member`. The filter is
  always-on when demand filtering is selected; there is no user
  knob for filter strategy.
- `templates/targets/haskell_wam/main.hs.mustache` emits the
  `IntSet` build inline; no dispatch shape.
- Cache infrastructure (`lmdb_cache_mode = per_hec / sharded /
  two_level / auto`) is implemented in
  `templates/targets/haskell_wam/lmdb_fact_source.hs.mustache:218+`
  but not coupled to the demand BFS — it caches kernel queries
  reactively.
- LMDB-resident Phase 1 (PR #1905) shipped the producer side:
  ingester writes `s2i` / `i2s` / `meta` / `category_parent` /
  `article_category` sub-dbs.

The plan below adds dispatch on the runtime side, ships the simple
sound default, and stubs the more advanced strategies for follow-up
PRs.

## 1. Phase 2 — `HopLimit` filter + dispatch shape

**Branch:** `feat/wam-haskell-demand-filter-dispatch`

### 1.1 Codegen support

Edit `src/unifyweaver/targets/wam_haskell_target.pl`:

- Parse `:- declare_demand_filter(Strategy, Opts).` directives at
  predicate-declaration time. Default is
  `declare_demand_filter(hop_limit, [max_hops(MaxDepth)])` where
  `MaxDepth` comes from the existing `max_depth/1` resolution.
- Emit a `DemandFilterSpec` literal in the generated `Main.hs`.
- Validate Strategy ∈ {`hop_limit`, `flux`, `none`}; reject unknown
  with a clear error.
- For `flux`, emit a panic stub: the spec compiles but the runtime
  errors out at startup with "Flux strategy not yet implemented;
  see Phase 2.5". Phase 2.5 fills this in.

### 1.2 Runtime dispatch in `main.hs.mustache`

- Add the `DemandFilterSpec` data declaration to `WamTypes` (or
  inline in `Main.hs` — Phase 2 chooses inline for simplicity, can
  promote later).
- Replace the inline `demandSet` build with a call to
  `runDemandBFS spec ctx rootId`, returning `DemandFilterResult`.
- Implement `runDemandBFS` for `HopLimit` (depth-bounded reverse
  BFS, no priority queue).
- Implement `runDemandBFS` for `None` (returns `dfrInSet =
  IS.fromList (IM.keys parents)`, sorted seeds = `Nothing`).
- Implement `runDemandBFS` for `Flux` as `error "..."` panic stub.

### 1.3 LMDB-cursor BFS

When `int_atom_seeds(lmdb)` mode (LMDB-resident Phase 2) is active,
the BFS walks LMDB cursors. Otherwise, it walks the in-memory
`IntMap` exactly as today. Same `runDemandBFS` interface either way;
just a different `EdgeLookup` argument.

The reverse-edge handling defaults to "build `reverseAdj` in memory
at startup" (LMDB-resident plan §6, option 1). If a reverse sub-db
is later added to the ingester (LMDB-resident Phase 1.x follow-up),
`runDemandBFS` reads from it directly without any change here.

### 1.4 Cache warming hook

When an `lmdb_cache_mode` is configured, the BFS visits each edge
once and writes it through the cache write path. Behind a feature
flag `dfWarmCache :: Bool` on the Spec (default `True` when a cache
mode is set, `False` otherwise). Trivial implementation — just an
extra `IORef`/`IOArray` write per visited edge.

For `Flux`, warming order honours flux scores (highest first). For
`HopLimit` and `None`, warming order is the BFS order (no priority).

### 1.5 Tests

`tests/test_wam_haskell_target.pl` extensions:

- `test_demand_filter_hop_limit_emits_dispatch` — generated `Main.hs`
  contains `runDemandBFS (HopLimit { dfMaxHops = 10 })`.
- `test_demand_filter_none_skips_filter` — generated `Main.hs`
  contains `runDemandBFS None` and `dfrInSet = IS.fromList`-style
  expression for the universe.
- `test_demand_filter_flux_emits_panic_stub` — generated `Main.hs`
  contains the `error "Flux strategy not yet implemented"` panic.
- `test_demand_filter_directive_validation` — invalid strategy
  rejected with diagnostic.
- `test_demand_filter_max_hops_overrides_max_depth` — explicit
  `max_hops(N)` in the directive overrides kernel's `max_depth`.

End-to-end: scale-300 matrix bench should produce sha
`70bbc9ffa4cf` unchanged, since `HopLimit` with `max_hops = max_depth`
is equivalent to today's behaviour.

### 1.6 Acceptance

- Tests green.
- Scale-300 stdout sha matches.
- `100k_cats` default root: `demand_set_size` and
  `demand_skipped_seeds` numbers unchanged from PR #1905 baseline.
- Stderr contains the new `demand_filter:` diagnostic line.

## 2. Phase 2.5 — `Flux` filter

**Branch:** `feat/wam-haskell-demand-flux`

### 2.1 Priority-queue BFS

Replace the panic stub with a real implementation:

- Use `Data.Heap` (or `Data.Heap.Internal.MaxHeap`) keyed by flux
  descending, ties broken by node ID ascending (for determinism).
- Walk reverse edges; on each pop, record `flux(n)` and update
  `flux(child)` for each reverse-edge child.
- Stop when `IS.size dfrInSet >= dfCacheTopK` or queue is empty.
- Compute `dfrSortedSeeds` as `sortBy (Down . flux)
  filteredSeedCats` if `dfSortSparks=True`.

### 2.2 Codegen: target_fraction → target_count

When the directive specifies `target_fraction(f)`:

- Codegen emits an expression like `Flux { dfCacheTopK = ceiling
  (f * fromIntegral universeSize), dfSortSparks = True }`.
- `universeSize` resolves to `meta.next_id` (LMDB) or
  `IM.size parentsIndexInterned` (in-memory IntMap).

When the directive specifies `target_count(K)`, the literal K is
emitted directly.

### 2.3 Cabal dep

Conditional on any kernel emitting `Flux`, add `heap >= 1.0` to the
generated cabal file. Existing `lmdb >= 0.2.5` conditional logic
serves as the pattern.

### 2.4 Tests

- Unit test on a small synthetic graph: known flux values, verify
  `runDemandBFS Flux` returns expected top-K.
- Sorted-sparks test: `dfrSortedSeeds` is in flux-descending order.
- Determinism test: two runs of the BFS produce the same set and the
  same sort order.
- Cap behaviour: `dfCacheTopK = 0` returns empty set; `dfCacheTopK >
  reachable count` returns full reachable set.
- E2E: scale-300 with `flux` strategy produces the same sha as the
  same workload with `hop_limit` (only the order changes, not the
  result; the kernel still runs all reachable seeds).

### 2.5 Acceptance

- Phase 2 tests still green.
- New flux-specific tests green.
- `100k_cats` `Deaths_by_year` root: `query_ms` with `flux` ≤
  `query_ms` with `hop_limit` (priority order should help cache
  hit rate at the very least; we'd accept flat as proof of
  no-regression).

## 3. Phase 2.6 — `Hybrid` and adaptive

**Branch:** `feat/wam-haskell-demand-hybrid`

Optional. Only worth implementing if Phase 2.5 measurements show
neither pure strategy dominates.

- Add `Hybrid { dfHopLimit :: !Int, dfFluxTopK :: !Int }`
  constructor: BFS bounded by hops AND flux-priority.
- Adaptive variant: measure hit-rate during the first few queries
  and adjust the floor / target_fraction. Probably gated behind an
  env flag rather than always-on.

## 4. Phase 5 — cross-target reuse (joint with LMDB-resident plan)

The demand-filter dispatch is target-agnostic in its semantics. The
WAM-Rust and WAM-Elixir backends could adopt the same strategy
sum-type. The implementation is per-target (each backend has its own
runtime), but the codegen directive (`declare_demand_filter`) and
the validation logic in `wam_haskell_target.pl` can be lifted to a
shared module under `src/unifyweaver/core/`.

This phase is exploratory; deliverable is a memo on whether to
unify, plus issues for the rollout to other targets.

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| `Flux` BFS is slow on broad roots | The `dfCacheTopK` cap bounds the BFS; once the priority queue has emitted `dfCacheTopK` nodes, expansion stops. Worst case is O(K log K). |
| Priority queue cost dominates for small `K` | For `K < 10000`, a sorted-array-based priority queue is faster than a heap. Phase 2.5 ships the heap version; if measurements show cost matters, swap implementations. |
| `target_fraction(f)` underspecifies behaviour when `meta.next_id` is unset (small fixtures, in-memory path) | Fall back to `IM.size parentsIndexInterned` as the universe estimate. Document this in the spec; warn at codegen if neither is present. |
| Sorted parMap input changes parallelism characteristics | The matrix bench should detect any regression in `query_ms`. If sorted input degrades parallelism (e.g. all high-flux work piles on HEC 0), investigate work-stealing tuning before reverting. |
| `Hybrid` adds complexity without proven value | Don't ship until measurements show it's needed. Phase 2.5 is the gate. |
| Reverse-edge in-memory build is itself expensive | Tracked in LMDB-resident plan §6; reverse sub-db follow-up at the ingester layer if measured to matter. |

## 6. Verification

End-to-end at every phase boundary:

1. Phase 2: scale-300 sha unchanged; `100k_cats` numbers match
   PR #1905; `demand_filter:` stderr line emitted.
2. Phase 2.5: small-graph unit test for known flux values; scale-300
   sha unchanged across `hop_limit` / `flux` / `none` strategies.
3. Phase 2.6: only if measurements warrant.

If any phase regresses scale-300 sha `70bbc9ffa4cf`, that is the
phase to debug before moving on. The strategy is "kernel-result
identical, performance characteristics differ" — anything else is a
bug.

## 7. Out of scope

These appear in the philosophy doc as future ideas but are not in
this implementation arc:

- Pre-computed demand sets per known root, written to a sub-db at
  ingest time (filed as ingester follow-up).
- Information-theoretic floors (entropy-based cutoffs).
- Adaptive memory budget with frontier trimming.
- Cross-root flux LRU cache.
- Parallel BFS workers.

Each of these is additive — they fit the dispatch shape Phase 2
establishes. Filed for measurement-driven future work.
