# Demand Filter: Specification

The technical shape of the demand-filter dispatch we're implementing.
For *why* each decision, see `WAM_DEMAND_FILTER_PHILOSOPHY.md`. For
the rollout sequence, see `WAM_DEMAND_FILTER_IMPLEMENTATION_PLAN.md`.

This design composes with the LMDB-resident interning work — see:

- `WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`
- `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`
- `WAM_LMDB_RESIDENT_INTERNING_IMPLEMENTATION_PLAN.md`

The demand BFS walks edge sub-dbs defined there; the filter strategies
defined here are independent of the storage layer (they work the same
way for in-memory `IntMap` edges or LMDB-cursor edges).

## 1. The DemandFilterSpec sum type

Carried from the codegen layer to the runtime; selected at compile
time per-predicate.

```haskell
data DemandFilterSpec
  -- Provably complete filter using the kernel's natural depth.
  = HopLimit
      { dfMaxHops :: !Int
      }
  -- Flux-weighted top-K with optional spark ordering.
  | Flux
      { dfCacheTopK   :: !Int   -- nodes to retain in cache (RAM-bounded)
      , dfSortSparks  :: !Bool  -- order parMap input by flux descending
      }
  -- No filter; rely on cache + kernel's own pruning.
  | None
  deriving (Eq, Show)
```

Three constructors cover the regimes the philosophy doc identifies.
The runtime dispatches once at startup (`runDemandBFS spec ctx
rootId`) and produces an `IntSet` of in-set node IDs, plus optionally
a sorted seed list for the spark stage.

## 2. Runtime contract

Every strategy honours the same shape:

```haskell
-- The output of a demand-filter run.
data DemandFilterResult = DemandFilterResult
  { dfrInSet        :: !IS.IntSet
    -- Nodes considered "in." Used for seed pre-filter and as the
    -- universe for cache warming.
  , dfrSortedSeeds  :: !(Maybe [Int])
    -- Seeds in descending priority order. Nothing for HopLimit/None.
  , dfrFluxScores   :: !(Maybe (IM.IntMap Double))
    -- Per-node flux. Populated by Flux only; consumed by cache
    -- warming when configured.
  }
```

The runtime then:

1. Filters seed list: `seedCats' = filter (`IS.member` dfrInSet) seedCats`.
2. If `dfrSortedSeeds` is `Just sorted`, replaces `seedCats'` with
   `take (length seedCats') sorted` (preserving sort order).
3. If a cache mode is active, warms the cache with edges to nodes in
   `dfrInSet` (highest-flux first when scores are present).
4. Hands `seedCats'` to `parMap`.

The seed pre-filter (step 1) is **always applied** when a filter is
declared — that's the spark-fanout fix from PR #1882, and it's
load-bearing regardless of strategy.

## 3. Strategy semantics

### 3.1 HopLimit

```
demand set = { n : reverse-BFS distance from root to n ≤ dfMaxHops }
sorted seeds = Nothing
flux scores = Nothing
```

Plain reverse BFS bounded by depth. Sound (no false negatives) when
`dfMaxHops` ≥ kernel's `max_depth`. Default `dfMaxHops = max_depth`
unless the user overrides.

### 3.2 Flux

```
1. Reverse-BFS from root with priority queue keyed by flux descending.
2. flux(root) = 1; flux(n) = sum over reverse-edges (n, parent) of
   flux(parent) / out_degree(parent).
3. Stop when |inSet| ≥ dfCacheTopK or queue exhausts (subtree smaller
   than target).
4. The flux of the last popped node is the implicit floor — emitted
   to stderr for diagnostics.
```

`dfSortSparks=True` returns the surviving seed list sorted by flux
descending; `False` skips the sort cost when the deployment doesn't
care about spark ordering.

A seed with `flux = 0` is by definition unreachable from root (in
reverse-BFS terms), so it's always excluded — that's the sound part.
A seed with `flux > 0` but rank > `dfCacheTopK` is excluded as an
**approximation**: in real workloads it has a path to root but the
path is "too diluted" by routing density to be worth scheduling.

### 3.3 None

```
demand set = full universe of node IDs (all keys in the edge sub-db)
sorted seeds = Nothing
flux scores = Nothing
```

The seed pre-filter degenerates to "always pass." All seeds get
sparked. Useful when:

- The cache mode is sufficient to bound work without filtering.
- Debugging / measuring filter overhead (vs. running with no filter).
- Small fixtures where filter setup costs more than the kernel work.

## 4. Codegen surface

The Prolog codegen emits a directive per kernel that uses demand
filtering:

```prolog
%% Default — sound, kernel-aligned:
:- declare_demand_filter(hop_limit, [max_hops(10)]).

%% Flux-weighted with sensible defaults (Phase 2.5):
:- declare_demand_filter(flux, [target_fraction(0.05), sort_sparks(true)]).

%% Flux with absolute cap:
:- declare_demand_filter(flux, [target_count(50000), sort_sparks(true)]).

%% No filter — cache-only mode:
:- declare_demand_filter(none).
```

The codegen reads this directive and emits a `DemandFilterSpec`
literal in `Main.hs`. Per-kernel option, not per-target option.

`target_fraction(f)` is resolved to `target_count = ceil(f *
meta.next_id)` at runtime, after the LMDB env is opened (see
`WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md` §1.1 for the `next_id`
meta key). For the in-memory IntMap path, `next_id` is replaced by
`IM.size parentsIndexInterned`.

## 5. Cache integration

The demand filter and the cache layer (`lmdb_cache_mode`, see
`templates/targets/haskell_wam/lmdb_fact_source.hs.mustache:218+`)
compose:

| Filter | Cache mode | Behaviour |
|---|---|---|
| `None` | `unset` | All seeds spark; LMDB cursors for every lookup. |
| `None` | `per_hec` / `sharded` / `two_level` | All seeds spark; cache absorbs hot lookups; cold falls through to LMDB. |
| `HopLimit` / `Flux` | `unset` | Seed pre-filter applied; LMDB cursors directly. |
| `HopLimit` / `Flux` | `per_hec` / `sharded` / `two_level` | Seed pre-filter applied; **cache warmed during BFS** with edges in `dfrInSet`. Kernel hits cache for in-set, cache misses fall through to LMDB. |

Cache warming during BFS is one extra `IORef` write per visited edge
— effectively free. With `Flux`, the warming order honours
`dfrFluxScores`: highest-flux edges go in first so eviction (when
the cache is smaller than the demand set) drops lowest-flux first.

## 6. Cabal/feature dependencies

No new Haskell packages required for `HopLimit` (uses only
`containers` for `IntSet` / `IntMap`).

`Flux` requires `Data.Heap` (or equivalent priority queue) for the
BFS frontier. Adds `heap >= 1.0` as a conditional dep when any
generated kernel emits a `Flux` filter.

## 7. Error handling and degenerate cases

| Condition | Behaviour |
|---|---|
| `HopLimit` with `dfMaxHops <= 0` | Demand set is `{rootId}` only; spark filter excludes everything. Logged as warning. |
| `Flux` with `dfCacheTopK = 0` | Demand set is empty; all seeds excluded. Logged as warning. |
| `Flux` with `dfCacheTopK > N` | All reachable nodes included; equivalent to unbounded reverse BFS. |
| Root not in LMDB / `IntMap` | Demand set is empty; warning emitted; all seeds excluded. |
| Edge sub-db empty | Demand set is `{rootId}` only. |

Warnings go to `stderr` via `hPutStrLn stderr "demand_filter: ..."`,
matching the existing `demand_set_size=` etc. convention.

## 8. Determinism

- `HopLimit` is fully deterministic given root and max_hops.
- `Flux` is deterministic given the input edges (priority queue ties
  broken by node ID ascending). Same LMDB → same demand set → same
  output sha.
- `None` is deterministic.

This matches the project-wide invariant that the same fixture should
produce the same `stdout_sha256` regardless of how many cores or
which strategy variant is selected.

## 9. Diagnostics

Every strategy emits a one-line summary on stderr after BFS:

```
demand_filter: strategy=hop_limit max_hops=10 in_set=1151 universe=84136
demand_filter: strategy=flux cache_top_k=4207 in_set=4207 implicit_floor=2.4e-5
demand_filter: strategy=none in_set=84136 universe=84136
```

The matrix bench harness already grep's `demand_set_size=` and
`demand_skipped_seeds=`; the new line follows the same pattern so
existing tooling sees both keys.
