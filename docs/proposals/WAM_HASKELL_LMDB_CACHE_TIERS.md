# WAM Haskell LMDB Cache Tiers

**Status:** Phases 1 + 2 shipped (PR #1640, PR #1641); Phase 3 pending
**Author:** John William Creighton (@s243a)
**Date:** 2026-04-25 (proposed); updated 2026-04-26

> **Update (2026-04-26):** Phases 1 (`per_hec`) and 2 (`sharded` +
> `two_level`) are shipped, with one notable design deviation from
> the original proposal: **L2 is a single shared lock-free `IOArray`
> rather than a 16-way sharded structure**.  See "Implementation
> notes — what changed during build" below for the rationale.
> The proposal text retains the original sharded sketch as the
> historical record; the *Implementation notes* and *Measured
> results* sections describe what actually shipped.

## Summary

The Haskell WAM target now exposes four LMDB-backed fact-source
strategies (PR #1637): `inline_data`, in-memory `IntMap`, direct
`lmdb` (dupsort with per-thread cursor), and `lmdb` plus
`lmdb_cache_mode(memoize)` — a demand-driven shared result cache.
Honest measurement on the 1000-seed enwiki dupsort workload showed
that the shared cache *regresses* under `+RTS -N4` (a 7× slowdown
vs the bare dupsort path) because every miss incurs an
`atomicModifyIORef'` write that serialises the four cores against
the same cache pointer.

The fix is a two-level cache: a per-HEC L1 with no synchronisation,
and a sharded L2 that recovers cross-HEC overlap without paying
single-pointer contention. This document specifies both tiers, sets
out the decision rules for selecting one, and connects the choice
to the cost-based optimisation work in
[`COST_BASED_OPTIMIZATION.md`](./COST_BASED_OPTIMIZATION.md).

## Motivation

The merged `lmdb_cache_mode(memoize)` option is correct (no race
conditions, identical results to direct LMDB) but only useful on
workloads with substantial subgraph overlap. On random shallow
seeds the cache hit rate is near zero and the IORef contention
dominates. From the PR #1637 benchmark, end-to-end through the WAM
target binary on 1000 seeds:

| Config | -N1 | -N4 | -N4 vs -N1 |
|---|---|---|---|
| dupsort (per-thread cursor only) | ~60ms | ~32ms | **1.9× faster** |
| dupsort + `lmdb_cache_mode(memoize)` | ~65ms | **~236ms** | **3.6× SLOWER** |

Three forces are competing:

1. **FFI cost per LMDB lookup** (~1 µs each). Eliminated by an
   in-memory cache on hit.
2. **Cache infrastructure cost** (~hundreds of ns per call:
   `readIORef` + `IM.lookup`, plus `atomicModifyIORef'` per miss).
3. **Cross-core CAS contention** on the shared cache pointer under
   `parMap rdeepseq` with `-N>1`.

The current shared cache pays (2) and (3) on every call but only
collects (1)'s benefit on cache hits. With near-zero hits the
arithmetic always loses. A well-designed cache must structure
itself so that (3) does not happen on the hot path — which is what
per-HEC L1 plus sharded L2 achieves.

This is the same hierarchy CPUs use (per-core L1, shared L2/L3)
and the same hierarchy Redis Cluster uses at network scope (local
client cache, sharded distributed cache). The patterns scale
across orders of magnitude precisely because contention has to be
distributed somehow.

## Core position

**Cache contention should be eliminated on the hot path, distributed
on the slow path, and avoided entirely when overlap doesn't justify
either.**

The hot path is whichever lookup the workload exercises most. For a
DFS over Wikipedia categories, the hot path is "key already seen by
this thread" — that's where per-HEC L1 belongs. The slow path is
"key seen by another thread but not this one" — that's where
sharded L2 helps. The third case, "key never seen by anyone," is
the irreducible LMDB FFI cost no cache can shorten.

When overlap is low (e.g. random shallow seeds), no cache helps.
The cost-based optimiser should recognise this and select the
direct dupsort path.

## When each tier is appropriate

A predicate's fact-access strategy is chosen at codegen time. The
existing decision points (compiled vs IntMap vs LMDB) are documented
in [`WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md`](./WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md).
This proposal extends that decision tree with cache-tier selection
*within* the LMDB option:

```
Fact source ─┬─ inline_data      (≤ 1k facts; tiny scale)
             ├─ IntMap (memory)  (medium scale; full graph fits in RAM)
             └─ LMDB ─┬─ no cache         (low overlap; or single-threaded)
                     ├─ L1 only           (high intra-thread reuse;
                     │                     many threads but workloads
                     │                     don't share keys)
                     ├─ L2 only           (cross-thread reuse; few threads;
                     │                     shared cache cheaper than
                     │                     duplicate per-thread state)
                     └─ L1 + L2           (both intra and cross-thread
                                          reuse; many threads; high
                                          subgraph overlap)
```

Heuristics for choosing a tier without statistics:

- **No cache** when reachable working set ≈ total LMDB key count
  (full-graph traversal). Cache infrastructure overhead exceeds
  any possible benefit.
- **L1 only** when each spark visits many keys but sparks rarely
  share keys with each other. Per-spark DFS over disjoint subgraphs
  is the canonical example.
- **L2 only** when sparks heavily share keys (e.g. all queries
  drill toward a common root) and `-N` is small (≤ 2). With few
  threads the L1 memory duplication wastes more than the L2
  contention costs.
- **L1 + L2** when both overlap dimensions are present and
  `-N` ≥ 4. The L1 absorbs the hot path; the L2 picks up cross-HEC
  hits the L1s would otherwise miss.

When statistics are available (next section), the optimiser computes
expected cache hit rates and picks the tier whose total cost is
minimal.

## Specification

### Prolog API

Replace the current `lmdb_cache_mode(memoize)` flag with a four-way
spelling:

```prolog
lmdb_cache_mode(none)          % default; bare dupsort path
lmdb_cache_mode(per_hec)       % L1 only
lmdb_cache_mode(sharded)       % L2 only
lmdb_cache_mode(two_level)     % L1 + L2
```

The current `memoize` value remains accepted as a synonym for
`sharded` to preserve compatibility with PR #1637 (with a
deprecation note).

All values are gated by `lmdb_layout(dupsort)`; selecting a cache
mode without dupsort is a Prolog warning and silently produces
`none` (matches existing behaviour).

### Codegen contract

Each cache mode emits a different `lmdbCachedEdgeLookup` definition
in the generated `WamRuntime.hs` from the
`templates/targets/haskell_wam/lmdb_fact_source.hs.mustache`
template. The mustache flags are:

| Flag | True when |
|---|---|
| `{{#lmdb_cache_l1}}` | mode ∈ {`per_hec`, `two_level`} |
| `{{#lmdb_cache_l2}}` | mode ∈ {`sharded`, `two_level`} |
| `{{#lmdb_cache_two_level}}` | mode = `two_level` |

The L1 and L2 sections compose: a `two_level` cache emits both,
plus the wiring that consults L1 first and falls through to L2.

### Runtime invariants

For all cache modes:

- Cache contents are never inconsistent with LMDB. The dupsort store
  is read-only during queries, so any cached value is correct
  forever.
- A cache miss triggers the underlying dupsort lookup (per-thread
  cursor cache as today). The result is inserted into the cache(s)
  before being returned.
- Caches are populated lazily; an empty cache is correct.
- Caches are never explicitly cleared during a run. Memory bound is
  enforced by capacity (see L1 design) or by relying on the
  workload's natural footprint (see L2 design).

## L1 design — per-HEC cache, no synchronisation

### Data structure

```haskell
-- One L1 cache per Haskell HEC (one thread per -N).  Stored in an
-- IORef with NO atomic update primitive — the IORef is read and
-- written only by the thread that owns it, so plain readIORef /
-- writeIORef suffice.  No CAS, no MVar, no STM.
data L1Cache = L1Cache
  { l1Map      :: {-# UNPACK #-} !(IORef (IM.IntMap [Int]))
  , l1Capacity :: {-# UNPACK #-} !Int  -- max entries (e.g. 4096)
  }
```

Per-HEC ownership is achieved by keying on `myThreadId`. The cache
registry is the same `Map ThreadId` pattern already used for
per-thread cursors (PR #1632), so it doesn't introduce a new
synchronisation point.

```haskell
type L1Registry = MVar (Map.Map ThreadId L1Cache)

getOrAllocL1 :: L1Registry -> Int -> IO L1Cache
getOrAllocL1 reg cap = do
    tid <- myThreadId
    m   <- readMVar reg
    case Map.lookup tid m of
      Just c  -> return c
      Nothing -> do
        ref <- newIORef IM.empty
        let c = L1Cache ref cap
        modifyMVar_ reg (\m' -> return (Map.insert tid c m'))
        return c
```

### Lookup

```haskell
l1Lookup :: L1Cache -> Int -> IO (Maybe [Int])
l1Lookup (L1Cache ref _) key = do
    m <- readIORef ref          -- thread-local: no contention
    return (IM.lookup key m)

l1Insert :: L1Cache -> Int -> [Int] -> IO ()
l1Insert (L1Cache ref cap) key vs = do
    m <- readIORef ref
    let m' | IM.size m >= cap = IM.insert key vs (evictOne m)
           | otherwise        = IM.insert key vs m
    writeIORef ref m'           -- thread-local: no contention
```

`evictOne` is intentionally simple — for example, drop the smallest
key, or randomly evict via `Data.Hashable`. Strict LRU is more
expensive than the cache savings on this scale; we prefer
overwrite-on-collision behaviour as suggested in the design
discussion (the user's "memory-bound hashtable" idea).

### Capacity

L1 capacity should be sized so the hot working set per thread fits.
For the enwiki workload (~5,200 distinct nodes touched across all
seeds, distributed roughly evenly across HECs), 1024 entries per
HEC is more than sufficient. Capacity becomes a tunable option:

```prolog
lmdb_cache_l1_capacity(1024)   % default
```

### Memory cost

`-N` HECs × `capacity` entries × ~80 bytes per `IntMap` entry
(spine + key + boxed `[Int]` cons cells). For `-N4` at capacity
1024 this is roughly 1.3 MiB total — negligible vs the LMDB working
set itself.

## L2 design — sharded shared cache

### Data structure

```haskell
-- Sharded shared cache.  N independent IORef-backed maps, each
-- protected by its own atomicModifyIORef'.  Contention is
-- distributed across N shards, so the expected serial fraction
-- drops by ~N (assuming uniform key distribution).
data L2Cache = L2Cache
  { l2Shards    :: {-# UNPACK #-} !(V.Vector (IORef (IM.IntMap [Int])))
  , l2NumShards :: {-# UNPACK #-} !Int  -- power of two
  }
```

Shard count of 16 is the recommended default. Lower shard counts
leave too much contention; higher counts inflate memory without
linear gain. Powers of two let `key .&. (numShards - 1)` replace
`mod`.

### Lookup

```haskell
l2Lookup :: L2Cache -> Int -> IO (Maybe [Int])
l2Lookup (L2Cache shards n) key = do
    let !shardIdx = key .&. (n - 1)
    m <- readIORef (shards V.! shardIdx)
    return (IM.lookup key m)

l2Insert :: L2Cache -> Int -> [Int] -> IO ()
l2Insert (L2Cache shards n) key vs = do
    let !shardIdx = key .&. (n - 1)
        !shard    = shards V.! shardIdx
    atomicModifyIORef' shard (\m -> (IM.insert key vs m, ()))
```

### Capacity

L2 has no per-shard capacity by default — it's bounded only by the
total LMDB working set, which the workload naturally caps. If the
workload pathologically caches the entire graph an explicit capacity
option could be added later; not in scope for the first
implementation.

### Memory cost

One shared cache total, sized to whatever the workload's reachable
set is. For enwiki at 1000 seeds (~5,200 nodes) this is roughly
0.4 MiB. Still negligible.

## Two-level composition

```haskell
twoLevelLookup :: L1Cache -> L2Cache -> EdgeLookup -> Int -> IO [Int]
twoLevelLookup l1 l2 fallback key = do
    h1 <- l1Lookup l1 key
    case h1 of
      Just vs -> return vs                  -- L1 hit (fast path)
      Nothing -> do
        h2 <- l2Lookup l2 key
        case h2 of
          Just vs -> do
            l1Insert l1 key vs              -- promote L2 hit to L1
            return vs
          Nothing -> do
            let !vs = fallback key          -- LMDB miss
            l1Insert l1 key vs
            l2Insert l2 key vs
            return vs
```

L1 hits never touch L2. L2 hits promote to L1 so subsequent same-key
accesses on the same thread skip the shared layer. Misses fill both
levels.

## Cost-analysis hook

The cache-tier choice is a natural client of the cost-based
optimisation infrastructure introduced in
[`COST_BASED_OPTIMIZATION.md`](./COST_BASED_OPTIMIZATION.md).
That work delivered:

- A `core/statistics.pl` module with `declare_stats/2`,
  `estimate_cost/4`, `load_stats/1`.
- A Go-target analyser that emits per-predicate cardinality and
  field selectivity into a JSON file.
- An optimiser that consumes those statistics for join ordering.

Cache-tier selection needs an *additional* statistic the current
schema doesn't capture: **expected subgraph overlap**, defined as
the average fraction of distinct keys touched per query that are
also touched by at least one other concurrent query. High overlap
favours caches that share results across threads (L2, two_level);
low overlap favours per-HEC L1 or no cache at all.

A first approximation can be derived from existing statistics:

```
overlap_factor(P) ≈ (avg_keys_per_query(P) × queries_per_run(P))
                    / cardinality(P)
```

When the numerator approaches `cardinality(P)`, every key is touched
multiple times across queries — L2 pays. When the numerator is
small relative to cardinality, queries hit disjoint regions — L1 or
no cache.

The cost model becomes:

```
cost(P, mode, N) =
    expected_misses(P, mode, N) × FFI_cost
  + expected_hits(P, mode, N)   × (mode-specific hit cost)
  + expected_writes(P, mode, N) × (mode-specific write cost)
  + contention_penalty(mode, N)
```

Where `contention_penalty` is roughly:

```
  none      → 0
  per_hec   → 0
  sharded   → α × misses × (N / num_shards)
  two_level → α × L2_misses × (N / num_shards)
```

`α` is calibrated empirically (the PR #1637 benchmark gives one
data point: at `-N4` with ~5,200 misses against 1 shard, total
penalty ≈ 200 ms; that implies `α` ≈ 10 µs per miss-thread-product
on this hardware, which seems plausible for cmpxchg with full
cache-line bouncing).

This is the hook to standardise the **C# target's cost analysis
direction** with the Haskell target's cache choice. Both targets
consume the same per-predicate statistics; both can produce
optimiser hints that propagate down to backend-specific decisions
(C# query plan, Haskell cache mode). Co-locating the statistics
schema in `core/` keeps the targets honest.

This integration is **out of scope for the first implementation**
of L1/L2. Phase 1 ships user-selectable cache modes; Phase 3 wires
the optimiser.

## Phased implementation

### Phase 1 — L1 cache (~1 day)

- Add `lmdb_cache_mode(per_hec)` option in `wam_haskell_target.pl`.
- Emit `L1Cache` type, registry, lookup/insert via new
  `{{#lmdb_cache_l1}}` template sections.
- Three new B1 tests: option emits the L1 code, options compose with
  existing dupsort path, default mode unchanged.
- End-to-end smoke: regenerate enwiki int-atom benchmark with
  `--cache-l1`, verify same 860 results, measure parallel scaling.

**Acceptance:** at `-N4` on the 1000-seed enwiki workload, L1 mode
matches or beats the bare dupsort path (≈ 32 ms). It does not have
to beat single-threaded baseline; it has to *not regress* under
parallelism.

### Phase 2 — L2 sharded cache + two-level (~1 day)

- Add `lmdb_cache_mode(sharded)` and `lmdb_cache_mode(two_level)`.
- Emit `L2Cache` type, sharded shards, lookup/insert via
  `{{#lmdb_cache_l2}}` and `{{#lmdb_cache_two_level}}` sections.
- Promote-on-L2-hit logic in the two-level lookup.
- Tests for option composition.
- Construct a workload with measurable subgraph overlap (deeper
  paths, seeds drilling toward a small root set) to demonstrate
  L2's value.

**Acceptance:** on the overlap-heavy workload, `two_level` mode
beats `per_hec` (which beats `none`). On the random-seed workload,
all modes are within 20% of `none`.

### Phase 3 — cost-analysis hook (separate proposal)

- Extend `core/statistics.pl` with the `overlap_factor/2`
  statistic.
- Teach the optimiser to choose `lmdb_cache_mode` automatically when
  statistics are present.
- Same hook serves the C# target's cost-driven plan selection.

This phase has its own design doc (the C# materialisation work it
ties into is referenced in
[`WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md`](./WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md#alignment-with-the-broader-system)).

## Implementation notes — what changed during build

This section records the deviations between the proposal text above
and the implementation that actually shipped in PR #1640 (Phase 1)
and PR #1641 (Phase 2).  The rest of the proposal is preserved as
originally written; this is the diff.

### L1 — IOArray instead of IORef IntMap

The proposal sketch used `IORef (IntMap [Int])` per HEC.  Profiling
the first implementation showed `IM.insert` consuming **42% of
total time** — every cache miss allocated ~10 IntMap tree nodes
under copy-on-write.  Replaced with `IOArray Int L1Entry` indexed
by `key .&. (cap - 1)` with overwrite-on-collision.  No allocation
per write, just a single `writeArray`.  Net: L1 went from **~145 ms
@ -N1 to ~62 ms** on the 1000-seed enwiki workload (2.3× faster
implementation).

### L1 — capability index instead of `Map ThreadId`

The proposal had an `IORef (Map ThreadId L1Cache)` registry.  The
shipped implementation pre-allocates `Array Int L1Cache` with one
entry per capability and uses `threadCapability` to index it
directly.  No `Map.lookup` on the hot path.  Cleaner and slightly
faster.

### L2 — single shared lock-free IOArray instead of sharded locks

The proposal specified 16 shards each protected by `atomicModifyIORef'`
to distribute contention.  The shipped implementation is **one
shared `IOArray`, lock-free**, because:

1. GHC's `IOArray` pointer writes are atomic on x86_64 (and most
   modern ISAs) — readers see either the old or the new pointer,
   never a torn write.
2. Every cached value is correct (LMDB is read-only during queries),
   so when two threads race to write the same slot, both stored
   values are equally valid.  The "winner" overwrites; correctness
   is preserved either way.

This eliminates both the per-shard `atomicModifyIORef'` overhead
*and* the sharding bookkeeping.  Simpler and faster than the
proposal's design.  The original sharded sketch (lines 259–307
above) is preserved as the design rationale; the shipped code is
the racy-write version.

### IO-direct fallback to skip nested `unsafePerformIO`

Both `lmdbL1EdgeLookup` and `lmdbRawEdgeLookup` are
`unsafePerformIO`-wrapped.  On a cache miss the L1 wrapper used to
unwrap `unsafePerformIO` twice (once for itself, once for the
fallback).  Split `lmdbRawEdgeLookup` into an IO version
(`lmdbRawEdgeLookupIO`) and a pure wrapper; the cache wrappers call
the IO version directly to avoid the nested unwrap.  Trimmed ~5 ms
off `-N4` enwiki.

### Memory-aware default L2 capacity

Per the design feedback, `defaultL2Capacity` reads
`/proc/meminfo` `MemAvailable` and computes
`min(half-available, available - 500MB) / 32` entries, bounded
`[1024, 1M]` and rounded down to a power of two.  Falls back to
1 GB / 32 = 32M, then bounded.  On non-Linux, defaults straight to
1 GB / 32 → 1M cap.  User override via
`lmdb_cache_l2_capacity_bytes/1` is **pending Phase 3**.

### `memoize` is now a deprecated synonym for `sharded`

PR #1637's `lmdb_cache_mode(memoize)` (shared `IORef (IntMap ...)`)
regressed 7× at `-N4` because every miss took
`atomicModifyIORef'`.  Phase 2 unified that path with the new
`sharded` mode (lock-free IOArray); the option is preserved as a
deprecated synonym so existing code compiles, but the underlying
implementation is the new IOArray.

## Measured results (consolidated)

All numbers warm-cache, alternating-order trials, `+RTS -A32M`
mandatory (without it, parallel GC dominates — see
`docs/design/WAM_HASKELL_BENCHMARK_STRATEGY.md`).

### Phase 1 — IOArray + capability index + IO-direct (PR #1640)

1000-seed enwiki, root 97688913 reachable from all measured seeds:

| Mode | -N1 | -N4 |
|---|---|---|
| BASE | ~62ms | ~31ms |
| L1 IntMap (initial impl) | ~145ms | ~46ms |
| L1 IOArray | ~62ms | ~32ms |
| **L1 IOArray + IO-direct (shipped)** | **~62ms** | **~27ms** |

**Phase 1 acceptance met**: L1 beats BASE at -N4 by ~13%.

### Phase 2 — L2 + two_level (PR #1641)

5000-seed simplewiki, all reaching root 265340:

| Mode | -N4 avg query_ms | vs BASE |
|---|---|---|
| BASE | ~62ms | 1.00× |
| L1 (per_hec) | ~52ms | 1.19× |
| L2 (sharded) | ~51ms | 1.22× |
| **two_level** | **~49ms** | **1.27×** |

1000-seed enwiki, four-way at -N4 (within-noise variance):

| Mode | avg query_ms |
|---|---|
| BASE | ~28ms |
| L1 | ~26ms |
| L2 | ~25ms |
| two_level | ~29ms |

**Phase 2 acceptance**: simplewiki shows the expected ordering
(`two_level` ≥ `sharded` ≥ `per_hec` > `none`).  Enwiki at 1000
seeds is too small to differentiate the cache modes statistically;
the hypothesis "advantage scales with workload" is consistent with
both data points but not yet decisively tested at enwiki scale.

### Memory cost on a constrained host (~5 GB RAM, 2.5 GB available)

Process RSS during simplewiki run, all four modes: **~80 MB total**,
of which the L2 IOArray (1M slots × 8 bytes) accounts for ~8 MB.
The 1 TB `VmSize` is the LMDB `mdb_env_set_mapsize` reservation;
only touched pages count against physical memory.

## Open questions

1. **L1 eviction policy.** Capacity-based overwrite is simplest;
   LRU is more accurate but more expensive. For Phase 1, simplest
   wins. Revisit if traces show pathological eviction.

2. **L2 capacity.** No bound by default; rely on the workload.
   This may be insufficient for very-large-graph traversals where
   every key gets cached forever. Add an option only if measurement
   demands it.

3. **L1 capacity per HEC vs total.** The proposal sizes per-HEC.
   An alternative is total-capacity-divided-by-N which preserves
   total memory budget across `-N` settings. Per-HEC is simpler
   and the memory cost is small enough that exact budgeting isn't
   necessary.

4. **Cross-run cache persistence.** Both L1 and L2 are in-process
   only. Surviving cache across program restarts would need either
   a serialisation pass or a backing store (Redis, mmap'd file).
   Not in scope; see "Where a server actually does help" in the
   PR #1637 discussion thread.

5. **Cache invalidation.** Not applicable: the dupsort store is
   read-only during queries. If we ever add write paths, the
   invariant breaks and the cache design needs revisiting.

## Non-goals

- **Replacing the existing direct LMDB path.** It remains the
  default and the right choice for low-overlap workloads.
- **Auto-selection without statistics.** Phase 1/2 are user-opt-in
  via Prolog options. Auto-selection is Phase 3 and depends on
  cost-analysis infrastructure that is not yet
  cache-mode-aware.
- **Persistence across runs.** In-process caches only.
- **Inter-process sharing.** A separate-server design would be a
  Redis-style cache, not a Haskell L1/L2. The use case for that
  is multi-process workers sharing cache, which UnifyWeaver's
  current single-binary deployment doesn't have.

## Connection to existing work

- **[`WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md`](./WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md)**
  established that the engine should decide what to retain, not
  the loader. Cache tiers are a runtime extension of that
  principle: the engine decides at codegen time *and* at runtime
  (via cache hit/miss) what stays in memory.

- **[`COST_BASED_OPTIMIZATION.md`](./COST_BASED_OPTIMIZATION.md)**
  delivered the statistics infrastructure that Phase 3 will
  consume. Both this proposal and any future C# optimiser-driven
  work plug into the same `core/statistics.pl` registry.

- **PR #1631** (int-atom seeds + dupsort layout) and
  **PR #1632** (per-thread cursor cache) made parallel LMDB
  access possible at all. This proposal extends parallel correctness
  to parallel performance.

- **PR #1637** (current cache-mode option) ships the foundation
  this proposal builds on. The `memoize` synonym preserves
  compatibility while the four-way spelling matures.
