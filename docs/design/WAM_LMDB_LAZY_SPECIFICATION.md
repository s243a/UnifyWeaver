# LMDB Lazy Access: Specification

**Status**: Design specification. Companion to
[`WAM_LMDB_LAZY_PHILOSOPHY.md`](WAM_LMDB_LAZY_PHILOSOPHY.md) (the
"why") and [`WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md`](WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md)
(the "when").

**Snapshot date**: 2026-05-20.

This document specifies the cross-target interface for the three
LMDB-access tiers `eager` / `lazy` / `cached` (see philosophy doc §2 for the
vocabulary), plus the scan-vs-seek axis and the workload-segregation
contract. The intent is that each target's implementation can be
audited against the same surface area.

## 1. Tier model

Three modes per target:

| Tier | Materialisation | Cache layer | Per-call cost |
| --- | --- | --- | ---: |
| `eager` (eager) | Yes — full demand-set Vec/HashMap built at startup | n/a (the materialisation *is* the cache) | O(1) HashMap lookup |
| `lazy` (lazy) | No | None | O(log B) LMDB cursor seek |
| `cached` (lazy + cache) | No | Yes — bounded-size LRU or sharded HashMap | O(1) on hit, O(log B) on miss |

A target MUST implement `eager` (which most already do). A target SHOULD
implement `cached` for any LMDB-backed fact source whose `fact_count`
exceeds the existing `use_lmdb(auto)` threshold. A target MAY
implement `lazy` for workloads with explicit segregation declarations.

## 2. The canonical lookup interface

All three tiers expose the **same shape** to callers. The contract:

```
lookup_parents(key) -> Iterator<Edge>
```

where `Iterator<Edge>` is each language's native lazy-sequence type:

| Target | Concrete type |
| --- | --- |
| Haskell | `Int -> [Int]` (lazy list) or `Int -> Conduit Int IO ()` |
| Rust | `fn lookup_parents(&self, key: i32) -> impl Iterator<Item = i32> + '_` |
| Go | `func (s *Source) LookupParents(key int32) <-chan int32` |
| C# | `IEnumerable<int> LookupParents(int key)` |
| Python | `def lookup_parents(self, key: int): yield ...` |
| Elixir | `def lookup_parents(source, key), do: Stream.unfold(...)` |

The iterator MUST yield ints (or whatever the canonical ID type is
for the predicate). String conversion happens at the caller's
discretion via i2s — this is consistent with the existing LMDB-resident
interning convention.

The iterator MUST be lazy in the sense that constructing it does no
LMDB work; only the first `.next()` call triggers the cursor. This
makes the iterator usable as a return value from a foreign predicate
without forcing materialisation.

## 3. Source trait / interface

Each target defines a source trait that all three tiers implement:

### Rust

```rust
pub trait LookupSource {
    type EdgeIter<'a>: Iterator<Item = i32> + 'a where Self: 'a;
    fn lookup_parents<'a>(&'a self, child_id: i32) -> Self::EdgeIter<'a>;
}
```

Implementations:

```rust
pub struct EagerVecLookup { ... }
pub struct LmdbCursorLookup<'env> { ... }  // `lazy`
pub struct CachedLookup<S: LookupSource> { inner: S, cache: ... }  // `cached` decorator
```

### Haskell

```haskell
class LookupSource s where
    lookupParents :: s -> Int -> [Int]
```

Haskell already has this conceptually inside `EdgeLookup` in
`lmdb_fact_source.hs.mustache`; the spec confirms the shape.

### Other targets

Same pattern: a trait/interface whose method returns the target's
native lazy iterator type. Each tier implements the trait.

## 4. Cache-tier spec (`cached`)

The cache layer wraps any `lazy` implementation via the Decorator pattern.
Spec:

- **Bounded size**: configurable via existing `cache_capacity(...)`
  option (already in `cache_strategy(auto)`).
- **Sharded for parallelism**: `cache_shards(...)` already exists in
  Haskell; the spec confirms it's the canonical knob. Default
  shards = number of capabilities / threads.
- **Eviction policy**: LRU on a per-shard basis. The cache contract
  is "correctness preserved on eviction" — any miss falls through to
  the inner `lazy` source.
- **Composition with `lazy`**: `cached` is a Decorator over `lazy`, not a replacement.
  Same `LookupSource` trait. This means `lazy` and `cached` are interchangeable
  in calling code — the kernel doesn't know which it has.
- **Concurrent access**: per-shard locking (Haskell uses `IORef`-protected
  Maps; Rust uses `dashmap` or sharded `RwLock<HashMap>`).

## 5. Scan vs seek

A second axis, orthogonal to the `eager`/`lazy`/`cached` tier. Some sources can
expose a *range* read in addition to point lookups.

### 5.1 Scan trait

```rust
pub trait ScanSource: LookupSource {
    type RangeIter<'a>: Iterator<Item = (i32, i32)> + 'a where Self: 'a;
    fn scan_range<'a>(&'a self, start_inclusive: i32, end_exclusive: i32)
        -> Self::RangeIter<'a>;
}
```

The iterator yields `(child, parent)` pairs from the half-open
range `[start, end)` of child IDs. Only `LmdbCursorLookup`
implements `ScanSource`; the eager Vec implementation can fall
through to a filter on its existing data.

### 5.2 When the kernel uses scans

The kernel uses scans when:

- The keys to look up are *contiguous in ID space* (a precondition
  the cost model checks), AND
- The number of keys is large enough that the per-call seek cost
  dominates (call it `scan_threshold_keys`, default ~16).

Otherwise the kernel uses repeated point seeks via `lookup_parents`.

### 5.3 Physical-layout signals

The cost model needs a `physical_layout_quality` signal to decide
whether scan-mode is worth using. Initial implementation: a manifest
field on the LMDB written by the ingest pipeline.

```
meta:
  layout_strategy:    insertion_order | topological | semantic | mst
  scan_threshold:     <integer>
```

Default: `insertion_order` (current state); cost model defaults to
seeks-only.

## 6. Workload-segregation contract

The signal that allows `lazy` to be picked over `cached`.

### 6.1 Caller-side declaration

A new option on the bench harness or the recursive_kernel
declaration:

```prolog
:- recursive_kernel(category_ancestor, category_ancestor/4, [
    max_depth(10),
    edge_pred(category_parent/2),
    workload_segregated(true),         % NEW
    segregation_cluster_id(physics)    % NEW (optional)
]).
```

When `workload_segregated(true)`, the cost-model resolver MAY pick
`lazy` over `cached`. When unset (default), the resolver picks `cached`.

The `segregation_cluster_id` is purely informational at this stage
— it provides a name for telemetry / cache-eviction-boundary
purposes. The cost model does not enforce that different cluster
IDs use different cache scopes; that's a future refinement.

### 6.2 What "segregated" means semantically

A workload is *segregated* iff:

- The query stream issued in a single process invocation does not
  revisit any LMDB key.
- OR: the cache hit rate would be effectively zero anyway (e.g., the
  workload is one query against a cold cache).

If the caller declares `workload_segregated(true)` and the
guarantee is violated, the result is **slower** `lazy` execution due
to repeated LMDB cursor walks — but **not incorrect**. `lazy` produces
the same answers as `cached` on identical input; the segregation flag is
strictly a performance hint.

### 6.3 Compiler-inferred segregation (deferred)

Shape analysis could in principle prove segregation:

- If `max_depth × branching_factor < total_seed_count` AND
- Visited sets between seeds are guaranteed disjoint (rare in
  general; common when seeds are partitioned by root)

But this requires the cost model to know about visited-set
intersection, which it doesn't today. Deferred to a future
revision.

## 7. Cost-model integration

The existing resolver predicates extend to cover the new axes:

### 7.1 New input fields

Added to the workload-metadata vocabulary used by `resolve_*` predicates:

| Field | Type | Default | Source |
| --- | --- | --- | --- |
| `workload_segregated` | bool | false | caller-declared |
| `segregation_cluster_id` | atom | _none_ | caller-declared |
| `physical_layout_quality` | enum | `insertion_order` | LMDB meta sub-db |
| `scan_threshold_keys` | int | 16 | LMDB meta sub-db |
| `cache_miss_rate_estimate` | float | 0.5 | empirical / heuristic |
| `expected_query_count_per_process` | int | 1 | caller-declared |

### 7.2 Resolver: `resolve_lmdb_lookup_tier/2`

New predicate, sits next to the existing `resolve_auto_lmdb_cache_mode/2`:

```prolog
resolve_lmdb_lookup_tier(Options, Tier) :-
    % Tier = eager | lazy | cached
    option(fact_count(F), Options),
    option(demand_set_estimate(D), Options),
    option(memory_budget(B), Options),
    option(workload_segregated(WS), Options, false),
    option(expected_query_count_per_process(NQ), Options, 1),
    edge_size_bytes(F, EdgeBytes),
    (   D * EdgeBytes > B
    ->  % Demand set doesn't fit; must use lazy
        ( WS == true -> Tier = `lazy` ; Tier = `cached` )
    ;   % Demand set fits; eager wins if NQ * ε amortises M
        crossover_eager_lazy(F, D, NQ, ChooseEager),
        (   ChooseEager == true
        ->  Tier = `eager`
        ;   ( WS == true -> Tier = `lazy` ; Tier = `cached` )
        )
    ).
```

### 7.3 Resolver: `resolve_lmdb_access_mode/2`

For the scan-vs-seek axis:

```prolog
resolve_lmdb_access_mode(Options, Mode) :-
    % Mode = seek | scan
    option(physical_layout_quality(Q), Options, insertion_order),
    option(expected_keys_per_call(K), Options, 1),
    option(scan_threshold_keys(T), Options, 16),
    (   Q \= insertion_order, K >= T
    ->  Mode = scan
    ;   Mode = seek
    ).
```

The two resolvers compose: `(eager|lazy|cached) × (seek|scan)` = 6 combinations,
but only 4 are meaningful (`eager` doesn't use either; `lazy`+scan, `cached`+seek,
`cached`+scan, and `lazy`+seek are the four lazy options).

## 8. Generated-code shape

### 8.1 Rust (per-tier)

For `eager` — current behaviour:

```rust
let runtime_category_parents: Vec<(String, String)> = (...materialisation...);
vm.register_indexed_atom_fact2("category_parent/2", build_indexed_fact2(&runtime_category_parents));
```

For `lazy`:

```rust
let lookup = LmdbCursorLookup::new(&lmdb, &reachable_ids, &i2s);
vm.register_foreign_lookup("category_parent/2", Box::new(lookup));
```

For `cached`:

```rust
let inner = LmdbCursorLookup::new(&lmdb, &reachable_ids, &i2s);
let lookup = CachedLookup::new(inner, /* cache_capacity */ 1024, /* shards */ 4);
vm.register_foreign_lookup("category_parent/2", Box::new(lookup));
```

`register_foreign_lookup` is the new runtime API the WAM-Rust runtime
needs. Today the kernel calls into `ffi_facts: HashMap<String,
HashMap<u32, Vec<u32>>>` for foreign-predicate edge lookups; the `lazy`/`cached`
path replaces that with a trait object call.

### 8.2 Haskell (per-tier)

`eager` — `resident` (IntMap) mode. Already implemented.

`lazy` — a hypothetical `lazyCursor` mode (not yet implemented). Would
remove the per-HEC L1 and sharded L2 caches from the
`lmdbFactSource.hs.mustache` template.

`cached` — `resident_cursor` mode with `lmdb_cache_mode(per_hec |
sharded | two_level)`. Already implemented.

So in Haskell, `cached` is the existing implementation; `lazy` would be an `cached`
instance with cache size = 0, which is a configuration value
already exposed. **No new Haskell code is required to support `lazy`**;
the spec just blesses it as a valid configuration.

### 8.3 Scan-mode hooks

Per-target, the scan iterator opens a cursor at `start` and iterates
until it sees a key ≥ `end`. The contract is the same across all
targets; the implementation uses each target's native LMDB binding's
range API.

## 9. Testing contract

For each target and each tier:

1. **Correctness**: a small fixture (`1k_cats`, 5933 edges) yields
   identical `tuple_count` and per-tuple `effective_distance` under
   `eager`, `lazy`, and `cached` modes.
2. **Cache behaviour**: at `cached`, count cache hits / misses via
   instrumentation; assert hits > 0 on the test workload.
3. **Segregation contract**: at `lazy` with `workload_segregated(true)`,
   confirm no cross-seed key revisits via instrumentation.
4. **Scan vs seek**: a smoke test on a topologically-sorted fixture
   confirms scan yields the same results as repeated seeks.

## 10. Versioning and migration

Adding `lazy` / `cached` / scan-mode is *additive*:

- Existing benchmarks default to `eager` (unchanged behaviour).
- The cost-model resolver opts into `cached` only when `lmdb_lazy_tier(auto)`
  is set; existing call sites without that option remain at `eager`.
- The LMDB on-disk format is unchanged. The new `layout_strategy` and
  `scan_threshold` fields in the meta sub-db are optional reads with
  documented defaults.

This means each target can adopt the `lazy`/`cached` work incrementally without
breaking existing benchmarks.

## 11. Out of scope (for this spec revision)

- **Multi-process shared cache**: would let `cached` share state across
  process invocations. Out of scope; could be added with the LMDB
  itself as a shared cache by definition (an `cached`-as-LMDB-table
  pattern).
- **Adaptive tier switching**: the cost model picks once per
  process. Switching tiers mid-run is more invasive and not
  motivated by any current measurement.
- **Compiler-inferred segregation**: see §6.3. Deferred.
- **MST-sort ingest pre-processor**: see philosophy doc §4.2.
  Deferred to its own design doc when motivated by a workload.

## 12. References

- `WAM_LMDB_LAZY_PHILOSOPHY.md` — the "why" doc.
- `WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md` — the "when" doc with
  phased rollout.
- `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md` — LMDB layout this
  spec reads from.
- `QUERY_PLAN_RUNTIME_PHILOSOPHY.md` — the broader runtime-planner
  pattern.
- `CACHE_COST_MODEL_PHILOSOPHY.md` — cost-model framework.
- `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache`
  — current Rust `LookupSource`-equivalent (eager only).
- `templates/targets/haskell_wam/lmdb_fact_source.hs.mustache` —
  current Haskell `cached` implementation.
