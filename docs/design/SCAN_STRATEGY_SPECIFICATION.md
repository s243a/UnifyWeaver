# Scan Strategy — Specification

This document specifies the concrete shapes (data structures, option
list, formulas, composition rules) of the design laid out in
`SCAN_STRATEGY_PHILOSOPHY.md`. Read the philosophy doc first if the
"why" isn't clear.

## Phases and data structures

### Phase W (warm-build)

| name | type sketch | role |
| --- | --- | --- |
| `WarmHeap`  | `Data.Heap.MinPrioHeap Cost (NodeId, Edges)` | bounded-capacity **min-heap** keyed by cost; on insertion when at capacity, the minimum (lowest-cost) entry is evicted to make room for a higher-scored candidate |
| `WarmIndex` | `IntMap NodeId Cost`                          | "do we already have this node, and at what cost?" — O(log N) lookup back into the heap |
| `Frontier`  | `IntMap NodeId Cost`                          | candidate nodes not yet visited; same shape, separate logical role |
| `Visited`   | `IntSet NodeId`                                | nodes whose edges we've already loaded |

Both `WarmHeap` and `WarmIndex` are needed because a heap alone doesn't
support "does this node already exist?" lookups in O(log N). The
parallel index is the usual trick.

The frontier is *candidates whose score has been updated but whose
edges haven't been loaded yet*. Cursor seeks happen against frontier
nodes selected by score.

### Phase S (snapshot)

Single conversion pass over `WarmHeap`:

```haskell
snapshotCache :: WarmHeap -> IntMap NodeId [EdgeTarget]
```

**Cost values are not retained in the `Cache`.** Steady-state
lookup is by-node only; there's no use for cost metadata once the
cache is built. If cost information is needed post-snapshot (e.g.
for spark routing in P5), it is retained separately via
`snapshotRanked` below, not embedded in `Cache` entries.

Optionally, also produce a flux-ranked view for retention:

```haskell
snapshotRanked :: WarmHeap -> [(NodeId, Cost)]   -- descending by cost
```

`RankedView` is immutable post-snapshot; concurrent reads from
multiple capabilities (the P5 spark-routing use case) are safe
without synchronisation by Haskell's GC contract on pure data.

After snapshot, `WarmHeap`, `WarmIndex`, `Frontier`, `Visited` can
all be GC'd. Steady-state holds only the cache (and optionally the
ranked view).

### Phase R (steady-state, "running")

| name | type sketch | role |
|---|---|---|
| `Cache`        | fixed-capacity hashtable, overwrite-on-collision | O(1) lookup; initialised from snapshot; memoises misses |
| `RankedView`   | `[(NodeId, Cost)]` | retained iff `tree_retention(snapshot_only)` |
| `WarmHeap` + `WarmIndex` | (carried over from warm) | retained iff `tree_retention(live)` |
| `LmdbFallback` | existing cursor handle | cache miss path |

The `Cache` is the same shape as the existing L1/L2 hashtables: a
fixed-bucket array, where insert into a non-empty bucket overwrites
the prior occupant. No LRU/LFU tracking, no metadata per entry,
no eviction policy beyond "the latest insert wins the bucket". This
keeps lookups branch-free and matches the existing code paths the
implementation can reuse.

Lookup logic:

```
edgesOf n =
  case lookupBucket n cache of
    Just (k', es) | k' == n  -> hit es
    _                         -> do        -- miss
      es <- cursor n
      insertBucket n es cache              -- memoise; may overwrite
      return es
```

The cache is **memoised on miss**. A repeated query for the same
node hits even when the cost function didn't predict it. This is
the standard caching contract; the warm snapshot just provides
the *initial state* of the cache, not its frozen contents.

If a snapshot entry collides with a miss-memoised insert, the new
insert wins the bucket. That's tolerable: re-cursoring the lost
snapshot entry on a later query is a single seek, symmetric to any
other miss. We're not trying to permanently protect the snapshot
from the workload's access pattern; the cache is a fixed-capacity
data structure that gets filled by whatever queries actually run.

## Cost functions

### Strategy slot

```prolog
%% Workload-author API on the Prolog side.
tree_cost_function(flux,                [iterations(1), parent_decay(0.5), child_decay(0.3)]).
tree_cost_function(hop_distance,        [max_hops(5)]).
tree_cost_function(semantic_similarity, [dim(128), embedding_path('foo.bin')]).
```

The Haskell-side generated code consumes a `CostFn` abstraction:

```haskell
data CostFn = CostFn
  { cfInitial :: !(NodeId -> Cost)
      -- initial score for an endpoint node (seed or root)
  , cfRelax   :: !(NodeId -> [NodeId] -> ScoreMap -> ScoreMap)
      -- one round of score propagation:
      -- given a node, its neighbours, and the current score map,
      -- update the score map with the relaxed neighbour scores
  }
```

The "score" of a node is just its value in the `ScoreMap` — we
intentionally don't have a separate `cfScore` field. Flux scoring
depends on the global score map (parent/child contributions
require already-scored neighbours), so a pure
`NodeId -> Edges -> Cost` signature can't express it. Keeping the
state in `ScoreMap` and treating `cfRelax` as the sole
score-producing operation avoids the overlap.

Each concrete cost function (flux, hop, semantic) implements
this record. The tree-building algorithm consumes a `CostFn`
and is generic over which one.

### Flux

For each candidate node `n`, flux is the sum of two single-direction
contributions:

```
flux(n) = parent_flux(n) + child_flux(n)

parent_flux(n) = Σ_{p ∈ parents(n)}  (parent_decay / |children(p)|) ^ hops_from_endpoint(p)
child_flux(n)  = Σ_{c ∈ children(n)} (child_decay  / |parents(c)|)  ^ hops_from_endpoint(c)
```

**`hops_from_endpoint` semantics**: the BFS distance from `n` (or its
parent/child neighbour, as in the formulas above) to the *nearest
endpoint*. Endpoints are nodes named by the algorithm's `seeds/1`
predicate. When the workload also declares `roots/1`, both seeds and
roots are treated as endpoints — `hops_from_endpoint(x) = min(BFS to
seeds, BFS to roots)`. This makes flux symmetric for workloads with
two-sided endpoint sets (e.g. effective-distance computing reachability
between an article and a target category).

Reading the formula:

- We decay flux by hops from the nearest endpoint (the standard
  PageRank-style decay).
- Within each hop, we also divide by the branching factor — many
  children means each one inherits less flux from the parent (the
  "probability of taking this specific edge" intuition).
- Parent and child legs use *separate* decay constants because
  Wikipedia (and most category hierarchies) is asymmetric: typically
  1–5 parents per category, 10–1000 children. The same decay
  constant for both would either underweight ancestors or overweight
  descendants.

### Hop distance

```
score(n) = max_hops - min(hops_to_endpoint, max_hops)
```

Simpler, cheaper, no branching-factor dependence. Useful when the
graph is roughly homogeneous in degree or when flux's branching
factor data isn't trusted.

### Semantic similarity

```
score(n) = dot_product(embedding(n), query_embedding) / (|embedding(n)| * |query_embedding|)
```

Requires an embeddings table indexed by `NodeId`. Out of scope for
the first cuts; included in the strategy enum because the slot
should be open to it.

## Iterations

Unified meaning across access patterns:

| iterations | meaning |
| --- | --- |
| 1 | one round of score propagation. Cursor mode: expand frontier by 1 hop, update scores within that hop. Scan mode: one full pass over the edge list, propagating from scored nodes to neighbours. |
| N (positive integer) | N rounds. Approaches converged scoring as N → ∞. |

The cost vs accuracy knob. Default `iterations(1)` for the cheap
path; workload-authors who need converged scoring opt up.

#### Future iteration modes (not yet exposed)

`iterations(auto)` — runs until score-change-per-iteration drops
below a threshold — is a natural endpoint of the cost/accuracy
curve but is **not in the public option space for P0–P5**. It
needs a convergence metric definition (e.g. max-delta or
sum-of-deltas across the score map) plus a per-cost-function
sensible default threshold. Filed as a Phase 6+ extension; until
then, callers passing `iterations(auto)` are coerced to
`iterations(1)` with a stderr warning so a workload author who
typed it doesn't get silent surprises.

## Stage-1 → Stage-2 crossover

The runtime tracks frontier size. When the frontier — i.e. *number
of unvisited candidate nodes whose scores indicate worthwhile
materialisation* — exceeds a threshold, switch from cursor seeks
to a scan pass.

```
K_cross_stage2 = W_remaining / (bandwidth_eff * latency_eff)
```

where `bandwidth_eff` / `latency_eff` come from the cost model's
formula (`CACHE_COST_MODEL_PHILOSOPHY.md`).

**Static threshold, runtime check.** `K_cross_stage2` is computed
**at codegen time** using the same formula as the cost model
(static estimates of `W_remaining` from `db_size_bytes`,
hardware constants from `cost_model_constants/1`). The frontier
size `|Frontier|` is measured **at runtime** during warm-build and
compared against the precomputed threshold. Implementors should
*not* recompute the threshold dynamically — the inputs would
require a runtime probe and the comparison is meant to be cheap.

Concretely:

- Maintain a running count `|Frontier|`.
- After each cursor batch (or every N seeks), check `|Frontier| >
  K_cross_stage2`.
- If yes: drain the frontier via a single scan pass that touches
  every candidate's edges in one go, marking them visited.
- Repeat the check after the scan — if the new frontier (revealed
  by relaxing scores from the just-scanned nodes) is again above
  threshold, scan again. This is the optional fixed-point loop.

For the first cut, the fixed-point is bounded to ≤ 2 scans to
avoid pathological cases.

## Warm budget

Two limits, whichever fires first:

| limit | option | default |
|---|---|---|
| node count | `warm_budget_nodes(N)` | 10% of fact_count |
| time | `warm_budget_ms(N)` | 10000 ms |

When either limit is reached, warm phase ends, snapshot runs.

## Eviction policies

Two distinct policies, for two distinct structures:

### Tree eviction (warm phase)

When the heap is at capacity:

- Compute candidate score.
- Peek at the heap's lowest-scored element.
- If candidate score > lowest, evict lowest, insert candidate.
- Else discard candidate without inserting.

Standard min-heap-bounded operations. The cost function's score
is the eviction key. No LRU/LFU; the score *is* the prediction
of relevance.

### Cache eviction (steady-state)

Overwrite-on-collision. The cache is a fixed-bucket hashtable:

- Lookup: hash `key`, probe bucket. If occupied with matching
  key, hit. Else miss.
- Insert (on miss memoisation): hash `key`, probe bucket. If
  empty, place. If occupied with a different key, overwrite.

This is the same mechanism as the existing L1/L2 caches in the
codebase. No metadata per entry, no priority queue, no LRU chain.
Hash distribution decides what stays.

The cache is sized at codegen time (`cache_capacity_buckets/1`,
default `4 × warm_budget_nodes`). At `B = 4N` the expected
collision rate during snapshot population is roughly
`N(1 - (1 - 1/B)^N) / N ≈ 1 - e^{-1/4} ≈ 22%`. That is, ~22% of
snapshot inserts will hit an occupied bucket and overwrite —
acceptable because:

1. The displaced snapshot entry is one cursor seek to recover on
   first access; the cache catches it via miss-memoisation
   thereafter.
2. Steady-state miss memoisation will overwrite some snapshot
   entries anyway as the workload's actual hot set diverges
   from the prediction.

Workloads that want a tighter floor on snapshot retention can
override (`8×` halves the collision rate again, at the cost of
~2× the cache memory). The Phase L#5/#7 measurements informed
the `4×` default; revisit at P3 when warm-build measurements
land.

## Cache miss handling

Steady-state lookup:

1. Hash key; probe bucket.
2. **Hit** (bucket holds matching key): return edges.
3. **Miss** (bucket empty or holds different key): cursor seek to
   LMDB, then insert into the cache (overwriting whatever was in
   the bucket).

Misses are memoised. The cache learns the workload's actual
access pattern over time, on top of the warm-start the snapshot
provided.

If a snapshot entry gets overwritten by a miss, no special
handling: the next access to the displaced node is just another
miss, paying one cursor seek and re-inserting it. The cache
churn settles to whatever the workload's hot set actually is.

A diagnostic counter (`cache_strategy_verbose(true)`) exposes
hit/miss rates so callers can see whether the snapshot's
prediction accuracy is matching reality.

## Tree retention modes

```prolog
tree_retention(discard).        % default
tree_retention(snapshot_only).  % retain frozen ranked view
tree_retention(live).           % retain full heap + index; allow post-warm updates
```

### `discard` (default)

After snapshot, `WarmHeap`, `WarmIndex`, `Frontier`, `Visited` are
all GC'd. Steady-state holds only the cache. Minimum memory
footprint; suitable when nothing downstream needs the ranked
view.

### `snapshot_only`

The snapshot produces both `Cache` and a frozen `RankedView`:

```haskell
type RankedView = [(NodeId, Cost)]   -- descending by cost; immutable
```

The warm structures themselves still get GC'd; only the immutable
ranked list lives on. This is the "spark routing" use case — the
consumer reads the ranked view to partition work without needing
to update it.

`RankedView` consumers (current target: MoE-style spark routing):

```haskell
-- Partition seeds by flux quartile across N capabilities.
partitionByRank :: Int -> RankedView -> [[NodeId]]
```

Implemented as a follow-up consumer in the implementation plan
(P5); the snapshot just produces the view.

### `live`

`WarmHeap` and `WarmIndex` persist past snapshot. Algorithms can
update the tree during steady-state via:

```haskell
treeInsert  :: NodeId -> Cost -> Edges -> WarmHeap -> WarmHeap
treeUpdate  :: NodeId -> Cost -> WarmHeap -> WarmHeap        -- re-rank
treeRebuild :: WarmHeap -> Cache -> (WarmHeap, Cache)         -- re-snapshot
```

Use cases (none implemented; the door is open):

- **Query-history-driven re-ranking**: observe which nodes are
  actually touched, bump their costs, re-rank.
- **Adaptive cost-function switching**: start with cheap
  `hop_distance`, after N queries swap to `flux` and re-rank
  via `treeRebuild`.
- **Hot-region expansion**: detect a sub-region with high miss
  rate; warm additional nodes around it via `treeInsert`.

**Thread safety in `live` mode**: the warm phase is
single-threaded by design. If steady-state code wants to update
the tree concurrently with cache lookups, the implementation
guards the heap behind an `MVar` or `IORef`. For the first cuts
(P3, P5) only `discard` and `snapshot_only` are exercised, so
the thread-safety question doesn't bite. `live`-mode landing
(deferred to P7+) brings the locking question with it.

## Option list

All options accepted by the warm-build path. Defaults in
parentheses.

| option | default | purpose |
|---|---|---|
| `tree_cost_function(F, Params)` | `flux` with default params | strategy slot |
| `iterations(N)` | `1` | score-propagation rounds |
| `warm_budget_nodes(N)` | `fact_count / 10` | node-count limit |
| `warm_budget_ms(N)` | `10000` | wall-clock limit |
| `stage2_scan_threshold(K)` | derived from cost model | crossover trigger |
| `stage2_max_scans(N)` | `2` | fixed-point bound |
| `tree_retention(R)` | `discard` | `discard` / `snapshot_only` / `live` |
| `cache_capacity_buckets(N)` | `4 × warm_budget_nodes` | hashtable bucket count |

### Flux-specific sub-options

| option | default | purpose |
|---|---|---|
| `parent_decay(D)` | `0.5` | parent-leg geometric decay per hop |
| `child_decay(D)` | `0.3` | child-leg geometric decay per hop |
| `flux_merge(M)` | `sum` | how to combine parent + child flux: `sum`, `max`, `weighted(W1,W2)` |

### Hop-distance sub-options

| option | default | purpose |
|---|---|---|
| `max_hops(N)` | `5` | hop cutoff |

### Semantic-similarity sub-options

| option | default | purpose |
|---|---|---|
| `dim(D)` | `128` | embedding dimension |
| `embedding_path(P)` | required | path to embeddings file |

## Composition with existing resolvers

Decision order at codegen time:

1. **(new)** `load_algorithm_manifest/2` — merge
   `user:algorithm_optimization/2` facts into the option list.
   Caller-provided options win on conflict. See
   `ALGORITHM_MANIFEST_SPECIFICATION.md`.
2. `resolve_auto_use_lmdb/2` — picks LMDB vs IntMap fact source.
3. `resolve_auto_cache_strategy/2` — picks cursor vs in_memory BFS.
4. `resolve_auto_lmdb_cache_mode/2` — picks cache tier (none / L1 /
   L2 / two_level).
5. **(new)** `resolve_auto_scan_strategy/2` — picks scan strategy
   options if `scan_strategy(auto)` is set.

The new scan-strategy resolver:

- Fires only when `scan_strategy(auto)` is in the (merged) options.
- Reads `tree_cost_function/1`, `expected_query_count/1`,
  `working_set_fraction/1`, `mem_available_bytes/1`, etc. — most of
  which typically come from the algorithm manifest, with caller
  overrides on top.
- Outputs concrete `tree_cost_function/2`, `warm_budget_nodes/1`,
  `stage2_scan_threshold/1`, `tree_retention/1`.

When `scan_strategy(auto)` is absent, the workload runs without
warm-build / snapshot — direct cursor + cache mode picked by the
existing resolvers. This is the default; warm-build is strictly
opt-in (either via a caller option or via an algorithm manifest
that declares it).

### `tree_cost_function` auto-selection

When `scan_strategy(auto)` is set but `tree_cost_function/2` is
not supplied (neither by the caller nor by the manifest), the
resolver picks one by priority:

1. **`semantic_similarity`** — only if `embedding_path(P)` is set
   AND the file exists at codegen time. Requires real
   embeddings; degenerate scoring without them.
2. **`flux`** — only if branching-factor data is available
   (typically from P2's at-scale measurements feeding
   `parent_branching_factor/1` / `child_branching_factor/1`
   options). Without this data, the default decay constants are
   arbitrary; better to fall through.
3. **`hop_distance`** — the safe fallback. Cheap, no
   data-dependent parameters, gives a useful (if coarse) ranking
   for any graph.

Each priority gate has its own `(if-available)` check. The
resolver emits a verbose trace explaining the pick:

```
[WAM-Haskell] scan_strategy(auto): tree_cost_function defaulted to hop_distance
  (no embedding_path; no branching_factor data)
```

Callers who want a specific function just set
`tree_cost_function(flux, [...])` (or whichever) in their
options or manifest; the auto-selection only fires when the slot
is unspecified.

## Edge cases

### Warm produces zero useful nodes

If after warm the heap has no entries whose score exceeds some
floor, snapshot produces an empty `Cache`. Steady-state runs
entirely on cursor fallback. Verbose trace warns:

```
[WAM-Haskell] scan_strategy(auto): warm produced 0 nodes above threshold; running cursor-only
```

### Cost function disagrees with workload

If the workload's actual access pattern doesn't match the cost
function's predictions, miss rate will be high. Measurement —
not architecture — catches this. Recommend exposing the
`scan_strategy_verbose(true)` option (defined in the
implementation plan's cross-cutting telemetry section) to get
hit/miss counters during steady-state for diagnostic runs. Note:
this is a separate option from `cache_strategy_verbose(true)`,
which is the cache-cost-model resolver's existing trace flag —
the two flags toggle different telemetry paths.

### Endpoint disappears mid-warm

Not applicable — warm reads endpoints at start, doesn't track
mid-workload changes. Re-warm-on-shift is out of scope.

### Bounded heap evicts a node we later need

By design. Steady-state misses go to cursor fallback. The
eviction policy is "lowest-score first"; the score *is* the
prediction of relevance; an evicted node is a node the cost
function predicted wouldn't matter. If misses are high, the
prediction was wrong.

## See also

- `SCAN_STRATEGY_PHILOSOPHY.md` — the why.
- `SCAN_STRATEGY_IMPLEMENTATION_PLAN.md` — the how (phased
  rollout).
- `CACHE_COST_MODEL_PHILOSOPHY.md` — the cost model this layer
  builds on; provides `bandwidth_eff` / `latency_eff` /
  `K_cross` formulas.
