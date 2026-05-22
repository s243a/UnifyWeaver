# LMDB Lazy Access: Philosophy

**Status**: Design discussion. Captures the architectural vocabulary for
lazy LMDB access patterns in the WAM targets and how they relate to
workload-segregation and physical-layout strategies.

**Snapshot date**: 2026-05-21.

**Companions**:
- [`WAM_LMDB_LAZY_SPECIFICATION.md`](WAM_LMDB_LAZY_SPECIFICATION.md) —
  cross-target interface specification.
- [`WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md`](WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md) —
  phased rollout for Rust (and reference catalogue for Haskell, which
  already implements much of `cached` mode).

This doc sits inside the broader framing of
[`QUERY_PLAN_RUNTIME_PHILOSOPHY.md`](QUERY_PLAN_RUNTIME_PHILOSOPHY.md)
(precursor for the runtime planner pattern) and
[`SCAN_STRATEGY_PHILOSOPHY.md`](SCAN_STRATEGY_PHILOSOPHY.md) (which
informs how plans get chosen).

## Vocabulary

This triad uses three distinct sets of identifiers — keep them separate:

- **Modes** (runtime materialisation behaviour): `eager`, `lazy`, `cached`.
  These name *what the runtime does on each lookup*. They are the
  three runtime behaviours described in §2. Selectable per kernel
  via the codegen option `lmdb_materialisation(auto|eager|lazy|cached)`.
- **Phases** (project sequencing labels): R7, R8, R9, R10, H1, X.
  These appear in
  [`WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md`](WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md);
  each implements some subset of the modes for some target.
- **Cache tiers** (Haskell-internal, *within* `cached` mode):
  per-HEC L1 cache + sharded L2 cache. These are the existing
  `lmdb_cache_mode(per_hec|sharded|two_level)` configurations and
  are *internal* to the `cached` mode — not to be confused with the
  modes above.

Earlier drafts of this triad named the modes `L0`/`L1`/`L2`, which
collided with the Haskell internal cache-tier names. The rename to
`eager`/`lazy`/`cached` removes the collision. If you find any
lingering `L0`/`L1`/`L2` references to *modes* (rather than to
Haskell's internal cache tiers), they are stale and should be
updated.

## 1. Context — where this came from

The Rust R5/R6 LMDB benchmark arc (PRs #2284 + #2288) measured the
WAM-Rust target against simplewiki (297 k edges) and enwiki (9.93 M
edges). The bench eagerly materialises every demand-set
`(child, parent)` edge into a `Vec<(String, String)>` at startup, then
walks that Vec inside the seed loop.

At simplewiki this wins (~31 ms warm vs Haskell's 226 ms `-N1`,
because the materialisation cost `M ≈ 17 ms` is tiny relative to
Haskell's demand-BFS setup). At enwiki it loses badly — ~148 s
total, dominated by `M ≈ 140 s` worth of cursor lookups and string
allocation before any kernel work runs. Haskell's `resident_cursor`
mode finishes the same workload in 733 ms at `-N4`.

The reversal is a design-choice difference: Rust eager-materialises;
Haskell lazy-streams from LMDB cursors during the kernel walk. Both
are native-compiled and both wrap the same C library. The choice of
*when* to read dominates the choice of *what language* to read with.

This document captures the design vocabulary for lazy LMDB access so
the same trade-off can be discussed consistently across targets, and
so future cost-model resolvers can pick between modes from workload
metadata rather than hand-tuning.

## 2. The lazy/eager spectrum (and the three modes)

Three runtime modes show up in measurements so far. Naming them up
front:

- **`eager`** — eager materialisation. Build an in-memory index of
  every demand-set edge before any kernel iteration runs. The kernel
  walks the index without touching LMDB after startup.
- **`lazy`** — pure lazy. No materialisation. Each kernel walk step
  does an LMDB cursor lookup on demand. No cache layer above the
  LMDB.
- **`cached`** — lazy with cache. Each kernel walk step queries a
  cache first; on miss, does an LMDB cursor lookup and populates
  the cache. Subsequent lookups for the same key hit the cache.

The Haskell `resident_cursor` runtime is **`cached` mode** — it's
lazy by default but the runtime maintains per-HEC L1 + sharded L2
cache tiers above the LMDB cursor reads. (Those cache tiers are
internal to `cached`; see Vocabulary above.) The Haskell `resident`
(IntMap) runtime is **`eager` mode** — eager in-process
materialisation, with the same memory wall the Rust bench has.

The Rust bench today is **`eager`**. We have no `lazy` or `cached`
variant yet, and the cross-target story this doc motivates is to add
both.

### 2.1 Where each mode wins

The naive cost model for one process invocation:

```
eager  wall ≈ M + N × ε     (M = materialisation cost, ε = per-seed kernel)
lazy   wall ≈ N × p         (p = lazy per-seed cost, larger than ε)
cached wall ≈ N × q + θ × p (q = cache-hit cost, θ = cache-miss rate)
```

`lazy wall ≤ cached wall` only when `θ × p ≤ N × (p − q)`, which
simplifies to `θ ≤ N(p−q)/Np` — i.e., when **the cache miss rate is
low enough that the cache machinery itself isn't worth its
overhead**. In a single isolated query that's almost always the case
(θ ≈ 100%, but N ≈ 1 — the cache never gets to amortise). Across
many seeds with graph locality, θ drops quickly and `cached` wins.

The non-obvious result is that **`lazy` only beats `cached` under
specific workload conditions**: high cache-miss rate due to no
locality between queries, OR a workload-segregation guarantee that
means cache hits won't happen anyway. Most real workloads have
*some* locality, so `cached` is the right default.

### 2.2 The picture in one table

| Mode | Wins for | Why |
| --- | --- | --- |
| **`eager`** | Batched workloads, demand-set fits in RAM | Pays `M` once, amortises across many seeds |
| **`lazy`** | Pure one-shot queries, OR segregated batches with provably-disjoint subgraphs | No `M` overhead, no cache machinery overhead |
| **`cached`** | Default for everything else | Bounded per-seed cost, locality benefits, scales past the memory wall |

The cost-model resolver should pick **`cached` by default**,
**`eager` when demand set is small enough and seed count is high
enough** (the crossover in
[`QUERY_PLAN_RUNTIME_PHILOSOPHY.md`](QUERY_PLAN_RUNTIME_PHILOSOPHY.md)
§2), and **`lazy` only when the caller explicitly signals workload
segregation** (see §3).

## 3. Workload segregation: the enabling condition for `lazy`

`lazy` has a real niche, but only when the bench harness can guarantee
that **no two queries in the same process share an LMDB key**. Then
the cache that `cached` would maintain has zero hits, and `lazy`'s
no-cache shape is pure win.

The natural model for this is **node clusters** — group seeds whose
upward walks touch disjoint subgraphs. The category graph of
Wikipedia has multiple top-level roots; if a workload is "compute
effective distances against root R₁ for batch A, then against root
R₂ for batch B," and R₁ / R₂ have non-overlapping descendant
subtrees, then within each batch `lazy` is correct, and across
batches the cache would never have warmed for the new keys anyway.

Two ways to surface this to the cost model:

1. **User-declared segregation**: caller passes a `cluster_id` or
   `workload_segregated(true)` flag. The simplest contract. Risk:
   user lies, `lazy` burns cycles on uncached repeats.
2. **Compiler-inferred segregation**: shape analysis proves that the
   query stream cannot revisit nodes (e.g., the seed set is bounded
   by the demand set, and `max_depth` limits the walk such that
   per-seed visited nodes are disjoint with other seeds). Harder but
   verifiable.

For initial implementation, option 1 is fine — the cost-model
default is `cached`, and `lazy` is opt-in via an explicit annotation.

### 3.1 Where segregation comes from in practice

- **Multi-root workloads**: each root's descendant subtree is one
  cluster. Easy to detect (the existing post-ingest tools already
  pick a root).
- **Topic-partitioned graphs**: if the graph is annotated with
  topic/community labels, queries restricted to one topic are
  segregated by construction.
- **Time-window queries**: in an append-only graph with edges
  tagged by insertion time, a query restricted to a time window
  touches a known subgraph.

These are all *application-level* facts the compiler can't infer
without help. The user-declared signal is the right initial interface.

## 4. Scans vs seeks: a separate axis

Independent of the lazy/eager choice is the question of **how each
LMDB lookup actually reaches data**. LMDB stores entries as B-tree
pages on disk:

- **Seek**: point lookup. Walk the B-tree from root to leaf for one
  key. O(log B) page touches, where B is the branching factor of
  the B-tree (typically ~256 for LMDB pages of 4 KB with int32 keys).
- **Scan**: range read. Position a cursor at a starting key, then
  call `next_key` repeatedly. Each subsequent key in the same B-tree
  leaf page is essentially free (no extra page walk). Each leaf-page
  transition costs one page touch.

If a query needs N edges and they're stored in K B-tree pages, then:

- Seek: N × log(B) page touches (one full descent per edge)
- Scan: K page touches (one descent + K-1 cheap next-key calls)

When N is large and the relevant edges cluster into a few pages,
scan crushes seek. When edges are scattered uniformly, seek and scan
are similar.

### 4.1 Scan-friendliness requires physical layout

Scans only beat seeks when **related keys are physically adjacent in
the B-tree**. LMDB sorts keys lexicographically, so adjacency
depends on the byte-encoding choice:

- **Random integer IDs (current `cl_target_id`)**: keys are
  scattered. A demand set of 14 k nodes occupies pages all over the
  B-tree. Scan offers no advantage.
- **Topologically-sorted IDs**: assign each node an ID equal to its
  BFS order from some seed root. Adjacent IDs are likely adjacent in
  the graph. Scan over an interval reads many edges from few pages.
- **Semantically-clustered IDs**: assign IDs by topic/community.
  Queries restricted to one topic scan a contiguous prefix.

### 4.2 Pre-processing strategies

Three physical-layout strategies, ordered by cost:

| Strategy | Pre-processing cost | Scan locality | Notes |
| --- | --- | --- | --- |
| **Insertion order (current default)** | 0 | Poor | What LMDB gives you from the existing ingest. |
| **Topological sort** | O(V + E), one pass | Good for DAG workloads | Trivial for DAGs; falls back to SCC + topological sort of the SCC DAG for cyclic graphs. |
| **Semantic clustering** | O(V·k) where k = embedding cost | Good if clusters match query patterns | Cluster by embedding distance or pre-computed topic labels. Works best with annotated graphs. |
| **MST sort** | O(E log V) | Provably good | Compute a minimum spanning tree, then DFS-order the nodes. Optimal in the sense that adjacent IDs minimise tree-edge weight; expensive enough that it only pays off for static benchmarks. |

The scan-friendliness story is **orthogonal** to the lazy/eager
choice. `lazy` + scan combines workload segregation with physical
locality — the strongest configuration for one-shot, no-cache
workloads. `cached` + scan still benefits from physical locality on
cache misses. `eager` doesn't care because it pays for
materialisation once anyway.

## 5. How the modes compose

The full space of mode choices:

```
                       physical-layout
                       quality
                            │
                  poor      │      good
                 ─────────────────────────
  eager          │   indifferent (M pays once)
  lazy           │   bad        │  great (workload-seg + scan)
  cached         │   ok         │  great (cache hits + scan misses)
```

The decision tree for the cost-model resolver becomes:

1. *Is demand_set_size × edge_size > available_memory_budget?* → must use lazy form (`lazy` or `cached`).
2. *Does the user declare `workload_segregated`?* → `lazy`.
3. *Otherwise?* → `cached`.
4. *Is physical layout known to be scan-friendly?* → enable scan-mode within the chosen mode.

The existing cost-model resolvers (`cache_strategy(auto)`,
`lmdb_cache_mode(auto)`, `resident_auto`) already cover parts of
this. The `lazy` / `cached` distinction is what's new here; scan-mode
is a related axis that would be added per-target. See
[`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md)
and [`COST_FUNCTION_PHILOSOPHY.md`](COST_FUNCTION_PHILOSOPHY.md)
for the cost-resolver pattern these new resolvers slot into. (That
pattern is the one we sometimes call "the C# cost analysis", since
the C# planner already does similar runtime mode selection — but
the pattern itself is target-agnostic and lives in
`src/unifyweaver/core/cost_model.pl`.)

## 6. Cross-language patterns this maps to

`lazy` and `cached` are general patterns. The iterator-based lookup
interface maps cleanly across targets:

| Target | Idiomatic lazy lookup return type | Cache layer notes |
| --- | --- | --- |
| Haskell | `[Int]` (lazy list) or `ConduitT () Int IO ()` | Already implemented (`cached`); per-HEC L1 + sharded L2 in `lmdb_cache_mode` |
| Rust | `impl Iterator<Item=i32> + 'a` | Closure-based foreign predicate; cache via `dashmap` or sharded `HashMap` |
| Go | `<-chan int` (channel) | Pipeline-chaining already has channel mode; same pattern |
| C# | `IEnumerable<int>` (LINQ) | Existing C# planner already does this for relations |
| Python | generator (`def parents(child): yield ...`) | Existing federated_query already uses `yield` for partial results |
| Elixir | `Stream` lazy enumeration | `generator_mode(true)` option already exists |

So the *interface* is the same shape everywhere; the implementation
just chooses each language's native iterator equivalent. The cache
layer composes on top via a Decorator pattern: a `cached`
implementation wraps a `lazy` implementation via the same interface.

## 7. What this rules out

- **Picking lazy vs eager at the language level.** The mode is
  workload-dependent, not language-dependent. Any target can
  implement any mode.
- **A single "best" mode that wins everywhere.** Each mode has a
  niche; the cost model has to choose.
- **Hand-tuning per benchmark.** The auto-resolvers exist precisely
  so that "which mode?" is determined by workload metadata, not
  by which mode the experimenter happened to remember to set. The
  `lmdb_materialisation(auto)` value triggers the resolver; explicit
  values (`eager`/`lazy`/`cached`) override it.
- **Treating scan-mode as a degree of laziness.** It's a separate
  axis. Any mode can use seeks or scans; the choice is determined
  by physical layout quality.

## 8. Open questions

1. **What is the workload-segregation contract?** Just a boolean
   `workload_segregated(true)` flag, or a richer cluster annotation?
   The richer version lets the resolver size the cache to one
   cluster at a time.
2. **Should scan-mode be a target-level option or a per-predicate
   option?** Target-level is simpler; per-predicate matches the
   layout-quality question more accurately.
3. **How does this compose with the runtime planner pattern
   ([`QUERY_PLAN_RUNTIME_PHILOSOPHY.md`](QUERY_PLAN_RUNTIME_PHILOSOPHY.md))?**
   The planner is the right place to decide `eager`/`lazy`/`cached`
   + seek/scan per query rather than per process. Question: is the
   decision cheap enough to do per query, or does it need to be
   amortised?
4. **MST sort + LMDB rewrite for very large graphs**: is the
   pre-processing cost ever worth it for a one-off benchmark? Or
   only for production query workloads that hit the same DB
   thousands of times?
5. **Cache invalidation under graph mutation**: the current Haskell
   `cached` mode assumes a static LMDB. Mutating workloads need a
   versioning story.
6. **Cross-target benchmark methodology**: when we measure
   `lazy`/`cached`/scan-mode against `eager`, we need a workload
   spec that exercises each mode's niche. The existing matrix-bench
   is `eager`-shaped; it'd over-amortise.

## 9. References

- `docs/design/QUERY_PLAN_RUNTIME_PHILOSOPHY.md` — broader runtime
  planner pattern that this fits inside.
- `docs/design/SCAN_STRATEGY_PHILOSOPHY.md` + `_SPECIFICATION.md` +
  `_IMPLEMENTATION_PLAN.md` — scan-strategy triad (warm-build core,
  cost-function-driven plan selection).
- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md` — cost-model framework
  this work feeds into.
- `docs/design/COST_FUNCTION_PHILOSOPHY.md` — Green's-function and
  flux cost-functions; the lazy-vs-eager decision is one of the
  inputs.
- `docs/design/WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md` — LMDB
  layout this lazy work reads from.
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase L#7–9 — Haskell
  `cached`-mode measurements at simplewiki and enwiki.
- `examples/more/graph/effect_dist/haskell/gen/reports/effective_distance_haskell_vs_rust.md` —
  Haskell-vs-Rust report that motivated this design.
