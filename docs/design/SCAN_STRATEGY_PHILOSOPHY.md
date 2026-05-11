# Scan Strategy — Philosophy

## What this is

A design for an **adaptive query strategy** that picks between
cursor-driven seeks and full-DB scans dynamically, builds a
cost-ranked tree of relevant nodes during a warm phase, then
snapshots the tree to a flat cache for steady-state lookups.

This is the next layer above the cost model that landed in Phase
2c (`docs/design/CACHE_COST_MODEL_PHILOSOPHY.md`). The cost model
makes a *one-shot* decision at codegen time: sort or scan, this
cache tier or that. The scan-strategy work generalises that to a
*staged, iterative* decision: start with cursor seeks while the
frontier is small, switch to a sequential scan pass that absorbs
the frontier's edge data when the frontier grows past the
seek-vs-scan crossover, optionally repeat as the tree grows.

The deliverable, in one sentence: a runtime that **warms a tree
of cost-ranked candidates, snapshots it into a fast lookup cache,
and optionally retains the ranking for downstream consumers like
parallelisation**.

## Why now

Two pressures motivate this work:

1. **The cost model's static decisions don't generalise.** Phase L#11
   showed the model picking `in_memory` at simplewiki scale with
   the wrong `working_set_fraction` default; Phase L#13 added the
   memory-budget guard because the cost model's "scan" abstraction
   said nothing about working-set footprint. Both fixes were
   band-aids on a structurally static decision. A staged strategy
   would let the runtime observe its own frontier size and switch
   modes accordingly.

2. **Empirically, cursor wins everywhere we've measured for BFS
   workloads.** Phase L#7–9: simplewiki 1.96× faster cursor over
   in_memory; enwiki cursor-only because in_memory can't allocate;
   1k_cats roughly parity but cursor still nominally faster. The
   matrix-bench `resident_auto` mode now picks cursor at every
   scale. **The interesting question is no longer cursor-vs-in_memory
   but cursor-vs-(cursor-then-scan)** — at what frontier size does
   absorbing the remaining workload in one scan beat continuing to
   seek edge-by-edge?

## The core insight

Three phases, three data structures:

| phase | structure | operations | optimised for |
|---|---|---|---|
| **warm-build** | sorted heap / priority queue, scored by a cost function | insert with score, evict lowest, peek top-K | O(log N) construction + ranking |
| **snapshot** | (one-shot) walk the heap, materialise an IntMap | iterate, build | one-time conversion |
| **steady-state** | IntMap (the cache) + optional flux-ranked view | O(1) `lookup`; iterate in order | hot-path throughput |

The split between warm and steady-state matters because they have
genuinely different access patterns:

- Warm phase: we're *exploring*. We don't know which 1–10% of the
  DB is worth keeping. We need ranked insertion and bounded-capacity
  eviction. O(log N) is fine because each insertion does meaningful
  work (a cursor seek, a flux update).
- Steady-state: we're *serving*. The set of relevant nodes is
  frozen. We need O(1) lookup by key. Any ranking we maintained
  during warm is either retained for downstream consumers
  (parallelisation) or discarded as dead weight.

Treating these as one structure forces a tradeoff: O(1)
hashtable lookup competes with O(log N) heap-ordered eviction.
Separating them lets each phase use the right shape.

## Architectural alternatives we considered

### 1. Tree-as-cache (single structure)

Keep one data structure across all phases — a hybrid hashtable +
parallel heap, with the heap providing eviction order. Lookups go
through the hashtable; eviction through the heap. The cache *is*
the tree.

**Why we didn't pick this**: After warm completes, we never need
the eviction ordering again. The heap becomes dead weight until
the next eviction event — and steady-state caches don't evict
during normal operation. So we'd be paying memory + cache-coherence
cost for a structure we don't use.

The hybrid is still the right call *during* warm. Just not after.

### 2. Tree + L2 spillover

Keep the tree for ranking and a separate L2 hashtable for
overflow. Tree miss + L2 hit = serve from L2; tree miss + L2 miss
= cursor + populate L2.

**Why we didn't pick this for the first cut**: Two parallel
structures means two lookups per query in the miss case. The
design we picked already gets the same benefit cheaply — the
overwrite-on-collision cache memoises misses, so a node that
the cost function didn't predict but the workload queries twice
becomes a hit on the second access. That's effectively a
single-level spillover, just inside the same hashtable rather
than in a parallel structure.

If miss-rate measurements show systematic mis-prediction (not
just random outliers), we'd fix the cost function or warm budget
rather than add a second structure.

### 3. Single-pass scan-only

Forget cursor entirely; one scan pass at startup, materialise a
flat cache. No staging, no priority queue.

**Why we didn't pick this**: Already loses at every scale we've
measured. Phase M1.b: even at 100% selection of a 10k subset of
simplewiki, sorted seeks beat scan 2.8×. Scan only wins when we'd
touch a large fraction of the DB *and* the DB exceeds free RAM
(cold-disk regime). Most workloads at category scale aren't there.

### 4. Hardcoded flux (or any single cost function)

Pick one ranking function and hardcode it. Simpler API.

**Why we didn't pick this**: We genuinely don't know which cost
function works best. Flux makes sense for graph-shaped workloads
where branching-factor decay tracks relevance. Hop distance is
cheap and works for symmetric reachability problems. Semantic
similarity needs embeddings but may dominate when those exist.
Hardcoding forecloses A/B comparison; pluggable lets the workload
author choose and lets the implementation evolve.

## Separating the algorithm from its optimizations

A meta-concern that surfaced during the design discussion: the
*algorithm* (the logical specification of what is being computed
— e.g. effective_distance — closer to the Datalog/SQL sense of
"a query" than the algorithms-textbook sense of "a step-by-step
procedure") and the *optimizations* (how to compile it —
cost-model knobs, cache mode, scan strategy, demand-filter spec)
have different owners and lifecycles. Bundling them in a single
bench-harness declaration forces every caller to know every
optimization. Splitting them lets the algorithm be declared once
with its best-known optimization profile, with callers free to
override on a per-knob basis. The terminology note in
`ALGORITHM_MANIFEST_SPECIFICATION.md` covers this explicitly.

We capture this split in a separate small specification
(`ALGORITHM_MANIFEST_SPECIFICATION.md`). The scan-strategy work
assumes the manifest exists — `tree_cost_function/2`,
`tree_retention/1`, `scan_strategy/1` are all options that live in
a manifest in practice. None of the scan-strategy mechanics depend
on the manifest abstraction internally; the manifest is a
deployment-time convenience layer.

## What we picked

**Warm-build with a heap + cost-function strategy slot, snapshot
to a hashtable cache (overwrite-on-collision), optional retention
of the flux-ranked view for parallelisation consumers, optional
live tree updates for adaptive algorithms.**

Specifically:

- **Cost function** is a strategy parameter (`flux`,
  `hop_distance`, `semantic_similarity`, future others). Each has
  its own parameter list (decay constants, max hops, embedding
  dim, etc.).
- **Iterations** count is unified across access patterns: 1 = one
  round of score relaxation (1 hop deep for cursor mode; 1 full
  pass for scan mode). N iterations approximate the converged
  score; pick a point on the cost/accuracy curve.
- **Flux specifically** treats parent and child legs separately
  because the branching factors are asymmetric (Wikipedia
  categories: few parents per node, many children). Total flux is
  the merge of two single-direction scores.
- **Warm budget** targets 1–10% of the DB by node count. Below 1%
  the tree is too small to be useful; above 10% we've defeated the
  point of being selective.
- **Cache misses** during steady-state fall back to LMDB cursor
  and **memoise into the cache** (overwrite-on-collision; same
  mechanism as the existing L1/L2 hashtables). The snapshot
  provides the *initial state* of a real cache, not a frozen
  lookup table. There's no separate spillover structure — the
  cache itself plays that role via miss memoisation.
- **Tree retention** has three modes (default `discard`):
  - `discard` — GC the tree after snapshot. Minimum memory.
  - `snapshot_only` — keep the flux-sorted view (a frozen list).
    Used by spark routing and other consumers that need a static
    ranked view.
  - `live` — keep the full heap + parallel index. Supports
    post-warm updates: query-history-driven re-ranking, adaptive
    cost-function switching, hot-region expansion. No current
    consumer implements `live`-mode algorithms, but the data
    structures we pick during warm support it without
    architectural changes, so the door stays open.

## Why retention matters: spark routing

The retained flux-ranked view is precisely what MoE-style spark
routing needs to break the per-HEC L1 cache-duplication problem.
Memory note: "MoE-style spark routing would unlock L1; until then
sharded is the right default" because parMap has no region
affinity. With flux-sorted seeds:

1. Partition seeds into N capability-local chunks by flux score
   (high-flux chunk to capability 0, next to capability 1, etc.).
2. Each thread's L1 cache accumulates non-overlapping coverage
   instead of duplicating hot edges.
3. Sharded L2 becomes a backup for cross-region hits, not the
   primary mechanism.

Retention is the bridge between "we built a tree for cache
warming" and "we have a tree that's useful for scheduling
parallelism". Same artefact, two consumers.

## When this matters

Same regime checklist as the cost model itself:

1. **Article-level data ingestion** — enwiki article texts
   (~250 GB uncompressed) will need adaptive cursor→scan
   transitions because pure cursor wouldn't visit the
   working set fast enough.
2. **Multi-fixture aggregation** — categories + page metadata +
   revisions + links; total working set may exceed even hot-regime
   RAM.
3. **Repeated queries from a hot root subset** — Phase M3 said
   warming doesn't pay off for random-BFS, but it *would* pay off
   for repeated queries against a small region. Warming a tree
   per region would capture that; we don't address sub-region
   partitioning in the current spec — it's scoped as future work
   when a workload that needs it lands.
4. **Workloads where parallelisation matters** — even without the
   cold regime, the retained flux ranking unlocks
   region-affinity-aware spark routing.

At today's fixture scale (simplewiki at 297k, enwiki at 9.9M),
this work is groundwork. Like the cost model itself.

## What this isn't doing

- **Not a runtime auto-tuner.** Warm parameters
  (cost function, iterations, budget) are picked at codegen or
  workload-author time, not at runtime via measurement.
- **Not a re-warming mechanism.** Once steady-state starts, the
  cache is frozen until the workload completes. If query patterns
  shift mid-workload, that's a follow-up problem.
- **Not a correctness device.** Misses still return correct
  results via LMDB cursor fallback. The strategy affects
  performance, not semantics.
- **Not a replacement for the cost model.** The model still picks
  cursor-vs-in_memory and cache tier; scan strategies add an
  adaptive runtime layer on top.

## See also

- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md` — the cost model
  this layer sits on top of.
- `docs/design/ALGORITHM_MANIFEST_SPECIFICATION.md` — the
  declarative split between algorithm and optimization manifest;
  prerequisite for the per-algorithm optimization profile this
  scan-strategy work assumes.
- `docs/design/SCAN_STRATEGY_SPECIFICATION.md` — concrete
  formulas, data structures, option list.
- `docs/design/SCAN_STRATEGY_IMPLEMENTATION_PLAN.md` — phased
  rollout with deliverables and dependencies.
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` — Phase M / Phase L
  measurements that motivate the regimes.
