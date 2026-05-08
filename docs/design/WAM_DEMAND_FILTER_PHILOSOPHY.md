# Demand Filter: Philosophy

This doc captures the design exploration around how the WAM-Haskell
runtime decides which seeds to spark, which edges to cache, and which
work to deprioritise. It walks through alternatives we considered,
explains why we landed where we did, and flags the questions that
should drive future revisions.

Companion docs:
- `WAM_DEMAND_FILTER_SPECIFICATION.md` — the technical shape we ship.
- `WAM_DEMAND_FILTER_IMPLEMENTATION_PLAN.md` — phased rollout.
- `WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md` — the storage layer this
  composes with; demand BFS walks LMDB cursors.

## What the demand set actually is

For the canonical workload (`category_ancestor` style: each seed
walks `category_parent` edges UP toward a `root`), a seed S can reach
root iff root is in S's ancestors — equivalently, S is in root's
**descendants** in tree terms. The demand set is built by **reverse**
BFS from root, and is therefore "everything below root in the
hierarchy."

Common confusion worth nailing here:

- *Ancestors of root* = the few super-categories above root (e.g.
  "Knowledge", "Things"). Tiny. Not what we want.
- *Descendants of root* = the categories that sit under root (e.g.
  for root=Physics: "Quantum_Mechanics", "Optics", "Lenses", ...).
  Potentially huge. **This is the demand set.**

The size of the demand set is therefore determined by root
generality, not by the size of the universe directly:

| Root example | Approx descendants on enwiki |
|---|---:|
| `0s_beginnings` (very specific) | ~10 |
| `Deaths_by_year` (mid) | ~1,000 |
| `Physics` (broad) | ~50k–500k |
| `Science` (very broad) | ~1M–3M |
| `Things` (top-of-tree) | ~7M ≈ universe |

So "the demand set is small" is workload-dependent, not a property of
the algorithm. A correct design has to handle the full range.

## Two artifacts that the current TSV path collapses together

The current code computes:

```haskell
!demandSet         = computeDemandSet parentsIndexInterned rootId
!filteredParents   = filterByDemand demandSet parentsIndexInterned
!filteredSeedCats  = filter (\c -> IS.member (iAtom c) demandSet) seedCats
```

These are **two different artifacts** that the existing path
implicitly couples:

| Artifact | Type | Role | Memory cost |
|---|---|---|---|
| `demandSet` | `IntSet` | Pre-filter `seedCats` before parMap | O(\|descendants\|), small Patricia trie |
| `filteredParents` | `IntMap [Int]` | Edge map the kernel walks | O(\|descendant edges\|), big when descendants are big |

These have different roles and different costs. Keeping them coupled
forces "if filtering is on, both are built; if off, neither is." In
the LMDB-resident world they decouple naturally:

- `demandSet` stays — small set, used for seed pre-filter, prevents
  the spark-fanout regression PR #1882 just fixed.
- `filteredParents` goes away — the kernel reads edges via LMDB
  cursors, optionally backed by a configured cache layer
  (`per_hec` / `sharded` / `two_level`). The cache *is* the
  in-memory edge tier, sized by mode rather than by the demand set.

## Hop-limit: the simple sound filter

The kernel already has `max_depth=10` (or whatever the user sets). A
seed more than 10 forward-hops from root **cannot** succeed
regardless of any filter. So bounding the demand BFS to `max_depth`
hops gives a **provably complete filter** — no false negatives, no
approximation:

```
demand set = { n : reverse-BFS distance from root to n ≤ max_depth }
```

Properties:

- Sound: every seed reachable within the kernel's actual reach is in
  the set.
- Cheap: the BFS only walks `max_depth` levels, naturally bounded.
- Aligned: it uses the kernel's existing depth knob, no new tuning.

The downside: **it doesn't account for routing density**. In a
small-world graph like Wikipedia categories, a 5-hop path through
high-degree hubs is fundamentally different from a 5-hop path
through a thin chain. Hop-limit treats them the same.

## The small-world wrinkle

Wikipedia's category graph is small-world: average shortest path is
~4-5 hops, dominated by a handful of high-degree hubs that
short-circuit otherwise distant subtrees. Statistical character:

- Average reachability distance L* ~ log(N), not Θ(\|tree\|).
- A node with 10,000 children is a routing hub: any path through it
  fans out into 10,000 alternatives.
- Two nodes at the same hop-distance from root can have very
  different "effective connectedness" — one funneling through hubs,
  the other walking thin chains.

For a filter that wants to capture "the part of the graph
meaningfully connected to root," hop-limit conflates these regimes.
We need something that compensates for routing density.

## Flux density as the principled alternative

Treat the BFS as a random walk from root, choosing reverse edges
uniformly at each step. The probability of landing at node n after
some hops is the **flux** at n:

```
flux(root) = 1
flux(n)    = sum over reverse-edges (n, parent_of_n) of
                 flux(parent_of_n) / out_degree(parent_of_n)
```

This is personalized PageRank from root without teleport. Properties:

- A node reachable by many short, low-branching paths has high flux.
- A node reachable by one long high-branching path has low flux.
- Flux is **intrinsic** to network structure — it captures
  "closeness to root" weighted by routing density.
- It's bounded: total flux mass sums to 1 by construction.

Flux gives us a *gradient* over nodes rather than a binary in/out
classification. That gradient is what makes the design compose well
with parallel scheduling (see "Top-K plus sorting" below).

## What "size" knob makes sense for the user

We considered two parametrisations of "how big should the demand
set be":

- **Target flux mass** — capture nodes whose cumulative flux ≥ M
  (e.g. M = 1−1/e ≈ 0.632, by analogy with EE characteristic time
  / network-science mixing time).
- **Target network fraction** — capture top-K nodes by flux where
  K = f × N (e.g. f = 0.05).

We considered the 1/e mass cutoff because it's parameter-free and
falls out of the natural scale of an exponential decay. But:

- The user can't predict the resulting **count** without knowing the
  flux distribution, which depends on the workload.
- Memory budgets are denominated in "elements" or "bytes," not
  "mass."
- "Capture 5% of N" maps directly to a known cache footprint.

So the user-facing knob is **target network fraction**, with flux
ordering as the *priority* (which K to keep), not the *budget* (how
many). The 1/e idea was a useful theoretical anchor but the wrong
shape for the API.

## Top-K plus sorted sparks: the cleanest formulation

The realisation that simplifies the whole design: there are
**two different K's** with two different meanings.

| Mechanism | Cap | Purpose |
|---|---|---|
| **Cache top-K** | Bounded by RAM (`f × N` or absolute byte budget) | Holds highest-flux edges; warmed during BFS |
| **Spark ordering** | Unbounded — sort *all* reachable seeds by flux | parMap creates high-flux sparks first; low-flux fill idle capacity |

The cache K is hard-capped by hardware. The spark K is unbounded —
every seed with `flux > 0` still runs, just in priority order. No
false negatives, no quality cliff, graceful degradation under any
configuration.

Implementation is trivial:

```haskell
let !filteredSeedCats   = filter (\c -> flux c > 0) seedCats
    !sortedFilteredSeeds = sortBy (comparing (Down . flux)) filteredSeedCats
let !seedResultsForced   = parMap rdeepseq queryBody sortedFilteredSeeds
```

Two-line change vs. the current code. `parMap rdeepseq` is lazy in
its input; sparks fire as the spine is forced; sorted-by-flux input
means high-priority sparks reach idle workers first. GHC's
work-stealing then pulls low-flux sparks onto idle threads. No
custom scheduler, no priority queue.

The `flux > 0` filter remains the **hard sound filter** (a seed with
flux 0 cannot reach root, definitionally). Sorting is the **soft
priority** within the surviving set.

## What about "ancestors are smaller than descendants"

A reasonable hope: per-node ancestor sets are tiny (depth × branching
≈ 5-50 nodes) while descendant sets can be enormous. Why not compute
the demand filter as the **union of seed ancestors** instead of root
descendants?

The math:

| Approach | BFSes computed | Per-set size | Total work |
|---|---:|---:|---:|
| Descendants of root (current) | **1** | O(\|root subtree\|) | O(\|subtree\|) |
| Ancestors of each seed | **N seeds** | O(depth × branching) | O(N × depth) |

For a single shared filter, **one BFS from root** is cheaper than
**N small BFSes from seeds**, even though each individual seed's
ancestor walk is tiny. The filter inherently considers all seeds at
once, so the shared computation wins.

Bidirectional BFS (walk forward from seeds AND backward from root,
intersect) is a classic √N speedup for shortest-path queries, but
for our filter — a binary "can-reach?" predicate shared across many
seeds — the one-shot reverse BFS is already the bidirectional
algorithm's cheap half.

So the demand set stays as descendants of root, and we manage size
via flux-priority + cap, not via a different BFS direction.

## Statistical anchors

Useful fixed points from network science / EE:

- **Logarithmic complexity**: small-world networks have characteristic
  path length L* ~ log(N). On enwiki, that's ~14-16 (natural log) or
  empirically ~4-5 because hubs short-circuit chains.
- **Mixing time**: O(log N) random-walk steps to come within 1/e of
  the stationary distribution. Same scale as L*.
- **1/e (≈ 0.632) characteristic decay**: the "natural scale" of an
  exponential. The cumulative flux mass after one mixing time is
  ≈ 1 − 1/e ≈ 63%.
- **Effective diameter** (Leskovec et al.): 90th-percentile shortest
  path. Convention from network-science papers but the percentile
  is arbitrary.

These don't directly drive the user-facing knob (we settled on
network fraction), but they inform sensible defaults and explain why
the runtime should *expect* the demand set to be reachable with very
few hops on real workloads — and why the worst case is the entire
universe when `root` is the tree root.

## Configurable, not hard-coded

The right framing is "the demand filter is a strategy, not a
mechanism." Different workloads want different filters:

- A query asking "which categories strongly connect to Physics"
  wants flux-weighted reachability — most-central first, prune the
  tail.
- A query asking "which categories can reach Physics in any way"
  wants strict hop-bounded reachability — no false negatives.
- A small-fixture dev workflow wants no filter — everything runs,
  cache absorbs the overhead.

So the surface is declarative:

```prolog
:- declare_demand_filter(hop_limit, [max_hops(10)]).
:- declare_demand_filter(flux, [target_fraction(0.05), sort_sparks(true)]).
:- declare_demand_filter(none).
```

The runtime's job is to honour the declared strategy. No magic
defaults that vary by workload size; deployments pick what they
want.

## What we think is best

For Phase 2 (the immediate next implementation step):

1. Ship `hop_limit` as the default — it's sound, cheap, and aligned
   with the kernel's existing `max_depth`.
2. Build the dispatch machinery so adding strategies is purely
   additive (`DemandFilterSpec` sum type, `runDemandBFS` consumer).
3. Stub the `Flux` constructor so the codegen can emit it; runtime
   panics if used until Phase 2.5 implements it.

For Phase 2.5:

4. Implement `Flux` properly — priority-queue BFS, top-K cache
   warming, sorted-by-flux parMap input.
5. Default `target_fraction(0.05)` and `sort_sparks(true)` when
   `Flux` is selected. These are pragmatic, not principled — we
   revise after measurements.

For later phases:

6. Hybrid (`HopLimit` AND `Flux`) for users who want both bounds.
7. Adaptive: measure hit-rate during the first few queries and tune
   the floor / target_fraction. Worth doing only if measurements
   show the hand-tuned defaults need adjusting.

## Open questions and future inspiration

- **Pre-computed demand sets per known root** — for production
  deployments where the same root is queried repeatedly, the
  ingester could write per-root demand sets to disk. Trades disk for
  startup time. Not Phase 2 scope.
- **Parallel BFS** — workers fan out by frontier slice, each owning
  a thread-local LMDB cursor. ~Ncores speedup on broad roots.
- **Reverse-edge sub-db** — option in the ingester to write a
  pre-computed reverse adjacency, avoiding the in-memory build at
  startup. Small additive change to the Phase 1 ingester.
- **Information-theoretic floors** — replace the network-fraction
  heuristic with an entropy-based cutoff (capture the K nodes that
  contribute most to the flux distribution's entropy). More
  principled but heavier math.
- **Adaptive memory budget** — measure RSS during BFS, trim
  lowest-flux frontier when the budget is exceeded. No user knob
  needed, just a cap.
- **Cross-root flux caching** — flux for root R₁ partially overlaps
  with flux for root R₂ if R₁ and R₂ share descendants. A flux LRU
  across root queries could amortise the BFS.

These are filed for future exploration. Phase 2 ships the simple
sound default and the dispatch shape; everything above is additive.

## Refs

- `WAM_DEMAND_FILTER_SPECIFICATION.md` — the chosen API, types, knob surface.
- `WAM_DEMAND_FILTER_IMPLEMENTATION_PLAN.md` — phased rollout.
- `WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md` — composes with demand
  filtering at the BFS layer (cursors over LMDB).
- `WAM_PERF_OPTIMIZATION_LOG.md` Phase L appendix #4 — the
  parallelisation regression that motivated keeping the seed pre-filter
  no matter what.
- `templates/targets/haskell_wam/lmdb_fact_source.hs.mustache` — the
  cache mode infrastructure (`per_hec`, `sharded`, `two_level`).
