# Boundary Distribution Splice — Complexity Headroom (P3, 2026-06-15)

P3 of the boundary-distribution-cache plan
(`WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md`): before building the live kernel
(P2), measure whether the boundary splice actually buys anything **on top of the
already-fast edge cache** — and *why*.

## The key distinction

The boundary distribution is **primarily a complexity reduction, and only
secondarily a cache.** The `effective_distance` kernel sums over **all** paths
seed→root — exponentially many in a graph with diamonds, capped at the budget.
The edge cache (PRs #3127–#3187) removes LMDB-seek cost but **cannot avoid that
enumeration**: it still walks every path. A boundary distribution caches the
path-**length histogram** of the shared upper cone, which represents the
exponentially-many paths *compactly*, so a query that reaches the boundary
**splices** (O(budget)) instead of re-enumerating. Reusing that precomputed
histogram across seeds is the secondary caching/amortization layer.

## Measurement

`examples/benchmark/boundary_splice_complexity_bench.rs` (std-only,
`rustc -O … && ./a.out`). A dense core near the root (diamonds → exponentially
many paths to root) behind a thin **boundary cut**, with a sparse periphery below
where 500 seeds attach only to boundary nodes (so every seed→root path crosses
the cut). Budget 10, weighting `weighted_power(N=2)`.

- **Method A** — full path enumeration seed→root (the kernel's behaviour; parent
  lookups are HashMap hits = the warm edge cache).
- **Method B** — walk seed→boundary, splice the cached suffix histogram, stop.

| core size | A: full enumeration | B: boundary splice | speedup | aggregates equal |
|-----------|---------------------|--------------------|---------|------------------|
| 120 |  71 ms | 0.21 ms | **332×** | yes |
| 160 | 123 ms | 0.30 ms | **414×** | yes |
| 200 | 124 ms | 0.23 ms | **535×** | yes |
| 240 | 153 ms | 0.22 ms | **687×** | yes |

## Reading it

- **It is a complexity reduction, not a constant factor.** Method A *grows* with
  core density/size (more paths to enumerate); method B is **flat** (the histogram
  is the same size regardless of how many paths it represents, and the splice is
  O(budget)). The speedup therefore *grows* with scale — 332× → 687× here — which
  is the signature of a complexity-class improvement, not a cache constant.
- **The edge cache cannot do this.** Method A *is* the edge-cached path (lookups
  are warm HashMap hits); it is still 300–700× slower because the cost is the
  enumeration itself, which the edge cache does not touch. This is the headroom
  the boundary distribution unlocks *on top of* the edge cache — the question P3
  was meant to answer, answered strongly in favour of building P2.
- **Correctness preserved.** The spliced aggregate equals the full-enumeration
  aggregate exactly on every row (the P1 identity, at scale).
- **Caching is the secondary lever.** Here the boundary suffix histograms are
  computed once (≈40 boundary nodes) and reused by 500 seeds; that amortization is
  what makes the splice's flat per-seed cost pay for the one-time precompute.

## Caveats

- **Synthetic, engineered to have a dense shared upper cone.** The *magnitude*
  depends on the upper-cone path multiplicity; the *mechanism* (compact histogram
  vs exponential enumeration) is structural. Real category graphs do have dense
  hub structure near the root (cf. the cache-scaling reports' 98–99% hub reuse),
  so a real win is expected, but its size needs the P2 real-graph measurement.
- This measures the algorithmic headroom, not the integrated kernel. P2 wires the
  splice into the live `category_ancestor_boundary` kernel and measures it on the
  LMDB fixtures end-to-end.

## Consequence

P3 justifies P2: the boundary distribution is a genuine complexity reduction that
the edge cache cannot provide, with the win growing with scale. Proceed to wire
the live kernel (gated, off by default).
