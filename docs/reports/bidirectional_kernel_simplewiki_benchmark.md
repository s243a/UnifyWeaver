# Bidirectional Kernel Benchmark Report

**Date**: 2026-05-26
**Fixture**: Simple English Wikipedia (simplewiki), 10k article scale
**Data**: 25,227 category_parent edges, 7,811 parent keys, 7,417 child keys, 18,126 interned strings, max depth 20
**Root**: Physics (id=5556)
**Seeds**: 50 categories at depth ~4 from root (e.g., Ice, Soviet_Union, Kingdom_of_France, History_of_London)
**Kernel**: DFS with path-cost pruning, parentCost=1.0, budget=15.0
**Pruning**: A*-style lower-bound pruning using precomputed minimum
parent-hop distance from each node to root. No expansion cap.

## 1. childCost sweep (A* pruned, complete search)

| childCost | Total paths | Mixed (child hops) | Upward-only | Expansions (k) | Time |
|-----------|------------|-------------------|-------------|---------------|------|
| 100.0 | 513 | 0 | 513 | 3 | 10ms |
| 10.0 | 933 | 420 | 513 | 6 | 10ms |
| 5.0 | 11,035 | 10,522 | 513 | 54 | 90ms |
| **3.0** | **558,017** | **557,504** | **513** | **2,804** | **1.5s** |
| 2.5 | 1,636,160 | 1,635,647 | 513 | 8,917 | 5s |
| 2.0 | 21,741,496 | 21,740,983 | 513 | 119,874 | 77s |
| 1.5 | 341,891,463 | 341,890,950 | 513 | 1,978,827 | 22min |

Each unit decrease in childCost multiplies the path count by roughly
15x, consistent with the empirical child branching factor of ~15 in
Wikipedia's category graph. This dimensionality is ~3x higher than
the ~5 estimated by graph spectral methods, likely because spectral
methods measure smooth connectivity while path enumeration sees the
full combinatorial branching including redundant routes.

## 2. Distance metrics

The uniform power-law metric `d = (Σ (hops+1)^(-n))^(-1/n)` does not
converge as childCost decreases — each new child path adds equal
weight, driving d_eff toward zero. A direction-weighted metric that
normalizes by path weight produces convergent, meaningful distances.

### Direction-weighted metric

Each path is weighted by direction:

```
w(path) = (1/D)^N * (1/(b*D))^M

where:
  N = parent hops, M = child hops
  D = graph dimensionality (~5)
  b = child branching factor (~15)
```

Two normalized metrics using these weights:

- **Weighted average**: `d = Σ(w_i * h_i) / Σ(w_i)`
- **Weighted power-mean**: `d = (Σ(w_i * (h+1)^(-n)) / Σ(w_i))^(-1/n)`

### Convergence comparison

| childCost | d_uniform | Δ% | d_weighted_avg | Δ% | d_weighted_pow | Δ% |
|-----------|----------|-----|---------------|-----|---------------|-----|
| 100 (up-only) | 3.04 | — | 4.23 | — | 5.17 | — |
| 10 | 2.02 | -33% | 4.26 | +0.9% | 5.20 | +0.5% |
| 5 | 1.01 | -50% | 4.30 | +0.7% | 5.22 | +0.3% |
| 3 | 0.29 | -72% | **4.31** | **+0.3%** | **5.22** | **+0.1%** |

The uniform metric keeps dropping (-33%, -50%, -72%) and never
stabilizes. The weighted metrics converge: +0.9%, +0.7%, +0.3%
(weighted average) and +0.5%, +0.3%, +0.1% (weighted power-mean).
By childCost=3, the weighted power-mean changes by only 0.1% —
effectively converged.

### Interpretation

The weighted metrics converge to ~4.3 hops (weighted average) and
~5.2 hops (weighted power-mean). These are meaningful: "this
category is about 4-5 hops from Physics when you account for
bidirectional connectivity, weighted by direction."

The slight increase from upward-only (3.04) to bidirectional (4.31)
is correct: child-heavy paths are longer in hops, and even
down-weighted they pull the average up. The weighted metric says
"the additional child paths don't make things closer — they reveal
that the true connectivity-weighted distance is somewhat longer
than the shortest upward path alone suggests."

### Per-category detail (childCost=3)

| Category | Paths | Mixed | d_uniform | d_weighted_avg | d_weighted_pow | d_up_only |
|----------|-------|-------|-----------|---------------|---------------|-----------|
| Ice | 82 | 61 | 1.17 | 4.20 | 5.15 | 2.20 |
| People_by_former_country | 1,197 | 1,195 | 0.32 | 4.19 | 5.14 | 3.84 |
| Soviet_Union | 1,264 | 1,258 | 0.32 | 4.28 | 5.20 | 2.76 |
| Kingdom_of_France | 31,757 | 31,715 | 0.06 | 4.63 | 5.43 | 1.46 |
| Former_countries_by_continent | 2,842 | 2,832 | 0.21 | 4.22 | 5.16 | 3.04 |
| Former_countries_by_status | 1,094 | 1,092 | 0.34 | 4.18 | 5.14 | 3.84 |
| History_of_London | 17,609 | 17,589 | 0.09 | 4.35 | 5.23 | 1.99 |
| 21st_century_by_city | 15,460 | 15,452 | 0.09 | 4.50 | 5.37 | 2.36 |

## 3. A* pruning effectiveness

Precomputing the minimum parent-hop distance from each node to root
(BFS from root via child edges) provides an admissible heuristic:
`lower_bound = current_cost + min_dist[node] * parentCost`. Branches
where this lower bound exceeds the budget are pruned.

This makes the search complete (no expansion cap needed) while
keeping it tractable:
- At childCost=3.0: 2.8M expansions for 558k paths (1.5s)
- At childCost=2.0: 120M expansions for 21.7M paths (77s)

## 4. Recommendations

- **childCost=3.0** is the recommended default. At this setting the
  weighted metrics have converged (0.1-0.3% delta), the search
  completes in 1.5s, and the 558k paths capture genuine lateral
  connectivity.

- **Direction-weighted metrics** should replace the uniform
  power-law for bidirectional search. The weighted power-mean
  `d = (Σ(w_i * (h+1)^(-n)) / Σ(w_i))^(-1/n)` with b=15, D=5
  gives convergent, interpretable distances in hops.

- **Budget=15** with parentCost=1.0 and childCost=3.0 allows:
  - Up to 15 pure parent hops
  - Up to 5 pure child hops
  - Mixed: 1 child + 12 parent, 2 child + 9 parent, etc.

- **Parameters b and D** can be calibrated per-graph from observed
  child branching factor and spectral dimensionality. The current
  b=15, D=5 are empirical estimates from this simplewiki fixture.

- Future work: test on full enwiki (9.93M edges); explore top-K
  path pruning as an alternative to exhaustive enumeration;
  compare with exponential decay flux from
  `docs/design/COST_FUNCTION_PHILOSOPHY.md`.

## 5. Stress test: budget=20, 100 seeds at depth 4

With a higher budget (20 vs 15) and more seeds (100 vs 50), the
path count grows dramatically:

| childCost | Paths | Mixed | d_wPow | Time |
|-----------|-------|-------|--------|------|
| 100 (up-only) | 6,260 | 0 | 5.343 | 88ms |
| 10 | 62,298 | 56,038 | 5.386 | 211ms |
| 5 | 4,535,301 | 4,529,041 | 5.391 | 10.3s |
| 3 | (>120s, killed) | — | — | >120s |

The weighted metric converges well (+0.8% then +0.1%), but
childCost=3 with budget=20 is computationally intractable even
on simplewiki (25k edges). For enwiki (9.93M edges) or real-time
applications, either:
- Use childCost >= 5 (10.3s for 4.5M paths, convergent)
- Implement top-K path pruning to bound enumeration
- Reduce budget to 15 (childCost=3 completes in 1.5s)

## 6. Scale limitations and next steps

The full enwiki benchmark (9.93M edges) could not be run because
Wikipedia dumps are blocked by the environment's network policy.
The simplewiki results (25k edges, depth 20) characterize the
kernel's behavior on real Wikipedia structure but at 400x smaller
scale.

Expected enwiki behavior:
- childCost=5 with budget=15: likely tractable (~minutes)
- childCost=3 with budget=15: likely multi-hour per seed
- Top-K pruning needed for any childCost < 5 at enwiki scale

## 7. References

- Kernel template: `templates/targets/fsharp_wam/kernel_bidirectional_ancestor.fs.mustache`
- Design: `docs/design/WAM_FSHARP_CSR_KERNEL_INTEGRATION.md`
- Fixture: `data/benchmark/10k/` (simplewiki 10k article scale)
- Cost function theory: `docs/design/COST_FUNCTION_PHILOSOPHY.md`
