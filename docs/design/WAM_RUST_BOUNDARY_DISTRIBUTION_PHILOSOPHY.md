# WAM-Rust Boundary Distribution Optimization — Philosophy

Why the Rust WAM graph-search target should grow a **boundary distribution**
optimization, what kind of thing it is, and the principles that should shape it.
Companions: `WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md` (precise semantics),
`WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md` (phased implementation + status),
`WAM_RUST_BOUNDARY_DISTRIBUTION_HOWTO.md` (the operator's manual — how to use it),
`WAM_RUST_BOUNDARY_MEASUREMENT_2026-06-16.md` (what was measured).

## 1. The problem

Root-anchored path-aggregate queries (e.g. `effective_distance`,
`d_eff = (Σ_paths Hops^(-N))^(-1/N)`) aggregate over **every** path from a seed up
to a root. In a graph with diamonds the number of such paths is **exponential**
in the depth budget. The kernel enumerates them all.

The edge cache (PRs #3127–#3196) makes each parent lookup cheap, but it **cannot
change the number of paths enumerated** — it attacks I/O cost, not the
combinatorics. So even with a perfect edge cache, a dense shared upper cone is
re-enumerated, in full, for every seed that passes through it.

## 2. The insight: a path-length histogram is a compact exponential

The aggregate is a **linear functional of the path-length histogram**
`H[L] = #{paths of length L}`:

```
WeightSum = Σ_L H[L] · L^(-N)        (and mass = Σ H[L], moment1 = Σ L·H[L], …)
```

and path histograms **compose by convolution** over any cut node B on the
seed→root frontier:

```
H_seed→root = H_seed→B  (∗)  H_B→root
```

The histogram is the key: a vector of ~`budget` integers represents
exponentially-many paths exactly. So if the upper cone's histogram `H_B→root` is
known, a query that reaches B **splices** it (an O(budget) convolution) instead of
re-enumerating the cone.

## 3. What this primarily is: a complexity reduction (caching is secondary)

The headline is **not** "another cache." It is a **complexity reduction**:
exponential per-seed path enumeration becomes a polynomial precompute plus an
O(budget) splice. Measured on a dense-core synthetic graph
(`boundary_splice_complexity_bench.rs`), the splice is **300–700× faster than full
enumeration and the speedup grows with scale** — the fingerprint of a
complexity-class improvement, not a constant factor.

**Caching is the secondary, related lever.** Precomputing the boundary histograms
once and reusing them across seeds (and across queries) is what amortizes the
precompute — important, but it rides on top of the complexity win, not the other
way around. (This is the inverse emphasis of the edge cache, whose entire value
*is* reuse.)

## 4. What kind of feature: a disablable compiler optimization with a family of outputs

This is best understood as a **compiler optimization**, gated and disablable:

- **Recognize** a query shape — an aggregate of a path-length functional over
  root-anchored paths, *or* a request for the path-length distribution itself.
- **Substitute** the enumerating kernel with a **histogram-producing boundary
  kernel** plus a **result extractor**.
- **Gate** it (off by default / `boundary_optimization(false)`), so the optimized
  and unoptimized outputs can be compared and the optimization can be disabled
  whenever its preconditions are not met.

Crucially, the natural output of the boundary kernel is the **histogram**, and a
scalar aggregate is just *one extraction* from it. So the optimization is a
**family**, not a scalar special case:

| query wants | extractor | result Value |
|-------------|-----------|--------------|
| a scalar aggregate (`d_eff`, mass, moment) | apply the functional to `H` | `Float`/`Integer` |
| the path-length **distribution** | return `H` directly | `List` of counts |
| (future) CDF / quantiles / moments | the corresponding read of `H` | as needed |

Both the scalar and distribution results emit through the existing
`finish_foreign_results` **`deterministic`** mode (one result, no choice point) —
no new result machinery. Designing for "scalar only" would be a premature
specialisation that the distribution output rightly rejects.

More precisely (see the specification §1): the fundamental object is, in general,
a **measure over the path-length variable** — the thing that assigns a *weight to
an interval* of lengths (a **sum** for a discrete/atomic measure, an **integral**
for a continuous one). A scalar aggregate, the distribution itself, a CDF, a
quantile, or a windowed/truncated mass are all reads of that measure. This is the
right abstraction precisely because it gives clean **interface hooks**
(`interval_mass`, `atom_mass`, splice, truncate) independent of representation, and
because it handles the awkward cases honestly: an atom / delta on an interval
endpoint is resolved by a right-continuous CDF (`interval_mass((a,b]) = F(b)−F(a)`,
`atom_mass(x) = F(x)−F(x⁻)`), so "centered/left/right" point masses are explicit
rather than ambiguous. Discreteness is a computational convenience (it makes the
splice an FFT-able convolution), not intrinsic; at very large scale a
continuous/parametric measure convolves by parameter arithmetic and answers
interval reads from a closed-form CDF — so the spec is written against the
*measure*, with the discrete histogram the exact default and continuous/cumulative
the large-scale / O(1)-read alternatives.

## 5. Principles

- **Re-derive cost crossovers for compiled Rust.** The Python prototype concluded
  the boundary cache was *slower* — an interpreter-overhead artifact (75.6%
  residual traversal), exactly as on the edge cache (Python: "slower"; Rust:
  2.5–3.5× faster). Inherit the *algorithm* and the *storage* decisions from the
  Python line; **re-measure every wall-time crossover** (e.g. precompute depth
  `D_pre`) in Rust.
- **Histogram-first, exact-by-default.** Cache the raw histogram (general, all
  functionals); a pre-weighted basis (`g_B`, a ~1 ns dot product) is a hot-path
  specialisation. Because the exact splice is ~ns, the exact→approximate ladder is
  justified by **storage** only, never compute.
- **Correctness is gated, not assumed.** The splice is exact only under a proper
  boundary cut (suffix disjoint from prefix on a DAG) and matching root/filter/
  cycle/budget policy. The optimization must *verify or assume* these and be
  disablable; correctness is checked against the production kernel (done:
  `test_wam_rust_boundary_kernel_exec.pl`).
- **Orthogonal to the edge cache.** Different levers: the edge cache removes seek
  cost; the boundary distribution removes enumeration. They compose.

## 6. Why now

The complexity win is measured (300–700×), the splice identity is proven in Rust
(P1), and the runtime kernel exists and matches the production kernel (P2a,
P2c-parity). What remains is the optimization *wiring* — and the histogram-output
generalisation above is what keeps that wiring from being built too narrowly.

## 7. What the measurements showed (hypotheses → results)

The wiring is now built and measured end to end (`WAM_RUST_BOUNDARY_MEASUREMENT_
2026-06-16.md`, `WAM_RUST_BOUNDARY_DISTRIBUTION_HOWTO.md`). The §3/§5 hypotheses held:

- **The win is real *on top of* the edge cache, and the crossover is shallow.** On
  the real emitted kernels (not the Python prototype, not a std-only model), with the
  baseline running over warm in-memory edges (= a perfect edge cache): ~3× at
  `D_pre=1`, 16–26× at `D_pre=2`, >100× at `D_pre=3`. The optimal `D_pre` is *shallow*,
  exactly as §3 predicted the Python overhead had been masking. Precompute is sub-ms —
  it amortizes within a single 500-seed batch.
- **It holds, and grows, on the LMDB lazy edge path.** When each parent lookup is an
  LMDB seek rather than a HashMap hit, production is ~4–5× slower but the boundary win
  *persists and is often larger* (20–25×): the cache removes the *walk*, so the more
  each avoided lookup costs, the more it saves. This confirms §3's "layer on top of the
  edge cache, not a replacement" — the boundary cache's headroom is exactly the
  native-DFS time the edge cache leaves behind.
- **Exact at scale, every time.** The boundary aggregate equals the production
  hop-stream aggregate exactly across all `D_pre`, scales, and on both edge paths. The
  "any boundary band is exact" property held under the real integrated path.
- **The band must be a thin cut, and value concentrates root-near.** The whole-region
  band blows up with cumulative node count; the *entry frontier* (the region's surface)
  gets the same speedup at a fraction of the storage — and the speedup *falls* once
  `D_pre` is large enough that the region swallows the periphery. "A boundary cache
  wants its seeds in the periphery" is now a measured rule, not just intuition.
- **Caching is genuinely secondary.** The complexity reduction stands on its own (the
  precompute pays for itself within one batch); cross-run persistence and the lazy /
  eager / spill strategy choices are amortization levers on top of an already-winning
  algorithm — the inverse emphasis of the edge cache, as §3 argued.

The remaining open questions are storage-shaped, not compute-shaped: the
exact→approximate ladder (tail-prune → binomial/beta-binomial/mixture, CDF-gated) and
the quantised-CDF / GMM rungs matter only at large budget / deep paths, where
histograms are long — never for the small-budget effective-distance query, where exact
is also fastest.

## 8. The generalization: carry the functionals, not the histogram

The deepest "what kind of feature" answer (§4) is that the histogram is one *payload* of a
more general idea: the cheap **functionals** of the path-length distribution each satisfy
their own recurrence and can be propagated and spliced *without ever forming the
histogram*. That reframes the boundary cache as a **`PathSemiring`** parameterised by what
it carries — the histogram (everything, exact), the **moment jet** `(mass,m₁,m₂,m₃)` (mean
/ variance / skew and a CLT distribution reconstruction, budget-free on the acyclic
ancestor space), the **min-plus distance** scalar (cycle-correct shortest-distance, where
the histogram's DFS is unsound on cycles), and the between-nodes **composite caret**
distance. Same architecture, a different per-node element. The full theory — why the
functionals factor, the inner-product / kernel-trick picture, the cumulants-vs-moments
fork, and the directed/undirected subtleties of the caret bound — lives in
**`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`**.

