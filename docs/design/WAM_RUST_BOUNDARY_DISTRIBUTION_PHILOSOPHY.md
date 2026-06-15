# WAM-Rust Boundary Distribution Optimization — Philosophy

Why the Rust WAM graph-search target should grow a **boundary distribution**
optimization, what kind of thing it is, and the principles that should shape it.
Companions: `WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md` (precise semantics),
`WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md` (phased implementation + status).

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

More precisely (see the specification §1): the fundamental object is not "a
histogram" but **a form from which we can read a PMF or CDF** — i.e. the *mass
between two points* (a **sum** for a discrete histogram, an **integral** for a
continuous one). A scalar aggregate, the distribution itself, a CDF, a quantile,
or a windowed/truncated mass are all reads of that form. Discreteness is a
computational convenience (it makes the splice an FFT-able convolution), not
intrinsic; at very large scale a continuous/parametric form convolves by parameter
arithmetic and answers ranges from a closed-form CDF — so the spec is written
against the *form*, with the discrete histogram as the exact default and a
continuous/cumulative form as the large-scale / O(1)-read alternative.

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
