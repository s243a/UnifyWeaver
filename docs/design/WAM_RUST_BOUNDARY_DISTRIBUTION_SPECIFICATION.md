# WAM-Rust Boundary Distribution Optimization — Specification

Precise semantics of the boundary distribution optimization. See
`WAM_RUST_BOUNDARY_DISTRIBUTION_PHILOSOPHY.md` (rationale),
`WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md` (phasing/status),
`WAM_RUST_BOUNDARY_DISTRIBUTION_HOWTO.md` (how to use it),
`WAM_RUST_BOUNDARY_MEASUREMENT_2026-06-16.md` (measured results), and
`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` (the algebraic generalization — propagating
*functionals* of the histogram without forming it; the basis for the next increments).

## 1. The object: a measure over path length

The cached object is, in general, a **measure** `μ` over the path-length variable
`ℓ` (the single variable here being the seed→root path length). A measure assigns
a non-negative **weight to a set** of lengths; everything the optimization needs is
a *read* of `μ`. This is the abstraction the implementation should expose as an
**interface** (the hooks), with the concrete representation behind it. It subsumes
the cases under one idea:

- a **discrete histogram** is an *atomic* measure — a sum of point masses at
  integer lengths;
- a **continuous approximation** is an *absolutely-continuous* measure — a density
  integrated;
- a **single deterministic length** is a *Dirac* point mass;
- mixtures of the above are measures too.

### 1a. The interface (one canonical read, and the endpoint care)

Domain: path length `ℓ ∈ [0, budget]` (a non-root seed has `ℓ ≥ 1`; `ℓ = 0` only at
the root). The interface is defined against a **right-continuous CDF**
`F(x) = μ([0, x])` (so `F` *includes* the atom at `x`). There is **one canonical
read** — the half-open interval mass — plus the atom jump:

```
interval_mass(a, b) := μ((a, b]) = F(b) − F(a)     // THE method; half-open (a, b] by convention
atom_mass(x)        := F(x) − F(x⁻)                // the point mass AT x (0 if no atom)
total_mass          := F(budget)
```

— a *sum* for an atomic (discrete) measure, an *integral* for a continuous one.
`interval_mass(a, b)` **always** means `(a, b]`; the closed and other variants are
*derived*, never second signatures:

```
closed_interval_mass(a, b) = interval_mass(a, b) + atom_mass(a)   // [a, b]
```

**The delta-function caveat, resolved.** A point mass on an endpoint is no longer
ambiguous: it belongs to the side the half-open convention assigns, and its own
mass is `atom_mass`. In the discrete case the mass at integer `L` is
`interval_mass(L−1, L) = F(L) − F(L−1)`. **Invariant:** for a non-root seed
`atom_mass(0) = 0` (no zero-length path), so `total_mass = interval_mass(0, budget)`
— stated rather than assumed, so the half-open default never silently drops a
future zero-length atom.

`truncate(μ, lo, hi)` is the **unnormalized** restriction of `μ` to `[lo, hi]`
(mass outside the window set to 0); renormalisation to a *conditional* measure, if
ever wanted, is a **separate** operation. `splice` is convolution (§3).

**Interface sketch (illustrative, not final):**

```rust
trait Measure {
    fn interval_mass(&self, a: i64, b: i64) -> f64;  // μ((a,b]); f64 — real mass, not raw u64 counts
    fn atom_mass(&self, x: i64) -> f64;              // F(x) − F(x⁻)
    fn total_mass(&self) -> f64;
    fn truncate(&self, lo: i64, hi: i64) -> Self;    // unnormalized restriction to [lo, hi]
    // NOTE: splice is deliberately NOT a Self×Self→Self method here. Convolving a
    // histogram with a parametric measure yields a *third* representation, so the
    // composition is a separate combinator with its own output type (§3), not a
    // trait method — this is the associated-type seam to decide at implementation.
}
```

The return type is `f64` (real mass) even for the discrete histogram (whose raw
counts are `u64`): mass is a measure value, and continuous / normalised backings
need reals. A single `cdf(x)` primitive could derive `interval_mass` and
`atom_mass` as thin helpers — fewer methods to keep coherent, and `atom_mass`
returns `0` on a continuous backing with no special path. We keep both explicit for
efficiency (a histogram answers a window with one partial-sum, not two CDF
evaluations) and clarity; an implementation MAY back them with a single `cdf`.

### 1b. Backing representations (concrete measures)

| representation | `interval_mass` | splice (compose over a cut) | when |
|----------------|-----------------|------------------------------|------|
| **discrete histogram** (atomic) `H[L]:u64` | partial sum | convolution — O(m²) direct, **O(m log m) FFT** | default; exact; small/medium support |
| **continuous / parametric** (density: normal, binomial, GMM…) | integral / closed-form CDF | parameter arithmetic (e.g. variances add) — often O(1) | **very large scale**, where the discrete grid is the cost; approximate |
| **cumulative / transformed** (CDF table, or a pre-weighted basis `g_B`) | O(1) difference of cumulatives | depends; for a fixed functional the basis is a dot product (~1 ns) | hot `(functional, budget)`; O(1) reads |
| **Dirac / truncated** (a point mass, or any of the above restricted to `[lo, hi]`) | trivial / as parent | trivial / as parent | a deterministic length; constrained functionals (budget caps, thresholds) need only the relevant window — still a measure |

Discreteness is a *computational* choice (it makes the splice an FFT-able
convolution); it is not intrinsic. At very large scales a continuous/parametric
measure convolves by parameter arithmetic and answers interval reads from a
closed-form CDF, avoiding the grid entirely — at the cost of being approximate
(gated on a CDF/W1 error bound, per `DISTRIBUTIONAL_COMPRESSION_THEORY.md`).

- **Boundary set** `Bset ⊆ V`: nodes near the root at which suffix measures are
  cached. The current backing is `boundary_suffix: FxMap<u32, Vec<u64>>` on
  `WamState` (each boundary node `B` → `H_B→root`) — which holds **only** the exact
  discrete histogram. Supporting the non-discrete §1b backings means the value type
  must evolve into an enum or boxed `dyn Measure`, e.g.
  `enum MeasureBacking { Histogram(Vec<u64>), Cdf(Box<[f64]>), Parametric(Params), Dirac(i64) }`.
  `FxMap<u32, Vec<u64>>` is the exact-discrete case, not a universal backing.
- **Budget** `max_depth: usize`: the path-length bound, as in the production
  `category_ancestor` kernel.

## 2. Aggregate functionals (weighted reads of the measure)

(Terminology bridge: "form", "histogram", and "distribution" in this and later
sections all denote a concrete §1b backing of the **measure** μ of §1; the
functionals and result modes are defined against the measure interface of §1a.)

A functional is a weighted integral over the measure — `∫ w(ℓ) dμ(ℓ)` — which for
the discrete backing is a weighted sum:

```
f = ∫ w(ℓ) dμ(ℓ)   =   Σ_L w(L) · μ{L}   (discrete: μ{L} is the atom at L)
  mass           : w = 1                  (= total_mass = interval_mass(0, budget))
  moment1        : w(L) = L
  weighted_power : w(L) = L^(-N)  (L>0)    -> WeightSum; d_eff = WeightSum^(-1/N)
  bounded variants: integrate w only over [lo, hi]  (truncate(μ, lo, hi) suffices)
```

Linearity makes the splice exact for *all* functionals at once, so the measure is
cached once and read many ways (`boundary_cache::{f_mass,f_moment1,f_weighted_power}`,
or `interval_mass` for arbitrary windows). A **pre-weighted cumulative basis** `g_B`
(the "transformed" backing) folds `w` into the cached object so the read is a dot
product — the right backing when one `(functional, budget)` dominates.

## 3. The splice identity

The splice is, representation-independently, the **convolution of two measures**
`μ_seed→root = μ_seed→B ∗ μ_B→root` (path lengths add, so the measure of the sum is
the convolution). Its realization depends on the §1b backing: for the discrete
histogram it is histogram convolution (below); for a parametric measure it is
parameter arithmetic (e.g. normal variances add) — same identity, different cost.
The discrete realization:

```
H_seed→root[L] = Σ_{a+b=L} H_seed→B[a] · H_B→root[b]
```

Over a boundary set, a seed→root path crosses the boundary at its **first**
boundary node (or reaches root with no crossing). The boundary kernel therefore:
walks seed→{root or first boundary node}; at a boundary node `B` reached at prefix
depth `d`, adds `H_B→root[b]` into `H[d+b]` for all `d+b ≤ max_depth`, and stops;
at root, adds `1` to `H[depth+1]`. (`WamState::collect_native_category_ancestor_
boundary_hist`.)

## 4. Correctness invariants (preconditions for exactness)

1. **Proper cut.** Every `B→root` suffix path must be node-disjoint from any
   `seed→B` prefix (so production's `visited` pruning never differs). On a DAG
   (parent-only category edges) with `Bset` chosen as a root-near antichain this
   holds. The optimization MUST be disabled, or fall back, when it cannot be
   established.
2. **Policy match.** Suffix histograms must be precomputed with the *same* root,
   edge filter, and cycle policy the query uses.
3. **Budget match.** The `d + b ≤ max_depth` truncation reproduces the production
   `visited.len() >= max_depth` gate at the cut. Suffixes are precomputed up to
   `max_depth` and truncated per prefix depth at splice time.
4. **First-crossing only.** The walk stops at the first boundary node, so each
   path is counted once.

Conformance is verified against the production kernel
(`collect_native_category_ancestor_hops`) in
`tests/test_wam_rust_boundary_kernel_exec.pl`: equal `weighted_power` across seeds.

## 5. Result modes (the output family)

The boundary kernel produces the **measure** μ (default backing: `H`); a **result
extractor** reads it into the foreign predicate's result. All use the existing
`finish_foreign_results` **`deterministic`** mode (one result, no choice point,
`tuple(1)` layout):

| mode | result `Value` | extractor (a read of μ, per §1a) |
|------|----------------|-----------------------------------|
| `scalar(functional)` | `Float`/`Integer` | `f` = `∫ w dμ` (e.g. `weighted_power`) |
| `effective_distance` | `Float` | `WeightSum^(-1/N)` |
| `shortest_distance` | `Integer` | shortest hop-distance to root (the support floor `min`) — read from the **min-plus distance cache** (cycle-correct), not the histogram; see `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` |
| `distribution` | `List` of counts (PMF) — or the parametric/CDF encoding for a non-discrete backing | μ itself (possibly truncated) |
| `interval_mass(a, b)` | `Float` | `interval_mass(a, b) = μ((a, b])` (§1a; half-open) |
| `cdf(x)` / `quantile(p)` | `Float` | cumulative reads of μ |

This is the generalisation the measure view makes explicit: the result is *any*
read of μ. `distribution` returns the whole measure (for downstream
distribution-compression consumers); `scalar` an aggregate; `interval_mass` /
`cdf` / `quantile` are the §1a interval/cumulative reads (a single sum/integral, or
an O(1) difference when the cached backing is already a cumulative/CDF form). Each
is a new extractor over the same measure, not new kernel work — and for the
cumulative/parametric backings the read is O(1). (`interval_mass` here is the §1a
method; the half-open `(a, b]` convention applies, so this mode is unambiguous at
its endpoints.)

## 6. Foreign-predicate interface

A native kind `category_ancestor_boundary`, registered like the existing kernels
(`recursive_kernel_detection.pl` + the `execute_foreign_predicate` dispatch arm),
result mode `deterministic`, layout `tuple(1)`. Configuration via the existing
`register_foreign_*_config` channels:

- `edge_pred` (e.g. `category_parent`), `max_depth`, `root`.
- `result_extractor` (the §5 modes): `distribution | scalar(weighted_power(N)) | effective_distance(N) | interval_mass(a,b) | cdf(x) | …`.

Args (illustrative): `category_ancestor_boundary(Seed, Out)` with `Root`,
`MaxDepth`, and the extractor bound by config; `Out` unifies with the scalar or the
distribution list (PMF counts), per the chosen `result_extractor`.

## 7. Eligibility (what the optimization recognises)

Gated by `boundary_optimization(true)` (default **false**). When on, the compiler
may substitute the boundary kernel for a predicate whose body is:

- an aggregate of a path-length functional over root-anchored paths
  (`aggregate_all(sum(W), (path_to_root(Seed,Root,Hops), W is f(Hops)), Out)`),
  emitted as `scalar(f)`; or
- a collection of root-anchored path lengths into a list/distribution, emitted as
  `histogram`.

Non-matching predicates compile unchanged. With the gate off, the production
kernel is used and output is identical — the basis for the disablability guarantee.

## 8. Boundary set & precompute

- `Bset` policy: root-near nodes (small `D_pre`); the *value* concentrates there
  (high reuse + large cone) — `D_pre` to be **measured** in Rust, not inherited
  from the Python curves (philosophy §5).
- Precompute: `WamState::build_boundary_suffix(Bset, root, max_depth, edge_pred)`
  enumerates each `B→root` histogram via `enum_ancestor_hist`.
- Persistence (later): a `boundary_basis` LMDB sub-db (node → packed histogram),
  loaded at setup like `s2i`/`min_dist`, so the precompute is not repeated per run.

### 8a. Precompute liveness & eviction

The suffix recurrence builds a node's distribution **from its parents'** (the nodes
one step *closer to the root*):

```
H_root = δ_0            (the empty path: atom of mass 1 at length 0)
H_v[L] = Σ_{p ∈ parents(v)} H_p[L−1]            (boundary_cache.rs:suffix_histogram)
```

So `H_p` is a *consumed input* to every child `v` with `p ∈ parents(v)`, and nothing
else. This gives an exact **liveness signal** over the precompute sweep:

> **`H_p` is live only from when it is first computed until its last child consumes
> it.** Once the distributions of *all* descendants that name `p` as a parent have
> been computed, `H_p` is **dead** — fully consumed — unless `p` is itself a retained
> boundary node (`p ∈ Bset`), whose distribution is kept for query-time splicing.

Track it with a per-node consumer count `refs(p) = #{v : p ∈ parents(v)} within the
precomputed cone`, decremented on each child completion; `refs(p) = 0 ∧ p ∉ Bset`
marks `p` **dead**.

**The trigger is pressure, the liveness is the priority — they are separate.**
Eviction is driven by a **memory / storage budget** (the reserved capacity for the
in-memory side-table and for the `boundary_basis` LMDB sub-db): entries are evicted
only when the budget is under pressure, *not* eagerly at `refs = 0`. When pressure
hits, the liveness signal supplies the **priority order**:

1. **Dead interior nodes first** (`refs = 0 ∧ p ∉ Bset`) — never read again in *this*
   sweep, so the highest-priority class. **Within this class, evict deepest-first**
   (largest depth-from-root): root-near distributions cover the largest cones and are
   the most-reused hubs (philosophy §3), so they are the most likely to be hit again
   by a *second query* — keep them resident longer; the deep, narrow-cone nodes have
   little cross-query reuse value and go first. (So "zero-regret" holds within a
   single sweep; across queries the regret rises toward the root, which is exactly
   what depth-first eviction minimises.)
2. then **still-live interior** scratch (by a secondary metric — e.g. furthest next
   use / lowest remaining `refs`, again breaking ties deepest-first), accepting a
   possible recompute;
3. **retained boundary band** (`p ∈ Bset`) last, and even then it spills to the
   `boundary_basis` LMDB sub-db rather than being dropped, since queries still need
   it.

Consequences:

- **Bounded working set.** Holding dead interiors until pressure (rather than freeing
  at `refs = 0`) keeps the resident scratch within the budget while preferring to
  keep what might still be reused; under steady pressure the resident set tends
  toward the cone's *frontier antichain width* plus the retained band — biased to
  retain the **root-near** end of that scratch, the part with cross-query reuse.
- **Exactness preserved.** Eviction only ever removes an entry that is either dead
  (no future read) or recomputable; a spilled/recomputed `H_p` yields the identical
  histogram. So the policy changes resource use, never any spliced result — it is a
  cache policy (recompute / reload on demand), not a one-shot deletion.

### 8b. Two budgets: evictable vs live, and the stop-at-depth recourse

The §8a budget governs entries we *can* drop (dead, or live-but-recomputable). But a
top-down sweep also holds a **live, not-yet-discardable** set — the current frontier
of nodes with `refs > 0`, whose parents are done but whose children are not. You
cannot evict your way out of pressure on *that* set: those distributions are still
required inputs to children not yet computed. So the precompute carries **two
limits**:

1. **Evictable budget** (§8a) — the cap on resident *droppable* entries (dead
   interiors + spillable/recomputable scratch + the retained band). Exceeding it
   triggers the priority-ordered eviction above.
2. **Live budget** — a separate cap on the **non-discardable frontier** (`refs > 0`,
   not yet spillable). This bounds the sweep's irreducible working set ≈ the cone's
   *frontier antichain width* at the current depth.

When the **live budget** is hit there is nothing to evict, and there are two
recourses:

- **Stop-at-depth (default).** Freeze the precompute frontier at the current depth,
  treat everything below it as un-cached (those seeds fall back to enumeration, still
  correct), and retain what was computed. This makes the effective `D_pre`
  **dynamically bounded by memory**, not a fixed a-priori choice (philosophy §5 /
  plan §3): deepen the band only while the live frontier fits, then stop.
- **Spill-and-continue (non-default, implemented).** Given a `SpillSink`,
  `build_boundary_suffix_sweep_with_spill` instead **spills** live frontier entries
  (deepest-first — keep the root-near hubs resident) to the sink, removing them from
  the memo, and **reloads** each on demand when a child consumes it. The whole cone
  is swept (`stopped_early` stays false); `spilled` counts the writes. It is **exact
  paging** — a spilled `H_p` reloads identically — so results match the unbudgeted
  sweep. The sink is an in-memory map (RAM-overflow / testing) or the
  `boundary_spill` LMDB sub-db (`impl SpillSink for LmdbFactSource`, one short txn per
  put/get — a fallback path, not a hot loop). Spill is **off unless a sink is
  supplied**, and engages only when the in-memory live budget is exceeded; stop-at-
  depth remains the recourse when neither memory nor a sink can hold a wider frontier.

### 8c. Why exact (vs path sampling)

The one alternative that could rival the splice on *speed* is **Monte-Carlo path
sampling**: estimate `WeightSum = Σ_L H[L]·L^(-N)` from a small random sample of
seed→root paths instead of the full histogram. It is rejected here on **variance**,
not speed: a sample estimator carries estimation error that (a) shrinks only as
`1/√(samples)`, (b) is worst exactly where the functional is most sensitive — the
short paths that dominate `L^(-N)` for `N>0` are rare in a uniform path sample of a
diamond-dense cone — and (c) gives a *different* answer per run, breaking
reproducibility and the linear-functional exactness the whole design rests on. The
boundary splice computes the **exact** functional at ~ns from the cached histogram,
so there is no accuracy/speed tradeoff to make: exact is also fast. Sampling would
only be considered if even the polynomial precompute became infeasible — at which
point the §9 storage ladder (continuous/parametric measure) is the principled
exact-in-the-limit fallback, again before sampling.

### 8d. Lazy vs eager precompute (evaluation strategy)

The band can be filled two ways — the lazy/eager axis of
`RECURRENCE_EVALUATION_STRATEGY` applied to the boundary suffixes:

- **Eager** (`build_boundary_suffix_*`): materialise the whole band up front
  (fixed-point / bottom-up). One batched shared-memo sweep; amortised across all
  seeds; needs the in-memory `ffi_facts` edge table for the polynomial variants.
  Best when the band is known and densely reused.
- **Lazy** (`lazy_boundary_weightsum`): start empty and compute each band node's
  suffix histogram **on first demand** (per-query / top-down), memoizing it — only
  the band-entry nodes the workload touches are ever computed. Identical results
  (any band is exact). Best when the touched subset is sparse or unknown, when there
  is no good precompute moment (streaming / interactive), or when `D_pre` is hard to
  pick a priori — the cache self-warms. Also the strategy available on the
  **lazy/LMDB edge path** (it enumerates via the `EdgeAccessor`, not `ffi_facts`).

Neither dominates, and the choice depends on **workload sparsity × query count K**,
not a single number. Steady state is identical (both splice once warm), so the
decision is entirely the *warmup* cost. Measured (`..._MEASUREMENT_2026-06-16.md`
lazy addendum): **sparse/unknown** workloads favour lazy at every K (eager wastes
precompute on a band it won't use); a **dense** workload with a **modest** K favours
eager (batched precompute beats on-demand warmup); a dense workload with **large** K
tips back to lazy (its smaller warm cache gives marginally faster rounds). Bigger
datasets shift it toward lazy (the eager band grows with the graph; the touched
subset is bounded by the workload).

## 9. Storage / approximation

Exact discrete histograms by default. The §1a backings are the storage/scale
ladder: when a node's histogram exceeds a storage budget — or when the support is
so large that the discrete grid itself is the cost — a **continuous/parametric
form** (binomial / discretised or continuous GMM, per
`DISTRIBUTIONAL_COMPRESSION_THEORY.md`) replaces it, gated on a CDF/W1 error bound;
or a **cumulative/CDF/basis** form replaces it when O(1) range/aggregate reads
dominate. Because the exact discrete splice is ~ns, choosing a non-exact form is a
**storage / very-large-scale** decision, not a compute one — at normal scale,
exact discrete is both fastest and simplest.

### 9a. The exact→approximate ladder (ported policy + defaults)

Ported from the Python distribution-cache line (`DISTRIBUTIONAL_COMPRESSION_
THEORY.md`); two principles that doc is emphatic about are carried over verbatim:

- **The point count is a *work trigger*, not an acceptance gate.** "Is this over 50
  points?" is *not* the decision rule — it only decides *when to try* compressing.
  Acceptance is an **error certificate** on the CDF: Kolmogorov
  `ε_K = max_t|F_exact(t) − F_approx(t)|` (and optionally Wasserstein-1
  `ε_W1 = Σ_t|ΔF|`, which bounds Lipschitz-functional error). Defaults:
  `min_points = 50` (trigger), `ε_K = 0.001` (certificate).
- **Discrete-first ladder**, cheapest representation within the error budget:
  exact histogram → **tail-pruned exact** → quantised CDF table → binomial →
  beta-/mixture-of-binomials → discretised GMM (*escalation only* — bounded integer
  path-length data favour binomial families; GMM is not the default fallback).

Implemented (Rust, `boundary_cache.rs`):

- **Error metrics** — `cdf_max_error` (Kolmogorov), `cdf_w1_error` (Wasserstein-1).
- **Rung 1 — tail-pruned exact** — `tail_prune` drops the longest suffix whose
  cumulative mass ≤ the budget; the dropped fraction *is* the Kolmogorov error.
- **Rung 2 — parametric binomial** — two fits: `fit_binomial` (method of moments:
  `trials = support-1`, `p = mean/trials`) and `fit_binomial_cdf` (**CDF-space**:
  `p` chosen to minimise the Wasserstein-1 / integral CDF distance by a 1-D search).
  The CDF fit optimises the *same cumulative space the gate judges in* (§3/§6 of the
  theory doc) — a smoother, lower-bandwidth target than the PMF — so it is at least
  as good as moment-matching on the gate metric and **strictly better when the mean
  is pulled by local PMF contamination** (validated). `binomial_pmf` uses a stable
  recurrence. A `HistRepr` (`Exact` | `Binomial{trials,p,total}`) carries `bytes()`,
  `pmf()`, and `expand()` (a binomial expands to a rounded count histogram, since
  the kernel consumes counts). `choose_representation(h, min_points, ε_K)` mirrors
  the Python `choose_distribution_representation`: the cheapest representation whose
  CDF error is within `ε_K` — exact (error 0), tail-pruned, or binomial (trying both
  fits and keeping the lower-error one). **The gate is a hard reject:** a fit that
  misses `ε_K` is dropped and the exact/tail-pruned form kept, so accuracy never
  silently degrades.
- **Rung 3 — beta-binomial** — `fit_beta_binomial(h)` (method of moments: the
  intra-class over-dispersion `rho` from the variance gives `alpha+beta = 1/rho-1`)
  fits a **unimodal but over-dispersed** node — variance above a binomial's
  `n*p*(1-p)` — that a single binomial rejects but which a mixture would over-model.
  `beta_binomial_pmf` uses a stable ratio recurrence (no `lgamma`, std-only).
  `HistRepr::BetaBinomial{trials,alpha,beta,total}` (3 params, cheaper than a
  mixture's 2K) joins the candidate set.
- **Rung 4 — mixture of binomials** — `fit_binomial_mixture(h, k, iters)` (EM,
  shared `trials`, PMF-weighted) fits the **multimodal** nodes a single binomial
  rejects (a bottleneck / topic-mixture cone). `HistRepr::Mixture{trials,comps,total}`
  joins the candidate set (`K = 2,3`). Costlier-per-mode escalation (discretised-GMM)
  is Rung 6 — bounded integer path-length data favour binomial families, so it is
  tried last.
- **Rung 5 — quantised CDF table** — `HistRepr::QuantCdf{qcdf,total}` stores the
  exact CDF as one fixed-point `u16` per support point (`quantize_cdf` /
  `dequantize_cdf`; `F(i) ≈ qcdf[i]/65535`). **Always admissible** (error bounded by
  the quantisation step, `≤ 2^-16`) at ~1/4 the bytes of the raw `u64` counts, so it
  is the fallback when *no* parametric form fits an irregular/multi-spike node — and
  it gives O(1) prefix-mass reads (a natural fit for the `cdf`/`quantile` result
  modes). The chooser still prefers a cheaper parametric form when one passes.
- **Rung 6 — discretised Gaussian mixture** — `fit_discretised_gmm(h, k, iters)` (EM
  over the integer support, PMF-weighted) fits a mixture whose components carry a
  **free** `(weight, mu, sigma)`. Every rung above is a *binomial* family, whose
  per-component variance is pinned to its mean (`n*p*(1-p)`); that coupling means a
  binomial mixture cannot place a *narrow* mode in the interior of the support (a tight
  mode forces `p` toward an edge). The GMM lifts the coupling, so it fits the
  arbitrarily-placed, arbitrarily-narrow interior modes the binomial families cannot.
  `discretised_gmm_pmf` evaluates the components at each integer bin and renormalises
  (truncation at the domain edges absorbed). `HistRepr::DiscGmm{support,comps,total}`
  costs 3 params/mode — the most of any parametric form — so the chooser selects it
  only when every cheaper rung misses the gate **and** it still undercuts the
  quantised-CDF table on bytes. **Escalation only**, present for shapes the binomial
  families genuinely cannot represent.
- **Choice is multi-objective.** `choose_representation` is *error-driven* (cheapest
  within `ε_K`); `choose_representation_budget` is the *storage-driven* complement
  (smallest CDF error within a byte budget). Error is not always the binding
  constraint — memory / storage (or, with a different cost model, compute) often is —
  so the `min_points` work trigger is best read as a **storage proxy** ("this is
  getting big"), not an intrinsic error threshold, and need not be exactly 50.
- **Persistence (storage win, end to end).** `WamState::boundary_suffix_reprs` chooses
  a representation per cached node; `encode_repr`/`decode_repr` pack it (self-
  describing tag + fields); `LmdbFactSource::save_boundary_reprs` / `load_boundary_
  reprs` persist to the `boundary_basis_repr` sub-db and **expand** on load (a
  binomial node stores ~21 bytes instead of a histogram; a fitted node reconstructs
  within its `ε_K` certificate).
- `compress_histogram` / `WamState::compress_boundary_suffix(min_points, ε_K)` apply
  rung 1 across the cached table.

**All of this is OPT-IN** — the boundary cache is exact unless invoked, and even
then a lossy node differs by at most `ε_K` (the splice becomes approximate within
that certified bound for the affected nodes only). For typical small-budget boundary
histograms (support ≤ `budget`+1) the work trigger never fires; this matters at
**large budget / deep paths**.

### 9b. Ladder status and future work

All six **fitted histogram-approximation rungs** are implemented (Rungs 1–6: tail-pruned
exact, binomial [moment + CDF fits], beta-binomial, mixture-of-binomials, quantised-CDF,
discretised-GMM) — i.e. the *histogram-representation* ladder is closed structurally.
This is a structural closure, not an empirical claim that every rung has been observed
necessary on a measured workload (the GMM test uses a synthetic ground truth; whether a
real boundary node escalates past Rung 5 is a measurement question, §6). The error/CDF-
gate machinery, the chooser (error- and storage-driven), and the persistence path are
general — adding a representation is just another candidate in
`representation_candidates` with a `bytes()` and a `pmf()`/`expand()`.

A **separate, cheaper reconstruction rung** — the moment-jet / CLT reconstruction of
`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` §7 — now exists as `HistRepr::MomentNormal`
(carry `(mass, m₁, m₂)`, reconstruct a discretised Normal; wire tag 6). It is *not* a
histogram fit — it is constructible from the moment jet **alone**, so it is the form a
jet-only propagation reconstructs — but it joins the same candidate set under the same CDF
gate. What remains is the *propagation* side: carrying the jet through the boundary
precompute without ever materialising the histogram (functional-semiring note §8, 1c), and
the higher-order Edgeworth/Pearson members.

Remaining, lower priority and added on demand:

- **Jet-only propagation + higher-order reconstruction** — carry `(min,max,mass,m₁,m₂)`
  through the precompute (never forming the histogram); Edgeworth/Pearson rungs (§7).
- **heed-backend parity** for the `boundary_basis_repr` persistence and the spill
  sub-db (the `lmdb-zero` backend has both today).
- **Higher-K / Bayesian model selection** for the mixture and GMM rungs (today `K =
  2,3` are tried and the gate + byte cost arbitrate); a BIC/MDL term would let the
  chooser explore more components without overfitting.

Each rung is justified by **storage at a tolerance**, never compute (the exact
splice is ~ns), so they are added on demand as large-budget / deep-path workloads
appear.

## 10. Non-goals (this spec)

- Changing the production kernel or the default (un-optimized) semantics.
- Cross-thread/query persistence beyond the side-table + optional LMDB sub-db.
- Non-DAG cycle exactness guarantees (handled by the cut precondition / fallback).
