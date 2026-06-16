# WAM-Rust Boundary Distribution Optimization — Specification

Precise semantics of the boundary distribution optimization. See
`WAM_RUST_BOUNDARY_DISTRIBUTION_PHILOSOPHY.md` (rationale) and
`WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md` (phasing/status).

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

1. **Dead interior nodes first** (`refs = 0 ∧ p ∉ Bset`) — these are the nodes
   identified above; they will never be read again in this sweep, so they are the
   highest-priority, zero-regret eviction class.
2. then **still-live interior** scratch (by a secondary metric — e.g. furthest next
   use / lowest remaining `refs`), accepting a possible recompute;
3. **retained boundary band** (`p ∈ Bset`) last, and even then it spills to the
   `boundary_basis` LMDB sub-db rather than being dropped, since queries still need
   it.

Consequences:

- **Bounded working set.** Holding dead interiors until pressure (rather than freeing
  at `refs = 0`) keeps the resident scratch within the budget while preferring to
  keep what might still be reused; under steady pressure the resident set tends
  toward the cone's *frontier antichain width* plus the retained band.
- **Exactness preserved.** Eviction only ever removes an entry that is either dead
  (no future read) or recomputable; a spilled/recomputed `H_p` yields the identical
  histogram. So the policy changes resource use, never any spliced result — it is a
  cache policy (recompute / reload on demand), not a one-shot deletion.

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

## 10. Non-goals (this spec)

- Changing the production kernel or the default (un-optimized) semantics.
- Cross-thread/query persistence beyond the side-table + optional LMDB sub-db.
- Non-DAG cycle exactness guarantees (handled by the cut precondition / fallback).
