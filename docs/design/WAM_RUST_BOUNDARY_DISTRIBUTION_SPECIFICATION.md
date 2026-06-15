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

### 1a. The interface (one operation, and the endpoint care)

The core read is the **mass of an interval**:

```
interval_mass(a, b) = μ of the length interval between a and b
```

— a *sum* for an atomic (discrete) measure, an *integral* for a continuous one.
Everything else (PMF, CDF, any bounded/constrained functional, truncations) is
built from it.

**Endpoint convention — the delta-function caveat.** A point mass sitting exactly
on an endpoint forces a half-open-vs-closed choice. We pin it with a
**right-continuous CDF** `F(x) = μ((-∞, x])` (so `F` *includes* the atom at `x`):

```
interval_mass((a, b]) = F(b) − F(a)              (half-open — the canonical form)
atom_mass(x)          = F(x) − F(x⁻)             (the point mass AT x; 0 if no atom)
interval_mass([a, b]) = F(b) − F(a) + atom_mass(a)
```

So a "centered / left-sided / right-sided" point mass is just where the atom is
placed relative to the half-open intervals — in the discrete case the mass at
integer `L` is `F(L) − F(L−1)`. The interface therefore exposes **both**
`interval_mass` (half-open by default) **and** `atom_mass`, so consumers never
guess the convention. `total_mass = F(∞)`; the splice (§3) and `truncate` are the
other hooks.

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
  cached. The default backing is `boundary_suffix: FxMap<u32, Vec<u64>>` on
  `WamState` (each boundary node `B` → `H_B→root`); an alternate backing stores a
  CDF/basis/parametric measure instead.
- **Budget** `max_depth: usize`: the path-length bound, as in the production
  `category_ancestor` kernel.

## 2. Aggregate functionals (range/cumulative reads of the form)

A functional is a weighted sum/integral over the distribution — i.e. a read of the
form:

```
f = Σ_L w(L) · H[L]   (discrete)   or   ∫ w(ℓ) dH(ℓ)   (continuous)
  mass           : w = 1                  (= range_mass(0, budget))
  moment1        : w(L) = L
  weighted_power : w(L) = L^(-N)  (L>0)    -> WeightSum; d_eff = WeightSum^(-1/N)
  bounded variants: integrate w only over [lo, hi]  (the truncated form suffices)
```

Linearity makes the splice exact for *all* functionals at once, so the form is
cached once and read many ways (`boundary_cache::{f_mass,f_moment1,f_weighted_power}`,
or `range_mass` for arbitrary windows). A **pre-weighted cumulative basis** `g_B`
(the "transformed form") folds `w` into the cached object so the read is a dot
product — the right backing when one `(functional, budget)` dominates.

## 3. The splice identity

For any cut node `B` on a seed→root path, lengths add, so histograms convolve:

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

The boundary kernel produces the **form** (default: `H`); a **result extractor**
reads it into the foreign predicate's result. All use the existing
`finish_foreign_results` **`deterministic`** mode (one result, no choice point,
`tuple(1)` layout):

| mode | result `Value` | extractor (a read of the form) |
|------|----------------|--------------------------------|
| `scalar(functional)` | `Float`/`Integer` | `f` = weighted sum/integral (e.g. `weighted_power`) |
| `effective_distance` | `Float` | `WeightSum^(-1/N)` |
| `distribution` | `List` of counts (PMF) — or the parametric/CDF encoding for a non-discrete backing | the form itself (possibly truncated) |
| `range_mass(a, b)` | `Float`/`Integer` | `range_mass(a, b)` — sum/integral between two lengths |
| `cdf(x)` / `quantile(p)` | `Float`/`Integer` | cumulative reads of the form |

This is the generalisation the histogram/PMF-vs-CDF view makes explicit: the
result is *any* read of the distribution form. `distribution` returns the whole
thing (for downstream distribution-compression consumers); `scalar` an aggregate;
`range_mass`/`cdf`/`quantile` are the "mass between two points" reads (a single
sum/integral, or an O(1) difference when the cached backing is already a cumulative
/ CDF form). Each is a new extractor over the same form, not new kernel work — and
for the cumulative/parametric backings the read is O(1).

## 6. Foreign-predicate interface

A native kind `category_ancestor_boundary`, registered like the existing kernels
(`recursive_kernel_detection.pl` + the `execute_foreign_predicate` dispatch arm),
result mode `deterministic`, layout `tuple(1)`. Configuration via the existing
`register_foreign_*_config` channels:

- `edge_pred` (e.g. `category_parent`), `max_depth`, `root`.
- `result_extractor`: `histogram | scalar(weighted_power(N)) | effective_distance(N) | …`.

Args (illustrative): `category_ancestor_boundary(Seed, Out)` with `Root`,
`MaxDepth`, and the extractor bound by config; `Out` unifies with the scalar or the
histogram list.

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
