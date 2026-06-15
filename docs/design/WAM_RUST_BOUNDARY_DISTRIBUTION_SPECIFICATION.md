# WAM-Rust Boundary Distribution Optimization — Specification

Precise semantics of the boundary distribution optimization. See
`WAM_RUST_BOUNDARY_DISTRIBUTION_PHILOSOPHY.md` (rationale) and
`WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md` (phasing/status).

## 1. Objects

- **Path-length histogram** `H: [u64]`, `H[L]` = number of root-anchored paths of
  edge-length `L` (within budget). Trailing zeros insignificant. Represented in
  Rust as `Vec<u64>` (`boundary_cache::Hist`).
- **Boundary set** `Bset ⊆ V`: nodes near the root at which suffix histograms are
  cached. `boundary_suffix: FxMap<u32, Vec<u64>>` on `WamState` maps each boundary
  node `B` to `H_B→root`.
- **Budget** `max_depth: usize`: the path-length bound, as in the production
  `category_ancestor` kernel.

## 2. Aggregate functionals (linear in `H`)

```
f(H) = Σ_L w(L) · H[L]
  mass          : w(L) = 1
  moment1        : w(L) = L
  weighted_power : w(L) = L^(-N)   (L>0)   -> WeightSum; d_eff = WeightSum^(-1/N)
```

Linearity is what makes the splice exact for *all* functionals simultaneously,
and what lets the histogram be cached once and read many ways
(`boundary_cache::{f_mass,f_moment1,f_weighted_power}`).

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

The boundary kernel produces `H`; a **result extractor** maps `H` to the foreign
predicate's result. All use the existing `finish_foreign_results` **`deterministic`**
mode (one result, no choice point, `tuple(1)` layout):

| mode | result `Value` | extractor |
|------|----------------|-----------|
| `scalar(functional)` | `Float` (or `Integer`) | `f(H)` for the named functional |
| `histogram` | `List([Integer(H[0]), …])` | identity |
| `effective_distance` | `Float` | `WeightSum^(-1/N)` |

This is the generalisation over a scalar-only design: `histogram` returns the full
distribution (e.g. for downstream distribution-compression consumers); `scalar`
returns an aggregate. Adding CDF/quantile/moment modes is a new extractor over the
same `H`, not new kernel work.

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

Exact histograms by default. When a node's histogram exceeds a storage budget, a
fitted form (binomial / discretised GMM, per `DISTRIBUTIONAL_COMPRESSION_THEORY.md`)
may replace it, gated on a CDF/W1 error bound. Because the exact splice is ~ns,
approximation is a **storage** decision only.

## 10. Non-goals (this spec)

- Changing the production kernel or the default (un-optimized) semantics.
- Cross-thread/query persistence beyond the side-table + optional LMDB sub-db.
- Non-DAG cycle exactness guarantees (handled by the cut precondition / fallback).
