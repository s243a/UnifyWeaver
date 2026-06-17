# WAM-Rust Boundary Distribution — How-To

A practical guide to **using** the boundary-distribution optimization in the Rust WAM
graph-search target. For *why* it works see
`WAM_RUST_BOUNDARY_DISTRIBUTION_PHILOSOPHY.md`; for the *precise semantics* see
`WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md`; for *what was measured* see
`WAM_RUST_BOUNDARY_MEASUREMENT_2026-06-16.md`; for the **functional-payload generalization**
(moment jet, distance, caret — carry the functionals, not the histogram) see
`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`. This doc is the operator's manual.

## What it is, in one paragraph

A root-anchored path-aggregate query (e.g. `effective_distance`,
`d_eff = (Σ_paths Hops^(-N))^(-1/N)`) enumerates **every** path from a seed up to a
root — exponential in a graph with diamonds. The boundary cache precomputes, at
root-near **boundary** nodes B, the path-length histogram `H_B→root`; a query that
reaches B **splices** the cached suffix (an O(budget) convolution) instead of
re-enumerating. Exponential per-seed enumeration becomes a polynomial precompute plus
a cheap splice. It is **exact** (any boundary band gives the same answer) and
**opt-in** (off by default).

---

## Architecture — the layers, the axes, and the strategy classes

### The layers (a query flows top to bottom; precompute fills the middle)

```
  QUERY KERNELS    histogram splice  │  g_B scalar splice  │  jet/distance/caret splice  │  foreign-dispatch wrapper
                   (boundary_hist)      (weightsum)           (functionals, §payloads)      (category_ancestor_boundary)
                          │ read
  BOUNDARY CACHE   boundary_suffix : node → histogram   (+ boundary_basis_wp : g_B,  boundary_jet : (jet,interval),  boundary_dist : dist→root)
                          │ filled by
  PRECOMPUTE       eager: enum | shared-memo | topological sweep (+evict/spill)   │   lazy: demand-driven
                          │ walks the cone via
  EDGE ACCESS      eager: ffi_facts (HashMap)   │   lazy: LookupSource → LMDB + two-tier L1/L2 edge cache
                          │ over
  GRAPH            category_parent edges   +   min_dist (distance-to-root: band selection + budget prune)
```

The **edge cache** (L1/L2) and the **boundary cache** are *orthogonal layers*: the
edge cache makes each parent lookup cheap; the boundary cache removes the *walk*
itself. They compose — the boundary win sits on top of a warm edge cache (philosophy
§3, measured).

### The axes (each an independent architectural choice)

| axis | options | default | knob |
|---|---|---|---|
| edge access | eager in-memory · lazy LMDB (+L1/L2 cache) | eager | `register_ffi_fact_pairs` / `register_lazy_lookup` |
| band (the cut) | whole region · entry frontier | frontier (for storage) | `boundary_band_root_near` / `boundary_band_entry_frontier` |
| precompute strategy | eager (enum / shared-memo / sweep) · lazy (demand-driven) | eager sweep | `build_boundary_suffix*` / `lazy_boundary_weightsum` |
| working set | unbudgeted · two-budget eviction · spill-and-continue | unbudgeted | `evict_budget` / `live_budget` / `..._with_spill` |
| payload (what the cache carries) | path-length histogram · moment jet `(mass,m₁,m₂,m₃)` + interval `(min,max)` · scalar distance-to-root | histogram | `build_boundary_suffix*` / `build_boundary_jets` / `build_boundary_distances` |
| representation | exact histogram · g_B pre-weighted basis · fitted (tail-prune/binomial/beta-binomial/mixture/quantised-CDF/discretised-GMM/moment-Normal) | exact | `build_boundary_basis_weighted_power` / `boundary_suffix_reprs` |
| persistence | ephemeral · LMDB histograms · LMDB fitted reprs | ephemeral | `save/load_boundary_basis` / `save/load_boundary_reprs` |
| result | scalar · effective_distance · distribution · shortest_distance · (mean/variance/skew · caret) | scalar | `boundary_result_extractor` / the extractor read |

They are genuinely independent — e.g. *lazy LMDB edges + entry-frontier band + fitted
reprs persisted to LMDB* is a valid configuration, as is *eager edges + region band +
exact in-memory*.

### Strategy classes (how this maps onto `RECURRENCE_EVALUATION_STRATEGY`)

The boundary cache is the **`cached`** strategy class — precompute the suffix relation
once, serve queries from it — applied to the *suffix* of the recurrence rather than
the whole answer. Within it:

- **Eager precompute** is a *fixed-point / bottom-up* materialisation of the suffix
  histograms over the cone (the topological sweep is semi-naïve evaluation of
  `H_v = Σ_p H_p[·-1]`).
- **Lazy precompute** is *per-query / top-down*: compute a node's suffix on first
  demand and memoise — demand-driven, only what is touched.
- **The query itself is a hybrid**: a lazy outer walk (`seed → boundary`, graph
  search) splicing a cached inner suffix (`boundary → root`). That is exactly the
  "lazy outer, eager inner" hybrid of the strategy taxonomy — the boundary cut is
  where the two meet.

So "eager vs lazy boundary cache" is not a new dichotomy; it is the same lazy/eager
*evaluation-order* axis the recurrence-strategy work defines, applied to the suffix.

### Composition rules (what fits with what)

- **Lazy/LMDB edges** ⇒ precompute with `build_boundary_suffix` (enumeration, walks via
  the `EdgeAccessor`) or run the **lazy** boundary cache; the shared-memo sweep needs
  the eager `ffi_facts` table.
- **Eager edges** ⇒ any precompute; `build_boundary_suffix_shared` / `_sweep` are the
  *polynomial* options (each node's `H` computed once).
- **Persistence** (LMDB `boundary_basis` / `boundary_basis_repr`) is independent of
  edge access — persist the side-table regardless of how it was built.
- **`g_B`** is the fixed-budget / fixed-functional hot path; **fitted reprs** are the
  large-budget storage play. Both are exact-or-certified, never silent approximation.

### Functional payloads — beyond the histogram (the graph-functional-semiring line)

The histogram is one payload; the cache can also carry the **cheap functionals** of the
path-length distribution *directly*, never forming the histogram. The why and the algebra
live in **`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`**; the operator surface is:

| you want | precompute | query / read | answers |
|---|---|---|---|
| mean / variance / skew of path length, or a cheap distribution reconstruction | `build_boundary_jets(band, root, edge_pred)` | `collect_native_category_ancestor_boundary_jet(...)` → `MomentJet`; `.mean()/.variance()/.skewness()`, `.to_normal_repr(..)` | the moment jet `(mass,m₁,m₂,m₃)` + interval `(min,max)`, budget-free, **exact on the acyclic ancestor space** |
| shortest hop-distance to root | `build_boundary_distances(band, root, edge_pred)` | `category_ancestor_boundary_distance(seed, root, acc)` → `u32`, or the `shortest_distance` extractor | the **cycle-correct** min-plus distance (the DFS histogram is unsound on cycles) |
| between-nodes distance | (the to-root distance cache above) | `caret_distance_upper(d_u, d_v)` / `caret_distance_lca(u, v, parents)` | the composite **caret** — an undirected *upper* bound (exact = tree distance); `directed_distance_lower` is the A* heuristic for *directed* queries (no undirected lower bound off a tree) |

All three carry only a handful of scalars per node and degrade gracefully (an empty cache
falls back to a correct full walk). They are the same boundary-cache architecture with a
different per-node element — see the note's `PathSemiring` framework.

---

## 0. The shape of a session

```
COMPILE          enable the lowering (optional — you can also call the kernels directly)
  └─ write_wam_rust_project([...], [boundary_optimization(true)], Dir)

RUNTIME (once at setup)
  1. load the graph        register_ffi_fact_pairs / register_lazy_lookup
  2. load distance-to-root set_min_dist            (drives band selection + pruning)
  3. pick a band           boundary_band_root_near | boundary_band_entry_frontier
  4. precompute            build_boundary_suffix* (eager)  OR  lazy_boundary_weightsum (on demand)
  (optional) compress / persist / build a g_B basis

RUNTIME (per query)
  splice                   collect_native_category_ancestor_boundary_hist  → a histogram
                           collect_native_category_ancestor_weightsum      → a scalar (g_B)
                           or the emitted 3-ary wrapper / foreign dispatch
```

---

## 1. Compiler: enabling the lowering (optional)

`boundary_optimization(true)` upgrades a detected `category_ancestor/4` kernel to a
3-ary `category_ancestor_boundary/3` foreign kernel and emits a public wrapper
`Pred(Cat, Root, Result)`:

```prolog
write_wam_rust_project([user:category_ancestor/4],
    [ module_name(myapp),
      boundary_optimization(true),          % off by default
      boundary_weight_n(2.0),               % the functional exponent N (default 2.0)
      boundary_result_extractor(scalar)     % scalar | effective_distance | distribution | shortest_distance
    ], 'output/myapp').
```

> `shortest_distance` returns the **cycle-correct shortest hop-distance to root** via the
> min-plus distance cache (precompute with `vm.build_boundary_distances(band, root_id,
> edge_pred)`, not `build_boundary_suffix`); with an empty cache it degrades to a plain
> correct BFS. The other three read the path-length histogram.

This is the *codegen* decision. At runtime the emitted wrapper splices against the
side-table you populate (§4). **With an empty side-table the kernel is still correct**
— it degrades to full enumeration — so the lowering is safe by default and the
speedup is unlocked once you precompute a band. You can skip the compiler option
entirely and call the runtime kernels directly (§5); the option just wires the
dispatch + wrapper for you.

---

## 2. Runtime setup — load the graph and distances

```rust
let mut vm = WamState::new(vec![], HashMap::new());

// (a) edges — eager (in memory) ...
vm.register_ffi_fact_pairs("category_parent", &[("5","4"),("4","0"), ...]);
// ... or lazy (LMDB-backed, for graphs too big for RAM):
//   vm.register_lazy_lookup("category_parent/2", Arc::new(my_lookup_source));
//   vm.set_int_native_edges(true);   // if node ids ARE the LMDB keys

// (b) distance-to-root: keyed by node id, value = min hops to root. Drives band
//     selection AND the kernel's budget prune. Load it like s2i/min_dist at setup.
vm.set_min_dist(&min_dist_map);       // HashMap<i32, i32>
```

`min_dist` is the only "extra" input — compute it once (a BFS from the root over the
reverse edges) and it powers both band selection and pruning.

---

## 3. Choosing the band

The band is the set of nodes whose `node→root` histograms you cache. **Any band is
exact** — it is purely a storage/coverage choice (spec §8).

| selector | what it caches | use when |
|---|---|---|
| `boundary_band_root_near(d_pre)` | the whole region `{1 ≤ min_dist ≤ d_pre}` | you want maximal coverage and storage isn't tight |
| `boundary_band_entry_frontier(d_pre, edge_pred)` | only the region's **surface** — nodes with a child *outside* the region | **the default** for the boundary-cache regime: storage scales with the cut, not the cumulative node count |

The kernel splices at the **first** cached node a seed reaches and stops, so periphery
seeds only ever use the entry frontier — `boundary_band_entry_frontier` caches just
that and is the storage-efficient choice (measurement addendum: band 38 vs region 313
at the same `D_pre`). Pick `d_pre` so the frontier sits **between** your seeds and the
dense core (a boundary cache wants its seeds in the periphery).

---

## 4. Precompute — fill the side-table

`boundary_suffix` (node → histogram) is the cache. Several ways to fill it:

| method | strategy | notes |
|---|---|---|
| `build_boundary_suffix(band, root, max_depth, edge_pred)` | per-node enumeration | simplest; works on **eager AND lazy/LMDB** edges (via the `EdgeAccessor`) |
| `build_boundary_suffix_shared(band, ...)` | shared-memo recurrence | polynomial (each node's `H` once); **eager-edge only**; returns `false` on lazy |
| `build_boundary_suffix_sweep(band, ..., evict_budget, live_budget)` | top-down topological sweep with two-budget eviction | bounded working set; returns `BoundarySweepStats`; eager-edge only |
| `build_boundary_suffix_sweep_with_spill(band, ..., evict_budget, live_budget, &mut sink)` | sweep + spill-and-continue | non-default; when the live frontier exceeds budget, spill to a `SpillSink` instead of stop-at-depth |
| `build_boundary_suffix_root_near(root, d_pre, max_depth, edge_pred)` | convenience: select root-near band + shared precompute (eager) with enumeration fallback | one call for the common case |
| `lazy_boundary_weightsum(seeds, root, max_depth, edge_pred, d_pre, n)` | **lazy** — compute each touched entry on first demand, memoize, splice | no precompute phase; only computes what the workload touches |

**Eager vs lazy (the strategy choice).** Eager amortises a *known, densely-reused*
band; lazy computes *only what's touched* and needs no precompute moment. Measured
(report lazy addendum): sparse/unknown workloads favour lazy at every query count;
a dense workload with a modest query count favours eager. Steady-state per-query cost
is identical (both splice). Lazy is also the strategy available on the lazy/LMDB edge
path.

**Budgets (`build_boundary_suffix_sweep`).**

- `evict_budget` (0 = unlimited): cap on resident memo entries; under pressure, evict
  **dead interior** nodes (`refs = 0`, non-band) **deepest-first** (keep root-near
  hubs). Never changes a result.
- `live_budget` (0 = unlimited): cap on the **non-discardable** frontier (`refs > 0`).
  When exceeded, the default is **stop-at-depth** (`stats.stopped_early`); supply a
  `SpillSink` to `..._with_spill` to **spill-and-continue** instead (exact paging).

---

## 5. Querying

```rust
let acc = vm.resolve_edge_accessor("category_parent");

// (a) full path-length histogram, then any functional:
let mut hist: Vec<u64> = Vec::new();
let mut visited = vec![seed];
vm.collect_native_category_ancestor_boundary_hist(seed, root, &mut visited, budget, &acc, 0, &mut hist);
let weight_sum = boundary_cache::f_weighted_power(&hist, n);   // d_eff = weight_sum.powf(-1/n)

// (b) the scalar WeightSum directly via the pre-weighted g_B basis (skip the
//     histogram + the powf loop) — build the basis once after precompute:
vm.build_boundary_basis_weighted_power(budget, n);
let mut ws = 0.0;
let mut vis = vec![seed];
vm.collect_native_category_ancestor_weightsum(seed, root, &mut vis, budget, &acc, 0, n, &mut ws);
```

`g_B` is a **fixed-budget, fixed-functional** specialisation — it bakes the budget and
`N` into one scalar per prefix length, so it serves only queries with that *same*
budget and `N`. For variable budgets / multiple functionals, keep the histogram form.

If you compiled with `boundary_optimization(true)`, just call the emitted wrapper
`Pred(Cat, Root, Result)` (or dispatch the foreign predicate) — it does (a) for you.

---

## 6. Approximation and persistence (optional)

All **opt-in**; the cache is exact unless you invoke these.

**Compress (storage):** for large-budget / deep-path graphs where histograms are long.

```rust
// rung 1 only (tail-prune) in place:
vm.compress_boundary_suffix(50 /* min_points trigger */, 0.001 /* ε_K certificate */);

// full ladder — choose the cheapest representation per node within ε_K:
let reprs = vm.boundary_suffix_reprs(50, 0.001);   // HashMap<u32, HistRepr>
```

The chooser (`boundary_cache::choose_representation`) tries, cheapest-first within the
`ε_K` gate: **exact → tail-pruned → binomial → beta-binomial → mixture →
quantised-CDF → discretised-GMM**. The gate is a **certified sup-CDF
(Kolmogorov-distance) tolerance** — `max_t |F_fit(t) − F_exact(t)| ≤ ε_K` against the
full exact CDF — *not* a Kolmogorov–Smirnov statistical test (no sample, null, or
critical value), so it is a deterministic bound, not a probabilistic one.
`choose_representation_budget(h, min_points, max_bytes)` is the storage-driven
complement (smallest error within a byte budget). The point count `min_points` is a
*work trigger* / storage proxy, not an acceptance gate — acceptance is always the CDF
certificate. A fit that misses `ε_K` is rejected back to exact, so accuracy never
silently degrades.

**Persist (don't recompute per run), LMDB:**

```rust
// histograms:
lmdb.save_boundary_basis(&table);                 // -> boundary_basis sub-db
let table = lmdb.load_boundary_basis()?;          // reload at setup
vm.set_boundary_suffix(&table);

// fitted representations (the storage win — a binomial node is ~21 bytes):
lmdb.save_boundary_reprs(&reprs);                 // -> boundary_basis_repr sub-db
let table = lmdb.load_boundary_reprs()?;          // decodes + EXPANDS to histograms
vm.set_boundary_suffix(&table);
```

`LmdbFactSource` also implements `SpillSink` (the `boundary_spill` sub-db) for the
mid-sweep spill of §4.

---

## 7. Cheat sheet (decision flow)

- **Just want it to work?** Compile with `boundary_optimization(true)`, then at setup:
  `set_min_dist` → `build_boundary_suffix_root_near(root, 2, budget, edge_pred)`.
- **Graph too big for RAM?** Lazy edges (`register_lazy_lookup` + `set_int_native_edges`)
  + `build_boundary_suffix` (enumeration, lazy-compatible) **or** `lazy_boundary_weightsum`.
- **Sparse / unknown query set?** Use `lazy_boundary_weightsum` (computes only what's
  touched).
- **Storage tight?** `boundary_band_entry_frontier` for the band, then
  `boundary_suffix_reprs` + `save_boundary_reprs` to persist fitted forms.
- **One fixed functional, hot path?** `build_boundary_basis_weighted_power` +
  `collect_native_category_ancestor_weightsum` (the ~ns dot-product splice).
- **Precompute frontier won't fit in memory?** `build_boundary_suffix_sweep_with_spill`
  with an LMDB or in-memory `SpillSink`.

---

## 8. Where everything lives

| concern | code | tests |
|---|---|---|
| splice kernels, band selection, precompute, lazy, sweep+eviction+spill, g_B, compress | `templates/targets/rust_wam/state.rs.mustache` | `boundary_kernel_tests` (in `state.rs.mustache`) |
| histogram ops, splice identity, error metrics, fitting ladder, repr (de)code | `templates/targets/rust_wam/boundary_cache.rs.mustache` | its `#[cfg(test)] mod tests` |
| LMDB persistence + `SpillSink` | `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache` | `tests/test_wam_rust_boundary_{basis,repr,spill}_lmdb.pl` |
| compiler lowering / options | `src/unifyweaver/targets/wam_rust_target.pl`, `src/unifyweaver/core/recursive_kernel_detection.pl` | `tests/test_wam_rust_boundary_{lowering,foreign_dispatch}.pl` |
| foreign dispatch + integrated scale | — | `tests/test_wam_rust_boundary_{kernel_exec,integrated_scale}.pl` |
| measurement harnesses | `examples/benchmark/wam_rust_boundary_{measurement,lazy_edge_measurement}.pl`, `examples/benchmark/boundary_splice_complexity_bench.rs` | — |
| design | `WAM_RUST_BOUNDARY_DISTRIBUTION_{PHILOSOPHY,SPECIFICATION,CACHE_PLAN}.md`, `WAM_RUST_BOUNDARY_MEASUREMENT_2026-06-16.md` | — |
