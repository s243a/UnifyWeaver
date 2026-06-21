# WAM-Rust Boundary Distribution Optimization — Implementation Plan

Status: in progress (P1, P2a, P2c-parity, P2c-wiring/dispatch, P2c-wiring/lowering, P2c-wiring/precompute/selection, P2c-wiring/precompute/eviction, P2c-wiring/precompute/persistence, P3-measurement, P4-g_B-basis, P4-entry-frontier, P4-approx-rung1, P4-approx-rung2-binomial, P4-approx-cdf-fit, P4-approx-mixture, P4-approx-budget-mode, P4-repr-persistence, lazy-boundary-cache, lmdb-lazy-edge-measurement, eviction-spill, P4-approx-beta-binomial, P4-approx-quant-cdf, P4-approx-disc-gmm DONE). A **separate workstream** generalizes the cache from the path-length histogram to the *functionals* of it (moment jet, min-plus distance, composite caret) — tracked in **`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`** §8 (increments 1a–1d, 2a–2d, 3a DONE). The implementation-plan member of
the design trio: **`WAM_RUST_BOUNDARY_DISTRIBUTION_PHILOSOPHY.md`** (why — a
disablable complexity-reduction compiler optimization, caching secondary),
**`WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md`** (precise semantics, the
scalar/histogram result-mode family, invariants, interface), and this plan (phased
work + status); plus **`WAM_RUST_BOUNDARY_DISTRIBUTION_HOWTO.md`** (the operator's
manual — how to use it) and **`WAM_RUST_BOUNDARY_MEASUREMENT_2026-06-16.md`** (the
measured results). Also companion to `DISTRIBUTION_CACHE_BENCHMARK_PLAN.md`,
`DISTRIBUTIONAL_COMPRESSION_THEORY.md`, `RECURRENCE_EVALUATION_STRATEGY_*.md`, and
the EnWiki splice-validation reports.

It ports the **boundary-splice / suffix-distribution** approach (prototyped and
validated correct in Python by the `codex/*` line) into the Rust WAM graph-search
target, as a gated compiler optimization. It deliberately **re-derives the cost
tradeoffs for compiled Rust** rather than inheriting the Python prototype's
performance conclusions — because, as with the edge cache, several of those
conclusions are interpreter-overhead artifacts, not properties of the algorithm.

## 1. What the boundary distribution is, and why the splice is valid

The effective-distance query aggregates over all bounded paths from a seed to a
root up `category_parent`:

```
d_eff = (Σ_paths  Hops(path)^(-N))^(-1/N)        (effective_distance.pl:90,108-114)
WeightSum = Σ_paths Hops^(-N)
```

`WeightSum` is a **linear functional of the path-length histogram**
`H[L] = #{paths of length L}`: `WeightSum = Σ_L H[L]·L^(-N)`
(cf. `weighted_power_cdf`, `tools/distribution_cache_support.py:303-304`).

Path-length histograms compose by **convolution** over a cut node B (boundary):
every seed→root path through B is a seed→B prefix path concatenated with a B→root
suffix path, and lengths add:

```
H_seed→root[L] = Σ_{a+b=L} H_seed→B[a] · H_B→root[b]            (boundary splice)
```

So if we precompute the **suffix** histogram `H_B→root` at boundary nodes B near
the root, a query that walks up and reaches B can **splice** the cached suffix
over its remaining budget instead of continuing to enumerate B→root for every
seed. The Python prototype proved this **exact** on EnWiki MTC — zero delta on
mass, value-sum, and mean length vs full DFS
(`enwiki_mtc_shallow_splice_validation_2026-06-14.md`). The algorithm is correct;
this plan ports it.

## 2. What does NOT transfer from the Python prototype (the interpreter-overhead trap)

The same dynamic we already proved on the **edge** cache applies here. The Python
edge-cache benchmark concluded "cached is *slower*" (ratio 1.66); the compiled
Rust port measured cached **2.5–3.5× faster** (`wam_rust_cached_scaling_sweep`),
because the residual that dominated Python was interpreter overhead, not algorithm
cost. The boundary-cache prototype shows the **identical signature**:

| Python conclusion | Source | Cost basis | Transfers to Rust? |
|---|---|---|---|
| "Cached search slower (ratio 1.66) despite many hits" | `enwiki_mtc_boundary_cache_runtime_attribution` | **75.6% unattributed traversal/interpreter overhead**; splice 0.1%, decode 0.4% | **No** — pure Python residual |
| "Still ~11.6% slower after decode optimisation" | `enwiki_mtc_boundary_cache_hit_overhead` | per-node Python recursion/bookkeeping | **No** |
| "Fewer nodes (0.82×) but slower (1.33×)" | `..._boundary_aggregate_root_cone_b2_t4...` | Python per-node cost > node savings | **No** — the node savings are real |
| Splice/decode are negligible | attribution buckets (0.1% / 0.4%) | algorithm cost | **Yes** (and even cheaper in Rust — see §3) |
| Boundary-hit fraction ~24% on EnWiki MTC | coverage probe | graph structure / reuse | **Yes** (durable) |
| 50-point exact→approximate threshold | `DISTRIBUTIONAL_COMPRESSION_THEORY` | **storage** bytes | **Yes** (durable; see §4) |
| `D_pre` / marginal-speedup crossovers | `DISTRIBUTION_CACHE_BENCHMARK_PLAN` ("hypotheses, not commitments") | **Python wall-time** — overhead so large the curves are unreadable | **No** — must be re-measured in Rust |

**Conclusion:** keep the algorithm, the storage-based decisions, and the
structural reuse facts. Re-derive every wall-time/compute crossover for Rust.

## 3. The sharp Rust criterion (the non-obvious part)

In Rust the boundary cache does **not** automatically win — and not for Python's
reasons. The kernel DFS is already native (`collect_native_category_ancestor_hops`,
`wam_rust_target.pl:2433-2499`) and, after the edge cache, parent lookups are
~L1-cache reads. So in Rust **both** the suffix walk *and* the splice are cheap.
The cache wins iff:

```
(native DFS work avoided in the B→root cone) × (reuse across seeds)  >  splice cost
```

The splice cost is essentially zero. Measured here: caching the suffix as a
**pre-weighted vector** `g_B[a] = Σ_b H_B→root[b]·(a+b)^(-N)` makes the query-time
splice a dot product `WeightSum += Σ_a H_seed→B[a]·g_B[a]` —

```
micro-bench (this machine): query-time dot-product splice = 1.23 ns
                            (raw-histogram convolve + powf  = 5.85 µs;
                             Python dict convolve            = 35 µs)
```

So the criterion reduces to: **the cached suffix cone must represent enough avoided
native DFS, reused enough times, to beat ~1 ns.** That is overwhelmingly true for
**root-near** boundaries (the B→root cone is the whole shared upper graph, traversed
by *every* seed — exactly the hub-reuse the cache-scaling work measured) and a wash
for deep boundaries (tiny cone, low reuse). Two consequences that differ from the
Python framing:

- **Cache only shallow / root-near boundaries.** The optimal `D_pre` is likely
  *shallower* in Rust than the Python curves suggest: Python overhead made even
  small caches look marginal, so its "diminishing returns past D_pre 3–4" is
  overhead-shaped. In Rust only the top reuse band clears the ~1 ns splice — but it
  clears it by orders of magnitude.
- **The boundary cache is a layer *on top of* the edge cache, not a replacement.**
  The edge cache removes LMDB-seek cost but each seed still *walks* the suffix cone
  (native DFS over cached edges). The boundary cache removes the walk itself. So its
  remaining headroom is exactly the native-DFS time left after the edge cache — to
  be measured, not assumed.

## 4. Representation: pre-weighted vector, exact by default

Cache, per boundary node and per hot functional, the **cumulative basis** vector
`g_B[a]` for `a = 0..max_depth` (the "cumulative bases" of
`DISTRIBUTION_CACHE_BENCHMARK_PLAN.md` — `mass`, `moment(1)`, `weighted_power(N)`).
For `d_eff` that is `g_B[a] = Σ_b H_B→root[b]·(a+b)^(-N)`, truncated at the budget.
This is:

- **Cheaper at query time** than caching the raw histogram and shift-adding
  (`add_shifted_hist`, `lmdb_parent_boundary_cache_benchmark.py:517-525`): a dot
  product, no convolution, no `powf`.
- **Smaller**: one `f64` vector of length `≤ max_depth` (~10–20) per boundary per
  functional, vs a variable-support histogram.

Because the exact splice is ~1 ns, the **exact→approximate ladder is justified by
storage only** (the 50-point/byte argument), never by compute. The Rust default is
**exact** `g_B`; fall back to a fitted form (binomial / discretised GMM, per the
compression theory) only when a node's basis exceeds the storage budget — decoupling
the two concerns the Python policy had to entangle.

(Keep the raw suffix histogram cacheable too, for cold/ad-hoc functionals; the
`g_B` vectors are the hot-path optimisation.)

### 4a. Two convolution modes: full vs budget-specialised

There is a real tradeoff between caching the **raw suffix histogram** and the
**pre-weighted `g_B`** vector, and the right default depends on query shape:

- **Raw histogram → full (truncated) convolution.** Splice yields the entire
  `H_seed→root` length distribution (up to budget), from which **any functional
  and any budget** can be read afterward — "compute the convolution once, then
  take the part of the distribution you care about." This is the general form and
  amortises when a node is queried under several functionals/budgets. It is also
  the form whose correctness is easiest to validate (P1 below). The aggregate is a
  *family* of weighting functions (`mass`, `moment(1)`, `weighted_power(N)`, …),
  not one — caching the histogram keeps all of them available.
- **Pre-weighted `g_B` → dot product.** Specialised to one `(functional, budget)`;
  cheapest at query time (~1 ns) but must be rebuilt per functional/budget.

So: cache the **histogram** by default (general, multi-functional); derive `g_B`
only for hot `(functional, budget)` pairs.

Two structural notes for the general case (when supports grow or many boundaries
combine):

- **FFT.** A length-`m` truncated convolution is `O(m²)` direct; for the small
  `m ≤ budget` (~10–20) here, direct wins (no FFT setup cost). But if supports grow
  (deep budgets, fitted continuous tails) or a query composes **several** boundary
  suffixes, the convolutions become an `O(m log m)` FFT / frequency-space multiply —
  worth it past a crossover to measure, not assume.
- **Multiple boundaries = a system, one boundary = a single equation.** With a
  single cut node the splice is one convolution (one multiply in frequency space,
  sub-quadratic). With several boundary cut nodes on the seed→root frontier it is a
  linear system over the per-boundary suffix distributions (matrix form in
  frequency space); the P1 enumeration handles the multi-boundary case by summing
  per-boundary spliced contributions, which is the explicit (non-FFT) solution of
  that system.

## 5. Rust integration (reuses existing scaffolding)

From the runtime map, the plug-in points already exist:

1. **Storage** — add a `boundary_basis` LMDB sub-db (node id → packed `g_B` /
   histogram bytes), alongside `s2i/i2s/category_parent/category_child`
   (`lmdb_fact_source_heed.rs.mustache:42-84`). Load once into a side-table
   (`HashMap<i32, Box<[f32]>>`), mirroring `load_s2i`. Built by a new ingest pass
   (the recurrence in `DISTRIBUTIONAL_COMPRESSION_THEORY.md:15-27`, computed in
   Rust or written by the existing Python precompute and just *read* by Rust).
2. **Kernel** — a `category_ancestor_boundary` native kind: walk up as today, but
   when a node is in the boundary side-table, **splice `g_B` and stop** instead of
   recursing. Register via `kernel_native_kind/2` +
   `recursive_kernel_detection.pl:54-68`; add the match arm in
   `execute_foreign_predicate` (`wam_rust_target.pl:1894-2346`); reuse the existing
   `min_dist` A* prune (`:2451-2460`).
3. **Result** — emit the same aggregate the current kernel does (so it drops into
   the existing `effective_distance` pipeline unchanged), gated so default output
   is identical to today.
4. **Value/side-table** — no `Value` change needed; the basis lives in a native
   side-table (like `min_dist`), not as heap `Value::List`.

Phasing:

- **P1 — splice identity in Rust [DONE].** A self-contained runtime module,
  `templates/targets/rust_wam/boundary_cache.rs.mustache`, implements the exact
  histogram form: `suffix_histogram` (the recurrence), `splice_truncated` (the
  truncated convolution), and the linear functionals (`f_mass`, `f_moment1`,
  `f_weighted_power`). Its `#[cfg(test)]` proves **boundary-spliced histogram ==
  full-DFS histogram**, bin-for-bin, across multiple boundary cut-sets and budgets
  (hence all functionals match) — the Rust analog of the Python exact-match splice
  validation. Compiled into every generated crate (`pub mod boundary_cache`) and
  verified via `cargo test --lib`. Deliberately the histogram (general) form, not
  the specialised `g_B`, so it validates all functionals at once (§4a). No kernel
  is wired to it yet — that is P2.
- **P2 — make it live.**
  - **P2a [DONE].** The runtime boundary kernel on `WamState` (`state.rs`):
    `boundary_suffix` side-table + `set_boundary_suffix`; `enum_ancestor_hist`
    (full enumeration, an exact mirror of the production
    `collect_native_category_ancestor_hops` walk); `collect_native_category_
    ancestor_boundary_hist` (splice-and-stop at boundary nodes, `d+b<=max_depth`
    truncation matching the production budget at the cut); `build_boundary_suffix`
    (precompute). Unit-tested through the real `register_ffi_fact_pairs` /
    `edge_parents_via` path (boundary splice == full enumeration).
  - **P2c-parity [DONE].** Exec test `test_wam_rust_boundary_kernel_exec.pl`:
    in a generated crate, the boundary kernel's `weighted_power` aggregate equals
    the **production** `collect_native_category_ancestor_hops` aggregate across
    seeds (incl. deep ones through a root-near boundary band) — closing the gap
    that P2a compared only against an in-module oracle.
  - **P2c-wiring/dispatch [DONE].** The boundary kernel is now reachable as a
    **foreign kernel** through `execute_foreign_predicate`: a
    `"category_ancestor_boundary"` dispatch arm in
    `compile_execute_foreign_predicate_to_rust` reads `A1`/`A2` (cat/root) and the
    `max_depth`/`edge_pred`/`weight_n`/`result_extractor` foreign config, runs
    `collect_native_category_ancestor_boundary_hist` over the cached side-table, and
    emits **one** result via `finish_foreign_results` **`deterministic`** mode
    (`tuple(1)`, no choice point). **Result-mode family** (spec §5): the kernel
    produces the histogram `H`; the `result_extractor` selects `scalar`
    (`f(H)`, e.g. `weighted_power`), `effective_distance` (`d_eff = WeightSum^(-1/N)`),
    or `distribution` (the histogram itself, `Value::List` of counts). It does *not*
    re-expand `H` into a hop stream (that re-introduces the exponential) — so the
    wiring adds extractors over one histogram kernel, not a scalar-only predicate.
    Exec test `test_wam_rust_boundary_foreign_dispatch.pl`: in a generated crate,
    registering the kernel as a foreign predicate and driving it through
    `execute_foreign_predicate` reproduces the **production** kernel's aggregate for
    both `scalar` and `effective_distance` across seeds.
  - **P2c-wiring/lowering [DONE].** The compiler now *chooses* this lowering under
    the gate `boundary_optimization(true)` (default off), mirroring the
    `kernel_mode(bidirectional)` opt-in upgrade. A detected `category_ancestor`
    kernel is upgraded to a `category_ancestor_boundary` kernel
    (`rust_maybe_upgrade_boundary/3`): the registry in
    `recursive_kernel_detection.pl` gains the boundary metadata (native kind,
    `tuple(1)` layout, **`deterministic`** result mode, arity 3), the existing
    `emit_kernel_registration/1` auto-generates the
    `register_foreign_native_kind(..., category_ancestor_boundary)` + `max_depth` /
    `edge_pred` / `weight_n` / `result_extractor` config, and a public 3-ary wrapper
    `Pred(Cat, Root, Result)` (`rust_boundary_wrapper_code/3`) is emitted.
    Tunables: `boundary_weight_n(N)` (default 2.0), `boundary_result_extractor`
    (`scalar` | `effective_distance` | `distribution`, default `scalar`). Exec test
    `test_wam_rust_boundary_lowering.pl`: the default keeps the 4-ary streaming
    kernel; the option upgrades to the 3-ary deterministic kernel; tunables flow
    through; and the emitted wrapper reproduces the production aggregate in a built
    crate. The boundary side-table (`build_boundary_suffix`) remains a separate
    precompute — with an empty side-table the kernel degrades to full enumeration
    (still correct), so the lowering is correct by default and the speedup is
    unlocked once boundary nodes are precomputed.
  - **P2c-wiring/precompute/selection [DONE].** Band *selection* is now automatic:
    `WamState::boundary_band_root_near(d_pre)` reads the distance-to-root table
    (`min_dist`, loaded at setup like `s2i`) and returns the root-near band
    `{ n : 1 <= min_dist(n) <= d_pre }` (§3); `build_boundary_suffix_root_near(root,
    d_pre, max_depth, edge_pred)` selects that band and precomputes + installs its
    suffix histograms in one call — the setup hook a harness calls after edges +
    `min_dist` are loaded. An empty `min_dist` yields an empty band → full
    enumeration (still correct). Unit tests (`boundary_kernel_tests`): the selection
    matches the expected band, the precompute populates the side-table, and the
    splice matches full enumeration; the lowering exec test
    (`test_wam_rust_boundary_lowering.pl`) drives the emitted 3-ary wrapper through
    the root-near selection path end-to-end. Any band is exact for the splice
    (kernel splices at the first band node and stops), so root-near is a
    coverage/performance choice, not a correctness one.
  - **P2c-wiring/precompute/eviction [DONE].** `build_boundary_suffix_sweep(band,
    root, max_depth, edge_pred, evict_budget, live_budget)` runs the precompute as a
    root-down topological (Kahn) sweep over the band's ancestor cone — `H_root =
    δ_0`, `H_v[L] = Σ_{p ∈ parents(v)} H_p[L-1]` — with the **liveness-prioritised
    eviction** of spec §8a/§8b. A per-node consumer count `refs(p)` marks a node
    **dead** once all its in-cone children are computed; the two budgets bound the
    working set: `evict_budget` evicts **dead interior** nodes (`refs = 0 ∧ p ∉ Bset`)
    **deepest-first** when the resident memo exceeds it (root-near hubs kept longest;
    a dead node has no future reader so this never changes a result), and
    `live_budget` caps the non-discardable frontier (`refs > 0`) — when exceeded the
    sweep **stops deepening** (`stopped_early`), freezing `D_pre` and leaving deeper
    band nodes uncached (they fall back to enumeration, still correct). Returns
    `BoundarySweepStats` (computed / retained / evicted / peak_resident /
    stopped_early) for the §6 measurement. The cone is the *full* upward closure (not
    `min_dist`-bounded), since a band node's complete histogram can include
    long-detour routes through parents whose own shortest distance exceeds the band
    depth. Tests: unbudgeted == enumeration table + full-enum splice; tight
    `evict_budget` evicts a dead interior yet leaves band results identical; tiny
    `live_budget` forces stop-at-depth with partial caching still correct; eager-only
    no-op. Eager-edge only; the lazy/LMDB path keeps the enumeration precompute.
  - **P2c-wiring/precompute/persistence [DONE].** The precompute persists to a
    `boundary_basis` LMDB sub-db (u32 node -> packed histogram) so it is **not
    repeated per run**: `LmdbFactSource::save_boundary_basis(table)` writes it in one
    transaction and `load_boundary_basis()` reads it back at setup (empty when the
    sub-db is absent -> kernel degrades to full enumeration). The packing format is
    crate-independent `boundary_cache::encode_hist`/`decode_hist`
    (`[len: u32 LE][counts: u64 LE * len]`), shared by both LMDB backends and
    unit-tested without LMDB. Tests: `encode_decode_hist_roundtrip` (boundary_cache)
    and `test_wam_rust_boundary_basis_lmdb.pl` — in a generated lmdb_zero crate,
    save -> load -> fresh re-open all return the identical table (cross-run
    persistence). lmdb_zero backend; heed parity is a follow-up.
  - **Lazy (demand-driven) precompute [DONE].** `lazy_boundary_weightsum` fills the
    band **on first demand** (per-query / top-down) instead of eagerly up front —
    only the band-entry nodes the workload touches are computed (spec §8d). Identical
    results to eager; best for sparse/unknown workloads, streaming queries, or when
    `D_pre` is hard to pick, and it is the strategy available on the lazy/LMDB edge
    path (enumerates via the `EdgeAccessor`, no `ffi_facts` needed). Measured (it
    depends on **workload sparsity × query count K**, steady state being splice-
    identical): **sparse** workloads favour lazy at every K; **dense + modest K**
    favours eager (batched precompute beats on-demand warmup); **dense + large K**
    tips back to lazy; bigger datasets shift toward lazy. Guarded by
    `lazy_boundary_caches_on_demand_and_matches_eager`; the harness prints the
    per-config winner.
  - **P2c-wiring/precompute/eviction/spill [DONE, NOT default].** When the live
    frontier exceeds its budget, `build_boundary_suffix_sweep_with_spill(..., spill:
    &mut dyn SpillSink)` spills live entries (deepest-first) to the sink and reloads
    them on demand — turning the §8b stop-at-depth into spill-and-continue-against-
    storage, so the whole cone is swept (`stopped_early` false, `spilled` counts the
    writes). **Off by default** (the plain `build_boundary_suffix_sweep` passes no
    sink); engages only when a sink is supplied AND the in-memory live budget is
    exceeded; falls back to stop-at-depth otherwise. `SpillSink` backends: an
    in-memory map and the `boundary_spill` LMDB sub-db (`impl SpillSink for
    LmdbFactSource`). Exact paging — results match the unbudgeted sweep. Tests:
    `sweep_spill_completes_and_matches` (in-memory: no-sink stops early, with-sink
    completes + spills + identical results) and cargo-gated
    `test_wam_rust_boundary_spill_lmdb.pl` (the LMDB backend end to end).
- **P3 — the measurement in §6 [DONE].** Measured on the **real emitted kernels**
  (`build_boundary_suffix_sweep` + `collect_native_category_ancestor_boundary_hist`
  vs the production `collect_native_category_ancestor_hops`), harness
  `examples/benchmark/wam_rust_boundary_measurement.pl`, report
  `WAM_RUST_BOUNDARY_MEASUREMENT_2026-06-16.md`. Result: **yes — the boundary cache
  adds wall-time on top of the warm edge cache, and the crossover is shallow** —
  ~3× at `D_pre=1`, 16–26× at `D_pre=2`, >100× at `D_pre=3` on a dense-core graph;
  precompute is sub-ms (amortizes within a single 500-seed batch); and the boundary
  aggregate equals production **exactly** at every `D_pre` (the §6 exact-match
  invariant, guarded by `tests/test_wam_rust_boundary_integrated_scale.pl`). The
  shallow crossover confirms philosophy §3 (Python overhead had masked it).
  - **LMDB lazy-edge path [DONE].** `wam_rust_boundary_lazy_edge_measurement.pl`
    writes the synthetic graph into a real LMDB env and registers it as the lazy
    `category_parent/2` `LookupSource` (each lookup an LMDB seek through the L1/L2
    edge cache), measuring production vs boundary there side by side with the eager
    path. Result: production is ~4–5× slower on LMDB (85–108 ms vs 17–23 ms), and the
    boundary win **persists and is often larger** (20–25× lazy vs 10–25× eager) —
    the cache removes the *walk*, so the more each avoided lookup costs, the more it
    saves (~82 ms removed per batch on LMDB vs ~16 ms in memory). Exact on both paths.
    Closes the §6 "to be confirmed on the LMDB run" caveat.
  - Boundary-hit-fraction vs `D_pre` remains a minor open measurement.
- **P4 — storage-gated approximate/specialised bases.**
  - **`g_B` pre-weighted basis [DONE].** `build_boundary_basis_weighted_power(max_depth,
    n)` collapses each cached histogram into `g_B[a] = Σ_b H_B[b]·(a+b)^(-N)`, and
    `collect_native_category_ancestor_weightsum` accumulates WeightSum directly (a
    dot-product splice — no convolution, no final `powf` loop). **Budget precondition
    (§4a "budget-specialised"):** `g_B` bakes in the budget and `N`, so it serves
    only queries with that *same fixed* budget and functional; variable budgets /
    multiple functionals keep the histogram (`boundary_suffix`), the general
    budget-flexible form. Validated `g_B` WeightSum == histogram `weighted_power` ==
    full enumeration.
  - **Entry-frontier band [DONE].** `boundary_band_entry_frontier(d_pre, edge_pred)`
    caches only the region's *surface* — nodes with `1 ≤ min_dist ≤ d_pre` that have a
    child *outside* the region — instead of the whole `boundary_band_root_near`
    *volume*. Since the kernel splices at the first cached node and stops, periphery
    seeds only use this cut, so storage scales with the region surface, not the
    cumulative root-near node count (measured: band 38–83 vs region 313–711 at the
    same `D_pre`, same-order speedup at the intended periphery-seed operating point).
    Any band is exact, so this is a storage/coverage choice. The whole-region band
    remains for maximal coverage when storage is not the constraint.
  - **Approximation ladder, rung 1 [DONE].** Ported the Python distribution-cache
    policy (spec §9a): the support count is a **work trigger** (default 50), NOT an
    acceptance gate — acceptance is the CDF **error certificate** (`ε_K` Kolmogorov,
    default 0.001; `ε_W1` Wasserstein available). `boundary_cache::{cdf_max_error,
    cdf_w1_error, tail_prune, compress_histogram}` + the opt-in
    `WamState::compress_boundary_suffix`. Tail-pruned-exact is rung 1 (the dropped
    fraction *is* the Kolmogorov error). OPT-IN: the cache stays exact unless called;
    triggers only for large-budget/deep-path histograms (support > min_points).
  - **Approximation ladder, rung 2 — parametric binomial [DONE].** `fit_binomial`
    (method of moments), `binomial_pmf` (stable recurrence), a `HistRepr`
    (`Exact` | `Binomial`) with `bytes`/`pmf`/`expand`, and `choose_representation`
    — the cheapest representation (exact / tail-pruned / binomial) within the `ε_K`
    CDF certificate (mirrors Python `choose_distribution_representation`). The gate
    is a hard reject: validated that a true binomial is fit + chosen (smaller) and a
    bimodal histogram is rejected back to exact. Bounded integer path-length data
    favour binomials.
  - **CDF-space fit [DONE].** `fit_binomial_cdf` chooses `p` by minimising the
    Wasserstein-1 (integral CDF) distance via a 1-D search, instead of moment-
    matching — optimising the same cumulative space the gate judges in (smoother /
    lower-bandwidth). `choose_representation` tries both fits and keeps the
    lower-error one. Validated: the CDF fit is ≤ the moment fit on W1 and strictly
    better when the mean is pulled by local PMF contamination.
  - **Mixture of binomials + multi-objective choice + persistence [DONE].**
    `fit_binomial_mixture` (EM, shared trials) + `HistRepr::Mixture` fit the
    multimodal nodes a single binomial rejects (validated: a true 2-binomial mixture
    is fit and chosen while a single binomial misses the gate).
    `choose_representation_budget` adds the **storage-driven** complement to the
    error-driven `choose_representation` — error is not always the binding
    constraint; `min_points` is a storage proxy, not an intrinsic error gate (need
    not be 50). End-to-end storage win wired: `WamState::boundary_suffix_reprs` +
    `encode_repr`/`decode_repr` + `LmdbFactSource::save_boundary_reprs` /
    `load_boundary_reprs` (the `boundary_basis_repr` sub-db; expand on load).
    Cargo-gated `test_wam_rust_boundary_repr_lmdb.pl`: a binomial-shaped node persists
    as ~21 bytes and reloads within `ε_K`; a short node round-trips exactly.
  - **Beta-binomial (over-dispersion) [DONE].** `fit_beta_binomial` (method of
    moments via the intra-class over-dispersion `rho`) + `beta_binomial_pmf` (stable
    ratio recurrence, no `lgamma`) + `HistRepr::BetaBinomial` fit a unimodal
    *over-dispersed* node (variance above a binomial's `n*p*(1-p)`) that a single
    binomial rejects but which a mixture would over-model — 3 params, cheaper than a
    mixture. Validated: a true beta-binomial is recovered by MoM, the single binomial
    misses the gate, and the chooser selects `BetaBinomial` over the costlier mixture.
    `encode_repr`/`decode_repr` tag 3.
  - **Quantised-CDF table [DONE].** `HistRepr::QuantCdf{qcdf,total}` stores the exact
    CDF as one `u16` per support point (`quantize_cdf`/`dequantize_cdf`); always
    admissible (error ≤ `2^-16`) at ~1/4 the bytes of raw counts, so it is the
    fallback when no parametric form fits an irregular/multi-spike node (also O(1)
    prefix-mass). `encode_repr`/`decode_repr` tag 4. Validated:
    `quant_cdf_is_the_no_fit_fallback` (five spikes — binomial/mixture rejected, the
    chooser falls back to QuantCdf, smaller than exact, within `ε_K`).
  - **Discretised-GMM [DONE].** `HistRepr::DiscGmm{support,comps,total}` —
    `fit_discretised_gmm` (EM over the integer support) fits free-`(weight,mu,sigma)`
    components, so it places the narrow interior modes the mean-coupled binomial
    families cannot. Most params/mode of any rung, so the chooser picks it only when
    every cheaper rung misses the gate AND it still undercuts the quant-CDF table on
    bytes. `encode_repr`/`decode_repr` tag 5. Validated:
    `disc_gmm_fits_narrow_interior_modes` (two σ=1.5 modes at 20/40 — binomial, beta-
    binomial, and K≤3 binomial mixture all rejected; the GMM fits and is chosen).
    **The fitted histogram-approximation ladder is now closed structurally (Rungs 1–6)**
    — every *histogram-representation* form is in place. (A separate, cheaper moment-jet
    / CLT *reconstruction* rung — carry `(M,m₁,m₂)`, no EM — remains unbuilt and is the
    first increment in `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` §7–§8; it is not part of
    this fitted ladder.) "Closed" is structural, not a claim that a measured workload has
    been shown to need Rung 6.

## 6. What to measure (re-derive, don't inherit)

Using the synthetic generator (`generate_synthetic_category_graph.py`) and the
cached/lazy harness, add a **boundary-cached** variant and measure, at enwiki-ish
scale, the three-way `lazy` vs `edge-cached` vs `edge-cached + boundary-cached`:

- Does the boundary cache add measurable wall-time **on top of** the edge cache,
  and from what `D_pre`? (The Rust crossover, expected shallower than Python's.)
- Boundary-hit fraction vs `D_pre` (durable from Python ~24%, confirm in Rust).
- Correctness invariant: boundary-cached output == edge-cached output, exactly
  (the Rust analog of the Python exact-match validation).

## 7. Risks / open questions

- **Headroom may be modest.** After the edge cache, the suffix walk is cheap native
  code; the boundary cache's win is bounded by that residual. P3 measures it before
  P4 is worth building. (This is the honest version of "Rust makes the cache win" —
  it might be a small win here precisely because Rust already made the *edge* cache
  win.)
- **Cycle / filter policy must match** between precompute and query (the Python
  validation's exactness depended on identical root, filter, and cycle handling) —
  port those invariants, not just the splice.
- **Multi-functional caching** multiplies storage by #hot-functionals; gate by the
  same demand/storage budget the edge cache uses (R8b).
