# WAM-Rust Boundary Distribution — §6 Measurement (2026-06-16)

The §6 measurement of `WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md`: does the
boundary distribution cache add measurable wall-time **on top of the (warm) edge
cache**, and from what `D_pre` — measured on the **real emitted kernels** in a
generated crate, not a std-only re-implementation.

Harness: `examples/benchmark/wam_rust_boundary_measurement.pl` (generates a crate,
builds `--release`, runs `examples/boundary_measure.rs`).

## What is compared

| | kernel | edge access |
|---|---|---|
| **baseline** | `collect_native_category_ancestor_hops` → `weighted_power` | eager `HashMap` (warm edge cache) |
| **boundary** | `build_boundary_suffix_sweep(root-near band @ D_pre)` once, then `collect_native_category_ancestor_boundary_hist` → `weighted_power` | same eager edges |

So the baseline **is** the edge-cached path (parent lookups are warm-cache hits);
the reported speedup is the boundary win *on top of* the edge cache — exactly the
§6 question. Graph: a dense core (diamonds → exponentially many seed→root paths,
capped at budget=8) behind a thin boundary cut; 500 periphery seeds attach only to
the cut. `N=2`, root-near band selected from `min_dist` via
`boundary_band_root_near(D_pre)`.

## Results (this machine, release)

```
config           Dpre   prod_ms  bound_ms    pre_ms   speedup   band   peakR   eq
core=120  cp=3      1     17.66     5.499     0.055      3.2x      8      34  yes
core=120  cp=3      2     17.66     0.667     0.160     26.5x     65     112  yes
core=120  cp=3      3     17.66     0.158     0.385    111.6x    313     322  yes
core=120  cp=3      4     17.66     0.119     0.661    148.3x    599     600  yes
core=180  cp=3      1     21.75     7.997     0.112      2.7x     11      70  yes
core=180  cp=3      2     21.75     1.113     0.271     19.5x    121     223  yes
core=180  cp=3      3     21.75     0.195     0.407    111.4x    342     362  yes
core=180  cp=3      4     21.75     0.152     0.672    142.6x    643     644  yes
core=240  cp=3      1     23.28     8.615     0.115      2.7x     11      70  yes
core=240  cp=3      2     23.28     1.441     0.186     16.2x     71     133  yes
core=240  cp=3      3     23.28     0.184     0.414    126.7x    335     364  yes
core=240  cp=3      4     23.28     0.161     0.799    144.6x    711     712  yes
```

`prod_ms` = production over 500 seeds; `bound_ms` = boundary query over 500 seeds;
`pre_ms` = one-time sweep precompute; `band` = retained band size; `peakR` = sweep
peak resident memo entries; `eq` = boundary aggregate == production exactly.

## Findings

1. **Yes — measurable win on top of the edge cache, and the crossover is shallow.**
   Even `D_pre=1` is ~3× faster; `D_pre=2` is 16–26×; `D_pre=3` exceeds 100×. This
   confirms the philosophy §3 prediction: the optimal `D_pre` is *shallower* in Rust
   than the Python curves suggested (Python's interpreter overhead made shallow
   caches look marginal). The win grows with `D_pre` because a larger root-near band
   splices away more of the exponential cone.
2. **Precompute is ~free relative to the query savings.** The one-time sweep is
   0.05–0.8 ms; it saves ~17–23 ms across one 500-seed batch — so it amortizes
   *within a single batch*, before any cross-run reuse (the LMDB-persisted
   `boundary_basis`, #3210) or cross-query reuse is counted.
3. **Exact correctness holds at scale.** Every row is `eq = yes`: the boundary
   aggregate equals the production hop-stream aggregate exactly. The integrated path
   — `min_dist` → `boundary_band_root_near` → `build_boundary_suffix_sweep` →
   `collect_native_category_ancestor_boundary_hist` — running the *real* emitted
   kernels (not the P3 std-only model) surfaced **no** correctness bug.
4. **Bounded working set.** `peakR` tracks the band size; the sweep does not blow up
   even at `D_pre=4` (where the band approaches "cache everything"). The interesting
   operating regime is `D_pre ∈ {1,2,3}`, the root-near cut — `D_pre=4` here caches
   most of the graph and the speedup plateaus (~145×).

## Caveats

- Single-thread, single machine, synthetic dense-core graph; absolute numbers are
  illustrative. The **shape** (shallow crossover, >100× by `D_pre=3`, exact match)
  is the durable result.
- This measures the eager (in-memory) edge path. The LMDB-backed lazy path adds
  seek cost to *both* baseline and boundary; the boundary still removes the walk, so
  the relative win should persist (to be confirmed in the end-to-end LMDB run).
- `boundary_splice_complexity_bench.rs` (P3, std-only) remains the complexity-class
  demonstration (splice flat vs full-enum growing); this report is its on-the-real-
  kernels counterpart.

Guarded by `tests/test_wam_rust_boundary_integrated_scale.pl` (the exact-match
invariant at a moderate synthetic scale, deterministic / debug).

## Addendum — entry-frontier band vs whole region (storage)

The original table used `boundary_band_root_near(D_pre) = {1 ≤ min_dist ≤ D_pre}`,
the **whole** root-near region. That `band` column grows with the *cumulative*
node count (313, 599, 711 …) — it caches the region's *volume*. But the boundary
kernel splices at the **first** cached node a seed reaches and stops, so a periphery
seed only ever uses the region's **entry frontier** (region nodes with a child
*outside* the region). `boundary_band_entry_frontier(D_pre, edge_pred)` caches just
that cut — the region's *surface*:

```
config           Dpre   prod_ms  bound_ms    pre_ms   speedup  region   front   eq
core=120  cp=3      1     18.06     5.602     0.052      3.2x       8       8  yes
core=120  cp=3      2     18.06     0.728     0.108     24.8x      65      38  yes
core=120  cp=3      3     18.06     2.191     0.130      8.2x     313      38  yes
core=120  cp=3      4     18.06    11.292     0.094      1.6x     599       8  yes
core=240  cp=3      2     23.64     1.477     0.179     16.0x      71      51  yes
core=240  cp=3      3     23.64     1.179     0.233     20.0x     335      83  yes
core=240  cp=3      4     23.64    14.913     0.161      1.6x     711      10  yes
```

`region` = whole-region band size; `front` = entry-frontier band size.

**Findings.**

- The entry frontier is the **thin cut** the design intends: at the intended
  operating point (`D_pre=2`, where most seeds are still in the *periphery*, outside
  the region) it gets ~16–25× with a band of 38–51 vs the region's 65–71 — same
  order of speedup, a fraction of the storage. This is the fix for the
  region-band "blowing up" with cumulative node count.
- At large `D_pre` the speedup *falls* (8.2×, then 1.6×) — and that is the lesson,
  not a regression: once `D_pre` is large enough that the region has swallowed the
  periphery (`region` ≈ all 620 nodes), there is almost nothing *outside* it, so the
  frontier collapses (8–10 nodes) and the now-in-region seeds enumerate the dense
  interior cone the frontier does not cache. **A boundary cache wants seeds in the
  periphery**: pick `D_pre` so the frontier sits between the seeds and the dense
  core. The whole-region band hides this by caching the interior too — at a storage
  cost that grows with the graph.
- Correctness is unchanged (`eq = yes` throughout): any band is exact for the
  splice, so the region-vs-frontier choice is purely a storage/coverage tradeoff.

So `boundary_band_entry_frontier` is the storage-efficient default for the boundary-
cache regime; `boundary_band_root_near` remains available when maximal coverage
(caching the interior too) is wanted and storage is not the constraint.

## Addendum — pre-weighted basis `g_B` (P4) and its budget precondition

For a **fixed** functional and budget, each cached histogram `H_B` collapses to a
single fixed-length vector `g_B[a] = Σ_b H_B[b]·(a+b)^(-N)` (prefix length `a`);
a query reaching `B` at depth `a` adds `g_B[a]` to WeightSum directly — a
dot-product splice that skips the histogram convolution and the final `powf` loop
(`build_boundary_basis_weighted_power` / `collect_native_category_ancestor_weightsum`).

**Precondition (plan §4a "budget-specialised").** `g_B` bakes the path-length budget
and `N` into each scalar (it pre-sums suffixes with `a+b ≤ max_depth`), so it is
valid only for queries with that *same fixed budget and N*. It cannot serve a query
with a different/smaller max path length, nor a different functional — those need
the **histogram** (`boundary_suffix`), which preserves the length breakdown and can
be re-truncated per query. `g_B` is the fixed-budget/fixed-functional specialisation;
the histogram is the general, budget-flexible form. (Validated:
`weighted_power_basis_equals_histogram` — `g_B` WeightSum == histogram
`weighted_power` == full enumeration, same fixed budget.)

## Addendum — lazy (demand-driven) vs eager boundary cache

The eager `build_boundary_suffix_*` precompute materialises the whole band up front
(the *fixed-point / eager* strategy of `RECURRENCE_EVALUATION_STRATEGY`).
`WamState::lazy_boundary_weightsum` is the *per-query / lazy* alternative: start
empty and compute each band node's `node->root` histogram **on first demand**,
memoizing it — only the band-entry nodes the workload actually touches are computed.

Which wins is **not** a single number — it depends on **workload sparsity** (how
much of the band the queries touch) and **query count K** (how many rounds amortize
the warmup). Eager precomputes the *whole region band* — it does not know which
seeds will be queried — then each round splices; lazy warms only the touched subset
during round 1, then splices. Apples-to-apples on the same `min_dist <= D_pre = 2`
band, total cost = (precompute or warmup) + K × steady-round:

```
                              eager        lazy        winner by K
config  workload   band  pre_ms touched  warm_ms   K=1     K=10    K=100
core120 dense 500   65   0.80    54      1.93      eager   eager   LAZY
core120 sparse 20   65   0.80    33      0.28      LAZY    LAZY    LAZY
core240 dense 500   71   0.94    51      3.36      eager   eager   LAZY
core240 sparse 20   71   0.94    45      0.85      LAZY    LAZY    LAZY
```

**Findings (this corrects the earlier "lazy is just slower" note).**

- **Steady-state is identical.** Once both caches are warm, a query round is a plain
  splice for both (eager ≈ lazy per round, e.g. 0.60 vs 0.58 ms) — so K rounds
  amortize the *same* way. The strategy choice is entirely about the *warmup* cost.
- **Sparse workload -> lazy wins at every K.** When the queries touch little of the
  band (20 seeds, ~33 of 65 nodes), eager wastes precompute on the whole region
  (0.80 ms) while lazy warms only what's asked (0.28 ms) — lazy is 3× cheaper at K=1
  and stays ahead as K grows.
- **Dense workload + few queries -> eager wins.** When the queries touch most of the
  band, eager's batched precompute (0.80 ms for 65 nodes) beats lazy's interleaved
  on-demand warmup (1.93 ms) — so for K=1..~50 eager is cheaper.
- **Dense workload + many queries -> lazy edges ahead** (K=100): lazy's smaller warm
  cache (54 vs 65) gives marginally faster steady rounds, and the upfront-precompute
  gap washes out. So even in the dense case the crossover is finite.
- **Dataset size shifts it toward lazy.** The eager band is the whole root-near
  region, which grows with the graph; the touched subset is bounded by the workload.
  So a bigger graph with the same query workload widens lazy's precompute saving.

**Takeaway:** lazy is the right strategy for **sparse or unknown** query
distributions (don't precompute a band you won't use), for streaming/interactive
workloads (self-warming, no precompute moment), and on the **lazy/LMDB edge path**
(it enumerates via the `EdgeAccessor`; the eager shared-memo sweep needs the
in-memory `ffi_facts`). Eager is the right strategy for a **known, densely-reused
band with a modest query count**. Neither dominates — the harness prints the
per-config winner.

Guarded by `lazy_boundary_caches_on_demand_and_matches_eager` (a query touching one
seed caches only that seed's entry; results match production).
