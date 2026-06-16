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
