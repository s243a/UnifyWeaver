# WAM-Rust Cached Graph-Search: Cache × Threads Interaction (2026-06-15)

Companion to `wam_rust_cached_scaling_sweep_2026-06-14.md`, which measured the
cached-vs-lazy win **single-threaded** to isolate the cache effect. The obvious
follow-up: do the cache and multi-threading *stack*? This sweeps thread count on
the largest fixture (10k, 25 227 edges) at fixed size.

## Method

Same cached / lazy crates as the scaling sweep. Fixed fixture `10k`, varying
`WAM_THREADS`. `query_ms` is the min over 3 reps; attribution on.
Reproduce: `THREADS=N examples/benchmark/run_wam_rust_cached_scaling_sweep.sh 10k`
(machine here: 4 physical cores).

## Result

| threads | cached `query_ms` | lazy `query_ms` | **speedup** | L1 hits | L2 hits | misses |
|---------|-------------------|-----------------|-------------|---------|---------|--------|
| 1 |  82 | 283 | **3.45×** | 700 847 |     0 | 4 493 |
| 2 |  53 | 142 | **2.68×** | 700 135 |   712 | 4 493 |
| 4 |  38 |  73 | **1.92×** | 699 281 | 1 565 | 4 494 |
| 8 |  36 |  74 | **2.06×** | 697 619 | 3 227 | 4 494 |

(8 threads on 4 cores is oversubscribed — no gain over 4, as expected.)

## Reading the numbers

- **Cache and parallelism are partially substitutable.** Both attack the same
  cost — LMDB parent-lookup seeks. So as threads independently hide seek latency,
  the cache's *relative* advantage shrinks (3.45× → ~2×): lazy has more
  parallelisable seek work to distribute, so it gains more from extra cores.
- **The cache still wins ~2× at full core count**, and does far less work: ~4 500
  inner seeks total vs lazy issuing a seek per lookup (~705 000). On this small
  fixture the LMDB pages are warm in the OS cache so lazy's "seeks" are cheap RAM
  reads; on a disk-bound graph (enwiki scale) each lazy miss is a real page
  fault, so the substitutability weakens and the cache advantage should *grow*
  with size even as cores rise.
- **The two-tier cache behaves as designed under parallelism.** Single-threaded,
  the per-thread L1 (65 536 slots) absorbs everything (L2 = 0). As threads rise,
  each worker's L1 starts cold, so the shared L2 picks up the cross-thread reuse
  (0 → 3 227 hits) — exactly its role. Miss count stays flat (~4 493), i.e. the
  unique working set is covered regardless of thread count.
- **Both optimisations are worth keeping**: parallelism cuts latency; the cache
  cuts work/IO (energy, and disk traffic on cold graphs). They compose to the
  fastest configuration (cached + all cores: 36 ms vs lazy single-thread 283 ms,
  ~7.9×).

## Scope

Single fixture, 4 cores, OS-warm LMDB. The headline caveat — that the cache
advantage is *understated* here because warm-cache lazy seeks are cheap — points
at the same next step as the size sweep: an enwiki-scale, disk-bound fixture,
where cache and parallelism should substitute less and the cache win widen.
