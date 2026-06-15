# WAM-Rust Cached Graph-Search at Scale (Synthetic, 2026-06-15)

The size/thread sweeps (`wam_rust_cached_scaling_sweep_2026-06-14.md`,
`wam_rust_cached_multithread_2026-06-15.md`) all carried the same caveat: the
in-repo fixtures are tiny (≤25k edges, working set ~1.7k ≪ the 65 536-slot
per-thread L1), so the cache never overflows L1 and L2 capacity is never
stressed. The clean next step was an enwiki-scale fixture.

**A real dump is not reachable here** — the network policy blocks
`dumps.wikimedia.org` (`403 host_not_allowed`) and no large graph is checked in.
So this uses a **synthetic** category DAG
(`examples/benchmark/generate_synthetic_category_graph.py`) sized to reach the
regime that matters: working set several times the L1, exercising L2 under
capacity pressure. Structure mimics the cache-relevant properties of a wiki
category graph — a single root, a small set of high-in-degree hub ancestors near
the root (path convergence → reuse), and otherwise locally-wired parents (depth).

## Scales reached

| graph | edges | seeds | total lookups | **working set** (distinct) | vs L1 (65 536) |
|-------|-------|-------|---------------|----------------------------|----------------|
| 200k cats | 398 832 | 86 476 | 6.93 M | 102 188 | **1.6×** |
| 500k cats | 999 126 | 174 670 | 63.7 M | 256 462 | **3.9×** |

(Correctness invariant holds at both: cached output == lazy output, exactly.)

## Cached vs lazy (single-threaded query time)

| graph | cached ms | lazy ms | **speedup** | hit rate | L1 hits | L2 hits | inner misses |
|-------|-----------|---------|-------------|----------|---------|---------|--------------|
| 200k | 1841 | 5176 | **2.81×** | 0.9853 | 6.80 M | 30 289 | 102 188 |
| 500k | 25 453 | 63 819 | **2.51×** | 0.9960 | 63.1 M | 346 345 | 256 462 |

The cache win **persists at scale** (2.5–2.8×) even with the working set 1.6–3.9×
the L1. Hit rate doesn't fall as the working set overflows L1 — it *rises*
(98.5% → 99.6%) because more total lookups concentrate ever harder on the hot
hub ancestors, which stay resident in the per-thread L1. The cache is
**L1-dominated**: L1 serves 98–99% of all lookups; L2 mops up a thin tail.

## L2 capacity sweep — how much cache do you actually need?

Single-threaded; the working set (102k / 256k) far exceeds every L2 size tried.

200k (working set 102 188):

| L2 capacity | hit rate | inner misses | query ms |
|-------------|----------|--------------|----------|
| 1 024 | 0.9841 | 109 863 | 1707 |
| 8 192 | 0.9850 | 103 597 | 1644 |
| 65 536 | 0.9852 | 102 687 | 1756 |
| 4 000 000 | 0.9853 | 102 188 | 2182 |

500k (working set 256 462):

| L2 capacity | hit rate | inner misses | query ms |
|-------------|----------|--------------|----------|
| 1 024 | 0.9954 | 295 898 | 22 120 |
| 65 536 | 0.9960 | 257 079 | 22 241 |
| 262 144 | 0.9960 | 256 462 | 23 310 |
| 4 000 000 | 0.9960 | 256 462 | 21 831 |

**Single-threaded, L2 capacity is largely insensitive.** A 1024-entry L2 (4 KB)
is within ~0.1 point of a 4 M-entry L2; by ~65 536 the hit rate is saturated.
The reason is skew: with one thread, the single per-thread L1 (65 536 slots)
holds the hot hub ancestors, so the L2-cacheable benefit is only the thin tail of
keys evicted from L1 and re-accessed soon. Once that tail fits, more L2 is dead
weight (the 4 M case is no faster, sometimes slower).

## Multi-threaded: the shared L2 carries cross-thread reuse

The single-threaded picture above is misleading for real multi-core use. The L1
is **per-thread**: with N worker threads each starts with a cold L1, so reuse of
a hub ancestor *across* threads cannot hit any thread's L1 — it must hit the
**shared L2**. So L2 capacity, irrelevant single-threaded, becomes load-bearing.
Same 200k graph (working set 102 188), L2 capacity sweep at 1 vs 4 threads:

| L2 capacity | 1 thread hit | 1 thread inner misses | 4 threads hit | 4 threads inner misses |
|-------------|--------------|-----------------------|---------------|------------------------|
| 1 024 | 0.9841 | 109 863 | **0.9745** | **176 944** |
| 16 384 | 0.9851 | 103 179 | 0.9806 | 134 083 |
| 65 536 | 0.9852 | 102 687 | 0.9838 | 112 303 |
| 262 144 | 0.9853 | 102 188 | 0.9853 | 102 189 |
| 4 000 000 | 0.9853 | 102 188 | 0.9853 | 102 188 |

At 4 threads, shrinking L2 from the working-set size (262 144) to 1024 raises
inner LMDB seeks from 102 k to **177 k (+73%)** and drops the hit rate a full
point (98.5% → 97.5%); the shared-L2 hit count grows 16 k → 93 k across the
sweep (vs only 23 k → 30 k single-threaded). The L2 recovers the single-threaded
hit rate once it reaches roughly the cross-thread working set (~262 144 ≈ 2.5×
the L1 here).

(On this RAM-warm fixture those extra 75 k seeks barely move wall time — 589 vs
678 ms — because parallelism hides them and a "seek" is a µs page-cache read. On
a disk-bound graph each extra seek is a page fault, so under-sizing L2 would cost
real time, not just I/O work.)

## Actionable conclusion

The cache is **L1-dominated per thread, L2-dominated across threads**:

- **Single-threaded / per-thread streams:** the L1 does the work; L2 can be tiny.
- **Multi-core (the real deployment):** the shared L2 carries the cross-thread
  hub reuse, so it should be sized to roughly the cross-thread working set, not
  minimised — under-sizing it costs ~1 point of hit rate and ~73% more LMDB seeks
  here (and proportionally more wall time when disk-bound). This matches the
  earlier observation that the L2 provides most of the cache benefit: that holds
  whenever reuse is cross-thread or cross-query (the L1 is per-thread and does not
  persist across either).

So the R8b cost-model clamp is right to bound capacity, but the floor it clamps
*to* should track the cross-thread working set on multi-core runs — not shrink to
L1 size, which is only safe single-threaded.

## Caveats

- **Synthetic, not real wiki.** The structure is chosen to be cache-realistic
  (hub ancestors + depth), but it is not a real category graph. The architectural
  conclusions (per-thread L1 dominance, shared-L2 cross-thread reuse, the
  persistent ~2.5× win) are structural and should carry over; exact numbers would
  differ on a real dump.
- **Not disk-bound.** The LMDB is RAM-resident here, so an inner "seek" is a ~µs
  page-cache read. On a graph too large for RAM each lazy miss would be a real
  page fault, so the cache advantage measured here is a *lower bound* — it would
  widen, and L2 would matter more, when seeks hit disk.

## Reproduce

```bash
python3 examples/benchmark/generate_synthetic_category_graph.py \
    --out /tmp/syn_src --categories 200000 --articles 200000 \
    --hubs 512 --local-window 2000 --hub-prob 0.35 --seed 1
mkdir -p /tmp/syn_fix
python3 examples/benchmark/ingest_resident_lmdb_fixture.py /tmp/syn_src /tmp/syn_fix/lmdb_resident
cp /tmp/syn_src/article_category.tsv /tmp/syn_fix/
# then run the cached/lazy crates against /tmp/syn_fix with
# UW_WAM_CACHE_ATTRIBUTION=1 and varying WAM_CACHE_CAPACITY / WAM_THREADS.
```
