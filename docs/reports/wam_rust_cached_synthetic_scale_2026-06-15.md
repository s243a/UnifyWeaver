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

**L2 capacity is largely insensitive.** A 1024-entry L2 (4 KB) is within ~0.1
point of a 4 M-entry L2; by ~65 536 the hit rate is saturated. Oversizing L2
buys nothing and can slightly hurt (allocation/locality — the 4 M case is
no faster, sometimes slower). The reason is the same skew: the L2-cacheable
benefit is only the small tail of keys evicted from L1 and re-accessed soon;
once that tail fits, more L2 is dead weight.

## Actionable conclusion

For this graph-search workload the per-thread L1 plus skewed hub access do the
work; **a modest L2 (on the order of the L1, not the working set) is sufficient**.
This supports the R8b cost-model approach of clamping cache capacity rather than
sizing to the full demand set — the demand set can be 4× L1 with no hit-rate
penalty. Sizing L2 to the working set wastes memory for ~0.1 point of hit rate.

## Caveats

- **Synthetic, not real wiki.** The structure is chosen to be cache-realistic
  (hub ancestors + depth), but it is not a real category graph. Conclusions about
  the cache architecture (L1-dominance, L2-insensitivity, the persistent ~2.5×
  win) are structural and should carry over; exact numbers would differ on a real
  dump.
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
