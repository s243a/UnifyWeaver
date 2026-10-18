# Store backend selection: theory, benchmarks, and the `auto` default

How the C++ WAM store-backed resolver chooses between the two D43 seek backends —
**indexed** (dependency-free UWFI/UWIX; whole-`.idx`-in-RAM binary search + mmap
in-place record read + shared L1/L2 row cache) and **lmdb** (system `liblmdb`;
keyed B-tree range-scan + the same shared cache).

`RESULTS.md` is the raw data appendix (full tables, per-cell jsonl). This file is
the decision doc: what we learned, the policy, and how to refine it.

## TL;DR

- Across four rounds of optimization, **the storage engine was never the story** —
  caching and the read path were. Once the indexed backend got (1) the index in
  RAM, (2) the shared L1/L2 row cache, and (3) an mmap in-place record read, it
  **matches lmdb** on every workload that fits RAM: within ~1.1-1.4x on reuse and
  **~1.0x** on resident pure-miss.
- lmdb used to pull ahead in exactly one regime: **disk-bound (store > RAM) AND
  multi-row-per-key**, where its key-clustered leaves save cold seeks (asymptotic
  win ≈ rows_per_key). **The clustered `.data` layout closes that gap.** Indexed
  now **stable-sorts `.data` by key at build time** so a key's rows are physically
  contiguous — a keyed lookup touches **~1 distinct `.data` page**, not
  rows_per_key. Measured (deterministic, from `.idx`): the 8-row/key synthetic
  store drops from **8.0 → 1.05** cold-reads/miss; the realistic scattered
  `revdep` table (4 rows/key, max 109) drops **3.79 → 1.02**; both matching lmdb's
  structural ~1. So indexed is now perf-competitive **everywhere**, disk-bound
  multi-row included.
- **lmdb is therefore no longer a performance necessity — it is an opt-in for its
  MATURITY**: battle-tested storage, crash-safety, and concurrent writers. The
  dependency-free indexed backend is the default and loses nothing on speed.
- **Default policy (`auto`): pick lmdb iff `store_size > 2 × available_RAM`
  AND `rows_per_key ≥ 2` AND lmdb is usable for the C++ lane, else indexed.**
  This round leaves the default unchanged (conservative), but with clustered
  `.data` the `rows_per_key ≥ 2` gate no longer reflects a real indexed
  disadvantage — see *Follow-up: the rows_per_key gate can be relaxed*.

## Theory: the cost model

A keyed lookup is a cache hit, or a miss that reads the key's records. Per lookup,
with hit rate `h`, rows-per-key `M`, and store/RAM ratio `r ≥ 1`:

```
T(backend) = h·t_hit + (1−h)·cold_reads·[ (1/r)·t_mem + (1−1/r)·t_seek ]
   P(a miss's page is evicted) ≈ 1 − 1/r      (fraction of store not resident)
   indexed (clustered .data): cold_reads ≈ 1  (a key's rows are contiguous → ~1 page)
   lmdb:                       cold_reads ≈ 1  (records clustered in ~1 B-tree leaf)
```

> **Clustered `.data` (this round).** The row `indexed: cold_reads = M` above was
> the *old* source-order layout. The builder now stable-sorts `.data` by key, so a
> key's `M` rows occupy `M` *contiguous* offsets → ~1 distinct 4KB page. The model
> below is retained to explain the historical crossover; with clustered `.data`
> **`M` collapses to 1 for indexed too**, so the `T_indexed/T_lmdb → M` asymptote
> and the whole `K` table below flatten to **1.0× at every r** — lmdb has no
> speed edge left. The measured `scatter` proof of this collapse is in the table.

Measured primitives on this box (WSL2; `cost_model.sh` + `cost_model_probe.c`):

| primitive | value | measured how |
|---|---|---|
| `t_seek` — cold single 4KB page | **~173-180 µs** (p10-p90 ≈ 150-225) | `posix_fadvise(DONTNEED)` + `pread`, ×3000, median |
| `t_mem` — warm single 4KB page | **~0.5 µs** | same pages resident |
| `t_seek / t_mem` | **~340-350×** | — |
| `t_hit` — cache-hit lookup | indexed ~0.3-0.6 µs, lmdb ~0.15 µs | 50k lookups / 100 hot keys |
| `t_miss_resident` — warm zero-reuse lookup | ~2.4-2.6 µs (indexed ≈ lmdb) | `unique` R=1 warm |
| **scatter** = indexed cold-reads/miss (source-order, old) | **= rows_per_key** (exact) | distinct 4KB `.data` pages a key spans, from `.idx` |
| **scatter** = indexed cold-reads/miss (**clustered, now**) | **≈ 1** | same measure; clustered `.data` → contiguous offsets |
| lmdb cold-reads/miss | **≈ 1** | structural (key-ordered B-tree) |

**Clustered `.data` scatter — measured before/after (`cost_model.mjs scatter`, deterministic, `.idx`-only):**

| store | rows_per_key | cold-reads/miss BEFORE (source-order) | cold-reads/miss AFTER (clustered) | lmdb (structural) |
|---|---|---|---|---|
| synthetic 8-row interleaved | 8.0 | **8.0** | **1.055** | 1 |
| resolver `revdep` (scattered, max 109) | 4.02 | **3.79** | **1.024** | 1 |
| resolver `dep` (already source-clustered) | 3.00 | 1.03 | 1.015 | 1 |
| resolver `pkg` (~1-row) | 1.50 | 1.003 | 1.002 | 1 |
| ABI `symprov` (~1-row) | ~1.03 | ~1.0 | ~1.0 (unchanged) | 1 |

The scatter collapse is **builder-only** and answer-identical: the record bytes,
count, and `.idx` format are unchanged (only the physical order of `.data` records
and their offsets change), so the runtime read path is byte-transparent to it. The
`revdep` line is the headline — a *realistically* scattered multi-row table where
lmdb previously won ~3.8× on cold reads now ties it at ~1.

**What the model says:**

- **Onset ≈ 1× RAM.** When `r ≤ 1` the store fits, `P(evicted) ≈ 0`, every miss is
  a resident page (`t_mem`), and indexed ≈ lmdb (measured 1.0x). No benefit below
  RAM. The crossover **onset** is where the store stops fitting: ~1× available RAM.
- **Ramps to asymptote by ~2× RAM.** Above RAM, `P(evicted) = 1 − 1/r` climbs:
  0 at r=1, 0.5 at r=2, →1 as r→∞. By r≈2 half the misses hit disk; the speedup is
  already most of the way to its asymptote.
- **Asymptotic magnitude = rows_per_key.** Fully disk-bound, `T_indexed/T_lmdb → M`
  (M scattered `t_seek`s vs 1). For `M = 1` the ratio is **1.0 at every r** —
  lmdb is never worth it.
- **Skew pushes the onset past 1× RAM.** A hot set (real ABI resolution
  re-touches a few libraries) keeps the working set resident even when the whole
  store exceeds RAM, so misses stay cheap longer. That makes **1× RAM a floor**
  for the onset and **2× a conservative trigger**.

Solving `T_lmdb < T_indexed` for the store/RAM ratio `K` (hit_rate 0.9, measured
primitives) — `K` is a **threshold near 1**, the **magnitude is rows_per_key**:

| rows_per_key | speedup at store ≫ RAM | K@1.2× | K@1.5× | K@2× |
|---|---|---|---|---|
| **1 (ABI symprov)** | **1.0×** | never | never | never |
| 2 | 2.0× | 1.0 | 1.05 | never |
| 4 | 4.0× | 1.0 | 1.0 | 1.05 |
| 8 | 7.9× | 1.0 | 1.0 | 1.0 |
| 16 | 15.8× | 1.0 | 1.0 | 1.0 |

## Benchmarks (summary; full tables in RESULTS.md)

Three-way min-of-3 wall, symbol scale (256k rows), warm:

| workload | R | idx-nocache | idx+cache | idx+cache+mmap | lmdb |
|---|---|---|---|---|---|
| skewed (reuse) | 10 | 1706 ms | 250 ms | ~250 ms | 213 ms |
| unique (zero-reuse) | 1 | 950 ms | 1284 ms | **593 ms** | 591 ms |

- The **cache** collapsed the reuse gap from 7-21× to ~1.1-1.4×.
- The **mmap** in-place record read collapsed the resident pure-miss gap from
  ~2.2× to **~1.0×** (unique R=1: indexed 593 ms vs lmdb 591 ms warm; 0.92× cold).
- Deterministic parity: after mmap the indexed miss read-count equals lmdb's
  (~1 read/record), and indexed's cache reports identical L1/L2/miss to lmdb.
- lmdb's disk-bound multi-row advantage was **modeled, not stress-tested** (see
  Honesty): the scatter (= rows_per_key) and `t_seek` were measured; the aggregate
  disk-bound regime was extrapolated. **Clustered `.data` now removes that
  advantage at its root** — indexed's scatter drops to ~1 (measured deterministically
  from `.idx`), so there is no rows_per_key gap left to stress-test.

## Default policy: `auto`

Implemented in `examples/pkg_resolver/store/ensure_lmdb.sh`
(`uw_resolve_store_backend`), wired into `cpp_store/build.sh` as the default when
`UW_STORE_BACKEND` is unset. Test: `store/test_auto_select.sh`.

```
choose LMDB  iff  store_size_bytes > UW_STORE_LMDB_RAM_FACTOR × available_RAM_bytes
             AND  rows_per_key >= UW_STORE_LMDB_MIN_ROWS_PER_KEY
             AND  lmdb is usable for the C++ lane
else INDEXED
```

- `UW_STORE_LMDB_RAM_FACTOR` — the headroom factor, **default 2** (named constant,
  tunable; a non-integer value is rejected with a warning and treated as 2). 2× is
  deliberately conservative: the model puts the *onset* at ~1× RAM and skew pushes
  it higher, so 2× only trips lmdb once the store clearly exceeds RAM and
  disk-bound misses are unavoidable.
- `UW_STORE_LMDB_MIN_ROWS_PER_KEY` — **default 2**. `rows_per_key` is aggregated
  cheaply from the UWIX `.idx` headers (`n_records / n_keys`, no scan) across all
  indexes in the store dir. Because the asymptotic lmdb win is ≈ rows_per_key, a
  ~1-row/key store (ABI symprov = 1.03) never benefits — this gate keeps it on
  indexed even above 2× RAM. When no `.idx` exists yet (pre-build), rows_per_key
  is *unknown* and the gate is skipped (size-only for that first build).
- `available_RAM` — `/proc/meminfo` `MemAvailable`, overridable with
  `UW_STORE_AVAIL_RAM_BYTES` (WSL2's `MemAvailable` balloons, so the override
  matters for tests/reproducibility).
- `store_size` — the **built** indexed store (`*.data` + `*.idx`) when present,
  else the source **P/2 JSONL** (`*.jsonl`, excluding `cases.jsonl`) as a
  pre-build estimate. One consistent measure; documented here.
- **`lmdb usable` for the C++ lane** is a real probe, not just "the npm module
  loads": (1) `uw_ensure_lmdb` (the v1-format module used to *build* the store),
  (2) system `liblmdb` links (`#include <lmdb.h>` + `-llmdb`) — the C++ reader
  needs it, and (3) best-effort: an already-built lmdb store under `DIR/lmdb/*`
  actually `mdb_env_open`s (catches `MDB_INVALID` from a wrong page format).
- **Fallback:** if the rule wants lmdb but it is not usable, it **WARNs loudly and
  uses indexed** — safe because indexed is answer-identical and ~as fast up to the
  disk-bound multi-row regime. It prints the chosen backend and the numbers.
- **Policy only, never answers.** Whichever backend is chosen returns identical
  rows. Proven: 503-case store differential + 51-case corpus stay **0
  divergences** (corpus verified through the auto path → indexed);
  `bench_crossover.sh` asserts `rows_found` identical across backends in every
  cell; and `test_auto_select.sh` checks the size rule and the rows_per_key gate
  (lmdb above 2× with rpk≥2, indexed below 2× or at rpk<2) via the RAM override.

## Storage note: the index is mmap'd, not slurped

The indexed backend **mmaps** both `.idx` and `.data` (`PROT_READ`,
`MAP_PRIVATE`), page-cache-backed and **evictable**. An earlier version slurped
the whole `.idx` into a heap `std::string` — but the `.idx` is **39-97% of the
store**, so that allocated ~that much *non-evictable anonymous* memory. Under the
exact pressure that routes a `> 2× RAM` store to indexed, the slurp risked
`bad_alloc` → caught as a goal failure → *silent under-answering*. mmap restores
the O(1)-heap, page-cache-friendly behavior (with an `ifstream` fallback for
non-POSIX). It also fixes the D43 counters (the mmap path charges nothing at open;
only actual record reads count).

## Done: key-sorted (clustered) indexed `.data`

The refinement previously deferred here is now **implemented** (builder-only, in
`scripts/js_wam/uw_fact_codec.js` `writeIndexedStore`). Indexed's only structural
disadvantage was source-order scatter (M scattered pages per multi-row key). The
builder now **stable-sorts `.data` by first-arg key** so a key's rows are physically
contiguous, clustering them into ~1 page (measured: the 8-row/key store drops from
8 to **1.055** pages/key; the scattered `revdep` table from **3.79 → 1.02**),
erasing lmdb's disk-bound edge **without the external dependency**. Stable = source
order preserved among equal keys, so intra-key answer order is unchanged; the `.idx`
format, record bytes, and record count are untouched, so the runtime read path is
byte-transparent (no goldens rebaseline). Answer-identity re-proven: store
differential **503/0**, corpus **51/0**, ABI verify **122/0**, and the bench
`row_digest` is byte-identical between the scattered and clustered layouts.

## Follow-up: the `rows_per_key` gate can be relaxed (discuss)

The `auto` policy's `rows_per_key ≥ 2` gate exists because, with source-order
`.data`, a multi-row key cost indexed ≈ rows_per_key cold reads — so lmdb's win
scaled with rows_per_key and the gate steered multi-row stores toward lmdb above
2× RAM. **Clustered `.data` removes that premise**: indexed's cold-reads/miss is
now ~1 regardless of rows_per_key, so lmdb no longer has a multi-row speed edge to
gate for. The gate could be **relaxed or dropped** — arguably the *entire* lmdb
auto-branch could go, since indexed now matches lmdb on speed in every measured
regime and lmdb's remaining value is maturity/crash-safety/concurrency (a
deployment choice, not something `auto` should infer from store shape).

**Recommendation:** do NOT change the default this round. Reasons: (1) the
disk-bound multi-row regime is *modeled*, not stress-tested on this box (no fair
memory cap — see Honesty), so keep the conservative gate until the clustered win is
confirmed under real pressure; (2) leaving `auto` as-is keeps this change
builder-only and low-risk. Flag for a follow-up decision: once the clustered
cold-read collapse is validated under real memory pressure, relax
`UW_STORE_LMDB_MIN_ROWS_PER_KEY` (or retire the size/rows auto-branch entirely and
make lmdb a purely explicit opt-in).

## Honesty

- Resident costs (`t_hit`, `t_miss_resident`, the ~1.0× pure-miss parity) and the
  per-page cold latency `t_seek` are **measured**. The **scatter** (indexed
  cold-reads/miss) is measured exactly from `.idx` offsets — **before** clustering
  it = rows_per_key, **after** it ≈ 1. That collapse is the deterministic,
  trustworthy signal for the clustered-`.data` win, and it is fully reproducible
  (`cost_model.mjs scatter`).
- **The clustered win is proven deterministically, not on wall time here.** On this
  box, an indexed-only cold-vs-clustered wall comparison (same binary, scattered vs
  clustered `.data`, `UW_BENCH_EVICT=1`, min-of-5) was **flat** (~144 ms both) —
  because the store is far smaller than RAM and `posix_fadvise(DONTNEED)` does not
  create real memory pressure, so "cold" ≈ warm and the OS page cache absorbs the
  reads. The `fact_io_reads`/`fact_io_bytes` D43 counters are also identical
  between layouts (logical reads are unchanged; only physical *page* locality
  improves). The wall payoff of fewer distinct cold pages only materializes under
  genuine memory pressure / real disk seeks, which this WSL2 host cannot produce
  unprivileged.
- The **aggregate disk-bound regime is modeled/extrapolated** from those
  primitives, **not stress-tested**: this WSL2 box has no fair memory-cap
  mechanism (no cgroup `memory.max` unprivileged; capping the app cache is unfair
  because lmdb still gets RAM via mmap/page cache; `MemAvailable` balloons). So the
  crossover magnitude and the `K` table are estimates grounded in measured
  primitives, and the `2×` factor is a conservative engineering choice, not a
  stress-tested optimum.
