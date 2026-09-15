# ABI store backend crossover — neutralizing the cache: indexed+L1/L2 vs lmdb

> Data appendix. The decision doc (theory + policy + the `auto` default) is
> [`BACKEND_SELECTION.md`](BACKEND_SELECTION.md).

Story so far:
1. **First run:** lmdb beat `indexed` by 13-72x — but mostly because the on-disk
   binary search re-read every probe (~37 `ifstream` syscalls/lookup).
2. **Fair fight:** optimized the indexed read path (whole `.idx` slurped into RAM
   once, in-memory binary search + one `.data` record read). Gap fell to ~1.6-2.2x
   on zero-reuse but stayed 7-21x on reuse — because lmdb still had an L1/L2 **row
   cache** and indexed had none.
3. **Cache lift:** lifted the L1/L2 row cache into the shared `SeekFactSource` so
   BOTH backends cache identically. Caching neutralized: indexed+cache matched
   lmdb within ~1.1-1.4x on reuse, leaving only a ~1.9-2.7x edge on pure-miss.
4. **This run:** **(a)** close the pure-miss edge — `mmap` the `.data` so a miss
   reads each record in-place (one page access like lmdb) instead of two
   positioned reads; **(b)** build a calibrated **cost model** to estimate when
   lmdb is worth it (store/RAM ratio × rows-per-key), since real memory pressure
   is not creatable on this box.

**TL;DR of this run:** the pure-miss ~2x **closed to ~1.0x** (mmap). The only
regime where lmdb still wins is **disk-bound + multi-row-per-key**, and the win
is ≈ rows_per_key; for the ABI ~1-row/key store lmdb is **never** worth it at any
store/RAM ratio.

Harness + entry script: `examples/pkg_resolver/abi/bench/` (drives the C++ WAM
`SeekFactSource` read path directly — not the JS lmdb backend).
Reproduce: `BUILD_OLD=1 bash examples/pkg_resolver/abi/bench/bench_crossover.sh`.

## The change (shared cpp_wam runtime)

`templates/targets/cpp_wam/runtime.h.mustache`: the L1 (direct-mapped) + L2
(FIFO) row cache — key → decoded row list — was **lifted out of the
`WAM_CPP_ENABLE_LMDB` gate** into an engine-agnostic cache in `rows()`, used by
both backends. `rows()` now does: open → `ensure_cache_config()` → L1 probe → L2
probe (promote on hit) → on miss, backend `fetch_keyed()` (indexed:
`lookup_offsets`+`read_record`; lmdb: `lmdb_range_scan`) → fill both tiers. Full
(unbound-arg1) scans are never cached. Shared sizing env `UW_WAM_FACT_L1_SLOTS` /
`UW_WAM_FACT_L2_CAP` (the `UW_WAM_LMDB_*` names still honored for back-compat).
Cache is orthogonal to storage, so this is a pure code move + one branch.

## Store sizes

| scale | indexed | lmdb (v1) | rows | distinct keys |
|---|---|---|---|---|
| **symbol** (ABI `symprov/2`, `/var/lib/dpkg/info`) | 42 MB (24 data + 18 idx) | 128 MB | 256,225 | 249,097 |
| **package** (`gen_scale_catalog` 5k, `pkg/2`) | 320 KB (164 + 156 KB) | 972 KB | 7,522 | 5,007 |

Benchmark cache sizing: `UW_WAM_FACT_L2_CAP=65536`, L1 default (1<<14 slots) —
identical for indexed+cache and lmdb (fair).

## Correctness (all guardrails pass — a cache must not change answers)

1. **Cross-check:** indexed+cache `rows_found` **==** lmdb `rows_found` **==**
   nocache, every cell, both scales. And indexed+cache reports **identical
   L1/L2/miss counts to lmdb** (e.g. symbol skewed R=10 = 283,810 L1 / 195,733 L2
   / 20,457 miss for both) — proof the shared cache behaves identically per
   backend.
2. **Resolver differential/corpus/ABI** (built at `-O0` under memory pressure,
   one at a time): `run_differential_cpp_store.sh` = **503 / 0 divergences**;
   `run_corpus_cpp_store.sh` = **51 / 0**; `run_abi_verify.sh` = **122 / 0**.
3. **Byte-frozen goldens** re-baselined (plain 91891→92370, lmdb 92132→92611;
   +479 each, gate-independent). Suite green. Runtime-source golden unchanged.

Frozen `resolver.pl` / `resolver_store.pl` / `debian/` untouched (`git diff` clean).

## Results (min-of-3 wall; WSL2 noisy — spreads in raw jsonl)

| scale | workload | cache | R | idx-nocache | **idx+cache** | lmdb | cache vs nocache | lmdb vs cache |
|---|---|---|---|---|---|---|---|---|
| package | skewed | warm | 1 | 122 | **19** | 10 | 6.6x | 1.9x |
| package | skewed | warm | 5 | 592 | **40** | 30 | 14.7x | 1.3x |
| package | skewed | warm | 10 | 1167 | **65** | 57 | 18.1x | 1.1x |
| package | uniform | warm | 10 | 1131 | **74** | 62 | 15.3x | 1.2x |
| package | unique | warm | 1 | 12 | **15** | 5 | 0.8x | 2.7x |
| package | unique | warm | 10 | 126 | **21** | 11 | 6.1x | 1.9x |
| symbol | skewed | warm | 1 | 181 | **121** | 62 | 1.5x | 2.0x |
| symbol | skewed | warm | 5 | 811 | **179** | 126 | 4.5x | 1.4x |
| symbol | skewed | warm | 10 | 1706 | **250** | 213 | 6.8x | 1.2x |
| symbol | skewed | cold | 10 | 1748 | **325** | 303 | 5.4x | 1.1x |
| symbol | uniform | warm | 1 | 165 | **198** | 96 | 0.8x | 2.1x |
| symbol | uniform | warm | 10 | 1591 | **353** | 254 | 4.5x | 1.4x |
| symbol | unique | warm | 1 | 950 | **1284** | 586 | 0.7x | 2.2x |
| symbol | unique | cold | 1 | 1196 | **1533** | 685 | 0.8x | 2.2x |

(Full warm+cold sweep, all R, both scales: `.out/bench/results.symbol.jsonl` +
`results.package.jsonl`. Cold ≈ warm — stores ≪ RAM, no hard cap available
unprivileged on WSL2, so no disk-bound regime; deterministic counters lead.)

### Deterministic I/O (warm; identical across repeats and warm/cold)

| scale | workload | R | nocache reads | cache reads | lmdb reads | L1 (cache=lmdb) | L2 (cache=lmdb) | miss | rows |
|---|---|---|---|---|---|---|---|---|---|
| package | skewed | 1 | 142,486 | 12,478 | 6,238 | 45,052 | 981 | 3,967 | 71,242 |
| package | skewed | 10 | 1,424,842 | 12,478 | 6,238 | 480,499 | 15,534 | 3,967 | 712,420 |
| package | unique | 10 | 150,442 | 15,046 | 7,522 | 33,597 | 11,466 | 5,007 | 75,220 |
| symbol | skewed | 1 | 132,190 | 52,910 | 26,454 | 21,280 | 8,263 | 20,457 | 66,094 |
| symbol | skewed | 10 | 1,321,882 | 52,910 | 26,454 | 283,810 | 195,733 | 20,457 | 660,940 |
| symbol | uniform | 10 | 898,482 | 82,208 | 41,103 | 111,415 | 348,829 | 39,756 | 449,240 |
| symbol | unique | 1 | 540,964 | 528,032 | 264,015 | 867 | 2,366 | 252,992 | 270,481 |

Note: with the cache, indexed's read count is **flat in R** (52,910 at every R,
like lmdb's 26,454) — the linear-in-R re-reads are gone. The residual ~2x reads
vs lmdb is that indexed's `read_record` does two positioned reads per record (len
prefix + payload) where lmdb's mmap cursor returns the value in one op.

## Answer: does indexed+cache now match lmdb?

**On reuse — yes, essentially.** The cache neutralized the 7-21x reuse gap:
indexed+cache is now within **1.1-1.4x** of lmdb at moderate/high reuse
(skewed/uniform R≥5, both scales; e.g. symbol skewed R=10 250 ms vs 213 ms =
1.2x; package skewed R=10 65 ms vs 57 ms = 1.1x). The huge wins were **entirely
the cache** — confirmed, because indexed+cache and lmdb now post identical
L1/L2/miss counts and their walls converge. The dependency-free backend gets the
same reuse win.

**On zero-reuse (pure miss) — lmdb keeps a ~1.9-2.7x edge.** This is the true
engine difference: mmap single-value fetch vs indexed's two positioned reads per
record (`fact_io` shows cache-indexed does ~2x the reads of lmdb on misses). And
because the cache can't help a pure-miss stream, it adds small overhead there —
`unique` R=1 is the one place indexed+cache is *slower than* indexed-nocache
(symbol 1284 vs 950 ms; package 15 vs 12 ms). So the cache is a clear win wherever
there is any reuse and a slight tax on pure-miss.

**Same conclusion at both scales.** Package (5k, the default's real domain) and
symbol (256k) agree: cache≈lmdb on reuse, lmdb ~2x on pure-miss. Package looks
more lopsided in the nocache column only because its small keyset means higher
reuse.

## Follow-up: mmap the .data — the resident pure-miss ~2x closes to ~1.0x

The pure-miss gap was `read_record` doing **two** positioned reads per record
(4-byte length prefix, then payload). Change: `mmap` the `.data` file and read
each record **in place** — one page access, no read syscall, exact byte count,
just like lmdb's mmap value fetch (POSIX; ifstream two-read path kept as a
fallback). Deterministic proof: on symbol `unique` R=2 the indexed read count
dropped to **512,452 ≈ lmdb 512,450** (was ~2x), and `rows_found` stays identical
(cross-check passes: indexed == lmdb == nocache, every cell).

Zero-reuse wall, symbol `unique` R=1 (min-of-5), before vs after the mmap:

| variant | warm indexed | warm lmdb | ratio | cold indexed | cold lmdb | ratio |
|---|---|---|---|---|---|---|
| before (two reads) | 1284 ms | 586 ms | 2.2x | 1533 ms | 685 ms | 2.2x |
| **after (mmap)** | **593 ms** | 591 ms | **1.00x** | **624 ms** | 676 ms | **0.92x** |

So on **resident** access the engines are now equal — indexed matches lmdb on
pure-miss (and is marginally faster cold, being a flat file vs a B-tree). The mmap
also verified the resident multi-row case: an 8-rows/key store, `unique` warm,
indexed 79 ms vs lmdb 83 ms — scatter is free when resident.

## Backend-selection cost model (calibrated estimate)

We cannot create fair memory pressure on this WSL2 box (no cgroup `memory.max`;
capping the app cache is unfair because lmdb still gets RAM via mmap/page cache).
So we **measure primitives without pressure and extrapolate** the disk-bound
regime. Resident numbers and the per-page cold latency are **measured**; the
aggregate disk-bound regime is **modeled**, not stress-tested — labeled as such.

**Measured primitives (this WSL2 host; `cost_model.sh` / `cost_model_probe.c`):**

| primitive | value | how |
|---|---|---|
| `t_seek` — cold single 4KB page read | **~173-180 µs** (median; p10-p90 ≈ 150-225) | `posix_fadvise(DONTNEED)` a page, `pread`, ×3000 |
| `t_mem` — warm single 4KB page read | **~0.5 µs** | same pages, resident |
| `t_seek / t_mem` | **~340-350x** | — |
| `t_hit` — cache-hit lookup | indexed ~0.3-0.6 µs, lmdb ~0.15 µs | 50k lookups over 100 hot keys |
| `t_miss_resident` — warm zero-reuse lookup | ~2.4-2.6 µs (indexed ≈ lmdb after mmap) | `unique` R=1 warm |
| **scatter** — indexed cold reads per miss | **= rows_per_key** (measured exact) | distinct 4KB `.data` pages a key's records span, from `.idx` |
| lmdb cold reads per miss | **≈ 1** (clustered leaf) | structural (key-ordered B-tree) |

The scatter is the crux and is **measured exactly** from the `.idx` offsets:
indexed stores records in **source order**, so with realistic key-interleaved
input a key's N rows land on **N distinct pages** (measured: ABI 1.03 rows/key →
1.03 pages/key; synthetic interleaved 2/4/8/16 rows/key → 2/4/8/16 pages/key).
lmdb keys its B-tree by key, so a key's rows cluster in ~1 leaf. **But** building
the indexed `.data` in **key order** clusters it too — measured: an 8-row/key
grouped store spans **1.05 pages/key**, erasing the scatter entirely.

**Model.** With hit rate `h`, rows_per_key `M`, store/RAM ratio `r ≥ 1`, and
`P(evicted for a miss) ≈ 1 − 1/r`:

```
T(backend) = h·t_hit + (1−h)·cold_reads·[ (1/r)·t_mem + (1−1/r)·t_seek ]
   indexed: cold_reads = M (scattered) ;  lmdb: cold_reads ≈ 1
```

Solving `T_lmdb < T_indexed` for the store/RAM ratio `K` (hit_rate 0.9, measured
`t_seek`/`t_mem`):

| rows_per_key | lmdb speedup at store ≫ RAM | K@1.2x | K@1.5x | K@2x |
|---|---|---|---|---|
| **1 (ABI)** | **1.0x** | never | never | never |
| 2 | 2.0x | 1.0 | 1.05 | never |
| 4 | 4.0x | 1.0 | 1.0 | 1.05 |
| 8 | 7.9x | 1.0 | 1.0 | 1.0 |
| 16 | 15.8x | 1.0 | 1.0 | 1.0 |

**Reading of the model.** The store/RAM ratio is a **threshold**, not a knob:
`K ≈ 1` — a benefit appears only once the store **exceeds RAM** (so misses hit
disk); below RAM everything is resident and indexed ≈ lmdb (measured 1.0x). The
**magnitude** of the benefit is set by **rows_per_key**: under pressure lmdb is
≈ `rows_per_key` times faster on the miss path (1 clustered leaf vs N scattered
`t_seek`s). For `rows_per_key = 1` the ratio is 1.0x at **every** store/RAM ratio
— **lmdb is never worth it**.

**Practical rule:** *Use lmdb only when BOTH (a) the working set exceeds RAM
(store > ~1× RAM, so misses go to disk) AND (b) rows_per_key ≳ 2 with
key-interleaved `.data`; the expected miss-path speedup is ≈ rows_per_key.* For
the ABI symbol store (~1.03 rows/key) neither the ratio nor the scatter ever
favors lmdb. And even for multi-row stores, **sorting the indexed `.data` by key
at build time** clusters the rows (measured ~1 page/key) and removes lmdb's edge
without the dependency.

## Final default recommendation

**Make optimized + cached indexed the universal default.** The numbers support it:

- **Dependency-free** (no `lmdb` npm, no system `liblmdb`, no Symas-vs-vanilla v1
  format dance), **3x smaller on disk**, works out of the box.
- **Matches lmdb on reuse** (within ~1.1-1.4x) — and real ABI/package resolution
  is reuse-heavy (a few hot libraries/packages touched constantly), exactly where
  the cache wins. At package scale both are effectively instant (sub-100 ms for
  50k lookups).
- **Matches lmdb on resident pure-miss too** now (~1.0x) after the `.data` mmap —
  the old ~2x is gone.
- Per the cost model, the **only** regime where lmdb still wins is **disk-bound
  (store > RAM) AND multi-row-per-key with key-interleaved data**, where it is
  ≈ rows_per_key faster on misses. The ABI store (~1 row/key) never enters that
  regime; and sorting the indexed `.data` by key would close it even for
  multi-row stores.

**Keep lmdb opt-in only for that narrow case** — a store larger than RAM with
several rows per key, queried at high volume with a cold working set — where the
external dependency buys ≈ rows_per_key on the miss path. For everything the
resolver actually runs (ABI ~1 row/key; reuse-heavy access; working sets that fit
RAM), dependency-free optimized+cached+mmap indexed is the better default.

## Honesty caveats

- No hard memory cap available unprivileged on this WSL2 host (no systemd user
  bus / cgroup delegation / root); stores ≪ RAM, so fadvise-cold ≈ warm and a
  disk-bound regime is unreachable. Deterministic read/cache counters lead; they
  are exact and reproducible.
- Wall is min-of-3 and noisy on WSL2 (spreads in the raw jsonl; a couple of cold
  cells show wide tails). The ratios are robust to the noise.
- EXPERIMENT branch. The shared-runtime cache lift is answer-identical and
  golden-rebaselined here, but touches every cpp_wam indexed-store consumer, so it
  needs its own PR/review if kept.
