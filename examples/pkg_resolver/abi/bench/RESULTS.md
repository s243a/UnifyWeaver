# ABI store backend crossover — neutralizing the cache: indexed+L1/L2 vs lmdb

Third run in the arc. Story so far:
1. **First run:** lmdb beat `indexed` by 13-72x — but mostly because the on-disk
   binary search re-read every probe (~37 `ifstream` syscalls/lookup).
2. **Fair fight:** optimized the indexed read path (whole `.idx` slurped into RAM
   once, in-memory binary search + one `.data` record read). Gap fell to ~1.6-2.2x
   on zero-reuse but stayed 7-21x on reuse — because lmdb still had an L1/L2 **row
   cache** and indexed had none.
3. **This run:** **lift the L1/L2 row cache into the shared `SeekFactSource`** so
   BOTH backends cache identically. Now caching is neutralized and we see the
   true engine-vs-engine comparison.

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

## Final default recommendation

**Make optimized + cached indexed the universal default.** The numbers support it:

- **Dependency-free** (no `lmdb` npm, no system `liblmdb`, no Symas-vs-vanilla v1
  format dance), **3x smaller on disk**, works out of the box.
- **Matches lmdb on reuse** (within ~1.1-1.4x) — and real ABI/package resolution
  is reuse-heavy (a few hot libraries/packages touched constantly), exactly where
  the cache wins. At package scale both are effectively instant (sub-100 ms for
  50k lookups).
- The **only** place lmdb wins is pure-miss high-volume scans (~1.9-2.7x), a
  workload the resolver does not run in steady state.

**Keep lmdb opt-in for pure-miss, high-volume symbol scans** where its mmap
read-path is ~2x and the external dependency is justified. The residual 2x is
purely `read_record`'s two-reads-per-record; a follow-up (read len+payload in one
`pread`, or mmap `.data`) would likely erase even that — leaving indexed
strictly competitive everywhere. (Out of scope here.)

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
