# ABI store backend crossover — the FAIR FIGHT: optimized indexed vs lmdb

Follow-up to the first crossover run. That run found the `indexed` backend lost
to `lmdb` by 13-72x, but mostly for an *incidental* reason: the on-disk
`SeekFactSource` binary search re-read every probe from the stream (~37 positioned
`ifstream` `read()` syscalls per lookup, re-paid on every repeat). This run
**optimizes the real indexed backend** and re-measures on a level field, at two
scales.

Harness + entry script: `examples/pkg_resolver/abi/bench/` (drives the C++ WAM
`SeekFactSource` read path directly — not the JS lmdb backend).
Reproduce: `bash examples/pkg_resolver/abi/bench/bench_crossover.sh`.

## The optimization (shared cpp_wam runtime)

`templates/targets/cpp_wam/runtime.h.mustache`, `SeekFactSource` indexed path:
at store open the **entire `.idx` key table is slurped into RAM once**
(`idx_blob_`); a keyed lookup is then an **in-memory** binary search
(`idx_key_compare` + rewritten `lookup_offsets`) plus **one** positioned `.data`
record read. The ~37 per-probe seek+read syscalls per lookup are gone. The `.data`
file is still read with positioned `ifstream` reads (no mmap — kept simple, per
the brief). Answer-identical; see correctness below.

Effect on the deterministic read count: at symbol scale, skewed R=1 indexed reads
fell from **1,884,629 → 132,190** (~14x); the remaining reads are the per-record
`.data` reads (len prefix + payload = 2 reads/record) plus the one-time `.idx`
slurp.

## Store sizes

| scale | indexed | lmdb (v1) | rows | distinct keys |
|---|---|---|---|---|
| **symbol** (ABI `symprov/2`, `/var/lib/dpkg/info`) | **42 MB** (24 data + 18 idx) | **128 MB** | 256,225 | 249,097 |
| **package** (`store/gen_scale_catalog.mjs` 5k catalog, `pkg/2`) | **320 KB** (164 + 156 KB) | **972 KB** | 7,522 | 5,007 |

(The lmdb store is rebuilt v1-format via `store/ensure_lmdb.sh` so vanilla system
`liblmdb` can read it; the shipped lmdb-js store is `MDB_INVALID` to vanilla
liblmdb.)

## Correctness (all three guardrails pass)

1. **Built-in cross-check:** optimized-indexed `rows_found` **==** lmdb
   `rows_found` in every cell, both scales (e.g. symbol skewed R=10 = 660,940;
   package unique R=1 = 7,522). Old-indexed matches too.
2. **Resolver differential/corpus** (answer-identical, indexed backend, with the
   optimized runtime): `run_differential_cpp_store.sh` = **503 cases, 0
   divergences**; `run_corpus_cpp_store.sh` = **51 cases, 0 divergences**;
   `run_abi_verify.sh` = **122 passed, 0 failed**.
3. **Byte-frozen goldens** (`tests/test_wam_cpp_templates.pl`): re-baselined the
   two header digests (plain 90019→91891, lmdb 90260→92132; +1872 chars each,
   gate-independent). Full suite green. Runtime-source golden unchanged (I did
   not touch `runtime.cpp.mustache`).

Frozen resolver files (`resolver.pl` / `resolver_store.pl` / `debian/`) untouched
(`git diff` clean).

## Results

min-of-3 wall (WSL2 — noisy; warm and fadvise-cold both shown). Reads / cache
counters are deterministic (identical warm/cold and across repeats).

### Headline: min wall (ms) and speedups

| scale | workload | cache | R | old-idx | **opt-idx** | lmdb | opt speedup vs old | lmdb vs opt |
|---|---|---|---|---|---|---|---|---|
| package | skewed | warm | 1 | 946 | **117** | 10 | 8.1x | 12.2x |
| package | skewed | warm | 5 | 4601 | **573** | 30 | 8.0x | 19.1x |
| package | skewed | warm | 10 | 9359 | **1193** | 56 | 7.8x | 21.5x |
| package | uniform | warm | 1 | 934 | **119** | 11 | 7.9x | 11.1x |
| package | uniform | warm | 10 | 9038 | **1122** | 61 | 8.1x | 18.3x |
| package | unique | warm | 1 | 88 | **12** | 5 | 7.3x | 2.2x |
| package | unique | warm | 5 | 465 | **64** | 8 | 7.3x | 8.2x |
| package | unique | warm | 10 | 926 | **133** | 10 | 6.9x | 12.9x |
| symbol | skewed | warm | 1 | 1412 | **167** | 63 | 8.5x | 2.6x |
| symbol | skewed | warm | 5 | 7593 | **911** | 218 | 8.3x | 4.2x |
| symbol | skewed | warm | 10 | 16406 | **1799** | 236 | 9.1x | 7.6x |
| symbol | uniform | warm | 1 | 1659 | **188** | 122 | 8.8x | 1.5x |
| symbol | uniform | warm | 10 | 16791 | **1760** | 252 | 9.5x | 7.0x |
| symbol | unique | warm | 1 | 8724 | **959** | 593 | 9.1x | 1.6x |
| symbol | unique | cold | 1 | 9556 | **1243** | 714 | 7.7x | 1.7x |

(Full warm+cold sweep, all R, both scales: `.out/bench/results.symbol.jsonl` and
`results.package.jsonl`. Cold ≈ warm everywhere — the stores are far smaller than
RAM, and no hard memory-cap mechanism is available unprivileged on this WSL2 box,
so a disk-bound regime is still unreachable; the read/cache counters are the
trustworthy signal, as in the first run.)

### Deterministic I/O (warm; the primary signal)

| scale | workload | R | old-idx reads | opt-idx reads | lmdb reads | lmdb L1 | lmdb L2 | lmdb miss | rows |
|---|---|---|---|---|---|---|---|---|---|
| package | skewed | 1 | 1,330,740 | 142,486 | 6,238 | 45,052 | 981 | 3,967 | 71,242 |
| package | skewed | 10 | 13,307,382 | 1,424,842 | 6,238 | 480,499 | 15,534 | 3,967 | 712,420 |
| package | uniform | 10 | 13,304,712 | 1,419,902 | 7,515 | 441,954 | 53,045 | 5,001 | 709,950 |
| package | unique | 1 | 133,879 | 15,046 | 7,522 | 0 | 0 | 5,007 | 7,522 |
| package | unique | 10 | 1,338,772 | 150,442 | 7,522 | 33,597 | 11,466 | 5,007 | 75,220 |
| symbol | skewed | 1 | 1,884,629 | 132,190 | 26,454 | 21,280 | 8,263 | 20,457 | 66,094 |
| symbol | skewed | 10 | 18,846,272 | 1,321,882 | 26,454 | 283,810 | 195,733 | 20,457 | 660,940 |
| symbol | uniform | 10 | 18,424,622 | 898,482 | 41,103 | 111,415 | 348,829 | 39,756 | 449,240 |
| symbol | unique | 1 | 9,482,081 | 540,964 | 264,015 | 867 | 2,366 | 252,992 | 270,481 |

## Verdict: does lmdb still win on a level field?

**Yes — lmdb still wins at BOTH scales, but the margin collapses, and the residual
gap is caching, not the storage engine.** The optimization removed ~8-9.5x of the
old indexed deficit uniformly (the per-probe syscalls). What is left:

- **Zero key reuse** (`unique` R=1, the fairest — caches are useless): lmdb wins
  only **1.6x** (symbol) / **2.2x** (package). This residual is purely
  read-path: opt-indexed does 2 positioned `.data` reads per record (len +
  payload) vs lmdb's single mmap value fetch — syscalls vs page faults on the
  ~same bytes.
- **Reuse-bearing** (`skewed`/`uniform`, and higher R): the gap grows with reuse
  — up to **7.6x** (symbol skewed R=10) and **21.5x** (package skewed R=10) —
  because lmdb's L1 (direct-mapped) + L2 (FIFO) **row cache** serves repeats with
  zero reads (its read count is FLAT in R: 6,238 / 26,454 regardless of R),
  while opt-indexed has no cache and re-reads every record every repeat (reads
  scale linearly with R). The deterministic columns make this explicit: at
  symbol skewed R=10, lmdb does 26,454 reads and 283,810+195,733 cache hits;
  opt-indexed does 1,321,882 reads.

**Same direction at both scales; the size of the win is set by key-reuse, not by
store size.** Package scale looks *more* lopsided only because a small keyset
under a fixed query count means heavy reuse (50k queries over ~5k keys), which is
exactly lmdb's cache regime. On the reuse-neutral control the two scales agree
(~1.6-2.2x).

Crucially, **the remaining lmdb advantage is its application cache, which is not
intrinsic to lmdb.** An equivalent L1/L2 row cache over decoded records could be
added to the indexed backend and would erase the reuse-driven 7-21x, leaving only
the ~2x cold read-path difference (which mmap-ing `.data` would further narrow).

## Recommendation

- **Make optimized-indexed the universal default.** It is dependency-free (no
  `lmdb` npm, no system `liblmdb`, no Symas-vs-vanilla v1 format dance), ~3x
  smaller on disk (42 MB vs 128 MB at symbol scale), works out of the box, and is
  now within **~1.6-2.2x** of lmdb on cache-neutral access. At package scale —
  the domain the default actually serves — both are effectively instant
  (sub-150 ms for 50k lookups), so the external dependency buys nothing that
  matters there. This directly tempers the "lmdb-by-default always" instinct.
- **Add the row cache to indexed** (follow-up, own PR): an L1/L2 over decoded
  records keyed by the encoded key would make the dependency-free backend
  competitive with lmdb across the board, since the cache — not the B-tree — is
  the remaining differentiator. (Out of scope here; the brief scoped the change
  to the `.idx` load.)
- **Keep lmdb as an opt-in for high-volume, high-reuse symbol resolution at
  scale** (re-touching a few hot libraries under sustained query load), where its
  row cache still gives 7x+ today and it avoids re-reads entirely — accepting the
  external dependency and the larger store.

## Honesty caveats (unchanged from the first run)

- Hard memory caps remain unavailable unprivileged on this WSL2 host (no systemd
  user bus, no cgroup delegation, no root); stores ≪ RAM, so `fadvise`-cold ≈
  warm and a disk-bound regime is not reachable. Lead with the deterministic
  read/cache counters; they are exact and reproducible.
- Wall time is noisy on WSL2 (min-of-3, spreads in the raw jsonl). The ratios
  above are robust to the noise; treat single-cell wall values as indicative.
- This is an EXPERIMENT branch. The shared-runtime `.idx`-in-RAM change is
  answer-identical and golden-rebaselined here, but if kept it needs its own
  PR/review (it affects every cpp_wam indexed store consumer, not just this bench).
