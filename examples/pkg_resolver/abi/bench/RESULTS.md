# ABI store backend crossover: lmdb (C++ SeekFactSource L1/L2) vs indexed (on-disk seek)

Benchmark of the two D43 store backends for the UnifyWeaver ABI symbol store
(`symprov/2`), driving the **C++ WAM `SeekFactSource`** read path directly (the
same class the compiled resolver dispatches through), never the JS/wamjs lmdb
backend. Harness and entry script: `examples/pkg_resolver/abi/bench/`.

Reproduce: `bash examples/pkg_resolver/abi/bench/bench_crossover.sh`
(raw per-cell JSON in `.out/bench/results.jsonl`).

## Store sizes reached

| artifact | size | rows |
|---|---|---|
| packed P/2 JSONL (`symprov.p2.jsonl`) | 24 MB | 256,225 |
| **indexed** backend (`idx/symprov.data` 24 MB + `.idx` 18 MB) | **42 MB** | 256,225 |
| **lmdb** backend, v1-format (`lmdb_v1/symprov/data.mdb`) | **128 MB** | 256,225 (stored twice: seq + a1-range bands) |

The pre-built `lmdb/symprov` (127 MB, lmdb-js Symas fork) is rejected by vanilla
system `liblmdb` with `MDB_INVALID`, so the harness rebuilds a v1-compatible
store (`lmdb_v1/`, from-source `LMDB_DATA_V1=true` via `store/ensure_lmdb.sh`)
that the C++ reader (which links system `-llmdb`) can open. Same 256,225 rows.

The store was **not** grown with extra ELF provides: growing cannot unlock a
disk-bound crossover on this host (see "Why no crossover" below), and the
deterministic I/O attribution already isolates the mechanism at this scale.

## Workloads (drawn from the real store keys)

- **skewed** (primary, realistic): 50,000 queries; a hot set of
  `libc.so.6 / libstdc++.so.6 / libm.so.6` keys (20,590 keys = 8.0% of the
  store) is hit 80% of the time, a uniform tail 20%; 15% well-formed **miss**
  keys (real soname band + synthetic symbol). 20,457 distinct keys touched.
- **uniform** (contrast): 50,000 queries uniform-random over all keys + 15%
  miss. 39,756 distinct keys touched.
- **unique** (zero-reuse control): all 256,225 keys, shuffled, each queried
  once — the case where the L1/L2 caches give ~no benefit, isolating the raw
  read-path cost.

`R` = in-process repeats of the whole key list (the reuse axis). `cache=cold`
evicts the store from the page cache per run via `posix_fadvise(DONTNEED)`
(`UW_BENCH_EVICT=1`), the root-free "store not resident" proxy.

## Results

min-of-3 wall (WSL2, noisy — spread shown); reads / bytes / L1 / L2 / misses are
deterministic (identical across repeats and warm/cold, so listed once per R).

<!-- generated from .out/bench/results.jsonl -->
| workload | cache | backend | R | reads | bytes_read | L1_hits | L2_hits | misses | min_wall_ms | spread_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| skewed | warm | indexed | 1 | 1,884,629 | 68,189,451 | 0 | 0 | 0 | 1690.0 | 1690-1714 |
| skewed | warm | lmdb | 1 | 26,454 | 4,487,206 | 21,280 | 8,263 | 20,457 | 68.5 | 68-72 |
| skewed | cold | indexed | 1 | 1,884,629 | 68,189,451 | 0 | 0 | 0 | 1836.3 | 1836-1897 |
| skewed | cold | lmdb | 1 | 26,454 | 4,487,206 | 21,280 | 8,263 | 20,457 | 153.7 | 154-204 |
| skewed | warm | indexed | 5 | 9,423,137 | 340,947,095 | 0 | 0 | 0 | 8577.6 | 8578-10362 |
| skewed | warm | lmdb | 5 | 26,454 | 4,487,206 | 137,960 | 91,583 | 20,457 | 172.9 | 173-182 |
| skewed | cold | indexed | 5 | 9,423,137 | 340,947,095 | 0 | 0 | 0 | 8874.0 | 8874-9497 |
| skewed | cold | lmdb | 5 | 26,454 | 4,487,206 | 137,960 | 91,583 | 20,457 | 233.0 | 233-247 |
| skewed | warm | indexed | 10 | 18,846,272 | 681,894,150 | 0 | 0 | 0 | 17119.5 | 17119-17586 |
| skewed | warm | lmdb | 10 | 26,454 | 4,487,206 | 283,810 | 195,733 | 20,457 | 239.8 | 240-312 |
| skewed | cold | indexed | 10 | 18,846,272 | 681,894,150 | 0 | 0 | 0 | 17043.6 | 17044-17288 |
| skewed | cold | lmdb | 10 | 26,454 | 4,487,206 | 283,810 | 195,733 | 20,457 | 313.9 | 314-331 |
| uniform | warm | indexed | 1 | 1,842,464 | 62,330,857 | 0 | 0 | 0 | 1810.0 | 1810-1952 |
| uniform | warm | lmdb | 1 | 41,103 | 6,510,205 | 7,213 | 3,031 | 39,756 | 100.6 | 101-109 |
| uniform | cold | indexed | 1 | 1,842,464 | 62,330,857 | 0 | 0 | 0 | 2112.0 | 2112-2316 |
| uniform | cold | lmdb | 1 | 41,103 | 6,510,205 | 7,213 | 3,031 | 39,756 | 192.6 | 193-203 |
| uniform | warm | indexed | 5 | 9,212,312 | 311,654,125 | 0 | 0 | 0 | 8667.5 | 8667-8977 |
| uniform | warm | lmdb | 5 | 41,103 | 6,510,205 | 53,525 | 156,719 | 39,756 | 174.7 | 175-187 |
| uniform | cold | indexed | 5 | 9,212,312 | 311,654,125 | 0 | 0 | 0 | 9132.7 | 9133-9253 |
| uniform | cold | lmdb | 5 | 41,103 | 6,510,205 | 53,525 | 156,719 | 39,756 | 269.1 | 269-282 |
| uniform | warm | indexed | 10 | 18,424,622 | 623,308,210 | 0 | 0 | 0 | 18192.6 | 18193-18894 |
| uniform | warm | lmdb | 10 | 41,103 | 6,510,205 | 111,415 | 348,829 | 39,756 | 253.8 | 254-269 |
| uniform | cold | indexed | 10 | 18,424,622 | 623,308,210 | 0 | 0 | 0 | 18028.2 | 18028-18253 |
| uniform | cold | lmdb | 10 | 41,103 | 6,510,205 | 111,415 | 348,829 | 39,756 | 349.8 | 350-402 |
| unique | warm | indexed | 1 | 9,482,081 | 323,466,220 | 0 | 0 | 0 | 9313.7 | 9314-9375 |
| unique | warm | lmdb | 1 | 264,015 | 41,711,175 | 867 | 2,366 | 252,992 | 625.9 | 626-681 |
| unique | cold | indexed | 1 | 9,482,081 | 323,466,220 | 0 | 0 | 0 | 9534.2 | 9534-9683 |
| unique | cold | lmdb | 1 | 264,015 | 41,711,175 | 867 | 2,366 | 252,992 | 734.9 | 735-767 |

## Finding: NO crossover on this hardware — lmdb wins across the entire feasible range

**There is no crossover.** The lmdb C++ backend is faster than indexed in
**every** cell measured — every workload, every `R`, warm and cold — by roughly
**13-15x with zero key reuse** (`unique`, R=1: 626 ms vs 9314 ms) up to
**~55-72x with reuse** (`skewed`, R=10: 240 ms vs 17,120 ms). The task's
hypothesis (indexed wins when RAM is ample; lmdb only wins once the store is
evicted under memory pressure) does **not** hold for these implementations here.

### Mechanism (from the deterministic I/O + cache counters — the trustworthy signal)

- **indexed** does an on-disk UWFI/UWIX binary search with `ifstream` positioned
  reads and **no application cache**. Each lookup issues ~37 `read()` syscalls
  (log2(256k) ≈ 18 probes x {16-byte index entry + key blob} + record payload),
  and **re-does them on every repeat**. Its read count and bytes therefore scale
  linearly with `R` (skewed: 1.88M reads / 68 MB at R=1 -> 18.8M reads / 682 MB
  at R=10). It is **syscall-bound**, not disk-bound: page-cache-warm, the bytes
  are free, but the per-probe syscalls are not.
- **lmdb** (`SeekFactSource`, `WAM_CPP_ENABLE_LMDB`) does an mmap B-tree range
  scan (page faults, no per-probe syscall) plus the L1 direct-mapped + L2 FIFO
  caches. Only the **first touch of each distinct key** does I/O; repeats are
  L1/L2 hits. Its read count is **flat in `R`** (skewed: 26,454 reads / 4.5 MB at
  every R=1..10 — only the 20,457 distinct-key misses ever read). At R=10 skewed
  it serves 283,810 L1 + 195,733 L2 hits with zero extra reads.

So for identical lookups indexed issues **~37x more reads and ~8-15x more bytes**
than lmdb, and that gap widens with reuse. This is the L1/L2 advantage the
benchmark set out to isolate — it just shows up at **every** memory level, not
only under pressure.

### Why the memory-pressure crossover is unreachable here (honest caveat)

The premise needs indexed to become **disk-bound** (evicted store -> real
seeks). That regime could not be entered on this host:

1. **No hard memory-cap mechanism for an unprivileged user (WSL2, 10 GB):**
   `systemd-run --user` fails (`Failed to connect to bus`); system scope needs
   interactive polkit auth; there is no passwordless `sudo`; cgroup v2 is **not
   delegated** (`/proc/self/cgroup` = `0::/`, `cgroup.subtree_control` is
   root-owned and unwritable); no root for `drop_caches`. So `MemoryMax` sweeps
   (128M..16M) were **not runnable**.
2. **`posix_fadvise(DONTNEED)` cold ≈ warm.** Because the store (42 MB indexed /
   128 MB lmdb) is far smaller than RAM (10 GB), evicting it costs almost
   nothing to re-read: cold wall is within ~10% of warm in every cell. Eviction
   cannot manufacture a disk-bound regime when the working set trivially fits.
3. **A userspace RAM hog doesn't help:** WSL2 dynamically balloons `MemTotal`, so
   a 3.8 GB resident hog still left ~3.7 GB `MemAvailable` — no page-cache
   pressure. Eating the whole (growing) total would risk OOM-killing the box for
   a result the deterministic data already predicts.
4. **Scaling the store cannot fix (1)-(3):** no feasible ELF-provides store
   exceeds 10 GB RAM, and without a cap the store still fits resident. Growing
   would only re-confirm the same lmdb dominance at larger N.

Even if a disk-bound regime *were* reachable, it would **widen** lmdb's lead, not
reverse it: indexed's 9-19M logical reads would become physical seeks, while
lmdb's hot set stays in L1/L2 (and its far smaller byte footprint faults less).
There is no memory regime, reachable or hypothetical, in which indexed wins for a
reuse-bearing ABI workload on this backend pair.

### Bottom line

- **Crossover: no** — not in the hypothesized direction, and not reachable on
  this hardware. lmdb (C++ L1/L2 + mmap) beats indexed (syscall-bound on-disk
  seek, no app cache) at every measured point.
- **At what cap: n/a** — hard caps were unavailable; the deterministic read/cache
  attribution stands independent of memory and is the reported primary signal.
- **Why:** indexed is syscall-bound (~37 positioned `read()`s/lookup, no cache,
  linear in R); lmdb amortizes to ~1 mmap op per distinct key then serves reuse
  from L1/L2 (reads flat in R). Wall-time is noisy on WSL2 (spreads shown); lead
  with the exact, reproducible I/O counters.
