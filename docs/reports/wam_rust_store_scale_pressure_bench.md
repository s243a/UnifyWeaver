# wam_rust store I/O lever — scale + cold-cache pressure benchmark (D107)

**Question.** D105 (#4272 key-clustering) and D106 (L1 decoded-row cache) landed on
the Rust store lane, but every benchmark so far was *warm* over a *small* store,
where the store I/O is <1% of a resolve and the levers are invisible (D105/D106
rows say so honestly). This report tests the "at scale" thesis directly: **does
the store I/O lever deliver a real WALL-CLOCK win in a large-store / cold-cache
regime, and does that justify building the deferred follow-ups (L2 raw-block
cache, LMDB tier)?**

Measurement only. No shipped runtime code, `resolver*.pl`, or the committed
`gen_scale_catalog.mjs` fixture was changed. All generators / stores live in
scratch or gitignored `store/.out/` paths.

**Verdict up front: NO — the lever does not convert to wall-clock for the
resolver's access pattern, and the cost model proves L2/LMDB will not pay
either.** The decisive fact is *scatter = rows-per-key*: the real ABI symbol
store and the pkg_resolver stores are **~1 row per key**, so a keyed lookup
already touches ~1 cold page whether the backend is indexed, clustered, or LMDB.
Clustering's cold-read win is real but *only exists for multi-row-per-key stores*
(proven: 8→1 at M=8), which these workloads are not. Recommendation: **do not
build L2 or the LMDB tier for the current workloads.**

---

## 1. Method

Three complementary measurements, mirroring the C++ store lane's methodology
(`examples/pkg_resolver/abi/bench/`) and adding the empirical disk-bound run the
C++ lane could not do (their WSL2 host had no writable `drop_caches`; **this host
is root with a writable `/proc/sys/vm/drop_caches`**).

1. **Real symbol store** (supersedes synthetic padding for the scatter question).
   `abi/ingest_symbols.mjs symbols-dir /var/lib/dpkg/info` → 107,312 `symprov/2`
   rows from 239 real Debian `.symbols` files → value `#`-packed to a scalar →
   indexed with the SHARED `scripts/js_wam/uw_fact_index.js build` (which
   key-clusters per #4272). The Rust `SeekFactSource` reads this UWFI/UWIX format
   byte-identically.
2. **Cost model** (`abi/bench/cost_model_probe.c` + `cost_model.mjs`): the
   single-page cold seek latency (`t_seek`, via `posix_fadvise(DONTNEED)`+`pread`),
   warm-page cost (`t_mem`), the scatter factor (distinct 4KB `.data` pages a key
   spans = cold reads/miss, read straight from `.idx`), then the closed-form
   store/RAM ratio `K` where a lower-scatter backend (LMDB/L2) reaches a speedup.
3. **Empirical Rust drop_caches run** over synthetic pkg_resolver stores at
   5k/100k/500k packages, `store_cache` ON vs OFF and clustered vs unclustered,
   evicting the page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`) before
   each cold process launch. Probe = one bound `resolve_layered p30` via the shim
   `--scale-probe` (reports D43 bytes-read/reads + `resolve_ms`).

Host: 16 GB RAM, ~25 GB free disk, root, no memory cgroup controller (so
`systemd-run -p MemoryMax` is a no-op here — confirmed the brief's expectation;
`drop_caches` is the pressure axis). Single box, so wall numbers carry
single-machine noise; the deterministic D43 read/byte counters and the `.idx`-derived
scatter are noise-free and are the primary evidence.

---

## 2. The real symbol store: scatter = 1 row/key

```
ingest_symbols  → 107,312 symprov rows (239 .symbols files, 0 rejected)
uw_fact_index   → 107,312 records / 107,312 keys   (9.6 MB .data + 7.1 MB .idx)
cost_model scatter:
  rows_per_key = 1
  distinct_data_pages_per_key = 1
  indexed_cold_reads_per_miss = 1   (== lmdb structural ~1)
```

Every symbol key maps to exactly one provenance row. A keyed lookup reads **one**
`.data` record on **one** page. LMDB's structural advantage is that a key's rows
cluster in ~1 B-tree leaf — but at 1 row/key the indexed store already touches 1
page. **There is no scatter for a different backend to remove.** (This container
holds fewer `.symbols` than the C++ lane's 256k-row store, but the *structure* —
1 row/key — is identical and is what the crossover depends on.)

## 3. Cost model: LMDB/L2 speedup ceiling = rows-per-key

**Disk primitive on this host** (`cost_model_probe`, 3000 random page reads):

| file | size | `t_seek` cold | `t_mem` warm | ratio | `proc/self/io` read_bytes Δ |
|---|---|---|---|---|---|
| real `symprov.data` | 9.6 MB | 540 ns | 367 ns | ~1.5× | **0** (stayed resident — fadvise did not reach disk) |
| synthetic `dep.data` | 54 MB | **50,585 ns** | 627 ns | **~80×** | **12.7 MB** (real disk faults accounted) |

The 9.6 MB file is small enough that eviction never reaches the device (cold ≈
warm — the same limitation the C++ WSL2 host hit). The 54 MB file **does** enter
the disk-bound regime: a cold single-page seek costs ~50 µs vs ~0.6 µs warm, and
the kernel accounts 12.7 MB physically read. So we genuinely measured the
disk-bound primitive.

**Scatter, clustered vs unclustered** (`cost_model.mjs scatter`; the committed
`uw_fact_index.js` key-clusters, so a scratch pre-#4272 builder was used to show
the unclustered layout):

| store | rows/key | indexed cold-reads/miss | LMDB (structural) |
|---|---|---|---|
| real ABI symprov | 1 | 1.00 | 1 |
| synthetic, **clustered** (#4272) | 2 / 4 / 8 / 16 | 1.008 / 1.024 / 1.056 / 1.125 | 1 |
| synthetic, **unclustered** (pre-#4272) | 8 | **8.00** | 1 |

This is the whole story in one table: **#4272 clustering collapses an 8-reads/miss
multi-row key to ~1** — that is the lever working, and it is exactly the win LMDB
would otherwise have. With clustering on, indexed ≈ LMDB regardless of rows/key.

**Crossover `K`** (store/RAM ratio at which a scatter=1 backend beats indexed),
with the measured disk-bound primitives (`t_seek`=50,585 ns, `t_mem`=627 ns,
`t_hit`=220 ns, hit_rate=0.9):

| rows/key | speedup at store ≫ RAM | K@1.2× | K@1.5× | K@2× |
|---|---|---|---|---|
| **1** (real ABI, pkg stores) | **1.00×** | never | never | never |
| 2 | 1.96× | 1.0 | 1.05 | never |
| 4 | 3.89× | 1.0 | 1.0 | 1.05 |
| 8 | 7.74× | 1.0 | 1.0 | 1.0 |
| 16 | 15.43× | 1.0 | 1.0 | 1.0 |

The asymptotic LMDB/L2 speedup **equals rows-per-key**; the onset `K` is ≈1 (it
appears as soon as the store stops fitting in RAM). **At 1 row/key the ceiling is
1.0× — no store/RAM ratio ever makes a different backend faster.** This
independently reproduces the C++ lane's finding
(`abi/bench/BACKEND_SELECTION.md`, `RESULTS.md`): post-clustering there is no
storage-engine crossover; LMDB keeps only ~1.6–2.2× on pure-miss, erased by
clustering on reuse.

## 4. Empirical Rust drop_caches run (synthetic pkg_resolver)

Probe = bound `resolve_layered p30`; the closure is bounded (deps point onto
earlier packages), so a probe touches a fixed slice of the store regardless of N
— growing N spreads those closure keys across a larger key-sorted `.data`, i.e.
the cold random-seek regime. Median of 5–7 runs; cold = `drop_caches` before each
launch. Answer verified correct (10-package closure) at every size.

| N_PKGS | `.data` total | config | warm ms | cold ms | cold−warm | D43 reads | D43 bytes |
|---|---|---|---|---|---|---|---|
| 5,000 | 1.1 MB | ON (clustered) | 23 | 27 | +4 | 580 | 7,361 |
| 100,000 | 24 MB | ON (clustered) | 19 | 32 | +13 | 766 | 10,040 |
| 500,000 | 122 MB | ON (clustered) | 19 | 38 | +19 | 858 | 11,549 |
| 500,000 | 122 MB | **OFF** (clustered) | 19 | 37 | +18 | **930** | **12,541** |
| 500,000 | 122 MB | ON, **unclustered** | 19 | 35 | +16 | 858 | 11,549 |

Readings:

- **Cold is real and grows with store size** (+4 → +13 → +19 ms): `drop_caches`
  does expose cold random-seek latency (each closure key on its own cold page as
  the store grows). So the regime the thesis asked for was reached.
- **L1 cache (D106): I/O-volume win, zero wall-clock.** ON vs OFF at 500k saves
  72 reads (930→858, −8%) and ~1 KB (−8%) — some keys re-seek within one resolve
  — but cold wall is **identical** (38 vs 37 ms, within noise). The saved reads
  are repeat hits on pages already faulted in, so they remove no distinct cold
  page fault.
- **Clustering (#4272): zero wall-clock here.** Clustered vs unclustered 500k
  cold is flat (38 vs 35 ms, unclustered even marginally faster within noise) —
  because the pkg_resolver stores are ~1–2 rows/key, so a key already fits ~1
  page either way. Clustering has nothing to collapse (consistent with §3: its
  win requires multi-row keys).

This matches D106's warm 5k finding (−25% reads, flat `resolve_ms`) and extends
it: **still flat even cold, even at 122 MB.**

---

## 5. Verdict and recommendation

**Does the store I/O lever show a wall-clock win at scale under cold cache? No —
not for these workloads, and the cost model says it structurally cannot.**

- The resolver's stores (ABI symprov, pkg_resolver pkg/dep/revdep) are **~1 row
  per key**. At 1 row/key the scatter is 1, so indexed, #4272-clustered, and LMDB
  all touch ~1 cold page per keyed miss. The crossover ceiling is **1.0×** at any
  store/RAM ratio (§3 K-table).
- The empirical cold Rust run confirms it: `store_cache` ON≡OFF and
  clustered≡unclustered on cold wall-clock at every size up to 122 MB, even
  though `drop_caches` genuinely exposed +19 ms of cold seek cost.
- The levers still earn their (near-zero) keep: D106 is a real −8–25% I/O-*count*
  win and D105 clustering is free and byte-transparent. Keep both.

**Should L2 (raw-block cache) or the LMDB tier be built? No — not now.**

- Their asymptotic payoff **equals rows-per-key** and only materialises when the
  store also exceeds RAM (K≈1). For a 1-row/key store that payoff is 1.0× — they
  would add code, memory, and (LMDB) a dependency for **no** wall-clock benefit.
- Even for a hypothetical multi-row store, **#4272 clustering already captures
  most of the win** (8→1.056 at M=8, §3), leaving LMDB only its ~1.6–2.2×
  pure-miss edge which reuse erases.
- **Build L2/LMDB only if** a genuine high-fan-in, **multi-row-per-key** store
  (rows/key ≥ ~4) that is also **larger than available RAM** and **miss-heavy**
  (low reuse) becomes a real workload. None of the current resolver stores are.
  Until then this is testing functionality, not utility.

## 6. Method caveats

- **`drop_caches` semantics.** `echo 3` drops page cache globally; each cold
  probe re-faults the store *and* the binary/libs (binary reload is excluded from
  `resolve_ms`, which times only the resolve). `posix_fadvise(DONTNEED)` in the C
  probe evicts per-file; on files small relative to RAM it may not reach the
  device (the 9.6 MB store: `read_bytes` Δ = 0), so the disk-bound primitive was
  taken on a 54 MB file where the kernel accounted real reads.
- **Path-baking gotcha (bit two prior agents, and this run once).** The generated
  store crate bakes the absolute `STORE_DIR` into `setup_foreign_predicates` at
  `build.sh` time; `--scale-probe DIR` only sets the total-size/`probe.json`
  source, the actual reads use the baked path. Every (size × config) here was a
  fresh `STORE_DIR=<dir> build.sh` and the baked path was grep-verified in
  `uw_resolve_wam_store/src/lib.rs` before each probe. Additionally, `build.sh`
  runs `dump_store_data` (overwriting a store dir's `*.jsonl`/`rich.jsonl` with
  the *corpus* dump) when `cases.jsonl` is absent — harmless to a probe that reads
  the pre-existing `.data`, but it clobbers the scale source files; a
  `cases.jsonl` sentinel was added to each scale dir to prevent it.
- **Synthetic vs real scale.** The 5k–500k pkg_resolver stores are synthetic
  (uniform padding); the ABI store is real Debian symbols but this container has
  239 `.symbols` (9.6 MB) vs the C++ lane's 256k-row/42 MB. Neither is
  Debian-archive scale. The scatter fact (1 row/key) and the cost-model crossover
  are structural and scale-independent; only the absolute cold-ms figures are
  box- and size-specific.
- **Single box.** Wall medians carry build-load noise (visible in the occasional
  27→34 ms warm outlier); the D43 counters and `.idx` scatter are deterministic
  and are the load-bearing evidence.

## 7. Reproduce

Scratch generators (session scratch dir, not committed):
`gen_scale_catalog_param.mjs` (parametrized N/padding copy of the fixture),
`build_unclustered.mjs` (pre-#4272 source-order store from the codec's exported
helpers), `bench.sh` (warm/cold `--scale-probe` runner). Real store + cost model:
`abi/ingest_symbols.mjs` + `abi/bench/{cost_model_probe.c,cost_model.mjs}` on
main. Full `abi/bench/bench_crossover.sh` additionally needs the npm `lmdb`
package + the C++ bench binaries (not run here; the schema-agnostic cost-model
pieces above were run standalone).
