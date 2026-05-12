# Cache Cost Model — Philosophy

## What this is

A small, opt-in cost model that lets UnifyWeaver targets choose between
sorted-seek and sequential-scan access patterns (and decide whether
cache warming will pay off) based on first-principles estimates of
cache regime, working-set size, and query count.

This is *not* a runtime ML model. It's a closed-form formula with a
handful of hardware constants that fit on one screen. The whole point
is that it should be auditable: a human can read the numbers and
predict what the model will recommend.

## When the model matters

The cache cost model is **groundwork**, not an immediate optimization.
Most current UnifyWeaver workloads (Wikipedia categories at simplewiki
or enwiki scale) fit comfortably in modern free RAM:

- simplewiki categories: ~15 MB
- enwiki categories: ~500 MB

At this scale, the page cache holds the entire working set, every
read pattern reduces to memory bandwidth, and "always use sorted
seeks" is the right answer.

The regime starts to bite when:

1. **Article-level data** is ingested. Full enwiki article texts are
   ~80 GB compressed, ~250 GB uncompressed — well past consumer RAM.
2. **Multi-fixture aggregation** combines categories with page
   metadata, revision counts, link tables, etc.
3. **Memory-pressured environments** — WSL2 on Windows is a notable
   case where the host dynamically reclaims VM pages.
4. **Spinning-disk targets** — random seek latency jumps 100× over
   SSD, shifting the scan-vs-seek crossover left aggressively.

## What "free RAM" means (and why it's tricky)

The cost model uses *free* RAM, not total. On Linux this is read from
`/proc/meminfo:MemAvailable` — the kernel's own estimate of how much
can be allocated without swapping. This subsumes:

- The page cache (which the kernel will evict if a process needs RAM).
- Reclaimable slab.
- Memory currently backing inactive anonymous pages.

`MemAvailable` is *not* the same as `MemFree`. A system with 0 KB
"free" but 12 GB of page cache has 12 GB available — and that's
exactly the RAM that the cost model can spend on warmed working set
or rely on for memory-bandwidth-speed reads.

On WSL2 the picture is more dynamic: the Linux VM's RAM ceiling is
set in `.wslconfig` but the Windows host reclaims unused pages when
under pressure. The cost model should re-read `MemAvailable` each
time it makes a regime decision, not cache it across long-running
processes.

## The formula

Variables:

| symbol | meaning |
|---|---|
| `W` | bytes a full scan would touch (≈ DB size) |
| `X` | working-set fraction the workload actually uses |
| `R_free` | free RAM (`MemAvailable`) |
| `S_mem_seq` | sequential bandwidth from page cache (~5 GB/s) |
| `S_disk_seq` | sequential bandwidth from cold storage (~500 MB/s SSD) |
| `t_mem_seek` | one cached B-tree seek (~1 µs) |
| `t_disk_seek` | one uncached B-tree seek (~100 µs SSD, ~10 ms HDD) |
| `K` | distinct keys looked up |

Cache regime weight:

```
W_working = X * W
f_hot     = min(1, R_free / W_working)
```

Effective costs:

```
T_scan = W_working * [f_hot / S_mem_seq + (1 - f_hot) / S_disk_seq]
T_sort = K         * [f_hot * t_mem_seek + (1 - f_hot) * t_disk_seek]
```

Crossover K (sort = scan):

```
K_cross = W_working / [bandwidth_eff * latency_eff]

where  bandwidth_eff = f_hot * S_mem_seq    + (1 - f_hot) * S_disk_seq
       latency_eff   = f_hot * t_mem_seek   + (1 - f_hot) * t_disk_seek
```

Warming-pays-off threshold:

```
M_warm ≥ T_warm_load / (T_query_cold - T_query_hot)
```

The denominator collapses to ≈ 0 when `R_free ≫ W_warm`, matching the
empirical Phase M result that warming is pointless when everything
already fits. The numerator is small because warming is one sequential
scan. Warming pays off when `R_free ≪ W_warm` and `M ≥ small`.

## Hardware constants and calibration

The four hardware constants (`S_mem_seq`, `S_disk_seq`, `t_mem_seek`,
`t_disk_seek`) need to be set correctly for the running machine. Three
options, in increasing accuracy:

1. **Defaults**. SSD-typical numbers are within an order of magnitude
   on most consumer hardware; the cost model uses these for regime
   selection, not for second-decimal optimization. This is the default
   and good enough for groundwork.
2. **User override**. The Prolog predicates accept a `constants/1`
   option list so callers can plug machine-specific numbers (e.g.
   from `dd if=/dev/zero of=test bs=1M count=1024` measurements).
3. **Calibration probe**. A future enhancement: at startup, time a
   small synthetic scan + seek workload and write the resulting
   constants to a per-machine cache file. This isn't built yet because
   we're not at a scale where second-decimal accuracy matters.

What does matter is that the *order of magnitude* is right. Mixing up
SSD-typical seek latency (100 µs) with HDD-typical (10 ms) shifts the
crossover by 100×. Knowing whether the storage is rotational or solid-
state is the single most important calibration input.

## Why first-principles, not measured

We could pre-compute K_cross by running M1.b on every fixture every
time. We don't, because:

1. **Generality**: a measured number describes the fixture you measured,
   not the next fixture. The formula generalises across DB sizes.
2. **Composability**: when the workload changes (different X, different
   M, different access pattern), measured numbers are stale. The
   formula tracks.
3. **Auditability**: a reader can trace why the model picks a given
   strategy. With a measured table, "the table said so" is the only
   explanation.
4. **Speed**: the formula evaluates in microseconds. Repeated
   measurement-driven recalibration is overkill at typical query
   rates.

Measurements still play a role — they're how we *validate* the formula's
predictions and how we *calibrate* the constants. But the day-to-day
regime decisions come from the closed form.

## Connection to other resolvers

UnifyWeaver already has two cost-style resolvers in production:

- `resolve_auto_use_lmdb/2` in `wam_haskell_target.pl` — picks LMDB
  vs IntMap from a `fact_count` threshold.
- `resolve_auto_demand_bfs_mode/2` — picks cursor BFS vs in-memory
  IntMap from the same threshold (50,000 facts).
- The C# query target's `source_mode` resolver in
  `docs/design/CSHARP_QUERY_SOURCE_MODE_*.md` follows a similar
  pattern.

The cache cost model is the third in this family. The pattern they
share:

- **Inputs are workload metadata + cheap system probes**. Not running
  the workload to find out.
- **Output is a structural choice**, not a tunable knob. Pick `cursor`
  or `in_memory`, not "set cache size to 9.7 MB".
- **Defaults are conservative**. When the model can't tell, fall back
  to the safest of the candidate strategies.
- **User override is always available**. The model is a default
  picker, not a gatekeeper.

Phase 2c added `resolve_auto_cache_strategy/2` as the fourth
resolver. It uses `cost_model.pl` to decide between cursor and
in-memory demand BFS via the workload-metadata channel. A
footprint guard (Phase L#11 follow-up) overrides `in_memory` →
`cursor` when `R_free < W`.

Phase 2c+ added a fifth resolver, `resolve_auto_lmdb_cache_mode/2`,
covered in the next section.

## The `lmdb_cache_mode` auto-resolver

`resolve_auto_cache_strategy/2` picks the *access pattern*
(cursor vs in-memory). `resolve_auto_lmdb_cache_mode/2` picks the
*cache tier* on top of that. The two compose.

### Inputs

| input | source | purpose |
|---|---|---|
| `lmdb_cache_mode(auto)` | option | triggers resolution |
| `workload_locality(L)` | option (opt-in signal) | tier choice — `L ∈ {intra_thread, cross_thread, mixed, unknown}` |
| `expected_query_count(M)` | option | reused from cost_model channel; whether caching pays off |
| `demand_bfs_mode/1` | already-resolved by cache_strategy | composition: `in_memory` → no cache |

The opt-in signal is `workload_locality/1`. Without it, the new
resolver is a no-op and the existing in-place auto resolution
(`statistics:select_cache_mode/2` consulted in
`generate_lmdb_functions/2`) handles `lmdb_cache_mode(auto)`. This
preserves backwards compatibility for callers using the older
`statistics:declare_cache_hints/1` channel.

### Decision matrix

| condition | pick | rationale |
|---|---|---|
| `demand_bfs_mode = in_memory` | `none` | the edge IntMap *is* the cache; another tier is redundant |
| `M = 1` | `none` | nothing to amortise the cache fill against |
| `M > 1` and `intra_thread` | `per_hec` (L1) | per-HEC L1 wins when sparks have region affinity |
| `M > 1` and `cross_thread` | `sharded` (L2) | single shared cache, mild contention, captures inter-thread hits |
| `M > 10` and `mixed` | `two_level` (L1+L2) | both intra- and inter-thread locality |
| `M > 1` and (`mixed` with low M, or `unknown`) | `sharded` (L2) | safe default |

### Why the `unknown → sharded` default

Mirrors the conventional wisdom documented in the matrix bench:
per-HEC L1 caches duplicate hot edges across threads when sparks
have no region affinity (parMap-scheduled). Until MoE-style spark
routing lands, L1 is dominated by sharded L2 in the unknown-
locality case.

### Composition with `cache_strategy(auto)`

The two resolvers run in order: `cache_strategy(auto)` first
(picks cursor or in_memory), then `lmdb_cache_mode(auto)` reads
the result. The composition rule "in_memory → none cache tier"
is the key piece of information they share — it's what lets one
opt-in (`workload_locality(unknown)` + `cache_strategy(auto)`)
produce a self-consistent end-to-end decision.

### Memory budget guard (Phase L#13)

Symmetric to the working-set footprint guard from Phase L#11. The
cache layer needs *some* RAM to be useful; under memory pressure
the resolver picks `none` rather than risk OOM. Concretely: when
`R_free < cache_tier_floor_bytes` the resolver short-circuits to
`none` regardless of locality/M.

- Default floor: 4 MB. Enough headroom for a modest L2 with the
  default capacity, biases toward enabling caching when RAM is
  plentiful, protects against thrashing when it isn't.
- Override per-call via `cache_tier_floor_bytes(N)`.
- Reads `mem_available_bytes(R)` (option) or
  `/proc/meminfo:MemAvailable` (fallback), with a 1 GB fallback on
  non-Linux — same chain as the cache_strategy resolver.

### What this isn't doing (yet)

- **No automatic locality inference.** `workload_locality(...)`
  comes from the workload author. Inferring it from purity
  analysis or kernel metadata is research-y and deferred.

## What this isn't

- **A runtime tuner**. The model picks one strategy at codegen time
  and sticks with it. Adaptive runtime switching (e.g. "scan first,
  detect that cache is filling, switch to seeks") is a different
  problem class.
- **A workload predictor**. The model takes workload metadata as
  input. It doesn't predict, from past queries, what the next query
  will look like. That's what cache warming with query-history
  policies is for, which is downstream of this groundwork.
- **A correctness device**. Both sort and scan return the same
  answer; the model is purely a performance hint. Pick the wrong
  one and you get a slow query, never a wrong one.

## See also

- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` — Phase M appendix #10
  (measurements that motivate the formula), Phase L#11 (the
  matrix-bench validation that surfaced the footprint-guard gap),
  and the resolvers' usage notes.
- `examples/benchmark/cache_warming_microbench/` — the M1/M1.b/M2/M3
  microbench that produced those measurements.
- `src/unifyweaver/core/cost_model.pl` — the formula implementation.
- `tests/core/test_cost_model.pl` — unit tests for the predicates
  with the simplewiki/enwiki crossover values from this document.
- `src/unifyweaver/targets/wam_haskell_target.pl` —
  `resolve_auto_cache_strategy/2` and `resolve_auto_lmdb_cache_mode/2`
  (the Phase 2c/2c+ resolvers).
- `docs/design/COST_FUNCTION_PHILOSOPHY.md` — companion design doc
  covering the *cost-function variants* used by scan-strategy
  warm-build (hop_distance, exp / power-law / additive flux,
  semantic similarity), with computational profiles and an
  approximation-vs-convergence spectrum. The cost-model formulas
  here (`K_cross`, `bandwidth_eff`, `latency_eff`) gate which
  cost-function variants are affordable at runtime; the
  cost-function doc covers what each variant *means*.
