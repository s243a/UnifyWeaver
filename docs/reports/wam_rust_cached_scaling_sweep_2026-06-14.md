# WAM-Rust Cached Graph-Search Scaling Sweep (2026-06-14)

Follow-up to the cached runtime-attribution work
(`wam_rust_cached_runtime_attribution_2026-06-14.md`, PRs #3127/#3140/#3151).
That established the instrumentation and a 96.9% hit rate on a single fixture but
explicitly was **not** a wall-clock result — the 10x fixture was too small for
cold LMDB seeks to dominate. This sweep runs cached vs lazy across four fixture
sizes to chart how the cache pays off as the graph grows, directly answering the
open question from PR #3120 (the Python boundary-cache benchmark, where cached
search was *slower*).

## Method

- One `cached` crate and one `lazy` (uncached, control) crate, built once
  (fixture-independent; cache capacity overridden at runtime).
- Each `data/benchmark/<fixture>` ingested to an `lmdb_resident` database via
  `ingest_resident_lmdb_fixture.py`.
- Single-threaded (`WAM_THREADS=1`) for deterministic attribution and a clean
  cache-vs-no-cache comparison — the cache benefit measured here is lookup
  avoidance, not parallelism. `query_ms` is the min over 3 reps.
- Reproduce: `examples/benchmark/run_wam_rust_cached_scaling_sweep.sh`.

## Result

| fixture | edges | cached `query_ms` | hit rate | misses | inner LMDB ms | lazy `query_ms` | **speedup** |
|---------|-------|-------------------|----------|--------|---------------|-----------------|-------------|
| 300     |  6008 |  8 | 0.9819 |  888 | 0.48 |  18 | **2.25×** |
| 1k      |  5933 | 10 | 0.9848 | 1010 | 0.60 |  25 | **2.50×** |
| 5k      | 12981 | 39 | 0.9923 | 2254 | 1.64 | 113 | **2.90×** |
| 10k     | 25227 | 88 | 0.9936 | 4493 | 3.86 | 279 | **3.17×** |

`total_ms` tracks the same win once load is included — at 10k: cached 131 ms vs
lazy 321 ms (~2.4× end-to-end; `load_ms` is ~15 ms for both).

## Reading the numbers

- **The cache is a clear wall-clock win, and it grows with graph size**
  (2.25× → 3.17× on query time). This is the result the single-fixture report
  could not show: at 6k edges the win is already 2.25×, and it widens as the
  graph grows and cold LMDB seeks would otherwise dominate.
- **This is the direct counter to PR #3120.** In the Python prototype, cached
  search was *slower* (ratio 1.66) because residual traversal/interpreter
  overhead swamped the cache savings. The compiled Rust kernel removes the
  interpreter, so the 98–99% hit rate translates into real speedup instead of
  being hidden.
- **Hit rate rises with size** (98.2% → 99.4%): larger graphs route more seeds
  through a denser set of shared root-near ancestors, so reuse increases.
- **Inner LMDB time stays a small fraction** of query time (≤ 4 ms across ~4500
  misses at 10k) — once warm, the cursor seeks are no longer the cost centre.
- **Correctness:** for every fixture, cached output is identical to lazy output
  (`c==l`), and matches the golden `reference_output.tsv` exactly on 300/1k/5k/10k.
  (The 198-edge `dev` toy fixture is excluded: cached==lazy there too, but its
  stored reference differs at ~1e-5 — a stale floating-point reference, not a
  cache discrepancy.)

## Caveats / next

- Largest fixture here is 25k edges; the trend says the win keeps widening, so
  an enwiki-scale `lmdb_resident` fixture (millions of edges, deep cones) should
  amplify it further — the natural next measurement when such a fixture is
  available.
- Measured single-threaded to isolate the cache effect; the multi-threaded path
  adds the T7 parallelism on top (orthogonal win).
