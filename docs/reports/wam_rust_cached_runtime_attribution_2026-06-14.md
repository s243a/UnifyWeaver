# WAM-Rust Cached Runtime Attribution — Wiring Validation (2026-06-14)

Rust-side analog of the Python boundary-cache attribution from PR #3120
(`scripts/lmdb_parent_boundary_cache_benchmark.py`). That benchmark found cached
search dominated by *residual cached traversal / interpreter overhead*. The
compiled Rust WAM kernel removes the interpreter, so this work instruments what
remains on the cached path so the same question can be asked on the Rust target.

## What landed

- `CacheAttribution` in the Rust runtime (`templates/targets/rust_wam/state.rs.mustache`):
  atomic L1-hit / L2-hit / miss counters plus nanoseconds spent in the inner LMDB
  cursor seek, recorded inside `CachedLookup::lookup_parents`. Opt-in via
  `UW_WAM_CACHE_ATTRIBUTION`; behind an `Option<Arc<…>>` branch so the default
  cached path takes no extra timer calls. (Merged in PR #3127.)
- Surfacing in the **matrix benchmark** main (`examples/benchmark/generate_wam_rust_matrix_benchmark.pl`,
  LMDB-resident main): reads back the process-global sink and prints `cache_attr_*`
  lines alongside the existing `eprintln!` diagnostics. This closes the gap where
  the matrix bench (the actual cached graph-search harness) emits its own main and
  did not surface attribution.

## Smoke run (tiny int-native fixture)

Built with `tests/helpers/build_tiny_int_native_lmdb.py` (7 nodes, 5 edges,
root=2, seeds {10,11,12,20}); cached crate generated with `materialisation=cached`,
built `--release`, run with `UW_WAM_CACHE_ATTRIBUTION=1`:

```text
cache_attr_lookups=4
cache_attr_l1_hits=0
cache_attr_l2_hits=0
cache_attr_misses=4
cache_attr_hit_rate=0.0000
cache_attr_inner_lookup_ms=0.026
```

This is a **wiring validation, not a performance result**: a 5-edge graph with
four distinct seed parent-lookups and no repeated traversal yields four cold
misses and zero reuse by construction. It confirms the counters record, the inner
LMDB seek is timed, and the report surfaces in the real cached binary. With the
env var unset (or `=0`) no `cache_attr_*` lines are emitted.

## Real-graph measurement (10x fixture)

Ingested `data/benchmark/10x` (3932 category-parent edges, 1768 atoms, 250 seeds
converging on a single root) into an `lmdb_resident` fixture with
`examples/benchmark/ingest_resident_lmdb_fixture.py`, then ran the cached crate
against it with `UW_WAM_CACHE_ATTRIBUTION=1`. A `lazy` (uncached) crate is the
control.

| metric | cached | lazy (control) |
|--------|--------|----------------|
| `cache_attr_lookups` | 16337 | 0 |
| `cache_attr_l1_hits` | 15840 (1 thread) / ~15656 (N threads) | 0 |
| `cache_attr_l2_hits` | 0 (1 thread) / ~181 (N threads) | 0 |
| `cache_attr_misses` | 497 | 0 |
| `cache_attr_hit_rate` | **0.9696** | 0.0000 |
| `cache_attr_inner_lookup_ms` | ~0.23–0.30 | 0.000 |

Reading the numbers:

- **96.96% hit rate.** Of 16337 parent lookups, only 497 (3%) reach LMDB; the
  rest are served from cache. This quantifies the root-near ancestor reuse the
  benchmark plan hypothesised — 250 seeds share a small set of ancestors near the
  single root.
- **Inner LMDB time is tiny** (~0.25 ms total across the 497 misses), i.e. the
  cursor seeks are no longer the cost centre once the cache is warm — the direct
  counter to the PR #3120 Python finding where cached time was dominated by
  residual traversal/interpreter overhead.
- **Tier split behaves as designed.** Single-threaded, the one per-thread L1
  absorbs every hit (L2=0). Multi-threaded, each worker's cold L1 pushes ~181
  lookups to the shared L2; cold misses tick up slightly (497→~500) but the hit
  rate is unchanged (~0.9695). `cache_attr_lookups=16337` is deterministic.
- **Lazy is a clean control:** it never constructs `CachedLookup`, so all
  counters read zero — the attribution is specific to the cached path.

Correctness was verified end-to-end: the cached output is an **exact match** to
`data/benchmark/10x/reference_output.tsv` (183 rows), and cached and lazy produce
identical output.

This is a reuse-and-correctness result, not a wall-clock speedup claim: at this
fixture size total runtime is ~7 ms for both cached and lazy, dominated by costs
other than LMDB seeks, so the two are timing-indistinguishable here. A wall-clock
cache win is expected only on a graph large enough for cold LMDB seeks to
dominate (enwiki scale), which the high hit rate and small inner-lookup time
above suggest the cache would capture.

## Reproduce

`examples/benchmark/run_wam_rust_cached_attribution.sh <fixture_dir> <out_dir>`
automates ingest → generate (cached + lazy) → build → run with attribution. By
hand:

```bash
# 1. Ingest a graph (dir with category_parent.tsv + article_category.tsv +
#    root_categories.tsv) into an lmdb_resident fixture.
BENCH=/tmp/wam_real
python3 examples/benchmark/ingest_resident_lmdb_fixture.py \
    data/benchmark/10x "$BENCH/lmdb_resident"
cp data/benchmark/10x/article_category.tsv "$BENCH/"   # root_ids.txt is written by the ingester

# 2. Generate + build a cached crate
cd examples/benchmark
swipl -q -s generate_wam_rust_matrix_benchmark.pl -- \
    effective_distance.pl /tmp/bench_cached accumulated functions kernels_on \
    cursor auto cached 3932
( cd /tmp/bench_cached && cargo build --release )

# 3. Run with attribution
UW_WAM_CACHE_ATTRIBUTION=1 /tmp/bench_cached/target/release/bench "$BENCH"
```

## Next

The instrumentation and the reuse measurement are now both in place. The
remaining step toward the full PR #3120 comparison is to run the same recipe on
an enwiki-scale `lmdb_resident` fixture, where cold LMDB seeks dominate and the
96.96% hit rate should translate into a wall-clock cache win — turning this reuse
result into a head-to-head cached-vs-lazy runtime comparison.
