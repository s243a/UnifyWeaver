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

## Reproduce

```bash
# 1. Generate + build a cached crate
cd examples/benchmark
swipl -q -s generate_wam_rust_matrix_benchmark.pl -- \
    effective_distance.pl /tmp/cached_bench accumulated functions kernels_on \
    cursor auto cached 297283
cd /tmp/cached_bench && cargo build --release

# 2. Build a fixture (tiny here; use a real lmdb_resident fixture for reuse numbers)
python3 <repo>/tests/helpers/build_tiny_int_native_lmdb.py /tmp/wam_fixture

# 3. Run with attribution
UW_WAM_CACHE_ATTRIBUTION=1 ./target/release/bench /tmp/wam_fixture
```

## Next

Real hit-rate / inner-lookup numbers — the actual Rust-vs-#3120 comparison —
require a fixture with genuine ancestor reuse (a SimpleWiki/enwiki
`lmdb_resident` fixture), where root-near ancestors are hit repeatedly across
seeds. The instrumentation is now in place to measure it; only the fixture is
outstanding.
