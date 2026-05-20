# LMDB Lazy Access: Implementation Plan

**Status**: Phased rollout. Companion to
[`WAM_LMDB_LAZY_PHILOSOPHY.md`](WAM_LMDB_LAZY_PHILOSOPHY.md) (the
"why") and [`WAM_LMDB_LAZY_SPECIFICATION.md`](WAM_LMDB_LAZY_SPECIFICATION.md)
(the "what").

**Snapshot date**: 2026-05-20.

This document sequences the work to bring L1, L2, scan-mode, and the
workload-segregation contract to each target. Haskell already has L2
(Phase L#7-9); the heaviest lift is Rust.

## 1. Current state across targets

| Target | L0 | L1 | L2 | Scan | Segregation |
| --- | :-: | :-: | :-: | :-: | :-: |
| Haskell | ✅ (`resident` IntMap) | ⚠️ degenerate (L2 with cache size 0) | ✅ (`resident_cursor` + sharded cache) | ❌ | ❌ |
| Rust | ✅ (R5/R6 matrix bench) | ❌ | ❌ | ❌ | ❌ |
| C# | ✅ + planner picks at query time | partial via planner | partial via planner | partial via planner | partial via planner |
| Go | ✅ (channel-pipeline) | ❌ | ❌ | ❌ | ❌ |
| Elixir | ✅ + `generator_mode(true)` | ❌ | ❌ | ❌ | ❌ |
| Python | ✅ + `yield from` pipeline | ❌ | ❌ | ❌ | ❌ |
| Others | ✅ | ❌ | ❌ | ❌ | ❌ |

So the gap matrix is wide. Prioritising Rust (because the R6 enwiki
result is the public-facing motivator).

## 2. Phase R7 — Rust L1 prototype

**Scope**: Add `lmdb_lazy_tier(l0|l1|l2)` codegen option to the
matrix-bench generator. Implement L1 path that skips the
`runtime_category_parents` materialisation block and registers
`category_parent/2` as a foreign predicate backed by an
`LmdbCursorLookup` struct.

**Files**:

- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` — gate
  the materialisation block on the tier option; emit a different
  registration call when `l1`.
- `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache`
  — add an `LmdbCursorLookup` struct implementing the
  `LookupSource` trait (§3 of spec doc).
- `templates/targets/rust_wam/lmdb_fact_source_heed.rs.mustache` —
  same for heed.
- `src/unifyweaver/runtime/rust/wam_rust/state.rs` (or equivalent) —
  add `register_foreign_lookup` API that accepts `Box<dyn LookupSource>`.
- `src/unifyweaver/runtime/rust/wam_rust/foreign.rs` (or equivalent) —
  modify the FFI path for predicates that have a registered
  foreign lookup, routing edge lookups through the trait instead of
  through `ffi_facts`.
- `tests/test_wam_rust_target.pl` — add codegen tests asserting
  that the lazy variant elides the materialisation block.

**Measurement**:

- Smoke test at 1k_cats: confirm tuple_count + effective_distance
  match L0.
- Bench at simplewiki (5,000 demand-set seeds): record cold T1 +
  warm median for L1 vs L0.
- Bench at enwiki (1,000 demand-set seeds): same.
- Update the Reddit report at `examples/more/.../reports/effective_distance_haskell_vs_rust.md`
  with the new numbers.

**Expected outcome**: L1 at enwiki should drop from ~148 s (L0) to
~1-10 s (lazy LMDB cursor reads + 1000 seeds × ~10 hops). If L1
beats L0 at enwiki, we've validated the lazy-streaming hypothesis.
If not, profiling reveals what's actually dominating cost.

**Estimated effort**: ~1 day. Branch: `feat/wam-rust-lmdb-lazy-l1`.

## 3. Phase R8 — Rust L2 cache layer

**Scope**: Add `CachedLookup<S>` decorator that wraps any
`LookupSource` with a sharded LRU cache. Add cost-model resolver
`resolve_lmdb_lookup_tier/2` that picks L0/L1/L2 from workload
metadata.

**Files**:

- `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache`
  — add `CachedLookup` struct.
- `src/unifyweaver/core/cost_model.pl` — add `resolve_lmdb_lookup_tier/2`
  per spec §7.2.
- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` — wire
  the resolver into the codegen so `lmdb_lazy_tier(auto)` works.
- `tests/test_wam_rust_target.pl` — codegen tests for the auto
  resolver.
- `tests/core/test_cost_model.pl` — resolver tests covering the
  decision tree (small fact_count + high seed count → l0; large +
  any seed count → l2; segregated → l1).

**Measurement**:

- Re-run the bench matrix at all three scales with `lmdb_lazy_tier(auto)`.
- Verify the resolver picks L0 at 1k, l2 at simplewiki and enwiki.
- Confirm L2 enwiki numbers match Haskell's `resident_cursor` shape
  (within an order of magnitude — the cross-language gap should
  collapse).

**Estimated effort**: ~1-2 days. Branch: `feat/wam-rust-lmdb-lazy-l2`.

## 4. Phase R9 — workload-segregation contract

**Scope**: Add the `workload_segregated(bool)` option to the
`recursive_kernel` declaration. Update the cost-model resolver to
prefer L1 when set. Thread the flag through the codegen.

**Files**:

- `src/unifyweaver/core/recursive_kernel_detection.pl` — accept
  the new option.
- `src/unifyweaver/core/cost_model.pl` — already covers this in
  R8's resolver work; verify.
- `examples/benchmark/effective_distance.pl` — example use of the
  declaration (does NOT set it by default since simplewiki/enwiki
  benchmarks aren't necessarily segregated).
- `tests/test_recursive_kernel_detection.pl` — declaration tests.

**Measurement**:

- Construct a synthetic segregated workload (e.g., 1000 seeds
  partitioned 100 each into 10 disjoint root-subtrees, run as 10
  separate process invocations).
- Verify L1 wins over L2 on the segregated workload.
- Verify L2 wins over L1 on the original non-segregated workload.

**Estimated effort**: ~1 day. Branch: `feat/wam-segregation-contract`.

## 5. Phase R10 — scan-mode (Rust)

**Scope**: Implement `ScanSource` trait + `scan_range` method on
`LmdbCursorLookup`. Add cost-model resolver
`resolve_lmdb_access_mode/2`. Update the ingest pipeline to write
`layout_strategy` to the LMDB meta sub-db.

**Files**:

- `templates/targets/rust_wam/lmdb_fact_source_*.mustache` — add
  `ScanSource` trait + impl.
- `src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py` —
  optionally pre-process IDs via topological sort if requested;
  write `layout_strategy` to meta.
- `examples/streaming/*_category_ingest*.pl` — pass through the
  optional `layout_strategy` knob.
- `src/unifyweaver/core/cost_model.pl` — `resolve_lmdb_access_mode/2`.
- Codegen wires the scan path into the kernel when chosen.

**Measurement**:

- Re-ingest 1k_cats with topological sort; compare seek vs scan at
  the same scale.
- If scan wins meaningfully on small fixtures, validate at
  simplewiki.

**Estimated effort**: ~2 days. Branch: `feat/wam-rust-lmdb-scan-mode`.

## 6. Phase H1 — Haskell L1 (validation only)

**Scope**: Haskell already implements L2 via `resident_cursor` +
sharded cache. Validate that `lmdb_cache_capacity(0)` produces a
correct L1 (no cache) behaviour and update the cost-model resolver
to recognise it.

**Files**:

- `src/unifyweaver/core/cost_model.pl` — extend `resolve_lmdb_lookup_tier/2`
  to handle the Haskell case: `l1` corresponds to
  `lmdb_cache_capacity(0)` plus `lmdb_cache_mode(none)`.
- `tests/test_wam_haskell_target.pl` — codegen tests.

**Measurement**:

- Sanity-check existing Haskell L2 results.
- Add a measurement at enwiki with cache-capacity 0 (L1) to compare
  against L2; expect L2 to win unless we contrive a segregated
  workload.

**Estimated effort**: ~0.5 day. Branch: `feat/wam-haskell-l1-recognition`.

## 7. Phase X — cross-target consolidation

**Scope**: Once Rust + Haskell both have L0/L1/L2 + scan-mode +
segregation, write a cross-target benchmark report and update the
Reddit report with all six (3 tiers × 2 access modes) data points
at simplewiki and enwiki.

**Files**:

- `examples/more/graph/effect_dist/haskell/gen/reports/effective_distance_haskell_vs_rust.md`
  — update with the post-R10 numbers; refer to this spec as the
  canonical taxonomy.
- A new `docs/design/WAM_LMDB_LAZY_BENCHMARK_REPORT.md` (or an
  appendix in the perf log) capturing the cross-target table.

**Estimated effort**: ~0.5 day. No branch — runs after R10 lands.

## 8. Dependencies

```
R7 (Rust L1) ──┐
               ├──► R8 (Rust L2 + resolver) ──┐
               │                              ├──► X (consolidation)
H1 (Haskell L1) ─────────────────────────────┤
                                              │
R10 (Rust scan) ──────────────────────────────┤
                                              │
R9 (segregation) ─────────────────────────────┘
```

- R7 is the only hard prerequisite for anything; it establishes the
  `LookupSource` trait that R8, R9, R10 all build on.
- H1 is independent (no Rust dependency).
- R10 (scan) is independent of R9 (segregation) but the cost model
  in R8 anticipates both.

## 9. Out-of-scope (for this plan revision)

- **Other targets (Go, C#, Elixir, Python) L1/L2 implementation**:
  C# already has substantial planner-level coverage; others are
  speculative. File separate implementation plans when motivated
  by a workload.
- **MST-sort ingest pre-processor**: see philosophy doc §4.2.
- **Multi-process shared cache**: see spec §11.
- **Adaptive tier switching mid-run**: see spec §11.

## 10. Open questions before R7 starts

1. **Does the WAM-Rust runtime's `ffi_facts` indirection allow a
   trait-object swap-in?** Need to read `state.rs` carefully — the
   `register_foreign_lookup` API might need a refactor of how
   foreign predicates dispatch.
2. **Cursor lifetime in Rust**: should L1 hold one shared cursor
   protected by a mutex (cheap but contended), per-thread cursors
   (more memory, lock-free), or open-per-call cursors (slow but
   simplest)? Likely per-thread for the parallel future, single
   for the L1 prototype.
3. **What's the right `cache_capacity` default for L2**: matches
   Haskell's existing default? Or sized by demand-set heuristic?
4. **Should scan-mode require a separate codegen option, or be
   chosen by the cost-model resolver automatically once
   `layout_strategy` is known**? Initial answer: automatic — keeps
   the user-facing surface small.

## 11. Concrete next step

**This PR (`docs/wam-lmdb-lazy-design`)**: lands the three design
docs. No code changes.

**Next PR (`feat/wam-rust-lmdb-lazy-l1`)**: implements Phase R7. The
follow-on PRs for R8/R9/R10 land independently as each phase
completes.

## 12. References

- `docs/design/WAM_LMDB_LAZY_PHILOSOPHY.md`
- `docs/design/WAM_LMDB_LAZY_SPECIFICATION.md`
- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md`
- `docs/design/QUERY_PLAN_RUNTIME_PHILOSOPHY.md`
- `docs/design/WAM_LMDB_RESIDENT_INTERNING_*.md` (existing LMDB layout)
- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` — current Rust eager bench
- `templates/targets/rust_wam/lmdb_fact_source_*.mustache` — current Rust LMDB source
- `templates/targets/haskell_wam/lmdb_fact_source.hs.mustache` — current Haskell L2
- `examples/more/graph/effect_dist/haskell/gen/reports/effective_distance_haskell_vs_rust.md` — the public-facing motivator
