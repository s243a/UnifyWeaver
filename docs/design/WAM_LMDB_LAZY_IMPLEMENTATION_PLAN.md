# LMDB Lazy Access: Implementation Plan

**Status**: Phased rollout. Companion to
[`WAM_LMDB_LAZY_PHILOSOPHY.md`](WAM_LMDB_LAZY_PHILOSOPHY.md) (the
"why") and [`WAM_LMDB_LAZY_SPECIFICATION.md`](WAM_LMDB_LAZY_SPECIFICATION.md)
(the "what"). Reverse-child artifact choices are tracked separately in
[`WAM_REVERSE_INDEX_ARTIFACTS.md`](WAM_REVERSE_INDEX_ARTIFACTS.md).

**Snapshot date**: 2026-05-21.

This document sequences the work to bring `lazy`, `cached`, scan-mode,
and the workload-segregation contract to each target. Haskell already
has `cached` (perf log Phase L#7-9); the heaviest lift is Rust.

## Vocabulary

This triad uses three distinct sets of identifiers — keep them separate:

- **Modes** (runtime materialisation behaviour): `eager`, `lazy`, `cached`.
- **Phases** (project sequencing labels): R7, R8, R9, R10, H1, X.
  These are *this document's* sequencing labels and are unrelated to
  the modes.
- **Cache tiers** (Haskell-internal, *within* `cached` mode):
  per-HEC L1 cache + sharded L2 cache.

See the Vocabulary block in
[`WAM_LMDB_LAZY_PHILOSOPHY.md`](WAM_LMDB_LAZY_PHILOSOPHY.md) for the
full discussion. Earlier drafts used `L0`/`L1`/`L2` as mode names;
those are stale.

## 1. Current state across targets

| Target | `eager` | `lazy` | `cached` | Scan | Segregation |
| --- | :-: | :-: | :-: | :-: | :-: |
| Haskell | ✅ (`resident` IntMap) | ⚠️ degenerate (`cached` with capacity 0) | ✅ (`resident_cursor` + sharded cache) | ❌ | ❌ |
| Rust | ✅ (R5/R6 matrix bench) | ❌ | ❌ | ❌ | ❌ |
| C# | ✅ + planner picks at query time | partial via planner | partial via planner | partial via planner | partial via planner |
| Go | ✅ (channel-pipeline) | ❌ | ❌ | ❌ | ❌ |
| Elixir | ✅ + `generator_mode(true)` | ❌ | ❌ | ❌ | ❌ |
| Python | ✅ + `yield from` pipeline | ❌ | ❌ | ❌ | ❌ |
| Others | ✅ | ❌ | ❌ | ❌ | ❌ |

So the gap matrix is wide. Prioritising Rust (because the R6 enwiki
result is the public-facing motivator).

## 2. Phase R7 — Rust `lazy` mode prototype

**Scope**: Add `lmdb_materialisation(auto|eager|lazy|cached)`
codegen option to the matrix-bench generator. Implement the `lazy`
path that skips the `runtime_category_parents` materialisation block
and registers `category_parent/2` as a foreign predicate backed by
an `LmdbCursorLookup` struct. R7 wires the `auto` token but leaves
the resolver returning `eager` until R8 — meaning callers who want
`lazy` must set it explicitly in R7.

**Files**:

- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` — gate
  the materialisation block on the mode option; emit a different
  registration call when `lazy`.
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
  that the `lazy` variant elides the materialisation block.

**Measurement**:

- Smoke test at 1k_cats: confirm tuple_count + effective_distance
  match `eager`.
- Bench at simplewiki (5,000 demand-set seeds): record cold T1 +
  warm median for `lazy` vs `eager`.
- Bench at enwiki (1,000 demand-set seeds): same.
- Update the Reddit report at `examples/more/.../reports/effective_distance_haskell_vs_rust.md`
  with the new numbers.

**Expected outcome**: `lazy` at enwiki should drop from ~148 s
(`eager`) to ~1-10 s (lazy LMDB cursor reads + 1000 seeds × ~10
hops). If `lazy` beats `eager` at enwiki, we've validated the
lazy-streaming hypothesis. If not, profiling reveals what's actually
dominating cost.

**Estimated effort**: ~1 day. Branch: `feat/wam-rust-lmdb-lazy`.

## 3. Phase R8 — Rust `cached` mode + auto-resolver

**Scope**: Add `CachedLookup<S>` decorator that wraps any
`LookupSource` with a sharded LRU cache. Add cost-model resolver
`resolve_auto_lmdb_materialisation/2` that picks `eager`/`lazy`/
`cached` from workload metadata. Wire the `auto` token to actually
call the resolver.

**Files**:

- `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache`
  — add `CachedLookup` struct.
- `src/unifyweaver/core/cost_model.pl` — add
  `resolve_auto_lmdb_materialisation/2` per spec §7.2.
- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` — wire
  the resolver into the codegen so `lmdb_materialisation(auto)` works.
- `tests/test_wam_rust_target.pl` — codegen tests for the auto
  resolver.
- `tests/core/test_cost_model.pl` — resolver tests covering the
  decision tree (small fact_count + high seed count → `eager`;
  large + any seed count → `cached`; segregated → `lazy`).

**Measurement**:

- Re-run the bench matrix at all three scales with
  `lmdb_materialisation(auto)`.
- Verify the resolver picks `eager` at 1k, `cached` at simplewiki
  and enwiki.
- Confirm `cached` enwiki numbers match Haskell's `resident_cursor`
  shape (within an order of magnitude — the cross-language gap
  should collapse).

**Estimated effort**: ~1-2 days. Branch: `feat/wam-rust-lmdb-cached`.

## 4. Phase R9 — workload-segregation contract

**Scope**: Add the `workload_segregated(bool)` option to the
`recursive_kernel` declaration. Update the cost-model resolver to
prefer `lazy` when set. Thread the flag through the codegen.

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
- Verify `lazy` wins over `cached` on the segregated workload.
- Verify `cached` wins over `lazy` on the original non-segregated
  workload.

**Estimated effort**: ~1 day. Branch: `feat/wam-segregation-contract`.

## 5. Phase R10 — scan-mode (Rust)

**Scope**: Implement `ScanSource` trait + `scan_range` method on
`LmdbCursorLookup`. Add cost-model resolver
`resolve_auto_lmdb_access_mode/2`. Update the ingest pipeline to
write `layout_strategy` to the LMDB meta sub-db.

**Files**:

- `templates/targets/rust_wam/lmdb_fact_source_*.mustache` — add
  `ScanSource` trait + impl.
- `src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py` —
  optionally pre-process IDs via topological sort if requested;
  write `layout_strategy` to meta.
- `examples/streaming/*_category_ingest*.pl` — pass through the
  optional `layout_strategy` knob.
- `src/unifyweaver/core/cost_model.pl` —
  `resolve_auto_lmdb_access_mode/2`.
- Codegen wires the scan path into the kernel when chosen.

**Measurement**:

- Re-ingest 1k_cats with topological sort; compare seek vs scan at
  the same scale.
- If scan wins meaningfully on small fixtures, validate at
  simplewiki.

**Estimated effort**: ~2 days. Branch: `feat/wam-rust-lmdb-scan-mode`.

## 6. Phase H1 — Haskell `lazy` mode (validation only)

**Scope**: Haskell already implements `cached` via `resident_cursor`
+ sharded cache. Validate that `lmdb_cache_capacity(0)` produces a
correct `lazy` behaviour and update the cost-model resolver to
recognise it.

**Files**:

- `src/unifyweaver/core/cost_model.pl` — extend
  `resolve_auto_lmdb_materialisation/2` to handle the Haskell case:
  `lazy` corresponds to `lmdb_cache_capacity(0)` plus
  `lmdb_cache_mode(none)`.
- `tests/test_wam_haskell_target.pl` — codegen tests.

**Measurement**:

- Sanity-check existing Haskell `cached`-mode results.
- Add a measurement at enwiki with cache-capacity 0 (i.e., `lazy`)
  to compare against `cached`; expect `cached` to win unless we
  contrive a segregated workload.

**Estimated effort**: ~0.5 day. Branch: `feat/wam-haskell-lazy-recognition`.

## 7. Phase X — cross-target consolidation

**Scope**: Once Rust + Haskell both have `eager`/`lazy`/`cached` +
scan-mode + segregation, write a cross-target benchmark report and
update the Reddit report with all six (3 modes × 2 access modes)
data points at simplewiki and enwiki.

**Files**:

- `examples/more/graph/effect_dist/haskell/gen/reports/effective_distance_haskell_vs_rust.md`
  — update with the post-R10 numbers; refer to this spec as the
  canonical taxonomy.
- A new `docs/design/WAM_LMDB_LAZY_BENCHMARK_REPORT.md` (or an
  appendix in the perf log) capturing the cross-target table.

**Estimated effort**: ~0.5 day. No branch — runs after R10 lands.

## 8. Dependencies

```
R7 (Rust lazy) ──┐
                 ├──► R8 (Rust cached + resolver) ──┐
                 │                                  ├──► X (consolidation)
H1 (Haskell lazy) ─────────────────────────────────┤
                                                    │
R10 (Rust scan) ────────────────────────────────────┤
                                                    │
R9 (segregation) ───────────────────────────────────┘
```

- R7 is the only hard prerequisite for anything; it establishes the
  `LookupSource` trait that R8, R9, R10 all build on.
- H1 is independent (no Rust dependency).
- R10 (scan) is independent of R9 (segregation) but the cost model
  in R8 anticipates both.

## 9. Out-of-scope (for this plan revision)

- **Other targets (Go, C#, Elixir, Python) `lazy`/`cached`
  implementation**: C# already has substantial planner-level
  coverage; others are speculative. File separate implementation
  plans when motivated by a workload.
- **Reverse-child artifact selection and CSR preprocessing**: see
  `WAM_REVERSE_INDEX_ARTIFACTS.md`. The R7-R10 plan keeps the hot
  parent-edge `lazy`/`eager`/`cached` mode selection separate from
  reverse child-index planning.
- **MST-sort ingest pre-processor**: see philosophy doc §4.2 and
  `WAM_REVERSE_INDEX_ARTIFACTS.md`.
- **Multi-process shared cache**: see spec §11.
- **Adaptive mode switching mid-run**: see spec §11.

## 10. Open questions before R7 starts

1. **Does the WAM-Rust runtime's `ffi_facts` indirection allow a
   trait-object swap-in?** Need to read `state.rs` carefully — the
   `register_foreign_lookup` API might need a refactor of how
   foreign predicates dispatch.
2. **Cursor lifetime in Rust**: should `lazy` hold one shared cursor
   protected by a mutex (cheap but contended), per-thread cursors
   (more memory, lock-free), or open-per-call cursors (slow but
   simplest)? Likely per-thread for the parallel future, single
   for the `lazy` prototype.
3. **What's the right `cache_capacity` default for `cached`**:
   matches Haskell's existing default? Or sized by demand-set
   heuristic?
4. **Should scan-mode require a separate codegen option, or be
   chosen by the cost-model resolver automatically once
   `layout_strategy` is known**? Initial answer: automatic — keeps
   the user-facing surface small.

## 11. Concrete next step

**This PR (`docs/wam-lmdb-lazy-rename`)**: tidies the
mode-vs-phase-vs-cache-tier vocabulary across the three design
docs. No code changes.

**Next PR (`feat/wam-rust-lmdb-lazy`)**: implements Phase R7. The
follow-on PRs for R8/R9/R10 land independently as each phase
completes.

## 12. References

- `docs/design/WAM_LMDB_LAZY_PHILOSOPHY.md`
- `docs/design/WAM_LMDB_LAZY_SPECIFICATION.md`
- `docs/design/WAM_REVERSE_INDEX_ARTIFACTS.md`
- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md`
- `docs/design/COST_FUNCTION_PHILOSOPHY.md`
- `docs/design/QUERY_PLAN_RUNTIME_PHILOSOPHY.md`
- `docs/design/WAM_LMDB_RESIDENT_INTERNING_*.md` (existing LMDB layout)
- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` — current Rust eager bench
- `templates/targets/rust_wam/lmdb_fact_source_*.mustache` — current Rust LMDB source
- `templates/targets/haskell_wam/lmdb_fact_source.hs.mustache` — current Haskell `cached`-mode source
- `examples/more/graph/effect_dist/haskell/gen/reports/effective_distance_haskell_vs_rust.md` — the public-facing motivator
