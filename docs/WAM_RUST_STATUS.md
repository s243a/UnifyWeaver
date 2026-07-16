# WAM Rust Target — Status

Living summary of the hybrid WAM-Rust backend. Distinct from the
**non-WAM** direct Rust stream/project compiler documented in
[`RUST_TARGET.md`](RUST_TARGET.md) (`rust_target.pl`).

Companion docs:

- Design set under `docs/design/WAM_RUST_*` (transpilation, LMDB crate
  decision, boundary distribution, bridge detector, state retrospective).
- Reports under `docs/reports/wam_rust_*` (T7 parallel, cached scaling,
  bidirectional kernel benches).
- Handoffs: [`handoff/wam_rust_simplewiki_blocker.md`](handoff/wam_rust_simplewiki_blocker.md),
  [`handoff/rust_fsharp_parity_campaign.md`](handoff/rust_fsharp_parity_campaign.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Single-core kernel king.** Native compile, register-allocated hot
loops, u32 atom interning in FFI kernels, and the densest graph-kernel
surface in the fleet.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_rust_target.pl` | ~7.1k |
| `src/unifyweaver/targets/wam_rust_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~44 files |

## What's shipped

**Dual lowering.** Full WAM instruction VM + lowered emitter for
**deterministic** predicates / clause-1 of multi-clause (ITE via
`wam_ite_structurer.pl` where applicable).

**Kernels.** All seven shared kinds (inlined in `wam_rust_target.pl`,
unlike Haskell’s mustache templates), plus:

- Effective-distance **matrix** FFI path.
- **bidirectional_ancestor** (F# parity port: calibration + A*-pruned
  direction-cost search; `kernel_mode(bidirectional)`).
- **`category_ancestor_boundary`** (Rust-only extra kind) + large
  `boundary_cache.rs.mustache` distribution-cache substrate.
- Child direction from CSR artifact (`csr_child_index(true)`), LMDB
  `category_child`, or derived reverse table.

**Materialisation.** `LookupSource` trait over LMDB. Eager / lazy /
cached branches live in `materialisation_setup.rs.mustache` and are
exercised by the matrix benchmark generator
(`examples/benchmark/generate_wam_rust_matrix_benchmark.pl`); June
2026 reports document cached vs lazy speedups. Default
`write_wam_rust_project` does **not** yet take a first-class
`lmdb_materialisation(...)` option the way F# does. Reverse-CSR
artifact reader ships. No full Haskell-style FactSource facade yet.

**Atom interning.** u32 IDs in FFI hot path — measured **~7.9×**
query speedup at scale 300 (134 ms → 17 ms) vs string-keyed maps
([`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md)).

**Parallelism.** T7 `parallel_aggregates(true)` / `parallel(true)`
with rayon (`par_aggregate.rs.mustache`); measured ~3.39× on fib
BASE=20 (`docs/reports/wam_rust_t7_speedup_benchmark.md`).

**Classic conformance.** Registered; opt-in; green (cons-cell /
placeholder / integer `is/2` conventions applied).

**Runtime parser.** Opt-in compiled mode; default off; generated-runtime
tests cover round-trips.

**Lowered emitter.** Deterministic / T4 multi-clause / T5 switch
cascade / T6 match / ITE (`wam_rust_lowered_emitter.pl`) — broader
than “clause-1 only,” but still det-centric vs Elixir’s full lowered
default path in tests.

## Performance notes

- Scale-300 Rust WAM + FFI + interning: **17 ms query / 32 ms total**
  (beats pruned native DFS on query).
- Cached vs lazy 10k: ~88 vs ~279 query_ms (~3.17×) —
  `docs/reports/wam_rust_cached_scaling_sweep_2026-06-14.md`.
- Simplewiki median ~370 ms query —
  `docs/handoff/wam_rust_simplewiki_blocker.md`.
- Best target for single-process tight numeric / graph recursion with
  no parallelism requirement
  ([`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md) paradigm table).

## Known issues / gaps

- Wire `lmdb_materialisation(...)` into default project writer (today
  mainly matrix-bench path).
- LMDB scan/segregation (R9/R10) still open.
- ISO three-form stack not adopted (catch/throw appears in builtin
  parity only).
- Simplewiki-scale bidirectional vs F# still an open measurement.
- Lowered TODO stub remains for some instruction shapes.

## Path forward

1. Simplewiki-scale bidirectional benchmark vs F#.
2. Promote LMDB lazy/cached into default project options + FactSource
   generalisation.
3. Distribution-cache / boundary phases + builtins parity sweep.
4. Optional ISO three-form adoption once C++/Elixir/F# patterns are
   extracted.

## Document status

Snapshot for the hybrid comparison branch. Prefer updating this file
when kernel, LMDB, or conformance milestones land rather than only
adding one-off report files.
