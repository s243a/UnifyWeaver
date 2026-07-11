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

**Kernels.** All seven shared kinds, plus:

- Effective-distance **matrix** FFI path.
- **bidirectional_ancestor** (F# parity port: calibration + A*-pruned
  direction-cost search; `kernel_mode(bidirectional)`).
- Child direction from CSR artifact (`csr_child_index(true)`), LMDB
  `category_child`, or derived reverse table.
- Boundary / distribution-cache phases (see design docs).

**Materialisation.** `LookupSource` trait over LMDB; eager shipped;
lazy/cached planned (R7/R8). Reverse-CSR artifact reader. No full
Haskell-style FactSource facade yet.

**Atom interning.** u32 IDs in FFI hot path — measured **~7.9×**
query speedup at scale 300 (134 ms → 17 ms) vs string-keyed maps
([`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md)).

**Classic conformance.** Registered; opt-in; green (cons-cell /
placeholder / integer `is/2` conventions applied).

**Runtime parser.** Opt-in compiled mode; default off; generated-runtime
tests cover round-trips.

## Performance notes

- Scale-300 Rust WAM + FFI + interning: **17 ms query / 32 ms total**
  (beats pruned native DFS on query).
- Best target for single-process tight numeric / graph recursion with
  no parallelism requirement
  ([`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md) paradigm table).

## Known issues / gaps

- Lowered emitter is deterministic-only (no full non-det prefix
  lowering like Elixir’s default lowered mode).
- LMDB lazy/cached/scan/segregation still on the R7–R10 plan.
- ISO errors mostly not adopted.
- Simplewiki-scale bidirectional vs F# still an open measurement
  (see simplewiki handoff).

## Path forward

1. Simplewiki-scale bidirectional benchmark vs F#.
2. Ship LMDB lazy/cached (R7/R8) and FactSource generalisation.
3. Distribution-cache phases + builtins parity sweep.
4. Optional ISO three-form adoption once C++/Elixir/F# patterns are
   extracted.

## Document status

Snapshot for the hybrid comparison branch. Prefer updating this file
when kernel, LMDB, or conformance milestones land rather than only
adding one-off report files.
