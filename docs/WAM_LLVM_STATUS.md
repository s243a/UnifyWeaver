# WAM LLVM Target — Status

Living summary of the hybrid WAM-LLVM backend (LLVM IR → native or
WASM). Distinct from the **non-WAM** direct LLVM IR compiler in
[`LLVM_TARGET.md`](LLVM_TARGET.md) (`llvm_target.pl`).

Companion docs:

- [`design/WAM_LLVM_TRANSPILATION_SPECIFICATION.md`](design/WAM_LLVM_TRANSPILATION_SPECIFICATION.md)
- [`design/WAM_LLVM_TRANSPILATION_PHILOSOPHY.md`](design/WAM_LLVM_TRANSPILATION_PHILOSOPHY.md)
- [`design/WAM_LLVM_TRANSPILATION_IMPLEMENTATION_PLAN.md`](design/WAM_LLVM_TRANSPILATION_IMPLEMENTATION_PLAN.md)
- [`design/WAM_LLVM_LESSONS_FROM_WAT.md`](design/WAM_LLVM_LESSONS_FROM_WAT.md)
- [`WAM_PERF_CROSS_TARGET.md`](WAM_PERF_CROSS_TARGET.md) — arena-reset
  and alwaysinline notes.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Portable native codegen.** One IR pipeline to a native binary or
WASM module without tying to Rust/GHC/.NET toolchains.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_llvm_target.pl` | ~20.6k |
| `src/unifyweaver/targets/wam_llvm_lowered_emitter.pl` | ~2.1k |
| Dedicated tests (`tests/core/test_wam_llvm*`, …) | ~52 files |

Largest hybrid WAM codegen module in the fleet; densest LLVM-specific
kernel/arena/WASM test surface.

## What's shipped

**Dual lowering.** Full `@step` WAM interpreter in IR + lowered
emitter milestones **M1–M4**: single-clause, multi-clause hybrid,
pattern matching, cross-predicate call/execute closures.

**Arena runtime (M5/M6).** Growable trail / stack / choice-point /
heap. `@wam_cleanup` is an arena **reset** (not destroy) — ~18%
per-query win on the dispatch microbench after removing per-iter
`malloc(1 MiB)` (`WAM_PERF_CROSS_TARGET.md`).

**Foreign kernels.** Seven LLVM-specific kinds with
`foreign_lowering(true)` autodetect — **not the same set** as the
shared detector’s seven:

| In LLVM set | Shared-only (missing here) | LLVM-only extras |
|---|---|---|
| `category_ancestor`, `transitive_closure2`, `transitive_distance3`, `weighted_shortest_path3`, `astar_shortest_path4` | `transitive_parent_distance4`, `transitive_step_parent_distance5`, `bidirectional_ancestor` | `countdown_sum2`, `list_suffix2` |

Execution smokes: BFS / Dijkstra / A* / TC / category ancestor /
countdown / list_suffix.

**Deploy shapes.** Native `.ll` projects and WASM project writer
(`write_wam_llvm_wasm_project/3`).

## Gaps (relative to Rust / Haskell / F#)

- **No LMDB / FactSource** — arena-only materialisation story.
- **Not registered** in the classic conformance harness.
- **Not in** `wam_runtime_parser_capability.pl`.
- Real-workload effective-distance matrix thinner than Rust/Haskell/F#
  (microbench + foreign-kernel harnesses exist;
  `test_wam_llvm_realdata_benchmark.pl` / effective-distance smoke are
  the main bridges).
- Hybrid clause-1 trail-rollback for partial bindings still called out
  as follow-up in the roadmap.

## Path forward

1. Register a conformance adapter (`CONFORMANCE_TARGETS=llvm`).
2. Effective-distance-class cross-target matrix parity.
3. LMDB / LookupSource fact-source.
4. Trail-rollback for hybrid clause-1 partial bindings.
5. Optional runtime-parser capability entry if compiled/native parsing
   is needed for term IO.

## Document status

Derived from the transpilation trilogy, roadmap Table 1, perf notes,
and the hybrid comparison branch. Update when arena, kernel, or
conformance milestones land.
