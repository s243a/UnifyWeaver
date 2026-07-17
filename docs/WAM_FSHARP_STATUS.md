# WAM F# Target — Status

Living summary of the hybrid WAM-F# backend. Distinct from the
**non-WAM** idiomatic F# compiler in [`FSHARP_TARGET.md`](FSHARP_TARGET.md).

Companion docs (prefer these for depth):

- [`WAM_FSHARP_TARGET.md`](WAM_FSHARP_TARGET.md) — usage guide, emit
  modes, builtins, LMDB/CSR options, runtime invariants.
- [`design/WAM_FSHARP_PARITY_AUDIT.md`](design/WAM_FSHARP_PARITY_AUDIT.md) —
  builtin/ISO/LMDB/CSR parity vs Haskell/Rust.
- Design: cost analyzer, CSR philosophy/parallel plan, program template
  migration.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Haskell-shaped WAM on .NET** with the fleet’s **richest LMDB
materialisation modes** (eager / lazy / cached + two-level L1/L2) and
query times competitive with Rust FFI at scale 300.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_fsharp_target.pl` | ~5.3k |
| `src/unifyweaver/targets/wam_fsharp_lowered_emitter.pl` | ~1.6k |
| `src/unifyweaver/bindings/fsharp_wam_bindings.pl` | shared helpers |
| Dedicated tests | ~27 files |

## What's shipped

**Dual lowering.** Interpreter + `emit_mode(functions)` lowered path
(mirrors Haskell; deterministic clause-1 for choicepoint-heavy bodies).

**Kernels.** Shared detection can recognize more kinds than F# accelerates.
The capability gate promotes a predicate to `CallForeign` only when the
selected kind is allow-listed **and** its F# handler exists; otherwise the
already-working WAM predicate remains the correctness path. Five F#
mustache handlers exist on disk
(`kernel_category_ancestor.fs.mustache`,
`kernel_bidirectional_ancestor.fs.mustache`,
`kernel_transitive_closure.fs.mustache`,
`kernel_transitive_distance.fs.mustache` — dist+ BFS,
`kernel_transitive_parent_distance.fs.mustache` — shortest-positive
parents). Missing handlers no longer emit undefined `nativeKernel_*`
calls. Benchmarks centre on `category_ancestor/4`. **Bidirectional**
upgrade is computed but **off by default** — requires
`allow_bidirectional_kernel_swap(true)`. CSR reverse-index reader
(`CsrLookupSource`) and cost/strategy analyzers ship.

**LMDB.** LightningDB 0.21; `ILookupSource` with eager, lazy, cached,
two-level cache, Dict unwrap; `lmdb_materialisation(...)` including
**`auto` resolver** (`resolve_auto_lmdb_materialisation_fs/2` —
parity audit “future” wording is stale) and `lmdb_l2_capacity(...)`.
Scale sweep: at 40k edges, lazy ~86× faster than eager Map
materialisation for partial demand.

**Parallelism.** TPL `Array.Parallel.map` / `runNegationParallel`
(`forkMinBranches = 3`). No Elixir-style `forkMinCost` /
`runtime_cost_probe` yet.

**ISO.** Partial adopter: catch/throw substrate, constructors,
`is_iso`/`is_lax`, arith compares, `succ` family; shared
`iso_errors.pl` consumer. Lax float divide via CLR nan/inf.

**Runtime parser.** Opt-in compiled mode; **42/42** smoke
(`test_wam_fsharp_parser_smoke.pl`). Default off.

**Emit modes.** Codegen default is **`interpreter`**;
`emit_mode(functions)` is opt-in and only ~1.0–1.07× on documented
best cases — kernels/LMDB dominate, not lowered emit.

**Perf fix.** In-place `WsRegs` mutation (PR #2428) — ~2–3× on
parser-heavy benches; pattern unique to F# among WAM targets
(`WAM_PERF_CROSS_TARGET.md`).

**CI note.** Dedicated main-workflow job
(`run_wam_fsharp_tests.pl` + LMDB oracle) — but **not** in the
classic `wam_conformance_smoke` matrix with Scala/Elixir.

## Gaps

- Finish F# kernel templates for the remaining three shared kinds
  (`transitive_step_parent_distance5`, `weighted_shortest_path3`,
  `astar_shortest_path4`) — or stop claiming full kernel parity without
  the templates.
- Bidirectional off by default; enable after cost-model confidence.
- Classic conformance (**CONF-FSHARP**, 2026-07-15): registered
  opt-in (`fsharp` / `fsharp_functions`) with additive
  `conformance_main(true)`. **Measured maturity:** all classic
  programs green on interpreter; append/reverse green under
  `emit_mode(functions)` after **FS-LIST-PARTIAL-TAIL**
  (GetValue→unifyVal); builtins also green under functions after
  **FS-FUNCTIONS-BUILTINS-LOWER** (last-slash `parse_functor_fs` for
  `///2`). No remaining fsharp ct_xfail/ct_skip.
- Dynamic database partial (facts via lowered mutation; prefer Python
  for full dynamic-DB semantics — target doc).
- LMDB scan-mode / workload-segregation wait on Rust R9/R10 reference.

## Performance notes

- Scale-300 functions mode: **~11 ms query** / ~159 ms total (.NET
  startup dominates total).
- LMDB cached fact-access-only path: ~2 ms (lookup throughput, not
  full WAM query).

## Path forward

1. CONF-FSHARP follow-ups closed (list + arith + functions/builtins).
2. Elixir-style cost-gate calibration for TPL fanout.
3. Follow Rust scan/segregation LMDB contract when available.
4. Keep ISO table in sync with C++/Elixir/Python.

## Document status

Status extract over `WAM_FSHARP_TARGET.md` +
`WAM_FSHARP_PARITY_AUDIT.md`. Prefer updating those for API detail;
update **this** file for milestone checkboxes and cross-target
ranking.
