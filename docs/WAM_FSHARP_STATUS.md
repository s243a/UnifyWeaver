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

**Kernels.** Shared graph-kernel set + **bidirectional_ancestor**
upgrade path; CSR reverse-index reader (`CsrLookupSource`); cost
analyzer for strategy selection.

**LMDB.** LightningDB 0.21; `ILookupSource` with eager, lazy, cached,
two-level cache, Dict unwrap; `lmdb_materialisation(...)` and
`lmdb_l2_capacity(...)` options. Scale sweep: at 40k edges, lazy
~86× faster than eager Map materialisation for partial demand
(parity audit / e2e benches).

**Parallelism.** TPL `Parallel.map` / `runNegationParallel` race-to-true.

**ISO.** Partial adopter: catch/throw substrate, constructors,
`is_iso`/`is_lax`, arith compares, `succ` family; shared
`iso_errors.pl` consumer. Lax float divide via CLR nan/inf.

**Runtime parser.** Opt-in compiled mode; **42/42** smoke
(`test_wam_fsharp_parser_smoke.pl`). Default off.

**Perf fix.** In-place `WsRegs` mutation (PR #2428) — ~2–3× on
parser-heavy benches; pattern unique to F# among WAM targets
(`WAM_PERF_CROSS_TARGET.md`).

## Gaps

- **Not registered** in classic conformance harness (roadmap: add
  adapter).
- Dynamic database partial (facts via lowered mutation; prefer Python
  for full dynamic-DB semantics — target doc).
- LMDB scan-mode / workload-segregation wait on Rust R9/R10 reference.
- Roadmap historically said materialisation “none documented” — that
  is **stale**; modes above are shipped (corrected in
  `WAM_TARGET_ROADMAP.md` on the hybrid comparison branch).

## Performance notes

- Scale-300 functions mode: **~11 ms query** / ~159 ms total (.NET
  startup dominates total).
- LMDB cached fact-access-only path: ~2 ms (lookup throughput, not
  full WAM query).

## Path forward

1. Register classic conformance adapter.
2. Elixir-style cost-gate calibration for TPL fanout.
3. Follow Rust scan/segregation LMDB contract when available.
4. Keep ISO table in sync with C++/Elixir/Python.

## Document status

Status extract over `WAM_FSHARP_TARGET.md` +
`WAM_FSHARP_PARITY_AUDIT.md`. Prefer updating those for API detail;
update **this** file for milestone checkboxes and cross-target
ranking.
