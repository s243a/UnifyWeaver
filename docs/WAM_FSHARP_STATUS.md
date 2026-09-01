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
already-working WAM predicate remains the correctness path. Handlers exist
for the allow-listed native kinds:
`category_ancestor`, `bidirectional_ancestor`, `transitive_closure2`,
`transitive_distance3`, `transitive_parent_distance4`,
`transitive_step_parent_distance5`, `weighted_shortest_path3`, and
`astar_shortest_path4`. WSP3 landed in #3847; A* landed in #3856.
Missing handlers no longer emit undefined `nativeKernel_*` calls.
Benchmarks centre on `category_ancestor/4`.
**Bidirectional** upgrade is computed but **off by default** — requires
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

- ASTAR4 (`astar_shortest_path4`) landed in #3856 (Mustache + allow-list
  + dual weighted edge/heuristic materialization + singleton
  `FFIStreamRetry`).
- No shared recursive kernel remains gated for a missing F# handler.
- Bidirectional off by default; enable after cost-model confidence.
- Classic conformance (**CONF-FSHARP**, 2026-07-15): registered
  opt-in (`fsharp` / `fsharp_functions`) with additive
  `conformance_main(true)`. **Measured maturity:** all classic
  programs green on interpreter; append/reverse green under
  `emit_mode(functions)` after **FS-LIST-PARTIAL-TAIL**
  (GetValue→unifyVal); builtins also green under functions after
  **FS-FUNCTIONS-BUILTINS-LOWER** (last-slash `parse_functor_fs` for
  `///2`). No remaining fsharp ct_xfail/ct_skip.
- Head-shape conformance (**CONF-FIX-FSHARP-LOWERED-GETVALUE**,
  2026-08-09): the four head-shape programs (`wide`, `nested`,
  `emptylist`, `repeatvar`) were run for the first time. The
  **interpreter arm was green outright**; `fsharp_functions` failed
  three queries because the lowered emitter kept its own inlined
  `get_value` — shallow `a = x` equality — and so had **never received
  the FS-LIST-PARTIAL-TAIL fix that the interpreter documents in a
  comment right next to its `unifyVal` call**. A list reached as a cons
  tail is `Str("[|]", [h; t])` while the same list as an argument is a
  compact `VList`; shallow equality rejects the two spellings, and an
  empty tail happens to match either way, which is exactly why only the
  multi-element cases failed. `emit_one_fs(get_value(...))` now
  delegates to `unifyVal`, as the interpreter does. **Standing lesson:**
  every unification fix has to be applied to both the interpreter and
  the lowered emitter — the two conformance arms exist to catch when it
  is not.
- Dynamic database partial (facts via lowered mutation; prefer Python
  for full dynamic-DB semantics — target doc).
- LMDB scan-mode / workload-segregation wait on Rust R9/R10 reference.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. F#'s audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | not in the step dispatch |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **suspected (strong)** | same design as the verified Haskell case: numeric `X→+100 / Y→+200` encoding (`wam_fsharp_target.pl:4190-4191`) plus per-frame `EfYRegs` (`:1136-1143`), which puts X101 ≡ Y1 into the *caller's* frame when the callee never allocates. The exact getReg/putReg Y-threshold was not directly confirmed, hence suspected rather than verified |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified partial** | `isIsoMetaBuiltin` (catch/throw/succ) routes through `BuiltinCall` with the correct PC→CP conversion (`:1041-1070`) — the pattern the Rust arm mirrors — but any other unlabelled builtin falls through label lookup to `None` = silent goal failure (`:1072-1083`) |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted literals intern as atoms |

The standing lesson recorded above (every unification fix must land in
both the interpreter and the lowered emitter) applies to these classes
too: an A2/A3 fix in the interpreter arm needs a matching audit of
`emit_mode(functions)`.

Pattern lane: `fsharp_target.pl` compiles facts from
`clause(Head, true)` — the G-A3-8 execute-at-compile-time hazard is
absent; the G-A3 machinery has no analogue there.

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
ranking. 2026-09-01: added the whole-program (A2) deficiency audit; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
