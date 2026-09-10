# WAM C++ Target — Status

Living summary of the hybrid WAM-C++ backend
(`wam_cpp_target.pl` + `wam_cpp_lowered_emitter.pl` + C++ runtime).

Companion docs:

- [`design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md`](design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md)
- [`design/WAM_CPP_ISO_ERRORS_PHILOSOPHY.md`](design/WAM_CPP_ISO_ERRORS_PHILOSOPHY.md)
- [`design/WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md`](design/WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md)
- [`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md)
- [`WAM_RUNTIME_PARSER_STATUS.md`](WAM_RUNTIME_PARSER_STATUS.md) —
  native default + compiled cost notes.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Systems substrate + ISO-error reference.** First concrete consumer
of the shared ISO three-form contract; mutable C++ runtime that
passed classic conformance on onboarding without backend fixes.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_cpp_target.pl` | ~11.0k |
| `src/unifyweaver/targets/wam_cpp_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~6 focused files (+ large generator suite) |

## What's shipped

**Dual lowering.** WAM instruction VM + lowered emitter for
deterministic / clause-1 / ITE shapes (mirrors Rust/Haskell design).

**ISO errors (reference adopter).** Config loader, rewrite, audit
(`wam_cpp_iso_audit/3`), `catch/3`+`throw/1`, error constructors,
`is_iso`/`is_lax`, ISO/lax arithmetic compares, `succ_iso`/`succ_lax`,
lax IEEE-754 float divide. Shared contract documented in
`WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` (Elixir is the second
reference; F#/Python are partial adopters).

**Materialisation.** Arity-2 LMDB FactSource (v1) per
`WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md`.

**Runtime parser.** Default `native(parse_term)`; also supports
`compiled(prolog_term_parser)`. Audit-clean quoted-atom handling.
Compiled-parser projects balloon `generated_program.cpp` (~42k lines)
and can take ~11 min to compile — `wam_runtime.o` caching mitigates
the shared runtime portion (`WAM_RUNTIME_PARSER_STATUS.md`).

**Classic conformance.** Registered (`CONFORMANCE_TARGETS=cpp`);
**conformant on first onboarding** — cons-cell / placeholder / `is/2`
conventions already present. CLI via `emit_main(true)` → `main.cpp`
exit 0/1.

**Foreign surface.** `call_foreign` trampolines and lowered foreign
comments; **no** `recursive_kernel_detection` / shared graph-kernel
set (unlike C, which ships all 7 + bidirectional). Graph-kernel
density and effective-distance matrix presence lag Rust/Haskell/C.

**Test surface.** Few dedicated kernel e2e *files*, but
`tests/test_wam_cpp_generator.pl` alone carries on the order of
**~400** `test/1` clauses spanning ISO + LMDB + e2e — do not read
“~6 files” as a thin suite.

## Gaps

- Graph-kernel density lags Rust/Haskell/C (no shared-kernel detector).
- Compiled runtime-parser path is a compile-time cost hazard (~11 min
  for parser-bundled projects).
- ISO implementation lives in C++ rather than solely in the shared
  `iso_errors.pl` Prolog module used by F#/Python/Elixir.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. C++ audits as the best-prepared
sibling — two of the three classes are already handled:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **handled (as an atom alias)** | `sub_string/5` dispatches to the `sub_atom/5` walk in both the `Call` and `Execute` arms (`wam_cpp_target.pl:7935,7998`) — the only sibling that has it. Results intern as atoms (see A4), so SWI `string/1`-typed programs still diverge on type tests |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **aliasing form absent; frameless-Y form BROKEN → FIXED (2026-09)** | The *aliasing* form is absent — registers are string-named and Y regs live in per-frame `y_regs` (`:3321-3336`), so `"X101"` cannot collide with `"Y1"`. But the per-frame `y_regs` were **not** immunity to the *frameless-Y-write* form: a frameless callee's `env_stack.back()` IS the caller's frame, so `get_level Y1` in an `Allocate`-less ITE clause clobbered the caller's permanent with a choicepoint depth (`pick_a(3,gte(1),tagX,Out)` failed; `pick_b(...)` gave `Out=0` vs SWI `tagX`). Fixed with the CP-levels model — see "Frameless-Y ITE barrier" below |
| A3 | `Execute` of a builtin doesn't return to the continuation | **handled** | the `Execute` arm ends in a general builtin fallback followed by the Proceed protocol (`:8035-8040`), after routing catch/throw, aggregates, sub_atom/sub_string, retract/clause, `^`, phrase and `call/N` — with R, the fleet's reference implementation |
| A4 | String fidelity | **rung 0** | `Value::Tag` is Uninit/Atom/Integer/Float/Unbound/Compound (`:3039`) — no string tag; D37's double-quoted literals intern as atoms |

C++ is therefore the natural **second target** for the
`examples/cli_args/` whole-program benchmark (Class C in the fleet
doc): the known blockers are the string-type rung, not control flow.

Pattern lane: `cpp_target.pl` compiles facts from `clause(Head, true)`
— the G-A3-8 execute-at-compile-time hazard is absent; the G-A3
machinery has no analogue there.

## Frameless-Y ITE barrier (A2 frameless-Y form) — BROKEN → FIXED 2026-09

wam_cpp passes `ite_use_y_level(true)` (`wam_cpp_target.pl:439,1425,2641`),
so `compile_if_then_else/7` in the shared emitter plants `get_level Y1` /
`cut Y1` in `Allocate`-less clauses — e.g. `sat(V,gte(G)) :- \+ lt(V,G)`
(resolver's `satisfies/2`). Every Y access routes through
`env_stack.back().y_regs`; for a frameless callee that back-frame is the
**caller's**, so the old `GetLevel` handler
(`bind_cell(get_cell("Y1"), Integer(choice_points.size()))`) overwrote the
caller's permanent variable with a choicepoint depth. Reproduced as a live
wrong answer, not a crash:

    pick_a(3, gte(1), tagX, Out)  ->  FAILED    (SWI: Out = tagX)
    pick_b(3, gte(1), tagX, Out)  ->  Out = 0   (a CP depth; SWI: Out = tagX)
    wpick(4, tagY, Out)           ->  FAILED    (SWI: Out = tagY)

**Fix (C++-idiomatic CP-levels model, runtime-side only):** the barrier
level lives on the if-then-else's own choice point, never in a register.

- `ChoicePoint::levels` — a `Y-name -> choicepoint-count` map.
- `WamState::record_ite_level` (`get_level Yn`) — snapshots
  `choice_points.size()`; when the next instruction is `try_me_else` it
  **parks** the level (`pending_level_*`), because a frameless clause can be
  entered with no choice point yet; the subsequent `TryMeElse` records the
  park on the guard CP it pushes.
- `WamState::lookup_ite_level` (`cut Yn`) — searches the CP stack
  innermost-first; a missing level means the guard was already cut away
  (no-op). The level never touches a Y register.
- A parked-but-unconsumed level is cleared on backtrack and on `query`.

Per-activation for free (nested activations each carry their own barrier),
entry-path independent. Both the **interpreter lane** and the
**lowered-caller lane** (whose `call p/N` dispatches the callee through the
interpreter at its label) are covered; the pure lowered-structural ITE was
already immune (`wam_cpp_lowered_emitter.pl:717-718` emits `get_level`/`cut`
as no-ops — the commit is a C++ `if`/`else`). Mirrors the wam_rust /
wam_python / wam_go precedent (`wam_target.pl` is FROZEN — the fix is always
runtime-side).

Probe: `tests/test_wam_cpp_frameless_ite_level.pl` — pins the emission shape
(`get_level Y` in an `allocate`-less clause; caller allocates and holds Y1)
AND behaviour vs SWI on both lanes. RED on the pristine tree (5 divergences).

## §9 cut-semantics probe corpus

`tests/test_wam_cpp_cut_semantics.pl` — the 35-probe corpus ported from
`tests/test_wam_javascript_cut_semantics.pl` (via the Go/Rust ports).
SWI-oracled, driven at the interpreter lane through the generated `main`
shim (`vm.query("pNN/0", [])`); generation runs with
`on_compile_error(throw)`, so an un-compilable shape fails LOUDLY at build
time rather than silently dropping a predicate. **35/35 match SWI, 0
refused.**

Two defect classes the corpus surfaced (each RED on pristine, fixed here,
probe-pinned):

- **`!` as a meta-called goal-term (`call/1`, `call((G,!))`)** — p09, p10.
  `invoke_goal_as_call` had no `!` case, so `call(!)` failed outright.
  Fix: a `!` case that cuts to the call-scope barrier
  (`meta_call_barrier`, recorded at `dispatch_call_meta` entry) — call/N is
  a cut barrier, so the cut prunes only choice points created inside the
  call, never the caller's alternatives.
- **`!` inside a `findall`/`bagof`/`setof` goal opened by the SAME clause**
  — p13 (`e(Y), findall(X,(d(X),!),L)`), p29 (findall-with-cut nested in
  `\+`). The `!/0` builtin cut to the clause `b0`, which sits BELOW the
  aggregate's generator base when a choice point exists before the
  aggregate, collapsing the aggregate's own backtracking and any live prior
  CP. Fix: `!/0` never prunes below the innermost open aggregate's
  `base_cp_count` (ISO — the goal is opaque to cut).

## Path forward

1. Keep ISO contract as the shared reference; sync Elixir/F#/Python
   status tables when variants change.
2. Grow shared-kernel parity if C++ is used for graph benches.
3. Treat compiled-parser compile cost as a first-class constraint
   (chunked setup, stronger `.o` caching, or native-only default for
   large projects).
4. Optional: more kernel smoke tests mirroring C’s seven-kind set.

## Related: WAM-C

The sibling **C** target (`wam_c_target.pl` ~6.1k lines, no separate
lowered-emitter module) has a living checklist in
[`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md) —
**all 7 shared kernels + `bidirectional_ancestor`**, reverse-CSR
child-index paths, LMDB FactSource, aggregates/bagof/setof, lowered
**helpers** prototype (fact/filter shapes, not a full WAM lowered
emitter). Conformance registered. Do not confuse C and C++ maturity
axes: C leads on kernels/CSR; C++ leads on ISO + runtime parser.

## Document status

Snapshot for the hybrid comparison branch. Update when ISO, LMDB,
kernel, or parser-compile milestones land. 2026-09-01: added the
whole-program (A2) deficiency audit — Execute-builtin fallback and
`sub_string/5` alias verified handled; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
2026-09: frameless-Y ITE-barrier re-audit — **BROKEN → FIXED** (CP-levels
model, probe `tests/test_wam_cpp_frameless_ite_level.pl`); added the §9
cut-semantics corpus `tests/test_wam_cpp_cut_semantics.pl` (35/35 vs SWI,
two defect classes fixed: `!` in `call/N`, and `!` inside an aggregate
goal cutting below the aggregate base).
