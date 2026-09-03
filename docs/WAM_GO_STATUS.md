# WAM Go Target — Status

Living summary of the hybrid WAM-Go backend
(`wam_go_target.pl` + `wam_go_lowered_emitter.pl`). Distinct from the
**non-WAM** direct Go stream/dataflow compiler documented in
[`GO_TARGET.md`](GO_TARGET.md) (`go_target.pl`), which remains Go's
**default** product path.

Companion docs:

- [`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md) — parity vs Rust/Haskell.
- [`GO_TARGET.md`](GO_TARGET.md) — non-WAM sibling compiler.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Strong Go runtime backend.** Broad builtin/IO/aggregate surface with
a full FFI graph-kernel set, reachable when a workload opts into the
WAM pipeline via `prefer_wam(true)`.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_go_target.pl` | ~4.3k |
| `src/unifyweaver/targets/wam_go_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~23 files |

## What's shipped

**All 7 FFI kernels.** Full shared-detector set via
`go_foreign_lowering` / FFI dispatch. (`go_supported_shared_kernel/1`
lists 5; weighted/A* are separate arms — the effective set is all 7.)

**Fact sources.** TSV and LMDB atom-fact paths. The LMDB source
carries the full `lmdb_materialisation(eager|lazy|cached|auto)` +
`lmdb_l2_capacity(N|auto)` tier set (LMDB-GO), matching F#: eager
materialises once at construction, lazy spawns the helper per lookup,
cached memoises through an L1/L2 pair. `auto` defers to the shared cost
model in `core/cost_model.pl`. Default is `cached`.

**ISO three-form errors.** Full adoption (ISO-GO): `catch/3` + `throw/1`
via panic/recover, `error(Formal, Context)` constructors, `is_iso`/
`is_lax`, ISO/lax forms of the six arithmetic comparisons, `succ_iso`/
`succ_lax`, per-predicate key rewrite, and `wam_go_iso_audit/3`.
Rewriting is opt-in: without an `iso_errors` option the WAM text keeps
its plain keys. `catch/3` is deterministic — it commits to the goal's
first solution.

**Effective-distance benchmark.** Scale-300 row populated (BENCH-GO):
query_ms 898 / total_ms 898, 5-rep median, output an exact multiset
match against the reference. See
[`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md).

**Compiled runtime parser.** `runtime_parser(compiled)` compiles the
portable `prolog_term_parser` and the target-agnostic wrappers into the
project, so a generated program can call `read_term_from_atom/2`
directly (PARSE-GO). Default stays `none`. Proven end-to-end by
`tests/test_wam_go_parser_smoke.pl` — 22 inputs including operator
precedence and associativity.

**Dual lowering.** WAM instruction VM plus the lowered emitter.

**Broad surface.** Builtins, IO, and aggregate coverage tracked in the
parity audit against Rust/Haskell.

**Conformance.** Registered `conformance_target(go)` and green — but
**requires `prefer_wam(true)`**, because the default Go strategy is the
non-WAM dataflow/stream compiler, not the WAM pipeline. Stays opt-in
(needs a `go` per-program build), not default CI.

## Gaps (relative to Rust / Haskell / F#)

- **Default product path is non-WAM** (`go_target.pl`); the WAM route
  is opt-in.
- **Effective-distance row is interpreter-bound** (~898 ms query vs
  Rust's 17 ms): the benchmark runs `category_ancestor/4` through the
  shared bytecode loop rather than the FFI kernel path.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Go's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | Go has `sub_atom/5` (`state.go.mustache:2792`) but no `sub_string/5` |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **aliasing form: verified (structural), Call Y-save is a partial mitigation, X101≡Y1 not hit by uw-resolve. Frameless-Y form: REPRODUCED as a live wrong answer on the lowered lane → FIXED 2026-09** | encoding `X_n→n+99 / Y_n→n+199`, so **X101 ≡ index 200 ≡ Y1**. Call snapshots Y 200..299 and Proceed restores; Execute does **not** push (LCO). Choice points snapshot the Y-save stack. The numeric alias is unchanged. See the frameless-Y section below. |
| A3 | `Execute` of a builtin doesn't return to the continuation | **handled for known builtins; `call/1` now classified** | `BuiltinExecute` takes Proceed's return path **including** `popCallFrame`. Residual: a missed classifier entry still silently fails. `call/1` is `wam_go_direct_builtin` → `BuiltinCall`. `member/2` in uw-resolve is `BuiltinCall`. |
| A4 | String fidelity | **rung 0** | `value.go.mustache` has Integer/Float/Atom/Compound/Structure/List/Ref/Unbound — no string type; D37's double-quoted literals intern as atoms |

Pattern lane: `go_target.pl` compiles facts from `clause(Head, true)`
(`:6790-6798`) — the G-A3-8 execute-at-compile-time hazard is absent.
The G-A3 machinery has no analogue; presume those gaps present until
`examples/cli_args/` is attempted through the Go pattern lane.

## Cut and choice-point barriers (2026-09)

Pinned by `tests/test_wam_go_cut_semantics.pl` (35 probes, SWI oracle,
`prefer_wam(true)` only — Go has no `emit_mode`). **35/35 vs SWI, 0
refused-loudly.** The JS audit found 12 divergences of this class (`!`
wiping ALL choice points). This backend now matches §9:

- Call = `pushCallFrame` (Y-save + push B0, then `PendingB0 = len(CPs)`).
- Execute = `enterExecute` (rebase `PendingB0` **without** pushing).
- Proceed / `BuiltinExecute` = `popCallFrame`.
- `!/0` truncates to `PendingB0`, not `EnvFrame.CutB0` (no-Allocate
  neck-cut).
- Every choice point snapshots `PendingB0`, `CutB0Stack`, and `YSaves`.

uw-resolve plus the 35-probe corpus also forced:

- **`call/1` opaque scope.** Previously a missing label (silent fail).
  Now a builtin whose `!` truncates only to the metacall entry height.
  Residual: nested *user* goals inside `call/1` are first-solution
  (leftover CPs are not resumed as extra metacall solutions). p09/p10
  do not need that path.
- **`inline_bagof_setof(true)`.** bagof/setof compile to
  `BeginAggregate` instead of `call bagof/3`. Empty bagof/setof fail;
  setof sorts.
- Aggregates still run in `Clone()`; an inner `!` cannot destroy the
  caller's CPs. `freezeTerm` copies collected templates out of the clone
  (findall nested Unbounds).

ITE condition / `\+` / `once` / `forall` remain the M17 soft-cut
rewrite. Probe count and any refused-loudly shapes are reported by the
suite (currently 35 compile).

## A2 frameless-Y: the if-then-else barrier (2026-09-03)

**Verdict: wam_go WAS exposed — reproduced as a live wrong answer on the
lowered lane, and fixed.** Pinned by
`tests/test_wam_go_frameless_ite_level.pl` (RED on the pristine tree with
2 divergences, green after).

The shape (ledger D50/D52). `compile_if_then_else/7` in `wam_target.pl`
reserves the ITE barrier's permanent register *after* it has decided
whether the clause needs an environment, so under `ite_use_y_level(true)`
— which **every** wam_go compile passes (`classify_predicates/3`, all
four `go_compile_predicate_to_wam` arms) — it plants `get_level Y1` …
`cut Y1` into clauses with **no `allocate`**. The canonical instance is
`sat(V, gte(G)) :- \+ lt(V, G)`, the resolver's `satisfies/2` shape.

What we found, lane by lane:

- **Interpreter lane: already safe, and here is why.** D51's §9 work
  added `vm.YSaves` — `Call` (`pushCallFrame`) snapshots `Regs[200:300]`
  and `Proceed` (`popCallFrame`) restores it, with every choice point
  saving and restoring the whole `YSaves` stack. That is the
  wam_javascript Call-snapshot model, and it repairs the caller's Y1
  after the frameless callee scribbles on it. Verified causally, not by
  inspection: with the `copy` in `popYSave` disabled, a 14-shape
  differential against SWI went from 14/14 correct to
  `pick_b(3,gte(1),tagX,Out) = 1` (a choice-point depth), `pick_a` failing
  outright, and a `findall` returning `[2,1]` instead of `[2,3]`.
  Restoring the copy made all 14 correct again.
- **Lowered lane: BROKEN.** `wam_go_lowered_emitter.pl` emits `call p/N`
  as raw label dispatch —
  `func() bool { if pc, ok := vm.Ctx.Labels["p/N"]; ok { vm.PC = pc; return vm.Run() }; return false }()`
  — with **no `pushCallFrame`**, so no Y save is pushed and nothing is
  restored; its `allocate` likewise pushes a bare `&EnvFrame{CP:…, B0:…}`
  with no `SavedYRegs` copy. A lowered method that parks a permanent in
  `vm.Regs[200]` and then calls an interpreted frameless-ITE callee
  therefore got its Y1 overwritten. On the pristine tree, against SWI's
  `Out = tagX`:

  ```
  vm.PredPick_b4(3, gte(1), tagX, Out)  ->  ok=true,  Out = 0   (a CP depth)
  vm.PredPick_a4(3, gte(1), tagX, Out)  ->  ok=false            (silent fail)
  ```

  while the same predicate entered at its interpreter label returned
  `tagX` for both. This lane is how `examples/pkg_resolver/go/shim.go`'s
  sibling embedders reach a predicate, so it is not hypothetical.

**The fix** takes the barrier out of the register file entirely — the
wam_rust `ChoicePoint::levels` / wam_python `ChoicePoint.levels` model:

- `ChoicePoint.Levels map[int]int` (nil until used) keyed by the
  Y-register index the emitter named.
- `recordIteLevel/1` replaces the `putReg` in the `GetLevel` case. The
  level is unchanged (`len(vm.ChoicePoints)` at that instant); only its
  home changes. Both emission shapes are handled: `get_level` immediately
  *before* the ITE `try_me_else` parks the level (`PendingLevel*` on the
  VM) for `fillBarrier` to record on the guard choice point that
  `try_me_else` pushes; `get_level` immediately *after* it (the shape used
  when the condition holds a top-level `!`) attaches to the guard that
  already exists. Parking is load-bearing: the clause can be entered with
  zero choice points, so there is nowhere to record until the guard is
  pushed.
- `lookupIteLevel/1` replaces the `getReg` in the `Cut` case, walking the
  choice-point stack innermost-first; "not found" means an inner commit
  already cut the guard away and the truncation is a no-op.
- `restoreBarrier` clears a parked-but-unconsumed level, so it cannot
  outlive a backtrack.

Because the level never touches `vm.Regs`, the fix is independent of how
the predicate was entered — it closes the lowered lane as well as the
interpreter lane, and it does not depend on `YSaves` (which stays, since
it still covers `get_variable Yn` in Allocate-less clauses).

Files: `templates/targets/go_wam/state.go.mustache` (`ChoicePoint.Levels`,
`PendingLevel*`, `recordIteLevel`, `lookupIteLevel`, `fillBarrier`,
`restoreBarrier`), `templates/targets/go_wam/instructions.go.mustache`
(doc), `src/unifyweaver/targets/wam_go_target.pl`
(`wam_go_case('GetLevel')`, `wam_go_case('Cut')`).

Gates re-run after the fix: cut probes **35/35** (0 refused), the five
D51 probe suites, the full 24-suite `tests/test_wam_go_*.pl` sweep,
corpus **39/39**, differential **2400/0/0**. The corpus and differential
are *not* evidence for this fix — `shim.go` enters every predicate by
label, never through a lowered method, so the resolver never exercised
the broken lane. They are evidence the fix is neutral (and it is:
pristine differential 27.9 s, fixed 25.2–27.1 s on the same box).

### Residual (separate defect, NOT this one)

A lowered method whose clause body contains a `call` into the
interpreter and then tail-`execute`s itself **loses the output binding**.
`wpick(4, tagY, Out)` returns `ok=true` with `Out` still unbound from
`vm.PredWpick3()`, while the interpreter lane returns `tagY`. This is not
the ITE barrier: the control `wcall/3` — same recursive shape, a plain
`call okp/1` in the body, no if-then-else or negation anywhere — fails
identically, and both fail the same way before and after the barrier fix
(a second control, `wplain/3`, with no `call` at all, is correct on both
lanes at every depth). Suspected cause, not verified: the lowered emitter
does not implement the §9 call-frame protocol at all — `emit_one(call)`
omits `pushCallFrame` and `emit_one(allocate)` omits the `SavedYRegs`
copy, so the interpreted callee's `Proceed`/`popCallFrame` pops a frame
the lowered caller never pushed. `tests/test_wam_go_frameless_ite_level.pl`
runs `wpick` on the interpreter lane only and says why.

## Whole-program exercise: uw-resolve (`examples/pkg_resolver/go/`)

P0.5 resolver compiled through `wam_go` (`prefer_wam(true)`). JSON shim
is term↔JSON IO only. Corpus **39/39** vs SWI; seeded differential
**2400/0**. Additional runtime bugs the program forced (not in the A2
table): empty-list `GetConstant` vs `*List`; `sort/2` unique-collapsing
compounds; `switch_on_structure` emission using `Val` instead of
`Functor`; 4-arg `begin_aggregate`; `allocVarId` aliasing driver Idx
10000–10999 (B3 `sort/2` unifying Acc with `[]`). **B3** on the 5k
catalog (`0xc0ffee01`): Go load **0.060s** / resolve **4.604s**, same
10-package selection as SWI (load **0.428s** / resolve **0.008s**). See
`examples/pkg_resolver/go/README.md`.

## Path forward

1. Decide whether the WAM route should become a first-class Go product
   path or stay the kernel-benchmarking arm.
2. Route the effective-distance benchmark through the registered
   foreign kernels — the single biggest lever on the Go perf row.
3. Consider registering `conformance_target(go)` as default CI rather
   than opt-in, now that the runtime has been hardened by the parser
   bring-up.

## Document status

Fleet-aligned snapshot; source-verified against `wam_go_target.pl`,
the parity audit, and the conformance harness (`prefer_wam(true)`
requirement confirmed) on 2026-07-11. Refreshed 2026-08-06 after all four Go
gap cards (LMDB-GO, ISO-GO, BENCH-GO, PARSE-GO) landed. Update the parity audit first,
then refresh here. 2026-09-01: added the whole-program (A2) deficiency
audit — X→Y aliasing and the `BuiltinExecute` mitigation verified by
source reading; see [`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
2026-09-03: uw-resolve on Go + 35-probe cut audit (35/35, 0 refused).
A2 still structural (Call Y-save is a partial mitigation). A3 residual
remains for unclassified builtins; `call/1` now classified. Cut suite
is `prefer_wam(true)` only. B3 5k `resolve_layered` matches SWI after
`allocVarId` skips Idx 10000–10999 (Go resolve 4.604s vs SWI 0.008s).
2026-09-03 (later): **A2's frameless-Y form reproduced and fixed** —
broken on the lowered lane (`vm.PredPick_b4` returned a choice-point
depth), already safe on the interpreter lane thanks to D51's `YSaves`
(proved causally by disabling `popYSave`'s restore). ITE barrier levels
now live on `ChoicePoint.Levels`, never in `vm.Regs`. Probe
`tests/test_wam_go_frameless_ite_level.pl`. A separate, pre-existing
lowered-lane defect (a lowered method that `call`s the interpreter and
then tail-recurses loses its output binding) is recorded as a residual
above — it is not the barrier and is not fixed here.
