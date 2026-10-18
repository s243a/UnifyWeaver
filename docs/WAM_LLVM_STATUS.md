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

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. LLVM's audit (light pass —
suspected entries were not source-confirmed here):

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | fleet grep: `sub_string/5` is dispatched only by C++ and (post-A2) JS |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | aliasing form: **absent** (disjoint X/Y ABI). Frameless-Y form: **EXPOSED AND UNFIXED on the bytecode lane, immune on the lowered lane** (2026-09-03) | `bindings/llvm_wam_bindings.pl reg_name_to_index` gives A/X/Y disjoint ranges (Y1..Y16 → regs[48..63]), so the numeric X→Y alias other targets have cannot occur. But registers are one flat `%WamState` array and only `allocate`/`deallocate` save/restore the Y window — see the frameless-Y section below |
| A3 | `Execute` of a builtin doesn't return to the continuation | **suspected** | the shared emitter's `deallocate` + `execute <builtin>` shape reaches this backend like every other; the `execute` lowering was not audited |
| A4 | String fidelity | **rung 0** | no string term tag in `value.ll`; D37's double-quoted literals intern as atoms |

With no conformance registration either (see Gaps above), emitted LLVM
output currently has *no* executed acceptance gate — resolve the
suspected rows as the first step of any conformance-adapter work.

## A2 frameless-Y: the if-then-else barrier (2026-09-03)

**Verdict: EXPOSED AND UNFIXED on the bytecode lane; structurally immune
on the lowered lane.** Static audit only — this round owned wam_go; llvm
got a verdict, not a fix. Pinned by
`tests/test_wam_llvm_frameless_ite_level.pl` (emission-level; runs with
no `llc`/`clang` present).

The trigger is not a wide ground fact — it is any if-then-else or `\+` in
a clause with no other permanent variable, e.g.
`sat(V, gte(G)) :- \+ lt(V, G)` (the resolver's `satisfies/2` shape).
All three preconditions hold:

1. **The flag is on by default.** `write_wam_llvm_project/3`
   (`wam_llvm_target.pl:1858-1861`) adds `ite_use_y_level(true)` unless
   the caller supplied one, so `compile_if_then_else/7` reserves the
   barrier's Y register *after* deciding the clause needs no environment
   and emits `get_level Yn` … `cut Yn` with no `allocate`.
2. **The emitter accepts it — no loud refusal.**
   `wam_line_to_llvm_literal(["get_level", 'Y1'], _)` yields
   `%Instruction { i32 33, i64 48, i64 0 }` (and `cut` yields opcode 34);
   a default project build for the shape above really does contain
   `i32 33, i64 48` in its code array. Operand 48 is Y1 under the
   disjoint X/Y ABI (Y1..Y16 → regs[48..63]).
3. **The opcode writes that register.** `wam_llvm_case('get_level', …)`
   (`:3861-3876`) ends in
   `call void @wam_set_reg(%WamState* %vm, i32 %gl.yn, %Value %gl.v)`,
   and `@wam_set_reg` (`templates/targets/llvm_wam/state.ll.mustache:199`)
   is a GEP straight into the ONE flat `%WamState` register array
   (field 1) — no environment-frame indirection.

The only Y protection this backend has is `wam_llvm_case('allocate')`,
which memcpys regs[48..63] (256 bytes) into the env frame for
`deallocate` to restore — and an `allocate`-less clause never runs it.
`do_call` (`:3454-3487`) saves the continuation and the cut barrier and
**no registers at all**; `proceed` restores nothing. There is no analogue
of wam_javascript's or wam_go's Call-time Y snapshot, so on the success
path the caller's Y1 keeps the choice-point count the callee wrote.

One partial mitigation worth naming precisely, so nobody over-reads it:
`get_level` calls `@wam_trail_binding(%vm, %gl.yn)` before the write, so
the caller's old value is on the trail. That only helps a backtrack past
that trail mark. The wrong answer in this shape is delivered on the
*success* path — `\+ lt(3,1)` succeeds and returns to its caller
normally — and the ITE choice point's trail mark is *above* the
`get_level` entry, so failing the condition does not unwind it either.
Trailing narrows the window; it does not close it.

**The lowered lane is immune, structurally.**
`wam_llvm_lowered_emitter.pl` declares `supported(get_level(_))` and
renders it as an explicit no-op (`emit_instr/4` emits a comment and a
branch, `:936-945`), because the soft cut there is realised by the
basic-block layout rather than by a choice-point level. Nothing is
written to a register at all. That is a real difference from wam_go,
whose lowered lane was the broken one.

**The fix, when someone takes it**: the wam_rust `ChoicePoint::levels` /
wam_python `ChoicePoint.levels` / wam_go `ChoicePoint.Levels` model —
keep the barrier level on the if-then-else's own choice point and never
write it into a register. See the A2 frameless-Y section of
[`WAM_GO_STATUS.md`](WAM_GO_STATUS.md) for a worked version, including
how the two emission shapes (`get_level` before vs after the ITE
`try_me_else`) are told apart.

## Path forward

0. **Close the A2 frameless-Y hole on the bytecode lane** (above) — a
   silent wrong answer on a program shape as ordinary as `\+ G` in a
   one-goal clause, and with no conformance arm nothing would catch it.
1. Register a conformance adapter (`CONFORMANCE_TARGETS=llvm`).
2. Effective-distance-class cross-target matrix parity.
3. LMDB / LookupSource fact-source.
4. Trail-rollback for hybrid clause-1 partial bindings.
5. Optional runtime-parser capability entry if compiled/native parsing
   is needed for term IO.

## Document status

Derived from the transpilation trilogy, roadmap Table 1, perf notes,
and the hybrid comparison branch. Update when arena, kernel, or
conformance milestones land. 2026-09-01: added the whole-program (A2)
deficiency audit (light pass); see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
2026-09-03: **A2 frameless-Y verdict — EXPOSED, UNFIXED on the bytecode
lane; immune on the lowered lane** (get_level is an explicit no-op
there). The A2 aliasing cell is also resolved from *suspected* to
*absent*: `reg_name_to_index` gives A/X/Y disjoint ranges. Verdict only:
the round that found it owned wam_go. Probe
`tests/test_wam_llvm_frameless_ite_level.pl`.
