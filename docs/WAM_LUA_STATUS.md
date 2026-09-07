# WAM Lua Target — Status

Living summary of the hybrid WAM-Lua backend
(`wam_lua_target.pl` + `wam_lua_lowered_emitter.pl`). Distinct from the
**non-WAM** direct Lua compiler (`lua_target.pl`).

Companion docs:

- [`design/WAM_LUA_PARITY_AUDIT.md`](design/WAM_LUA_PARITY_AUDIT.md) — 2026 builtin parity pass.
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Lightweight embed for builtin/control parity.** A small WAM backend
focused on a 2026 builtin/control/aggregate parity pass, for embedding
in Lua hosts.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_lua_target.pl` | ~0.8k |
| `src/unifyweaver/targets/wam_lua_lowered_emitter.pl` | ~0.6k |
| Dedicated tests | ~5 files (~44 plunit cases) |

## What's shipped

**Builtin / control / aggregate parity.** Focused 2026 parity pass per
the parity audit.

**Lowered T4–T6.** Dual WAM-instr + lowered emitter covering T4–T6
shapes.

**Narrow IO surface.**

## Gaps (relative to Rust / Haskell / F#)

- **No graph kernels** (zero kernel surface).
- **No LMDB / fact source** (zero LMDB surface).
- **No conformance registration** — no `conformance_target(lua)`.
- **No runtime-parser capability entry.**
- **No ISO three-form contract adoption.**

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that apply fleet-wide. Lua is the JS runtime's model sibling
(same term tags, same register encoding), so it inherits the findings
almost verbatim:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | not in the dispatcher; `sub_atom/5` is missing too — the builtin surface is ~10 predicates (`runtime.lua.mustache:910-927`) |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural) — and the frameless-Y form is EXPOSED AND UNFIXED (2026-09-03)** | `X→+100 / Y→+200` encoding (`wam_lua_target.pl:136-137`) means X101 ≡ Y1; registers are one flat table (`runtime.lua.mustache:158-159`); `Allocate` frames store only `cp` (`:1023`) and `Call` takes no Y snapshot (`:1127-1133`). A ground fact needing >99 X placeholders overwrites the caller's Y registers — exactly what `default_registry/1` did to the JS runtime before its Call-snapshot fix. See the frameless-Y section below for the *other*, much cheaper trigger |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified** | `Execute`/`Call` have *no* builtin fallback at all: an unresolved label is a silent goal failure (`runtime.lua.mustache:1141-1148`, `:1127-1133`), so a last-goal builtin outside `is_builtin_pred/2` fails instead of running |
| A4 | String fidelity | **rung 0** | no string term tag; compiled `"foo"` literals and string builtins intern as atoms (D37's double-quoted WAM tokens degrade gracefully to atoms here) |

Compounding A3/B-H2: `wam_lua` has **no conformance arm** (CONF-LUA
still open in [`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md)), so
its emitted output is never executed by the shared harness — the fleet's
weakest position against silent-wrong-code regressions.

Pattern lane: `lua_target.pl` compiles facts from `clause(Head, true)`
(no compile-time execution — the G-A3-8 hazard is absent), but none of
the G-A3 machinery (cross-predicate calls, multi-output loops, semidet
sentinel, compound-term data) has an analogue there; presume those gaps
present until the `examples/cli_args/` benchmark is attempted.

## A2 frameless-Y: the if-then-else barrier (2026-09-03)

**Verdict: EXPOSED, UNFIXED.** Static audit only — this round owned
wam_go; lua got a verdict, not a fix. Pinned by
`tests/test_wam_lua_frameless_ite_level.pl` (emission-level; runs with no
Lua interpreter present).

The A2 row above scores the *aliasing* form, which needs a fact with >99
X placeholders. The frameless-Y form needs no such thing — just an
if-then-else or a `\+` in a clause with no other permanent variable — and
all three of its preconditions hold on wam_lua:

1. **The flag is always on.** `compile_lua_predicate_wam/3` (both arms,
   `wam_lua_target.pl:541,543`) compiles with `[ite_use_y_level(true)]`,
   so `compile_if_then_else/7` reserves the barrier's Y register *after*
   deciding the clause needs no environment and emits `get_level Yn` …
   `cut Yn` with no `allocate`.
2. **The emitter accepts it — no loud refusal.** `wam_parts_to_lua`
   (`:282-287`) renders `get_level Yn` / `cut Yn` as `I.GetLevel(R)` /
   `I.Cut(R)` with `R = n + 200` (`reg_to_int/2`, `:130-139`). Generated
   output for `sat(V, gte(G)) :- \+ lt(V, G)`:

   ```lua
   I.TrustMe(),
   I.GetVariable(101, 1),
   I.GetStructure(7, 2, 1),
   I.UnifyVariable(102),
   I.GetLevel(201),          -- the barrier, with NO I.Allocate()
   I.TryMeElse("L_ite_else_1"),
   ...
   I.Cut(201),
   ...
   I.Allocate(),             -- the framed caller that follows
   I.GetVariable(201, 3),    -- ... parks ITS first permanent in slot 201
   ```
3. **The runtime routes it into shared, live storage.**
   `runtime.lua.mustache:1110-1113`:
   `Runtime.put_reg(state, instr.yn, #state.cps)`, and
   `Runtime.put_reg(state, idx, val)` is `state.regs[idx] = val`
   (`:158-159`) — one flat, VM-global table.

And nothing repairs it. `Allocate` pushes only
`{ cp = state.cp, locals = {} }` (`:1023`) and `Deallocate` restores only
`cp`; the `Call` arm (`:1127-1133`) sets `state.cp` and nothing else.
There is no analogue of wam_javascript's or wam_go's Call-time Y
snapshot, so the caller's Y1 is overwritten with a choice-point depth and
**never** restored on the success path.

wam_lua is therefore worse placed than wam_go was before its fix: Go's
interpreter lane was covered by `vm.YSaves` and only the lowered lane
broke; Lua has no cover on any lane. It is also the hardest target to
prove it on, because **wam_lua has no conformance arm at all**
(`WAM_FLEET_GAPS.md` B-H2) — there is no execution oracle to diff
against without first building one. The probe is consequently
emission-level: it asserts the three preconditions from the generated
Lua and from the shipped runtime template, and its three
`..._is_still_exposed` tests deliberately assert the defect, so they go
RED the moment someone fixes it.

**The fix, when someone takes it**: the wam_rust `ChoicePoint::levels` /
wam_python `ChoicePoint.levels` / wam_go `ChoicePoint.Levels` model —
keep the barrier level on the if-then-else's own choice point and never
write it into a register. See the A2 frameless-Y section of
[`WAM_GO_STATUS.md`](WAM_GO_STATUS.md) for a worked Go version,
including how the two emission shapes (`get_level` before vs after the
ITE `try_me_else`) are told apart.

## Path forward

0. **Close the A2 frameless-Y hole** (above) — it is a silent wrong
   answer on a program shape as ordinary as `\+ G` in a one-goal clause,
   and it outranks everything below.
1. Register a conformance adapter once the builtin surface stabilises.
2. Add a fact-source path (TSV then optional LMDB) if fact-backed
   workloads are wanted.
3. Foreign/graph kernels if Lua moves beyond embed-parity duty.
4. ISO three-form adoption if Lua joins the error-fidelity set.

## Document status

Fleet-aligned snapshot; source-verified line counts, the lowered
T4–T6 coverage, and the absence of kernels/LMDB/conformance against
`wam_lua_target.pl`, the parity audit, and the conformance harness
(2026-07-11). 2026-09-01: added the whole-program (A2) deficiency
audit — Y-clobber, Execute-of-builtin and `sub_string/5` verified by
source reading; see [`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
2026-09-03: **A2 frameless-Y verdict — EXPOSED, UNFIXED**, verified from
the generated Lua and the shipped runtime template (all three
preconditions hold and there is no Y save anywhere). Verdict only: the
round that found it owned wam_go. Probe
`tests/test_wam_lua_frameless_ite_level.pl`.
