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
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural)** | `X→+100 / Y→+200` encoding (`wam_lua_target.pl:136-137`) means X101 ≡ Y1; registers are one flat table (`runtime.lua.mustache:158-159`); `Allocate` frames store only `cp` (`:1023`) and `Call` takes no Y snapshot (`:1127-1133`). A ground fact needing >99 X placeholders overwrites the caller's Y registers — exactly what `default_registry/1` did to the JS runtime before its Call-snapshot fix |
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

## Path forward

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
