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
(2026-07-11).
