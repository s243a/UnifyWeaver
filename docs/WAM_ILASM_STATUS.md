# WAM ILAsm Target — Status

Living summary of the hybrid WAM-ILAsm backend (`wam_ilasm_target.pl`),
which emits .NET CIL from WAM. For a **mature** .NET route prefer
**F#** ([`WAM_FSHARP_STATUS.md`](WAM_FSHARP_STATUS.md)); ILAsm is an
early scaffold for when raw CIL is specifically wanted.

Companion docs:

- [`WAM_FSHARP_STATUS.md`](WAM_FSHARP_STATUS.md) — the mature .NET target.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Raw CIL scaffold.** Emits .NET CIL directly from WAM with a `switch`
dispatch shape — an early scaffold, not a production .NET path.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_ilasm_target.pl` | ~1.9k |
| Lowered emitter | none |
| Dedicated tests | ~2 files (~46 plunit cases) |

## What's shipped

- **CIL emit** from WAM with `switch`-based dispatch.
- **~46 plunit cases** across 2 files — denser than the file count
  suggests.

## Gaps

- **No lowered emitter.**
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No conformance registration** — no `conformance_target(ilasm)`.
- **No runtime-parser capability entry.**

## Path forward

1. Decide whether ILAsm stays a raw-CIL scaffold or grows a lowered
   emitter — F# already covers the mature .NET route.
2. Conformance adapter if it graduates from scaffold status.

## Document status

Fleet-aligned snapshot; source-verified line count, absence of a
lowered emitter, ~46-case density, and absence of conformance
registration against `wam_ilasm_target.pl` and the conformance harness
(2026-07-11).
