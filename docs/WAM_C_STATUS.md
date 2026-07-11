# WAM C Target — Status

Living summary of the hybrid WAM-C backend
(`wam_c_target.pl` + `wam_c_runtime/`). Distinct from the **non-WAM**
direct C compiler (`c_target.pl`). The running checklist that has
historically functioned as this target's status is
[`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md); this doc
is the fleet-aligned snapshot.

Companion docs:

- [`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md) — living checklist.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Portable C ABI / FFI-glue substrate.** Small-footprint native
runtime that doubles as shared C glue other systems targets (Rust/Go
FFI kernels) can lean on. Historically undercounted as "907 lines" —
the current codegen is far larger.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_c_target.pl` | ~6.6k |
| `src/unifyweaver/targets/wam_c_runtime/` (header + runtime) | ~1k header |
| Lowered emitter | none as a separate module — lowered **helpers** prototype inside the target |
| Dedicated tests | ~9 files |

## What's shipped

**All 7 shared kernels + `bidirectional_ancestor`.** Full shared
detector set plus the bidirectional ancestor kernel, with reverse-CSR
child-index paths for reverse traversal.

**Fact sources.** TSV plus an LMDB FactSource — `lmdb` is a heavy
theme in the target (mmap-backed fact storage, not TSV-only).

**Meta / aggregates.** Aggregates and `bagof`/`setof` meta-goals.

**Lowered helpers prototype.** No standalone
`wam_c_lowered_emitter.pl`; deterministic helper emission lives in the
target module.

**Conformance.** Registered as `conformance_target(c)` and passes the
whole spec with no `ct_xfail`/`ct_skip`; stays opt-in (needs a `gcc`
per-program build) rather than default CI.

## Gaps (relative to Rust / Haskell / F#)

- **No ISO three-form contract adoption** — not a reference adopter
  alongside C++/Elixir (see
  [`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md)).
- **No runtime-parser capability entry** in
  `wam_runtime_parser_capability.pl` — no source-term parsing path.
- **No two-level lazy/cached LMDB policies** (F#/Haskell tier); the
  FactSource is present but the policy surface is thinner.
- Effective-distance cross-target matrix presence is thinner than the
  Tier-A kernel benches.

## Path forward

1. Adopt the ISO three-form error contract (C++ is the reference).
2. Add a runtime-parser capability entry if compiled/native term IO is
   wanted.
3. Promote the lowered-helpers prototype into a first-class lowered
   emitter if C moves off "substrate" duty.
4. Richer LMDB lazy/cached policy tiers.
5. Publish a dedicated effective-distance bench row.

## Document status

Fleet-aligned snapshot; source-verified line/kernel/LMDB/conformance
facts against `wam_c_target.pl`, the conformance harness, and the
parser-capability module (2026-07-11). Update the living checklist
first when milestones land, then refresh here.
