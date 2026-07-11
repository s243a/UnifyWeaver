# WAM R Target — Status

Living summary of the hybrid WAM-R backend
(`wam_r_target.pl` + `wam_r_lowered_emitter.pl`). Distinct from the
**non-WAM** direct R compiler (`r_target.pl`). Usage guide:
[`WAM_R_TARGET.md`](WAM_R_TARGET.md).

Companion docs:

- [`WAM_R_TARGET.md`](WAM_R_TARGET.md) — usage guide.
- [`handoff/wam_r_session_handoff.md`](handoff/wam_r_session_handoff.md) — parity-campaign handoff.
- [`WAM_RUNTIME_PARSER_STATUS.md`](WAM_RUNTIME_PARSER_STATUS.md) — R native parser default.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Statistical-scripting embed.** Product of a ~30-PR builtin/parity
campaign; a full-kernel WAM backend that lands inside an R runtime with
a native source-term parser as the default.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_r_target.pl` | ~2.0k |
| `src/unifyweaver/targets/wam_r_lowered_emitter.pl` | ~1.0k |
| Dedicated tests | ~7 files (~100 plunit cases; generator suite denser) |

## What's shipped

**All 7 shared kernels.** Full graph-kernel set from the parity
campaign.

**Rich builtins.** Broad builtin/control/aggregate coverage per the
session handoff.

**Native parser default.** `target_runtime_parser_default(wam_r,
native(parse_term))` — R is one of only two backends (with C++) whose
runtime-parser default is native, with the compiled
`prolog_term_parser` available as an opt-in fallback.

**Optional LMDB.** Load-everything LMDB fact path (not the
lazy/cached tiers of F#/Haskell).

## Gaps (relative to Rust / Haskell / F#)

- **Not in the conformance harness** — no `conformance_target(r)`;
  correctness rests on the generator suite, not the shared spec.
- **LMDB is load-everything** — no lazy/cached two-level policies.
- **ISO error support** is `tryCatch`-level only, not the three-form
  contract.
- No dedicated effective-distance cross-target bench row.

## Path forward

1. Register a conformance adapter (`CONFORMANCE_TARGETS=r`) so R joins
   the shared spec.
2. Lazy/cached LMDB policies over the load-everything path.
3. ISO three-form adoption if R joins the error-fidelity set.
4. Effective-distance cross-target matrix row.

## Document status

Fleet-aligned snapshot; source-verified line/kernel counts, the native
runtime-parser default, and the **absence** of conformance registration
against `wam_r_target.pl`, the parser-capability module, and the
conformance harness (2026-07-11).
