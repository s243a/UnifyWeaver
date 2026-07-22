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

**Optional LMDB.** Legacy `lmdb(Path)` remains load-everything.
Versioned `lmdb_arg1_v1(Path)` supports `lmdb_materialisation(lazy)`
(default when omitted), `eager`, `cached` (+ `lmdb_l2_capacity`,
default 4096), and explicit `auto` (codegen resolves via shared
`resolve_auto_lmdb_materialisation/2`). LMDB-R-0/R-1/R-2A/R-2B complete.

## Gaps (relative to Rust / Haskell / F#)

- **Classic conformance (CONF-R):** opt-in adapter (`r` / `r_functions`).
  All classic programs green. Indexing hints — including remaining A2
  forms (`switch_on_constant_a2`, `_a2_fallthrough`, `switch_on_structure_a2`)
  — map to the existing `SwitchOnTerm()` linear no-op (not optimized A2
  dispatch), preserving the full try/retry/trust chain.
- **ISO error support** is a **partial adopter** after ISO-R-0: shared
  config/rewrite/audit + `is/2` → `is_iso`/`is_lax` vertical slice.
  The pre-existing catch/throw substrate is already covered end-to-end;
  comparisons/`succ` remain ISO-R-2A/R-2B. Do not claim complete ISO-R yet.
- No dedicated effective-distance cross-target bench row.

## Path forward

1. ISO-R-2A arithmetic comparisons; ISO-R-2B successor + adoption closeout.
2. Effective-distance cross-target matrix row.

## Document status

Fleet-aligned snapshot updated for CONF-R landing (2026-07-15): opt-in
harness adapter + measured classic-program readout.
