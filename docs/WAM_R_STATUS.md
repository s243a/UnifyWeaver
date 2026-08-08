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
- **ISO error support** meets the seven-item adoption unit after ISO-R-0/
  2A/2B: shared config/rewrite/audit, `is/2`, six arithmetic comparisons,
  and `succ/2` three-form (`succ_iso`/`succ_lax`), plus pre-existing
  catch/throw. Same caveat as Python/F#: remaining concrete builtins
  without three-form keys are out of scope for “complete for every
  builtin.” Do not describe R as fully ISO-compatible across the entire
  builtin surface.
- Effective-distance cross-target matrix row populated (BENCH-R): scale-300
  `emit_mode(functions)` + auto `category_ancestor/4` kernel, reference
  parity match — see `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`.
  Further optimized by PERF-R-CA-DIRECT / IDDFS / IDCACHE / STACK / IDTABLE /
  WAM-STEP / AGGREGATE-LOWER / AGG-BATCH / NATIVE-HOPS-0. Hosted-CI
  NATIVE-HOPS-0 median query_ms=270 / total_ms=776 (prior AGG-BATCH
  1111/1621; AGGREGATE-LOWER 1564/2077; IDTABLE 3412/4076).
  PERF-R-CA-LOOPBODY, PATHMARK, ED-BOUNDARY, LOWERED-DIRECTCALL, and
  CA-BULK-HOPS profiled/declined where measured. PERF-R-WAM-STEP
  short-circuits `is_lax` `**/^` and unary `-`. PERF-R-AGGREGATE-LOWER
  capability-gates scalar `sum|count|min|max` regions over `bulk_collect`
  generators (~1.43–1.44× same-host query; steps 137817→14540).
  PERF-R-AGG-BATCH adds typed numeric batches + vectorized closed `is_lax`
  reduce (~1.18× same-host query over AGGREGATE-LOWER; steps unchanged at
  14540). PERF-R-CA-BULK-HOPS found post-AGG-BATCH query time ~84% inside
  `category_ancestor_hops_ids`; R-level membership/buffer trials did not
  clear the ≥1.05× gate (best ~1.016×). PERF-R-NATIVE-HOPS-0 adds an
  optional base-R `.Call` hop kernel (`R CMD SHLIB`, pure-R fallback;
  ~1.95× same-host query over AGG-BATCH; steps unchanged at 14540).
  PERF-R-POST-NATIVE-0 profiled the post-native residual: hops ~3–5% of
  warm query; `run`/`step` ~84–90%, aggregate fast path ~32%, `put_reg`
  ~23%. These are nested/inclusive measurements and are not additive.
  Plan-cache + closed `(Batch ⊕ scalar) ** Exp` trials did not
  clear ≥1.05× (best ~1.00–1.03×); no production change retained.
  PERF-R-BULK-REDUCE-REGION-0: shape-gated fused bulk-reduce prototype
  ≈2.04×; thin step-driver ≈0.97×; direct-fusion five-repetition smoke
  ≈1.20× but ≈320 LOC (>180 soft-stop) — reverted, docs-only. The
  official two-sequence gate was not run after the soft-stop fired.

## Path forward

1. Optional follow-up: three-form keys for additional audited builtins
   beyond `is`/compares/`succ` (same class of work as ISO-PYTHON/ISO-FSHARP).
2. Next ED performance lever: retain shape-gated direct bulk-reduce
   fusion under LOC discipline (generate-time plan packing or raised
   budget) — thin step-drivers do not clear ≥1.05×.

## Document status

Fleet-aligned snapshot updated for PERF-R-BULK-REDUCE-REGION-0
(2026-08-05): hosted NATIVE-HOPS-0 row unchanged at 270/776; fused
bulk-reduce declined on LOC soft-stop despite a ≥1.05× smoke result.
