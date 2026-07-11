# WAM C++ Target — Status

Living summary of the hybrid WAM-C++ backend
(`wam_cpp_target.pl` + `wam_cpp_lowered_emitter.pl` + C++ runtime).

Companion docs:

- [`design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md`](design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md)
- [`design/WAM_CPP_ISO_ERRORS_PHILOSOPHY.md`](design/WAM_CPP_ISO_ERRORS_PHILOSOPHY.md)
- [`design/WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md`](design/WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md)
- [`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md)
- [`WAM_RUNTIME_PARSER_STATUS.md`](WAM_RUNTIME_PARSER_STATUS.md) —
  native default + compiled cost notes.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Systems substrate + ISO-error reference.** First concrete consumer
of the shared ISO three-form contract; mutable C++ runtime that
passed classic conformance on onboarding without backend fixes.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_cpp_target.pl` | ~11.0k |
| `src/unifyweaver/targets/wam_cpp_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~6 focused files (+ large generator suite) |

## What's shipped

**Dual lowering.** WAM instruction VM + lowered emitter for
deterministic / clause-1 / ITE shapes (mirrors Rust/Haskell design).

**ISO errors (reference adopter).** Config loader, rewrite, audit
(`wam_cpp_iso_audit/3`), `catch/3`+`throw/1`, error constructors,
`is_iso`/`is_lax`, ISO/lax arithmetic compares, `succ_iso`/`succ_lax`,
lax IEEE-754 float divide. Shared contract documented in
`WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` (Elixir is the second
reference; F#/Python are partial adopters).

**Materialisation.** Arity-2 LMDB FactSource (v1) per
`WAM_CPP_LMDB_FACT_SOURCE_DESIGN.md`.

**Runtime parser.** Default `native(parse_term)`; also supports
`compiled(prolog_term_parser)`. Audit-clean quoted-atom handling.
Compiled-parser projects balloon `generated_program.cpp` (~42k lines)
and can take ~11 min to compile — `wam_runtime.o` caching mitigates
the shared runtime portion (`WAM_RUNTIME_PARSER_STATUS.md`).

**Classic conformance.** Registered (`CONFORMANCE_TARGETS=cpp`);
**conformant on first onboarding** — cons-cell / placeholder / `is/2`
conventions already present. CLI via `emit_main(true)` → `main.cpp`
exit 0/1.

**Foreign surface.** `call_foreign` trampolines and lowered foreign
comments; **no** `recursive_kernel_detection` / shared graph-kernel
set (unlike C, which ships all 7 + bidirectional). Graph-kernel
density and effective-distance matrix presence lag Rust/Haskell/C.

**Test surface.** Few dedicated kernel e2e *files*, but
`tests/test_wam_cpp_generator.pl` alone carries on the order of
**~400** `test/1` clauses spanning ISO + LMDB + e2e — do not read
“~6 files” as a thin suite.

## Gaps

- Graph-kernel density lags Rust/Haskell/C (no shared-kernel detector).
- Compiled runtime-parser path is a compile-time cost hazard (~11 min
  for parser-bundled projects).
- ISO implementation lives in C++ rather than solely in the shared
  `iso_errors.pl` Prolog module used by F#/Python/Elixir.

## Path forward

1. Keep ISO contract as the shared reference; sync Elixir/F#/Python
   status tables when variants change.
2. Grow shared-kernel parity if C++ is used for graph benches.
3. Treat compiled-parser compile cost as a first-class constraint
   (chunked setup, stronger `.o` caching, or native-only default for
   large projects).
4. Optional: more kernel smoke tests mirroring C’s seven-kind set.

## Related: WAM-C

The sibling **C** target (`wam_c_target.pl` ~6.1k lines, no separate
lowered-emitter module) has a living checklist in
[`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md) —
**all 7 shared kernels + `bidirectional_ancestor`**, reverse-CSR
child-index paths, LMDB FactSource, aggregates/bagof/setof, lowered
**helpers** prototype (fact/filter shapes, not a full WAM lowered
emitter). Conformance registered. Do not confuse C and C++ maturity
axes: C leads on kernels/CSR; C++ leads on ISO + runtime parser.

## Document status

Snapshot for the hybrid comparison branch. Update when ISO, LMDB,
kernel, or parser-compile milestones land.
