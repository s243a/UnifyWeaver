# WAM Go Target — Status

Living summary of the hybrid WAM-Go backend
(`wam_go_target.pl` + `wam_go_lowered_emitter.pl`). Distinct from the
**non-WAM** direct Go stream/dataflow compiler documented in
[`GO_TARGET.md`](GO_TARGET.md) (`go_target.pl`), which remains Go's
**default** product path.

Companion docs:

- [`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md) — parity vs Rust/Haskell.
- [`GO_TARGET.md`](GO_TARGET.md) — non-WAM sibling compiler.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Strong Go runtime backend.** Broad builtin/IO/aggregate surface with
a full FFI graph-kernel set, reachable when a workload opts into the
WAM pipeline via `prefer_wam(true)`.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_go_target.pl` | ~4.3k |
| `src/unifyweaver/targets/wam_go_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~17 files |

## What's shipped

**All 7 FFI kernels.** Full shared-detector set via
`go_foreign_lowering` / FFI dispatch. (`go_supported_shared_kernel/1`
lists 5; weighted/A* are separate arms — the effective set is all 7.)

**Fact sources.** TSV and LMDB atom-fact paths. The LMDB source
carries the full `lmdb_materialisation(eager|lazy|cached|auto)` +
`lmdb_l2_capacity(N|auto)` tier set (LMDB-GO), matching F#: eager
materialises once at construction, lazy spawns the helper per lookup,
cached memoises through an L1/L2 pair. `auto` defers to the shared cost
model in `core/cost_model.pl`. Default is `cached`.

**ISO three-form errors.** Full adoption (ISO-GO): `catch/3` + `throw/1`
via panic/recover, `error(Formal, Context)` constructors, `is_iso`/
`is_lax`, ISO/lax forms of the six arithmetic comparisons, `succ_iso`/
`succ_lax`, per-predicate key rewrite, and `wam_go_iso_audit/3`.
Rewriting is opt-in: without an `iso_errors` option the WAM text keeps
its plain keys. `catch/3` is deterministic — it commits to the goal's
first solution.

**Effective-distance benchmark.** Scale-300 row populated (BENCH-GO):
query_ms 898 / total_ms 898, 5-rep median, output an exact multiset
match against the reference. See
[`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md).

**Compiled runtime parser.** `runtime_parser(compiled)` compiles the
portable `prolog_term_parser` and the target-agnostic wrappers into the
project, so a generated program can call `read_term_from_atom/2`
directly (PARSE-GO). Default stays `none`. Proven end-to-end by
`tests/test_wam_go_parser_smoke.pl` — 22 inputs including operator
precedence and associativity.

**Dual lowering.** WAM instruction VM plus the lowered emitter.

**Broad surface.** Builtins, IO, and aggregate coverage tracked in the
parity audit against Rust/Haskell.

**Conformance.** Registered `conformance_target(go)` and green — but
**requires `prefer_wam(true)`**, because the default Go strategy is the
non-WAM dataflow/stream compiler, not the WAM pipeline. Stays opt-in
(needs a `go` per-program build), not default CI.

## Gaps (relative to Rust / Haskell / F#)

- **Default product path is non-WAM** (`go_target.pl`); the WAM route
  is opt-in.
- **Effective-distance row is interpreter-bound** (~898 ms query vs
  Rust's 17 ms): the benchmark runs `category_ancestor/4` through the
  shared bytecode loop rather than the FFI kernel path.

## Path forward

1. Decide whether the WAM route should become a first-class Go product
   path or stay the kernel-benchmarking arm.
2. Route the effective-distance benchmark through the registered
   foreign kernels — the single biggest lever on the Go perf row.
3. Consider registering `conformance_target(go)` as default CI rather
   than opt-in, now that the runtime has been hardened by the parser
   bring-up.

## Document status

Fleet-aligned snapshot; source-verified against `wam_go_target.pl`,
the parity audit, and the conformance harness (`prefer_wam(true)`
requirement confirmed) on 2026-07-11. Refreshed 2026-08-06 after all four Go
gap cards (LMDB-GO, ISO-GO, BENCH-GO, PARSE-GO) landed. Update the parity audit first,
then refresh here.
