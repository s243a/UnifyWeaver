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
| `src/unifyweaver/targets/wam_go_target.pl` | ~3.7k |
| `src/unifyweaver/targets/wam_go_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~13 files (~38 plunit cases) |

## What's shipped

**All 7 FFI kernels.** Full shared-detector set via
`go_foreign_lowering` / FFI dispatch. (`go_supported_shared_kernel/1`
lists 5; weighted/A* are separate arms — the effective set is all 7.)

**Fact sources.** TSV and LMDB atom-fact paths.

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
- **No two-level lazy/cached LMDB policies** (F#/Haskell tier).
- **No ISO three-form contract adoption** (low ISO surface in source).
- **No runtime-parser capability entry** in
  `wam_runtime_parser_capability.pl`.

## Path forward

1. Decide whether the WAM route should become a first-class Go product
   path or stay the kernel-benchmarking arm.
2. Richer LMDB policy tiers to match F#/Haskell.
3. ISO three-form adoption if Go joins the error-fidelity set.
4. Effective-distance cross-target matrix row.

## Document status

Fleet-aligned snapshot; source-verified against `wam_go_target.pl`,
the parity audit, and the conformance harness (`prefer_wam(true)`
requirement confirmed) on 2026-07-11. Update the parity audit first,
then refresh here.
