# WAM WAT Target — Status

Living summary of the hybrid WAM-WAT (WebAssembly text) backend
(`wam_wat_target.pl` + `wam_wat_lowered_emitter.pl`). Emits a fused
bytecode VM as WebAssembly text; there is no non-WAM WAT sibling —
WASM is inherently the compiled deploy shape.

Companion docs:

- [`targets/wam-wat.md`](targets/wam-wat.md) — overview + architecture.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Browser / sandboxed WASM deploy.** A fused WAM bytecode VM that runs
in a browser or any WASM sandbox without a native toolchain. Browser
deployment is the differentiator.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_wat_target.pl` | ~6.7k |
| `src/unifyweaver/targets/wam_wat_lowered_emitter.pl` | ~0.5k |
| Dedicated tests | ~6 files (~76 plunit cases) |

## What's shipped

**Fused bytecode VM.** Substantial WAM-instruction lowering pipeline.

**Hybrid lowered fast paths.** T4–T6 deterministic clause-1 lowered
paths plus a `$run_loop` interpreter fallback: opt-in hybrid mode
mirrors Rust/Scala-style public-entry replay — lowered success returns
immediately; lowered failure reinitialises and falls back to
`$run_loop`.

**Conformance.** Registered `conformance_target(wat)` and green;
opt-in (needs `wat2wasm` + `node` per program), not default CI.

## Gaps (relative to Rust / Haskell / F#)

- **No foreign graph kernels** — dispatch stays interpreter-bound; the
  differentiator is deploy shape, not kernel throughput.
- **No LMDB / fact source** — no mmap materialisation story.
- **No runtime-parser capability entry.**
- Interpreter-bound dispatch is the standing perf ceiling.

## Path forward

1. Explore foreign/host-imported graph kernels via WASM imports if
   perf-class graph work is wanted.
2. A host-side fact-source bridge (IndexedDB / host LMDB) for larger
   fact sets.
3. Widen lowered T4–T6 coverage to shrink the `$run_loop` fallback
   surface.

## Document status

Fleet-aligned snapshot; source-verified line counts, the T4–T6 +
`$run_loop` hybrid path, absence of kernels/LMDB, and opt-in
conformance registration against `wam_wat_target.pl` and the
conformance harness (2026-07-11).
