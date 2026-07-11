# WAM Scala Target — Status

Living summary of the hybrid WAM-Scala backend
(`wam_scala_target.pl` + `wam_scala_lowered_emitter.pl`). Distinct
from the **non-WAM** direct Scala compiler (`scala_target.pl`). Usage
guide: [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md).

Companion docs:

- [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md) — usage guide.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Generalization anchor + default conformance backend.** Carries the
classic-program suite (n-queens, Ackermann, Fibonacci, …) that sets
the transpiler's generalisation upper bound, and runs as a **default
CI** conformance target alongside Elixir.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_scala_target.pl` | ~1.4k |
| `src/unifyweaver/targets/wam_scala_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~11 files (~137 plunit cases) |

## What's shipped

**Dual lowering.** WAM instruction VM plus per-predicate emitter
(`emit_mode(functions)`) — clause-1 fast path with interpreter
fallback.

**All 7 shared kernels.** Opt-in via `kernel_dispatch(true)`. An
intra-Scala mode bench (`benchmarks/wam_scala_mode_bench.md`, cited —
not re-run here) shows kernel dispatch ~4×@depth100, ~9×@depth300;
shallow queries can regress.

**Four fact backends.** Inline / file CSV / grouped TSV / **arity-N
LMDB** (validated end-to-end); auto-inline ≤128 rows.

**Atom interning.** Aggressive compile-time atom interning.

**Conformance.** `ct_default_target(scala)` — one of only two
backends (with Elixir) that run conformance by default, not opt-in.

## Gaps (relative to Rust / Haskell / F#)

- **No ISO three-form contract adoption** (low ISO surface in source).
- **No runtime-parser capability entry** in
  `wam_runtime_parser_capability.pl`.
- **No two-level lazy/cached LMDB policies** — the LMDB backend is
  arity-N but flat, without F#-style eager/lazy/cached tiers.
- Cross-target effective-distance bench vs Elixir/Haskell still open.

## Path forward

1. ISO three-form adoption.
2. Cross-target effective-distance benchmark vs Elixir/Haskell.
3. Richer LMDB policy tiers.
4. Runtime-parser capability entry if term IO is wanted.

## Document status

Fleet-aligned snapshot; source-verified line/kernel/LMDB facts and the
`ct_default_target(scala)` default-CI registration against
`wam_scala_target.pl` and the conformance harness (2026-07-11). Perf
figures cited from the mode bench, not re-run.
