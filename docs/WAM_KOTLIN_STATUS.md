# WAM Kotlin Target — Status

Living summary of the hybrid WAM-Kotlin backend
(`wam_kotlin_target.pl`), an early scaffold. For **mature** JVM routes
prefer **Scala** ([`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md)) or
**Clojure** ([`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md)).

Companion docs:

- [`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md),
  [`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md) — mature JVM routes.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Kotlin hybrid-partition scaffold.** Partitions predicates into a
lowered fast path with a WAM-instruction fallback; smallest target
module in the fleet.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_kotlin_target.pl` | ~0.5k |
| `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl` | ~0.3k (deterministic single-clause) |
| Dedicated tests | ~1 file (~12 plunit cases, incl. Gradle e2e when available) |

## What's shipped

- **Hybrid partition** emit with a WAM fallback.
- **Gradle e2e** hook — the test suite includes a Gradle end-to-end
  path when the toolchain is available.
- **WAM-lowered native dispatch (first cut):** `wam_kotlin_lowered_emitter.pl`
  lowers **flat single-clause facts + register unification** (no structure/list
  construction — see below) to `fun (state: WamState): Boolean`, registered via
  `WamProgram.registerNative`. `WamRuntime.run` tries native first and falls back
  to the bytecode interpreter on `false` (WAT-style snapshot restore). `functions`
  / `mixed` modes route lowerable preds through this path; everything else
  (including any structure/list builder) stays on the interpreter registrars.

## Gaps

- **Lowered emitter scope** — only flat single-clause facts + register
  unification. Structure/list **construction** was silently wrong in the first
  cut (unbound vars in the result) and now declines to the interpreter; fixing it
  is follow-up **EMIT-KOTLIN-2**. No T4/T5 multi-clause, ITE, or `call`/`execute`
  in lowered bodies.
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No conformance registration** — no `conformance_target(kotlin)`.
- **No runtime-parser capability entry.**

## Path forward

1. Extend lowered emitter (T4/T5/ITE) following Lua/Scala models.
2. Add conformance adapter if Kotlin graduates beyond scaffold.

## Document status

Fleet-aligned snapshot; source-verified against `wam_kotlin_target.pl`,
`wam_kotlin_lowered_emitter.pl`, and `tests/test_wam_kotlin_target.pl`
(2026-07-12).
