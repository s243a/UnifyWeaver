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
| Lowered emitter | none |
| Dedicated tests | ~1 file (~9 plunit cases, incl. Gradle e2e when available) |

## What's shipped

- **Hybrid partition** emit with a WAM fallback.
- **Gradle e2e** hook — the test suite includes a Gradle end-to-end
  path when the toolchain is available.

## Gaps

- **No lowered emitter** (partition logic lives in the target).
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No conformance registration** — no `conformance_target(kotlin)`.
- **No runtime-parser capability entry.**

## Path forward

1. Decide whether Kotlin earns investment given Scala/Clojure already
   cover the JVM maturely.
2. Add a lowered emitter and conformance adapter if it graduates.

## Document status

Fleet-aligned snapshot; source-verified line count, the
hybrid-partition + Gradle-e2e shape, ~9-case density, and absence of
conformance registration against `wam_kotlin_target.pl` and the
conformance harness (2026-07-11).
