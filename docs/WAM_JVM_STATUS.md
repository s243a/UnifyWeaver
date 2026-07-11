# WAM JVM Target — Status

Living summary of the hybrid WAM-JVM backend (`wam_jvm_target.pl`),
which emits JVM bytecode. For **mature** JVM routes prefer **Scala**
([`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md)) or **Clojure**
([`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md)); this generic
bytecode route is the third, and earliest, JVM path.

Companion docs:

- [`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md),
  [`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md) — mature JVM routes.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Generic JVM bytecode scaffold.** Emits JVM bytecode directly (rather
than through a JVM language), a third route after Scala and Clojure.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_jvm_target.pl` | ~0.7k |
| Lowered emitter | none |
| Dedicated tests | ~1 file |

## What's shipped

- **Dual bytecode emit** via Jamaica / Krakatau assembler routes
  (both referenced in the target).
- Smallest of the three JVM-family entries.

## Gaps

- **No lowered emitter.**
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No conformance registration** — no `conformance_target(jvm)`.
- **No runtime-parser capability entry.**
- Thin test surface (1 file).

## Path forward

1. Decide whether the generic-bytecode route earns investment given
   Scala and Clojure already cover the JVM maturely.
2. Add a lowered emitter and conformance adapter if it graduates.

## Document status

Fleet-aligned snapshot; source-verified line count, the
Jamaica/Krakatau dual-emit references, absence of a lowered emitter,
and absence of conformance registration against `wam_jvm_target.pl`
and the conformance harness (2026-07-11).
