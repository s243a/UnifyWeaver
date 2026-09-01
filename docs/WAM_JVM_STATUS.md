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

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. JVM's audit (light pass — this
scaffold-tier backend was not source-read in depth):

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | fleet grep: only C++ and (post-A2) JS dispatch it |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **suspected** | register layout not audited |
| A3 | `Execute` of a builtin doesn't return to the continuation | **suspected** | the emitter pattern reaches this backend; the execute lowering was not audited |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted literals intern as atoms |

With no conformance registration (Gaps above), no emitted JVM output is
ever executed by the shared harness; resolve the suspected rows as part
of any graduation from scaffold status.

## Path forward

1. Decide whether the generic-bytecode route earns investment given
   Scala and Clojure already cover the JVM maturely.
2. Add a lowered emitter and conformance adapter if it graduates.

## Document status

Fleet-aligned snapshot; source-verified line count, the
Jamaica/Krakatau dual-emit references, absence of a lowered emitter,
and absence of conformance registration against `wam_jvm_target.pl`
and the conformance harness (2026-07-11). 2026-09-01: added the
whole-program (A2) deficiency audit (light pass); see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
