# WAM Clojure Target — Status

Living summary of the hybrid WAM-Clojure backend
(`wam_clojure_target.pl` + `wam_clojure_lowered_emitter.pl`). Distinct
from the **non-WAM** direct Clojure compiler (`clojure_target.pl`).
Design proposals live under `docs/**/WAM_CLOJURE_*`.

Companion docs:

- `WAM_CLOJURE_*` proposal docs (design/proposals).
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**LMDB-on-the-JVM niche.** First-class LMDB JNI data tier with cache
policies; a JVM WAM route distinct from Scala's, where the lowered
emitter is actually larger than the target module.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_clojure_target.pl` | ~0.9k |
| `src/unifyweaver/targets/wam_clojure_lowered_emitter.pl` | ~1.5k |
| Dedicated tests | ~5 files (~147 plunit cases) |

Note the lowered emitter is **larger** than the target module — most
of the codegen weight is in the lowered path.

## What's shipped

**LMDB JNI tier.** Production-grade JNI loader (delay-wrapped) with
cache policies: `memoize` / `shared` / `two_level`. LMDB is the
dominant theme in the source.

**Foreign category handlers.** Foreign handlers for
`category_parent` / `category_ancestor` — **not** the shared-7 FFI
kernel set.

**Deterministic-prefix lowering.** Dual WAM-instr + lowered emitter;
T4 lowering **strips `switch_on_constant` prefixes** rather than
emitting a switch table (it is not "no switch handling").

## Gaps (relative to Rust / Haskell / F#)

- **No shared-7 FFI graph kernels** — only the two foreign category
  handlers.
- **No conformance registration** — no `conformance_target(clojure)`.
- **Sequential-only tests**; lowered emitter covers the deterministic
  prefix, not non-deterministic prefixes.
- **No runtime-parser capability entry.**
- **No emitted switch tables** for T4.

## Path forward

1. Extend the lowered emitter to non-deterministic prefixes and emit
   real switch tables.
2. Add parallelism gates.
3. Register a conformance adapter.
4. Broaden foreign handlers toward the shared-7 kernel set if graph
   perf becomes a goal.

## Document status

Fleet-aligned snapshot; source-verified line counts (lowered > target),
the LMDB-JNI + cache-policy surface, the foreign category handlers, the
`switch_on_constant` prefix-stripping T4 behavior, and the absence of
conformance registration (2026-07-11).
