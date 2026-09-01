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

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Clojure's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | `sub_atom` appears in the builtin table but `sub_string/5` nowhere |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **suspected (low)** | registers are string-named and env frames are per-`Allocate` maps (`runtime.clj.mustache:2846-2855`), so the numeric X→Y aliasing form is absent; the frameless-Y-write form was not fully audited |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified** | the `:execute` arm special-cases only `variant/2`; any other unresolved name → `backtrack` = silent goal failure (`runtime.clj.mustache:3275-3283`) |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted literals intern as atoms |

Compounding A3/B-H2: `wam_clojure` has **no conformance arm**
(CONF-CLOJURE still open in
[`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md)) — with Lua, one of
only two WAM backends whose emitted output the shared harness never
executes.

Pattern lane: `clojure_target.pl` compiles facts from
`clause(Head, true)` — the G-A3-8 execute-at-compile-time hazard is
absent; the G-A3 machinery has no analogue there.

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
conformance registration (2026-07-11). 2026-09-01: added the
whole-program (A2) deficiency audit; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
