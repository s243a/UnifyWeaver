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
- **WAM-lowered native dispatch:** `wam_kotlin_lowered_emitter.pl` lowers
  **deterministic single-clause** predicates — flat facts, register unification,
  and write/read-mode structure/list construction (`get/put_structure`,
  `get/put_list`, `set_*`, `unify_*`) — to `fun (state: WamState): Boolean`,
  registered via `WamProgram.registerNative`. `WamRuntime.run` tries native first
  and falls back to the bytecode interpreter on `false` (WAT-style snapshot
  restore). `functions` / `mixed` modes route lowerable preds through this path.

## Gaps

- **Lowered emitter scope** — deterministic single-clause only. No T4/T5
  multi-clause, ITE, or `call`/`execute` in lowered bodies (separate cards).
- **Conformance (opt-in)** — `conformance_target(kotlin)` /
  `kotlin_functions` registered. **append/3 green**; member/reverse/builtins/fib/ack
  are `ct_xfail` on measured interpreter gaps (list placeholder clobber under
  backtrack; `///2` arith functor parse; Y-register bind-through across recursion).
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No runtime-parser capability entry.**

## Path forward

1. Retire kotlin `ct_xfail`s (list structure-sharing, `//` functor strip, Y-env).
2. Extend lowered emitter (T4/T5/ITE) following Lua/Scala models.
3. Add ISO / kernels if Kotlin graduates beyond scaffold.

## Document status

Fleet-aligned snapshot; source-verified against `wam_kotlin_target.pl`,
`wam_kotlin_lowered_emitter.pl`, and `tests/test_wam_kotlin_target.pl`
(2026-07-12). EMIT-KOTLIN-2 + CONF-KOTLIN landed.
