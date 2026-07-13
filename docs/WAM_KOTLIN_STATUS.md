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
| `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl` | ~0.4k (T1 + T4 + T5) |
| Dedicated tests | ~1 file (plunit + Gradle e2e when available) |

## What's shipped

- **Hybrid partition** emit with a WAM fallback.
- **Gradle e2e** hook — the test suite includes a Gradle end-to-end
  path when the toolchain is available.
- **WAM-lowered native dispatch:** `wam_kotlin_lowered_emitter.pl` lowers:
  - **T1** deterministic single-clause — flat facts, register unification,
    write/read-mode structure/list construction.
  - **T5** `clause_chain` — multi-clause with distinct first-arg
    `get_constant` discriminators (bound A1 if-cascade; unbound A1
    returns `false` so `tryRun` falls back to the interpreter).
  - **T4** `multi_clause_n` — all supported deterministic clauses
    inlined, tried in order with `snapshotForNative` /
    `restoreFromSnapshot` between attempts.
  Registered via `WamProgram.registerNative`. `WamRuntime.run` /
  `tryRun` tries native first and falls back to the bytecode
  interpreter on `false`. `functions` / `mixed` modes route lowerable
  preds through this path.

## Gaps

- **`call`/`execute` in lowered bodies** — still declined (member,
  append, reverse, fib, ack stay on the interpreter). Follow-up
  **EMIT-KOTLIN-4**.
- **ITE/soft-cut, cut, aggregates** — not lowered.
- **Conformance (opt-in)** — `conformance_target(kotlin)` /
  `kotlin_functions` registered. **All classic programs green** (append,
  member, reverse, builtins, fib, ack) — no remaining `ct_xfail`s after
  KT-ARITH-SLASH-FUNCTOR + KT-LIST-BACKTRACK + KT-Y-ENV-RECURSION.
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No runtime-parser capability entry.**

## Path forward

1. EMIT-KOTLIN-4: emit `call`/`execute` in lowered bodies (recursion /
   inter-predicate).
2. Optional: ITE/soft-cut lowering; ISO / kernels if Kotlin graduates
   beyond scaffold.

## Document status

Fleet-aligned snapshot; source-verified against `wam_kotlin_target.pl`,
`wam_kotlin_lowered_emitter.pl`, and `tests/test_wam_kotlin_target.pl`
(2026-07-12). EMIT-KOTLIN-2 + CONF-KOTLIN + KT-ARITH-SLASH-FUNCTOR +
KT-LIST-BACKTRACK + KT-Y-ENV-RECURSION + EMIT-KOTLIN-3 landed
(T4/T5 multi-clause without call/execute; EMIT-KOTLIN-4 next).
