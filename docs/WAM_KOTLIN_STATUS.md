<!--
SPDX-License-Identifier: MIT OR Apache-2.0
-->
# WAM Kotlin Target — Status

Living summary of the hybrid WAM-Kotlin backend
(`wam_kotlin_target.pl`), an early scaffold. For **mature** JVM routes
prefer **Scala** ([`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md)) or
**Clojure** ([`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md)).

Companion docs:

- [`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md),
  [`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md) — mature JVM routes.
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).
- [`WAM_KOTLIN_BENCH.md`](WAM_KOTLIN_BENCH.md) — interpreter vs lowered timing.
- [`design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md`](design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md).

## Role

**Kotlin hybrid-partition scaffold.** Partitions predicates into a
lowered fast path with a WAM-instruction fallback; smallest target
module in the fleet.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_kotlin_target.pl` | ~0.5k |
| `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl` | ~0.6k (T1 + T4 + T5 + execute + mid-body call) |
| Dedicated tests | ~1 file (plunit + Gradle e2e when available) |

## What's shipped

- **Hybrid partition** emit with a WAM fallback.
- **Gradle e2e** hook — the test suite includes a Gradle end-to-end
  path when the toolchain is available.
- **WAM-lowered native dispatch:** `wam_kotlin_lowered_emitter.pl` lowers:
  - **T1** deterministic single-clause — flat facts, register unification,
    write/read-mode structure/list construction, last-call `execute`,
    arithmetic `builtin_call`, deterministic mid-body `call`.
  - **T5** `clause_chain` — multi-clause with distinct first-arg
    `get_constant` discriminators (bound A1 if-cascade; unbound A1
    returns `false` so `tryRun` falls back to the interpreter).
  - **T4** `multi_clause_n` — all supported deterministic clauses
    inlined, tried in order with `snapshotForNative` /
    `restoreFromSnapshot` between attempts (incl. clauses ending in
    `execute` / mid-body `call`). Leading `get_constant` peel skips the
    entry snapshot on closed fail (KT-HEAP-SNAPSHOT-OPT-2).
  - **Last-call `execute` (EMIT-KOTLIN-4):** `return dispatch("P/N", state)`.
  - **Mid-body `call` + arith (EMIT-KOTLIN-5):**
    `if (!dispatch(...)) return false` + `kotlinLoBuiltinCall` — **only**
    when every mid-body callee is self-recursion or single-clause
    deterministic. Fib/ack lower; nondet mid-body callers decline.
    Self-recursion exemption is sound via top-level tryRun fallback
    (**KT-SELF-REC-SOUNDNESS** — do not remove that fallback lightly).
  Registered via `WamProgram.registerNative`. `functions` / `mixed`
  modes route lowerable preds through this path.

## Perf signal

See [`WAM_KOTLIN_BENCH.md`](WAM_KOTLIN_BENCH.md). After dispatch/snapshot
opts + EMIT-KOTLIN-5: append_500 ~**28×**, fib_15 ~**1.85×**, ack_23
~**1.78×**, member ~1.5×.

## Gaps

- **Nondeterministic mid-body `call`** — declined (first-solution hazard).
- **ITE/soft-cut, cut, aggregates** — not lowered.
- **Native recursion depth:** mid-body/tree recursion uses the JVM call
  stack. Measured ~**750–780** frames before `StackOverflowError` on the
  default stack (linear mid-body probe). Conformance `fib(10)` /
  `ack(2,3)` are fine; prefer decline over wrong answers if a workload
  would overflow.
- **Conformance (opt-in)** — `conformance_target(kotlin)` /
  `kotlin_functions` registered. **All classic programs green** (append,
  member, reverse, builtins, fib, ack) — no remaining `ct_xfail`s.
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No runtime-parser capability entry.**

## Path forward

1. Optional: heap-outside-map / trail-with-old-values if non-peeled T4 or
   CP snapshots dominate a new workload.
2. Optional: ITE/soft-cut; ISO / kernels if Kotlin graduates beyond scaffold.

## Document status

Fleet-aligned snapshot; source-verified against `wam_kotlin_target.pl`,
`wam_kotlin_lowered_emitter.pl`, and `tests/test_wam_kotlin_target.pl`
(2026-07-15). Through KT-SELF-REC-SOUNDNESS.
