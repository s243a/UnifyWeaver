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
| `src/unifyweaver/targets/wam_kotlin_lowered_emitter.pl` | ~0.5k (T1 + T4 + T5 + execute) |
| Dedicated tests | ~1 file (plunit + Gradle e2e when available) |

## What's shipped

- **Hybrid partition** emit with a WAM fallback.
- **Gradle e2e** hook — the test suite includes a Gradle end-to-end
  path when the toolchain is available.
- **WAM-lowered native dispatch:** `wam_kotlin_lowered_emitter.pl` lowers:
  - **T1** deterministic single-clause — flat facts, register unification,
    write/read-mode structure/list construction, last-call `execute`.
  - **T5** `clause_chain` — multi-clause with distinct first-arg
    `get_constant` discriminators (bound A1 if-cascade; unbound A1
    returns `false` so `tryRun` falls back to the interpreter).
  - **T4** `multi_clause_n` — all supported deterministic clauses
    inlined, tried in order with `snapshotForNative` /
    `restoreFromSnapshot` between attempts (incl. clauses ending in
    `execute`).
  - **Last-call `execute` (EMIT-KOTLIN-4):** native fns take
    `(state, dispatch)` where `dispatch` is `WamRuntime.tryRun`; emit
    `return dispatch("P/N", state)`. Tail-recursive member/append and
    accumulator reverse lower.
  Registered via `WamProgram.registerNative`. `functions` / `mixed`
  modes route lowerable preds through this path.

## Perf signal (BENCH-KOTLIN + KT-DISPATCH-SNAPSHOT-OPT)

In-process `tryRun` timing (not JVM startup) — see
[`WAM_KOTLIN_BENCH.md`](WAM_KOTLIN_BENCH.md) and
[`design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md`](design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md).
Recursive native hops **skip** `snapshotForNative` (top-level fallback
kept). Profile: snaps were ~31% of `append_500` wall. **After:**
append_100 ~1.03×, append_500 ~0.85× (was ~0.55×); member ~1.4–1.6×.
Remaining recursive tax: T4 `_t4` map copy per entry.

## Gaps

- **Mid-body `call`** — still declined (fib, ack with non-tail call).
  Follow-up **EMIT-KOTLIN-5** (correctness; re-measure after).
- **T4 `_t4` snapshot cost** on deep recursion (blocks append_500 ≥1.0×).
- **ITE/soft-cut, cut, aggregates** — not lowered.
- **Native recursion depth:** `execute` uses the JVM call stack.
  Measured ~1000 peano-depth OK, ~2000 → `StackOverflowError` on the
  default stack. Conformance depths (list length 3) are fine.
- **Conformance (opt-in)** — `conformance_target(kotlin)` /
  `kotlin_functions` registered. **All classic programs green** (append,
  member, reverse, builtins, fib, ack) — no remaining `ct_xfail`s after
  KT-ARITH-SLASH-FUNCTOR + KT-LIST-BACKTRACK + KT-Y-ENV-RECURSION.
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No runtime-parser capability entry.**

## Path forward

1. Cheapen T4 `_t4` restore (trail-with-old-values / heap separation) if
   append_500 must reach ≥1.0×.
2. EMIT-KOTLIN-5 for mid-body `call` (correctness; re-measure perf).
3. Optional: ITE/soft-cut; ISO / kernels if Kotlin graduates beyond scaffold.

## Document status

Fleet-aligned snapshot; source-verified against `wam_kotlin_target.pl`,
`wam_kotlin_lowered_emitter.pl`, and `tests/test_wam_kotlin_target.pl`
(2026-07-14). Through KT-DISPATCH-SNAPSHOT-OPT.
