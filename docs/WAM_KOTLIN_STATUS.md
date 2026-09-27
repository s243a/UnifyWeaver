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

**Nested head terms (fixed 2026-08-09, CONF-FIX-KOTLIN-NESTED).** The
runtime carried a single `var context: WamContext?`, which
`beginStructure` replaced outright. A term nested inside another — the
`[tk(X)|R]` shape every tokenizer uses — therefore discarded the
enclosing list's pending arguments, and the `unify_variable` that should
have taken the cons tail got `null` and failed the clause. The write
path had it too: building `[tk(X)|[]]`, the nested `tk/1` frame
completed and cleared the context, so the tail was never written.
`WamContext` now carries an `abstract val parent`, and
`nextReadArg`/`pushWriteArg` restore it. Because `ChoicePoint`,
`CallFrame` and `WamNativeSnapshot` already carry `context`, the whole
chain is saved and restored by the existing machinery. Both emit modes
share these helpers, which is why `kotlin` and `kotlin_functions` failed
identically and are fixed together.

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
  `kotlin_functions` registered. **All programs green** — the six
  classics (append, member, reverse, builtins, fib, ack) plus the four
  head-shape programs (`wide`, `nested`, `emptylist`, `repeatvar`) — no
  remaining `ct_xfail`s.
- **No foreign kernels, no LMDB / fact source, no ISO contract.**
- **No runtime-parser capability entry.**

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Kotlin's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | no `sub_atom/5` either — the string-builtin surface is thin |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **absent (aliasing form)** | registers are string-named and Y regs resolve through per-frame environment slots (`WamRuntime.kt.mustache:104-110`), so numeric X→Y aliasing cannot occur; the frameless-Y-write form was not fully audited |
| A3 | `Execute` of a builtin doesn't return to the continuation | **suspected** | the native/bytecode dispatch (`tryRun` for last-call `execute P/N`, EMIT-KOTLIN-4) was not audited for the shape "last goal is a builtin with no label"; the shared emitter emits it (`deallocate` + `execute <builtin>`) for anything outside `is_builtin_pred/2` |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted literals intern as atoms |

Pattern lane: `kotlin_target.pl` compiles facts from
`clause(Head, true)` — the G-A3-8 execute-at-compile-time hazard is
absent; the G-A3 machinery has no analogue there.

## Path forward

1. Optional: heap-outside-map / trail-with-old-values if non-peeled T4 or
   CP snapshots dominate a new workload.
2. Optional: ITE/soft-cut; ISO / kernels if Kotlin graduates beyond scaffold.

## Document status

Fleet-aligned snapshot; source-verified against `wam_kotlin_target.pl`,
`wam_kotlin_lowered_emitter.pl`, and `tests/test_wam_kotlin_target.pl`
(2026-07-15). Through KT-SELF-REC-SOUNDNESS. 2026-09-01: added the
whole-program (A2) deficiency audit; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
