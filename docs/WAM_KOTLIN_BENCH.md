<!--
SPDX-License-Identifier: MIT OR Apache-2.0
-->
# Kotlin WAM — interpreter vs lowered (BENCH-KOTLIN)

**Question:** does `emit_mode(functions)` (native lowering) beat the bytecode
interpreter on wall time?

**Method:** in-process timing only via `benchmark_main(Pred, Iterations)`:
**5 warmup + 15 timed** `tryRun` batches inside one JVM; report **min**
(robust for JIT’d code) and median. **Not** timed: `gradle` / JVM startup.

Harness: `examples/benchmark/run_wam_kotlin_lowered_vs_interpreter.pl`

```bash
LANG=C.UTF-8 swipl -q -s examples/benchmark/run_wam_kotlin_lowered_vs_interpreter.pl -g main -t halt
```

Profiling (functions mode; attributes **all** `snapshotForNative`, incl. T4 `_t4`):

```bash
WAM_KT_PROFILE=1 gradle -q run --args='80 2 5 PROFILE=1'
```

## Results after EMIT-KOTLIN-5 (2026-07-15)

Hardened harness (5/15, min + median). All functions-mode cases confirmed
`registerNative` / `lowered_*` (incl. fib/ack).

| program | interp min | lowered min | **speedup min** | speedup med | notes |
|---|---:|---:|---:|---:|---|
| fact | 3.10 | 4.02 | 0.77× | 0.70× | ~parity (short) |
| list_builder | 6.39 | 5.56 | 1.15× | 1.02× | slight win |
| t5_color | 7.82 | 3.22 | 2.43× | 2.10× | win |
| t4_second_arg | 8.06 | 5.58 | 1.44× | 1.11× | win |
| member_100 | 70.44 | 47.19 | **1.49×** | 1.43× | win |
| member_500 | 70.21 | 46.60 | **1.51×** | 1.43× | win |
| append_100 | 216.07 | 28.64 | **7.54×** | 7.46× | win |
| append_500 | 872.58 | 31.31 | **27.87×** | 27.60× | win |
| fib_15 | 9848.71 | 5310.43 | **1.85×** | 1.81× | mid-body call win |
| ack_23 | 94.20 | 52.87 | **1.78×** | 1.77× | mid-body call win |

Speedup = interpreter_ms / lowered_ms (min batch).

## Append trajectory (KT-DISPATCH → heap peel)

| program | BENCH-KOTLIN | + KT-DISPATCH | + KT-HEAP-SNAPSHOT-OPT-2 |
|---|---:|---:|---:|
| append_100 | ~0.75× | ~1.03× | **~7×** |
| append_500 | ~0.55–0.64× | ~0.85× | **~28–30×** |

## Honest readout

- Tail `execute` (append) and mid-body `call` (fib/ack) both **win** after
  the dispatch/snapshot work — EMIT-KOTLIN-5 is not coverage-only.
- Fib/ack ~**1.8×**; tree recursion is still JVM-stack-bound (~750 mid-body
  frames on the default stack; conformance depths are fine).
- Nondeterministic mid-body call **declines** (first-solution would be wrong).
- Short cases: treat sub-10ms rows cautiously.

See [`design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md`](design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md).
