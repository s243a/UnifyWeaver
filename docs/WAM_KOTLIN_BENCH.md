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

## Results after KT-HEAP-SNAPSHOT-OPT-2 (2026-07-14)

Hardened harness (5/15, min + median). All functions-mode cases confirmed
`registerNative` / `lowered_*`.

| program | interp min | lowered min | **speedup min** | speedup med | notes |
|---|---:|---:|---:|---:|---|
| fact | 3.64 | 3.92 | 0.93× | 0.75× | ~parity (short) |
| list_builder | 6.28 | 5.58 | 1.13× | 1.01× | slight win |
| t5_color | 8.63 | 3.35 | 2.58× | 1.96× | win |
| t4_second_arg | 6.62 | 6.12 | 1.08× | 0.97× | ~parity |
| member_100 | 65.68 | 46.94 | **1.40×** | 1.37× | win |
| member_500 | 66.12 | 48.34 | **1.37×** | 1.31× | win |
| append_100 | 235.26 | 32.63 | **7.21×** | 6.88× | win (was ~1.03×) |
| append_500 | 884.52 | 29.16 | **30.33×** | 28.94× | win (was ~0.85×) |

Speedup = interpreter_ms / lowered_ms (min batch).

## Before → after (append, the real signal)

| program | BENCH-KOTLIN | + KT-DISPATCH | **+ KT-HEAP-SNAPSHOT-OPT-2** |
|---|---:|---:|---:|
| append_100 | ~0.75× | ~1.03× | **7.21×** |
| append_500 | ~0.55–0.64× | ~0.85× | **30.33×** |

### Profile evidence (append_500, 80×5, functions)

| config | snap_fraction_of_wall | snap_count / native_entries | max_register_map_size |
|---|---:|---|---:|
| AFTER KT-DISPATCH (entry `_t4` every hop) | **48.1%** | 200800 / 200400 | 508 |
| AFTER peel leading `get_constant` | **8.6%** | 800 / 200400 | 508 |

See [`design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md`](design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md).

## Honest readout

- **tryRun per-hop snapshot** was the first recursive tax (~31% of wall);
  KT-DISPATCH fixed that and left append_500 at ~0.85×.
- **T4 `_t4` per-entry map copy** was the residual (~48% of wall once
  properly attributed). Peeling a leading fail-closed `get_constant` makes
  append’s cons path snap-free → **≥1.0× with large margin**.
- Member stays ~1.4× (first instr is `get_list`, not peeled).
- Short cases: treat sub-10ms rows cautiously; member/append are durable.

## Implications for EMIT-KOTLIN-5

Mid-body `call` still needs continuation machinery. Tail `execute`
recursion is no longer snapshot-bound for peelable T4 heads; re-measure
after EMIT-KOTLIN-5 lands.
