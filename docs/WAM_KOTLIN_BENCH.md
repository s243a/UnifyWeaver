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

Profiling A/B (functions mode only):

```bash
WAM_KT_PROFILE=1 gradle -q run --args='SKIP_RECURSIVE_SNAP=0'   # legacy per-hop snap
WAM_KT_PROFILE=1 gradle -q run --args='SKIP_RECURSIVE_SNAP=1'   # optimized (default)
```

## Results after KT-DISPATCH-SNAPSHOT-OPT (2026-07-14)

Hardened harness (5/15, min + median). All functions-mode cases confirmed
`registerNative` / `lowered_*`.

| program | interp min | lowered min | **speedup min** | speedup med | notes |
|---|---:|---:|---:|---:|---|
| fact | 3.23 | 3.43 | 0.94× | 0.87× | ~parity (short) |
| list_builder | 6.22 | 5.53 | 1.13× | 1.10× | slight win |
| t5_color | 8.66 | 2.72 | 3.19× | 1.79× | win (still noisy med) |
| t4_second_arg | 6.25 | 5.51 | 1.14× | 0.92× | ~parity |
| member_100 | 68.48 | 43.83 | **1.56×** | 1.55× | win |
| member_500 | 64.45 | 44.62 | **1.44×** | 1.44× | win |
| append_100 | 222.85 | 216.14 | **1.03×** | 1.04× | ~parity (was ~0.75×) |
| append_500 | 841.41 | 985.75 | **0.85×** | 0.85× | improved; still under 1.0× |

Speedup = interpreter_ms / lowered_ms (min batch).

## Before → after (append, the real signal)

| program | before (BENCH-KOTLIN, ~median) | after (speedup min) |
|---|---:|---:|
| append_100 | ~0.75× | **1.03×** |
| append_500 | ~0.55–0.64× | **0.85×** |

Functions-only self-speedup on append_500 profile run (80×5): legacy
per-hop snap **419.7 ms** min → skip-recursive **270.9 ms** min (~1.55×
faster native path).

### Profile evidence

| config | snap_fraction_of_wall | snap_count / native_entries |
|---|---:|---|
| BEFORE (`SKIP_RECURSIVE_SNAP=0`) | **30.7%** | 200400 / 200400 |
| AFTER (default skip) | **~0%** | 400 / 200400 |

See [`design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md`](design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md).

## Honest readout

- **tryRun per-hop snapshot was the dominant recursive tax** (~31% of wall;
  one snap per native entry). Skipping it on recursive hops fixes
  append_100 to parity and lifts append_500 from ~0.55× to ~0.85×.
- **append_500 still does not beat the interpreter.** Remaining cost:
  T4’s `val _t4 = snapshotForNative()` once per recursive lowered entry
  (still copies the growing `H<n>` map). Documented follow-up in the
  optimization history (trail-with-old-values / heap separation).
- Short cases are more stable with 5/15 + min, but treat sub-10ms rows
  cautiously; member/append are the durable signals.

## Implications for EMIT-KOTLIN-5

Mid-body `call` still adds continuation machinery on top of this seam.
Dispatch is cheaper now for tail `execute`, but EMIT-KOTLIN-5 should not
claim a win until measured; T4 snapshot cost remains on multi-clause
recursion.
