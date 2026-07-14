# Kotlin WAM — interpreter vs lowered (BENCH-KOTLIN)

**Question:** does `emit_mode(functions)` (native lowering) beat the bytecode
interpreter on wall time?

**Method:** in-process timing only. Each generated project uses
`benchmark_main(Pred, Iterations)`: warm up, then take the **median** of
several timed batches of `tryRun` loops inside one JVM. **Not** timed:
`gradle` / JVM startup.

Harness: `examples/benchmark/run_wam_kotlin_lowered_vs_interpreter.pl`

```bash
LANG=C.UTF-8 swipl -q -s examples/benchmark/run_wam_kotlin_lowered_vs_interpreter.pl -g main -t halt
```

## Results (2026-07-14, this environment)

Median batch wall-ms (higher speedup = lowered faster). All functions-mode
cases confirmed `registerNative` / `lowered_*` present.

| program | interpreter ms | lowered ms | speedup | notes |
|---|---:|---:|---:|---|
| fact (flat) | 17.2 | 37.0 | **0.46×** | lowered slower |
| list_builder | 35.9 | 45.0 | **0.80×** | lowered slower |
| t5_color (first-arg) | 17.0 | 30.8 | **0.55×** | lowered slower |
| t4_second_arg | 20.1 | 18.8 | **1.07×** | slight win |
| member_100 | 72.0 | 60.6 | **1.19×** | modest win |
| member_500 | 73.3 | 64.1 | **1.14×** | modest win |
| append_100 | 217.4 | 279.1 | **0.78×** | lowered slower |
| append_500 | 932.7 | 1682.2 | **0.55×** | lowered slower |

Speedup = interpreter_ms / lowered_ms. Values are rough (scaffold target,
batch median); treat as a 2–3×-or-not signal, not microbench precision.

## Honest readout

**Lowering does not broadly pay off yet.** Flat facts, T5 dispatch, list
construction, and especially **tail-recursive append** are slower in
functions mode. The only clear (small) wins are T4 second-arg dispatch and
`member` (~1.1–1.2×).

Likely cost drivers (not proven here, but consistent with the numbers):

1. **`tryRun` snapshots** every native entry — including every recursive
   `execute` → `dispatch` → `tryRun` hop (EMIT-KOTLIN-4). The interpreter's
   `execute` only resets PC on an explicit WAM stack; no per-call heap snapshot.
2. **Native false → restore → bytecode fallback** path adds work when a T4
   clause fails before a later clause succeeds (member's first-clause miss).
3. For tiny predicates (facts), snapshot + Kotlin call overhead dominates the
   short bytecode body.

## Implications for EMIT-KOTLIN-5

Mid-body `call` (fib/ack) would need a continuation-friendly dispatch. If it
reuses today's `tryRun`+snapshot seam, it is **unlikely** to win on performance
until recursive/native dispatch avoids per-hop snapshots (e.g. direct native
tail call, or interpreter-style PC transfer for same-predicate execute).
Correctness still matters; performance ROI for EMIT-KOTLIN-5 looks weak unless
the dispatch seam is cheapened first.

## Stability

Same harness, one machine, median of 5 timed batches after 2 warmup batches.
Re-runs can jitter tens of percent on short cases; the qualitative pattern
(append/facts regress; member slight win) is the durable signal.
