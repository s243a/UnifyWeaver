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

**⚠️ Single run — the short rows are not repeatable** (see Reproduction /
variance below). Kept here as one sample; read only the `append` rows as a
finding.

| program | interpreter ms | lowered ms | speedup | notes |
|---|---:|---:|---:|---|
| fact (flat) | 17.2 | 37.0 | 0.46× | noise (see variance: 0.46–3.77×) |
| list_builder | 35.9 | 45.0 | 0.80× | noise (0.80–3.68×) |
| t5_color (first-arg) | 17.0 | 30.8 | 0.55× | noise (0.55–2.08×) |
| t4_second_arg | 20.1 | 18.8 | 1.07× | ~parity (noisy) |
| member_100 | 72.0 | 60.6 | 1.19× | ~parity (noisy) |
| member_500 | 73.3 | 64.1 | 1.14× | ~parity |
| append_100 | 217.4 | 279.1 | **0.78×** | **lowered slower (reproducible)** |
| append_500 | 932.7 | 1682.2 | **0.55×** | **lowered slower (reproducible)** |

Speedup = interpreter_ms / lowered_ms. Short cases run in sub-30ms batches and
are dominated by JIT/GC noise — treat only the `append` regression as real.

## Reproduction / variance — the short cases are NOT stable

The table above is a single run. Two independent re-runs (same harness, same
machine) show the **short/non-recursive cases swing wildly** and cannot support
any conclusion, while the **deep-recursion regression reproduces**:

| program | run A (above) | run B | run C | reproducible? |
|---|---:|---:|---:|---|
| fact | 0.46× | 3.77× | 0.74× | **no — 8× swing (noise)** |
| list_builder | 0.80× | 3.68× | 0.84× | **no (noise)** |
| t5_color | 0.55× | 2.08× | 1.26× | **no (noise)** |
| t4_second_arg | 1.07× | 1.09× | 1.31× | ~parity (noisy) |
| member_100 | 1.19× | 1.15× | 0.94× | ~parity (noisy) |
| member_500 | 1.14× | 0.99× | 1.02× | ~parity |
| append_100 | 0.78× | 0.80× | 0.71× | **yes — slower** |
| append_500 | 0.55× | 0.61× | 0.64× | **yes — slower** |

The short cases run in sub-30ms batches, so JIT/GC/scheduling noise dominates —
`fact` alone ranged 0.46×→3.77×. Do **not** read the short-case rows as
regressions; they are inconclusive.

## Honest readout

The one **reproducible** signal is that **deep tail-recursion regresses**:
`append` is ~0.6–0.8× (lowered slower) across all three runs, worsening with
depth (500 slower than 100) — consistent with an O(depth) cost per query.
Everything else (facts, T5, list-builder, T4, member) is **noise-dominated /
inconclusive** at these batch sizes — the harness cannot currently say whether
lowering helps or hurts on short, non-recursive predicates.

Likely cost driver for the recursion regression (well-supported by the shape,
not yet profiled):

- **`tryRun` snapshots on every native/recursive hop.** Each `execute` →
  `dispatch` → `tryRun` (EMIT-KOTLIN-4) calls `snapshotForNative`, which
  deep-copies the register/heap map — and heap vars (`H<n>`) accumulate
  unbounded within a query, so per-hop snapshot cost grows with depth →
  ~O(depth²) for a length-`depth` recursion. The interpreter's `execute` only
  resets PC on an explicit WAM stack; no per-call heap snapshot.

The other hypotheses (fallback path cost, tiny-predicate call overhead) are
**not supported** by the data — those cases are within noise.

## What to do next

1. **Fix the harness first** (`KT-DISPATCH-SNAPSHOT-OPT` and/or a benchmark
   hardening): more warmup/timed batches and report **min** batch-ms (the
   standard robust estimator for JIT'd code) so the short cases stop swinging.
   Only then can facts/T5/list-builder be judged.
2. **Profile + optimize the recursion path** — avoid the per-hop full-map
   snapshot (trail-based undo like the interpreter, or skip the snapshot on a
   same-predicate tail `execute`). This is the one measured regression and the
   highest-value optimization.

## Implications for EMIT-KOTLIN-5

Mid-body `call` (fib/ack) is *more* recursion on the same `tryRun`+snapshot
seam, so on today's dispatch it would inherit (or worsen) the append-style
regression. **Cheapen the dispatch seam (KT-DISPATCH-SNAPSHOT-OPT) before
EMIT-KOTLIN-5**, or treat EMIT-KOTLIN-5 as correctness-only with no perf claim.

## Stability caveat

Numbers here are from one machine; short-case batches are **not** repeatable
(see the variance table — up to 8× run-to-run). Median of 5 batches / 2 warmup
is too few for the short cases. The durable signal is the deep-recursion
regression, which held across three runs.
