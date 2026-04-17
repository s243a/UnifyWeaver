# Haskell WAM FFI Profiling Report

**Date:** 2026-04-13
**Dataset:** 300-scale effective_distance benchmark (386 seeds, 6788 category_parent facts)
**Profiling flags:** `-prof -fprof-auto -rtsopts`, run with `+RTS -p`

## Summary

Profiling with FFI enabled reveals that the pure-interpreter bottleneck
(`step` at 59% time) is **not** the real performance issue when FFI
kernels handle the hot predicate. With FFI enabled, `step` drops to
~2.7%, and a different bottleneck emerges: **fact index construction**
that the FFI path doesn't even need.

## The 4 configurations

| Config | emit_mode | no_kernels | Description |
|--------|-----------|-----------|-------------|
| A: pure-interp | interpreter | true | All predicates via WAM interpreter |
| B: interp-ffi | interpreter | false | Interpreter + category_ancestor FFI kernel |
| C: lowered-only | functions | true | Lowered Haskell functions, no FFI |
| D: lowered-ffi | functions | false | Lowered functions + FFI kernel |

## Results (with profiling overhead, ~3-4x slower than non-profiled)

| Config | query_ms | total_time | total_alloc |
|--------|----------|------------|-------------|
| A: pure-interp | 9531 | 10.47s | 10.9 GB |
| B: interp-ffi | 791 | 1.50s | 2.3 GB |
| C: lowered-only | 10689 | 10.85s | 10.9 GB |
| D: lowered-ffi | 700 | 1.42s | 2.3 GB |

**Non-profiled baseline** (from prior benchmarks): B ~= 285ms, D ~= 260ms.

## Top cost centres

### Config A (pure interpreter) — interpreter-bound, as expected

| Cost centre | %time | %alloc |
|-------------|-------|--------|
| `step` (WamRuntime) | 59.7 | 54.1 |
| `run` (WamRuntime) | 10.1 | 4.1 |
| `buildFact2Code.buildGroup` (Main) | 6.5 | 19.3 |
| `==` (WamTypes) | 3.8 | 0.0 |

### Config B (interpreter + FFI) — **buildFact2Code dominates**

| Cost centre | %time | %alloc |
|-------------|-------|--------|
| `buildFact2Code.buildGroup` (Main) | **42.8** | **85.5** |
| `hashWithSalt1` (Data.Hashable) | 17.2 | 0.3 |
| `nativeKernel_category_ancestor.recHits.\` | 11.3 | 1.3 |
| `nativeKernel_category_ancestor.directParents` | 9.2 | 0.4 |
| `step` (WamRuntime) | **2.8** | 1.6 |
| `nativeKernel_category_ancestor.baseHits` | 2.5 | 0.3 |

### Config C (lowered, no FFI) — essentially same as A

Lowered emitter only converted helper predicates (power_sum_bound, etc.)
to native Haskell. The hot predicate `category_ancestor/4` still goes
through the interpreter because it can't be lowered (recursive, uses
member/2 for cycle check). So the interpreter still dominates.

**Conclusion:** lowering helpers does nothing when the hot predicate
isn't lowered.

### Config D (lowered + FFI) — same profile as B, marginal speedup

Lowering the helpers shaves ~11% off B (791ms → 700ms) but the same
`buildFact2Code` dominates (45.9% time, 85.5% alloc).

## Root cause: double fact construction

`Main.hs` builds each fact set **twice**:

1. **`buildFact2Code "category_parent" ...`** — emits WAM instructions
   for each fact (`[GetConstant child, GetConstant parent, Proceed]`
   per fact, plus a `SwitchOnConstant` HashMap for dispatch). For
   6788 facts: ~20k instructions and ~6788 HashMap entries.

2. **`parentsIndex = Map.fromListWith (++) ...`** — builds the FFI-side
   `HashMap String [String]` used by `nativeKernel_category_ancestor`.

In configs B and D, the FFI kernel handles ALL category_parent lookups.
**The WAM-compiled version is never executed** — but it's still built
and its thunks are forced through lazy evaluation, costing ~60% of
total time.

Even worse: `article_category` and `root_category` are compiled to WAM
instructions but **never referenced by any compiled predicate**. They're
used only inside `main` to compute the seed list (in pure Haskell).

## Confirmed: no compiled predicate references article_category or root_category

```
$ grep -r 'article_category\|root_category' src/Predicates.hs
(no matches)
```

All three fact tables are WAM-compiled dead code in configs B and D.

## Proposed optimization

Add a mechanism to **skip WAM-compilation of fact predicates that are:**
1. Owned by an FFI kernel (e.g., `category_parent` owned by `category_ancestor`)
2. Not referenced by any compiled predicate (e.g., `article_category`,
   `root_category` in this benchmark)

Options:
- **Auto-detect** from `DetectedKernels` + label references in compiled predicates
- **Explicit** via `skip_fact_predicates([list])` option
- **Both** — auto-detect by default, override via option

### Expected impact

If `buildFact2Code` accounts for 42% of B's 1.5s profiled runtime, that's
~630ms of waste. Extrapolating to non-profiled runtime (B's baseline was
285ms), the `buildFact2Code` overhead is likely ~80-100ms of real time.

**Potential speedup: 30-40% on FFI-dominant workloads.**

This could bring the Haskell target's 300-scale time from ~260ms to
~160-180ms — closing most of the gap to Rust's 126ms without requiring
parallelization.

## Secondary findings

- **`step` at 2.8% time in FFI configs** — the interpreter is essentially
  free once the hot predicate is handled by FFI. Any further interpreter
  optimization (atom interning, ST monad) is effectively pointless for
  FFI-dominant workloads.
- **`hashWithSalt1` at 17%** — HashMap hashing during fact index
  construction. Would also be eliminated by the skip-facts optimization.
- **Lowering has near-zero impact** when the hot predicate can't be
  lowered. Lowering helps only when it removes the interpreter from a
  frequently-executed path.

## Implications for the roadmap

Before proceeding to **Phase 1 (seed-level parallelism)** in the vision
roadmap, implement the **skip-facts optimization** — it's a simple,
high-impact change that directly addresses the bottleneck revealed by
profiling. Parallelism won't help if 42% of the work is redundant fact
table construction.

## Reproduction

```bash
swipl -q -s examples/benchmark/gen_prof_matrix.pl -- \
    data/benchmark/300/facts.pl /tmp/wam-prof-matrix

for cfg in A-pure-interp B-interp-ffi C-lowered-only D-lowered-ffi; do
    (cd /tmp/wam-prof-matrix/$cfg && cabal build)
done

for cfg in A-pure-interp B-interp-ffi C-lowered-only D-lowered-ffi; do
    cd /tmp/wam-prof-matrix/$cfg
    ./dist/build/wam-prof-$cfg/wam-prof-$cfg \
        data/benchmark/300 +RTS -p -RTS
    head -20 wam-prof-$cfg.prof
done
```
