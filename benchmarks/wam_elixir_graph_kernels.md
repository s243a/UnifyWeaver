# WAM-Elixir graph kernels: measurements

Companion to `docs/WAM_TARGET_ROADMAP.md` and `benchmarks/wam_elixir_tier2_findall.md`.
Tracks the kernel-based-lowering work — the largest perf lever
identified in the roadmap. Two PRs landed the substrate; this doc
collects their measurements and the validation that auto-routing
preserves the wins.

## TL;DR

Kernel-based lowering of `transitive_closure` produces **3 orders
of magnitude** speedup at scale, and the compile-time pattern
recognition wrapper (PR #1800) preserves nearly all of that perf
when the user writes the canonical `tc/2` shape.

## Background

The roadmap doc identified three layers of perf headroom:

1. WAM-instruction lowering alone — modest speedup over plain
   bytecode interpretation.
2. Per-predicate native fast-path emitters (`wam_*_lowered_emitter.pl`)
   — closes part of the gap, bounded.
3. **Kernel-based lowering** — dispatching specific hot graph
   operations to hand-tuned native code. Go's `category_ancestor`
   FFI kernel hit 52× at scale-300; the kernel-vs-WAM benchmarks
   below validate the same lever for Elixir.

PR #1799 added the first Elixir kernel
(`WamRuntime.GraphKernel.TransitiveClosure`). PR #1800 added
compile-time pattern recognition that auto-routes the canonical
`tc/2` shape through the kernel — users no longer need to call the
kernel manually.

## Measurement: PR #1799 — kernel direct call vs WAM-compiled tc/2

Chain graph `1 → 2 → ... → N`. Driver code calls
`WamRuntime.GraphKernel.TransitiveClosure.reachable_from/2`
directly vs invoking WAM-compiled `tc/2` through `findall`.

| N | Kernel direct (μs) | WAM (μs) | Speedup |
| ---: | ---: | ---: | ---: |
| 10 | 3 | 27 | 9× |
| 50 | 21 | 107 | 5× |
| 200 | 85 | 18,541 | **218×** |
| 1000 | 320 | 500,567 | **1564×** |

Super-linear scaling: WAM's naive recursive `tc/2` is O(N²) on a
chain (no memoisation — explores every path), the kernel is O(N) BFS
with explicit visited tracking.

Caveat: this comparison favours the kernel because the WAM `tc/2` is
naive. A tabled / memoised WAM `tc` would close some of the gap. For
the realistic case where the user writes the obvious recursive
Prolog, the kernel is the win.

## Measurement: PR #1800 — auto-routing via kernel_dispatch(true)

Same chain graph, three measurement cells per workload:

- **default**: `tc/2` compiled to WAM, called via `findall`. Same as
  PR #1799's "WAM" baseline above.
- **kernel_auto**: `kernel_dispatch(true)` is set; `tc/2` is auto-
  routed through the dispatch wrapper. Called via `findall`.
- **kernel_direct**: control case — driver calls `reachable_from`
  directly, bypassing both WAM and the wrapper.

| N | default WAM (μs) | kernel_auto (μs) | kernel_direct (μs) | speedup vs WAM | wrapper overhead vs direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 22 | 11 | 4 | 2.0× | 2.75× |
| 50 | 102 | 45 | 36 | 2.3× | 1.25× |
| 200 | 19,097 | 139 | 94 | **137×** | 1.48× |
| 1000 | 659,177 | 671 | 615 | **982×** | 1.09× |

(The PR #1799 cells differ slightly from this run's PR #1800 cells —
single-machine variance plus a couple PRs of intervening codegen
changes. The crossover and order-of-magnitude conclusions are stable
across both runs.)

## Reading the wrapper-overhead column

The `kernel_auto / kernel_direct` ratio is the cost of the dispatch
wrapper: pattern recognition at codegen time produces a thin Elixir
module that fetches a FactSource handle from the registry, builds a
`neighbors_fn` closure, calls the kernel, dumps results into the
calling aggregate frame, and throws fail to drive finalisation.

- **At small N (10-50)**: wrapper overhead is 1.25-2.75×. The kernel
  itself is microsecond-scale; even small per-call costs (registry
  lookup, closure allocation, FactSource indirection) show up as a
  measurable ratio.
- **At large N (200-1000)**: wrapper overhead drops to 1.09-1.48×.
  Per-call costs are amortised across N node visits.

Conclusion: at N≥200 the auto-routing essentially preserves the
direct-call performance. At smaller N the wrapper has visible cost
but is still microsecond-scale absolute.

## Why these numbers will look different on your machine

- WAM's `tc/2` is O(N²) on a chain because it has no tabling. A
  tabled / memoised version would close some of the gap.
- The kernel uses a `MapSet` for visited tracking — BEAM's MapSet is
  O(log N) per op. Replacing with a process-dictionary or `:ets` set
  would shave constant factors at very large N.
- ETS lookup cost depends on table type (we used `:bag`); an
  `:ordered_set` with a different access pattern could change the
  per-step cost.
- The kernel's BFS is naive — DAG-aware optimisations (topological
  sort + linear pass) would beat it on acyclic inputs.

## Reproducing

PR #1799 measurement: `examples/benchmark/...` (script is in the
referenced PR; not shipped permanently — it's an ad hoc measurement
script used during the PR's development).

PR #1800 measurement: same approach, with `kernel_dispatch(true)`
and `source_module(user)` in `write_wam_elixir_project/3` Options.

Both rely on:
- `:ets` or another FactSource backing for the `edge/2` predicate.
- Registration via `WamRuntime.FactSourceRegistry.register("edge/2", handle)`
  before invoking `findall(Z, tc(X, Z), L)`.

## Known limitations of the current TC kernel + dispatch

- **One pattern only.** Only the canonical 2-clause TC shape
  (`tc(X, Z) :- edge(X, Z). tc(X, Z) :- edge(X, Y), tc(Y, Z).`) is
  recognised. Variants out of scope:
  - clause-order swap
  - right-recursive form `tc(X, Z) :- tc(Y, Z), edge(X, Y)`
  - non-2-ary edges
- **One kernel only.** `transitive_closure`. Future kernels:
  `shortest_path`, `strongly_connected_components`,
  `effective_distance`.
- **Single-machine measurements.** The 4-vCPU box numbers above
  don't tell you what happens on 16+ cores or in clustered Elixir
  setups. The kernel is single-process and won't scale to
  distributed graph workloads as-is.
- **No tabling for the WAM baseline.** A tabled `tc/2` would be
  faster than the naive version measured here.

## Related

- `docs/WAM_TARGET_ROADMAP.md` — strategic context.
- `benchmarks/wam_elixir_tier2_findall.md` — parallel-vs-sequential
  measurements (different perf lever; same workload shape).
- `benchmarks/wam_elixir_builtin_coverage.md` — runtime-builtin
  audit; the kernel layer sits on top of the builtin layer.

## History

- **PR #1799** — first Elixir graph kernel
  (`WamRuntime.GraphKernel.TransitiveClosure`); kernel-direct vs
  WAM measurement reported 218×–1564×.
- **PR #1800** — compile-time pattern recognition auto-routes
  canonical `tc/2` to the kernel; this doc validates the wrapper
  preserves the perf.
