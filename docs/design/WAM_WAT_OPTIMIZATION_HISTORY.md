<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025-2026 John William Creighton (@s243a)
-->
# WAM-WAT optimization history

This document records the WAM-WAT target's performance-optimization
track from the Phase 6 baseline (April 2026) through PR #1543.
It has two purposes:

1. **Capture what we did, with commit references** — so that when
   someone asks "why is `type_dispatch_a1` the way it is?" the
   answer is a specific PR with a rationale, not a 20-PR
   archaeology dig.
2. **Document what we considered but didn't do, and why** — so
   future contributors can pick up where we stopped (or not)
   with accurate cost/benefit estimates rather than rediscovering
   the trade-offs.

Complements [`WAM_WAT_ARCHITECTURE.md`](WAM_WAT_ARCHITECTURE.md),
which describes the **current state**. This doc is the **log of
how we got there**.

---

## Baseline

Before the optimization track, the WAM-WAT target's post-Phase-6
state (see
[`WAM_TERM_BUILTINS_PHASE_6_PERF.md`](WAM_TERM_BUILTINS_PHASE_6_PERF.md))
was 1.1–5.7× faster than SWI-Prolog on 9 of 13 bench workloads
and within 1.75× on the remaining 4. The remaining 4 (deep
recursive term walking) were the starting point for the
optimization work documented here.

Reference numbers from
[`WAM_TERM_BUILTINS_PHASE_6_PERF.md`](WAM_TERM_BUILTINS_PHASE_6_PERF.md):

| Workload | WAM-WAT baseline (ns) | Target |
|---|---|---|
| bench_sum_small | 1,686 | improve |
| bench_sum_medium | 4,172 | improve |
| bench_sum_big | 8,003 | improve |
| bench_term_depth | 8,930 | improve |
| bench_fib10 | 36,811 | improve |

Current (post-optimization) on the same bench hardware:

| Workload | Post-optimization (ns) | Δ vs Phase 6 baseline |
|---|---|---|
| bench_sum_small | ~1,100 | −35% |
| bench_sum_medium | ~2,700 | −35% |
| bench_sum_big | ~5,500 | **−31%** (cumulative) |
| bench_term_depth | ~7,000 | −22% |
| bench_fib10 | ~35,000 | −5% |

(Min-of-3-pair measurements on thermally-constrained Termux
hardware. Exact numbers vary ±3% between runs.)

---

## Optimizations made

Grouped by family. Each entry lists commit SHA, PR number, and
a one-line rationale.

### Pre-optimization groundwork

These PRs set up the measurement infrastructure and did the
initial easy wins that this document's real track built on.

| Commit | PR | Change |
|---|---|---|
| `27d057e9` | #1316 | Phase 6 benchmark + O(1) term-builtin dispatch + writeup |
| `a5b91b06` | #1386 | Reduce CP save from 32 to 8 A-regs (1.5× recursive speedup) |
| `2c7f5d2a` | #1399 | `neck_cut_test` peephole introduced (had a latent bug — didn't actually fire; see PR #1528 for the fix) |
| `930fe607` | #1450 | Phase 6 rerun + lexicographic atom comparison |

### Fused arithmetic

Collapse the 5-instruction `X is A OP B` window into a single
instruction that computes the result in place.

| Commit | PR | Change |
|---|---|---|
| `9dcd4cc7` | #1456 | `fused_is_add` — the template pattern |
| `5fff3a33`, `e3f47c5f` | #1463 | `fused_is_sub` / `fused_is_mul` / `_const` variants; direct-dispatch `arg_direct` and `functor_direct` |
| `34c7f77d` | #1479 | More direct dispatchers (`copy_term_direct`, `univ_direct`, `is_list_direct`) + nested arithmetic peephole |

Rationale: `is/2` dominates arithmetic-heavy benchmarks
(fib, sum). The fused instruction skips the `put_structure +
set_value + builtin_call(is/2)` dispatch chain for the common
`Dest is Src1 OP Src2` shape.

### arg/3 + call fusion family

The largest single optimization family. Fuses `arg(N, T, V)`
with the surrounding register-shuffling and call-setup
instructions that dominate term-walking predicates like
`sum_ints_args/5` and `term_depth_args/5`.

| Commit | PR | Change |
|---|---|---|
| `6a4ad386` | #1479 | `arg_reg_direct` / `arg_lit_direct` — 4-instruction fuse (N/T/Dest + arg call) |
| `119ebde1`, `654b68f2` | #1484 | Inline arg/3 read-mode fast path; `arg_to_a1` fusion (arg + put_value to A1) |
| `bca55ff9`, `b2ae8d5b` | #1492 | `arg_call_reg_3` — 4-way fusion (arg + A2 setup + A3 setup + call); liveness-based `_dead` variants |
| `dba07aaf`, `e7962359` | #1495 | K-family extension (K=1, K=2 with IsVar flag for put_value vs put_variable A2) |
| `7fd68a46` | #1497 | Look-through liveness across `try_me_else` (enables dead-variant selection for clauses with in-body disjunctions) |

Rationale: profiling showed `$step` dispatch dominated, and
each per-instruction dispatch had fixed overhead. Fusing the
typical 4- to 7-instruction windows around arg/3 cuts dispatch
count proportionally.

### Tail-call + clause-end fusion

Collapses the end-of-clause sequences that every allocated
predicate pays for on every call.

| Commit | PR | Change |
|---|---|---|
| `3af28873` | #1501 | `tail_call_5` — fuse `put_value × 5 + deallocate + execute` (7 → 1) |
| `f3eccfba` | #1510 | `deallocate_proceed` — fuse `deallocate + proceed` |
| `e4ce1ef7` | #1510 | `tail_call_5_c1_lit` — K=5 tail with literal first arg |
| `dcd6e83f` | #1513 | `deallocate_builtin_proceed` — `deallocate + builtin_call + proceed` |
| `8d1e16dd` | #1518 | `deallocate_<X>_direct_proceed` family — 5 variants for the direct-dispatch builtins |
| `dfa3f19c` | #1518 | `builtin_proceed` — no-deallocate variant |

Rationale: every clause ends with some form of
"clean up + return." Each clause in the bench predicates was
paying 2–7 separate dispatches for this sequence. The fused
instructions do the work in one `$step` iteration.

### First-argument indexing (type dispatch)

Dispatch directly on A1's runtime tag for multi-clause
predicates whose clauses are selected by the type of the first
argument (e.g., `term_depth/2`'s integer / atom / compound
clauses).

| Commit | PR | Change |
|---|---|---|
| `de324bb7` | #1520 | `type_dispatch_a1` — 3-clause variant for `term_depth/2` |
| `78fbbd00`, `4be16e1a` | #1524 | 2-clause variant (fires on `sum_ints/3`); `default_tgt` operand for total dispatch |
| `6a50e089` | #1526 | Drop the `try_me_else` / `retry_me_else` / `trust_me` chain when dispatch is total (dead code) |

Rationale: the WAM compiler doesn't emit first-argument
indexing for clauses with variable first arguments — even when
the body clearly selects on type via `integer/1` or `atom/1`
guards. A WAM-WAT peephole recognizes the pattern and emits a
tag-based dispatch at the predicate entry, routing directly to
the correct clause without the try/retry/trust chain.

### Choice-point elision (neck cut)

Replace the `try_me_else + guard + !/0` window with a
combined `neck_cut_test` instruction that evaluates the guard
inline and never pushes a choice point.

| Commit | PR | Change |
|---|---|---|
| `015292b1` | #1528 | Fix `peephole_neck_cut` — it had never fired because of a `label(Pred)` prefix mismatch (latent since PR #1399!) |
| `6aaaaf2f` | #1531 | Trail-aware variant — env frame gains a `trail_mark` slot; `neck_cut_test` unwinds the trail on guard failure (unlocks `term_depth_args/5`) |

Rationale: guard+cut patterns are cut-deterministic by
construction. The choice point that `try_me_else` pushes is
never used on the guard-success path, and its push/pop cost
dominates for tight recursive loops (`sum_ints_args/5`).

### Profile-guided investigation

| Commit | PR | Change |
|---|---|---|
| `9fc101b5` | #1533 | Inline `$fetch_instr_*` calls in `$step`; document findings |

Rationale: after the fusion work, profiling showed `$step`
dispatch at ~50% of hot-loop time and individual `$do_*`
handlers at <2% each. The inlining itself gave <1%
(V8 was already inlining small fetches); the main value was
confirming the cost distribution to inform future decisions.

### Documentation (post-optimization wrap-up)

| Commit | PR | Document |
|---|---|---|
| `76df57b9` | #1536 | [`WAM_WAT_ARCHITECTURE.md`](WAM_WAT_ARCHITECTURE.md) — design-level reference |
| `d247c665` | #1538 | [`docs/targets/wam-wat.md`](../targets/wam-wat.md) — user-facing target page; overview.md mention |
| `63ed08ce` | #1539 | [`docs/targets/comparison.md`](../targets/comparison.md) — WAM-WAT column in comparison matrix |
| `11654459` | #1543 | [`SHARED_WAM_INDEXING_PROPOSAL.md`](SHARED_WAM_INDEXING_PROPOSAL.md) — forward-looking proposal to share optimizations with WAM-family backends |

---

## Optimizations considered but not made

Four distinct reasons for not pursuing an optimization:

- **Low ceiling given profile**: the optimization would work
  but the profile shows it can't deliver a meaningful win.
- **Implementation cost disproportionate to ceiling**:
  technically viable but too much code growth / complexity
  relative to expected gain.
- **Requires structural/runtime change**: needs more than a
  peephole; scope beyond what was in hand.
- **Beyond the target's scope**: involves changes to shared
  code or infrastructure outside WAM-WAT.

### Heap-slab allocator (low ceiling)

**What it would do**: replace `$heap_push_val`'s per-call bump
allocation with batched allocation (e.g., allocate 256 cells
at once, amortize the counter traffic).

**Status**: not done.

**Why not**: profile (PR #1533) showed heap management at <1%
of `bench_sum_big` time. Even eliminating all counter traffic
caps at ~1%. V8 JIT likely already coalesces the
load/add/store pair. The ceiling is too low to justify even
a small implementation.

**When to revisit**: a workload with very high heap allocation
pressure (e.g., deep compound construction in a tight loop).

### Inline hot builtin bodies (partial — low remaining ceiling)

**What it would do**: inline the bodies of `$builtin_is`,
`$builtin_arith_cmp`, `$builtin_functor`, `$builtin_arg`
directly into the dispatch sites (rather than calling them via
a function-call boundary from `$execute_builtin` or `$do_*`).

**Status**: _partially_ done. Direct-dispatch wrappers
(`arg_direct`, `functor_direct`, `copy_term_direct`,
`univ_direct`, `is_list_direct` — PR #1463/#1479) skip
`$execute_builtin`'s br_table for these five common builtins.
Dispatch-fused clause ends
(`deallocate_arg_direct_proceed` etc. — PR #1518) further
collapse the dispatch with clause-end cleanup.

What was **not** done: inlining the _body_ of the called
builtin (e.g., having `$do_arg_direct` contain the arg/3
logic inline rather than calling `$builtin_arg`).

**Why not**: profile showed `$execute_builtin` at ~4% (and
that's inclusive of the builtin body). The function-call
boundary to the `$builtin_*` helper is a smaller fraction.
Inlining each builtin body would require per-site WAT case
variants — a large code-size impact for perhaps 2–3% total.

**When to revisit**: a workload heavy in builtins not in the
direct-dispatch set (e.g., `is/2` outside the fused-arith
peephole).

### 3-clause neck_cut for fib-shaped predicates (medium cost, narrow target)

**What it would do**: extend `peephole_neck_cut` to handle
3-clause predicates with `guard+cut`, `head-unify+cut`,
default — the shape of `fib/3`.

**Status**: not done.

**Why not**: the clean implementation requires new instruction
variants like `try_get_constant_else(C, Ai, FailLbl)` that
handle head-unification failure by jumping rather than
backtracking (~4 new WAT cases plus peephole rewrites). The
realistic ceiling is ~10% on `bench_fib10` — narrow impact
on one bench workload.

**When to revisit**: if `bench_fib10` becomes a priority or a
non-bench workload stresses this shape. Full cost-benefit
analysis is in the PR #1531 follow-up section.

### `$step` inlining of `$do_*` bodies (structural)

**What it would do**: inline each `$do_*` body directly into
its `br_table` case in `$step`, eliminating the indirect-call
overhead.

**Status**: not done.

**Why not**: substantial code-size explosion (`$step` would
grow from ~200 lines to 2000+). Uncertain V8 handling of a
very large function (JIT tiering behaviour, inliner budgets).
Profile shows `$step` at ~50% of time, so the ceiling is
attractive, but the implementation is structural — not a
peephole.

**When to revisit**: if profile re-analysis on a cooler host
shows that most of that 50% really _is_ the indirect call (as
opposed to the br_table dispatch, fetch, and bookkeeping).

### Shared indexing/neck_cut across WAM-family backends (cross-cutting)

**What it would do**: port the WAM-WAT peephole optimizations
up into the shared `wam_target.pl` layer so Go/Rust/LLVM/
ILAsm/Elixir/JVM backends can opt into them.

**Status**: proposed in PR #1543
([`SHARED_WAM_INDEXING_PROPOSAL.md`](SHARED_WAM_INDEXING_PROPOSAL.md)),
not executed.

**Why not now**: No sibling backend has near-term performance
pressure that motivated the port. The proposal captures the
plan so the work is ready to pick up when needed.

**When to revisit**: when a sibling WAM backend has a perf
concern that these techniques would address.

### Profile-guided investigation on non-Termux hardware

**What it would do**: re-run the profiler on a desktop Linux
host (or similar non-thermally-constrained environment) to
resolve sub-percent optimizations that are currently in noise.

**Status**: not done; Termux is what we have.

**Why not**: not in our hands. Termux has thermal variance
of ±1–3% per run, which obscures potential 0.5–1% wins from
smaller optimizations.

**When to revisit**: whenever a contributor with
non-thermally-constrained hardware picks up the thread.

### Cooler measurement + re-audit existing optimizations

**What it would do**: verify that every peephole we made is
actually earning its keep (vs adding dispatch overhead for a
pattern that only fires occasionally).

**Status**: not done.

**Why not**: no regressions observed; wouldn't likely surface
issues worth the audit cost given the cumulative result.

**When to revisit**: if adding a new peephole seems to not
improve the bench despite firing — that would suggest
investigation is needed.

---

## Cumulative impact

A final snapshot of the trajectory, for posterity:

| Milestone | `bench_sum_big` ratio |
|---|---|
| Phase 6 baseline (PR #1450) | 1.00× |
| Fused arith + direct dispatch (#1456, #1463) | ~0.87× |
| arg_call family + liveness (#1481–#1497) | ~0.69× |
| tail_call_5 + K-family extension (#1501, #1510) | ~0.58× |
| type_dispatch_a1 family (#1520, #1524, #1526) | ~0.52× |
| neck_cut_test fix + trail-aware (#1528, #1531) | ~0.45× |
| **Profiling investigation & wrap-up** | ~0.45× |

Total ≈ **−55% on `bench_sum_big`** vs Phase 6 baseline
(noting exact numbers fluctuate with thermal variance).

---

## Meta-observations

A few things worth remembering from this track that don't
belong in individual PRs:

### Dispatch becomes the bottleneck eventually

After ~20 fusion PRs, profile showed per-instruction `$do_*`
work at <2% each and dispatch (`$step` + br_table + indirect
call) at ~50%. The fusion work moved cost from per-instruction
execution into pure dispatch overhead. Further fusion has
diminishing returns once most remaining dispatches are
already fused windows.

### V8 does a lot for you

Several "obvious" micro-optimizations we considered (e.g.,
inlining the 3 fetch calls in `$step`) had <1% measured
impact because V8's JIT was already doing the inlining
invisibly. The lesson: at the WAT level, focus on things V8
can't discover (new instructions that bundle related work)
rather than things it can.

### Trust profile data over intuition

Several proposed optimizations we initially thought would
deliver 10% (heap-slab, builtin inlining) turned out to have
<5% ceilings once we ran the profiler. Measure before
implementing when you can.

### Some bugs hide for months

`peephole_neck_cut` was introduced in PR #1399 but never
actually fired until PR #1528 fixed a `label(Pred)` prefix
mismatch. Latent bugs in optimization passes are especially
sneaky because "no regression" is the silent failure mode.
Writing "did the peephole fire?" probes early in any new
peephole work is a cheap sanity check.

---

## See also

- [`WAM_WAT_ARCHITECTURE.md`](WAM_WAT_ARCHITECTURE.md) — the
  current-state design reference.
- [`docs/targets/wam-wat.md`](../targets/wam-wat.md) —
  user-facing target overview.
- [`WAM_TERM_BUILTINS_PHASE_6_PERF.md`](WAM_TERM_BUILTINS_PHASE_6_PERF.md)
  — the baseline state before this track.
- [`SHARED_WAM_INDEXING_PROPOSAL.md`](SHARED_WAM_INDEXING_PROPOSAL.md)
  — forward-looking plan to share these optimizations with
  sibling WAM-family backends.
