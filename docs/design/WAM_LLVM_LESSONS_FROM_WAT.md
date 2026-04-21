<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025-2026 John William Creighton (@s243a)
-->
# What the WAM-WAT track teaches us about the WAM-LLVM target

This document surveys lessons from the WAM-WAT optimization
thread (PRs #1456–#1549, captured in
[`WAM_WAT_OPTIMIZATION_HISTORY.md`](WAM_WAT_OPTIMIZATION_HISTORY.md))
and asks: which of them transfer to the hybrid LLVM/WAM target
(`wam_llvm_target.pl`)?

The short version: **a lot** of the pattern-recognition work
transfers (the peepholes are target-neutral), **some** of the
specific instruction fusions would duplicate LLVM's own
optimizer, and **none** of the WAT-specific runtime tricks
(env-frame trail_mark slot, br_table dispatch layout) apply
directly.

Actionable framing at the end.

---

## Current state — a gap analysis

| Dimension | WAM-WAT (after PR #1549) | WAM-LLVM (current) |
|---|---|---|
| Base WAM instruction cases | 31 | 31 (parity) |
| Optimization-specialized instructions | 42 (tags 31–72) | 0 |
| Peephole passes | 8 | 0 |
| First-arg type dispatch (guards) | Yes (`type_dispatch_a1`) | No |
| Neck-cut elision | Yes (trail-aware) | No |
| arg/3 + call fusion family | Yes (K=1, 2, 3 + liveness dead variants) | No |
| Tail-call fusion (`tail_call_5` etc.) | Yes | No — relies on `musttail` instead |
| Clause-end fusion (`deallocate_proceed` etc.) | Yes | No |
| Fused arithmetic (`fused_is_add` etc.) | Yes | No — relies on LLVM optimizer |
| Benchmark suite | `examples/wam_term_builtins_bench/` (13 workloads) | None |
| Profile-guided investigation | Yes (PR #1533) | None |

WAM-LLVM has feature parity with the base WAM instruction set
but none of the pipeline-level optimizations. Some of this is
deliberate — LLVM's optimizer does work that WAT has to do by
hand — but not all of it.

---

## What transfers directly

### Pipeline-level peepholes (pattern recognition)

The WAM-WAT peepholes operate on **WAM text / instruction terms
before encoding**. They recognize patterns like
`try_me_else + guard + cut → trust_me` or
`arg_to_a1 + call(Pred, K)`. These patterns exist in the WAM
stream **regardless of backend**; rewriting them is purely a
front-end job.

Candidates that transfer well to LLVM:

- **`peephole_type_dispatch`** (PR #1520/#1524/#1526) —
  Recognizes 2- and 3-clause predicates whose clauses are
  guarded by type tests (`integer/1`, `atom/1`, etc.) and emits
  a tag dispatch at the predicate entry. LLVM can handle the
  resulting dispatch via `switch i32`; no LLVM optimizer pass
  discovers this pattern on its own because it requires
  reading the clause guard, not just the head.
- **`peephole_neck_cut`** (PR #1528/#1531) — Recognizes the
  `try_me_else + guard + !/0` shape and emits a `neck_cut_test`
  that bypasses the choice-point push. The CP push is an
  explicit runtime operation; LLVM's optimizer has no way to
  infer that it's dead for this pattern.
- **`peephole_arg_call_k`** family (PRs #1492/#1495/#1497) —
  Fuses the common arg/3-then-call window that dominates
  `sum_ints_args/5` and `term_depth_args/5`. These patterns
  involve calls into runtime helpers (`$builtin_arg`) that
  LLVM can't see through to optimize further.
- **`peephole_tail_call_k`** (PRs #1501/#1510) — Fuses
  `put_value × N + deallocate + execute(Pred)` into a single
  tail-call instruction. On LLVM this might be _partially_
  subsumed by `musttail` (see "What's different" below) but
  the fusion still helps when multiple put_value sources come
  from the same Y slot.
- **`peephole_tail_call_k`**'s clause-end family
  (`deallocate_proceed`, `deallocate_builtin_proceed`, etc. —
  PRs #1513/#1518) — Collapse the `deallocate + builtin +
  proceed` shape at clause ends. LLVM can't fuse these across
  its runtime-helper boundary.

The [`SHARED_WAM_INDEXING_PROPOSAL.md`](SHARED_WAM_INDEXING_PROPOSAL.md)
PR already described moving these peepholes up to the shared
`wam_target.pl` layer with per-backend capability gates. The
WAT track is the motivating example; LLVM would be the first
test case for the shared design.

### The meta-lessons

From [`WAM_WAT_OPTIMIZATION_HISTORY.md`](WAM_WAT_OPTIMIZATION_HISTORY.md)'s
"Meta-observations" section — these are experience lessons,
not target-specific:

1. **Dispatch becomes the bottleneck eventually.** On WAT,
   once fusion reduced per-instruction work, dispatch (~50%)
   became the remaining target. LLVM's dispatch model is
   different (see below), but the phenomenon — per-instruction
   work shrinks as fusion progresses; dispatch cost rises in
   proportion — is general.
2. **JIT/optimizer does a lot for you.** V8 inlined small
   fetch functions; several WAT-level micro-optimizations
   had <1% impact. LLVM's optimizer is even more aggressive,
   which **cuts the other way**: some WAT peepholes (e.g.,
   `fused_is_add`) might be subsumed by LLVM's arithmetic
   optimizer.
3. **Trust profile data over intuition.** Multiple
   WAT-proposed optimizations had 10%-feel but ~3%-reality.
   Applies to LLVM too — maybe more so since LLVM's cost
   model is less predictable than V8's interpreter+tiering.
4. **Some bugs hide for months.** `peephole_neck_cut`
   didn't fire from PR #1399 to PR #1528 due to a
   `label(Pred)` prefix mismatch. If peepholes go into
   `wam_target.pl`, each backend needs a "did the peephole
   fire?" probe test early.

### Specific bug classes to watch for on LLVM

These came up during WAT work and would likely recur:

- **String vs atom label comparison** — `label` markers from
  the WAM text parser are strings; `try_me_else` operand
  references are atoms. Any peephole that matches labels
  needs `atom_string/2` normalization. (PR #1526 fix.)
- **Fail-closed `remove_label_trust_me`** — silently
  succeeding when the pattern isn't found lets the peephole
  fire on 3-clause predicates where it shouldn't. (PR #1528
  fix.)
- **Trail unwind on guard failure** — `neck_cut_test` needs
  to unwind trailed bindings from the pre-guard prelude, not
  just pop the env frame. (PR #1531 fix.) The LLVM runtime
  has its own trail; the env-frame layout is LLVM-specific,
  but the _requirement_ to unwind is universal.

### Benchmark infrastructure

WAT has 13 workloads with documented performance trajectory.
LLVM has **no benchmark suite** (grep for `bench_` in
`test_wam_llvm_target.pl` returns nothing).

Adding a suite parallel to `examples/wam_term_builtins_bench/`
is Phase 0 for any LLVM perf work. Without it, optimization
is blind.

---

## What's different — and won't transfer directly

### LLVM optimizer subsumes some peepholes

**Fused arithmetic** (`fused_is_add`, `fused_is_sub`,
`fused_is_mul`, `fused_is_add_const`, etc. — PRs #1456, #1463,
#1479) was a big WAT win because V8 can't see through the
interpreter dispatch. On LLVM, the arithmetic is expressed
directly as `i64` add/sub/mul after the switch dispatches; if
the generated IR exposes the arithmetic as direct ops rather
than routed through `@wam_builtin_is`, LLVM's optimizer
performs constant-folding, strength reduction, etc.
automatically.

**Lesson:** port the WAT fused-arith peephole to LLVM **if**
the current LLVM generation routes `is/2` through a runtime
helper that hides the arithmetic from the optimizer. Check
first; don't port blindly.

### `musttail` changes the tail-call landscape

WAT has an explicit `$run_loop` that repeatedly invokes
`$step`. `tail_call_5` (PR #1501) fuses the tail-call setup
because each separate instruction means another `$step`
iteration.

LLVM uses **`musttail`** for constant-stack tail calls — the
instruction stream is already compiled to native code,
tail-call setup is compiled instructions that the LLVM
backend reorders and schedules directly. `tail_call_5` might
be redundant here: LLVM's own tail-call optimization + register
allocation could achieve similar results.

**Lesson:** before porting `tail_call_5`, compare the LLVM IR
generated for a WAM-compiled `sum_ints_args/5` against
hand-written native LLVM. If the generated IR has 7 separate
instructions that LLVM collapses during its passes, porting
would help the front-end only (faster compilation) but not the
output. If the dispatch model _prevents_ LLVM from seeing the
tail-call chain as a single unit, porting would help runtime too.

### Dispatch model is fundamentally different

WAT uses a `br_table` on instruction tag in a single `$step`
function, with each case calling a `$do_<instr>` helper.

LLVM uses a `switch i32` on the tag, typically with
`musttail`-called case handlers that chain to the next
instruction. The cost model is different:

- WAT's br_table is O(1) but crosses the `$step` / `$do_*`
  function boundary per instruction (V8 mitigates via
  inlining).
- LLVM's switch is also O(1) but can be lowered to a jump
  table or balanced binary search; per-case handlers can be
  inlined by the LLVM optimizer.

**Lesson:** don't assume the 50%-dispatch-is-the-bottleneck
finding from WAT applies to LLVM. Profile first (`perf` on
Linux, `Instruments` on macOS — both work better than Termux
V8 for this).

### Runtime data structure differences

The trail-mark slot in WAT's env frame (PR #1531) is
WAT-specific storage. LLVM's env frame is a different layout
entirely (structs, not hand-packed bytes). The _abstract_
requirement (neck_cut must unwind the trail) ports; the
_concrete_ storage (add a 4-byte slot at offset +392) does not.

Same for the choice-point frame, heap layout, and register
file.

### Native lowering vs WAM fallback split

LLVM has a hybrid model that WAT doesn't: simpler predicates
can be **natively lowered** to direct LLVM IR (musttail,
typed i64, no tag dispatch), with WAM used only as the
fallback for predicates that resist native lowering.

Some WAT peepholes (fused_is_add, the direct-dispatch
builtins) may belong in the **native lowering** pipeline on
LLVM rather than the WAM interpreter path. This would be a
more significant refactor than "port the peephole"; it's a
rethink of where the optimization belongs in the LLVM
pipeline.

---

## Recommended starting points

Ordered from easiest/cheapest to largest/most-speculative.

### Phase 0: Benchmark suite

Port `examples/wam_term_builtins_bench/` to compile to LLVM
and run natively. Same 13 workloads, same correctness checks,
native executable instead of `.wasm`. Establishes a baseline
so all subsequent work has A/B numbers.

**Estimated cost**: ~1 PR. The workloads are pure Prolog; the
harness is a different driver.

### Phase 1: Profile the baseline

Run `perf record` (or equivalent) on the native executable
against the hot workloads. Produce a per-function breakdown
like the WAT profiling investigation (PR #1533). Expected
output: a clear identification of the dominant cost on LLVM
(likely different from WAT's dispatch-dominance finding).

**Estimated cost**: ~1 PR. Profile data + a writeup.

### Phase 2: First targeted port

Based on Phase 1 findings, pick **one** peephole to port:

- If WAM dispatch dominates → port `peephole_type_dispatch`
  (biggest WAT win for `term_depth/2` / `sum_ints/3`).
- If CP management dominates → port `peephole_neck_cut`
  (biggest WAT win for `sum_ints_args/5`).
- If clause-end dispatch dominates → port the
  `deallocate_proceed` / `deallocate_builtin_proceed` fusions.

Execute the port as a shared peephole per the
[`SHARED_WAM_INDEXING_PROPOSAL.md`](SHARED_WAM_INDEXING_PROPOSAL.md)
plan: move the peephole to `wam_target.pl`, add a capability
gate, implement the new instruction in the LLVM backend, and
measure.

**Estimated cost**: ~2–3 PRs. One for the shared peephole
refactor, one per peephole ported.

### Phase 3: Re-profile and iterate

After the first port, re-measure and decide what (if anything)
to port next. The WAT track took ~20 PRs to hit diminishing
returns; LLVM likely won't take that many because (a) LLVM
optimizer subsumes some of them and (b) the remaining ones
are higher-per-PR impact since they target explicit runtime
operations.

**Estimated cost**: open-ended, driven by profile data.

### Phase 4: Consider native-lowering promotions

For patterns where LLVM's optimizer could do the work if only
the IR was structured right, consider promoting the
optimization from the WAM interpreter path into the native
lowering path. Example: if a predicate's body is
`X is A + B` and both A and B are registers with known
integer provenance, emit `add i64` directly in native LLVM
rather than emitting `fused_is_add` (still via the WAM
interpreter).

**Estimated cost**: requires coordinated design work across
native lowering and WAM paths; not a pure peephole.

---

## What NOT to do (based on WAT experience)

### Don't port everything at once

The WAT track succeeded in part because each PR was tight
(one peephole, one measurement). Bulk-porting 8 peepholes to
LLVM in one PR makes regressions and per-peephole cost/benefit
analysis muddier.

### Don't skip the benchmark suite

WAT progress without measurement would have been impossible.
Multiple "felt like a win" optimizations turned out <1% or
negative. Don't guess; measure.

### Don't replicate env-frame layout decisions blindly

The 4-byte trail-mark slot added in PR #1531 was specific to
WAT's env frame. LLVM's env frame (a proper LLVM struct) has
a different layout; adding the trail mark there should
respect LLVM IR idioms, not port the byte offset.

### Don't try to out-compete LLVM's optimizer without measurement

Several WAT peepholes (fused arith, direct-dispatch builtins)
solved problems that LLVM's own passes can solve on their
own if the IR is structured right. Porting them can be
redundant — worse, it can make the output _harder_ for LLVM
to optimize further. Profile first to confirm the optimization
isn't already happening.

---

## Summary — what to port, what to skip

| Optimization | Port to LLVM? | Reason |
|---|---|---|
| `peephole_type_dispatch` | **Yes** | Guard-based dispatch is target-neutral; LLVM can't discover it. |
| `peephole_neck_cut` (trail-aware) | **Yes** | CP elision saves explicit runtime ops; concrete per-PR gain. |
| `peephole_arg_call_k` family | **Yes** (probably) | Arg/3 + call fusion spans runtime helpers LLVM can't see through. |
| `peephole_tail_call_k` (fixed-arity) | **Maybe** | Check if musttail + LLVM tail-call opt already does this. |
| `peephole_tail_call_k` (clause-end family) | **Yes** | deallocate + builtin + proceed spans runtime boundaries. |
| `peephole_fused_arith` | **Maybe** | LLVM likely handles arithmetic; verify before porting. |
| `peephole_direct_builtins` | **Maybe** | LLVM's switch lowering may subsume this. |
| `peephole_nested_arith` | **No** | LLVM optimizer handles nested arithmetic directly. |
| Env-frame trail-mark slot | **Concept only** | Semantics port; layout is backend-specific. |

---

## See also

- [`WAM_WAT_ARCHITECTURE.md`](WAM_WAT_ARCHITECTURE.md) —
  current-state reference for the WAT target these lessons
  come from.
- [`WAM_WAT_OPTIMIZATION_HISTORY.md`](WAM_WAT_OPTIMIZATION_HISTORY.md) —
  the full optimization track with commit references.
- [`SHARED_WAM_INDEXING_PROPOSAL.md`](SHARED_WAM_INDEXING_PROPOSAL.md) —
  proposal for moving peepholes to shared `wam_target.pl`,
  which LLVM would be the first consumer of.
- [`WAM_LLVM_TRANSPILATION_PHILOSOPHY.md`](WAM_LLVM_TRANSPILATION_PHILOSOPHY.md) —
  the LLVM target's design principles, which constrain what
  _can_ port.
- [`WAM_LLVM_TRANSPILATION_IMPLEMENTATION_PLAN.md`](WAM_LLVM_TRANSPILATION_IMPLEMENTATION_PLAN.md) —
  the LLVM target's existing implementation roadmap.
