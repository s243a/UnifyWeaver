<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust lowered tier (Stage 2) — external soundness review

**Reviewer:** Kimi K2 (external, routed via the maintainer). **Date:** 2026-09-06.
**Subject:** the resumable native-lowering design in
`docs/proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md` (§5 Stage 2, §9
generalization directive). **Posture:** adversarial — every invariant treated as
guilty until proven innocent, every optimization a potential bug until a
concrete failure mode is ruled out.

This is a design review (no code existed yet). The five gaps below are recorded
as **hard soundness gates** for the Stage 2 build; the concrete failure
scenarios are the regression targets. Reproduced faithfully; wording is the
reviewer's.

## Verdict

The core insight — amortizing snapshot cost over larger native regions via
M-Prolog-style resumption — is sound in theory. The sketch under-specifies the
live set (omits argument registers and frame pinning), lacks the static and
dynamic guards P1 needs, and omits the activation identity and cut-invalidation
that M-Prolog uses to keep compiled nondet code correct. Each omission has a
concrete wrong-answer failure mode (not merely a crash). Fix the five gaps and
the design is ready to build.

## The five gaps

### G-1 — the resume tuple `(resume_arm, minimal_saved_locals, clause_index)` is insufficient
Resumption jumps into the *middle* of a compiled predicate, so the resumed
clause still needs the **original argument registers** — but a forward/liveness
pass from the fall-through path may mark them dead and drop them. The minimal
live set must be computed by **backward** liveness from *every* `resume_arm`,
and must **force all argument registers live** at every internal choice point.
It must also pin the **activation frame** (see G-5) and account for `trail_len`,
`heap_len`, and the cut barrier.
- *Failure:* `p(f(A),B):-q(A),r(B). p(g(C),D):-s(C,D).` — clause 1 overwrites the
  arg register holding `f(A)`; on retry into clause 2 the head unifies against
  garbage. Wrong answer.

### G-2 — P1 (direct native calls) removes the interpreter's choice-point-leak guard
Today the interpreter round-trip is what detects a callee that unexpectedly left
a choice point; direct calls remove it. Replace it with **both**: (a) a
whole-program `det` lattice (a predicate is `det` only if clause heads are
pairwise non-unifiable AND every body subgoal is `det` AND no unwrapped
meta-call of a non-`det` goal — poison on any non-`det` callee), and (b) a
**runtime post-call guard** at every lowered→interpreted / lowered→unknown
boundary: check the choice-point depth against expected; if the callee left a
CP, either commit (only in a `once`/first-solution context) or deoptimize to an
interpreter stub. Static analysis alone is insufficient for dynamic `call/1`.
- *Failure:* a `det`-compiled `p(X):-q(X),r(X)` calling an actually-nondet `q`;
  later backtracking resumes `p` after its frame is gone. Wrong answer / crash.

### G-3 — cut across the lowered/interpreted boundary
Correct **iff** (a) all CP types share one linear stack with uniform depth
addressing, (b) the cut barrier is passed **caller→callee as an explicit value**
(never looked up from a frame the frameless lowered predicate does not have),
and (c) the barrier is **saved in the CP at original-call time and restored on
resumption, never recomputed from the current stack top**.
- *Failures:* frameless lowered `p` calls interpreted `q`; `q`'s `!` reads a
  stale barrier and over-prunes → missing solutions. And `p:-a,!,b. p:-c.` where
  resumption reinitializes the barrier → `!` fails to commit → duplicate `c`.

### G-4 — the determinism decision: pairwise non-unifiable heads is necessary but NOT sufficient
It proves deterministic *clause selection*, not a single *solution*: a matched
clause body can still be nondet (`p(a,X):-member(X,[1,2,3]).`). The correct
minimal condition to drop the CP is a transitive, whole-predicate
**at-most-one-solution** proof (clause selection deterministic AND every body
subgoal single-solution AND no implicit nondet builtins). Otherwise the call
must leave a resume-state CP unless the call site is explicitly first-solution
(`once`, deterministic `->`).
- *Failure:* a single-clause `q(X):-member(X,[1,2])` wrongly marked `det`;
  committing after `X=1` loses `X=2`. Missing solution.

### G-5 — M-Prolog mechanisms omitted: activation identity + cut-invalidation
(a) Each resume-state CP must carry an **activation handle** (frame pointer +
generation/depth) that uniquely identifies the activation and is validated on
resumption, so nested self-calls do not confuse an inner activation's CP with an
outer one's (M-Prolog's per-predicate activation-depth counter). (b) `cut` must
**invalidate the current predicate's own resume-state CP** at the barrier level
(disabled flag, or overwrite `resume_arm` to a fail label, or sentinel
`clause_index`) — pruning CPs *above* the barrier is not enough.
- *Failures:* recursive lowered nondet predicate resumes into the wrong
  activation's locals → crash/wrong binding; `p:-a,!,b. p:-c.` where the
  predicate's own CP survives `!` → duplicate solution.

## Minimal soundness checklist (must pass before landing)

**Liveness & state:** backward liveness from every `resume_arm` with all
argument registers forced live; every CP pins its activation frame (LCO disabled
while a CP references the frame); heap cells reachable from saved locals proven
allocated before the CP's `heap_len`.

**Determinism & guards:** whole-program `det` lattice; runtime post-call
choice-point-depth guard at every lowered→interpreted/unknown boundary; no
`det`-compiled predicate contains an unanalyzable `call/1` unless wrapped in
`once`.

**Cut & barriers:** one linear CP stack, uniform depth numbering; barrier passed
as an explicit parameter across every tier boundary; cut invalidates the
resume-state CP at the barrier level; resumption restores the original barrier
from the CP, never recomputes it.

**Nondet & resumption:** every resume-state CP carries a validated activation
handle; nested self-calls get distinct handles; clause/subgoal retry order
matches the interpreter exactly (no reordering for optimization).

**Testing:** the full differential (2,600 term + 503 store) at 0 divergences;
explicit stress tests for cut across 3+ mixed tiers, recursive lowered nondet,
`once(call(nondet))` boundary, LCO with an active CP; fuzz programs with nested
`!`, `-> ;`, and self-calls vs SWI at 0 divergences.

## Bearing on our runtime

Several gates map onto machinery the Rust runtime already has, which lowers the
risk: the existing sound-intermediate `lowered_dispatch` already does a
post-call choice-point-depth check (the G-2 runtime guard, in first-solution
form); `ChoicePoint` already Arc-clones the environment `stack` (a form of the
G-5 activation snapshot) and records `trail_len`/`heap_len`; the §9 cut-barrier
model and the §8 ITE-levels-on-the-CP fix already exist (the G-3 substrate). The
review's net effect is to convert these from "present" to "provably sufficient
for the nondet case," and to add the two genuinely new obligations: **forced
argument-register liveness at resume points (G-1)** and the **whole-program
`det` lattice + cut-invalidation of the own CP (G-4/G-5)**.
