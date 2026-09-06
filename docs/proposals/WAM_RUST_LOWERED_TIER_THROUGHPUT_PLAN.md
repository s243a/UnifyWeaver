<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — the throughput plan (beat SWI on B2)

Status: **Proposed** (2026-09-06). Scope: the Rust WAM target only. Goal: close
the raw-throughput gap vs SWI-Prolog on the uw-resolve differential (B2), where
every transpiled target — Rust included at ~8.8× SWI — still trails. The
recorded profile says the lever is **fewer WAM dispatches**, not faster
dispatch (Phase-K verdict, `WAM_PERF_OPTIMIZATION_LOG.md`; profile ~65 %
step-dispatch, ~21 % backtrack, ~12 % restore_regs, `WAM_RUST_STATUS.md`), i.e.
lowering more predicates to native code.

This plan is **staged deliberately**. The hard win (a resumable native tier)
was deferred twice for concrete reasons; the groundwork records a lower-risk
step that must come first. Every stage keeps the full gate set green — contract
corpus 51/51, term differential 2600/0, store differential 503/0 — and is
justified by measurement, not by expected speedup. No stage touches
`resolver.pl` or `resolver_store.pl` (frozen spec).

## 1. What already exists (and why it is neutral today)

The Rust target has a lowered emitter (`wam_rust_lowered_emitter.pl`) that emits
`pub fn lowered_<name>_<arity>(vm: &mut WamState) -> bool` for deterministic /
multi-clause-n / clause-chain / ITE predicates, dispatched at the two
`crate::lowered_call(...)` sites in the `Call`/`Execute` arms. It is the "sound
intermediate" (landed 2026-09-05) and it is **performance-neutral**, for one
structural reason: every lowered function returns `bool` (first solution only)
and lowered-to-lowered calls route through the interpreter `run()`. So dispatch
is restricted by a greatest-fixpoint to predicates whose every transitive callee
is choice-point-clean; the hot code — the multi-clause recursion drivers — is
nondet and cannot be dispatched. The win is bounded by how much work sits in
cp-clean leaves, which for the resolver is little.

## 2. The two recorded obstructions to a resumable tier (2b)

From `WAM_RUST_STATUS.md` (item 2b, the deferral write-up). A resumable tier —
the mprolog **F3** idea: a choice point is a saved resume point, a second
solution is a *jump* not a re-call; the portable analogue is a state-machine
enum driven by a trampoline, resumed by restoring saved locals and matching to
the right arm — was deferred, not for budget alone, but because two things block
it:

- **O1 — the cross-call convention.** Lowered functions today call each other
  through `run()`, which is exactly what forces the cp-clean restriction. A
  resumable tier must make lowered→lowered calls **direct** (a resume-entry, not
  `run()`), or nondet callees keep leaking choice points into their callers.
  (mprolog is the negative example: zero-overhead nondet→nondet via `goto`, but
  full dispatch cost on the nondet→det crossing.)
- **O2 — the snapshot cost.** The sound intermediate already shows the soundness
  snapshot (`save_regs`, a full register-file clone) costs about what dispatch
  saves. A resumable protocol pushes a choice point **per re-entry**; unless the
  saved-locals set is much smaller than the whole register file, per-solution
  cost will not beat the interpreter's own choice-point machinery (the very 21 %
  backtrack + 12 % restore_regs being measured). 2b must **measure per-solution
  cost against that baseline before it lands.**

Neither obstruction is cleared today. Charging straight at 2b risks repeating
the neutral result at higher risk. Hence the staging below.

## 3. Stage 0 — census (decides everything, cheap)

Before writing any lowering code, profile the B2/differential workload and
classify the hot predicates (top contributors to `step` count and `backtrack`
count) into:

- **F11-eligible** — single recursive clause, recursion in tail position,
  `independ_head` (head has no repeated argument variable), deterministic body.
  These compile to a native loop with **zero** choice-point machinery (Stage 1).
- **2b-only** — genuinely nondet multi-clause recursion drivers (real
  backtracking across solutions). These need the resumable protocol (Stage 2)
  or stay interpreted.
- **already-lowered / leaf** — covered by the sound intermediate.

Deliverable: a table of the hot predicates with their class and their share of
B2 `step`/`backtrack`. **Decision gate:** if a large share of the hot path is
F11-eligible, Stage 1 alone moves B2 and is worth doing now. If the hot path is
almost entirely 2b-only, Stage 1 is a small safe bank and the real lever is
Stage 2 — which then must clear O1+O2 first. Either way the census is decisive
and must be reported before implementation proceeds past Stage 1's eligible set.

## 4. Stage 1 — F11: self-tail-recursion → native loop (lowest risk)

Recorded as the lowest-risk next lowered-tier step (`WAM_RUST_STATUS.md` Q2;
mprolog **F11**, `gen_tail_pred` family). A predicate that is a single
deterministic recursive clause with the recursive call in tail position and an
independent head compiles to a `loop { }` inside its lowered function: bind the
head, run the body, on the tail self-call rebind the argument registers and
`continue` instead of pushing an environment and re-dispatching. No choice
point, no `run()` round-trip, no `Allocate`/`Deallocate` per iteration.

- **Why it dodges O1+O2:** it is deterministic (no resumable protocol, no
  per-re-entry choice point → O2 does not arise) and it does not make a
  nondet lowered→lowered call (the loop is self-contained → O1 does not arise).
- **Correctness:** gated behind a new eligibility predicate
  (`independ_head`-style check + "single recursive clause, recursion in tail
  position"), with interpreter fallback for anything that does not qualify —
  exactly the decline-if-unsure discipline of `wam_rust_lowerable/3`. Must
  preserve the trail (bindings trail-recorded, truncated on the loop's own
  failure path), cut barriers, and Y-register frame discipline (§5).
- **Measure:** B1/B2/B3 and every gate, before/after, on this box. Report the
  B2 ratio vs SWI. Bank it if it helps; it is safe either way.

## 5. Stage 2 — 2b resumable trampoline (conditional; explicit approval)

Only after Stage 1 is measured, and only with an explicit go-ahead, because it
is the twice-obstructed hard part. It must not start until both prerequisites
are designed:

- **P1 (clears O1):** a direct lowered→lowered resume entry — not `vm.run()`.
  The emitter's `emit_one(call)`/`emit_one(execute)` interpreter round-trip is
  replaced on the lowered→lowered edge by a direct call into the callee's resume
  entry.
- **P2 (clears O2):** a **minimal saved-locals** representation per predicate
  (the live set at each resume point), not a `save_regs` full-register clone.
  The resume state rides on `ChoicePoint` (extend `builtin_state` or add a
  `lowered_state` field carrying `(resume-arm, saved live locals, clause
  index)`); `backtrack()`'s resume branch re-enters the state machine at the
  saved arm — copying the two working CP-leaving precedents already in the
  runtime: `builtin_state`/`resume_builtin` and T9's `fact_table_attempt`.

**Hard requirement:** prototype-and-measure per-solution cost against the
interpreter's 21 % backtrack + 12 % restore_regs baseline **before** landing.
If per-solution cost does not beat that baseline, 2b does not land — the same
prove-then-commit rule that governed the H4/H1 and store rounds.

Keep the dispatch loop small: the LLVM hybrid found that inlining the whole
state machine regressed 3–5 % from icache pressure (`WAM_PERF_OPTIMIZATION_LOG`).

## 6. Correctness invariants (all stages)

From the runtime integration surface. Any lowered body must preserve, exactly as
`backtrack()` does:

- **Trail:** every binding written through the `bindings` table and
  trail-recorded; on any failure/re-entry, `trail` truncated to the choice
  point's `trail_len` and `heap` to `heap_len`. Mind the Y-register-trail-entry
  hazard that `call_goal_once` works around.
- **Choice-point floor:** never pop/resume a choice point at or below
  `backtrack_floor` (D71); a first-solution meta-call context (`call_goal_once`,
  `call/1`) raises it, and a lowered call reached inside one must respect it.
- **Cut barriers (§9):** honor `cut_barrier` and `pending_cut_barrier`; a `!`
  prunes choice points back to the barrier, never to zero (mprolog F3's
  `rp[th]=P_Arp` is the same rule). ITE barrier levels ride on
  `ChoicePoint.levels`, never a Y-register (the §8 frameless-Y hazard).
- **Y-registers:** live in the topmost `StackEntry::Env` frame; a lowered
  predicate that allocates an environment saves/restores the Arc-cloned `stack`
  on its choice point and must not corrupt a caller's Y-frame on unwind.
- **Sentinels:** `pc==0` = halt; `cp==0` = top-level return; nested runs halt at
  the top-level sentinel (the `lowered_dispatch` `cp=0` discipline).
- **Multiplicity / order / side-effects:** solutions in the same order and
  multiplicity as the interpreter (validated against SWI, sequence-oracle
  style); only side-effect-free builtins are safe on any path that can roll
  back (`rust_dispatch_pure_builtin` whitelist).

## 7. Attribution

The loop shape (F11) and the resumable-choice-point shape (F3) are **ideas**
adopted from Kenichi Sasagawa's M-Prolog / N-Prolog
(`https://github.com/sasagawa888/mprolog`, Modified BSD), never code, per
`MPROLOG_MINING_NOTES.md`. Any change that adapts logic recognizably derived
from mprolog source must carry the Modified BSD notice and attribution
alongside it. The portable state-machine/trampoline analogue is the standard
async/await / Scheme-to-C CPS transform and is not mprolog-specific.

## 8. Sequencing

1. **Stage 0 census** → report the eligibility table (decision gate).
2. **Stage 1 (F11)** → implement for the eligible set, measure, bank if it helps.
3. **Stage 2 (2b)** → only on explicit approval after Stage 1's numbers, and
   only with P1+P2 designed and the per-solution cost proven against baseline.

Ledger each landed stage in `JS_TARGETS_PARITY_PUNCHLIST.md`.
