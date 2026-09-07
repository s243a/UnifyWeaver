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

## 6a. Soundness gates from external review (Kimi K2, 2026-09-06)

An adversarial design review (`docs/reports/wam_rust_stage2_kimi_soundness_review.md`)
found five load-bearing gaps, each with a concrete wrong-answer scenario. These
are **hard gates** for the Stage 2 build, not advisories; the scenarios are
regression targets. Several map onto machinery the runtime already has (the
`lowered_dispatch` post-call CP-depth check, the Arc-cloned `stack` on
`ChoicePoint`, the §9 cut-barrier and §8 ITE-levels models) — the review's job is
to make these provably sufficient for the nondet case and to add the two new
obligations (G-1, G-4/G-5).

- **G-1 — resume state is more than the tuple.** `minimal_saved_locals` must be
  computed by **backward** liveness from *every* `resume_arm`, and must **force
  all argument registers live** at every internal choice point (they are live for
  the *resumed* clause even if dead on the fall-through path). Plus frame pinning
  (G-5) and `trail_len`/`heap_len`.
- **G-2 — P1 needs a replacement guard.** Removing the interpreter round-trip
  removes the choice-point-leak detector. Replace with (a) a whole-program `det`
  lattice (poison on any non-`det` callee) and (b) a runtime post-call
  choice-point-depth guard at every lowered→interpreted/unknown boundary
  (commit only in a first-solution context, else deoptimize). The current
  `lowered_dispatch` guard is the first-solution form of (b) to generalize.
- **G-3 — cut across tiers.** One linear CP stack with uniform depth addressing;
  the cut barrier **passed caller→callee as an explicit value** (frameless
  lowered code has no frame slot to hold it); the barrier **saved in the CP at
  original-call time and restored on resumption, never recomputed** from the
  stack top.
- **G-4 — determinism ≠ non-unifiable heads.** Pairwise non-unifiable heads
  prove deterministic clause *selection*, not a single *solution* (a matched
  body can still be nondet, e.g. `member/3`). Dropping the CP requires a
  transitive whole-predicate **at-most-one-solution** proof; otherwise leave a
  resume-state CP unless the call site is explicitly first-solution. (Stage 1's
  F11 gate — non-unifiable heads + *pure body* — is a conservative special case;
  the general mechanism needs the full lattice.)
- **G-5 — activation identity + cut-invalidation.** Each resume-state CP carries
  a validated **activation handle** (frame pointer + generation) so nested
  self-calls do not confuse activations; and `cut` must **invalidate the
  predicate's own resume-state CP** at the barrier level, not only prune CPs
  above it.

The build proceeds only against the review's full soundness checklist, with the
seven concrete failure scenarios as explicit stress tests alongside the 2600/503
differential at 0 divergences.

## 6b. Committed-choice recursion is deterministic (classify before building 2b)

Refinement (project owner, 2026-09-07), applied to Stage 0 classification and to
every driver before it is treated as a 2b case: **a recursion whose body makes a
nondeterministic call is still deterministic overall if backtracking is
forbidden from re-entering that call past the iteration boundary** — i.e. the
per-step nondet call is *committed* before the recursive step (a `once/1`, a cut
that commits the choice ahead of the tail call, or an `->`/if-then-else that
discards alternatives before recursing). Such a recursion yields exactly one
solution and never re-enters an earlier iteration's nondet call on backtracking.

Consequence: it lowers with the **cheap deterministic mechanism** (native loop /
explicit-stack + P2 minimal snapshot, **no resume-state choice point**) — exactly
like regions 1–4 — and the hard gates are vacuous for it (G-5 activation identity
and G-4 multiplicity/order do not arise when no CP is left). So the classifier
must, for each apparent backtracking driver (`pick/7`, `blocked_from/4`,
`dep_breaks/5`), first decide: **(a)** committed-per-iteration → deterministic
path (preferred: safer and faster), or **(b)** genuinely exposes alternatives to
its caller → the resume-state trampoline. Prefer (a) wherever the commit/cut sits
before the recursive call.

**In-project reference:** PLAWK uses exactly this principle — a committed-choice
recursion compiled to a deterministic native loop despite containing a
nondet/meta call. See `examples/plawk/core/plawk_core.pl`,
`examples/plawk/codegen/plawk_native_codegen.pl`, the loop/meta-call probes under
`examples/plawk/probes/`, and `examples/plawk/TUTORIAL.md`. The deterministic-path
eligibility test mirrors PLAWK's: "the per-iteration nondet call is committed
(once/cut/->) before the tail recursion, so backtracking cannot cross the
iteration boundary."

A **second** deterministic pattern PLAWK uses (project owner, 2026-09-07):
**turn the nondeterministic part into an aggregation and iterate over that.**
An aggregation (`findall`/`bagof`/`setof`/`aggregate_all`) is itself
deterministic — it fully explores the nondet goal and returns exactly one list —
so `findall(X, NondetGoal, Xs)` followed by a deterministic iteration over `Xs`
is fully deterministic: the backtracking is contained at the aggregate boundary
and the hot outer structure is a deterministic fold with **no resume-state
choice point** (again lowering the cheap way; G-4/G-5 vacuous). The aggregate
call itself stays an interpreted builtin (or is lowered separately later); what
lowers deterministically is the iteration over its result. PLAWK's native
codegen is full of this shape — `findall(X, member(X, L), Xs)` and
`findall(..., ( member(N, Arities), ... ), ...)` in
`examples/plawk/codegen/plawk_native_codegen.pl` — aggregate the candidates,
then iterate the list.

So per-driver classification is **three-way**: (a) committed-per-iteration
recursion, (b) aggregate-then-iterate, (c) genuinely exposes alternatives to its
caller. (a) and (b) take the cheap deterministic path (preferred); only (c) needs
the resume-state trampoline. (b) applies only where a predicate ALREADY has the
aggregate-then-iterate shape — never by editing the frozen spec.

## 6c. The deterministic-recursion family is broader than tail recursion

Guidance (project owner, 2026-09-07): this project has **more deterministic
recursion patterns than tail recursion**, and the lowering recognizer must
target the whole family, not assume a tail-recursive loop. Patterns already
lowered here prove the point, and PLAWK carries more:

- **Tail recursion → native loop** (F11; the region 1/2/3a walks).
- **Non-tail recursion via explicit accumulator / stack** — region 3b
  (`group_keyed`, nested loops over a materialised run) and region 4
  (`build_tree`, bounded native recursion, balanced O(log N) depth). NOT loops.
- **Committed-choice recursion** — the per-step nondet call is committed
  (once/cut/->) before the tail (§6b; `dep_breaks/5`, region 5).
- **Aggregate-then-iterate** — the nondeterminism is bounded in
  findall/bagof/setof and the outer structure is a deterministic fold (§6b).
- **…and more the project actually uses** (project owner): **linear recursion**
  (one recursive call per clause, not necessarily in tail position),
  **transitive closures** (reachability/ancestor-style closure over a relation —
  deterministic when computed as a set / via aggregation or memoised to
  terminate), **tree recursion** (multiple recursive calls per clause — region 4
  `build_tree` is one), and **mutual recursion** (predicates that call each other
  recursively). These are to be enumerated precisely from PLAWK
  (`examples/plawk/`) and the resolver so the recognizer generalises to the
  family rather than special-casing each shape.

Design consequence: the eligibility recognizer is a **classifier over a family
of deterministic recursion shapes** (each with its native emission — loop,
explicit stack, committed-`->`, fold), with interpreter-decline for anything
outside it. A survey of the project's deterministic recursion patterns should
drive a general recognizer, target-agnostic so every backend's transpiler
applies it.

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

## 9. Generalization directive (2026-09-06)

Steering decision: **generalize as much as possible — do not special-case one
family, and do not cap the ambition at deterministic code.** The end goal is a
single *general* lowered execution model, not a bag of per-pattern hacks.

- **General deterministic class.** The project has many deterministic recursion
  patterns already identified — tail-recursive accessors (Stage 1's set),
  non-tail recursive walkers (`lookup_held`, `long_enough`, `scan_base_holds`),
  deterministic-in-practice bodies reached via `run()` (`matching_deps`,
  `matching_versions`, `direct_on`, `no_acc_conflicts`), and index builders
  (`same_key`, `build_tree`). The region-fusion mechanism (P1 direct calls + P2
  minimal snapshot) should be built to cover this general class, characterized
  against the census + the T1–T11 taxonomy, so the win reaches the whole
  deterministic dispatch share, not the ~26% of one family.

- **Nondeterministic code too (via F3).** mprolog's compiler handles nondet code
  (F3: a choice point is a saved resume label, a second solution is a *jump*,
  not a re-call). The census shows the true hot path is largely nondet (2b), so
  a deterministic-only tier leaves most of the lever on the table. The target is
  **one mechanism** whose choice point can carry a nondet resume state
  (resume-arm + minimal saved locals + clause index) — the same minimal-snapshot
  representation P2 requires, extended from "restore-and-fail" to
  "restore-and-resume-at-arm." Whether one mechanism cleanly covers both is a
  measured question, not an assumption.

- **Two cruxes, measured before the general build.** (1) The *deterministic*
  crux — does minimal-snapshot + direct-call beat the interpreter on a fused
  region? (spike 1, in flight). (2) The *nondet* crux — does a resume-state
  choice point beat the interpreter's own choice-point machinery per solution,
  against the 21 % backtrack + 12 % restore_regs baseline? (spike 2). The
  general build proceeds only when both cruxes read GO; a NO-GO on the nondet
  crux narrows the general build to the deterministic class rather than
  abandoning it.
