# T7 embedded aggregates — codegen+dispatch wiring (design)

**Status:** design / spec / implementation plan. Prerequisite landed:
`lift_embedded_aggregate/6` (PR #3141, on `main`) — the pure, proven source
transform. This doc covers the remaining *invasive* part: applying the lift
during compilation and making the lifted helper WAM-callable. Companion docs:
`wam_rust_t7_RESUME.md`, `wam_rust_t7_2b2_fork_analysis.md`, `wam_rust_t7_NEXT.md`.

---

## 1. Philosophy

**Problem.** The whole-body parallel-aggregate path (on `main`) fires only when a
predicate's *entire* body is a forkable aggregate. Real programs usually embed an
aggregate inside a larger clause:
`p(X,R) :- setup(X,Y), findall(D, Inner, R), use(R).`
Such aggregates currently compile sequentially.

**Design principles** (in priority order):

1. **Reuse, don't reinvent.** The whole-body machinery (cost gate → split →
   transform → `par_collect` wrapper, all reduce types, exec-tested, 3.39×) is
   correct and proven. Embedded support should *funnel into it*, not duplicate it.
2. **No new WAM instruction.** The earlier `par_aggregate`-instruction design
   (fork_analysis.md) works but means new bytecode + parser + interpreter arm +
   WAM-text surgery. The lift-to-helper approach avoids all of that: an embedded
   aggregate becomes a *whole-body helper predicate* + a call — i.e. it reduces
   the new problem to the already-solved one.
3. **The call must cross WAM→native.** A lifted helper compiles to a native
   `par_collect` wrapper (no bytecode). The enclosing clause is WAM. The bridge
   already exists: the `Call` instruction dispatches to `execute_foreign_predicate`
   when the callee is in `self.foreign_predicates`. So register the helper as
   foreign — no interpreter change to `Call` itself.
4. **Never corrupt the user's program.** Clause rewriting for compilation must not
   mutate the user's module destructively. Work on copies / a scratch module.
5. **Gated + reversible.** Behind `parallel_aggregates(true)`; off ⇒ byte-identical
   output, as every prior T7 slice.
6. **Correctness first, then speed.** The lift is already proven result-preserving;
   the wiring must preserve that (exec test: embedded `findall` == sequential).

---

## 2. Specification

### 2.1 What is lifted
For each clause `H :- B` being compiled (with the gate on): if
`lift_embedded_aggregate(H, B, Model, Seed, NewBody, Helper)` succeeds, the clause
is compiled as `H :- NewBody` and `Helper` is added to the compile set. `Seed`
must be unique per lifted site (e.g. `Pred_Arity_Index`).

### 2.2 The helper contract
`Helper = (__lift_agg_Seed(Ins..., R) :- Aggregate)` is a whole-body aggregate
predicate. It MUST be handled by the existing `rust_inject_parallel_aggregates`
(which turns it into a native `par_collect` wrapper + `__par_enum`/`__par_body`
sub-helpers). I.e. lifting produces input for the path already on `main`.

### 2.3 WAM-callability (the crux)
`__lift_agg_Seed/(n+1)` must be callable from the enclosing clause's WAM:
- register its `name/arity` string in `self.foreign_predicates` (so `Call`
  dispatches to `execute_foreign_predicate`);
- `execute_foreign_predicate(pred, arity)` must, for that pred, read `A1..An`
  (the inputs) + `A(n+1)` (the result var) from registers and invoke the native
  wrapper `crate::__lift_agg_Seed_k(self, a1..an, a_{n+1})`, returning its bool.
  The wrapper already does par_collect + reduce + unify-result; the foreign arm
  is a thin shim.

### 2.4 Semantics
The lifted call must be observationally identical to the original embedded
aggregate: same bindings to `R`, deterministic (one solution), no effect on the
clause's other goals. (The lift is proven result-preserving; the wiring must not
change argument order, miss an input var, or mis-deref the result.)

### 2.5 Gating / decline
- Off unless `parallel_aggregates(true)`.
- Decline (compile clause unchanged) when: no embedded eligible aggregate; the
  aggregate is the whole body (handled by the existing path); the body has a cut
  spanning the aggregate (soundness — `lift` already refuses cut/control/side-
  effects inside the *aggregate*, but a cut elsewhere in the clause that the
  aggregate is control-dependent on should also decline — see edge cases).

### 2.6 Edge cases
- **Multiple embedded aggregates in one clause:** lift iteratively (each with a
  distinct Seed) until none remain; each becomes its own helper.
- **Cut in the enclosing clause:** if a `!` precedes the aggregate, the lift is
  still sound (the aggregate runs after the cut commits); if the aggregate is
  inside an `( _ -> _ ; _ )` arm, decline for v1 (the lift only handles
  conjunction-level embedding).
- **Result var also an input** (aggregate's R reused as an input to a later goal):
  R is the output arg; later goals see it bound — unchanged.
- **Aggregate whose inputs include a var bound *after* it** (mode violation):
  cannot happen for a well-moded clause; if `Ins` contains an unbound var at call
  time, the wrapper enumerates over an unbound — same as the sequential findall
  would, so behaviourally consistent (just not parallel-profitable).

---

## 3. Implementation plan

Work in `wam_rust_target.pl` (+ small `parallel_gate.pl` reuse). Test-first
against an exec harness.

### Step A — apply the lift during compilation (no DB corruption)
In `compile_predicates_for_project` (already runs `rust_inject_parallel_aggregates`
as Pass 0), add a Pass -1 that, for each user predicate with the gate on:
1. read its clauses (`clause/2`);
2. for each clause, repeatedly apply `lift_embedded_aggregate` until it declines,
   collecting `Helper` clauses and the rewritten body;
3. assert the rewritten clauses + helpers into a **scratch module** (e.g.
   `gensym`-named, or retract-on-cleanup), and compile the predicate *from the
   scratch module* — never mutate the user's module. (Mirror how
   `rust_assert_helper` already asserts whole-body helpers, but scoped + cleaned
   up via `setup_call_cleanup`.)
4. add the lifted helper PIs to the predicate compile set so Pass 0 turns them
   into `par_collect` wrappers.

Simplest robust form: build the full set of (rewritten-clauses + helpers) up
front into a scratch module, point all subsequent passes at it, `retractall` on
cleanup.

### Step B — register lifted helpers as foreign
- Collect the lifted helper PIs; emit them into `setup_foreign_predicates`
  (`self.foreign_predicates.insert("__lift_agg_Seed/N")`) and into
  `foreign_pred_keys()`.
- Make `execute_foreign_predicate` extensible: today it's a fixed match
  (`compile_execute_foreign_predicate_to_rust`). Add a generated arm per lifted
  helper:
  ```rust
  "__lift_agg_Seed/N" => {
      let a1 = self.get_reg("A1").unwrap_or(Value::Uninit);
      // ... aN, a_{n+1}
      crate::__lift_agg_Seed_k(self, a1, /*…*/ a_np1)
  }
  ```
  Generate these arms from the lifted-helper list and splice into the function
  (make the template take an injected-arms parameter).

### Step C — exec test (acceptance)
`tests/test_wam_rust_embedded_exec.pl` (cargo-gated): a predicate with an
*embedded* `findall` over a recursive body + goals before/after; build, run,
assert the result == the sequential answer. Pattern:
`tests/test_wam_rust_parallel_injection_exec.pl`. Add a `parallel_aggregates(false)`
control proving default output is unchanged.

### Step D — regression + matrix
- Full `test_wam_rust_target.pl` must stay green (gate-off ⇒ no change).
- Update the matrix note: rust T7 `~` scope widens from "whole-body" to
  "whole-body + embedded".

### Risk register
- **Highest:** clause-rewrite-during-compile leaking into the user's module →
  use a scratch module + `setup_call_cleanup`, and a test that the user's module
  is unchanged after compilation.
- **Medium:** `execute_foreign_predicate` arm generation (string splicing into a
  large generated fn) — keep arms tiny (delegate to the wrapper).
- **Low:** arg/register read order in the foreign arm — covered by the exec test.

### Order of work (each independently committable + tested)
1. Step A as a *pure* pass returning (rewritten-clauses, helper-clauses, foreign-PIs)
   without yet emitting foreign arms — unit-test it produces the right set and the
   user module is untouched.
2. Step B foreign registration + dispatch arms — cargo-check a generated project.
3. Step C exec test — embedded == sequential.
4. Step D regression + matrix.

---

## 4. Concrete seams (implementation-ready — from investigation)

Step 1 (the pure clause-lifting pass `rust_lift_predicate_clauses/4`) is **done +
proven** (PR #3156). The remaining integration has these exact seams:

- **Rewritten-clause compilation (no module mutation):**
  `compile_predicate_to_wam_text/3` is `wam_predicate_clauses/5` +
  `compile_clauses_to_wam(Pred, Arity, Clauses, Options, Code)`. So a lifted
  predicate's WAM is `compile_clauses_to_wam(Pred, Arity, RewrittenClauses, …)` —
  compiled directly from the lifted clause list, **no need to touch the module**.
- **WamCode seam:** `classify_predicates/3` (`wam_rust_target.pl`) acquires per-pred
  WAM at `wam_target:compile_predicate_to_wam(Module:Pred/Arity, WamOptions,
  WamCode)` (~line 5617). For a lifted predicate, substitute `WamCode` with the
  rewritten-clause compile above (e.g. via a `lifted_wamcode(PI, Code)` entry in
  `Options` populated by a Pass -1, checked before the `compile_predicate_to_wam`
  call).
- **Cost-model correctness:** lifted helpers must be analysed against the **full
  program** — so assert helper clauses into the **user module** (not a scratch
  module), so `rust_inject_parallel_aggregates`' `build_cost_model(user)` sees
  `ew_down` etc. and classifies the helper parallel. (Step 1's pass is read-only;
  the integration adds a guarded assert+cleanup of helpers only.)
- **Foreign dispatch:** `execute_foreign_predicate` (`compile_execute_foreign_predicate_to_rust`,
  ~line 1798) is a fixed `match native_kind { … }`. `Call` already routes any
  `pred ∈ foreign_predicates` here (line ~669). Add a **conditional** arm
  `"lift_agg" => match pred_key { "__lift_agg_K/N" => { read A1..A_{n+1};
  crate::__lift_agg_K_n(self, …) } … }`, generated only when ≥1 lifted helper
  exists (so gate-off projects get no new arm ⇒ **zero blast radius**). Register
  each helper in `setup_foreign_predicates` (`foreign_predicates.insert("__lift_agg_K/N")`
  + `foreign_native_kinds.insert(key, "lift_agg")`) and `foreign_pred_keys()`.
- **Wrapper reuse:** the lifted helper, added to the compile set, is turned into a
  native `par_collect` wrapper by the **existing** `rust_inject_parallel_aggregates`
  / `rust_parallel_aggregate_wrapper` — no new emission needed; the foreign arm
  just calls that wrapper.

**Assessment:** mechanically clear, gated to near-zero blast radius, but a
multi-part change to the core compile pipeline + the foreign-dispatch template,
with cargo exec/regression cycles between steps. Best executed as a focused,
undistracted pass (Step-1 — the subtle-correctness half — is already banked).

## 5. Progress + the 2b dispatch wrinkle (live)

- **Step 1** (pure clause-lifting pass `rust_lift_predicate_clauses/4`) — DONE,
  proven (PR #3156).
- **Step 2a** (live Pass -1 `rust_apply_embedded_lifts` + `lifted_wam` WamCode
  substitution + helper-into-user assert + Pass-0 wrapping) — DONE, verified: a
  gate-on embedded project compiles (`eb_pred` Executes `__lift_agg_…`, whose
  `par_collect` sub-helpers are in the shared table), and the gate-off regression
  stays green. The lifted helper's `Execute` does **not** yet dispatch at runtime.
- **Step 2b** (runtime dispatch) — REMAINING. Structural wrinkle found: in
  `write_wam_rust_project`, `compile_wam_runtime_to_rust` (which emits
  `execute_foreign_predicate`) and `generate_setup_foreign_predicates_rust` run
  **before** `compile_predicates_for_project` (where Pass -1 discovers the
  helpers). So the helper list must be discovered **before** runtime-gen.

  **Plan for 2b:**
  1. Hoist lift *discovery* to the top of `write_wam_rust_project` (compute the
     lifted-helper list + `lifted_wam` subs + assert helpers once; cleanup at the
     end). `compile_predicates_for_project` then *consumes* the precomputed subs
     instead of re-lifting.
  2. Thread the lifted-helper list (each: foreign key, the `__par_enum`/`__par_body`
     **labels** already in the shared table, agg type, input arity, result reg)
     into both `generate_setup_foreign_predicates_rust` (register
     `foreign_predicates` + `foreign_native_kinds="lift_agg"` + the labels/type as
     `foreign_*_configs`) and `compile_execute_foreign_predicate_to_rust`.
  3. Add a **generic** `"lift_agg"` arm to `execute_foreign_predicate` (data-driven
     from the configs — not per-helper codegen) that runs a new label-based
     `par_collect_labels(self, enum_pc, body_pc)` in `par_aggregate.rs` (clone,
     set pc + A1=Unbound, `run()`, deref-collect, `backtrack()+run()` loop; per
     input clone + body_pc), reduces by agg type, and unifies the result arg.
  4. Exec test: `eb_pred(X,R)` (embedded findall) builds + runs == sequential.

  This is the final increment; everything it needs (the sub-helper labels, the
  reduce logic, the gate) already exists.
