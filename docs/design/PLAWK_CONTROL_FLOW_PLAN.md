<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk control-flow runtime — implementation plan

**Status**: PR 2 **LANDED**. The `while` and `do-while` **surfaces** parse (they
lower to `while_loop(Cond, Body)` / `do_while_loop(Body, Cond)`) and their
**runtime** now compiles: an arithmetic/`print` body over an i64 counter,
condition `VAR CMP int`. PRs 3–4 (general condition, `break`/`continue`, nested
loops in multi-pass) remain. This doc scopes the runtime, grounded in how the
single-pass driver lowers scalar state.

> **Implementation note (what actually shipped).** The §2 "bracket with memory"
> sketch (alloca/load/store) turned out to be unnecessary: the emitter already
> has a working **loop-header phi** for exactly this shape — `foreach_loop`'s
> head phis (`plawk_foreach_head_phi_lines`). The while/do-while lowering reuses
> that machinery directly (SSA back-edge phis, no memory slots), which is
> simpler and matches the existing style. §2 is kept below as the original
> reasoning; the head-phi route is what landed.

## 1. The blocking fact: scalar state is SSA/phi, not memory

plawk's single-pass driver threads scalar variables (`s`, `i`, counters) as
**SSA values joined by phi nodes**, not memory slots — there are no `alloca`s
for user scalars. Each rule's body recomputes new SSA values for the scalars it
touches (`plawk_scalar_match_update_ir`), and the per-rule/per-record boundaries
phi them together (`plawk_scalar_rule_input_phi_ir`, the `if`-join phis in
`plawk_scalar_if_join_pairs`). An `if` is already lowered this way: then/else
branches produce candidate SSA values, and a join block phis them.

A **loop** is the hard case for this model: a variable mutated in the loop body
depends on its own value from the previous iteration, which SSA expresses with a
**loop-header phi** (`%i = phi [%i_init, %preheader], [%i_next, %body]`). Nothing
in the current scalar lowering emits loop-header phis — the `if` join is a
*forward* phi (two disjoint predecessors), not a *back-edge* phi.

## 2. Approach: bracket the loop with memory (SSA → mem → loop → SSA)

Rather than retrofit loop-header phis through the whole phi-threaded scalar
machinery (invasive, and every downstream phi site would need to learn about the
back edge), **bracket each loop with a memory slot** for the scalars it mutates:

```
; preheader: the scalars are SSA values here (%i_in, %s_in, ...)
  %i.slot = alloca i64
  store i64 %i_in, i64* %i.slot        ; seed from the incoming SSA value
  br label %while.head.N
while.head.N:
  %i.cur = load i64, i64* %i.slot
  %cond  = icmp <op> i64 %i.cur, <bound>
  br i1 %cond, label %while.body.N, label %while.after.N
while.body.N:
  ; body actions load/store the slot(s); print reads %i.cur; i++ stores i+1
  br label %while.head.N
while.after.N:
  %i_out = load i64, i64* %i.slot      ; feed back into the SSA scalar chain
```

- **Self-contained.** The `alloca`/`load`/`store` bracket lives entirely inside
  the loop action's IR. Outside the loop the scalar stays SSA; the loop consumes
  the incoming SSA value (`store` at the preheader) and produces an outgoing SSA
  value (`load` at `while.after`) that flows into the next phi exactly like an
  `if` join's result. No change to the forward-phi machinery.
- **`do-while`** is the same with the condition test *after* the first body
  pass: `preheader → body → head(cond) → body|after`.
- **Which scalars to bracket.** The set the body mutates (walk the body for
  `set` / `inc` / `+=` targets) ∪ the condition variable. Read-only scalars stay
  SSA and are read once into the preheader.

## 3. PR sequence

1. **Surface — LANDED.** `while (VAR CMP int) { BODY }` and
   `do { BODY } while (VAR CMP int)` parse; not-yet compile error.
2. **Runtime, arithmetic body — LANDED.** Loop-header-phi lowering (reusing
   `foreach_loop`'s head phis, **not** memory slots) for a body of `set` /
   `inc` / `+=` over i64 scalars + `print`, condition `VAR CMP int`. `while`
   tests before the body (exit values are the head phis, so zero iterations is
   fine); `do-while` tests after (exit values are the body outputs, so the body
   always runs once). Deliverable met: `{ i = 0; while (i < 3) { print i; i++ }
   }` prints `0/1/2`; the `do-while` mirror runs the body once even when the
   condition starts false. Two enablers landed alongside: **printing a bare
   scalar var** (`print i` — substituted to the slot's SSA value) and a
   **body-printing scalar chain with no `END`** driver clause (all output from
   the per-record body). Tests: `tests/test_plawk_while.pl`.
3. **General condition — LANDED.** The loop condition is now a boolean
   combination of scalar comparisons: each comparison is `VAR CMP (int | VAR)`
   (the right side may be another loop variable, not just a literal), combined
   with `&&` / `||` (`&&` binds tighter). Lowered as a block of i64 `icmp`s
   folded by `and`/`or i1` at the loop's condition point (`plawk_while_cond_ir`
   / `plawk_while_cond_build`); every named variable must be an i64 slot
   (`plawk_while_cond_vars` registers them). A single `VAR CMP int` still parses
   to a bare `cmp(...)`, so PR 2 is a strict subset. Tests:
   `tests/test_plawk_while.pl` (var-bound, `&&`, `||`, do-while var-bound).
3b. **`break` / `continue` — SURFACE + GUARD LANDED; runtime is the follow-on.**
   The `continue` keyword parses (to a `continue` action); `break` already did.
   A `break` inside a loop body, or any `continue`, is now a **clean not-yet
   error** (`check_loop_control`) — this **guards a silent mis-compile**: inside
   a loop `break` used to lower to the rule-level stream-break, stopping the
   record stream entirely instead of leaving the loop. `break` *outside* a loop
   keeps its existing stream-break meaning. The runtime still to wire:
   - **SSA phi merge.** A `break` jumps to the loop's `after`; that block then
     needs a phi merging the normal exit (head-phi values, condition false) with
     the value set at *each* break point. A `continue` adds a back-edge into the
     head phi (`while`) / body-condition (`do-while`) from each continue point.
     The loop emitter collects the body's `branch_break` / `branch_continue`
     exits (delivered via `InnerNextExits`) and builds these merge phis instead
     of letting them propagate to the record loop. A **loop-context stack in a
     global** (pushed/popped around each loop body, read by `break`/`continue`
     and by `plawk_branch_to_done_ir`) redirects the branches to the enclosing
     loop's labels without threading a new argument through the ~14-clause
     sequence walker; nested loops fall out because each loop consumes only the
     exits generated while it was innermost.
   - **`break` semantics.** plawk uses `break` at *rule-body* level to mean "stop
     the record stream" (`break_close_stream`) — non-standard awk. In a loop,
     `break` must mean *loop* break; the loop intercepts its own body's break
     exits (the guard above makes this a hard boundary today). `next`-inside-loop
     stays "next record" (propagates past the loop).
   - **Dependency — scalar `if` conditions.** For a *counter-based* break
     (`if (i > 2) break`) to be useful the `if` condition must accept a scalar
     variable; today `if (COND)` only takes field/pattern conditions (`$1 > 2`),
     not `i > 2` (parse error). So this PR also needs the `if` condition grammar
     extended to scalar comparisons (the same `VAR CMP int/VAR` shape the loop
     condition already uses). Do it alongside, or first.
4. **Nested loops / loop in multi-pass `pass { }`.** Unique slot naming per loop
   nesting; the multi-pass driver's scalar handling.

## 4. Related AWK gaps found while scoping (see the audit)

These surfaced during the control-flow investigation and are tracked in
`PLAWK_AWK_FEATURE_AUDIT.md`; noted here because they share the scalar/if
lowering:

- **`if` with a non-accumulator body doesn't compile.** `{ if (COND) { print
  $1 } }` (a plain guarded print, no scalar update) is currently *outside the
  compilable surface* (exit 3) — the scalar `if` lowering assumes the branch
  bodies update scalar slots. This also blocks **regex/general conditions in
  `if`** (`if ($0 ~ /re/) { … }`): the condition grammar accepts `~` (rule
  patterns compile it), but the guarded-print body is the blocker, not the
  regex. Fix belongs with the `if`-body lowering, not the loop runtime.
- **Numeric user-function args in text mode.** `print f($1)` returns `0` because
  a text field is passed to the foreign call as an *atom*, and the synthesised
  `f(X,R) :- R is X*2` fails `is` on a non-number. In `BINFMT` (typed i64) mode
  it works. This is a **type-model** decision: auto-coerce a field arg to i64
  when the callee uses it numerically (needs body inference), or an explicit
  `num($1)` / `int($1)` coercion at the call site (currently `int(...)` is not
  accepted as a call arg). Prioritised #2 in the audit; wants a small design
  call before coding.
- **brace-less `if`/loop bodies.** `if (COND) print` (no `{ }`) doesn't parse;
  plawk requires a braced block. Minor parser follow-on.
