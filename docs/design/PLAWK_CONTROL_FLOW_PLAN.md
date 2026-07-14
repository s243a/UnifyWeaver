<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk control-flow runtime — implementation plan

**Status**: plan. The `while` and `do-while` **surfaces** parse (they lower to
`while_loop(Cond, Body)` / `do_while_loop(Body, Cond)` and today emit a clean
not-yet error); this doc scopes the **runtime**, grounded in how the single-pass
driver actually lowers scalar state. Prioritised #1 in
`PLAWK_AWK_FEATURE_AUDIT.md`.

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
2. **Runtime, arithmetic body.** Bracket-with-memory lowering for a body of
   `set` / `inc` / `+=` over i64 scalars + `print`, condition `VAR CMP int`.
   Deliverable: `{ i = 0; while (i < 3) { print i; i++ } }` prints `0/1/2`, and
   the `do-while` mirror runs the body once even when the condition starts false.
3. **General condition + `break`/`continue`.** A boolean/expression condition
   (reuse the guard grammar) and loop-control statements.
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
