# WAM-Elixir — Correctness Gaps

Running status doc for correctness issues in the Elixir target surfaced
while attempting to benchmark Phase A+B+C optimizations on
`examples/benchmark/effective_distance.pl`.

## Fixed

### `!/0` (cut) not implemented in `execute_builtin`

`WamRuntime.execute_builtin/3` had no arm for `!/0`; every clause with a
cut threw `:fail` at the cut call. Added an arm that clears
`choice_points` and advances `pc`.

### A and X register namespace collision in `reg_id/2`

Both `A1` and `X1` mapped to integer id `1`. Reg banks now map to
distinct ranges (`A: 1-99`, `X: 101-199`, `Y: 201-299`).

### `write_ctx` leak from `put_structure` / `put_list` in lowered emitter

Lowered put-mode ops pushed a `write_ctx` frame that `set_variable` /
`set_value` never consumed, corrupting the stack for `deallocate`.
Fixed by removing the push from put-mode ops.

### `WamDispatcher.call` had no builtin fallback

Default clause now falls through to `WamRuntime.execute_builtin` before
throwing `{:undefined_predicate, _}`, so meta-calls like `\+ member(...)`
resolve.

### `backtrack/1` let clause exceptions escape

Added `try`/`catch` that cascades to the next CP on `:fail` and unwraps
`{:return, _}`.

### Caller-supplied unbound id collided with register numbers

`run(args)` now rewrites each caller-supplied `{:unbound, _}` to
`{:unbound, make_ref()}` so the id cannot collide with any reg slot.

### A-registers used as scratch; driver couldn't read outputs

Added `WamState.arg_vars`, `WamRuntime.materialise_args/1`, and
`WamRuntime.next_solution/1`. Drivers read outputs from `state.regs[i]`
transparently after the wrappers deref the tracked unbound.

### Y-registers globally shared across recursive calls

Y-registers (ids 201-299) lived in the global `state.regs` map, so
recursive calls to the same predicate overwrote each other's
permanents. `allocate` now saves the current Y-reg subset into
`env.y_regs_saved` and clears those slots; `deallocate` discards
current Y-regs and merges the saved ones back. Correctness improvement
but not observable in the current benchmark because of the
non-tail-call continuation bug below.

## Remaining: non-tail recursive calls lose the outer continuation in lowered mode

The lowered emitter represents each clause as a standalone Elixir
function. `WamDispatcher.call` invokes a predicate's clause; the clause
runs to completion and returns `{:ok, state}`. For a non-tail body like
`body_goal(...), is/2` (`ancestor_h(X,Y,N) :- parent(X,Z), ancestor_h(Z,Y,N1), N is N1+1`):

- First solution works: outer's clause runs straight through, calls
  inner, gets its return, runs outer's `is/2`, returns.
- On backtrack: `backtrack/1` pops inner's CP and invokes
  `&clause_LAncestorH32/1` directly with the restored state. Inner's
  clause runs to completion — including its own `is/2` — and returns.
  The result propagates through `backtrack` to `next_solution` and out
  to the driver. **Outer's post-call code (reading Y3 for N, running
  its own is/2) is never re-entered.**

Consequence: alternative solutions show the inner's `N` binding but
not the outer's. In `examples/debug_wam_elixir_ancestor.pl` Test 3
with a 4-level fact chain:

```
Y="b" N="1"                                (1st — base case, ok)
Y="c" N=2.0                                (2nd — 1-hop recursion, ok)
Y="d" N={:unbound, <caller's N ref>}      (3rd — 2-hop, N not bound)
Y="e" N={:unbound, <caller's N ref>}      (4th — 3-hop, N not bound)
```

Traces confirm: on the 3rd+ backtrack, inner's `is/2` binds inner's
local `N` ref (~206372) but outer's caller-`N` ref (~206345) stays
unbound because outer's `is/2` never runs.

### Fix shape

The lowered emitter needs to preserve the outer's continuation across
a `call`. Options:

1. **Continuation passing.** A clause takes a `k` (continuation)
   parameter and tail-calls it with the final state. Every `call` in
   the body captures the post-call code in a closure and passes it as
   `k` to the callee. Backtrack's `cp.pc.(state, k)` would then invoke
   the retry with the saved continuation.
2. **State.cp as function ref.** Store the post-call function in
   `state.cp` before each `call`; have `proceed` invoke `state.cp`.
   Matches the interpreter-mode abstraction but requires the lowered
   emitter to synthesise a fresh `defp` for each call-point's
   continuation.
3. **Interpreter fallback.** Detect non-tail recursive calls at
   codegen time and emit interpreter-style dispatch for those
   predicates.

Option 2 is probably the cleanest given the existing `state.cp` field
already exists. Option 3 is the smallest change but splits the
generated code into two regimes.

Fact-only predicates, plain tail-recursive predicates, and
deterministic clauses work correctly today; only non-tail recursion is
affected.

## Out of scope for this doc

- `switch_on_constant` duplicate-key warnings — fixed in
  `fix/wam-elixir-switch-dedup-arms`.
- Phase A/B/C perf work is orthogonal.

## Implication for benchmarking

`effective_distance` at dev scale still produces 3 rows vs the 19-row
reference because `category_ancestor/4` has non-tail recursion (cut
and `H is H1 + 1` after the recursive call). Meaningful perf
comparisons still require the continuation fix.
