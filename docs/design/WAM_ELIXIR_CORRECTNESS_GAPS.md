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
current Y-regs and merges the saved ones back.

### Non-tail recursive calls lost the outer continuation (CPS refactor)

Fixed in `fix/wam-elixir-cps-continuations`. Each body call now
terminates its sub-segment; subsequent instructions go into a fresh
`defp BaseFunc_kN/1` continuation function. Before each call,
`state.cp` is set to `&BaseFunc_kN/1`; `proceed` tail-calls
`state.cp.(state)`. Top-level callers seed `state.cp =
&WamRuntime.terminal_cp/1`. BEAM TCO collapses the stack across
predicates, so deep recursion doesn't grow it. `backtrack` retries
invoke the stored continuation, so outer post-call code runs on every
solution. `examples/debug_wam_elixir_ancestor.pl` now reports 4/4
correct solutions with bound `N` values across all three tests.

### Catch-snapshot hiding mid-body choice points

The initial CPS wrapping used `catch :fail -> backtrack(state)` where
`state` was the pre-try snapshot — any choice points pushed during the
body were invisible to the catch, causing either lost CPs or infinite
retry loops (seen as a hang). Fixed by threading state through the
throw: every `throw(:fail)` is now `throw({:fail, state})`, and every
catch pattern-matches on `{:fail, s}` and passes `s` to `backtrack`.

### `switch_on_constant` duplicated solutions

Inline dispatch from `switch_on_constant` into the target clause ran
in the presence of the outer `try_me_else` CP (pushed just before the
switch) — on backtrack, that CP retried the same clause, producing
duplicates. The inline-dispatch arm now pops that outer CP before
calling the target clause.

## Remaining / follow-ups

Codegen time dominates benchmark runs at scale 300+ — emitting the
lowered Elixir for six predicates on 300-row facts takes ~80s in the
Prolog side. Not a correctness issue, but worth investigating before
meaningful perf comparison on larger scales.

## Out of scope for this doc

- Phase A/B/C perf work is orthogonal.
