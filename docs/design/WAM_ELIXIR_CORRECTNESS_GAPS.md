# WAM-Elixir — Correctness Gaps

Running status doc for correctness issues in the Elixir target surfaced
while attempting to benchmark Phase A+B+C optimizations on
`examples/benchmark/effective_distance.pl`. The pipeline compiles and
executes but has produced progressively less-wrong output as each
blocker has been removed.

## Fixed

### `!/0` (cut) not implemented in `execute_builtin`

`WamRuntime.execute_builtin/3` had no arm for `!/0`; unknown ops hit the
`_ -> :fail` catch-all, so every clause with a cut (including
`category_ancestor/4`'s recursive arm) threw `:fail` at the cut call.
Added an arm that clears `choice_points` and advances `pc` — a
conservative approximation matching the green-cut usage the compiler
emits. Proper WAM cut-barrier semantics would prune only to the
clause's saved barrier; not needed by current codegen.

### A and X register namespace collision in `reg_id/2`

`wam_elixir_utils.pl`'s `reg_id/2` mapped both `A1` and `X1` to integer
id `1`. The WAM compiler freely emits sequences like `get_variable X3,
A1` while `A3` is still live — under the collision, the store
clobbered `A3` before the next `A3` read, so head unification
consistently failed. Reg banks now map to distinct integer ranges:

- `A1..A99` → `1..99`
- `X1..X99` → `101..199`
- `Y1..Y99` → `201..299`

This matches the Haskell target's `reg_name_to_int` convention.

### `write_ctx` leak from `put_structure` / `put_list` in lowered emitter

The lowered emitter emitted `[{:write_ctx, N} | state.stack]` after
`put_structure` / `put_list`, mirroring the interpreter. But the
lowered forms of `set_variable` / `set_value` inline heap writes
directly — they never consume the ctx. Subsequent `deallocate` popped
`{:write_ctx, 2}` expecting an env map and raised `BadMapError`.
Solution: stop pushing the ctx in put-mode — `set_*` don't need it.
`get_structure` / `get_list` still push it because `unify_variable` /
`unify_value` go through the runtime, which does consume it.

### `WamDispatcher.call` had no builtin fallback

Meta-calls (e.g. `\+ member(Parent, Visited)`) reach
`WamDispatcher.call("member/2", state)` at runtime. The dispatcher only
knew about user-compiled modules; unknown predicates threw
`{:undefined_predicate, pred}` which surfaced as a crash on every
negation-as-failure over a builtin. The generated default clause now
falls through to `WamRuntime.execute_builtin` before throwing.

### `backtrack/1` didn't catch exceptions from retried clauses

When `backtrack/1` invoked `cp.pc.(state)` to resume the next clause,
that clause could `throw :fail` (its own guards failed) or `throw
{:return, result}`. Both exceptions escaped `backtrack` instead of
being translated back into the `{:ok, state} | :fail` contract —
callers that hold state across multiple backtrack steps (e.g. the bench
driver's `execute_backtrack/2` loop) crashed on the first failure.
Added a `try`/`catch` that re-enters `backtrack` on `:fail` and unwraps
`{:return, _}`.

## Status

At dev scale (19-row reference), the pipeline now produces 3 rows with
non-trivial effective distances — up from 2 rows of pure root-matches
before any fix. That means single-hop `category_ancestor` succeeds and
some backtracking works, but multi-hop recursion still does not find
all paths. Reference expects articles like `Nuclear_physics`,
`Gravitational_constant`, `Statistical_mechanics` at distance ≈ 2.5–2.9;
these are absent from the generated output.

## Likely remaining root causes

Speculative — investigation hasn't pinned these yet:

- **Choice-point state capture for recursive calls.** CPs save
  `regs`, `heap`, `heap_len`, `trail`, `trail_len`, `stack`, `cp`. But
  the `stack` field holds both env frames and transient unify-mode
  markers; if any get-structure path leaves an unbalanced stack, CP
  restore reinstates the wrong env height.
- **`allocate` / `deallocate` pairing across recursive `call`s.** The
  lowered emitter's `allocate` pushes a fresh env; `deallocate` pops
  one. Recursive `category_ancestor(Mid, Ancestor, H1, [Mid|Visited])`
  calls back into the same predicate with new args; if the env stack
  isn't saved/restored correctly across nested `call`s, Y-register
  values go stale.
- **`unify_value` / `unify_variable` in read mode against already-bound
  heap cells.** Lowered `step_get_structure_ref` sets up the
  read-mode context; if the following `unify_value` paths don't
  correctly dereference, unification against the parent category may
  silently succeed when it shouldn't (or fail when it should).

A minimal reproducer — a recursive hand-written `ancestor/2` over a
tiny fact set — would isolate which of these is the issue without the
full benchmark's scaffolding.

## Out of scope for this doc

- `switch_on_constant` duplicate-key warnings were fixed in an earlier
  PR (`fix/wam-elixir-switch-dedup-arms`).
- Performance-oriented work (Phase A/B/C) landed separately and is
  unrelated to these correctness gaps.

## Implication for benchmarking

Phase A+B+C numbers remain unmeasurable until the remaining recursion
correctness is resolved. The existing benchmark driver will accept the
partially-correct output and produce a timing number, but those
numbers would compare against Haskell/Rust results for different
computations — not meaningful.
