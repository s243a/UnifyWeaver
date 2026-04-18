# WAM-Elixir — Correctness Gaps

Running status doc for correctness issues in the Elixir target surfaced
while attempting to benchmark Phase A+B+C optimizations on
`examples/benchmark/effective_distance.pl`. The pipeline compiles and
executes but still produces fewer result rows than the reference.

## Fixed

### `!/0` (cut) not implemented in `execute_builtin`

`WamRuntime.execute_builtin/3` had no arm for `!/0`; unknown ops hit the
`_ -> :fail` catch-all, so every clause with a cut (including
`category_ancestor/4`'s recursive arm) threw `:fail` at the cut call.
Added an arm that clears `choice_points` and advances `pc` — a
conservative approximation matching the green-cut usage the compiler
emits.

### A and X register namespace collision in `reg_id/2`

Both `A1` and `X1` mapped to integer id `1`. The WAM compiler freely
emits sequences like `get_variable X3, A1` while `A3` is still live, so
the store clobbered `A3` before its next read. Reg banks now map to
distinct ranges (`A: 1-99`, `X: 101-199`, `Y: 201-299`), matching the
Haskell target's `reg_name_to_int` convention.

### `write_ctx` leak from `put_structure` / `put_list` in lowered emitter

The lowered emitter pushed `[{:write_ctx, N} | state.stack]` mirroring
the interpreter, but lowered `set_variable` / `set_value` inline heap
writes directly — they never consume the ctx. Subsequent `deallocate`
popped the residual tuple expecting an env map → `BadMapError`. Stop
pushing the ctx in put-mode.

### `WamDispatcher.call` had no builtin fallback

Meta-calls like `\+ member(Parent, Visited)` route through
`WamDispatcher.call` at runtime. The dispatcher threw
`{:undefined_predicate, pred}` for anything outside the user-compiled
module table. Default clause now falls through to
`WamRuntime.execute_builtin` before throwing.

### `backtrack/1` let clause exceptions escape

`backtrack/1` invokes `cp.pc.(state)` to resume the next clause, which
may `throw :fail` or `throw {:return, result}`. Both escaped instead of
translating back into the `{:ok, state} | :fail` contract. Added
`try`/`catch` that cascades to the next CP on `:fail` and unwraps
`{:return, _}`.

### Caller-supplied unbound id collided with register numbers

`run(args)` put caller-supplied `{:unbound, N}` literally at
`state.regs[N]`. When a subsequent `put_variable` during the body
overwrote `regs[N]` with a fresh unbound, any saved copy of the
original unbound (e.g., `Y2` holding `{:unbound, N}`) would deref
through the overwritten slot and pick up the unrelated new chain.

Fix: `run(args)` now rewrites every caller-supplied unbound to
`{:unbound, make_ref()}` — the make_ref cannot collide with any reg
slot, so overwriting an A-reg no longer corrupts the binding chain of
the original variable.

### A-registers used as scratch; driver couldn't read outputs

Even with the unbound-id rewrite, A-regs get used as scratch during
body execution (e.g., `put_structure +/2, A2` overwrites A2 with a
heap ref for the arithmetic expression). After the predicate returns,
`state.regs[i]` holds whatever intermediate value the body last wrote,
not the caller's output binding.

Fix: track each caller-supplied unbound's ref in `state.arg_vars`,
added as a field to `WamState`. The driver-facing `run/1` calls
`WamRuntime.materialise_args/1` before returning, which derefs each
tracked ref against the final state and writes the resolved value
back to `regs[i]`. Added `WamRuntime.next_solution/1` wrapping
`backtrack/1` with the same materialisation so enumerating alternatives
gives correct output regs on every iteration.

## Remaining: Y-register clobbering across recursive calls

Y-registers (permanent variables) live in `state.regs` at ids 201-299
— the same flat map that holds A/X regs. `allocate` pushes an empty
env frame but doesn't isolate Y-reg slots; `deallocate` pops but
doesn't restore. So recursive calls to the same predicate overwrite
each other's Y-regs.

Concretely, for `ancestor_h(X, Y, N) :- parent(X, Z), ancestor_h(Z, Y, N1), N is N1 + 1`:

- Outer's `Y3` (holding caller's `N` ref) lives at `state.regs[203]`.
- Inner recursive call's `Y3` (holding inner's `N1` ref) also writes
  `state.regs[203]`, overwriting the outer's.
- After the inner returns, outer's `put_value Y3, A1` picks up the
  inner's ref instead of its own — `is/2` binds the wrong variable.

Observable in `examples/debug_wam_elixir_ancestor.pl`:

- **Test 1** (`ancestor(X, Y) :- parent(X, Y); parent(X, Z), ancestor(Z, Y)`) — plain
  variable propagation, no body-side A-reg clobbering. Returns all 3
  solutions correctly.
- **Test 2** (`ancestor_h("a", "d", N)` — bound Y, unbound N). Returns
  no solutions. The 3-hop case requires recursion; the Y-reg corruption
  makes the final `is/2` fail.
- **Test 3** (`ancestor_h("a", Y, N)` — both unbound). First two
  solutions partially correct (Y right, N right for depth 1 and 2). Third
  solution has Y correct ("d") but N unbound — the deepest recursion
  clobbers the outer-most Y-reg.

### Fix shape

Y-regs need to live in the env frame, not the global regs map. Minimal
approach:

- `allocate` saves the current Y-reg subset (keys 201-299 from
  `state.regs`) into `env.y_regs`, then clears those keys.
- `deallocate` pops the env, removes the current Y-reg subset, and
  merges the saved ones back.
- Trail entries for Y-regs during the body need careful handling —
  either scope the trail per env frame, or ensure `deallocate` strips
  body-time trail entries for Y-regs before they can corrupt a later
  unwind.

This is a larger refactor than the run/next-solution API fix and is
staged as its own PR.

## Out of scope for this doc

- `switch_on_constant` duplicate-key warnings were fixed earlier
  (`fix/wam-elixir-switch-dedup-arms`).
- Phase A/B/C perf work is orthogonal.

## Implication for benchmarking

`effective_distance` at dev scale still produces 3 rows vs the 19-row
reference. The driver-API fixes in this PR don't change the benchmark
number because the benchmark's bottleneck is multi-hop recursion
through `category_ancestor/4`, which trips the Y-reg clobbering bug on
every non-trivial path. Meaningful perf comparisons against
Haskell/Rust still require the Y-reg fix.
