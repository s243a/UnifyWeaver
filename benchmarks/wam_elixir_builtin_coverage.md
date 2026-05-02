# WAM-Elixir builtin coverage audit

## TL;DR

The WAM-Elixir runtime's `execute_builtin/3` (`wam_elixir_target.pl`)
implements a subset of what the WAM compiler accepts as builtin
goals. Anything compiled to `builtin_call P/N` whose `P/N` doesn't
match an `execute_builtin` arm hits the default `_ -> :fail` arm and
**silently returns `:fail`** — user code looks like it just doesn't
match. There's no warning, no log, no crash.

This audit cross-references the compiler-recognised builtin set
against the runtime-implemented set, lists the gap, and prioritises
fixes by "how likely real Prolog code uses this". Generated as
follow-up to PRs #1776 / #1777 which surfaced the integer-quoting
and missing-comparison-operator regressions.

## Implemented in `execute_builtin/3` (wam_elixir_target.pl:773)

| Op       | Notes |
| -------- | ----- |
| `is/2`     | arithmetic eval |
| `</2`      | numeric < |
| `>/2`      | numeric > (added in #1777) |
| `=</2`     | numeric ≤ (added in #1777) |
| `>=/2`     | numeric ≥ (added in #1777) |
| `=:=/2`    | numeric equality (added in #1777) |
| `=\=/2`    | numeric disequality (added in #1777) |
| `length/2` | list length |
| `member/2` | list membership |
| `\+/1`     | negation as failure |
| `!/0`      | cut |

## Compiler-handled (no runtime arm needed)

| Goal | How |
| ---- | --- |
| `true/0` | compiles to nothing — control falls through |
| Head-match `=/2` | compiled to `get_constant`/`get_value` etc. (NOT body `=/2`) |

## Compiles but silently fails at runtime

These all show up as `builtin_call P/N` in WAM bytecode and lower to
`execute_builtin(state, "P/N", N)` — but `execute_builtin` has no arm
for them, so they fall through to `_ -> :fail`. **Any user code using
these gets `:fail` with no diagnostic.**

| Goal | Likelihood of real-world use | Notes |
| ---- | ---------------------------- | ----- |
| `=/2`        | **very high**  | body unification — `X = Y` in any rule body silently fails. Verified with `eq_test(X) :- X = 5.` returning `:fail`. |
| `\=/2`       | **high**       | negated unification — `X \= 5` |
| `fail/0`     | medium         | explicit failure — usually compiler-handled but isn't here; `builtin_call fail/0` emitted; also silently fails (which happens to be the right semantics, but for the wrong reason — landing on default `:fail` arm, not a deliberate dispatch) |
| `append/3`   | **very high**  | list concatenation |
| `write/1`    | high           | output — debug/trace/print |
| `nl/0`       | high           | output |
| `format/1`   | high           | output |
| `format/2`   | high           | output |
| `functor/3`  | medium         | term inspection |
| `arg/3`      | medium         | term inspection |
| `=../2`      | medium         | univ — decompose/compose terms |
| `copy_term/2`| medium         | fresh-variable copy |

## Why this hasn't surfaced widely

The existing test suite (Phase 3 / 4c / 4d, target tests) exercises
the implemented set thoroughly: head matching, `is/2` for findall
counts, recursive findall over fact-only predicates, cut, negation.
Body unification (`X = Y`) and explicit list/output ops aren't
exercised — Phase scenarios use head-match and findall accumulation
exclusively. Any user code that uses Prolog more conventionally
(building lists with `append`, debugging with `write`, deconstructing
terms with `functor`/`=..`/`arg`) will silently no-op.

## Surface signature: `:fail` with no error

The WamDispatcher fallback (`wam_elixir_target.pl`, dispatcher
emission) is:

```elixir
def call(pred, state) do
  arity = WamRuntime.parse_functor_arity(pred)
  case WamRuntime.execute_builtin(state, pred, arity) do
    :fail -> :fail
    new_state when is_map(new_state) -> {:ok, new_state}
    _ -> throw({:undefined_predicate, pred})
  end
end
```

The `_ -> :fail` default arm in `execute_builtin` returns `:fail`,
which the caller treats as "predicate failed" — same as any other
clause body that didn't match. There's no distinction between
"`append([1,2], [3,4], L)` failed because no clause matched" and
"`append/3` is unimplemented and the dispatch silently returned
fail". Both look identical to user code.

## Recommended priority order for a follow-up PR

1. **`=/2`** and **`\=/2`** — body unification is fundamental;
   anything beyond fact-only predicates needs them. Implementation
   is small (call existing `unify/3`).
2. **`fail/0`** — though the default-arm behaviour happens to coincide
   with the right semantics, an explicit arm avoids the
   "right answer for wrong reason" trap and would let the catch-all
   arm be tightened to `throw({:unknown_builtin, pred, arity})` for
   real diagnostics.
3. **`append/3`** — pervasive in list-handling code.
4. **`write/1`, `nl/0`, `format/1`, `format/2`** — output operations
   are needed for any debugging story; without them debugging Prolog
   code requires reading inspected state structs.
5. **`functor/3`, `arg/3`, `=../2`, `copy_term/2`** — meta-programming
   primitives, less universally needed but used by reflection-heavy
   code.

## Hardening: surface unknown builtins

Independent of which arms get implemented: change the default arm in
`execute_builtin/3` from `_ -> :fail` to something that distinguishes
"clause failed" from "this builtin doesn't exist". Options:

- **Throw** `{:unknown_builtin, op, arity}` and catch at the top-
  level `run/1` to emit a clear error.
- **Return a tagged tuple** that the WamDispatcher interprets as
  "undefined" (and propagates via `throw({:undefined_predicate, op})`,
  matching the existing fallback arm at the bottom of `call/2`).

Either choice would have surfaced both the `>/2` and `=\=/2` bugs in
PR #1777 the moment they were exercised, instead of waiting for a
benchmark to notice "iterate runs in microseconds at all sizes".

## Reproducing the audit

For each candidate builtin: write a one-clause predicate that uses
it, compile it, generate the Elixir project, run it, and check
whether the result is `:fail` (silent failure, builtin missing) or
`{:ok, ...}`. Skeleton in `examples/benchmark/wam_elixir_tier2_findall_benchmark.pl`.

## History

- Audit triggered by the parallel-vs-sequential benchmark (PR #1774)
  surfacing the integer-quoting bug, which in turn surfaced the
  missing comparison operators. PRs #1776 / #1777 fixed those; this
  audit is the post-mortem map of what else is in the same shape.
