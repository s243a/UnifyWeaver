# WAM Fact Shape Philosophy

## Summary

Facts should live in the generated program as **data**, not as **code**.
The lowered WAM emitters (Elixir today; others to follow) currently
compile each clause of a fact-only predicate into its own host-language
function. That choice is source of one class of failure — the target
host compiler refuses, or refuses-in-practice, to compile a module with
thousands of functions in it.

This document argues the fix is architectural, not a micro-optimisation,
and should mirror the direction the parameterized query engine is
already taking: the parser / code generator prefers *streaming* tuples
into the engine, and the engine decides how much to retain.

## Core position

The emitter should not eagerly decide to turn every fact into a
standalone function.

That decision belongs in the WAM runtime because only the runtime knows:

- which predicate is being called
- which argument positions are ground at call time
- whether backtracking will demand another solution
- whether the predicate will be queried once, thousands of times, or as
  part of a hot loop
- whether first-argument indexing will pay for itself

So the preferred ownership boundary is:

- emitter responsibility: emit facts as **data** (or point at an
  external data source), and emit small *code* wrappers around them
- runtime responsibility: iterate, index, and retain that data
  according to the query pattern

## Why this matters

### Observed failure mode

Profiling the `effective_distance` benchmark exposed this:

| predicate                | clauses | compile (Prolog) | compile (Elixir) |
| ------------------------ | ------- | ---------------- | ---------------- |
| `category_ancestor/4`    | 2       | 1 ms             | OK               |
| `article_category/2`     | 771     | 40 ms            | ~seconds         |
| `category_parent/2`      | 6009    | 320 ms           | **> 10 minutes** |

After fixing the Prolog-side codegen slowness (PR #1505), the Prolog
emitter produces 6.9 MB of Elixir source in 1.9 s. The Elixir
compiler then fails to produce a loaded module in under ten minutes.
The source is valid; it is simply the wrong **shape**. A module with
6000 `defp clause_*/1` functions is a pathological input for the host
compiler, regardless of size.

Once this shape is eliminated, facts scale with data-structure cost
(`Map`, `list`, ETS) rather than with compiler cost.

### Why the same fix helps across targets

The problem is not specific to Elixir. Every WAM target that compiles
each fact into a host function — Go, Clojure, Haskell, Rust, C#-native
WAM — inherits the same scaling failure. The proposed boundary applies
to all of them, though each target picks the local "data" representation
that fits best.

### Alignment with the query engine

The C# parameterized query engine already distinguishes `Streaming`,
`Replayable`, `ExternalMaterialized`, and operator-owned retained state
(see `QUERY_ENGINE_MATERIALIZATION_PHILOSOPHY.md`). WAM targets making
the same distinction means both layers of the system can speak the same
language about retention.

## Preferred order of strategies

When the emitter has a choice, it should prefer these shapes in roughly
this order. Each step trades a small amount of codegen complexity for
much better host-compile and runtime behaviour:

1. **Host literal data, streamed** — emit the fact tuples as a
   host-language literal (e.g. Elixir `@facts [{...}, {...}]`); the
   runtime iterates lazily.
2. **Host literal data, indexed** — same literal tuples, plus a
   compile-time-built index on the first ground arg (Map).
3. **External data source loaded at boot** — facts live outside the
   module source (TSV, ETS, database). Runtime hits a `FactSource`
   adaptor.
4. **Compiled clauses (current default)** — each fact becomes its own
   host function. Reserved for predicates with dynamic heads, guarded
   unification, or very small clause counts where the overhead does
   not matter.

Defaults lean toward shapes that scale well. Users override explicitly
via Prolog-side configuration predicates.

## What does not change

- Recursive or rule-bearing predicates stay in the current CPS-lowered
  shape. The fact-shape question is orthogonal to non-tail recursion
  handling.
- The `run/1`, `next_solution/1`, `materialise_args/1` contract stays
  identical at the module boundary, so drivers and caller tests are
  unaffected.
- Existing `emit_mode(lowered)` / `emit_mode(interpreted)` continue to
  work; fact-shape is a third axis ("fact layout"), not a fourth
  emit-mode.

## Non-goals

- This document does not specify an index format, a serialisation
  format, or a query-time planner. Those are deferred to the spec and
  plan docs.
- It does not propose removing the compiled-clause path. Small
  fact-only predicates and all non-fact predicates continue to use it.
- It does not specify which targets receive the change first beyond
  "start with Elixir, where the scaling failure is already observable."
