<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk `over query(Goal)` reader — implementation plan (phase 6)

**Status**: plan, not yet implemented. Sequences the work for the query-driven
reader (`PLAWK_MULTIPASS_CACHE.md` §3.4, phase 6): a pass whose records are the
**solutions of a Prolog goal**. This is the "most beyond awk" reader and the
first place plawk admits controlled non-determinism, so it is planned carefully
and grounded in the actual runtime surface before any build.

## 1. Where we are (the blocking facts)

- **The goal-call surface is single-solution.** Every `@wam_object_call_*`
  primitive (`_i64`, `_bytes`, `_record`, `_posarray`, …) runs a predicate to
  its **first** solution and returns `{value, i1 success}`, discarding the
  choicepoint stack. `dyncall` / `dyncall_at` are built on these — they call a
  predicate and take one deterministic result. Nothing iterates solutions.
- **The WAM engine does have backtracking** (`try_me_else` / `retry_me_else` /
  choicepoints / trail rewind), so the solutions exist internally; they are
  just not exposed to a caller as an enumeration.
- **`over TABLE` (`pass_over`) is the reader template.** It materialises a set
  (a table's entries) and iterates it, binding each element and running the
  body. A query reader is the same shape with a different source.

The gap is therefore **not** backtracking itself but a way to turn a goal's
solution set into a materialised sequence the pass can iterate.

## 2. Approach: materialise, then iterate (not re-entrant coroutining)

Do **not** expose re-entrant WAM iteration (retain choicepoints across the pass
loop). Retaining a live choicepoint stack across records is exactly the
"residual choicepoint in the hot loop" the determinism model (§1) forbids — it
would pin the trail/heap and break constant-memory streaming. Instead, **run the
goal to completion, collect all solutions into a buffer, then iterate the
buffer** — the same move `over TABLE` already makes.

This is the **bounded-multiplicity principle**
(`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md`, Principle 1) at the reader boundary: the
query's multiplicity is collapsed into a materialised set *before* the body
runs, so the body iterates a fixed, deterministic sequence. **Snapshot
semantics fall out for free** — the solution set is frozen before the first
record, so writes during the pass cannot perturb it.

**The low-risk realisation — a `findall` wrapper.** Rather than build a new
"drive the engine and capture each binding" runtime primitive, lower
`over query(goal(X))` to a **`findall(Template, Goal, List)`** call through the
*existing* single-solution object-call surface: `findall/3` is one deterministic
result (the list), and the list materialises through the already-built posarray
/ list machinery (`@wam_object_call_posarray`). The pass then iterates that list
exactly like `over TABLE`. So the hard part (enumeration) is delegated to the
WAM builtin that already does it, and no new backtracking-driver primitive is
needed. (A direct engine-driven collector is the alternative if `findall/3` is
unavailable or too costly; the plan keeps the surface identical either way.)

## 3. Design decisions (the surface)

- **What the goal is.** `over query(pred(A1, …, An))` where `pred` is a loaded
  predicate — from an `@prolog { … }` block in the program or a `.wamo` object,
  the same predicate universe `dyncall` / `dyncall_at` already reach.
- **How a solution becomes a record.** Positional first: the goal's argument
  bindings in each solution map to fields `$1..$n` of the record (mirroring the
  awk field model and the `rows of` positional reader). A named/record binding
  (`as r`, `r["field"]`) is a follow-on once the positional core works.
- **Template.** The `findall` template is the tuple of the goal's arguments, so
  each list element carries that solution's ground bindings; the reader walks
  the element into `$1..$n`.
- **Determinism.** The goal may be non-deterministic (many solutions) — that is
  the point — but every solution is a ground record, and the collapse to a list
  happens at the boundary, so the pass body stays deterministic (§1 intact).

## 4. PR sequence

Each step its own PR, green regressions before the next.

### PR 1 — Surface + AST (parser), codegen stub — **LANDED**

`pass over query(PRED(V1, …, Vn)) { BODY }` parses to `pass_query(query(Pred,
Vars), Body)` (a new `pass_clauses` clause, tried before `pass over TABLE` — the
`(` after the predicate name distinguishes a query from a bare table). `Vars` is
the goal's positional output-variable list; the body's `$1..$n` will address the
solution's bindings. `plawk_pass_dynentry_rewrite` gains a passthrough clause so
the top-level program parse accepts it. `check_query_reader/2` in
`examples/plawk/bin/plawk` emits a clean, specific compile error (naming the
goal as `pred/arity`) until the runtime lands, rather than the generic "outside
the multi-pass surface". `tests/test_plawk_query_reader.pl`: parse (two-arg,
one-arg, `over TABLE` unchanged), the not-yet error (exit 2), and a non-query
program unaffected. No runtime.

### PR 2 — `findall` wrapper + solution materialisation (runtime/build)

At build time, synthesise the `findall(Template, Goal, List)` wrapper for each
query site and route it through the object-call so the solution list is
materialised (list → posarray/table). This is the crux; it reuses the posarray
list machinery. Verify with a fixed loaded predicate that N solutions produce an
N-element materialised set.

### PR 3 — Codegen: the query pass function

Model on `pass_over`: after materialising the solution set (PR 2), iterate it,
bind each solution's arguments to `$1..$n`, and run the body (`print $1, $2`).
Deliverable: `pass over query(pred(X, Y)) { print $1, $2 }` prints one line per
solution. Reader-guards (`if (…)`) compose for free if the body reuses the
row-reader emitter.

### PR 4 — Determinism guarantees + snapshot test + docs

A test that a write during the query pass does not change the solution set
(snapshot), and that a non-deterministic goal yields exactly its solutions in
order. Document the containment (collapse-at-boundary) and cross-link
`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md` Principle 1 and the §3.10 contained-search
sketch, of which this is the first concrete step.

## 5. Risks / open questions

- **Enumeration cost / memory.** `findall` materialises the whole solution set
  — fine for bounded queries, unbounded for a generator. Document it; a lazy /
  streaming query is explicitly out of scope (it would reintroduce the retained
  choicepoint §1 forbids).
- **Predicate universe & modes.** The goal's variables must be unbound at call
  (outputs) or bound from the record (inputs); the first cut is all-output
  (pure generator). Input binding from prior fields is a follow-on and connects
  to the §3.10 `X in domain` producer.
- **If `findall/3` is not a usable builtin here**, PR 2 becomes a direct
  engine-driven collector (drive `retry_me_else` to exhaustion, capturing each
  solution's registers) — deeper, but the surface (PRs 1, 3, 4) is unchanged.
