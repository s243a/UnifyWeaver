<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk `over query(Goal)` reader — implementation plan (phase 6)

**Status**: PRs 1–5 landed; a goal of any arity with a `print $K...` body —
optionally gated by an `if ($K CMP int)` reader guard — runs end-to-end, in an
all-query program or mixed with ordinary passes.
Sequences the work for the query-driven reader (`PLAWK_MULTIPASS_CACHE.md`
§3.4, phase 6): a pass whose records are the **solutions of a Prolog goal**.
This is the "most beyond awk" reader and the first place plawk admits
controlled non-determinism, so it is planned carefully and grounded in the
actual runtime surface before any build.

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
needed.

**Feasibility confirmed (the plan's biggest risk, retired).** `findall/3` +
`call/1` already run end-to-end in a **compiled plawk binary** on the LLVM WAM:
`tests/test_plawk_eval_compile.pl` has a passing test whose runtime-compiled
grammar executes `findall(Q, call(G), L), sum_list(L, R)` and returns the folded
result. So the enumeration builtin the approach depends on is real and works in
the generated code; PR 2 is wiring, not a new engine capability. (A direct
engine-driven collector remains the fallback, with the surface unchanged, but is
no longer expected to be needed.)

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

### PR 2 — `findall` wrapper + materialisation + query pass function — **LANDED**

The runtime crux, delivered as one end-to-end vertical for the first supported
shape (arity-1 goal, body `print $1`) so the materialisation is observable:

- **Injected findall wrapper.** For each query goal the build synthesises
  `__plawk_query_pred(L) :- findall(V, pred(V), L)` and adds it to the
  program's `@prolog` predicate set (`plawk_query_helper_clause/3` in the
  codegen; injected in `bin/plawk`), so `write_wam_llvm_project` compiles the
  enumeration into the same binary as the user's predicates.
- **Materialise via the posarray path.** A new query driver
  (`plawk_program_query_driver_ir/3`) emits, per query pass, a self-contained
  `@plawk_pass_N` that runs the wrapper on a shared `%WamState`
  (`@plawk_foreign_vm`, emitted standalone since a query program has no
  foreign-call sites) through `@wam_object_call_posarray`, walking the flat
  solution list into a fresh assoc table by position (keys `1..N`).
- **Ordered, deterministic iteration.** The pass walks keys `1..count` in
  order (not hash-slot order) and prints each solution's integer at `$1`, then
  frees the table. The multiplicity collapses at the boundary before the body
  runs (§1 intact); snapshot semantics fall out for free.
- **Scope + guard.** v1 is an all-query program (no input is read — the records
  come from the goal). Any query program outside the shape (higher arity,
  richer body, `END` block, or a query pass mixed with ordinary passes) is a
  clean not-yet compile error (exit 2, naming `pred/arity`), not a miscompile.
- **Verified end-to-end.** `tests/test_plawk_query_reader.pl`: facts and a
  disjunctive (non-deterministic) goal both materialise and print in order; the
  higher-arity and mixed-pass shapes hit the not-yet error; a non-query program
  is unaffected.

### PR 3 — Higher arity + positional fields — **LANDED**

Generalise beyond the arity-1 flat-list slice to a goal of any arity with a
`print` of `$K` fields:

- **Per-column materialisation.** Rather than a tuple template + a new
  list-of-compounds walker, each goal column gets its own findall wrapper
  `__plawk_query_pred_C(L) :- findall(VC, pred(V1..Vn), L)` and materialises
  into its own assoc table via the *existing* flat-list posarray path. `findall`
  preserves solution order and multiplicity identically across columns, so
  column *i*'s *k*-th element is the *k*-th solution's *i*-th argument — correct
  for a pure generator (the reader's snapshot boundary already assumes no
  interleaved writes).
- **Positional body.** The pass walks keys `1..count` (of column 1) in order
  and binds `$K` to `%qtable_K[pos]`; the body's `print` may list the fields in
  any order, repeat a column (`print $1, $1`), or print a subset (`print $2`) —
  joined by the output separator. Deliverable met: `pass over query(pred(X, Y))
  { print $1, $2 }` prints one line per solution.
- **Guard.** A field referencing a column outside the goal (`$3` for a 2-arg
  goal) stays a clean not-yet error, not a read of an absent column table.
- **Verified end-to-end.** `tests/test_plawk_query_reader.pl`: arity-2 facts,
  a reordered arity-3 disjunctive goal, repeated/subset columns, and the
  out-of-range-column not-yet error.

### PR 4 — Reader guards in the query body — **LANDED**

A query pass body may now be gated by an `if (COND)` reader guard — a WHERE over
the goal's solutions:

```
pass over query(edge(X, Y)) { if ($1 >= 2 && $2 < 40) print $1 }
```

- **Reuses the surface grammar.** `for_in_body` already parses `if (GUARD)
  print ...` to `[if(Guard, [print(Fields)], [])]` with `$K CMP int` leaves
  (`rfield_cmp`) combined by `&&` / `||`; the query driver just consumes it.
- **Pure evaluation.** Each leaf reads `%qtable_K[pos]` (an i64) and compares
  it (`plawk_icmp_pred`); `and` / `or` combine child i1s. Because the reads are
  side-effect-free, the tree lowers to plain i1 `and`/`or` with a single branch
  to a `q_print` block — no short-circuit control flow. The guard may read a
  column the body does not print.
- **Guard.** Only integer comparisons are meaningful (columns are i64); a
  float/string RHS or an out-of-range column stays a not-yet error.
- **Verified.** `tests/test_plawk_query_reader.pl`: a `>` filter, an `&&`
  reading a non-printed column, and an `||`.

### PR 5 — Mixed passes + determinism test — **LANDED**

A query pass may now run **alongside ordinary passes** in one program:

```
pass over query(edge(X, Y)) { print $1, $2 }   # materialised from the goal
pass { print $1 }                               # scans the input file
```

- **Integrated into the general multi-pass driver.** A `pass_query` clause on
  `plawk_multipass_pass_fn` emits the query pass-fn with the *standard*
  multi-pass signature (`%Value %mp_path` + shared-table params, all ignored —
  a query reads no input and shares no table), so main's call site is uniform.
  Query passes contribute no table (`plawk_passes_tables` skips them), so the
  table/cache machinery is untouched.
- **VM getter spliced once.** The general driver does not emit the shared
  `%WamState` a query goal runs on; `plawk_query_mixed_support_ir` adds it (sized
  by the module code/label counts in `Options`) when a query pass is present.
  The i64 print-format global the body uses is already among the driver's
  runtime globals.
- **Order preserved.** Passes run in program order; a query pass emits its
  solutions, an ordinary pass scans the input — in either order, with guards on
  the query pass composing as in PR 4.
- **Determinism.** A test that the same query in two passes yields byte-identical
  output (the collapse to a materialised set is order-stable and repeatable; a
  disjunctive goal appears in the same solution order each pass). A *write*-
  snapshot test (a pass mutating state the goal reads) is deferred: goals reach
  the `@prolog` predicate universe, not mutable plawk tables, so there is no
  path yet for a pass write to perturb a goal's solution set — that awaits goals
  reading mutable plawk state (a later capability).
- **Verified.** `tests/test_plawk_query_reader.pl`: query-then-ordinary,
  ordinary-then-guarded-query, and the two-pass determinism check.

### PR 6 — String columns

Non-integer (atom / string) goal columns via the posarray-str path. The open
question is per-column typing: columns are i64 today, and the goal's arguments
are untyped at build. Likely a tagged materialisation primitive
(`@wam_object_call_posarray_value` storing int-or-atom-id with a tag) so a
column can carry either without a surface type annotation — its own PR.

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

## 6. Future sketch: generator blocks (the producer dual)

The query reader is the **consumer** side of the plawk/Prolog boundary: a
Prolog goal *drives* a plawk pass (Prolog → plawk records). The natural dual is
the **producer** side — a plawk `{}` block that a Prolog goal can *call*
(plawk → Prolog solutions):

```
gen { ... emit E ... } as edges     # defines a callable relation edges/1

pass over query(path(edges, A, B)) { print $1, $2 }
```

Here `gen { … } as edges` is a block that produces a stream of values; naming
it exposes a predicate `edges(X)` reachable from any goal — including a query
reader's, so the two features compose (a gen block feeds a query pass).

**Why blocks, not lazy functions (the user's framing).** AWK users read `{}`
as "streaming work" and functions as eager call-and-return; making a *function*
lazy would violate that intuition, whereas a `{}` block that emits is already
the AWK mental model for producing a stream. So streaming is driven from block
syntax, not from function laziness. Iterating an iterable (array, assoc, DB
column) is a third, orthogonal producer, but the block form is the one that
reads as AWK.

**Why it fits the architecture (materialise, don't stream).** A gen block need
not retain a live choicepoint (which §1 forbids in the hot loop). It runs to
completion collecting each `emit E` into a list, then exposes that set as an
ordinary non-deterministic relation — effectively `edges(X) :- member(X,
Collected)`. That is exactly the **bounded-multiplicity** move
(`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md` Principle 1) this reader already makes,
run in the other direction: the block's multiplicity is collapsed to a
materialised set at definition time, and consumers (findall in a query pass)
iterate it deterministically. The runtime is the mirror of PR 2 — where the
query reader *reads* a findall list into a table, a gen block *writes* an
emitted list into one and wraps it as a callable relation.

**Open questions.** When the block runs (once at load, or per call with
arguments as inputs — the mode story connects to the §3.10 `X in domain`
producer and the "inputs bound from the record" follow-on above); whether
`emit` carries a tuple (arity > 1, matching the per-column materialisation of
PR 3); and lifetime/caching of the collected set (a durable gen block could
back its set with a cache/LMDB store, reusing phase 5). Sequenced after the
query reader's remaining PRs — it shares their materialisation runtime and is
best built on top of it.
