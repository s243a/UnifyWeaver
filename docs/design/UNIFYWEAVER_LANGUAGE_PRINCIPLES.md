<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# UnifyWeaver language principles

**Status**: living document. A catalog of the *language-design* principles that
recur across UnifyWeaver's surfaces (plawk, the query/cache layer, the WAM
targets) and how each maps onto established programming traditions — SQL,
Prolog, functional (Haskell), and imperative (Python). The point is not novelty
but *coherence*: when the same principle shows up in four idioms, adopting its
well-understood form keeps our surfaces predictable and lets us borrow the laws
those idioms already proved.

Each principle is stated once here and *referenced* from the design docs where
it is applied, so the rationale lives in one place.

---

## Principle 1 — Bounded multiplicity: contain non-determinism by collapsing it at a boundary

**Statement.** Wherever a construct can yield *many* answers — a set of matching
rows, a goal with several proofs, a generator over a domain — that multiplicity
must be **collapsed to a deterministic value at a syntactic boundary** before it
flows into ordinary (imperative, single-valued) code. Multiplicity is allowed to
*exist* only inside a bounded scope, and a **collapse operator** both opens that
scope and closes it. Raw use of a multi-valued result as if it were a scalar is
an error, not a convenience.

This is the single idea behind UnifyWeaver's determinism stance: *"backtracking
is contained and annotated, never the default in the hot loop"*
(`PLAWK_PHILOSOPHY.md §2`, `PLAWK_MULTIPASS_CACHE.md §1`). "Bounded multiplicity"
is that stance named as a reusable rule.

### The same principle, four idioms

| tradition | the multiplicity | the collapse (the boundary) | raw use |
|---|---|---|---|
| **SQL** | rows in a group | `GROUP BY` + an aggregate (`COUNT`, `SUM`, `array_agg`) | selecting a non-grouped, non-aggregated column is an error |
| **Prolog** | a goal's solution set | `findall/3`, `aggregate_all/3`, `forall/2` (encapsulated search) | a nondet goal left in a det context leaks a choicepoint |
| **functional (Haskell)** | the list monad | a list comprehension `[ e \| x <- xs, p x ]` / `concatMap` | — (the type *is* the list; the comprehension is the boundary) |
| **imperative (Python)** | a generator / iterable | `[e for x in xs if p]`, `sum(...)`, `any(...)` | consuming a generator where a scalar is expected |

They are not analogies — they are the same construct. A SQL `GROUP BY … COUNT`,
a Prolog `aggregate_all(count, Goal, N)`, a Haskell `length [ () | x <- xs, p x
]`, and a Python `sum(1 for x in xs if p(x))` compute the identical thing by the
identical shape: **generate a multiplicity, prune it, fold it to one value.**

### The unifying shape: producer / scope / eliminator

Every instance decomposes into three parts, and naming them avoids the common
confusion that the *eliminator keyword* is what "creates" the non-determinism:

- **Producers** — what introduces multiplicity: a generator drawing from a
  domain (`x <- xs`, `FROM table`, `X in domain`), or a relation/goal with more
  than one solution.
- **Scope** — the brackets in which multiplicity may exist: the comprehension
  body, the `WHERE`/`GROUP BY`, the encapsulated-search goal, plawk's `where`.
- **Eliminator (collapse operator)** — what consumes the scope and yields a
  deterministic result: `collect`/`findall` (all → array), `find` (first),
  `the` (require exactly one), `count`/`sum`/`min`/`max` (fold → scalar),
  `forall` (universal test → boolean).

The eliminator is chosen for *how* to collapse; the producers are where the
non-determinism actually lives. "`collect` triggers the non-determinism" is
backwards — `collect` is the *eliminator*; the generators are the trigger.

### Refinement: multiplicity is a property of the *call*, not the *function*

A predicate is not statically "deterministic" or "non-deterministic." Its
multiplicity depends on its **mode** — which arguments are bound at the call.
`adjacent(A, B)` with both unbound *generates* pairs; `adjacent(a, B)` *extends*
one; `adjacent(a, b)` is a yes/no *test*. So the rule "multiplicity must be
collapsed at a boundary" is enforced not by coloring functions but by **mode
analysis** (`demand_analysis` / `binding_state_analysis`): a call carrying an
unbound logic variable — the only calls that can backtrack — is precisely the
one that must sit inside a collapse scope. A relation used "backwards" (a test
run as a generator) is the extra power Prolog has over a comprehension guard,
and it is exactly the power the mode determines.

Equivalently, the three faces of the plawk containment rule are one statement:

> "multiplicity must be collapsed at a boundary"
> = "unbound logic variables must live inside a `where`"
> = "no choicepoint escapes an iteration bracket."

### Why it matters (not just tidiness)

The boundary is a *performance* invariant, not a style rule. UnifyWeaver streams
in constant memory because per-record state lives in native SSA slots, not a
managed heap. A multiplicity that escaped its scope would force the engine to
retain choicepoints (and the trail/heap they pin) *across* records — turning the
constant-memory loop into one that grows with the input
(`PLAWK_MULTIPASS_CACHE.md §1`). Collapsing at the boundary is what guarantees
the hot loop stays choicepoint-free.

### Where this principle is applied in UnifyWeaver

- **The determinism model** — `PLAWK_MULTIPASS_CACHE.md §1`, `PLAWK_PHILOSOPHY.md
  §2`: the per-record body is deterministic by default; backtracking is
  contained and annotated.
- **Non-unique secondary indexes** — `PLAWK_MULTIPASS_CACHE.md §3.5` (Phase 7):
  a non-unique lookup matches a *set*, so it **must** be consumed by an
  aggregation (`collect` → array, `count`, `sum`/`min`/`max`); raw scalar use is
  a compile-time error. This is the SQL `GROUP BY` face.
- **Reader guards** — `PLAWK_MULTIPASS_CACHE.md` (landed): the `WHERE`-style row
  filter is the *prune* step of the shape, applied to a row reader.
- **Loops vs determinism** — `PLAWK_MULTIPASS_CACHE.md §3.8`: a `while` re-adds
  *unboundedness*, not multiplicity; it lets you enumerate a solution set by
  hand (explicit collapse) instead of folding it (implicit collapse).
- **The contained-search sketch** — `PLAWK_MULTIPASS_CACHE.md §3.10`: the
  future `collect (…) where { … }` construct is this principle made into a
  first-class surface — the Prolog/comprehension face, with the collapse
  operator as the boundary.

---

## Principle 2 — Static structure elides coercion and serialization

**When the compiler statically knows a value's type or a boundary's shape, it
can carry the value in its native representation — skipping the runtime
coercion, parsing, or serialize/deserialize round-trip a dynamically-typed
crossing would pay.** The unit of work you *don't* do is the point.

UnifyWeaver compiles a dynamic-feeling surface (awk-like text fields, Prolog
terms) onto a **typed** substrate (typed WAM values, typed LLVM). Every place
the surface would otherwise force a value through text or an untyped box is an
opportunity: if the type/shape is known at compile time, the crossing is free;
if it is not, a runtime coercion pays for the dynamism. So the design lever is
**push type/shape knowledge to compile time**, and let the dynamic path remain
the correct-but-slower default.

### The same principle, several faces

- **Explicit `emit` vs implicit field-separator yield** (`PLAWK_GENERATOR_BLOCKS.md`
  §2.1): `emit E` materialises a *typed* value straight into the tagged column
  table — no text exists. An FS-split yield would `print` to text and re-parse
  it, a serialize→text→deserialize cost per value. Known type ⇒ no round-trip.
- **Optional function-arg typing over auto-coerce** (LANDED —
  `PLAWK_AWK_FEATURE_AUDIT.md` gap 2): the awk-faithful default auto-coerces a
  text field used numerically (`atom → number` every call); a declared
  `function f(num x)` lets the compiler pass an already-typed value and skip the
  coercion goal entirely — the typed-fast path over the dynamic-correct default.
  (The safety face — turning a type mismatch into a compile error — is the
  follow-on; today the annotation buys the elision.)
- **Typed binary records (`BINFMT`)** vs text fields: a declared record layout
  reads fields as native `i64`/`f64` with no per-field string parse.
- **The query reader's per-column tagged materialisation**
  (`PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md` PR 6): each column carries its
  kind (int/atom), so consumers read the native value rather than re-typing text.

### Relationship to gradual typing

This is why **gradual typing fits UnifyWeaver naturally** rather than as a
bolt-on: the substrate is already typed, so an *optional* annotation is not new
machinery — it is a compile-time assertion that unlocks the representation the
compiler can already emit. The dynamic default stays ergonomic (no annotations
required, awk semantics preserved); the annotation buys the elision. Correctness
comes from the dynamic path; performance and static checking come from the typed
path layered on top — never one instead of the other.

### The tension with Principle 1

Principle 1 (collapse multiplicity at a boundary) and Principle 2 (elide work
when the shape is known) pull the same direction: a **boundary with a known
shape** is both where non-determinism is contained *and* where representation is
fixed, so the compiler can specialise it. A well-typed collapse boundary is the
cheapest one.

---

*Add further principles below as they recur across surfaces.*
