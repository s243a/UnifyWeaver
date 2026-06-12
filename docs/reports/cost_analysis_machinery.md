# Static cost-analysis machinery

**Status:** phase 1 built + tested (`src/unifyweaver/core/cost_analysis.pl`,
`tests/test_cost_analysis.pl`, 16 tests green). Foundational analysis; consumers
are wired incrementally.

## Why

Several lowering decisions are currently gated by **hard-coded magic numbers** or
a **runtime probe**:

- T6 first-arg indexing — `t6_min_clauses(8)`.
- T4/T5 multi-clause lowering — fixed thresholds.
- **T7 parallel aggregate** — the `gated_collect` substrate
  (`rust_runtime/par_aggregate.rs`) samples a few branches *at runtime* per call
  to decide whether to fan out. That works, but it's reactive (per-call probe
  overhead) and can't decide at compile time.

These all want the same missing thing: a **compile-time estimate of how much work
a goal / clause body / predicate does**. This module is that shared machinery.
It was the explicit prerequisite for T7 phase 2 — we deferred the interpreter
wiring until the cost model existed, rather than ship a magic-number gate.

## The model

Cost is `cost(Weight, Boundedness)`:

- **Weight** — relative units. Builtins are weighted by a small, overridable
  table (`builtin_cost/2`: unification 1, arithmetic 2, term build 3, list/atom
  ops 3–6, I/O 5–8…). Conjunction adds weights; disjunction / if-then-else takes
  the worst single-solution branch (`cost_alt`); a predicate call adds its
  memoised body cost plus a small call overhead.
- **Boundedness** — `bounded` or `unbounded`. A goal is **unbounded** when it
  (transitively) calls a **recursive** predicate or enumerates an **open-ended
  aggregate** (`findall`/`bagof`/`setof`/`aggregate_all`/`forall`): its cost
  grows with input. That is exactly the *"expensive, worth parallelising"*
  signal T7 needs.

**Recursion** is detected once per program from the **call graph**: a predicate
reachable from itself (self- or mutually-recursive) is `unbounded`. The remaining
predicates form a DAG whose costs are computed bottom-up with memoisation, so the
analysis terminates and is cheap. The cost table and tier thresholds are
`multifile`/`dynamic`, so callers tune them without editing the module.

### Tiers

`cost_tier/2` maps a cost to `trivial | cheap | moderate | expensive | recursive`
(weight bands on a bounded cost; `recursive` = any unbounded cost). `expensive`
and `recursive` are the "do parallelise / the host compiler won't flatten this"
tiers.

## API

```prolog
build_cost_model(+Module, -Model)              % all clause-defined predicates
build_cost_model(+Module, +Preds, -Model)      % a chosen subset
predicate_cost(+PI, +Model, -cost(W,B))
goal_cost(+Goal, +Model, -cost(W,B))           % arbitrary goal vs the model
clause_body_cost(+Body, +Model, -cost(W,B))
cost_tier(+cost(W,B), -Tier)
goal_cost_tier/3, predicate_cost_tier/3        % convenience
recursive_predicate(+PI, +Model)
builtin_cost/2, cost_tier_threshold/2          % overridable tuning
```

## How the gates consume it (next)

- **T7 (parallel aggregate).** For a forkable `findall(Tmpl, Generator, _)`,
  estimate the **per-branch body** cost (`goal_cost` of the generator's work
  goal). `recursive`/`expensive` ⇒ parallelise-eligible; `trivial`/`cheap` ⇒
  stay sequential. This replaces the runtime probe's *decision* (the probe can
  remain as a cheap confirmation, or be dropped). The `par_aggregate.rs`
  `ParConfig` was deliberately built pluggable for exactly this.
- **T6 / T4 / T5.** Replace the magic-number thresholds with cost-tier checks on
  the candidate clauses' bodies (e.g. only bother lowering when bodies aren't
  `trivial`).

## Limitations (honest)

- Aggregate **cardinality** is treated as unknown (⇒ unbounded). A later
  refinement can estimate cardinality from fact-table sizes (the selectivity
  data already computed in `rust_target.pl`'s query path).
- Weights are **relative**, not wall-clock; they rank workloads, they don't
  predict microseconds. The runtime probe remains the source of absolute timing
  when one is needed.
- Clause combination uses the **worst single-clause** branch (a call commits to
  one clause). Exhaustive contexts (findall over all clauses) are already covered
  by the unbounded aggregate rule.
