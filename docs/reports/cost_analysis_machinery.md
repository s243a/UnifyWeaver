# Static cost-analysis machinery

**Status:** built + tested. The machinery (`src/unifyweaver/core/cost_analysis.pl`,
16 tests) **plus its first real consumer** — the T7 parallel-profitability gate
(`src/unifyweaver/core/parallel_gate.pl`, 10 tests) — are green. Further consumers
wired incrementally.

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

## How the gates consume it

- **T7 (parallel aggregate) — gate built (`parallel_gate.pl`) AND wired into Rust
  codegen (slice 2a).**
  `aggregate_parallel_decision(+Goal, +Model, -Decision)` recognises a forkable
  aggregate (`findall`/`aggregate_all`/`bagof`/`setof`) and returns `parallel`
  vs `sequential` from the generator's cost tier. Key simplification: an
  aggregate generator's own boundedness/weight *already* reflects per-branch
  work — a pure enumerator (`member/2`, fact lookups) is cheap+bounded, a
  generator calling a recursive predicate is unbounded, heavy per-solution work
  has a high weight — so the decision needs no fragile "enumerator vs body"
  split. Default policy: only `expensive`/`recursive` fan out (overridable via
  `parallel_worthy_tier/1`); everything else stays sequential, the safe default
  that never regresses. This is the *compile-time* decision the
  `par_aggregate.rs` substrate's `ParConfig` was built pluggable for — the
  runtime probe becomes a cheap confirmation rather than the decision.
  **Slice 2a (`wam_rust_target.pl`):** `compile_wam_predicate_to_rust` now
  consults this gate (per predicate, over its clause bodies' forkable
  aggregates) and annotates the generated function — `/// T7: parallel-eligible
  aggregate (cost gate tier: …)` — when a generator is parallel-worthy. This is
  the first time the cost machinery drives real codegen. Decision-only, no
  semantics change; gated behind `parallel_aggregates(true)` so default output
  is byte-identical. Tested end-to-end (`test_wam_rust_parallel_aggregate_gate.pl`):
  recursive/heavy generators get the annotation, cheap/no-aggregate/feature-off
  do not. **Slice 2b (next):** make `WamMachine: Clone + Send` and have the
  annotated path actually call `gated_collect`, with an exec harness proving
  parallel result-set == sequential + speedup.
- **T6 — *not* a fit (deliberately not wired).** `t6_min_clauses` gates on
  clause **count**, not body cost; first-argument indexing helps regardless of
  per-clause work, so cost analysis adds nothing there. Documented so the
  threshold isn't "fixed" with the wrong tool.
- **T4 / T5 — candidate.** Could use a cost-tier check on candidate clause
  bodies (e.g. prefer lowering all clauses when later clauses carry heavy
  bodies), but this is multi-target and behaviour-changing — a separate step.

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
