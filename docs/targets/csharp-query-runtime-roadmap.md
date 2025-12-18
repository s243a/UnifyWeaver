# C# Query Runtime Roadmap (`target(csharp_query)`)

This roadmap tracks the next features for UnifyWeaver’s managed C# query engine (IR + runtime). It is ordered to minimise rework (build foundations first) and, when trade-offs are unclear, to prefer smaller/easier increments that unlock the next steps.

## Scope
- Runtime: new plan nodes, execution strategies, diagnostics, caching, indexing.
- Compiler: mapping new Prolog constructs onto IR (only as needed per feature).
- Tests: extend `tests/core/test_csharp_query_target.pl` and the smoke runner to cover new semantics.

## Guiding Principles
- Correctness first: match existing UnifyWeaver semantics (Bash distinct/joins + Prolog constraints) before optimising.
- Observability before optimisation: add `explain/trace` so perf work is measurable and debuggable.
- Opt-in determinism: keep hash-set semantics as the default; add stable ordering/paging as explicit options.
- PR-sized slices: each milestone should decompose into small, reviewable PRs that add a node + tests.

## Milestones (Proposed Order)

### M1 — Explain/Trace + Plan/Query Caching (foundation, low risk)
**Why first:** instrumentation and caching reduce future debugging time and repeated work; they also help validate later optimiser/index changes.

- Explain/Trace
  - `Explain()`/plan printer with node tree, predicate IDs, arities, and parameter slots.
  - Optional execution trace: per-node row counts, timings, and fixpoint iteration stats.
  - Acceptance: a failing query can be debugged via a single “print plan + trace” output.
- Plan caching (prepared queries)
  - Cache “compiled” plans per predicate (and per parameter schema) so repeated runs do not rebuild node graphs.
  - Ensure caches remain correct under different parameter values (no accidental capture of bound values).
  - Acceptance: repeated invocations show stable plan identity and reduced allocations/time (via trace counters).

### M2 — Bound-Argument Indexing + Join Strategy Selection (performance, medium risk)
**Why next:** indexing depends on having stable plan shapes and good diagnostics (M1). It also benefits every other feature.

- Bound-argument indexing
  - In-memory hash indexes for relations keyed by one or more columns (selected based on observed bound arguments).
  - Support multi-arity keys and “prefix” binding patterns.
  - Acceptance: common patterns like `p(X, 42)` avoid full scans; trace shows index hit rates.
- Join strategy selection
  - Prefer joins that use indexes and/or smaller inputs first (simple heuristics before a full optimiser).
  - Acceptance: joins on bound keys run measurably faster on synthetic “large fact set” tests.

### M3 — Deterministic Output + Paging (ergonomics, medium risk)
**Why here:** it can be implemented as IR/runtime nodes without changing core semantics; it also improves reproducibility for users/tests.

- Deterministic ordering (opt-in)
  - `order_by` plan node (or ordering hints) with a clear, well-defined comparer for supported value types.
  - Stable `distinct(strategy(ordered))` (optional) when determinism matters more than speed.
  - Acceptance: same query + same inputs yields identical output ordering across runs.
- Paging
  - `limit` and `offset` nodes (or `limit/offset` hints) that work after ordering.
  - Acceptance: `limit/offset` behaves like SQL paging when used with `order_by`.

### M4 — Aggregates + `group_by` (expressiveness, medium/high risk)
**Why after M3:** grouping and aggregation often benefit from deterministic ordering and from the hashing/index infrastructure added earlier.

- Aggregates (non-grouped)
  - `count`, `sum`, `min`, `max`, `avg` with type/Null rules defined and tested.
  - Acceptance: aggregate outputs match expected rows in generated projects.
- Grouped aggregates
  - `group_by` with multiple keys; grouped aggregates per group.
  - Acceptance: grouped aggregate queries match expected results and remain performant with large inputs.

### M5 — Stratified Negation / Anti-Join (“NOT EXISTS”) (correctness-sensitive, high risk)
**Why last:** negation interacts with recursion and evaluation order; it needs careful analysis and good test coverage.

- Stratification analysis (compiler-side)
  - Detect and reject/diagnose unsafe negation in recursive SCCs; accept stratified programs.
  - Acceptance: clear errors for non-stratified negation; correct results for stratified cases.
- Runtime anti-join node
  - Implement `NOT EXISTS` / anti-join semantics with parameter binding support.
  - Acceptance: negation tests cover both ground and partially-bound cases.

## “Nice Next” (post-milestones)
- External fact providers: pluggable relation sources (CSV/JSONL streaming, DB-backed providers).
- Async execution: `IAsyncEnumerable` output + cancellation tokens for long-running fixpoints.
- Memory strategy controls: spilling/materialisation controls for large result sets.
- Plan serialisation: persist/query plans as JSON DTOs for offline inspection and execution.

## Tracking
- This file is the canonical roadmap.
- Each milestone should be turned into GitHub issues (one per PR-sized slice) and linked back here as they’re implemented.
