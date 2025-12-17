# Query vs Generator for Recursive Arithmetic (Fibonacci) – Analysis & Tradeoffs

**Date:** 2025-12-10  
**Author:** Codex (analysis of prior findings)

## Summary
- **Generator mode** can compile and run recursive arithmetic (e.g., Fibonacci) because it emits standalone C# with a fixpoint loop; it is not constrained to relation-first pipelines.
- **Query mode** in its default **all‑output** form cannot compile Fibonacci-style clauses because it expects a relation scan to seed variable bindings, and arithmetic (`is/2`) requires bound RHS variables. Clauses that begin with constraints/arithmetic have no bindings to work from, and recursive calls need arguments that aren’t produced by any relation.
- **Parameterized query mode** (via `mode/1`, e.g. `mode(fib(+, -)).`) supports Fibonacci‑style recursion for eligible predicates/SCCs by seeding inputs and using a demand‑closure (`pred$need`) fixpoint to scope recursion. This preserves the bottom‑up query model while enabling function‑style calls.

## Observations
- Query pipeline:
  - Starts from relation scans; head variables are unbound until a relation produces them.
  - Constraints/arithmetic are valid only after RHS vars are bound.
  - Clauses beginning with constraints are rejected; even if allowed, VarMap is empty, so arithmetic fails.
- Fibonacci needs to compute `N1/N2` from `N` before recursive calls; `N` itself is not produced by any relation—hence circular in the all‑output model.
- With parameterized modes, `N` is pre‑bound from parameters, so `N1 is N-1` / `N2 is N-2` are legal; demand closure materializes the reachable `N` values so the fixpoint does not enumerate the whole numeric space.
- Generator mode has no such limitation: it emits facts + rules + a fixpoint loop; arithmetic can run as part of rule code.

## Pros/Cons: Query Mode vs Generator Mode
### Query Mode
- **Pros:** Integrates with managed QueryRuntime, LINQ-style pipelines, relation-centric; good for datalog-style joins, filters, aggregates on known data. Parameterized modes add function‑style entry points without breaking datalog behavior.
- **Cons:** All‑output recursion still cannot synthesize arguments before recursion; constraints/arithmetic require bound vars. Parameterized recursion works for single recursion and (when modes are compatible) mutual recursion via `$need`, but has restrictions (e.g., aggregates are disallowed in need-closure prefixes; aggregates over SCC predicates are rejected; negation is bound-only/stratified).

### Generator Mode
- **Pros:** Can handle recursive arithmetic; standalone C# (no runtime dependency); supports aggregates (aggregate_all/3 and grouped aggregate_all/4), arg0/arg1 indexing, early pruning of bound-only builtins.
- **Cons:** No shared managed runtime; separate codegen path; not as optimized for relation-heavy scenarios as the query runtime might be.

## Potential Directions
- **Fallback strategy:** Use generator mode for patterns query mode can’t yet handle (e.g., mutual recursion with computed args), with clear errors pointing to generator mode.
- **Parameterized queries (in progress):** Track A demand‑closure is implemented; Track 2 would extend this to mutual recursion, stratified negation, and aggregates, or add a memoized/procedural fallback path for non‑datalog patterns.
- **Specialized compilation (optional):** Pattern‑match pure numeric recursion and emit iterative/memoized C# helpers when performance or eligibility demands it.

## Current Actions Taken
- Added generator playbook (`playbooks/csharp_generator_playbook.md`) showing Fibonacci works in generator mode and noting default query limitations.
- Implemented parameterized query support in C# query mode (modes metadata, `param_seed`, demand‑closure `pred$need` with `materialize`, runtime support, and tests in `tests/core/test_csharp_query_target.pl`).
- Expanded query mode coverage for practical programs (bound-only stratified negation, correlated/grouped aggregates including aggregate subplans and nested aggregates, disjunction expansion, and performance work like `KeyJoinNode` and per-execution caches).

## Recommendation
- Keep default query mode focused on relation‑first datalog semantics; reject Fibonacci‑style recursion without inputs and point to generator mode or parameterized modes.
- Continue parameterized‑query Track 2 only after semantics are solid: broaden eligibility carefully (mutual recursion / negation / aggregates) or introduce a memoized fallback if needed.
