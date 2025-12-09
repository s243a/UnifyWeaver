# Query vs Generator for Recursive Arithmetic (Fibonacci) – Analysis & Tradeoffs

**Date:** 2025-12-10  
**Author:** Codex (analysis of prior findings)

## Summary
- **Generator mode** can compile and run recursive arithmetic (e.g., Fibonacci) because it emits standalone C# with a fixpoint loop; it is not constrained to relation-first pipelines.
- **Query mode** (C# query runtime) currently cannot compile Fibonacci-style clauses because it expects a relation scan to seed variable bindings, and arithmetic (`is/2`) requires bound RHS variables. Clauses that begin with constraints/arithmetic have no bindings to work from, and recursive calls need arguments that aren’t produced by any relation.
- A hard architectural change (parameterized queries or a “bind-before-recursion” node) would be needed to support this pattern in query mode.

## Observations
- Query pipeline:
  - Starts from relation scans; head variables are unbound until a relation produces them.
  - Constraints/arithmetic are valid only after RHS vars are bound.
  - Clauses beginning with constraints are rejected; even if allowed, VarMap is empty, so arithmetic fails.
- Fibonacci needs to compute `N1/N2` from `N` before recursive calls; `N` itself is not produced by any relation—hence circular in the current model.
- Generator mode has no such limitation: it emits facts + rules + a fixpoint loop; arithmetic can run as part of rule code.

## Pros/Cons: Query Mode vs Generator Mode
### Query Mode
- **Pros:** Integrates with managed QueryRuntime, LINQ-style pipelines, relation-centric; good for datalog-style joins, filters, aggregates on known data.
- **Cons:** Cannot synthesize arguments before recursion; constraints/arithmetic require bound vars; recursive arithmetic (Fibonacci) not supported without architectural change.

### Generator Mode
- **Pros:** Can handle recursive arithmetic; standalone C# (no runtime dependency); supports aggregates (aggregate_all/3 and grouped aggregate_all/4), arg0/arg1 indexing, early pruning of bound-only builtins.
- **Cons:** No shared managed runtime; separate codegen path; not as optimized for relation-heavy scenarios as the query runtime might be.

## Potential Directions
- **Fallback strategy:** Use generator mode for patterns the query runtime can’t handle (e.g., recursive arithmetic), with clear documentation/error messages in query mode.
- **Parameterized queries (future):** If we want query mode to accept head parameters, we’d need:
  - A way to seed bindings from query inputs.
  - A “bind expression” node to compute new args before recursion.
  - Runtime changes to accept synthesized tuples not coming from relation scans.
- **Specialized compilation:** Pattern-match pure numeric recursion and emit an iterative/memoized C# helper, bypassing the relational pipeline (opt-in/path-specific).

## Current Actions Taken
- Added generator playbook (`playbooks/csharp_generator_playbook.md`) showing Fibonacci works in generator mode and noting that query mode will reject it.
- No code changes in query mode yet; query remains scoped to datalog-style patterns.

## Recommendation
- Keep query mode focused on relation-first, bound-variable constraints; reject Fibonacci-style recursion with a clear error and point to generator mode.
- If parameterized queries are a priority, design a new IR node/runtime path for bind-before-recursion, or add a specialized numeric-recursion compilation path. Otherwise, document the limitation and rely on generator mode as the fallback.
