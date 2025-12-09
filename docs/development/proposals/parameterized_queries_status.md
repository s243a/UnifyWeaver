# Parameterized Queries (Query Mode) – Work-in-Progress (WIP)

**Branch:** `feat/parameterized-queries-querymode`  
**Status:** Exploration WIP (do not merge to main until stable)

## Motivation
- Query mode today assumes head vars are unbound and seeded only by relation scans. Recursive arithmetic (e.g., Fibonacci) fails because arguments must be computed before recursion.
- Generator mode already handles these patterns; we want query-mode parity for function-style calls without giving up the relational pipeline for datalog queries.

## Proposed approach (from `PARAMETERIZED_QUERIES_PROPOSAL.md`)
1) **Mode declarations** (`+/-/?`) to mark input/output args (defaults to all-output to preserve current behavior).
2) **IR extensions**:
   - Parameter seed node (inputs as initial bindings).
   - `bind_expr` node to compute new bindings from bound vars before recursion.
   - Parameterized recursive refs (arg mappings).
3) **Codegen/runtime**: parameterized entry points, new node renderers; keep existing paths for all-output queries.
4) **Tests**: recursive arithmetic (Fibonacci), negation, aggregates, mixed-mode predicates to guard semantics.

## Risks/concerns
- Correctness: interactions with stratified negation/aggregates/recursion need careful testing.
- Scope: adds a parallel path; must keep all-output/datalog path unchanged.
- Types/inference: parameter types may need hints when relations aren’t used to seed bindings.

## Current plan
- Implement incrementally on the feature branch:
  - Parse mode declarations and store in plan metadata.
  - Add seed + bind_expr + param_recursive_ref nodes to the IR.
  - Update query codegen/runtime for new nodes.
  - Add targeted tests (Fibonacci-style recursion; ensure existing tests still pass).
- Keep generator mode as fallback; only merge to main when stable.
