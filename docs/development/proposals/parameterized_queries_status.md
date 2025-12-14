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

## Progress update (current branch snapshot)
- Modes are parsed and threaded through all query plans.
- Implemented a `param_seed` plan node; pipelines now seed inputs (when declared) before body evaluation, preserving the existing all-output path when no inputs are declared.
- Implemented bottom-up demand closure for non-mutual parameterised recursion: a synthetic `pred$need` fixpoint is built from recursive clause prefixes, materialized once, and used to seed/filter the main predicate’s base and recursive pipelines.
- Broadened query-mode arithmetic constraints:
  - Bound-LHS `is/2` compiles as “evaluate into temp column, then equality filter”.
  - Arithmetic expressions in comparisons (e.g. `X+1 =:= 6`) compile via temp arithmetic columns; `=:=`/`=\\=` use numeric `CompareValues(...)` semantics.
- Added a `materialize` plan node and matching C# `MaterializeNode` runtime support to cache subplan results (used by demand closure).
- C# QueryRuntime now understands `ParamSeedNode`, accepts parameters at execution time, and filters outputs by declared input positions.
- Rendered plans emit input-position metadata into `QueryPlan`.
- Added a plan-structure test for parameterised Fibonacci (`tests/core/test_csharp_query_target.pl`) to assert the need/materialize shape is present and rendered.
- Added end‑to‑end runtime coverage for parameterised Fibonacci and parameter‑passing plumbing in the dotnet harness.
- Added bound-only stratified negation in query mode (`\+` / `not/1`) via a `negation` plan node and C# `NegationNode`, including need-closure support when negation appears before recursive calls.
- Added query-mode aggregates (`aggregate_all/3,4`, including correlated aggregates) via an `aggregate` plan node and C# `AggregateNode` runtime support.
  - Grouped aggregates now support multi-key grouping (group term containing multiple variables maps to multiple `group_indices`).
  - Aggregate goals can now be conjunctions/subplans (e.g. joins, comparisons, stratified negation) via an `aggregate_subplan` plan node and C# `AggregateSubplanNode` runtime support; simple single-predicate aggregate goals still use the faster `aggregate` node path.
  - Rule bodies now support disjunction (`;/2`) by expanding into multiple clause variants and emitting a `union` plan node; `->`/`*->` and cut (`!`) remain unsupported.
  - Relation/recursive literals in query-mode bodies (including aggregate subplan goals) may now include simple constants (atomic/string); they are normalized to fresh variables plus equality constraints (e.g. `p(alice, X)` → `p(A, X), A = alice`).
- Parameterised mutual recursion:
  - Input modes are accepted for mutually-recursive SCCs (previously rejected).
  - When every predicate in the SCC declares compatible input modes (same input count), a tagged `$need` fixpoint is built and shared to seed each member’s base/recursive pipelines (demand-driven mutual SCC evaluation).
  - SCC members without explicit mode declarations can inherit the head predicate’s input positions (when arities permit), enabling `$need` closure for common patterns like even/odd; failures while building `$need` are treated as a silent fallback (no noisy `user_error` output).
  - Otherwise, SCC evaluation falls back to full mutual fixpoint + final parameter filtering.
- Current aggregate constraints:
  - Aggregate goals may be a single predicate call, or a goal built from conjunction/disjunction of relations/constraints/negation/aggregates (compiled as a union-of-branches subplan); `->`/`*->` remain unsupported.
  - Aggregates over SCC predicates are rejected (stratification requirement).
  - Need-closure prefixes still reject aggregates (allowed after recursion in the clause body).
- Next: broaden coverage (caching/memoization for correlated subplans, and optional memoized/procedural fallback once semantics are locked down).
