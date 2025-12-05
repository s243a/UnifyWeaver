# Proposal: Aggregate Support in C# Generator Mode

## Goals
- Enable `aggregate_all/3` and `aggregate/4` in generator mode, emitting C# that computes aggregates over the current fixpoint.
- Keep semantics aligned with Prolog: aggregates are evaluated over the current solution set and can contribute new facts in the same dependency group.
- Preserve existing guards: fail fast on unsupported forms (e.g., non-ground templates, aggregates over recursive predicates) until explicitly implemented.

## Scope
- Targets: C# generator mode (fixpoint solver). Other targets can reuse the strategy later.
- Aggregates: `aggregate_all/3`, `aggregate_all/4`, `aggregate/4` with standard operators (`count`, `sum`, `min`, `max`, `set`, `bag`).
- Dependency groups: aggregates may appear in recursive groups but must be *stratified* (aggregate over relations whose contents are already fixed in the current iteration) to avoid non-termination.

## Semantics
- Evaluation point: during each fixpoint iteration, before firing a rule containing an aggregate, materialize the aggregate over the *current total* (`HashSet<Fact>`). Aggregates do not mutate total directly; they feed into the head of the enclosing rule.
- Monotonicity: Only allow aggregates that are monotone w.r.t. set growth (e.g., `count`, `sum` over non-negative inputs, `set/bag` accumulation). Non-monotone aggregates (e.g., `min`, `max` decreasing) require care; start by allowing them but note that they may stabilize only when input stabilizes.
- Stratification: Disallow aggregates over predicates in the same SCC as the head (similar to negation). Allow aggregates over predicates from lower strata (already computed or non-recursive in the current iteration).
- Groundness: Require the template term to be ground after variable substitution from the join bindings; otherwise fail with a clear error.

## Codegen Strategy
1) **Detection**: Extend generator clause walk to detect aggregate goals and annotate rule plans.
2) **Extraction**: For each aggregate goal:
   - Identify target predicate and arguments from the aggregate template (e.g., `Rel(Args)` in the template).
   - Identify grouping keys from `GroupVars` (for aggregate/4) and aggregate operator (`count`, `sum`, etc.).
3) **Materialization**: Emit a C# LINQ-style computation over `total`:
   - Filter `total` by relation name and argument equality (using `argN` keys).
   - Project template arguments to anonymous objects or tuples.
   - Apply operator (`Count()`, `Sum(...)`, `Min/Max(...)`, `Distinct()` for set, list for bag).
4) **Join Integration**: Treat the aggregate result as a bound value in the rule’s var map; subsequent goals and head arguments can use it. For `aggregate_all/3`, no grouping; for `aggregate/4`, group by `GroupVars`.
5) **Fixpoint Update**: Aggregates are recomputed each iteration; rules using them can emit new facts, and fixpoint continues until no new facts are added.

## Validation Rules
- Reject if aggregate references a predicate in the same SCC as the head (non-stratified aggregate).
- Reject if template contains unbound variables after joins.
- Reject unknown operators; initially support `{count,sum,min,max,set,bag}`.
- Warn about potential non-monotone aggregates (`min/max`) but allow with the understanding they stabilize only when inputs stabilize.

## Testing Plan
- Unit tests (Prolog plunit) to ensure:
  - Simple `aggregate_all(count, ...)` over extensional facts compiles and runs (via Janus/dotnet).
  - Grouped aggregate (`aggregate(sum(Val), by=[Key], ...)`) produces expected facts.
  - Non-stratified aggregate is rejected.
  - Ungrounded template is rejected.
- Integration tests: extend existing Janus harness to compile and run a small aggregate program and assert output.

## Future Work
- Performance: consider caching aggregate results per iteration to avoid recomputation when inputs unchanged.
- Non-monotone aggregates: detect convergence and possibly short-circuit when no change in aggregate outputs.
- Cross-target reuse: mirror the same semantics and validation in Python/Go generators.
