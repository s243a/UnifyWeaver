# Fixed-Point Generator Mode (C#) and Cross-Target Direction

## What it is
- A codegen path that emits self-contained C# with a local fixpoint engine (`Solve()`): facts + `ApplyRule_*` methods + a worklist loop.
- Supports joins, builtins, negation (stratified), and `aggregate_all/3` (count, sum, min, max, set, bag). Grouped sum via `aggregate/4` is handled specially.
- Runs without the managed QueryRuntime; useful for embedding, quick tests, or environments where only generated code is desired.

## How it works (today)
- **Fact store**: `HashSet<Fact>` with structural equality for dictionaries and enumerable values (so list-valued aggregates dedupe correctly).
- **Rule expansion**:
  - Partition body: relations, builtins/negation, optional single aggregate (must be last when mixed with joins).
  - Generate nested `foreach` over `total` for joins, using `VarMap` to translate Prolog variables to `fact.Args["argN"]`.
  - Builtins/negation become inlined C# boolean checks; negation uses `!total.Contains(Fact)`.
  - Aggregates build `aggQuery = total.Where(...)` with emit guards; `set/bag` select list values, `count/sum/min/max` use LINQ reducers.
  - Fixpoint loop unions `newFacts` until no growth.
- **Supported shapes**:
  - Pure aggregate rules: `aggregate_all/3` (count, sum, min, max, set, bag).
  - Aggregate after joins/negation: `rel..., builtins..., aggregate_all/3`.
  - Grouped sum via `aggregate/4` (sum only).
  - Guarded: one aggregate goal per rule; aggregate must be last when mixed with relations.

## Toward a common cross-target API
- The generator path already consumes the shared `common_generator` helpers (`build_variable_map/2`, `translate_expr_common/4`, `translate_builtin_common/4`), mirroring the Python generator.
- The fixed-point loop + `Fact` abstraction can be lifted into a target-neutral core:
  - Define a minimal runtime interface (Fact, equality, fixpoint driver) used by all generated languages.
  - Factor aggregate translation to a shared IR (count/sum/min/max/set/bag, grouped) and per-target renderers.
  - Align negation and stratification checks across targets for consistency.
- Benefits:
  - Less duplicated code (C#, Python, Bash) for joins/negation/aggregates.
  - Easier to add new aggregates or dependency-group semantics once in the shared layer.
  - Clear surface for future unified “generator mode” across targets.

## Near-term work
- Expand grouped aggregates (min/max/set/bag) and eventually `aggregate_all/4`.
- Improve indexing in generated code to reduce full-scan `total.Where` for large fact sets.
- Formalize the shared generator IR (joins, negation, aggregates) and hook additional targets into the same helpers.
- Add docs/examples showing how to call generated C# without the managed runtime and how this maps to other targets.
