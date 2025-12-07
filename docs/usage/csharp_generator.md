# C# Generator Mode Quickstart (Standalone Fixed-Point)

This path emits self-contained C# (facts + ApplyRule\_* + Solve) that runs without the managed QueryRuntime.

## Generate
- In Prolog: `compile_predicate_to_csharp(pred/arity, [mode(generator)], Code).`
- The output is a single C# module with:
  - `Fact` record (structural equality for dicts + enumerable args).
  - `GetInitialFacts` (base facts from user clauses).
  - `ApplyRule_*` methods for rules.
  - `Solve()` fixpoint loop.

## Run (tests use Janus)
- Tests call into `csharp_test_helper:compile_and_run(Code, Name)` which writes a temp `Program.cs` + csproj, builds with `dotnet`, and executes.
- You can do the same manually: write `Code` to a file, create a minimal csproj targeting `net9.0`, `dotnet build`, then run the produced exe/dll.

## What’s supported
- Joins, builtins, stratified negation.
- Aggregates:
  - `aggregate_all/3`: count, sum, min, max, set, bag.
  - `aggregate/4` grouped: sum, min, max, set, bag.
  - At most one aggregate goal per rule; when present with joins, it must be last.
- Dependency groups and fixpoint evaluation are handled inside the generated module.

## Performance notes
- Indexing (default on): `Solve()` builds per-relation and arg0/arg1 buckets each iteration.
  - `relIndex: Dictionary<string, List<Fact>>` for relation-level scans.
  - `relIndexArg0/relIndexArg1` for arg-position buckets; joins/aggregates prefer arg0, then arg1, else relation list.
  - Can be disabled via `enable_indexing(false)` option if needed.
- Constraint pruning: builtins/negation whose variables are already bound at the current join depth are evaluated early to cut work; others stay in place, preserving semantics.

## Cross-target direction
- The generator shares helpers (`common_generator`) with other targets; the goal is a common generator API across languages (joins/negation/aggregates) with per-target renderers.
- This C# generator is the current reference for the standalone fixed-point path.
