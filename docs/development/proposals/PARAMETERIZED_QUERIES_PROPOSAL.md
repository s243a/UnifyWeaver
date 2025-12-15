# Technical Proposal: Parameterized Queries for C# Query Runtime

**Date:** 2025-12-09
**Author:** Claude Opus 4.5
**Status:** Draft - For Review by Codex-5.1

## Abstract

This proposal introduces **parameterized queries** to the C# query runtime, enabling support for predicates like Fibonacci where head arguments are provided as inputs rather than derived from relation scans. This extends the query model from pure enumeration to function-style invocation while maintaining compatibility with existing datalog patterns.

## Implementation Update (Branch Snapshot)

This proposal has been partially implemented (and, in places, superseded) on the working branch `feat/parameterized-queries-querymode`. The current branch design is centred on:
- **Mode declarations** provided as `user:mode/1` facts (not a built-in `:- mode(...)` directive in SWI-Prolog).
- A **`param_seed`** plan node to seed bindings from caller parameters.
- A demand-driven **`$need` closure** (a synthetic fixpoint) that computes reachable input bindings and is then used to seed/filter the main predicate’s base/recursive pipelines.
- A **`materialize`** plan node to cache the `$need` closure result and share it across plans.
- Existing **`arithmetic`** plan nodes for derived columns; no dedicated `bind_expr` IR node is required for the current approach.

See `docs/development/proposals/parameterized_queries_status.md` for the up-to-date implementation status and current constraints.

## Motivation

The current query runtime assumes all head variables are unbound and get bound only through relation scans. This works well for datalog-style queries:

```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
% Query: enumerate all grandparent pairs
```

But fails for function-style predicates:

```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, fib(N1, F1), fib(N2, F2), F is F1 + F2.
% Query: what is fib(10)?
```

The Fibonacci pattern requires:
1. `N` to be provided as input (bound before execution)
2. `N1 is N - 1` to compute new bindings from input
3. Recursive calls with computed arguments

## Proposed Solution

### 1. Mode Declarations

Introduce optional mode declarations to specify input/output arguments:

```prolog
% In this repo, modes are read from `user:mode/1` facts:
mode(fib(+, -)).            % First arg is input (+), second is output (-)
mode(grandparent(-, -)).    % Both outputs (current behavior, default)
mode(lookup(+, -)).         % Key is input, value is output

% Tests often set these dynamically:
%   assertz(user:mode(fib(+, -))).
```

Mode symbols:
- `+` : Input - must be bound by caller
- `-` : Output - bound by query execution
- `?` : Either - proposed future feature for multi-entrypoint compilation; currently the C# query target rejects `?` (use explicit `+`/`-`, or declare multiple concrete modes).

### 2. IR Extensions

#### 2.1 Query Plan Metadata

Extend the query plan to include mode information:

```prolog
% In csharp_target.pl, extend the plan structure
Plan = plan{
    head: HeadSpec,
    root: Root,
    relations: Relations,
    metadata: _{
        classification: Classification,
        options: Options,
        modes: [input, output]  % NEW: argument modes
    },
    is_recursive: IsRecursive
}.
```

#### 2.2 Possible Future IR Node: BindExpression

Add a node type that computes new bindings from existing bound variables:

```prolog
% New node type for pre-recursion binding
bind_expr{
    type: bind_expr,
    input_node: InputNode,
    bindings: [
        binding{var: N1, expr: N - 1},
        binding{var: N2, expr: N - 2}
    ],
    width: NewWidth
}
```

This node:
- Takes an input node (or initial parameter bindings)
- Computes new column values from arithmetic expressions
- Extends the tuple width with new columns
- Outputs tuples with the computed values appended

Implementation note: the current branch uses existing `arithmetic` nodes to compute derived columns and does not introduce a separate `bind_expr` node yet.

#### 2.3 Possible Future IR Node: Parameterized Recursive Reference

Extend the recursive reference node to pass computed arguments:

```prolog
% Current: recursive_ref just references the predicate
recursive_ref{type: recursive_ref, predicate: HeadSpec, role: delta, width: 2}

% Proposed: include argument mappings
param_recursive_ref{
    type: param_recursive_ref,
    predicate: HeadSpec,
    role: delta,
    arg_sources: [
        column(3),  % N1 comes from column 3
        unbound     % F1 is output
    ],
    width: 2
}
```

Implementation note: the current branch keeps the existing `recursive_ref` / `cross_ref` nodes and instead uses `$need` demand-closure seeding to restrict recursion to the reachable subspace for the given inputs.

### Future: `?` (“any”) modes / multi-entrypoint compilation

Supporting `?` properly in query mode is not just a matter of final output filtering: the declared **input positions** influence the seeded pipeline and the `$need` demand-closure that scopes recursion. As a result, `?` implies compiling **multiple concrete mode variants**.

Two pragmatic approaches:

1. **Multiple explicit `user:mode/1` declarations per predicate** (recommended incremental path):
   - Users declare the concrete modes they actually want (e.g., `mode(p(+, -)).`, `mode(p(-, +)).`).
   - Codegen emits one entrypoint per declaration.

2. **Treat `?` as sugar that expands to concrete modes**:
   - Expand `mode(p(?, -)).` into a bounded set of concrete modes (e.g., `{mode(p(+, -)), mode(p(-, -))}`).
   - Emit one plan/entrypoint per expanded mode, optionally with a dispatcher that selects a plan based on which arguments are provided.

Guardrails to avoid a `2^k` explosion (for `k` question-marks) may be required: explicit opt-in, a hard cap on expansions, or only expanding patterns observed in tests.

### 3. Compilation Changes

#### 3.1 Mode Analysis (`csharp_target.pl`)

Add a new analysis phase after clause gathering:

```prolog
%% analyze_modes(+HeadSpec, +Clauses, -Modes)
% Infer or validate modes from clause structure
analyze_modes(HeadSpec, Clauses, Modes) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    % Check for explicit mode declaration
    (   mode_declaration(Pred/Arity, DeclaredModes)
    ->  Modes = DeclaredModes
    ;   % Infer modes from clause structure
        infer_modes(Clauses, Arity, Modes)
    ).

%% infer_modes(+Clauses, +Arity, -Modes)
% Heuristic: if recursive clauses need arithmetic on arg N before recursion,
% arg N is likely an input
infer_modes(Clauses, Arity, Modes) :-
    length(Modes, Arity),
    maplist(=(output), Modes).  % Default: all outputs
```

#### 3.2 Clause Reordering for Parameterized Mode

When inputs are known, reorder clause body to:
1. First: arithmetic that computes from inputs
2. Then: recursive calls with computed arguments
3. Finally: result arithmetic

```prolog
%% reorder_for_parameterized(+Body, +InputVars, -ReorderedBody)
% Move arithmetic that only depends on inputs before recursive calls
reorder_for_parameterized(Body, InputVars, ReorderedBody) :-
    body_to_list(Body, Goals),
    partition(goal_depends_only_on(InputVars), Goals, PreGoals, PostGoals),
    append(PreGoals, PostGoals, ReorderedGoals),
    list_to_body(ReorderedGoals, ReorderedBody).
```

#### 3.3 Build Parameterized Pipeline

New pipeline construction for parameterized mode:

```prolog
%% build_parameterized_pipeline(+HeadSpec, +Modes, +Clauses, -Plan)
build_parameterized_pipeline(HeadSpec, Modes, Clauses, Plan) :-
    % Identify input positions
    findall(Pos, (nth0(Pos, Modes, input)), InputPositions),

    % Build initial node from input parameters (not relation scan)
    build_parameter_seed_node(HeadSpec, InputPositions, SeedNode, InitialVarMap),

    % Process body with pre-bound inputs
    ...
```

### 4. Code Generation Changes

#### 4.1 Method Signature Generation

Generate methods with input parameters:

```csharp
// Current (all-output mode)
public static IEnumerable<(long, long)> FibQuery(IRelationProvider provider)

// Proposed (parameterized mode)
public static IEnumerable<long> FibQuery(IRelationProvider provider, long n)
// Or for multiple outputs:
public static (long F) FibQuery(IRelationProvider provider, long n)
```

#### 4.2 BindExpression Node Rendering

```csharp
// Render bind_expr node as LINQ Select that computes new columns
.Select(tuple => (
    tuple.Item1,  // Preserve existing
    tuple.Item2,
    tuple.Item1 - 1,  // N1 = N - 1
    tuple.Item1 - 2   // N2 = N - 2
))
```

#### 4.3 Parameterized Recursive Call Rendering

```csharp
// Instead of joining with recursive_ref, call recursively with computed args
.SelectMany(tuple => {
    var f1 = FibQuery(provider, tuple.Item3);  // fib(N1, F1)
    var f2 = FibQuery(provider, tuple.Item4);  // fib(N2, F2)
    return f1.Zip(f2, (a, b) => (tuple.Item1, a + b));
})
```

### 5. Runtime Changes

#### 5.1 Parameter Seeding

Add runtime support for initial parameter bindings:

```csharp
public class ParameterSeedNode : IQueryNode
{
    private readonly object[] _parameters;

    public ParameterSeedNode(params object[] parameters)
    {
        _parameters = parameters;
    }

    public IEnumerable<object[]> Execute(IRelationProvider provider)
    {
        // Emit single tuple with parameter values
        yield return _parameters;
    }
}
```

#### 5.2 BindExpressionNode

```csharp
public class BindExpressionNode : IQueryNode
{
    private readonly IQueryNode _input;
    private readonly Func<object[], object[]> _binder;

    public IEnumerable<object[]> Execute(IRelationProvider provider)
    {
        foreach (var tuple in _input.Execute(provider))
        {
            yield return _binder(tuple);
        }
    }
}
```

### 6. Example: Fibonacci Compilation

#### Input Prolog

```prolog
:- mode(fib(+, -)).

fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
```

#### Generated C# (Conceptual)

```csharp
public static class FibModule
{
    // Base case facts
    private static readonly Dictionary<long, long> BaseCases = new()
    {
        { 0, 0 },
        { 1, 1 }
    };

    // Parameterized query entry point
    public static long Fib(long n)
    {
        // Check base cases first
        if (BaseCases.TryGetValue(n, out var baseResult))
            return baseResult;

        // Guard: N > 1
        if (n <= 1)
            throw new InvalidOperationException("No matching clause");

        // Compute arguments for recursive calls
        var n1 = n - 1;
        var n2 = n - 2;

        // Recursive calls
        var f1 = Fib(n1);
        var f2 = Fib(n2);

        // Result arithmetic
        return f1 + f2;
    }
}
```

### 7. Backward Compatibility

- Predicates without mode declarations default to all-output (current behavior)
- Existing query runtime code paths remain unchanged
- Parameterized mode is opt-in via mode declarations
- The `?` mode can generate both enumeration and parameterized entry points

### 8. Implementation Phases

#### Phase 1: Mode Declaration Parsing
- Add `mode/1` directive parsing to `csharp_target.pl`
- Store mode information in predicate metadata
- No code generation changes yet

#### Phase 2: Compile-Time Analysis
- Implement mode inference heuristics
- Add clause reordering for parameterized mode
- Validate that inputs are sufficient for arithmetic

#### Phase 3: IR Extensions
- Add `bind_expr` node type
- Add `param_recursive_ref` node type
- Extend plan metadata with mode information

#### Phase 4: Code Generation
- Generate parameterized method signatures
- Render new node types to C#
- Handle base case optimization

#### Phase 5: Runtime Support
- Add `ParameterSeedNode` and `BindExpressionNode` to runtime
- Optimize for memoization (optional)

### 9. Open Questions

1. **Memoization**: Should parameterized recursive queries automatically memoize? This would make Fibonacci O(n) instead of O(2^n).

2. **Multiple Modes**: How to handle predicates that work in multiple modes (e.g., `append/3` can be used to split or join lists)?

3. **Partial Binding**: What if only some inputs are provided? Generate a partially-specialized query?

4. **Type Inference**: How to infer C# types for parameters? Currently relations provide type hints.

5. **Error Handling**: What if a parameterized query has no solutions? Return default? Throw? Optional type?

### 10. Alternatives Considered

#### A. Generator Mode Only
Keep query mode limited to datalog patterns, use generator mode for recursive arithmetic.
- **Pro**: No query runtime changes
- **Con**: Two separate code paths, no unified model

#### B. Specialized Numeric Recursion
Pattern-match simple numeric recursion (like Fibonacci) and generate iterative C#.
- **Pro**: Efficient for specific patterns
- **Con**: Limited applicability, pattern matching complexity

#### C. Full Prolog Interpreter
Embed a Prolog interpreter for complex queries.
- **Pro**: Complete Prolog semantics
- **Con**: Performance overhead, defeats purpose of compilation

### 11. Conclusion

Parameterized queries provide a principled extension to the query runtime that:
- Maintains the relational/pipeline model
- Enables function-style invocation
- Supports recursive predicates with computed arguments
- Is backward compatible with existing queries

The key insight is that inputs are just pre-bound columns in the tuple, and the pipeline can start from a parameter seed rather than a relation scan. This unifies enumeration and function queries under the same model.

## References

- `docs/development/analysis/IS_PREDICATE_COMPATIBILITY_ANALYSIS.md` - Initial analysis
- `docs/development/analysis/QUERY_VS_GENERATOR_FIBONACCI.md` - Codex analysis
- `src/unifyweaver/targets/csharp_target.pl` - Current query target implementation
