# Analysis: `is/2` Predicate Compatibility with C# Query Runtime

**Date:** 2025-12-08
**Analyst:** Claude Opus 4.5
**Request:** Review by Codex-5.1 for potential resolution

## Status Update (2025-12-14)

Since this analysis was written, the query-mode compiler/runtime has expanded support for:
- `is/2` with a **bound** LHS (or numeric literal LHS) as a **check**: compile RHS into a temp column and filter by equality.
- Arithmetic expressions inside comparisons (e.g. `X+1 =:= 6`) by rewriting through temp arithmetic columns.
- Disjunction (`;/2`) in **rule bodies** by expanding into multiple clause variants and emitting a `union` plan node.
- Disjunction in **aggregate goals** (including nested disjunction inside conjunction) by compiling the aggregate goal as a union-of-branches subplan.
- Multi-mode declarations (`user:mode/1`) for a predicate now result in multiple query-mode plans/entrypoints; Prolog helpers exist to build/select variants (`build_query_plans/3`, `build_query_plan_for_inputs/4`).
- The C# query runtime (`src/unifyweaver/targets/csharp_query_runtime/`) is now dependency-free at the core, with opt-in runtime bundles:
  - `UnifyWeaver.QueryRuntime.Core.csproj` (no external deps)
  - `UnifyWeaver.QueryRuntime.Pearltrees.csproj` (LiteDB integration)
  - `UnifyWeaver.QueryRuntime.Onnx.csproj` (ONNX embedding provider)

These changes improve practical compatibility, but they do not change the core limitation: all-output query mode still cannot compile Fibonacci-style recursion because it has no relation scan to seed bindings for `N`, and recursion arguments must be computed before recursive calls.

## Executive Summary

While attempting to run `playbooks/csharp_query_playbook.md`, I discovered that the Fibonacci example fails to compile. After investigation, I found that:

1. The `is/2` predicate IS supported for derived-column computation **when all RHS variables are already bound**.
2. In **all‑output** query mode (default), Fibonacci‑style recursion still fails due to a fundamental mismatch with the relation‑first join pipeline.
3. In **parameterized** query mode (via `mode/1` input positions), the C# query target seeds bindings from inputs and uses a bottom‑up demand‑closure (`pred$need`) fixpoint so Fibonacci‑style recursion can compile and run for eligible predicates (currently non‑mutual).
4. A secondary bug (module prefix not being stripped) was fixed during investigation.
5. Since this analysis was written, query mode gained **bound-only stratified negation** and **correlated/grouped aggregates**; these features still have safety/stratification restrictions and do not change the all-output Fibonacci limitation.
6. Since this analysis was written, query mode also gained **bound-LHS `is/2` checks**, **arithmetic-in-comparisons**, and **disjunction support** (rule bodies and aggregate goals).

> Note on `mode/1` syntax: SWI-Prolog does not provide `:- mode(...)` as a built-in directive. In this repo, modes are read from `user:mode/1` facts (e.g., `assertz(user:mode(fib(+, -))).` in tests).

## Task Context

The playbook at `playbooks/csharp_query_playbook.md` references example code from `playbooks/examples_library/csharp_examples.md` which originally contained a Fibonacci example:

```prolog
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

## Issues Discovered

### Issue 1: Module Prefix Not Stripped (FIXED)

**Location:** `src/unifyweaver/targets/csharp_target.pl:466-475`

When clauses are retrieved via `clause/2`, the body may be wrapped in a module qualifier like `user:(A, B, C, ...)`. The original `body_to_list/2` did not handle this:

```prolog
% Original code (lines 466-471)
body_to_list(true, []) :- !.
body_to_list((A, B), Terms) :- !,
    body_to_list(A, TA),
    body_to_list(B, TB),
    append(TA, TB, Terms).
body_to_list(Goal, [Goal]).
```

**Error produced:** `C# query target: no facts available for :/2`

**Fix applied:** Added clause to strip module prefix:

```prolog
% Fixed code (lines 466-475)
body_to_list(true, []) :- !.
body_to_list(Body, Terms) :-
    compound(Body),
    Body = _Module:InnerBody, !,
    body_to_list(InnerBody, Terms).
body_to_list((A, B), Terms) :- !,
    body_to_list(A, TA),
    body_to_list(B, TB),
    append(TA, TB, Terms).
body_to_list(Goal, [Goal]).
```

### Issue 2: Clause Beginning with Constraint (Architectural)

**Location:** `src/unifyweaver/targets/csharp_target.pl:482-486`

After fixing Issue 1, the Fibonacci example fails with:
```
C# query target: clause for fib/2 begins with a constraint; reorder body literals.
```

The query compiler requires clauses to begin with a relation (to start the LINQ pipeline), not a constraint:

```prolog
% Lines 482-486
build_initial_node(HeadSpec, _GroupSpecs, _Term, constraint, _Node, _Relations, _VarMap, _Width) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    format(user_error, 'C# query target: clause for ~w/~w begins with a constraint; reorder body literals.~n', [Pred, Arity]),
    fail.
```

The Fibonacci recursive clause starts with `N > 1` (a comparison constraint), but the architecture expects a relation scan as the first operation.

### Issue 3: Recursive Arithmetic Pattern Mismatch (Fundamental)

**This is the core architectural issue.**

Even if we reorder the clause to put recursive calls first:

```prolog
fib(N, F) :-
    fib(N1, F1),      % Recursive call first
    fib(N2, F2),
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    F is F1 + F2.
```

This fails with: `C# query target: variable _X not bound before constraint evaluation.`

**Why this happens:**

The query runtime uses a **relational join model** where:
1. Relations are scanned to produce tuples
2. Joins combine tuples based on shared variables
3. Constraints filter tuples based on bound values
4. Arithmetic computes derived columns from existing bound columns

**Location:** `src/unifyweaver/targets/csharp_target.pl:567-620` (constraint node building)

```prolog
% Lines 567-620 show how arithmetic goals are processed
build_constraint_node(Term, InputNode, VarMapIn, WidthIn,
        NodeOut, VarMapOut, WidthOut) :-
    (   arithmetic_goal(Term)
    ->  Term = (Left is Right),
        (   var(Left)
        ->  (   var_bound(Left, VarMapIn)
            ->  format(user_error, 'C# query target: variable ~w already bound before is/2 evaluation.~n', [Left]),
                fail
            ;   compile_arithmetic_expression(Right, VarMapIn, ArithExpr),
                ...
            )
        ;   format(user_error, 'C# query target: left operand of is/2 must be an unbound variable (~q).~n', [Term]),
            fail
        )
    ...
```

The key constraint is at line 601: `compile_arithmetic_expression(Right, VarMapIn, ArithExpr)` - the right-hand side of `is/2` must only reference **already bound** variables.

**In Fibonacci:**
- `N1 is N - 1` requires `N` to be bound
- But `fib(N1, F1)` is called BEFORE `N1` is computed
- The recursive call `fib(N1, F1)` uses `N1` as an INPUT to the join
- This creates a circular dependency

## Architectural Analysis

### How the Query Runtime Works

**Location:** `src/unifyweaver/targets/csharp_target.pl:252-341` (recursive plan building)

The query runtime generates C# code that:
1. Builds an initial relation scan from base case facts
2. Iteratively expands via fixpoint computation
3. Uses joins to combine recursive results

```prolog
% Lines 324-341: Recursive variant building
build_recursive_variants(HeadSpec, GroupSpecs, Clauses, Variants, Relations) :-
    ...
    findall(variant(Node, RelList),
        (   member(Head-Body, Clauses),
            build_recursive_clause_variants(HeadSpec, GroupSpecs, Head-Body, VariantStructs),
            member(variant(Node, RelList), VariantStructs)
        ),
        VariantPairs),
    ...
```

### What Fibonacci Needs vs What the Runtime Provides

| Fibonacci Requirement | Query Runtime Capability |
|----------------------|-------------------------|
| Compute N-1, N-2 first | Cannot - arithmetic requires bound inputs |
| Use computed values in recursive calls | Cannot - join keys must come from scanned tuples |
| Accumulate F1 + F2 | CAN do this - final arithmetic projection |

### Working `is/2` Pattern

The `sum_pair/3` example works because it follows the supported pattern:

```prolog
sum_pair(X, Y, Sum) :-
    num_pair(X, Y),    % Relation scan binds X, Y
    Sum is X + Y.      % Arithmetic uses bound X, Y
```

Generated C# code (`csharp_target`):
```csharp
new ArithmeticNode(
    new RelationScanNode(new PredicateId("num_pair", 2)),
    new BinaryArithmeticExpression(
        ArithmeticBinaryOperator.Add,
        new ColumnExpression(0),  // X from tuple position 0
        new ColumnExpression(1)   // Y from tuple position 1
    ),
    2,  // Input width
    3   // Output width (adds Sum column)
)
```

### Failing Fibonacci Pattern

Fibonacci requires **computing argument values before the recursive call**, which the join-based model cannot express:

```
Standard Fibonacci evaluation:
  fib(5, F)
    → N1 = 5-1 = 4, N2 = 5-2 = 3
    → fib(4, F1), fib(3, F2)
    → F = F1 + F2

Query Runtime model:
  Cannot compute 4, 3 before issuing the recursive "query"
  The recursive call IS the query - arguments must be known
```

## Potential Resolutions for Codex-5.1 Review

### Option A: Detect and Reject with Clear Error

Add explicit detection for "recursive predicates with arithmetic argument computation" and provide a clear error message directing users to alternative approaches.

**Location to modify:** `src/unifyweaver/targets/csharp_target.pl` around line 343 (`build_recursive_clause_variants`)

### Option B: Transform to Iterative C# Code

For specific patterns like Fibonacci, generate iterative C# code with memoization:

```csharp
public static long Fib(int n) {
    var memo = new Dictionary<int, long> { {0, 0}, {1, 1} };
    for (int i = 2; i <= n; i++) {
        memo[i] = memo[i-1] + memo[i-2];
    }
    return memo[n];
}
```

This would require:
1. Pattern recognition for "simple recursive arithmetic"
2. New code generation path in the renderer
3. Special handling in `build_recursive_plan` (line 252)

### Option C: Extend the Query Plan IR

Add a new node type that represents "computed argument binding" before recursive calls:

```prolog
% Hypothetical new node type
compute_and_recurse{
    bindings: [N1 is N - 1, N2 is N - 2],
    recursive_calls: [fib(N1, F1), fib(N2, F2)],
    result: F is F1 + F2
}
```

This would require changes to:
- `src/unifyweaver/targets/csharp_target.pl` (IR construction)
- C# code renderer (new node type handling)
- Query runtime library (new node executor)

### Option D: Use Different Target for Recursive Arithmetic

Document that `csharp_stream_target` and `csharp_target` are for datalog-style queries, and recursive arithmetic predicates should use a different compilation strategy (e.g., direct Prolog interpretation or a dedicated functional target).

### Option E: Broaden `is/2` Support (Quality-of-Life)

This does **not** solve all-output Fibonacci, but improves compatibility for non-recursive code and for parameterized code that uses `is/2` as a *check*.

Historically, query mode treated `Var is Expr` as “compute a new column” and rejected `Value is Expr` / `BoundVar is Expr`.

Two incremental extensions:
1. **Allow bound-LHS `is/2` as a filter**: compile `X is Expr` (where `X` is already bound) into “compute Expr into a temp column, then compare to X”.
2. **Allow arithmetic expressions in comparisons** (`=:=`, `<`, `>`, etc.) by using the same temp-column rewrite.

Downside: more planner complexity, and it encourages using `is/2` as a constraint rather than separating “compute” (`is/2`) from “compare” (`=:=`). The rewrite approach keeps runtime changes minimal.

**Status:** Implemented in `feat/parameterized-queries-querymode` and covered by `tests/core/test_csharp_query_target.pl` (`verify_is_check_literal_plan`, `verify_is_check_bound_var_plan`, `verify_arith_expr_eq_plan`, etc.).

### Option F: Allow Ground Constants in Relation Literals (Quality-of-Life)

Query-mode joins currently expect relation call arguments to be variables; writing `p(alice, X)` (with a ground constant in a relation argument position) triggers a `domain_error(variable, alice)` during pipeline construction.

To make query mode feel closer to normal Prolog usage (and reduce boilerplate), we can desugar:

```prolog
p(alice, X)
```

into:

```prolog
p(A, X), A = alice
```

This can be done during clause preprocessing (before role assignment) without changing runtime semantics for pure constraints. It pairs naturally with the disjunction-expansion work now in place.

**Status:** Implemented in `feat/parameterized-queries-querymode` by normalizing query-mode clause terms (including aggregate subplan goals) so relation/recursive literals with simple constants are rewritten to fresh variables plus equality constraints.

## Files Referenced

| File | Lines | Purpose |
|------|-------|---------|
| `src/unifyweaver/targets/csharp_target.pl` | 466-475 | `body_to_list/2` - module stripping fix |
| `src/unifyweaver/targets/csharp_target.pl` | 482-486 | Constraint-first clause rejection |
| `src/unifyweaver/targets/csharp_target.pl` | 567-620 | Arithmetic constraint handling |
| `src/unifyweaver/targets/csharp_target.pl` | 252-341 | Recursive plan building |
| `src/unifyweaver/targets/csharp_target.pl` | 343-359 | Recursive variant construction |
| `playbooks/examples_library/csharp_examples.md` | All | Updated to use `sum_pair` example |

## Test Commands

```bash
# Working example (sum_pair with is/2)
swipl -l /tmp/test_csharp_target.pl  # Uses sum_pair/3

# All-output Fibonacci still fails (demonstrates the limitation)
# fib/2 without a mode declaration will still be rejected.

# Parameterized Fibonacci now works
swipl -q -t test_csharp_query_target -s tests/core/test_csharp_query_target.pl
# (See verify_parameterized_fib_plan / verify_parameterized_fib_runtime)
```

## Conclusion

The `is/2` predicate is supported for computing derived columns from bound tuple values. Fibonacci‑style recursion fails in the default (all‑output) relational pipeline because it requires computing recursive call arguments before those calls.

However, with explicit input modes (`mode/1`) the query target can treat head inputs as pre‑bound, compute argument deltas legally, and (via demand closure) restrict the fixpoint to the reachable subspace. This resolves the Fibonacci case without abandoning the bottom‑up query model, while preserving the existing datalog semantics for predicates without inputs.

Codex-5.2 recommendation:
- Keep default query mode focused on the relation-first/datalog pipeline (all-output), and continue to reject Fibonacci-style recursion in that mode.
- Support function-style recursive arithmetic via **parameterized query mode** (`user:mode/1`) plus demand-closure (`pred$need`), or use generator mode for patterns outside query mode’s scope.
- Prefer incremental UX improvements (Option A: clearer errors) over invasive IR changes (Option C) unless all-output recursion over computed arguments becomes a hard requirement.

Option F is now implemented and removes a common source of boilerplate/errors for query-mode bodies.
