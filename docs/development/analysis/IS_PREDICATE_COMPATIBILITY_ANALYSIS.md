# Analysis: `is/2` Predicate Compatibility with C# Query Runtime

**Date:** 2025-12-08
**Analyst:** Claude Opus 4.5
**Request:** Review by Codex-5.1 for potential resolution

## Executive Summary

While attempting to run `playbooks/csharp_query_playbook.md`, I discovered that the Fibonacci example fails to compile. After investigation, I found that:

1. The `is/2` predicate IS supported for derived-column computation **when all RHS variables are already bound**.
2. In **all‑output** query mode (default), Fibonacci‑style recursion still fails due to a fundamental mismatch with the relation‑first join pipeline.
3. In **parameterized** query mode (via `mode/1`, e.g. `:- mode(fib(+, -)).`), the C# query target now seeds bindings from inputs and uses a bottom‑up demand‑closure (`pred$need`) fixpoint so Fibonacci‑style recursion can compile and run for eligible predicates (non‑mutual, no negation/aggregates).
4. A secondary bug (module prefix not being stripped) was fixed during investigation.

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

I recommend Codex-5.1 review Options B or C if supporting recursive arithmetic predicates is a priority, or Option D if the current scope should be limited to datalog-style queries.
