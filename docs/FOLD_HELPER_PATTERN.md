# Tree Recursion with Fold Helper Pattern

## Overview

The **fold helper pattern** is a two-phase approach to tree recursion that separates structure building from computation:

1. **Phase 1: Build Structure** - Recursively construct a tree/graph representing dependencies
2. **Phase 2: Fold Structure** - Traverse the structure to compute the final value

## Why Use This Pattern?

### Benefits

- **Visualization**: The intermediate structure can be inspected, exported, or visualized
- **Debugging**: Easier to understand and debug complex recursive relationships
- **Caching**: Structure and computation can be cached separately
- **Parallelization**: Structure makes parallelization opportunities explicit
- **Separation of Concerns**: Structure logic separate from computation logic

### When to Use

Use this pattern when:
- You need to visualize or debug recursive dependencies
- The structure itself has value beyond just the computed result
- You want to cache intermediate structures
- Multiple computations share the same structure (e.g., Pascal's triangle)
- You're using `forbid_linear_recursion/1` to force tree recursion

## Pattern Structure

### General Template

```prolog
% Phase 1: Build structure
predicate_graph(BaseCase, leaf(Value)).
predicate_graph(Input, node(Input, [Child1, Child2, ...])) :-
    RecursiveCondition,
    ComputeInputs,
    predicate_graph(Input1, Child1),
    predicate_graph(Input2, Child2),
    ...

% Phase 2: Fold structure
fold_predicate(leaf(V), V).
fold_predicate(node(_, Children), Result) :-
    maplist(fold_predicate, Children, Values),
    ComputeResult(Values, Result).

% Wrapper: Build then fold
predicate_fold(Input, Result) :-
    predicate_graph(Input, Graph),
    fold_predicate(Graph, Result).
```

## Example 1: Fibonacci Sequence

### Traditional Fibonacci (Linear or Tree)

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

**Problem**: Can't see the dependency structure, hard to visualize.

### Fibonacci with Fold Helper

```prolog
% Force tree recursion instead of linear
:- forbid_linear_recursion(fib/2).

% Phase 1: Build dependency tree
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib_graph(N1, L),
    fib_graph(N2, R).

% Phase 2: Fold to compute value
fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :-
    fold_fib(L, VL),
    fold_fib(R, VR),
    V is VL + VR.

% Wrapper
fib_fold(N, F) :-
    fib_graph(N, Graph),
    fold_fib(Graph, F).
```

**Benefits**:
- Can inspect `fib_graph(5, G)` to see the tree structure
- Can visualize the recursion tree
- Can cache the graph structure separately from values

### Example Session

```prolog
?- fib_graph(3, G).
G = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]).

?- fib_fold(5, F).
F = 5.

?- fib_graph(4, G), fold_fib(G, F).
G = node(4, [node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]), node(2, [leaf(1), leaf(0)])]),
F = 3.
```

## Example 2: Binomial Coefficients (Pascal's Triangle)

### Traditional Binomial

```prolog
binom(_, 0, 1).
binom(N, N, 1).
binom(N, K, C) :-
    N > 0, K > 0, K < N,
    N1 is N - 1,
    K1 is K - 1,
    binom(N1, K1, C1),
    binom(N1, K, C2),
    C is C1 + C2.
```

### Binomial with Fold Helper

```prolog
% Phase 1: Build Pascal's triangle structure
binom_graph(_, 0, leaf(1)).
binom_graph(N, N, leaf(1)).
binom_graph(N, K, node(N, K, [L, R])) :-
    N > 0, K > 0, K < N,
    N1 is N - 1,
    K1 is K - 1,
    binom_graph(N1, K1, L),
    binom_graph(N1, K, R).

% Phase 2: Fold to compute coefficient
fold_binom(leaf(V), V).
fold_binom(node(_, _, [L, R]), V) :-
    fold_binom(L, VL),
    fold_binom(R, VR),
    V is VL + VR.

% Wrapper
binom_fold(N, K, C) :-
    binom_graph(N, K, Graph),
    fold_binom(Graph, C).
```

**Extra benefit**: Can compute entire rows efficiently:

```prolog
% Compute full row of Pascal's triangle
pascal_row(N, Row) :-
    findall(C, (between(0, N, K), binom(N, K, C)), Row).

?- pascal_row(5, Row).
Row = [1, 5, 10, 10, 5, 1].
```

## Pattern Detection and Compilation

### Current State

Currently, this pattern requires **manual implementation** - you write both the `_graph` and `fold_` predicates yourself.

### Future: Automatic Detection

In the future, UnifyWeaver could:

1. Detect when `forbid_linear_recursion/1` is used
2. Automatically generate both phases:
   - Phase 1: Tree builder returning structure
   - Phase 2: Fold operation computing result
3. Generate bash code for both phases
4. Allow visualization/export of structures

### Forcing This Pattern

```prolog
% Mark predicate to use tree recursion instead of linear
:- forbid_linear_recursion(fib/2).

% Your regular recursive definition
fib(0, 0).
fib(1, 1).
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.
```

UnifyWeaver will then compile using tree recursion instead of linear recursion with memoization.

## Visualization

The structure can be visualized using various formats:

### DOT Format (Graphviz)

```prolog
% Convert tree to DOT format
tree_to_dot(leaf(V), DotCode) :-
    format(atom(DotCode), '  "~w";', [V]).
tree_to_dot(node(N, Children), DotCode) :-
    format(atom(NodeCode), '  "node_~w";', [N]),
    maplist(tree_to_dot, Children, ChildCodes),
    findall(Edge,
        (   member(Child, Children),
            child_id(Child, ChildId),
            format(atom(Edge), '  "node_~w" -> "~w";', [N, ChildId])
        ),
        Edges),
    atomic_list_concat([NodeCode | ChildCodes, Edges], '\n', DotCode).
```

### Tree Visualization

```
fib(5) tree structure:
         5
        / \
       4   3
      / \ / \
     3  2 2  1
    /\ /\ /\
   2 1 1 0 1 0
  /\
 1 0
```

## Performance Considerations

### Memory

- **Structure Phase**: O(tree size) memory to store structure
- **Fold Phase**: O(tree depth) stack space

### Computation

- **Without Memoization**: O(2^n) for fibonacci (exponential)
- **With Structure Caching**: Compute structure once, fold multiple times
- **Compared to Linear**: Linear with memo is O(n), but fold pattern allows visualization

### Trade-offs

| Aspect | Fold Pattern | Direct Recursion |
|:-------|:-------------|:-----------------|
| Visualization | ✓ Easy | ✗ Hard |
| Debugging | ✓ Clear structure | ✗ Opaque |
| Memory | Higher (stores structure) | Lower |
| Speed | Slower (two passes) | Faster (one pass) |
| Flexibility | ✓ Structure reusable | ✗ Single use |

## Best Practices

1. **Use for Complex Recursion**: When dependencies are hard to understand
2. **Use for Shared Structures**: When multiple computations use same structure
3. **Use for Debugging**: When you need to see what's happening
4. **Don't Use for Simple Cases**: Overkill for simple linear recursion
5. **Consider Caching**: Cache structures if used multiple times

## Integration with UnifyWeaver

### Current Usage

```prolog
% Define your _graph and fold_ predicates manually
% Then use them in your code
?- fib_fold(10, F).
F = 55.
```

### Future Auto-Generation

```prolog
% Just mark the predicate
:- use_fold_pattern(fib/2).

% UnifyWeaver generates both phases automatically
% Compile to bash with tree+fold pattern
?- compile_recursive(fib/2, [pattern(tree_fold)], BashCode).
```

## Examples

See the `examples/` directory for complete working examples:

- `examples/fibonacci_fold.pl` - Fibonacci with fold pattern
- `examples/binomial_fold.pl` - Binomial coefficients (Pascal's triangle)

## Testing

```bash
# Test fibonacci fold pattern
swipl -g "['examples/fibonacci_fold'], fib_fold(10, F), format('Result: ~w~n', [F]), halt."

# Test binomial fold pattern
swipl -g "['examples/binomial_fold'], binom_fold(5, 2, C), format('Result: ~w~n', [C]), halt."

# Visualize structure
swipl -g "['examples/fibonacci_fold'], fib_graph(5, G), writeln(G), halt."
```

## References

- `examples/fibonacci_fold.pl` - Complete fibonacci example
- `examples/binomial_fold.pl` - Complete binomial example
- `src/unifyweaver/core/advanced/tree_recursion.pl` - Tree recursion compiler
- `src/unifyweaver/core/advanced/pattern_matchers.pl` - Pattern detection with `forbid_linear_recursion/1`

## Future Work

- [ ] Automatic detection of fold-suitable patterns
- [ ] Auto-generation of `_graph` and `fold_` predicates
- [ ] Bash compilation for two-phase execution
- [ ] Structure visualization tools
- [ ] Structure caching and reuse
- [ ] Parallelization based on structure
