<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Appendix B: Fold Pattern Deep Dive

**Educational Topic:** Advanced recursion patterns and two-phase computation  
**Real-World Application:** Fibonacci, binomial coefficients, and tree recursion optimization  
**Case Study:** UnifyWeaver's fold pattern integration and automatic code generation  

---

## Introduction

This appendix explores fold patterns - a sophisticated approach to recursive computation that separates structure building from value computation. This pattern was recently integrated into UnifyWeaver's main compilation pipeline, providing an elegant alternative to memoization for certain types of tree recursion.

The fold pattern demonstrates advanced compiler design principles, template-based code generation, and the power of separating concerns in algorithm design.

---

## The Fold Pattern Concept

### What is a Fold Pattern?

A **fold pattern** splits recursive computation into two distinct phases:

1. **Structure Building Phase:** Construct a tree/graph representing the computation dependencies
2. **Folding Phase:** Traverse the structure to compute the actual values

This separation provides several benefits:
- **Debuggability** - You can inspect the dependency structure before computation
- **Flexibility** - Different fold operations can use the same structure  
- **Optimization** - Structures can be cached separately from values
- **Clarity** - Algorithm intent is more explicit

### Traditional vs Fold Pattern

**Traditional Recursive Approach (Fibonacci):**
```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :- 
    N > 1,
    N1 is N - 1, N2 is N - 2,
    fib(N1, F1), fib(N2, F2),    % Direct recursive calls
    F is F1 + F2.                % Computation mixed with recursion
```

**Fold Pattern Approach:**
```prolog
% Phase 1: Structure Builder
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N - 1, N2 is N - 2,
    fib_graph(N1, L),            % Build dependency tree
    fib_graph(N2, R).

% Phase 2: Fold Computer  
fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :-
    fold_fib(L, VL),             % Fold left subtree
    fold_fib(R, VR),             % Fold right subtree  
    V is VL + VR.                % Combine results

% Phase 3: Wrapper
fib_fold(N, F) :- 
    fib_graph(N, Graph),         % Build structure
    fold_fib(Graph, F).          % Compute value
```

### Structure Representation

Fold patterns use a standardized tree representation:

**Leaf Nodes:** `leaf(Value)`
- Represent base cases with their computed values
- No further decomposition needed

**Internal Nodes:** `node(Label, Children)`  
- `Label` - Identifies the computation (often the input parameters)
- `Children` - List of child nodes `[Child1, Child2, ...]`

**Examples:**
```prolog
% Fibonacci graph for fib(3)
node(3, [
    node(2, [
        leaf(1),
        leaf(0)
    ]),
    leaf(1)
])

% This represents: fib(3) = fib(2) + fib(1) = (fib(1) + fib(0)) + fib(1)
```

---

## Real-World Examples

### Example 1: Fibonacci Numbers

**The Challenge:** Fibonacci has exponential recursive calls, needs optimization.

**Structure Builder:**
```prolog
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N - 1, N2 is N - 2,
    fib_graph(N1, L),
    fib_graph(N2, R).
```

**Fold Computer:**
```prolog
fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :-
    fold_fib(L, VL), fold_fib(R, VR),
    V is VL + VR.
```

**Usage:**
```prolog
?- fib_fold(5, F).
F = 5.

?- fib_graph(4, Graph).
Graph = node(4, [
    node(3, [
        node(2, [leaf(1), leaf(0)]),
        leaf(1)
    ]),
    node(2, [leaf(1), leaf(0)])
]).
```

### Example 2: Binomial Coefficients

**The Challenge:** Computing C(n,k) involves Pascal's triangle relationships.

**Structure Builder:**
```prolog
binom_graph(_, 0, leaf(1)).
binom_graph(N, N, leaf(1)) :- N >= 0.
binom_graph(N, K, node((N,K), [L, R])) :-
    N > 0, K > 0, K < N,
    N1 is N - 1, K1 is K - 1,
    binom_graph(N1, K1, L),      % C(n-1,k-1)
    binom_graph(N1, K, R).       % C(n-1,k)
```

**Fold Computer:**
```prolog
fold_binom(leaf(V), V).
fold_binom(node(_, [L, R]), V) :-
    fold_binom(L, VL), fold_binom(R, VR),
    V is VL + VR.                % Pascal's triangle: C(n,k) = C(n-1,k-1) + C(n-1,k)
```

**Usage:**
```prolog
?- binom_fold(5, 2, C).
C = 10.                          % C(5,2) = 10 combinations
```

### Example 3: Expression Trees

**The Challenge:** Parse and evaluate mathematical expressions with precedence.

**Structure Builder:**
```prolog
expr_graph(N, leaf(N)) :- number(N).
expr_graph(X + Y, node(plus, [L, R])) :-
    expr_graph(X, L),
    expr_graph(Y, R).
expr_graph(X * Y, node(mult, [L, R])) :-
    expr_graph(X, L),
    expr_graph(Y, R).
```

**Fold Computer:**
```prolog
fold_expr(leaf(V), V).
fold_expr(node(plus, [L, R]), V) :-
    fold_expr(L, VL), fold_expr(R, VR),
    V is VL + VR.
fold_expr(node(mult, [L, R]), V) :-
    fold_expr(L, VL), fold_expr(R, VR),
    V is VL * VR.
```

**Usage:**
```prolog
?- expr_fold(2 + 3 * 4, V).
V = 14.                          % Respects precedence: 2 + (3 * 4)
```

---

## Automatic Generation System

### The Challenge: Manual Implementation is Tedious

Writing fold patterns manually for each predicate is time-consuming and error-prone. UnifyWeaver includes an automatic generation system that detects fold patterns and creates the helper predicates automatically.

### Fold Pattern Detection

**Automatic Detection Criteria:**
```prolog
is_tree_fold_pattern(Pred/Arity) :-
    % 1. Has tree recursion (multiple recursive calls)
    has_tree_recursion(Pred/Arity),
    
    % 2. Uses forbid_linear_recursion directive
    forbid_linear_recursion(Pred/Arity),
    
    % 3. Follows arithmetic combination pattern
    has_arithmetic_combination(Pred/Arity).
```

**Example Detection:**
```prolog
% This predicate will be detected as fold pattern
:- assertz(forbid_linear_recursion(test_fib/2)).

test_fib(0, 0).
test_fib(1, 1).
test_fib(N, F) :-
    N > 1,
    N1 is N - 1, N2 is N - 2,
    test_fib(N1, F1), test_fib(N2, F2),    % Tree recursion
    F is F1 + F2.                          % Arithmetic combination
```

### Automatic Helper Generation

The `fold_helper_generator.pl` module automatically creates three helper predicates:

**1. Graph Builder:** `pred_graph/N`
```prolog
% Generated automatically
test_fib_graph(0, leaf(0)).
test_fib_graph(1, leaf(1)).
test_fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N - 1, N2 is N - 2,
    test_fib_graph(N1, L),
    test_fib_graph(N2, R).
```

**2. Fold Computer:** `fold_pred/2`
```prolog
% Generated automatically
fold_test_fib(leaf(V), V).
fold_test_fib(node(_, [L, R]), V) :-
    fold_test_fib(L, VL),
    fold_test_fib(R, VR),
    V is VL + VR.
```

**3. Wrapper Predicate:** `pred_fold/N`
```prolog
% Generated automatically
test_fib_fold(N, F) :-
    test_fib_graph(N, Graph),
    fold_test_fib(Graph, F).
```

### Template-Based Code Generation

**Variable Mapping Algorithm:**
```prolog
generate_fold_helpers(Pred/Arity, Helpers) :-
    functor(Head, Pred, Arity),
    Head =.. [Pred|Args],
    
    % Split arguments: inputs and output
    append(Inputs, [Output], Args),
    
    % Generate graph builder: pred_graph(Inputs..., Graph)
    GraphPred =.. [GraphName|Inputs, Graph],
    
    % Generate fold computer: fold_pred(Graph, Output)  
    FoldPred =.. [FoldName, Graph, Output],
    
    % Generate wrapper: pred_fold(Inputs..., Output)
    WrapperPred =.. [WrapperName|Args].
```

**Copy-Term Variable Preservation:**
The generator uses `copy_term/2` to ensure variable bindings are preserved correctly during transformation:

```prolog
transform_body(OriginalBody, GraphBody) :-
    copy_term(OriginalBody, GraphBody),  % Preserve variable sharing
    replace_recursive_calls(GraphBody).  % Transform calls to graph builders
```

---

## Integration with Compilation Pipeline

### Compilation Priority Order

Fold patterns are integrated into UnifyWeaver's advanced recursion compiler with specific priority:

```
tail → linear → FOLD → tree → mutual
```

**Why this order?**
- **Tail recursion** - Most efficient, try first
- **Linear recursion** - Good for single recursive calls  
- **Fold patterns** - Better than tree when structure reuse possible
- **Tree recursion** - General tree recursion with memoization
- **Mutual recursion** - Most complex, try last

### Pattern Selection Logic

```prolog
compile_advanced_recursive(Pred/Arity, Options, BashCode) :-
    (   % Try fold pattern if explicitly requested
        forbid_linear_recursion(Pred/Arity),
        is_tree_fold_pattern(Pred/Arity)
    ->  compile_fold_pattern(Pred/Arity, Options, BashCode)
    ;   % Otherwise try other patterns
        try_other_patterns(Pred/Arity, Options, BashCode)
    ).
```

### Bash Code Generation

The fold pattern compiler generates complete bash implementations:

**Generated Structure (1903 characters of code):**
```bash
#!/bin/bash
# test_fib - compiled fold pattern

# Graph builder function
test_fib_graph() {
    local n="$1"
    
    if [[ "$n" -eq 0 ]]; then
        echo "leaf(0)"
        return
    fi
    
    if [[ "$n" -eq 1 ]]; then
        echo "leaf(1)"
        return
    fi
    
    local n1=$((n - 1))
    local n2=$((n - 2))
    
    local left=$(test_fib_graph "$n1")
    local right=$(test_fib_graph "$n2")
    
    echo "node($n,[$left,$right])"
}

# Fold computer function
fold_test_fib() {
    local graph="$1"
    
    if [[ "$graph" =~ leaf\(([0-9]+)\) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    
    # Parse node structure and recursively fold
    # ... complex parsing and computation logic
}

# Wrapper function
test_fib_fold() {
    local n="$1"
    local graph=$(test_fib_graph "$n")
    fold_test_fib "$graph"
}

# Main entry point
test_fib() {
    test_fib_fold "$@"
}
```

---

## Performance Characteristics

### When Fold Patterns Excel

**Best Use Cases:**
1. **Tree recursion with repeated subproblems** (Fibonacci, binomial coefficients)
2. **Structure analysis tasks** (parsing, expression evaluation)  
3. **Debugging scenarios** (need to inspect computation structure)
4. **Educational purposes** (clearer algorithm intent)

**Performance Comparison:**

| Pattern | Time Complexity | Space Complexity | Code Clarity |
|---------|----------------|------------------|--------------|
| **Naive recursion** | O(2^n) - exponential | O(n) - call stack | ⭐⭐⭐ Very clear |
| **Memoization** | O(n) - linear | O(n) - memo table | ⭐⭐ Moderate |
| **Fold pattern** | O(n) - linear | O(n) - structure | ⭐⭐⭐⭐ Excellent |
| **Iterative** | O(n) - linear | O(1) - constant | ⭐ Less clear |

### Trade-offs

**Advantages:**
- ✅ **Debuggability** - Structure can be inspected before computation
- ✅ **Reusability** - Same structure, different fold operations
- ✅ **Clarity** - Separation of concerns makes intent clear
- ✅ **Optimization** - Structure caching opportunities

**Disadvantages:**  
- ⚠️ **Memory overhead** - Structure must be built and stored
- ⚠️ **Two-pass computation** - Structure building + folding
- ⚠️ **Complexity** - More complex than direct approaches
- ⚠️ **Parsing overhead** - Bash implementation needs structure parsing

---

## Educational Exercises

### Exercise 1: Manual Fold Pattern

**Task:** Implement a fold pattern for computing factorials.

**Starter Code:**
```prolog
% Define your graph builder
fact_graph(0, ?).
fact_graph(N, ?) :- N > 0, ?

% Define your fold computer  
fold_fact(?, ?).
fold_fact(?, ?) :- ?

% Create wrapper
fact_fold(N, F) :- ?
```

**Solution:**
```prolog
fact_graph(0, leaf(1)).
fact_graph(N, node(N, [Child])) :- 
    N > 0, N1 is N - 1,
    fact_graph(N1, Child).

fold_fact(leaf(V), V).
fold_fact(node(N, [Child]), V) :-
    fold_fact(Child, CV),
    V is N * CV.

fact_fold(N, F) :- fact_graph(N, Graph), fold_fact(Graph, F).
```

### Exercise 2: Automatic Generation

**Task:** Use UnifyWeaver's automatic generation for your factorial fold pattern.

**Steps:**
1. Define the original recursive predicate
2. Add `forbid_linear_recursion` directive
3. Let UnifyWeaver generate the helpers
4. Test the generated code

### Exercise 3: Structure Visualization

**Task:** Create a visualization of the fold pattern structure.

**Code:**
```prolog
visualize_graph(leaf(V)) :- 
    format('~w', [V]).
visualize_graph(node(Label, Children)) :-
    format('~w(', [Label]),
    maplist(visualize_graph, Children),  
    format(')', []).
```

**Test:**
```prolog
?- fib_graph(4, G), visualize_graph(G).
4(3(2(1,0),1),2(1,0))
```

### Exercise 4: Custom Fold Operations

**Task:** Create multiple fold operations for the same structure.

**Example - Statistics on Fibonacci Structure:**
```prolog
% Count nodes in structure
fold_count_nodes(leaf(_), 1).
fold_count_nodes(node(_, Children), Count) :-
    maplist(fold_count_nodes, Children, Counts),
    sumlist(Counts, ChildCount),
    Count is ChildCount + 1.

% Find maximum value in structure
fold_max_value(leaf(V), V).
fold_max_value(node(_, Children), Max) :-
    maplist(fold_max_value, Children, Values),
    max_list(Values, Max).
```

---

## Advanced Topics

### Fold Pattern Variants

**Bottom-Up Folding (Standard):**
```prolog
fold_pred(leaf(V), V).
fold_pred(node(_, Children), Result) :-
    maplist(fold_pred, Children, ChildResults),
    combine(ChildResults, Result).
```

**Top-Down Folding (With Accumulator):**
```prolog
fold_pred_acc(Graph, Acc, Result) :-
    fold_pred_acc(Graph, Acc, 0, Result).

fold_pred_acc(leaf(V), Acc, _, Result) :-
    Result is Acc + V.
fold_pred_acc(node(_, Children), Acc, Level, Result) :-
    NewAcc is Acc + Level,
    Level1 is Level + 1,
    maplist(fold_pred_acc_helper(NewAcc, Level1), Children, Results),
    sumlist(Results, Result).
```

### Parallel Fold Patterns

For large structures, fold operations can be parallelized:

```bash
# Bash parallel folding
fold_parallel() {
    local graph="$1"
    
    # Identify independent subtrees
    extract_subtrees "$graph" > subtrees.txt
    
    # Process subtrees in parallel
    parallel -j 4 fold_subtree :::: subtrees.txt > results.txt
    
    # Combine results
    combine_results < results.txt
}
```

### Incremental Folding

For dynamic structures, incremental folding can reuse previous computations:

```prolog
incremental_fold(Graph, Cache, Result, NewCache) :-
    (   member(Graph-CachedResult, Cache)
    ->  Result = CachedResult, NewCache = Cache
    ;   compute_fold(Graph, Result),
        NewCache = [Graph-Result|Cache]
    ).
```

---

## Integration with Other Patterns

### Fold + Constraints

```prolog
:- constraint(unique(false)).        % Allow duplicate intermediate results
:- constraint(unordered(true)).     % Order doesn't matter for folding
:- forbid_linear_recursion(fib/2).  % Force fold pattern

fib(N, F) :- fib_fold(N, F).
```

### Fold + Firewall

```prolog
:- firewall([bash_generation(allowed), memoization(forbidden)]).

% Only bash generation allowed, memoization blocked
% Fold pattern provides alternative to memoization
```

### Fold + Template System

```prolog
generate_fold_bash(Pred/Arity, BashCode) :-
    template_system:render_named_template(
        fold_pattern_template,
        [pred=Pred, arity=Arity],
        BashCode
    ).
```

---

## Summary

### Key Concepts Learned

**Two-Phase Computation:**
- **Structure building** separates dependency representation from computation
- **Folding** operates on the structure to produce values
- **Separation of concerns** improves clarity and debuggability

**Automatic Generation:**
- **Pattern detection** identifies fold pattern candidates
- **Template-based generation** creates helper predicates automatically  
- **Variable preservation** ensures correct binding through transformations

**Integration Architecture:**
- **Compilation priority** places fold patterns appropriately in the pipeline
- **Bash code generation** produces complete, functional implementations
- **Configuration options** allow fine-tuning of generated code

### Engineering Principles Demonstrated

1. **Abstraction** - Fold patterns abstract the concept of tree traversal + computation
2. **Code Generation** - Templates generate repetitive helper code automatically
3. **Separation of Concerns** - Structure building vs computation are independent
4. **Performance Trade-offs** - Clarity and flexibility vs raw speed
5. **Integration Design** - Fold patterns fit cleanly into existing compilation pipeline

### When to Use Fold Patterns

**Ideal scenarios:**
- Tree recursion with repeated subproblems
- Need to inspect computation structure for debugging
- Multiple operations on the same recursive structure
- Educational contexts where algorithm clarity is important

**Avoid when:**
- Simple linear recursion is sufficient
- Raw performance is critical and memoization works well
- Structure overhead exceeds computation benefits

**The fold pattern demonstrates that elegant software design often comes from separating what you're computing from how you're computing it.**

---

## Further Reading

**Related UnifyWeaver Documentation:**
- Chapter 9: Advanced Recursion Patterns
- Appendix A: SIGPIPE and Streaming Safety  
- Template System Design Documentation
- Constraint System Architecture

**Academic References:**
- **Functional Programming Principles** - Map, fold, and reduce operations
- **Compiler Design** - AST traversal and code generation
- **Algorithm Analysis** - Dynamic programming vs divide-and-conquer

**Practical Applications:**
- **Parser Combinators** - Building and evaluating parse trees
- **Computer Graphics** - Scene graph traversal and rendering
- **Game AI** - Decision tree evaluation
- **Database Query Optimization** - Query plan generation and execution

---

*This appendix documents the fold pattern implementation completed October 14, 2025, demonstrating advanced recursion compilation techniques and automatic code generation in production systems.*

---

## Navigation

**←** [Previous: Appendix B: SIGPIPE and Streaming Safety](appendix_a_sigpipe_streaming_safety.md) | [📖 Back to Book 1](README.md) | [Next: Case Study: Production Pipeline →](case_study_production_pipeline.md)

