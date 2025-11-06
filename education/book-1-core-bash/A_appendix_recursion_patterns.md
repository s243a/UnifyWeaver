<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Appendix: Formal Recursion Patterns

This appendix provides formal definitions for the recursion patterns recognized by UnifyWeaver.

## 1. Transitive Closure

**Definition:** A transitive closure pattern is used to find all reachable nodes in a graph from a starting node. It is defined by a predicate that has a base case and a recursive step, where the recursive step calls itself with a different argument that is derived from the previous one.

**Example:** The `ancestor/2` predicate is a classic example of a transitive closure.

```prolog
% Base case: A is an ancestor of D if A is a parent of D.
ancestor(A, D) :- parent(A, D).

% Recursive step: A is an ancestor of D if A is a parent of P,
% and P is an ancestor of D.
ancestor(A, D) :-
    parent(A, P),
    ancestor(P, D).
```

**UnifyWeaver Optimization:** UnifyWeaver optimizes transitive closures by using a breadth-first search (BFS) algorithm, which is implemented using a work queue. This is highly efficient and avoids the risk of stack overflows for deep graphs.

## 2. Tail Recursion

**Definition:** A tail-recursive predicate is one where the recursive call is the very last operation in the predicate. There is no computation, aggregation, or other logic after the recursive call returns. This pattern is often characterized by the use of an accumulator argument that carries the state.

**Example:** A tail-recursive predicate to count the items in a list.

```prolog
% Base Case: When the list is empty, the final count is the accumulator.
count_items([], Acc, Acc).

% Recursive Step: For a non-empty list, increment the accumulator
% and recurse on the tail of the list.
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N).
```

**UnifyWeaver Optimization:** UnifyWeaver transforms tail-recursive predicates into highly efficient iterative loops in Bash. This avoids the overhead of recursive function calls and prevents stack overflow errors for very deep recursions.

## 3. Linear Recursion

**Definition:** A linear-recursive predicate is one that has one or more **independent** recursive calls. The arguments to these calls are computed as scalar values before the calls are made, and the results are aggregated after the calls return.

**Example:** The `fibonacci/2` predicate is a classic example of linear recursion with multiple independent calls.

```prolog
% Base Cases
fib(0, 0).
fib(1, 1).

% Recursive Step: Two independent recursive calls
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
```

**UnifyWeaver Optimization:** UnifyWeaver uses **memoization** to optimize linear recursion. It caches the results of each call in an associative array. If the same sub-problem is encountered again, the result is returned from the cache, avoiding re-computation.

## 4. Tree Recursion

**Definition:** A tree-recursive predicate is one that operates on a data structure and recursively calls itself on smaller, decomposed parts of that structure. This pattern is defined by its use of **structural decomposition**.

**Example:** A predicate to sum the values in a binary tree.

```prolog
% Tree is represented as [Value, LeftSubtree, RightSubtree]

% Base Case: The sum of an empty tree is 0.
tree_sum([], 0).

% Recursive Step: The sum of a tree is the value of the root
% plus the sum of the left and right subtrees.
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),
    tree_sum(R, RS),
    Sum is V + LS + RS.
```

**UnifyWeaver Optimization:** UnifyWeaver generates Bash code that can parse the data structure. It does not use memoization by default for tree recursion, as the structure itself is the primary focus.

## 5. Mutual Recursion

**Definition:** Mutual recursion occurs when two or more predicates call each other in a cycle.

**Example:** The `is_even/1` and `is_odd/1` predicates are a classic example of mutual recursion.

```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```

**UnifyWeaver Optimization:** UnifyWeaver detects mutually recursive predicates by finding **Strongly Connected Components (SCCs)** in the call graph. It then generates a **shared memoization table** for all predicates in the SCC, ensuring that results are cached and shared correctly across the entire group of functions.