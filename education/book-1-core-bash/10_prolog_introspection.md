<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 10: Prolog Introspection and Theory

## Introduction

How does UnifyWeaver know whether your predicate uses tail recursion, linear recursion, or mutual recursion? How does it analyze your code to detect these patterns automatically?

The answer lies in **Prolog introspection** - the ability of Prolog programs to examine and analyze other Prolog code. This chapter explores the sophisticated analysis techniques UnifyWeaver uses to understand your code and select the best compilation strategy.

UnifyWeaver's pattern detection system consists of three main components:
1. **Call Graph Construction** - Building a dependency graph of predicate calls
2. **SCC Detection** - Finding Strongly Connected Components (mutual recursion cycles)
3. **Pattern Matching** - Identifying specific recursion patterns

## 1. Prolog Introspection Basics

### The `clause/2` Predicate

The foundation of code analysis in Prolog is the `clause/2` predicate, which allows you to inspect the clauses (rules and facts) that define a predicate.

**Syntax:**
```prolog
clause(Head, Body)
```

- `Head` is the head of a clause (the part before `:-`)
- `Body` is the body of the clause (the part after `:-`)
- For facts, `Body` is `true`

**Example: Inspecting a predicate**

```prolog
% Define a simple predicate
parent(abraham, isaac).
parent(isaac, jacob).

% Query its clauses
?- clause(parent(X, Y), Body).
X = abraham, Y = isaac, Body = true ;
X = isaac, Y = jacob, Body = true.
```

### The `functor/3` Predicate

The `functor/3` predicate extracts or constructs terms with a specific functor and arity:

```prolog
functor(Term, Functor, Arity)
```

**Example:**
```prolog
?- functor(parent(abraham, isaac), F, A).
F = parent, A = 2.

?- functor(Head, ancestor, 2).
Head = ancestor(_A, _B).
```

This is crucial for pattern detection - we can construct a generic head for any predicate and then inspect its clauses.

### The `user:clause/2` Pattern

UnifyWeaver uses `user:clause/2` to access predicates from any module, including test predicates:

```prolog
find_predicate_calls(Pred/Arity, CalledPred/CalledArity) :-
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    extract_calls_from_body(Body, CalledPred/CalledArity).
```

This cross-module visibility is essential for analyzing code that might be defined in different contexts.

## 2. Call Graph Construction

A **call graph** is a directed graph where:
- **Nodes** are predicates (represented as `Pred/Arity`)
- **Edges** represent calls: `A -> B` means "predicate A calls predicate B"

### Why Call Graphs Matter

Call graphs reveal:
- **Self-recursion**: A predicate that calls itself
- **Mutual recursion**: Cycles where predicates call each other
- **Dependencies**: What predicates must be compiled first
- **Reachability**: All predicates used by a given predicate

### Building a Call Graph

UnifyWeaver's `call_graph.pl` module constructs call graphs automatically:

**Implementation (from `src/unifyweaver/core/advanced/call_graph.pl`):**

```prolog
%% build_call_graph(+Predicates, -Graph)
%  Build a directed graph of predicate calls
%  Graph is a list of edges: [pred1/arity1 -> pred2/arity2, ...]
build_call_graph(Predicates, Graph) :-
    findall(Edge,
        (   member(Pred/Arity, Predicates),
            find_predicate_calls(Pred/Arity, CalledPred),
            Edge = (Pred/Arity -> CalledPred)
        ),
        Graph).
```

### Extracting Calls from Bodies

The key challenge is extracting all predicate calls from a clause body, which may contain:
- Conjunctions: `(A, B)`
- Disjunctions: `(A; B)`
- Conditionals: `(A -> B)`
- Negations: `\+ A`

**Implementation:**

```prolog
extract_calls_from_body(true, _) :- !, fail.
extract_calls_from_body((A, B), Pred/Arity) :-
    !,
    (   extract_calls_from_body(A, Pred/Arity)
    ;   extract_calls_from_body(B, Pred/Arity)
    ).
extract_calls_from_body((A; B), Pred/Arity) :-
    !,
    (   extract_calls_from_body(A, Pred/Arity)
    ;   extract_calls_from_body(B, Pred/Arity)
    ).
extract_calls_from_body((A -> B), Pred/Arity) :-
    !,
    (   extract_calls_from_body(A, Pred/Arity)
    ;   extract_calls_from_body(B, Pred/Arity)
    ).
extract_calls_from_body(\+ A, Pred/Arity) :-
    !,
    extract_calls_from_body(A, Pred/Arity).
extract_calls_from_body(Call, Pred/Arity) :-
    compound(Call),
    functor(Call, Pred, Arity),
    Pred \= ',',
    Pred \= ';',
    Pred \= '->'.
```

This recursively walks the clause body structure and extracts all predicate calls.

### Example: Analyzing a Predicate

```prolog
% Load UnifyWeaver's call graph module
?- use_module('src/unifyweaver/core/advanced/call_graph').

% Define test predicates
?- assertz(user:ancestor(X, Y) :- parent(X, Y)).
?- assertz(user:(ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z))).

% Find what ancestor/2 depends on
?- get_dependencies(ancestor/2, Deps).
Deps = [ancestor/2, parent/2].

% Check if it's self-recursive
?- is_self_recursive(ancestor/2).
true.
```

## 3. Strongly Connected Components (SCC)

A **Strongly Connected Component (SCC)** is a maximal set of nodes in a directed graph where every node is reachable from every other node in the set.

### Why SCCs Matter for Mutual Recursion

Consider `is_even` and `is_odd`:

```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```

The call graph is:
```
is_even/1 -> is_odd/1
is_odd/1  -> is_even/1
```

This forms a cycle! Both predicates are in the same SCC, indicating **mutual recursion**.

### Tarjan's Algorithm

UnifyWeaver implements **Tarjan's algorithm** for finding SCCs in linear time O(V + E).

**Key idea:**
- Perform a depth-first search (DFS)
- Track two numbers for each node:
  - `index`: The order in which nodes are visited
  - `lowlink`: The smallest index reachable from this node
- When a node's `index == lowlink`, it's a **root** of an SCC

**Algorithm structure (from `src/unifyweaver/core/advanced/scc_detection.pl`):**

```prolog
%% Tarjan state structure
%  tarjan_state(Index, Stack, IndexMap, LowLinkMap, SCCs, OnStack)

tarjan_strongconnect(Node, Graph, State0, StateFinal) :-
    State0 = tarjan_state(Index0, Stack0, IndexMap0, LowLinkMap0, SCCs0, OnStack0),

    % Set node index and lowlink
    Index1 is Index0 + 1,
    put_assoc(Node, IndexMap0, Index0, IndexMap1),
    put_assoc(Node, LowLinkMap0, Index0, LowLinkMap1),

    % Push node onto stack
    Stack1 = [Node|Stack0],
    OnStack1 = [Node|OnStack0],

    % Visit successors (nodes this node calls)
    findall(Succ, member(Node->Succ, Graph), Successors),
    tarjan_visit_successors(Successors, Node, Graph, State1, State2),

    % Check if node is a root node
    get_assoc(Node, IndexMap2, NodeIndex),
    get_assoc(Node, LowLinkMap2, NodeLowLink),

    (   NodeIndex =:= NodeLowLink ->
        % Node is root - pop SCC from stack
        pop_scc(Node, Stack2, Stack3, OnStack2, OnStack3, SCC),
        SCCs2 = [SCC|SCCs1]
    ;   StateFinal = State2
    ).
```

### Example: Detecting Mutual Recursion

```prolog
% Load SCC detection module
?- use_module('src/unifyweaver/core/advanced/scc_detection').

% Build call graph for even/odd
?- build_call_graph([is_even/1, is_odd/1], Graph).
Graph = [is_even/1 -> is_odd/1, is_odd/1 -> is_even/1].

% Find SCCs
?- find_sccs(Graph, SCCs).
SCCs = [[is_even/1, is_odd/1]].

% Check if SCC is non-trivial (more than one node)
?- is_trivial_scc([is_even/1, is_odd/1]).
false.

?- is_trivial_scc([ancestor/2]).
true.
```

A non-trivial SCC (more than one predicate) indicates mutual recursion!

## 4. Pattern Matching

Once we have the call graph and SCCs, UnifyWeaver uses **pattern matchers** to identify specific recursion patterns.

### Pattern 1: Tail Recursion with Accumulator

**Key characteristics:**
- Recursive call is the **last action** in the clause (tail position)
- Uses an **accumulator** to carry intermediate results
- Commonly has arity 3: `pred(Input, Accumulator, Result)`

**Detection (from `src/unifyweaver/core/advanced/pattern_matchers.pl`):**

```prolog
is_tail_recursive_accumulator(Pred/Arity, AccInfo) :-
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Partition into base and recursive clauses
    partition(is_recursive_for(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],

    % Check if recursive calls are in tail position
    forall(
        member(clause(_, RecBody), RecClauses),
        is_tail_call(RecBody, Pred)
    ),

    % Identify accumulator position
    identify_accumulator_position(Pred/Arity, BaseClauses, RecClauses, AccPos),

    AccInfo = acc_pattern(BaseClauses, RecClauses, AccPos).
```

**What is tail position?**

```prolog
% Tail position: recursive call is LAST
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N).  % ← TAIL position

% NOT tail position: work after recursive call
list_length([_|T], N) :-
    list_length(T, N1),       % ← NOT tail position
    N is N1 + 1.              % ← work happens after!
```

### Pattern 2: Linear Recursion

**Key characteristics:**
- **Exactly ONE** recursive call per clause
- Recursive call arguments are **pre-computed** (not from pattern matching)
- No structural decomposition (not tree patterns like `[V, L, R]`)
- Body is a conjunction (AND chain)

**Detection:**

```prolog
is_linear_recursive_streamable(Pred/Arity) :-
    % Check if forbidden first
    \+ is_forbidden_linear_recursion(Pred/Arity),

    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Must have at least one recursive clause
    member(clause(_, SomeBody), Clauses),
    contains_call_to(SomeBody, Pred),

    % For each recursive clause, verify exactly ONE recursive call
    forall(
        (   member(clause(_, RecBody), Clauses),
            contains_call_to(RecBody, Pred)
        ),
        (   count_recursive_calls(RecBody, Pred, Count),
            Count =:= 1,
            recursive_args_are_precomputed(RecBody, Pred)
        )
    ),

    Arity > 1.
```

**Example patterns that match:**

```prolog
% Factorial - linear recursion
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,           % Pre-computed
    factorial(N1, F1),      % One recursive call
    F is N * F1.

% List length - linear recursion
list_length([], 0).
list_length([_|T], N) :-
    list_length(T, N1),     % One recursive call
    N is N1 + 1.
```

### Pattern 3: Forbidding Linear Recursion

Sometimes you want to force a different compilation strategy, even if a predicate matches the linear pattern.

**Use case:** Predicates with ordering dependencies or tree structures

```prolog
% Fibonacci could be linear, but we want tree recursion
:- forbid_linear_recursion(fibonacci/2).

fibonacci(0, 0).
fibonacci(1, 1).
fibonacci(N, F) :-
    N > 1,
    N1 is N - 1, N2 is N - 2,
    fibonacci(N1, F1),
    fibonacci(N2, F2),      % Two calls - tree recursion
    F is F1 + F2.
```

**Implementation:**

```prolog
:- dynamic forbidden_linear_recursion/1.

forbid_linear_recursion(Pred/Arity) :-
    retractall(forbidden_linear_recursion(Pred/Arity)),
    assertz(forbidden_linear_recursion(Pred/Arity)).

is_forbidden_linear_recursion(Pred/Arity) :-
    forbidden_linear_recursion(Pred/Arity).
```

### Pattern Selection Priority

UnifyWeaver tries patterns in this order:

1. **Tail recursion** (most efficient when applicable)
2. **Linear recursion** (unless forbidden)
3. **Tree recursion** (structural decomposition)
4. **Mutual recursion** (detected via SCC)
5. **Basic recursion** (fallback with BFS)

## 5. Practical Examples

### Example 1: Analyzing Your Own Predicate

Let's analyze a custom predicate to see what pattern it matches:

```prolog
% Load UnifyWeaver modules
?- use_module(unifyweaver(core/advanced/call_graph)).
?- use_module(unifyweaver(core/advanced/scc_detection)).
?- use_module(unifyweaver(core/advanced/pattern_matchers)).

% Define a predicate
?- assertz(user:sum_list([], 0)).
?- assertz(user:(sum_list([H|T], Sum) :- sum_list(T, S1), Sum is S1 + H)).

% Check if it's linear recursion
?- is_linear_recursive_streamable(sum_list/2).
true.

% Get dependencies
?- get_dependencies(sum_list/2, Deps).
Deps = [sum_list/2].

% Check if self-recursive
?- is_self_recursive(sum_list/2).
true.

% Count recursive calls in the body
?- user:clause(sum_list([_|T], _), Body),
   count_recursive_calls(Body, sum_list, Count).
Body = (sum_list(T, S1), _ is S1+_),
Count = 1.
```

### Example 2: Detecting Mutual Recursion

```prolog
% Define mutually recursive predicates
?- assertz(user:is_even(0)).
?- assertz(user:(is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))).
?- assertz(user:is_odd(1)).
?- assertz(user:(is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))).

% Build call graph
?- build_call_graph([is_even/1, is_odd/1], Graph).
Graph = [is_even/1 -> is_odd/1, is_odd/1 -> is_even/1].

% Find SCCs
?- find_sccs(Graph, SCCs).
SCCs = [[is_odd/1, is_even/1]].  % One SCC with both predicates!

% Check if non-trivial (mutual recursion)
?- SCCs = [SCC|_], \+ is_trivial_scc(SCC).
true.
```

### Example 3: Extracting Accumulator Patterns

```prolog
% Define accumulator-based predicate
?- assertz(user:count([], Acc, Acc)).
?- assertz(user:(count([_|T], Acc, N) :- Acc1 is Acc + 1, count(T, Acc1, N))).

% Check if tail recursive with accumulator
?- is_tail_recursive_accumulator(count/3, AccInfo).
AccInfo = acc_pattern([...base clauses...], [...rec clauses...], 2).

% Extract detailed pattern
?- extract_accumulator_pattern(count/3, Pattern).
Pattern = pattern(Acc, arithmetic(_ is _+1), unify).
```

## 6. Exercises

### Exercise 1: Classify Your Predicates

For each of the following predicates, determine what pattern UnifyWeaver would detect:

```prolog
% A) Reverse with accumulator
reverse_acc([], Acc, Acc).
reverse_acc([H|T], Acc, R) :- reverse_acc(T, [H|Acc], R).

% B) Tree depth
tree_depth([], 0).
tree_depth([_V, L, R], D) :-
    tree_depth(L, DL),
    tree_depth(R, DR),
    D is max(DL, DR) + 1.

% C) Maximum in list
max_list([X], X).
max_list([H|T], Max) :-
    max_list(T, MaxT),
    Max is max(H, MaxT).
```

**Answers:**
- A) Tail recursion (tail position, accumulator pattern)
- B) Tree recursion (structural decomposition `[V,L,R]`, two recursive calls)
- C) Linear recursion (one recursive call, pre-computed arguments)

### Exercise 2: Build a Call Graph

Given these predicates, manually draw the call graph:

```prolog
path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).

reachable(X, Y) :- path(X, Y).
reachable(X, Y) :- path(Y, X).

edge(a, b).
edge(b, c).
```

**Answer:**
```
path/2 -> edge/2
path/2 -> path/2  (self-recursive)
reachable/2 -> path/2
```

### Exercise 3: Detect Mutual Recursion

Write Prolog code to determine if these predicates are mutually recursive:

```prolog
positive(X) :- X > 0.
positive(X) :- X < 0, negative(X).

negative(X) :- X < 0.
negative(X) :- X > 0, positive(X).
```

**Solution:**
```prolog
?- build_call_graph([positive/1, negative/1], Graph).
?- find_sccs(Graph, SCCs).
% Check if any SCC has more than one node
?- member(SCC, SCCs), \+ is_trivial_scc(SCC).
```

### Exercise 4: Write a Pattern Detector

Write a predicate `has_multiple_recursive_calls/1` that checks if a predicate makes more than one recursive call in any clause:

**Solution:**
```prolog
has_multiple_recursive_calls(Pred/Arity) :-
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    count_recursive_calls(Body, Pred, Count),
    Count > 1.
```

## 7. Integration with Advanced Recursion Compiler

Now that we understand pattern detection, here's how it integrates with the compilation pipeline:

```prolog
% From recursive_compiler.pl (simplified)

compile_recursive(Pred/Arity, Options, BashCode) :-
    % 1. Build call graph
    build_call_graph([Pred/Arity], Graph),

    % 2. Find SCCs (mutual recursion)
    find_sccs(Graph, SCCs),

    % 3. Try patterns in order
    (   is_tail_recursive_accumulator(Pred/Arity, AccInfo) ->
        compile_tail_recursion(Pred/Arity, AccInfo, Options, BashCode)
    ;   is_linear_recursive_streamable(Pred/Arity) ->
        compile_linear_recursion(Pred/Arity, Options, BashCode)
    ;   member([Pred/Arity|Others], SCCs), Others \= [] ->
        compile_mutual_recursion([Pred/Arity|Others], Options, BashCode)
    ;   % Fallback to basic recursion
        compile_basic_recursion(Pred/Arity, Options, BashCode)
    ).
```

## Summary

UnifyWeaver's pattern detection system is built on three powerful Prolog introspection techniques:

1. **Call Graph Construction** using `clause/2` and `functor/3`
   - Extracts all predicate calls from clause bodies
   - Builds dependency graphs
   - Identifies self-recursion

2. **SCC Detection** using Tarjan's algorithm
   - Finds cycles in call graphs
   - Detects mutual recursion
   - Provides topological ordering

3. **Pattern Matching** using heuristics
   - Tail recursion: checks tail position and accumulator
   - Linear recursion: counts calls, checks pre-computation
   - Tree recursion: detects structural decomposition
   - Forbid system: manual override for special cases

These techniques enable UnifyWeaver to automatically select the optimal compilation strategy for your Prolog predicates, generating highly efficient Bash code without manual intervention.

### Key Takeaways

- **Prolog introspection** allows programs to analyze other programs
- **Call graphs** reveal predicate dependencies and recursion patterns
- **Tarjan's algorithm** efficiently finds mutual recursion cycles
- **Pattern matchers** use heuristics to classify recursion types
- **Pattern priority** ensures the most efficient compilation strategy is chosen

## Next Steps

- **Chapter 11:** Test Runner Inference - Automatic test generation and discovery
- **Appendix A:** Detailed recursion pattern theory and classification algorithms

## References

- `src/unifyweaver/core/advanced/call_graph.pl` - Call graph construction
- `src/unifyweaver/core/advanced/scc_detection.pl` - Tarjan's algorithm implementation
- `src/unifyweaver/core/advanced/pattern_matchers.pl` - Pattern detection heuristics
