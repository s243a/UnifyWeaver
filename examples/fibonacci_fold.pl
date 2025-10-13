% fibonacci_fold.pl - Fibonacci using two-phase fold pattern
% Demonstrates tree recursion with explicit structure building + fold
%
% This pattern:
% 1. Builds a dependency graph/tree structure
% 2. Folds over the structure to compute the final value
%
% Benefits:
% - Structure can be visualized/exported
% - Intermediate results can be cached
% - Makes parallelization explicit
% - Separates structure from computation

:- use_module(library(lists)).

% Load pattern matchers to use forbid_linear_recursion
:- catch(use_module('../src/unifyweaver/core/advanced/pattern_matchers'), _, true).

% Mark fibonacci as NOT linear recursive to force tree recursion pattern
% (Only works if pattern_matchers is loaded)
:- catch(forbid_linear_recursion(fib/2), _, true).

% Traditional fibonacci (will use tree recursion due to forbid)
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% Alternative: Explicit graph-building approach
% (Not yet auto-compiled, but demonstrates the pattern)

% fib_graph/2 - Build the dependency graph structure
% Returns: node(Value, [LeftChild, RightChild])
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib_graph(N1, L),
    fib_graph(N2, R).

% fold_fib/2 - Fold over the graph to compute value
fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :-
    fold_fib(L, VL),
    fold_fib(R, VR),
    V is VL + VR.

% fib_fold/2 - Two-phase fibonacci: build then fold
fib_fold(N, F) :-
    fib_graph(N, Graph),
    fold_fib(Graph, F).

% Example usage:
% ?- fib(5, F).
% F = 5.
%
% ?- fib_fold(5, F).
% F = 5.
%
% ?- fib_graph(3, G).
% G = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]).
