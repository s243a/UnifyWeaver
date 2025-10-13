% binomial_fold.pl - Binomial coefficients using Pascal's triangle fold pattern
% Demonstrates tree recursion with structure building for Pascal's triangle
%
% Binomial coefficient C(n, k) = "n choose k"
% Recursive formula: C(n, k) = C(n-1, k-1) + C(n-1, k)
% Base cases: C(n, 0) = 1, C(n, n) = 1

:- use_module(library(lists)).

% Traditional binomial coefficient
binom(_, 0, 1).
binom(N, N, 1).
binom(N, K, C) :-
    N > 0, K > 0, K < N,
    N1 is N - 1,
    K1 is K - 1,
    binom(N1, K1, C1),
    binom(N1, K, C2),
    C is C1 + C2.

% Graph-building approach for Pascal's triangle

% binom_graph/3 - Build the dependency graph
binom_graph(_, 0, leaf(1)).
binom_graph(N, N, leaf(1)).
binom_graph(N, K, node(N, K, [L, R])) :-
    N > 0, K > 0, K < N,
    N1 is N - 1,
    K1 is K - 1,
    binom_graph(N1, K1, L),
    binom_graph(N1, K, R).

% fold_binom/2 - Fold over the graph to compute value
fold_binom(leaf(V), V).
fold_binom(node(_, _, [L, R]), V) :-
    fold_binom(L, VL),
    fold_binom(R, VR),
    V is VL + VR.

% binom_fold/3 - Two-phase binomial: build then fold
binom_fold(N, K, C) :-
    binom_graph(N, K, Graph),
    fold_binom(Graph, C).

% Compute an entire row of Pascal's triangle
pascal_row(N, Row) :-
    findall(C, (between(0, N, K), binom(N, K, C)), Row).

% Visualize Pascal's triangle (first N rows)
pascal_triangle(N) :-
    between(0, N, Row),
    pascal_row(Row, Values),
    format('Row ~w: ~w~n', [Row, Values]),
    fail.
pascal_triangle(_).

% Example usage:
% ?- binom(5, 2, C).
% C = 10.
%
% ?- binom_fold(5, 2, C).
% C = 10.
%
% ?- pascal_row(4, Row).
% Row = [1, 4, 6, 4, 1].
%
% ?- pascal_triangle(5).
% Row 0: [1]
% Row 1: [1, 1]
% Row 2: [1, 2, 1]
% Row 3: [1, 3, 3, 1]
% Row 4: [1, 4, 6, 4, 1]
% Row 5: [1, 5, 10, 10, 5, 1]
