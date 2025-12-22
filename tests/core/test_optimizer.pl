:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/core/optimizer').

:- begin_tests(optimizer).

test(reorder_filter_pushdown) :-
    % Input: X > 10, generator(X)
    % Expected: generator(X), X > 10 (since > needs X)
    Body = (X > 10, json_record([x-X])),
    optimize_clause(head(_), Body, [unordered(true)], Optimized),
    Optimized = (json_record([x-X]), X > 10).

test(reorder_ground_first) :-
    % Input: parent(X,Y), X=alice
    % Expected: X=alice, parent(X,Y) (X bound first)
    Body = (parent(X,Y), X=alice),
    optimize_clause(head(_), Body, [unordered(true)], Optimized),
    Optimized = (X=alice, parent(X,Y)).

test(complex_chain) :-
    % Input: Z > 5, b(Y, Z), a(X, Y)
    Body = (Z > 5, b(Y, Z), a(X, Y)),
    optimize_clause(head(_), Body, [unordered(true)], Optimized),
    Optimized = (b(Y, Z), Z > 5, a(X, Y)).

test(ordered_no_optimization) :-
    % Input: X > 10, generator(X)
    % Expected: X > 10, generator(X) (unchanged if unordered=false)
    Body = (X > 10, json_record([x-X])),
    optimize_clause(head(_), Body, [unordered(false)], Optimized),
    Optimized = (X > 10, json_record([x-X])).

:- end_tests(optimizer).
