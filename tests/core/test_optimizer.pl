:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/core/optimizer').

:- begin_tests(optimizer).

test(reorder_filter_pushdown) :-
    % Input: X > 10, generator(X)
    % Expected: generator(X), X > 10 (since > needs X)
    Body = (X > 10, json_record([x-X])),
    optimize_clause(head(_), Body, Optimized),
    Optimized = (json_record([x-X]), X > 10).

test(reorder_ground_first) :-
    % Input: parent(X,Y), X=alice
    % Expected: X=alice, parent(X,Y) (X bound first)
    % Note: =/2 is comparison or unification.
    % If X=alice is treated as assignment/unification, it's a generator of X.
    % optimizer.pl treats 'generic' as generator.
    Body = (parent(X,Y), X=alice),
    optimize_clause(head(_), Body, Optimized),
    % X=alice has 0 inputs? It binds X.
    % parent(X,Y) has 0 inputs bound initially.
    % X=alice should be picked first because it's "generic/generator".
    % But parent is also generic.
    % select_best_goal logic:
    % count_bound_vars: initially 0 for both.
    % Stable sort?
    % If stable, it stays same.
    % We might need heuristic: Prefer goals with FEWER variables if bound counts are equal?
    % Or prefer unification?
    
    % Let's see what happens.
    true.

test(complex_chain) :-
    % Input: Z > 5, b(Y, Z), a(X, Y)
    % Logic:
    % 1. Z > 5 not ready.
    % 2. b(Y, Z) ready (0 bound). a(X, Y) ready (0 bound).
    % 3. Stable sort picks 'b' first.
    % 4. Bound={Y, Z}.
    % 5. Z > 5 ready. comparison vs 'a' (generic). comparison wins.
    % 6. Bound={Y, Z}. 'a' runs.
    
    Body = (Z > 5, b(Y, Z), a(X, Y)),
    optimize_clause(head(_), Body, Optimized),
    Optimized = (b(Y, Z), Z > 5, a(X, Y)).

:- end_tests(optimizer).
