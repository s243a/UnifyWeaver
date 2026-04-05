:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/core/clause_body_analysis').
:- use_module('../src/unifyweaver/targets/go_target').

:- begin_tests(go_goal_parallel).

test(classify_parallel_simple) :-
    % p(X, Y) :- g1(X, Y1), g2(X, Y2), Y is Y1 + Y2.
    Clause = p(X, Y) - (g1(X, Y1), g2(X, Y2), Y is Y1 + Y2),
    classify_parallelism(p/2, [Clause], Strategy),
    assertion(Strategy = goal_parallel(Head, Pars, Results)),
    assertion(functor(Head, p, 2)),
    assertion(length(Pars, 2)),
    assertion(Results = [_ is _ + _]).

test(classify_parallel_impure) :-
    % p(X) :- write(X), g1(X).
    Clause = p(X) - (write(X), g1(X)),
    classify_parallelism(p/1, [Clause], Strategy),
    assertion(Strategy = sequential).

test(classify_parallel_dependent) :-
    % p(X, Y) :- g1(X, Z), g2(Z, Y).
    Clause = p(X, Y) - (g1(X, Z), g2(Z, Y)),
    classify_parallelism(p/2, [Clause], Strategy),
    assertion(Strategy = sequential).

test(classify_declared_independent) :-
    assertz(clause_body_analysis:order_independent(my_pred/2)),
    classify_parallelism(my_pred/2, [h1-b1, h2-b2], Strategy),
    assertion(Strategy = clause_parallel),
    retract(clause_body_analysis:order_independent(my_pred/2)).

test(parallel_safe_hook) :-
    assertz(clause_body_analysis:parallel_safe(my_impure_looking_pred/2)),
    Clause = p(X, Y) - (my_impure_looking_pred(X, Y1), g2(X, Y2), Y is Y1 + Y2),
    classify_parallelism(p/2, [Clause], Strategy),
    assertion(Strategy = goal_parallel(_, [_,_], _)),
    retract(clause_body_analysis:parallel_safe(my_impure_looking_pred/2)).

test(compile_parallel_code) :-
    Clause = node_score(X, Y) - (feature_a(X, Y1), feature_b(X, Y2), Y is Y1 + Y2),
    classify_parallelism(node_score/2, [Clause], goal_parallel(Head, ParallelGoals, ResultGoals)),
    compile_goal_parallel_to_go(node_score/2, Head, ParallelGoals, ResultGoals, Code),
    assertion(sub_string(Code, _, _, _, 'sync.WaitGroup')),
    assertion(sub_string(Code, _, _, _, 'go func()')),
    assertion(sub_string(Code, _, _, _, 'wg.Wait()')).

% ============================================================================
% Phase 5b: Clause-level parallelism
% ============================================================================

test(clause_parallel_codegen) :-
    % Simulate an order-independent predicate with 2 clauses
    assertz(clause_body_analysis:order_independent(node_color/2)),
    Clauses = [
        node_color(X, Y) - (X = red, Y = 1),
        node_color(X, Y) - (X = blue, Y = 2)
    ],
    classify_parallelism(node_color/2, Clauses, Strategy),
    assertion(Strategy == clause_parallel),
    compile_clause_parallel_to_go(node_color/2, Clauses, Code),
    assertion(sub_string(Code, _, _, _, 'sync.WaitGroup')),
    assertion(sub_string(Code, _, _, _, 'go func()')),
    assertion(sub_string(Code, _, _, _, 'wg.Wait()')),
    assertion(sub_string(Code, _, _, _, 'results')),
    assertion(sub_string(Code, _, _, _, 'chan interface{}')),
    assertion(sub_string(Code, _, _, _, 'clause 1')),
    assertion(sub_string(Code, _, _, _, 'clause 2')),
    retract(clause_body_analysis:order_independent(node_color/2)).

test(clause_parallel_has_close_channel) :-
    assertz(clause_body_analysis:order_independent(test_cp/1)),
    Clauses = [test_cp(X) - (X = a), test_cp(X) - (X = b)],
    compile_clause_parallel_to_go(test_cp/1, Clauses, Code),
    assertion(sub_string(Code, _, _, _, 'close(results)')),
    retract(clause_body_analysis:order_independent(test_cp/1)).

test(clause_parallel_dispatched_from_native_body) :-
    assertz(clause_body_analysis:order_independent(dispatched/1)),
    Clauses = [dispatched(X) - (X = 1), dispatched(X) - (X = 2)],
    % This should route through the clause_parallel path in native_go_clause_body
    go_target:native_go_clause_body(dispatched/1, Clauses, Code),
    assertion(sub_string(Code, _, _, _, 'Clause-parallel')),
    retract(clause_body_analysis:order_independent(dispatched/1)).

:- end_tests(go_goal_parallel).
