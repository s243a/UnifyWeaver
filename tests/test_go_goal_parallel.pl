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

:- end_tests(go_goal_parallel).
