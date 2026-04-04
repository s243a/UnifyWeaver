:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/core/clause_body_analysis').
:- use_module('../src/unifyweaver/targets/go_target').

:- begin_tests(go_goal_parallel).

test(classify_parallel_simple) :-
    % p(X, Y) :- g1(X, Y1), g2(X, Y2), Y is Y1 + Y2.
    Clause = p(X, Y) - (g1(X, Y1), g2(X, Y2), Y is Y1 + Y2),
    classify_parallelism(p/2, [Clause], Strategy),
    assertion(Strategy = goal_parallel(p(X, Y), [g1(X, Y1), g2(X, Y2)], Y is Y1 + Y2)).

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

test(compile_parallel_code) :-
    Clause = node_score(X, Y) - (feature_a(X, Y1), feature_b(X, Y2), Y is Y1 + Y2),
    classify_parallelism(node_score/2, [Clause], goal_parallel(Head, ParallelGoals, ResultGoal)),
    compile_goal_parallel_to_go(node_score/2, Head, ParallelGoals, ResultGoal, Code),
    format('Generated code:~n~s~n', [Code]),
    assertion(sub_string(Code, _, _, _, 'sync.WaitGroup')),
    assertion(sub_string(Code, _, _, _, 'go func()')).

:- end_tests(go_goal_parallel).
