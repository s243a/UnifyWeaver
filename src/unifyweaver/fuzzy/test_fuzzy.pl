/**
 * Test suite for Fuzzy Logic DSL
 */

:- use_module(core).
:- use_module(operators).
:- use_module(eval).
:- use_module(boolean).

run_tests :-
    format('~n=== Fuzzy Logic DSL Tests ===~n~n'),

    % Set up term scores
    assertz(fuzzy_core:term_score(bash, 0.8)),
    assertz(fuzzy_core:term_score(shell, 0.6)),

    % Test core operations
    format('--- Core Operations ---~n'),
    test_f_and,
    test_f_or,
    test_f_dist_or,
    test_f_union,
    test_f_not,

    % Test eval module
    format('~n--- Eval Module ---~n'),
    test_eval_fuzzy_expr,
    test_fallback,

    % Test operators
    format('~n--- Operators Module ---~n'),
    test_expand_fuzzy,
    test_fuzzy_and,
    test_fuzzy_or,

    % Test boolean operations
    format('~n--- Boolean Operations ---~n'),
    test_b_and,
    test_b_or,

    % Clean up
    retractall(fuzzy_core:term_score(_,_)),

    format('~n=== All Tests Complete ===~n').

% Core tests
test_f_and :-
    f_and([w(bash, 0.9), w(shell, 0.5)], Result),
    Expected is 0.9 * 0.8 * 0.5 * 0.6,
    (abs(Result - Expected) < 0.0001
    ->  format('f_and: PASS (~w)~n', [Result])
    ;   format('f_and: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

test_f_or :-
    f_or([w(bash, 0.9), w(shell, 0.5)], Result),
    Expected is 1 - (1 - 0.9*0.8) * (1 - 0.5*0.6),
    (abs(Result - Expected) < 0.0001
    ->  format('f_or: PASS (~w)~n', [Result])
    ;   format('f_or: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

test_f_dist_or :-
    f_dist_or(0.7, [w(bash, 0.9), w(shell, 0.5)], Result),
    Expected is 1 - (1 - 0.7*0.9*0.8) * (1 - 0.7*0.5*0.6),
    (abs(Result - Expected) < 0.0001
    ->  format('f_dist_or(0.7): PASS (~w)~n', [Result])
    ;   format('f_dist_or(0.7): FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

test_f_union :-
    f_union(0.7, [w(bash, 0.9), w(shell, 0.5)], Result),
    OrResult is 1 - (1 - 0.9*0.8) * (1 - 0.5*0.6),
    Expected is 0.7 * OrResult,
    (abs(Result - Expected) < 0.0001
    ->  format('f_union(0.7): PASS (~w)~n', [Result])
    ;   format('f_union(0.7): FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

test_f_not :-
    f_not(0.3, Result),
    Expected is 0.7,
    (abs(Result - Expected) < 0.0001
    ->  format('f_not: PASS (~w)~n', [Result])
    ;   format('f_not: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

% Eval tests
test_eval_fuzzy_expr :-
    eval_fuzzy_expr(
        f_and([w(bash, 0.9), w(shell, 0.5)]),
        [bash-0.8, shell-0.6],
        Result
    ),
    Expected is 0.9 * 0.8 * 0.5 * 0.6,
    (abs(Result - Expected) < 0.0001
    ->  format('eval_fuzzy_expr: PASS (~w)~n', [Result])
    ;   format('eval_fuzzy_expr: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

test_fallback :-
    eval_fuzzy_expr(
        f_and([w(unknown_term, 1.0)]),
        [],
        Result
    ),
    Expected is 0.5,
    (abs(Result - Expected) < 0.0001
    ->  format('fallback score: PASS (~w)~n', [Result])
    ;   format('fallback score: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

% Operators tests - using functor construction to avoid parse-time issues
test_expand_fuzzy :-
    % Build bash:0.9 & shell:0.5 dynamically
    Expr = &(':'(bash, 0.9), ':'(shell, 0.5)),
    expand_fuzzy(Expr, Expanded),
    (Expanded = f_and([w(bash, 0.9), w(shell, 0.5)])
    ->  format('expand_fuzzy AND: PASS~n')
    ;   format('expand_fuzzy AND: FAIL (~w)~n', [Expanded])
    ).

test_fuzzy_and :-
    % Build bash:0.9 & shell:0.5 dynamically
    Expr = &(':'(bash, 0.9), ':'(shell, 0.5)),
    fuzzy_and(Expr, Result),
    Expected is 0.9 * 0.8 * 0.5 * 0.6,
    (abs(Result - Expected) < 0.0001
    ->  format('fuzzy_and: PASS (~w)~n', [Result])
    ;   format('fuzzy_and: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

test_fuzzy_or :-
    % Build bash:0.9 v shell:0.5 dynamically
    Expr = v(':'(bash, 0.9), ':'(shell, 0.5)),
    fuzzy_or(Expr, Result),
    Expected is 1 - (1 - 0.9*0.8) * (1 - 0.5*0.6),
    (abs(Result - Expected) < 0.0001
    ->  format('fuzzy_or: PASS (~w)~n', [Result])
    ;   format('fuzzy_or: FAIL (got ~w, expected ~w)~n', [Result, Expected])
    ).

% Boolean tests
test_b_and :-
    b_and([true, true], Result1),
    b_and([true, false], Result2),
    ((Result1 == 1.0, Result2 == 0.0)
    ->  format('b_and: PASS~n')
    ;   format('b_and: FAIL (got ~w, ~w)~n', [Result1, Result2])
    ).

test_b_or :-
    b_or([false, false], Result1),
    b_or([true, false], Result2),
    ((Result1 == 0.0, Result2 == 1.0)
    ->  format('b_or: PASS~n')
    ;   format('b_or: FAIL (got ~w, ~w)~n', [Result1, Result2])
    ).

:- initialization(run_tests, main).
