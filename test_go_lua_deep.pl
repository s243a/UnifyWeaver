:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

:- dynamic label/2, classify_temp/2, safe_double/2, bounded_sq/2.

label(X, L) :- (X > 0 -> L = positive ; L = negative).

classify_temp(T, freezing) :- T =< 0.
classify_temp(T, cold) :- T > 0, T < 15.
classify_temp(T, warm) :- T >= 15, T < 30.
classify_temp(T, hot) :- T >= 30.

safe_double(X, Y) :- X > 0, Y is X * 2.
safe_double(X, 0) :- X =< 0.

bounded_sq(X, Y) :- Y is X * X, Y < 1000.

try(Label, Target, Pred/Arity, Checks) :-
    format('~w (~w): ', [Label, Target]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(Target)], Code), _, fail)
    ->  run_checks(Code, Checks, AllOk),
        (AllOk == true -> writeln(ok) ; true)
    ;   writeln('COMPILE FAIL')
    ).

run_checks(_, [], true).
run_checks(Code, [check(Label, Substr)|Rest], AllOk) :-
    (   sub_string(Code, _, _, _, Substr)
    ->  run_checks(Code, Rest, AllOk)
    ;   format('MISS(~w) ', [Label]),
        AllOk = false,
        run_checks(Code, Rest, _)
    ).

run :-
    %% Go tests
    try('ite output', go, label/2, [check('if', "if ")]),
    try('multi-clause', go, classify_temp/2, [check('freezing', "freezing")]),
    try('guard+output', go, safe_double/2, [check('* 2', "* 2")]),
    try('guarded tail', go, bounded_sq/2, [check('< 1000', "< 1000")]),
    nl,
    %% Lua tests
    try('ite output', lua, label/2, [check('if then', "if ")]),
    try('multi-clause', lua, classify_temp/2, [check('freezing', "freezing")]),
    try('guard+output', lua, safe_double/2, [check('* 2', "* 2")]),
    try('guarded tail', lua, bounded_sq/2, [check('< 1000', "< 1000")]),
    nl,
    writeln('=== GO + LUA DEEP TESTS DONE ===').
