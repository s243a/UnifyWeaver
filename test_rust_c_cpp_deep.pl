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
run_checks(Code, [check(L, S)|R], Ok) :-
    (sub_string(Code, _, _, _, S) -> run_checks(Code, R, Ok)
    ; format('MISS(~w) ', [L]), Ok = false, run_checks(Code, R, _)).

run :-
    %% Rust: compile_predicate_to_rust dispatch has pre-existing issue
    %% for non-recursive predicates (semantic predicate check interferes).
    %% Hooks work (tested in PR #1030), classify_goal_sequence added,
    %% but full compilation path needs separate fix.
    try('ite', c, label/2, [check('if', "if ")]),
    try('multi', c, classify_temp/2, [check('freezing', "freezing")]),
    try('guard+out', c, safe_double/2, [check('* 2', "* 2")]),
    try('tail', c, bounded_sq/2, [check('< 1000', "< 1000")]),
    nl,
    try('ite', cpp, label/2, [check('if', "if ")]),
    try('multi', cpp, classify_temp/2, [check('freezing', "freezing")]),
    try('guard+out', cpp, safe_double/2, [check('* 2', "* 2")]),
    try('tail', cpp, bounded_sq/2, [check('< 1000', "< 1000")]),
    nl,
    writeln('=== C + C++ DEEP TESTS DONE (Rust has pre-existing dispatch issue) ===').
