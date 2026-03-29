:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

:- dynamic classify_sign/2, safe_double/2.

classify_sign(0, zero).
classify_sign(X, positive) :- X > 0.
classify_sign(X, negative) :- X < 0.

safe_double(X, Y) :- X > 0, Y is X * 2.
safe_double(X, 0) :- X =< 0.

test_hook(Target) :-
    format('~w hook: ', [Target]),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(Target, Goal, VarMap, Code, _, _), Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

test_compile(Target, Pred/Arity) :-
    format('~w ~w/~w: ', [Target, Pred, Arity]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(Target)], Code), _, fail),
        Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

run :-
    test_hook(wat), test_hook(jamaica), test_hook(krakatau),
    nl,
    test_compile(wat, classify_sign/2),
    test_compile(wat, safe_double/2),
    test_compile(jamaica, classify_sign/2),
    test_compile(jamaica, safe_double/2),
    test_compile(krakatau, classify_sign/2),
    test_compile(krakatau, safe_double/2),
    nl,
    writeln('=== ASSEMBLY DEEP TESTS DONE ===').
