:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

:- dynamic classify_sign/2.
classify_sign(0, zero).
classify_sign(X, positive) :- X > 0.
classify_sign(X, negative) :- X < 0.

test_hook(Target) :-
    format('~w hooks: ', [Target]),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(Target, Goal, VarMap, Code, _, _),
        Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

run :-
    test_hook(haskell),
    test_hook(fsharp),
    test_hook(clojure),
    test_hook(elixir),
    test_hook(scala),
    test_hook(kotlin),
    test_hook(jython),
    nl, writeln('=== ALL 7 FUNCTIONAL TARGET HOOK TESTS DONE ===').
