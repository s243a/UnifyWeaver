:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

:- dynamic label/2.
label(X, L) :- (X > 0 -> L = positive ; L = negative).

test_hook(Target) :-
    format('~w: ', [Target]),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(Target, Goal, VarMap, Code, _, _), Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

run :-
    test_hook(perl), test_hook(ruby), test_hook(typescript),
    test_hook(haskell), test_hook(fsharp), test_hook(elixir), test_hook(clojure),
    nl, writeln('=== ALL 7 ADDITIONAL DEEPENED TARGETS DONE ===').
