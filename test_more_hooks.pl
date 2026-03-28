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
    ;   writeln('FAIL (compile_expression)')
    ).

test_compile(Target) :-
    format('~w compile: ', [Target]),
    (   catch(recursive_compiler:compile_recursive(classify_sign/2, [target(Target)], Code), _, fail),
        Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

run :-
    %% Test hooks
    test_hook(rust),
    test_hook(c),
    test_hook(cpp),
    test_hook(perl),
    test_hook(ruby),
    test_hook(typescript),
    nl,
    %% Test full compilation
    test_compile(rust),
    test_compile(c),
    test_compile(cpp),
    test_compile(perl),
    test_compile(ruby),
    test_compile(typescript),
    nl,
    writeln('=== ALL 6 TARGET HOOK TESTS DONE ===').
