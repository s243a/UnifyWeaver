:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

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
    test_hook(wat),
    test_hook(jamaica),
    test_hook(krakatau),
    test_hook(llvm),
    nl, writeln('=== ALL 4 ASSEMBLY TARGET HOOK TESTS DONE ===').
