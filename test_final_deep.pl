:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

test_hook(Target) :-
    format('~w: ', [Target]),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(Target, Goal, VarMap, Code, _, _), Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

run :-
    test_hook(scala), test_hook(kotlin), test_hook(jython),
    test_hook(awk), test_hook(vbnet),
    nl, writeln('=== ALL 5 FINAL DEEPENED TARGETS DONE ===').
