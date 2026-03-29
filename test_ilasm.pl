:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

:- dynamic classify_sign/2, safe_double/2, label/2.
classify_sign(0, zero).
classify_sign(X, positive) :- X > 0.
classify_sign(X, negative) :- X < 0.

safe_double(X, Y) :- X > 0, Y is X * 2.
safe_double(X, 0) :- X =< 0.

label(X, L) :- (X > 0 -> L = positive ; L = negative).

test_hook :-
    writeln('=== ILAsm hook test ==='),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(ilasm, Goal, VarMap, Code, _, _), Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

test_compile(Pred/Arity) :-
    format('ilasm ~w/~w: ', [Pred, Arity]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(ilasm)], Code), _, fail),
        Code \= ""
    ->  writeln(ok)
    ;   writeln('FAIL')
    ).

show(Pred/Arity) :-
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(ilasm)], Code), _, fail)
    ->  writeln(Code)
    ;   writeln('(compile failed)')
    ).

run :-
    test_hook,
    test_compile(classify_sign/2),
    test_compile(safe_double/2),
    test_compile(label/2),
    nl,
    writeln('=== classify_sign output ==='),
    show(classify_sign/2),
    nl, writeln('=== ILASM TESTS DONE ===').
