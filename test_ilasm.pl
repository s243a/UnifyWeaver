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

:- dynamic factorial/2, fibonacci/2, is_even/1, is_odd/1.
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

fibonacci(0, 0).
fibonacci(1, 1).
fibonacci(N, F) :- N > 1, N1 is N-1, N2 is N-2, fibonacci(N1, F1), fibonacci(N2, F2), F is F1 + F2.

is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).
is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).

:- dynamic typed_add/3.
:- type_declarations:assert(uw_type(typed_add/3, 1, integer)).
:- type_declarations:assert(uw_type(typed_add/3, 2, integer)).
:- type_declarations:assert(uw_return_type(typed_add/3, integer)).
typed_add(X, Y, Z) :- Z is X + Y.

:- dynamic parent/2, ancestor/2.
parent(alice, bob). parent(bob, charlie). parent(bob, diana).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

run :-
    test_hook,
    test_compile(classify_sign/2),
    test_compile(safe_double/2),
    test_compile(label/2),
    test_compile(factorial/2),
    test_compile(fibonacci/2),
    test_compile(is_even/1),
    test_compile(ancestor/2),
    %% Type test
    format('ilasm typed_add/3: ', []),
    (   catch(recursive_compiler:compile_recursive(typed_add/3, [target(ilasm)], Code), _, fail),
        Code \= "",
        sub_string(Code, _, _, _, "int64 arg1, int64 arg2")
    ->  writeln(ok)
    ;   writeln('FAIL')
    ),
    nl, writeln('=== ILASM TESTS DONE ===').
