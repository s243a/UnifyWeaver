:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

try_compile(Pred/Arity, Code) :-
    catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail).

show(Code, FuncName) :-
    split_string(Code, "\n", "", Lines),
    format(atom(Prefix), "def ~w", [FuncName]),
    (   nth1(I, Lines, L), sub_string(L, _, _, _, Prefix)
    ->  Start is I, End is min(I + 12, 99999),
        forall((between(Start, End, J), nth1(J, Lines, LJ),
                (LJ \= "" ; J =:= Start)), writeln(LJ))
    ;   writeln('  (not found)')
    ).

%% --- Predicates to quality-check ---

:- dynamic checked_double/2, abs_custom/2, classify_temp/2, direction/2, between_check/4.

checked_double(X, Y) :- X > 0, Y is X * 2.
checked_double(X, 0) :- X =< 0.

abs_custom(X, X) :- X >= 0.
abs_custom(X, Y) :- X < 0, Y is -X.

classify_temp(T, freezing) :- T =< 0.
classify_temp(T, cold) :- T > 0, T < 15.
classify_temp(T, warm) :- T >= 15, T < 30.
classify_temp(T, hot) :- T >= 30.

direction(0, zero).
direction(X, positive) :- X > 0.
direction(X, negative) :- X < 0.

between_check(X, Lo, Hi, in_range) :- X >= Lo, X =< Hi.
between_check(_, _, _, out_of_range).

run :-
    writeln('--- checked_double ---'),
    (try_compile(checked_double/2, C1) -> show(C1, checked_double) ; writeln('  FAIL')), nl,

    writeln('--- abs_custom ---'),
    (try_compile(abs_custom/2, C2) -> show(C2, abs_custom) ; writeln('  FAIL')), nl,

    writeln('--- classify_temp ---'),
    (try_compile(classify_temp/2, C3) -> show(C3, classify_temp) ; writeln('  FAIL')), nl,

    writeln('--- direction ---'),
    (try_compile(direction/2, C4) -> show(C4, direction) ; writeln('  FAIL')), nl,

    writeln('--- between_check ---'),
    (try_compile(between_check/4, C5) -> show(C5, between_check) ; writeln('  FAIL')), nl,

    writeln('=== QUALITY CHECK DONE ===').
