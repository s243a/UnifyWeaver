:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

%% Arity-1 mutual (already works)
:- dynamic is_even/1, is_odd/1.
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).
is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).

%% Arity-2 mutual: value-returning
:- dynamic count_down_even/2, count_down_odd/2.
count_down_even(0, 0).
count_down_even(N, R) :- N > 0, N1 is N - 1, count_down_odd(N1, R1), R is R1 + 1.
count_down_odd(0, 0).
count_down_odd(N, R) :- N > 0, N1 is N - 1, count_down_even(N1, R1), R is R1 + 1.

try(Label, Pred/Arity) :-
    format('~w: ', [Label]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail)
    ->  atom_string(Pred, PS),
        split_string(Code, "\n", "", Lines),
        (   nth1(I, Lines, L), (sub_string(L, _, _, _, PS) ; sub_string(L, _, _, _, "def ")),
            sub_string(L, _, _, _, "def ")
        ->  End is min(I + 10, 99999),
            forall((between(I, End, J), nth1(J, Lines, LJ), (LJ \= "" ; J =:= I)), writeln(LJ))
        ;   true
        ),
        writeln(ok)
    ;   writeln('FAIL')
    ).

run :-
    try('is_even/1 (arity-1 mutual)', is_even/1),
    nl,
    try('count_down_even/2 (arity-2 mutual)', count_down_even/2),
    nl,
    writeln('=== MUTUAL RECURSION TESTS DONE ===').
