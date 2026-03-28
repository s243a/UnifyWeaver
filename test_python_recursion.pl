:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

:- dynamic factorial/2, sum_list/2, fibonacci/2, list_length/2.

factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

sum_list([], 0).
sum_list([H|T], S) :- sum_list(T, S1), S is S1 + H.

fibonacci(0, 0).
fibonacci(1, 1).
fibonacci(N, F) :- N > 1, N1 is N-1, N2 is N-2, fibonacci(N1, F1), fibonacci(N2, F2), F is F1 + F2.

list_length([], 0).
list_length([_|T], N) :- list_length(T, N1), N is N1 + 1.

try(Label, Pred/Arity) :-
    format('~w: ', [Label]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail)
    ->  (sub_string(Code, _, _, _, "def ") -> writeln(ok) ; writeln('compiled but no def'))
    ;   writeln('FAIL')
    ).

show(Pred/Arity) :-
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail)
    ->  atom_string(Pred, PS),
        split_string(Code, "\n", "", Lines),
        format(atom(Prefix), "def ~w", [PS]),
        (   nth1(I, Lines, L), sub_string(L, _, _, _, Prefix)
        ->  End is min(I + 15, 99999),
            forall((between(I, End, J), nth1(J, Lines, LJ),
                    (LJ \= "" ; J =:= I)), writeln(LJ))
        ;   writeln('  (function not found)')
        )
    ;   writeln('  (compile failed)')
    ).

run :-
    try('factorial/2 (linear)', factorial/2),
    try('sum_list/2 (linear list)', sum_list/2),
    try('fibonacci/2 (tree)', fibonacci/2),
    try('list_length/2 (linear list)', list_length/2),
    nl,
    writeln('=== factorial ==='), show(factorial/2), nl,
    writeln('=== sum_list ==='), show(sum_list/2), nl,
    writeln('=== fibonacci ==='), show(fibonacci/2), nl,
    writeln('=== list_length ==='), show(list_length/2), nl,
    writeln('=== RECURSION TESTS DONE ===').
