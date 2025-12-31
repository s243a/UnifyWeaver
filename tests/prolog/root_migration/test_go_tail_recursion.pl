:- encoding(utf8).
% Test Go target with tail-recursive patterns

:- use_module('src/unifyweaver/targets/go_target').

% Tail-recursive factorial
% Base case: factorial(0, Acc, Acc)
% Recursive: factorial(N, Acc, F) :- N > 0, N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, F)
factorial(0, Acc, Acc).
factorial(N, Acc, F) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, F).

% Tail-recursive sum
sum_to_zero(0, Acc, Acc).
sum_to_zero(N, Acc, Sum) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum_to_zero(N1, Acc1, Sum).

test_factorial :-
    write('=== Test: Tail-Recursive Factorial ==='), nl,
    go_target:compile_predicate_to_go(factorial/3, [], Code),
    write(Code), nl, nl.

test_sum :-
    write('=== Test: Tail-Recursive Sum ==='), nl,
    go_target:compile_predicate_to_go(sum_to_zero/3, [], Code),
    write(Code), nl, nl.

run_all :-
    test_factorial,
    test_sum,
    write('All tail recursion tests completed!'), nl.

% Usage:
% ?- consult('test_go_tail_recursion.pl').
% ?- run_all.
