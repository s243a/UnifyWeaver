:- encoding(utf8).
% Test AWK target with tail-recursive patterns

:- use_module('src/unifyweaver/targets/awk_target').

% Tail-recursive factorial
% Base case: factorial(0, Acc, Acc)
% Recursive: factorial(N, Acc, F) :- N > 0, N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, F)
factorial(0, Acc, Acc).
factorial(N, Acc, F) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, F).

% Tail-recursive sum from N to 0
sum_to_zero(0, Acc, Acc).
sum_to_zero(N, Acc, Sum) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum_to_zero(N1, Acc1, Sum).

% Tail-recursive countdown
countdown(0, Acc, Acc).
countdown(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    countdown(N1, N1, Result).

test_factorial :-
    write('=== Test: Tail-Recursive Factorial ==='), nl, nl,
    awk_target:compile_predicate_to_awk(factorial/3, [], AwkCode),
    write(AwkCode), nl, nl.

test_sum :-
    write('=== Test: Tail-Recursive Sum ==='), nl, nl,
    awk_target:compile_predicate_to_awk(sum_to_zero/3, [], AwkCode),
    write(AwkCode), nl, nl.

test_countdown :-
    write('=== Test: Tail-Recursive Countdown ==='), nl, nl,
    awk_target:compile_predicate_to_awk(countdown/3, [], AwkCode),
    write(AwkCode), nl, nl.

run_all :-
    test_factorial,
    test_sum,
    test_countdown,
    write('All tail recursion tests completed!'), nl.

% Usage:
% ?- consult('test_awk_tail_recursion.pl').
% ?- run_all.
