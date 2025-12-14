:- encoding(utf8).
% Test suite for tail and linear recursion in Go and Rust targets
% Usage: swipl -g run_tests -t halt tests/test_recursion_patterns.pl

:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').
:- use_module('../src/unifyweaver/targets/rust_target').

%% Test predicates - tail recursion with accumulator
test_sum([], Acc, Acc).
test_sum([H|T], Acc, S) :- Acc1 is Acc + H, test_sum(T, Acc1, S).

%% Test predicates - linear recursion
test_triangular(0, 0).
test_triangular(1, 1).
test_triangular(N, F) :- N > 1, N1 is N - 1, test_triangular(N1, F1), F is F1 + N.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Go Tail Recursion Tests
test_go_tail_recursion :-
    Test = 'Go: compile_tail_recursion_go/3',
    (   go_target:compile_tail_recursion_go(test_sum/3, [], Code),
        sub_atom(Code, _, _, _, 'for _, item := range'),
        sub_atom(Code, _, _, _, 'acc +=')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Go for loop')
    ).

%% Go Linear Recursion Tests
test_go_linear_recursion :-
    Test = 'Go: compile_linear_recursion_go/3',
    (   go_target:compile_linear_recursion_go(test_triangular/2, [], Code),
        sub_atom(Code, _, _, _, 'Memo'),
        sub_atom(Code, _, _, _, 'map[int]int')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Go memoization')
    ).

%% Rust Tail Recursion Tests
test_rust_tail_recursion :-
    Test = 'Rust: compile_tail_recursion_rust/3',
    (   rust_target:compile_tail_recursion_rust(test_sum/3, [], Code),
        sub_atom(Code, _, _, _, 'for &item in items'),
        sub_atom(Code, _, _, _, 'result +=')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Rust for loop')
    ).

%% Rust Linear Recursion Tests
test_rust_linear_recursion :-
    Test = 'Rust: compile_linear_recursion_rust/3',
    (   rust_target:compile_linear_recursion_rust(test_triangular/2, [], Code),
        sub_atom(Code, _, _, _, 'MEMO'),
        sub_atom(Code, _, _, _, 'HashMap')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Rust memoization')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('Tail and Linear Recursion Test Suite~n'),
    format('========================================~n~n'),
    
    format('--- Go Target Tests ---~n'),
    test_go_tail_recursion,
    test_go_linear_recursion,
    
    format('~n--- Rust Target Tests ---~n'),
    test_rust_tail_recursion,
    test_rust_linear_recursion,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
