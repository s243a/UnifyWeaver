:- encoding(utf8).
% Test suite for VB.NET and F# targets
% Usage: swipl -g run_tests -t halt tests/test_vbnet_fsharp_targets.pl

:- use_module('../src/unifyweaver/targets/vbnet_target').
:- use_module('../src/unifyweaver/targets/fsharp_target').

%% Test predicates
test_person(john, 25).
test_person(jane, 30).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% VB.NET Tests
test_vbnet_facts :-
    Test = 'VB.NET: compile_facts',
    (   vbnet_target:compile_predicate_to_vbnet(test_person/2, [], Code),
        sub_atom(Code, _, _, _, 'Public Class'),
        sub_atom(Code, _, _, _, 'GetAll'),
        sub_atom(Code, _, _, _, 'Stream')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected VB.NET constructs')
    ).

test_vbnet_tail_recursion :-
    Test = 'VB.NET: compile_tail_recursion',
    (   vbnet_target:compile_tail_recursion_vbnet(sum/2, [], Code),
        sub_atom(Code, _, _, _, 'Do While'),
        sub_atom(Code, _, _, _, 'accumulator')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Do While loop')
    ).

test_vbnet_linear_recursion :-
    Test = 'VB.NET: compile_linear_recursion',
    (   vbnet_target:compile_linear_recursion_vbnet(fib/2, [], Code),
        sub_atom(Code, _, _, _, 'Dictionary'),
        sub_atom(Code, _, _, _, '_memo')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Dictionary memoization')
    ).

test_vbnet_mutual_recursion :-
    Test = 'VB.NET: compile_mutual_recursion',
    (   vbnet_target:compile_mutual_recursion_vbnet([is_even/1, is_odd/1], [], Code),
        sub_atom(Code, _, _, _, 'is_even'),
        sub_atom(Code, _, _, _, 'is_odd')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing mutual functions')
    ).

%% F# Tests
test_fsharp_facts :-
    Test = 'F#: compile_facts',
    (   fsharp_target:compile_predicate_to_fsharp(test_person/2, [], Code),
        sub_atom(Code, _, _, _, 'type'),
        sub_atom(Code, _, _, _, 'getAll'),
        sub_atom(Code, _, _, _, 'stream')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected F# constructs')
    ).

test_fsharp_tail_recursion :-
    Test = 'F#: compile_tail_recursion',
    (   fsharp_target:compile_tail_recursion_fsharp(sum/2, [], Code),
        sub_atom(Code, _, _, _, 'let rec loop'),
        sub_atom(Code, _, _, _, 'acc')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing let rec loop')
    ).

test_fsharp_linear_recursion :-
    Test = 'F#: compile_linear_recursion',
    (   fsharp_target:compile_linear_recursion_fsharp(fib/2, [], Code),
        sub_atom(Code, _, _, _, 'Dictionary'),
        sub_atom(Code, _, _, _, 'memo')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Dictionary memoization')
    ).

test_fsharp_mutual_recursion :-
    Test = 'F#: compile_mutual_recursion',
    (   fsharp_target:compile_mutual_recursion_fsharp([is_even/1, is_odd/1], [], Code),
        sub_atom(Code, _, _, _, 'let rec is_even'),
        sub_atom(Code, _, _, _, 'and is_odd')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing mutual recursion with and')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('VB.NET and F# Target Test Suite~n'),
    format('========================================~n~n'),
    
    format('--- VB.NET Tests ---~n'),
    test_vbnet_facts,
    test_vbnet_tail_recursion,
    test_vbnet_linear_recursion,
    test_vbnet_mutual_recursion,
    
    format('~n--- F# Tests ---~n'),
    test_fsharp_facts,
    test_fsharp_tail_recursion,
    test_fsharp_linear_recursion,
    test_fsharp_mutual_recursion,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
