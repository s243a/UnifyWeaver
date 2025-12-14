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
    Test = 'VB.NET: compile_facts_to_vbnet/3',
    (   vbnet_target:compile_predicate_to_vbnet(test_person/2, [], Code),
        sub_atom(Code, _, _, _, 'Public Class'),
        sub_atom(Code, _, _, _, 'GetAll')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected VB.NET constructs')
    ).

%% F# Tests
test_fsharp_facts :-
    Test = 'F#: compile_facts_to_fsharp/3',
    (   fsharp_target:compile_predicate_to_fsharp(test_person/2, [], Code),
        sub_atom(Code, _, _, _, 'type'),
        sub_atom(Code, _, _, _, 'getAll')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected F# constructs')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('VB.NET and F# Target Test Suite~n'),
    format('========================================~n~n'),
    
    format('--- VB.NET Tests ---~n'),
    test_vbnet_facts,
    
    format('~n--- F# Tests ---~n'),
    test_fsharp_facts,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
