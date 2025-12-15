:- encoding(utf8).
% Test suite for LLVM target
% Usage: swipl -g run_tests -t halt tests/test_llvm_target.pl

:- use_module('../src/unifyweaver/targets/llvm_target').

%% Test predicates (for facts test)
test_person(john, 25).
test_person(jane, 30).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Tests
test_llvm_tail_recursion :-
    Test = 'LLVM: compile_tail_recursion',
    (   llvm_target:compile_tail_recursion_llvm(sum/2, [], Code),
        sub_atom(Code, _, _, _, 'musttail call'),
        sub_atom(Code, _, _, _, 'define i64 @sum')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing musttail or function definition')
    ).

test_llvm_tail_recursion_export :-
    Test = 'LLVM: tail_recursion with export',
    (   llvm_target:compile_tail_recursion_llvm(sum/2, [export(true)], Code),
        sub_atom(Code, _, _, _, 'dllexport'),
        sub_atom(Code, _, _, _, '_ext')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing export wrapper')
    ).

test_llvm_linear_recursion :-
    Test = 'LLVM: compile_linear_recursion',
    (   llvm_target:compile_linear_recursion_llvm(fib/2, [], Code),
        sub_atom(Code, _, _, _, '@memo'),
        sub_atom(Code, _, _, _, 'getelementptr')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing memo table')
    ).

test_llvm_mutual_recursion :-
    Test = 'LLVM: compile_mutual_recursion',
    (   llvm_target:compile_mutual_recursion_llvm([is_even/1, is_odd/1], [], Code),
        sub_atom(Code, _, _, _, '@is_even'),
        sub_atom(Code, _, _, _, '@is_odd'),
        sub_atom(Code, _, _, _, 'musttail call i1 @is_odd'),
        sub_atom(Code, _, _, _, 'musttail call i1 @is_even')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing mutual recursion with musttail')
    ).

test_llvm_facts :-
    Test = 'LLVM: compile_facts',
    (   llvm_target:compile_predicate_to_llvm(test_person/2, [], Code),
        sub_atom(Code, _, _, _, '@str.'),
        sub_atom(Code, _, _, _, '_count')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing string constants or count')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('LLVM Target Test Suite~n'),
    format('========================================~n~n'),
    
    test_llvm_tail_recursion,
    test_llvm_tail_recursion_export,
    test_llvm_linear_recursion,
    test_llvm_mutual_recursion,
    test_llvm_facts,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
