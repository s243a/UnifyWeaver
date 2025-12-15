:- encoding(utf8).
% Test suite for TypeScript target
% Usage: swipl -g run_tests -t halt tests/test_typescript_target.pl

:- use_module('../src/unifyweaver/targets/typescript_target').

%% Test data (facts)
test_person(tom, 25).
test_person(bob, 30).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Tests
test_typescript_target_info :-
    Test = 'TypeScript: target_info',
    (   typescript_target:target_info(Info),
        Info.name == "TypeScript",
        Info.family == javascript,
        member(types, Info.features)
    ->  pass(Test)
    ;   fail_test(Test, 'Missing info fields')
    ).

test_typescript_tail_recursion :-
    Test = 'TypeScript: tail_recursion',
    (   typescript_target:compile_recursion(sum/2, [pattern(tail_recursion)], Code),
        sub_string(Code, _, _, _, 'export const sum'),
        sub_string(Code, _, _, _, ': number')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing TypeScript function')
    ).

test_typescript_list_fold :-
    Test = 'TypeScript: list_fold',
    (   typescript_target:compile_recursion(listSum/2, [pattern(list_fold)], Code),
        sub_string(Code, _, _, _, 'reduce'),
        sub_string(Code, _, _, _, 'number[]')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing reduce or array type')
    ).

test_typescript_linear_recursion :-
    Test = 'TypeScript: linear_recursion (fibonacci)',
    (   typescript_target:compile_recursion(fib/2, [pattern(linear_recursion)], Code),
        sub_string(Code, _, _, _, 'Map<number, number>'),
        sub_string(Code, _, _, _, 'fib(n - 1)')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing memoization or recursion')
    ).

test_typescript_module :-
    Test = 'TypeScript: compile_module',
    (   typescript_target:compile_module(
            [pred(sum, 2, tail_recursion), pred(factorial, 1, factorial)],
            [module_name('PrologMath')],
            Code),
        sub_string(Code, _, _, _, 'Module: PrologMath'),
        sub_string(Code, _, _, _, 'export const sum'),
        sub_string(Code, _, _, _, 'export const factorial')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing module or functions')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('TypeScript Target Test Suite~n'),
    format('========================================~n~n'),
    
    test_typescript_target_info,
    test_typescript_tail_recursion,
    test_typescript_list_fold,
    test_typescript_linear_recursion,
    test_typescript_module,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
