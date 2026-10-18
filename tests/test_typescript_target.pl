:- encoding(utf8).
% Test suite for TypeScript target
% Usage: swipl -q -g test_typescript_target -t halt tests/test_typescript_target.pl
%    or: swipl -g run_tests -t halt tests/test_typescript_target.pl

:- use_module('../src/unifyweaver/targets/typescript_target').

%% Test data (facts)
test_person(tom, 25).
test_person(bob, 30).

%% Fact data used by the fact-array regression test.
tsfact_edge(alice, bob).
tsfact_edge(bob, carol).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    assertz(test_failed).

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

%% Regression: compile_facts must emit populated fact rows (was empty due to
%% length/2 never being evaluated in format_ts_tuple/2 -> field names []).
test_typescript_facts_populated :-
    Test = 'TypeScript: compile_facts emits populated rows',
    (   typescript_target:compile_facts(tsfact_edge, 2, Code),
        % Both fact rows present
        sub_string(Code, _, _, _, 'arg1: "alice"'),
        sub_string(Code, _, _, _, 'arg1: "bob"'),
        sub_string(Code, _, _, _, 'arg2: "carol"'),
        % The array literal must not be empty
        \+ sub_string(Code, _, _, _, "[\n  \n]")
    ->  pass(Test)
    ;   fail_test(Test, 'Fact array is empty or rows missing')
    ).

%% Regression: the isXxx membership helper must contain a real JS match
%% expression, not the leaked Prolog term generate_match_expr([...]).
test_typescript_facts_no_leaked_term :-
    Test = 'TypeScript: membership helper has no leaked Prolog term',
    (   typescript_target:compile_facts(tsfact_edge, 2, Code),
        \+ sub_string(Code, _, _, _, 'generate_match_expr'),
        sub_string(Code, _, _, _, 'f.arg1 === arg1 && f.arg2 === arg2')
    ->  pass(Test)
    ;   fail_test(Test, 'Leaked generate_match_expr term or missing match expr')
    ).

%% Regression: transitive_closure must emit real closure logic, not fall
%% through to the tail_recursion branch.
test_typescript_transitive_closure :-
    Test = 'TypeScript: transitive_closure emits real closure logic',
    (   typescript_target:compile_recursion(path/2, [pattern(transitive_closure)], Code),
        sub_string(Code, _, _, _, 'Pattern: transitive_closure'),
        sub_string(Code, _, _, _, 'export const pathClosure'),
        sub_string(Code, _, _, _, 'Map<string, string[]>'),
        sub_string(Code, _, _, _, 'queue.shift()'),
        % must NOT be the tail-recursion fallthrough
        \+ sub_string(Code, _, _, _, 'Pattern: tail_recursion'),
        \+ sub_string(Code, _, _, _, 'acc + n')
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_closure fell through or missing BFS logic')
    ).

%% Run all tests
run_tests :-
    retractall(test_failed),
    format('~n========================================~n'),
    format('TypeScript Target Test Suite~n'),
    format('========================================~n~n'),

    test_typescript_target_info,
    test_typescript_tail_recursion,
    test_typescript_list_fold,
    test_typescript_linear_recursion,
    test_typescript_module,
    test_typescript_facts_populated,
    test_typescript_facts_no_leaked_term,
    test_typescript_transitive_closure,

    format('~n========================================~n'),
    (   test_failed
    ->  format('SOME TESTS FAILED~n'),
        format('========================================~n'),
        fail
    ;   format('All tests passed~n'),
        format('========================================~n')
    ).

%% Named entry point used by the acceptance command.
test_typescript_target :- run_tests.
