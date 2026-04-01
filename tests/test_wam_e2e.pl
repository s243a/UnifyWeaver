:- encoding(utf8).
% End-to-end test for WAM target and runtime
% Usage: swipl -g run_tests -t halt tests/test_wam_e2e.pl

:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/runtime/wam_runtime').

%% Test data
e2e_parent(alice, bob).
e2e_parent(bob, charlie).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Tests
test_wam_compilation_and_execution :-
    Test = 'WAM E2E: Fact compilation and execution',
    (   % 1. Compile facts to WAM
        wam_target:compile_facts_to_wam(user:e2e_parent, 2, Code),
        % 2. Execute WAM code with query: e2e_parent(alice, bob)?
        wam_runtime:execute_wam(Code, e2e_parent(alice, bob), _FinalRegs)
    ->  pass(Test)
    ;   fail_test(Test, 'E2E execution failed')
    ).

test_wam_backtracking_simple :-
    Test = 'WAM E2E: Simple backtracking (second fact)',
    (   % 1. Compile facts
        wam_target:compile_facts_to_wam(user:e2e_parent, 2, Code),
        % 2. Execute with query: e2e_parent(bob, charlie)?
        % This exercises the backtracking path since the first fact (alice, bob) will fail.
        wam_runtime:execute_wam(Code, e2e_parent(bob, charlie), _)
    ->  pass(Test)
    ;   fail_test(Test, 'E2E execution failed for second fact')
    ).

run_tests :-
    format('~n========================================~n'),
    format('WAM Target E2E Test Suite~n'),
    format('========================================~n~n'),
    
    test_wam_compilation_and_execution,
    test_wam_backtracking_simple,
    
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All E2E tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
