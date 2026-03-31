:- encoding(utf8).
% End-to-end test for WAM target and runtime
% Usage: swipl -g run_tests -t halt tests/test_wam_e2e.pl

:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/runtime/wam_runtime').

%% Test data
e2e_parent(alice, bob).
e2e_parent(bob, charlie).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

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
        % Note: our simple runtime doesn't do full backtracking yet, 
        % but it should find the second fact via labels and CP if implemented.
        % For now, execute_wam starts at the predicate label.
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
    format('E2E tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
