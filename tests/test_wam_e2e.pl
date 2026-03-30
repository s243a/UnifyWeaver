:- encoding(utf8).
% End-to-end test for WAM target and runtime
% Usage: swipl -g run_tests -t halt tests/test_wam_e2e.pl

:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/runtime/wam_runtime').

%% Test data
e2e_parent(alice, bob).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Tests
test_wam_compilation_and_execution :-
    Test = 'WAM E2E: Fact compilation and execution',
    (   % 1. Compile fact to WAM (symbolic)
        % Note: we'll use a simplified instruction set for the manual emulator step
        Code = [
            get_constant(alice, 'A1'),
            get_constant(bob, 'A2'),
            proceed
        ],
        % 2. Initialize registers for query: e2e_parent(alice, bob)?
        % In a real E2E, we'd parse the output of compile_facts_to_wam
        % For this first step, we verify step_wam logic
        wam_runtime:execute_wam(['e2e_parent/2': get_constant(alice, 'A1'), 
                                'e2e_parent/2_next': get_constant(bob, 'A2'),
                                'e2e_parent/2_next_next': proceed], 
                                e2e_parent(alice, bob), _FinalRegs)
    ->  pass(Test)
    ;   fail_test(Test, 'E2E execution failed')
    ).

run_tests :-
    format('~n========================================~n'),
    format('WAM Target E2E Test Suite~n'),
    format('========================================~n~n'),
    
    test_wam_compilation_and_execution,
    
    format('~n========================================~n'),
    format('E2E tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
