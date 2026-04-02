:- encoding(utf8).
% Test suite for WAM-to-Rust transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_target.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Tests

test_step_wam_generation :-
    Test = 'WAM-Rust: step() match arms generation',
    (   compile_step_wam_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn step'),
        sub_string(S, _, _, _, 'match instr'),
        sub_string(S, _, _, _, 'GetConstant'),
        sub_string(S, _, _, _, 'GetVariable'),
        sub_string(S, _, _, _, 'PutValue'),
        sub_string(S, _, _, _, 'Allocate'),
        sub_string(S, _, _, _, 'TryMeElse'),
        sub_string(S, _, _, _, 'Proceed'),
        sub_string(S, _, _, _, 'SwitchOnConstant')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected match arms in step()')
    ).

test_helpers_generation :-
    Test = 'WAM-Rust: helper functions generation',
    (   compile_wam_helpers_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn run'),
        sub_string(S, _, _, _, 'fn backtrack'),
        sub_string(S, _, _, _, 'fn unwind_trail'),
        sub_string(S, _, _, _, 'fn execute_builtin'),
        sub_string(S, _, _, _, 'fn eval_arith')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected helper functions')
    ).

test_full_runtime_generation :-
    Test = 'WAM-Rust: full runtime impl block',
    (   compile_wam_runtime_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'impl WamState'),
        sub_string(S, _, _, _, 'fn step'),
        sub_string(S, _, _, _, 'fn run'),
        sub_string(S, _, _, _, 'fn backtrack')
    ->  pass(Test)
    ;   fail_test(Test, 'Incomplete impl WamState block')
    ).

test_all_instruction_arms :-
    Test = 'WAM-Rust: all instruction types covered',
    (   compile_step_wam_to_rust([], Code),
        atom_string(Code, S),
        % Head unification
        sub_string(S, _, _, _, 'GetConstant'),
        sub_string(S, _, _, _, 'GetVariable'),
        sub_string(S, _, _, _, 'GetValue'),
        sub_string(S, _, _, _, 'GetStructure'),
        sub_string(S, _, _, _, 'GetList'),
        sub_string(S, _, _, _, 'UnifyVariable'),
        sub_string(S, _, _, _, 'UnifyValue'),
        sub_string(S, _, _, _, 'UnifyConstant'),
        % Body construction
        sub_string(S, _, _, _, 'PutConstant'),
        sub_string(S, _, _, _, 'PutVariable'),
        sub_string(S, _, _, _, 'PutValue'),
        sub_string(S, _, _, _, 'PutStructure'),
        sub_string(S, _, _, _, 'PutList'),
        sub_string(S, _, _, _, 'SetVariable'),
        sub_string(S, _, _, _, 'SetValue'),
        sub_string(S, _, _, _, 'SetConstant'),
        % Control
        sub_string(S, _, _, _, 'Allocate'),
        sub_string(S, _, _, _, 'Deallocate'),
        sub_string(S, _, _, _, 'Call('),
        sub_string(S, _, _, _, 'Execute('),
        sub_string(S, _, _, _, 'Proceed'),
        sub_string(S, _, _, _, 'BuiltinCall'),
        % Choice points
        sub_string(S, _, _, _, 'TryMeElse'),
        sub_string(S, _, _, _, 'TrustMe'),
        sub_string(S, _, _, _, 'RetryMeElse'),
        % Indexing
        sub_string(S, _, _, _, 'SwitchOnConstant'),
        sub_string(S, _, _, _, 'SwitchOnStructure'),
        sub_string(S, _, _, _, 'SwitchOnConstantA2')
    ->  pass(Test)
    ;   fail_test(Test, 'Not all instruction types have match arms')
    ).

test_builtin_dispatch :-
    Test = 'WAM-Rust: builtin dispatch covers all ops',
    (   compile_wam_helpers_to_rust([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'is/2'),
        sub_string(S, _, _, _, '>/2'),
        sub_string(S, _, _, _, '==/2'),
        sub_string(S, _, _, _, 'true/0'),
        sub_string(S, _, _, _, 'fail/0'),
        sub_string(S, _, _, _, '!/0'),
        sub_string(S, _, _, _, 'atom/1'),
        sub_string(S, _, _, _, 'number/1')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing builtin dispatch cases')
    ).

test_predicate_wrapper :-
    Test = 'WAM-Rust: predicate wrapper generation',
    (   compile_wam_predicate_to_rust(test_pred/2, "dummy", [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn test_pred'),
        sub_string(S, _, _, _, 'a1: Value'),
        sub_string(S, _, _, _, 'a2: Value'),
        sub_string(S, _, _, _, 'set_reg')
    ->  pass(Test)
    ;   fail_test(Test, 'Incorrect predicate wrapper')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('WAM-Rust Target Test Suite~n'),
    format('========================================~n~n'),

    test_step_wam_generation,
    test_helpers_generation,
    test_full_runtime_generation,
    test_all_instruction_arms,
    test_builtin_dispatch,
    test_predicate_wrapper,

    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
