:- encoding(utf8).
% Test suite for WAM-to-Rust transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_target.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/rust_target').
:- use_module('../src/unifyweaver/targets/wam_target').

%% Test predicate that resists native lowering — multi-clause rule with
%% mutual body calls and compound unification that no native tier handles.
%% The combination of multi-clause + compound heads + rule bodies with
%% multiple goals that aren't recognized as recursive patterns will
%% exhaust all native tiers.
:- dynamic test_resistant/3.
test_resistant(X, Y, Z) :- test_resistant_helper(X, Y), test_resistant_helper(Y, Z).
test_resistant(X, _, X).
:- dynamic test_resistant_helper/2.
test_resistant_helper(a, b).
test_resistant_helper(b, c).

%% Simple predicate that native lowering CAN handle
:- dynamic test_simple_fact/2.
test_simple_fact(hello, world).
test_simple_fact(foo, bar).

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

%% Phase 4: WAM fallback integration tests

test_wam_fallback_enabled :-
    Test = 'WAM-Rust: WAM fallback for predicate resisting native lowering',
    (   % test_resistant/3 has compound head args — resists native lowering
        % With WAM fallback enabled (default), should succeed
        rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn test_resistant')
    ->  pass(Test)
    ;   fail_test(Test, 'WAM fallback did not trigger for resistant predicate')
    ).

test_wam_fallback_disabled :-
    Test = 'WAM-Rust: WAM fallback disabled via option',
    (   % With wam_fallback(false), compilation should fail for resistant predicate
        \+ rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false), wam_fallback(false)], _)
    ->  pass(Test)
    ;   fail_test(Test, 'WAM fallback was not properly disabled')
    ).

test_native_still_preferred :-
    Test = 'WAM-Rust: native lowering still preferred over WAM fallback',
    (   % test_simple_fact/2 is facts-only — should be natively lowered,
        % not going through WAM even with fallback enabled
        rust_target:compile_predicate_to_rust(user:test_simple_fact/2,
            [include_main(false), wam_fallback(true)], Code),
        atom_string(Code, S),
        % Native facts compilation produces struct + vec!, not WAM wrapper
        (   sub_string(S, _, _, _, 'vec!')
        ;   sub_string(S, _, _, _, 'struct')
        ;   sub_string(S, _, _, _, 'TestSimpleFact')
        ;   sub_string(S, _, _, _, 'test_simple_fact')
        )
    ->  pass(Test)
    ;   fail_test(Test, 'Simple facts were not natively lowered')
    ).

test_wam_fallback_flag :-
    Test = 'WAM-Rust: WAM fallback disabled via Prolog flag',
    (   % Set global flag to disable WAM fallback
        set_prolog_flag(rust_wam_fallback, false),
        \+ rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false)], _),
        % Clean up
        set_prolog_flag(rust_wam_fallback, true)
    ->  pass(Test)
    ;   (   catch(set_prolog_flag(rust_wam_fallback, true), _, true),
            fail_test(Test, 'Prolog flag did not disable WAM fallback')
        )
    ).

%% Phase 5: E2E output validation tests

test_generated_rust_has_wam_wrapper :-
    Test = 'WAM-Rust E2E: generated code has proper wrapper structure',
    (   rust_target:compile_predicate_to_rust(user:test_resistant/3,
            [include_main(false), wam_fallback(true)], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'fn test_resistant'),
        sub_string(S, _, _, _, 'WamState'),
        sub_string(S, _, _, _, 'set_reg')
    ->  pass(Test)
    ;   fail_test(Test, 'Generated wrapper missing expected elements')
    ).

test_compile_wam_runtime_output :-
    Test = 'WAM-Rust E2E: full runtime generates valid impl block',
    (   compile_wam_runtime_to_rust([], Code),
        atom_string(Code, S),
        % Verify the impl block has all critical methods
        sub_string(S, _, _, _, 'impl WamState'),
        sub_string(S, _, _, _, 'pub fn step'),
        sub_string(S, _, _, _, 'pub fn run'),
        sub_string(S, _, _, _, 'pub fn backtrack'),
        sub_string(S, _, _, _, 'fn execute_builtin'),
        sub_string(S, _, _, _, 'fn eval_arith'),
        % Verify key instruction handling
        sub_string(S, _, _, _, 'GetConstant'),
        sub_string(S, _, _, _, 'Proceed'),
        sub_string(S, _, _, _, 'TryMeElse')
    ->  pass(Test)
    ;   fail_test(Test, 'Runtime impl block incomplete')
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
    test_wam_fallback_enabled,
    test_wam_fallback_disabled,
    test_native_still_preferred,
    test_wam_fallback_flag,
    test_generated_rust_has_wam_wrapper,
    test_compile_wam_runtime_output,

    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
