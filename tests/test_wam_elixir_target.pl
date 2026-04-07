:- encoding(utf8).
% Test suite for WAM-to-Elixir transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_elixir_target.pl

:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Core generation tests

test_step_generation :-
    Test = 'WAM-Elixir: step/2 case expression generation',
    (   compile_step_wam_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def step(state, instr)'),
        sub_string(S, _, _, _, 'case instr do'),
        sub_string(S, _, _, _, ':get_constant')
    ->  pass(Test)
    ;   fail_test(Test, 'step generation missing expected content')
    ).

test_helpers_generation :-
    Test = 'WAM-Elixir: helper functions generation',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'def run(state)'),
        sub_string(S, _, _, _, 'def backtrack(state)'),
        sub_string(S, _, _, _, 'def unwind_trail(state, mark)')
    ->  pass(Test)
    ;   fail_test(Test, 'helper generation missing expected content')
    ).

test_runtime_assembly :-
    Test = 'WAM-Elixir: full runtime assembly',
    (   compile_wam_runtime_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'defmodule WamRuntime'),
        sub_string(S, _, _, _, 'defmodule WamState'),
        sub_string(S, _, _, _, 'defstruct'),
        sub_string(S, _, _, _, 'def step(state, instr)'),
        sub_string(S, _, _, _, 'def run(state)')
    ->  pass(Test)
    ;   fail_test(Test, 'runtime assembly missing expected content')
    ).

test_instruction_count :-
    Test = 'WAM-Elixir: instruction arm count',
    (   findall(N, wam_elixir_case(N, _), Cases),
        length(Cases, Count),
        Count >= 26
    ->  pass(Test),
        format('  (~w instruction arms)~n', [Count])
    ;   fail_test(Test, 'fewer than 26 instruction arms')
    ).

%% Instruction category tests

test_head_unification_instructions :-
    Test = 'WAM-Elixir: head unification instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(get_constant, Cases),
        member(get_variable, Cases),
        member(get_value, Cases),
        member(get_structure, Cases),
        member(get_list, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing head unification instruction arms')
    ).

test_body_construction_instructions :-
    Test = 'WAM-Elixir: body construction instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(put_constant, Cases),
        member(put_variable, Cases),
        member(put_value, Cases),
        member(put_structure, Cases),
        member(put_list, Cases),
        member(set_variable, Cases),
        member(set_value, Cases),
        member(set_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing body construction instruction arms')
    ).

test_unification_instructions :-
    Test = 'WAM-Elixir: unification instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(unify_variable, Cases),
        member(unify_value, Cases),
        member(unify_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing unification instruction arms')
    ).

test_control_flow_instructions :-
    Test = 'WAM-Elixir: control flow instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(call, Cases),
        member(execute, Cases),
        member(proceed, Cases),
        member(allocate, Cases),
        member(deallocate, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing control flow instruction arms')
    ).

test_choice_point_instructions :-
    Test = 'WAM-Elixir: choice point instructions present',
    (   findall(N, wam_elixir_case(N, _), Cases),
        member(try_me_else, Cases),
        member(retry_me_else, Cases),
        member(trust_me, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing choice point instruction arms')
    ).

test_choice_point_bytecode :-
    Test = 'WAM-Elixir: choice point code saves/restores state',
    (   wam_elixir_case(try_me_else, TryCode),
        wam_elixir_case(retry_me_else, RetryCode),
        wam_elixir_case(trust_me, TrustCode),
        sub_string(TryCode, _, _, _, 'choice_points'),
        sub_string(RetryCode, _, _, _, 'choice_points'),
        sub_string(TrustCode, _, _, _, 'choice_points')
    ->  pass(Test)
    ;   fail_test(Test, 'choice point code missing state save/restore')
    ).

test_builtin_call_delegates :-
    Test = 'WAM-Elixir: builtin_call delegates to execute_builtin',
    (   wam_elixir_case(builtin_call, Code),
        sub_string(Code, _, _, _, 'execute_builtin')
    ->  pass(Test)
    ;   fail_test(Test, 'builtin_call does not delegate to execute_builtin')
    ).

%% Code style tests

test_elixir_idioms :-
    Test = 'WAM-Elixir: uses Elixir idioms (pipe, pattern match)',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '|>'),
        sub_string(S, _, _, _, 'Map.put'),
        sub_string(S, _, _, _, 'match?')
    ->  pass(Test)
    ;   fail_test(Test, 'missing Elixir idioms')
    ).

test_immutable_state_updates :-
    Test = 'WAM-Elixir: uses immutable state updates (%{state | ...})',
    (   wam_elixir_case(put_constant, Code),
        sub_string(Code, _, _, _, '%{state |')
    ->  pass(Test)
    ;   fail_test(Test, 'not using immutable struct update syntax')
    ).

test_functional_run_loop :-
    Test = 'WAM-Elixir: run loop is recursive (not imperative)',
    (   compile_wam_helpers_to_elixir([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'run(new_state)')
    ->  pass(Test)
    ;   fail_test(Test, 'run loop not recursive')
    ).

%% Test runner

run_tests :-
    format('~n=== WAM-Elixir Target Tests ===~n~n'),
    test_step_generation,
    test_helpers_generation,
    test_runtime_assembly,
    test_instruction_count,
    test_head_unification_instructions,
    test_body_construction_instructions,
    test_unification_instructions,
    test_control_flow_instructions,
    test_choice_point_instructions,
    test_choice_point_bytecode,
    test_builtin_call_delegates,
    test_elixir_idioms,
    test_immutable_state_updates,
    test_functional_run_loop,
    format('~n=== WAM-Elixir Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
