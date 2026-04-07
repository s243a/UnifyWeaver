:- encoding(utf8).
% Test suite for WAM-to-C transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_c_target.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Core generation tests

test_step_generation :-
    Test = 'WAM-C: wam_step() switch generation',
    (   compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'int wam_step(WamState'),
        sub_string(S, _, _, _, 'switch (instr->tag)'),
        sub_string(S, _, _, _, 'WAM_GET_CONSTANT')
    ->  pass(Test)
    ;   fail_test(Test, 'step generation missing expected content')
    ).

test_helpers_generation :-
    Test = 'WAM-C: helper functions generation',
    (   compile_wam_helpers_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'int wam_run(WamState'),
        sub_string(S, _, _, _, 'int wam_backtrack(WamState'),
        sub_string(S, _, _, _, 'void wam_unwind_trail(WamState'),
        sub_string(S, _, _, _, 'int wam_unify(WamState')
    ->  pass(Test)
    ;   fail_test(Test, 'helper generation missing expected content')
    ).

test_runtime_assembly :-
    Test = 'WAM-C: full runtime assembly',
    (   compile_wam_runtime_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '#include <stdio.h>'),
        sub_string(S, _, _, _, '#include "wam_runtime.h"'),
        sub_string(S, _, _, _, 'wam_step'),
        sub_string(S, _, _, _, 'wam_run')
    ->  pass(Test)
    ;   fail_test(Test, 'runtime assembly missing expected content')
    ).

test_instruction_count :-
    Test = 'WAM-C: instruction arm count',
    (   findall(N, wam_c_case(N, _), Cases),
        length(Cases, Count),
        Count >= 26
    ->  pass(Test),
        format('  (~w instruction arms)~n', [Count])
    ;   fail_test(Test, 'fewer than 26 instruction arms')
    ).

%% Instruction category tests

test_head_unification_instructions :-
    Test = 'WAM-C: head unification instructions present',
    (   findall(N, wam_c_case(N, _), Cases),
        member(get_constant, Cases),
        member(get_variable, Cases),
        member(get_value, Cases),
        member(get_structure, Cases),
        member(get_list, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing head unification instruction arms')
    ).

test_body_construction_instructions :-
    Test = 'WAM-C: body construction instructions present',
    (   findall(N, wam_c_case(N, _), Cases),
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
    Test = 'WAM-C: unification instructions present',
    (   findall(N, wam_c_case(N, _), Cases),
        member(unify_variable, Cases),
        member(unify_value, Cases),
        member(unify_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing unification instruction arms')
    ).

test_control_flow_instructions :-
    Test = 'WAM-C: control flow instructions present',
    (   findall(N, wam_c_case(N, _), Cases),
        member(call, Cases),
        member(execute, Cases),
        member(proceed, Cases),
        member(allocate, Cases),
        member(deallocate, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing control flow instruction arms')
    ).

test_choice_point_instructions :-
    Test = 'WAM-C: choice point instructions present',
    (   findall(N, wam_c_case(N, _), Cases),
        member(try_me_else, Cases),
        member(retry_me_else, Cases),
        member(trust_me, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing choice point instruction arms')
    ).

test_choice_point_content :-
    Test = 'WAM-C: choice point code uses push/update/pop',
    (   wam_c_case(try_me_else, TryCode),
        wam_c_case(retry_me_else, RetryCode),
        wam_c_case(trust_me, TrustCode),
        sub_string(TryCode, _, _, _, 'wam_push_choice_point'),
        sub_string(RetryCode, _, _, _, 'wam_update_choice_point'),
        sub_string(TrustCode, _, _, _, 'wam_pop_choice_point')
    ->  pass(Test)
    ;   fail_test(Test, 'choice point bytecode missing expected functions')
    ).

test_builtin_call_delegates :-
    Test = 'WAM-C: builtin_call delegates to wam_execute_builtin',
    (   wam_c_case(builtin_call, Code),
        sub_string(Code, _, _, _, 'wam_execute_builtin')
    ->  pass(Test)
    ;   fail_test(Test, 'builtin_call does not delegate')
    ).

%% C idiom tests

test_c_pointer_access :-
    Test = 'WAM-C: uses pointer access (state->)',
    (   wam_c_case(get_constant, Code),
        sub_string(Code, _, _, _, 'state->pc'),
        sub_string(Code, _, _, _, 'wam_reg_get(state')
    ->  pass(Test)
    ;   fail_test(Test, 'not using C pointer access idiom')
    ).

test_c_return_pattern :-
    Test = 'WAM-C: uses return 1/0 pattern',
    (   wam_c_case(get_constant, Code),
        sub_string(Code, _, _, _, 'return 1'),
        sub_string(Code, _, _, _, 'return 0')
    ->  pass(Test)
    ;   fail_test(Test, 'not using C return 1/0 pattern')
    ).

test_c_memory_management :-
    Test = 'WAM-C: helpers include malloc/realloc/free patterns',
    (   compile_wam_helpers_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'realloc'),
        sub_string(S, _, _, _, 'strdup')
    ->  pass(Test)
    ;   fail_test(Test, 'missing memory management patterns')
    ).

test_c_while_loop :-
    Test = 'WAM-C: run loop uses while (not recursion)',
    (   compile_wam_helpers_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'while (state->pc != WAM_HALT)')
    ->  pass(Test)
    ;   fail_test(Test, 'run loop not using while')
    ).

%% Test runner

run_tests :-
    format('~n=== WAM-C Target Tests ===~n~n'),
    test_step_generation,
    test_helpers_generation,
    test_runtime_assembly,
    test_instruction_count,
    test_head_unification_instructions,
    test_body_construction_instructions,
    test_unification_instructions,
    test_control_flow_instructions,
    test_choice_point_instructions,
    test_choice_point_content,
    test_builtin_call_delegates,
    test_c_pointer_access,
    test_c_return_pattern,
    test_c_memory_management,
    test_c_while_loop,
    format('~n=== WAM-C Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
