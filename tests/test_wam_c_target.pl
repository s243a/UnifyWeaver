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
        sub_string(S, _, _, _, 'bool step_wam(WamState'),
        sub_string(S, _, _, _, 'switch (instr->tag)'),
        sub_string(S, _, _, _, 'INSTR_GET_CONSTANT')
    ->  pass(Test)
    ;   fail_test(Test, 'step generation missing expected content')
    ).

test_helpers_generation :-
    Test = 'WAM-C: helper functions generation',
    (   compile_wam_helpers_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'void wam_state_init(WamState'),
        sub_string(S, _, _, _, 'void wam_free_state(WamState'),
        sub_string(S, _, _, _, 'int wam_run_predicate(WamState'),
        sub_string(S, _, _, _, 'resolve_predicate_hash')
    ->  pass(Test)
    ;   fail_test(Test, 'helper generation missing expected content')
    ).

test_runtime_assembly :-
    Test = 'WAM-C: full runtime assembly',
    (   compile_wam_runtime_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, '#include "wam_runtime.h"'),
        sub_string(S, _, _, _, 'step_wam'),
        sub_string(S, _, _, _, 'wam_run')
    ->  pass(Test)
    ;   fail_test(Test, 'runtime assembly missing expected content')
    ).

test_instruction_count :-
    Test = 'WAM-C: instruction arm count',
    (   implemented_wam_c_cases(Cases),
        length(Cases, Count),
        Count >= 21
    ->  pass(Test),
        format('  (~w instruction arms)~n', [Count])
    ;   fail_test(Test, 'fewer than 26 instruction arms')
    ).

%% Instruction category tests

test_head_unification_instructions :-
    Test = 'WAM-C: head unification instructions present',
    (   implemented_wam_c_cases(Cases),
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
    (   implemented_wam_c_cases(Cases),
        member(put_constant, Cases),
        member(put_variable, Cases),
        member(put_value, Cases),
        member(put_structure, Cases),
        member(put_list, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing body construction instruction arms')
    ).

test_unification_instructions :-
    Test = 'WAM-C: unification instructions present',
    (   implemented_wam_c_cases(Cases),
        member(unify_variable, Cases),
        member(unify_value, Cases),
        member(unify_constant, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing unification instruction arms')
    ).

test_control_flow_instructions :-
    Test = 'WAM-C: control flow instructions present',
    (   implemented_wam_c_cases(Cases),
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
    (   implemented_wam_c_cases(Cases),
        member(try_me_else, Cases),
        member(retry_me_else, Cases),
        member(trust_me, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing choice point instruction arms')
    ).

test_choice_point_content :-
    Test = 'WAM-C: choice point code uses push/update/pop',
    (   compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'push_choice_point(state'),
        sub_string(S, _, _, _, 'cp->next_pc = target'),
        sub_string(S, _, _, _, 'pop_choice_point(state)')
    ->  pass(Test)
    ;   fail_test(Test, 'choice point bytecode missing expected functions')
    ).

test_switch_on_term_list_dispatch :-
    Test = 'WAM-C: switch_on_term dispatches lists directly',
    (   compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'cell->tag == VAL_LIST'),
        sub_string(S, _, _, _, 'instr->list_target_pc >= 0'),
        sub_string(S, _, _, _, 'state->P = instr->list_target_pc')
    ->  pass(Test)
    ;   fail_test(Test, 'switch_on_term missing direct list dispatch')
    ).

%% C idiom tests

test_c_pointer_access :-
    Test = 'WAM-C: uses pointer access (state->)',
    (   compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'state->P'),
        sub_string(S, _, _, _, 'resolve_reg(state')
    ->  pass(Test)
    ;   fail_test(Test, 'not using C pointer access idiom')
    ).

test_c_return_pattern :-
    Test = 'WAM-C: uses boolean return pattern',
    (   compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'return true'),
        sub_string(S, _, _, _, 'return false')
    ->  pass(Test)
    ;   fail_test(Test, 'not using C boolean return pattern')
    ).

test_c_memory_management :-
    Test = 'WAM-C: helpers include malloc/realloc/free patterns',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(HelpersS, _, _, _, 'malloc'),
        sub_string(HelpersS, _, _, _, 'free'),
        sub_string(S, _, _, _, 'realloc'),
        sub_string(S, _, _, _, 'free')
    ->  pass(Test)
    ;   fail_test(Test, 'missing memory management patterns')
    ).

test_c_while_loop :-
    Test = 'WAM-C: run loop uses while (not recursion)',
    (   compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'while (state->P >= 0 && state->P < state->code_size)')
    ->  pass(Test)
    ;   fail_test(Test, 'run loop not using while')
    ).

test_predicate_hash_registration :-
    Test = 'WAM-C: predicates register and resolve through hash table',
    WamCode = 'foo/1:\n    get_constant a, A1\n    proceed',
    (   compile_wam_predicate_to_c(user:foo/1, WamCode, [], PredCode),
        atom_string(PredCode, PredS),
        compile_step_wam_to_c([], StepCode),
        atom_string(StepCode, StepS),
        sub_string(PredS, _, _, _, 'wam_register_predicate_hash(state, "foo/1", 0)'),
        sub_string(StepS, _, _, _, 'resolve_predicate_hash(state, instr->pred)')
    ->  pass(Test)
    ;   fail_test(Test, 'predicate hash registration/lookup missing')
    ).

test_list_target_pc_emission :-
    Test = 'WAM-C: list-headed clauses emit list_target_pc',
    assertz((user:wam_c_list_case([_|_]) :- true)),
    assertz((user:wam_c_list_case(a) :- true)),
    (   compile_predicate_to_wam(user:wam_c_list_case/1, [], WamCode),
        compile_wam_predicate_to_c(user:wam_c_list_case/1, WamCode, [], CCode),
        atom_string(CCode, S),
        sub_string(S, _, _, _, '.list_target_pc = 1')
    ->  pass(Test)
    ;   fail_test(Test, 'list_target_pc not emitted')
    ),
    retractall(user:wam_c_list_case(_)).

implemented_wam_c_cases(Cases) :-
    compile_step_wam_to_c([], Code),
    atom_string(Code, S),
    findall(Name,
            ( implemented_case(Name, CEnum),
              sub_string(S, _, _, _, CEnum)
            ),
            Cases).

implemented_case(get_constant, 'case INSTR_GET_CONSTANT').
implemented_case(get_variable, 'case INSTR_GET_VARIABLE').
implemented_case(get_value, 'case INSTR_GET_VALUE').
implemented_case(put_constant, 'case INSTR_PUT_CONSTANT').
implemented_case(put_variable, 'case INSTR_PUT_VARIABLE').
implemented_case(put_value, 'case INSTR_PUT_VALUE').
implemented_case(allocate, 'case INSTR_ALLOCATE').
implemented_case(deallocate, 'case INSTR_DEALLOCATE').
implemented_case(proceed, 'case INSTR_PROCEED').
implemented_case(call, 'case INSTR_CALL').
implemented_case(execute, 'case INSTR_EXECUTE').
implemented_case(try_me_else, 'case INSTR_TRY_ME_ELSE').
implemented_case(retry_me_else, 'case INSTR_RETRY_ME_ELSE').
implemented_case(trust_me, 'case INSTR_TRUST_ME').
implemented_case(switch_on_constant, 'case INSTR_SWITCH_ON_CONSTANT').
implemented_case(switch_on_structure, 'case INSTR_SWITCH_ON_STRUCTURE').
implemented_case(switch_on_term, 'case INSTR_SWITCH_ON_TERM').
implemented_case(get_structure, 'case INSTR_GET_STRUCTURE').
implemented_case(put_structure, 'case INSTR_PUT_STRUCTURE').
implemented_case(get_list, 'case INSTR_GET_LIST').
implemented_case(put_list, 'case INSTR_PUT_LIST').
implemented_case(unify_variable, 'case INSTR_UNIFY_VARIABLE').
implemented_case(unify_value, 'case INSTR_UNIFY_VALUE').
implemented_case(unify_constant, 'case INSTR_UNIFY_CONSTANT').

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
    test_switch_on_term_list_dispatch,
    test_c_pointer_access,
    test_c_return_pattern,
    test_c_memory_management,
    test_c_while_loop,
    test_predicate_hash_registration,
    test_list_target_pc_emission,
    format('~n=== WAM-C Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
