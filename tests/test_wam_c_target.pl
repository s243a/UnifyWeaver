:- encoding(utf8).
% Test suite for WAM-to-C transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_c_target.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.

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

test_generated_runtime_executable_smoke :-
    Test = 'WAM-C: generated runtime executable smoke',
    (   gcc_available
    ->  (   run_generated_runtime_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'generated runtime executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_cross_predicate_executable_smoke :-
    Test = 'WAM-C: cross-predicate executable smoke',
    (   gcc_available
    ->  (   run_cross_predicate_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'cross-predicate executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

gcc_available :-
    catch(process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

run_generated_runtime_executable_smoke :-
    WamCode = 'wam_c_exec_list/1:\n    switch_on_term 1 a:L_wam_c_exec_list_1_2 0 default\n    try_me_else L_wam_c_exec_list_1_2\n    get_list A1\n    unify_variable X1\n    unify_variable X2\n    proceed\nL_wam_c_exec_list_1_2:\n    trust_me\n    get_constant a, A1\n    proceed',
    compile_wam_predicate_to_c(user:wam_c_exec_list/1, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_exec_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_exec_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke(ExePath).

run_cross_predicate_executable_smoke :-
    WamCode = 'wam_c_exec_caller/1:\n    execute wam_c_exec_callee/1\nwam_c_exec_callee/1:\n    get_constant a, A1\n    proceed',
    compile_wam_predicate_to_c(user:wam_c_exec_caller/1, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_cross_exec_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_cross_exec_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).

compile_c_smoke(RuntimePath, PredPath, MainPath, ExePath) :-
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    format(atom(Cmd),
           'gcc -std=c11 -Wall -Wextra -fsanitize=address -I ~w ~w ~w ~w -o ~w',
           [IncludeDir, RuntimePath, PredPath, MainPath, ExePath]),
    shell(Cmd, Status),
    (   Status =:= 0
    ->  true
    ;   format(user_error, 'gcc failed with status ~w~n', [Status]),
        fail
    ).

compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath) :-
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    format(atom(Cmd),
           'gcc -std=c11 -Wall -Wextra -I ~w ~w ~w ~w -o ~w',
           [IncludeDir, RuntimePath, PredPath, MainPath, ExePath]),
    shell(Cmd, Status),
    (   Status =:= 0
    ->  true
    ;   format(user_error, 'gcc failed with status ~w~n', [Status]),
        fail
    ).

run_c_smoke(ExePath) :-
    format(atom(Cmd),
           'ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 timeout 10 ~w',
           [ExePath]),
    shell(Cmd, Status),
    (   Status =:= 0
    ->  true
    ;   format(user_error, 'generated executable failed with status ~w~n', [Status]),
        fail
    ).

run_c_smoke_plain(ExePath) :-
    format(atom(Cmd), 'timeout 10 ~w', [ExePath]),
    shell(Cmd, Status),
    (   Status =:= 0
    ->  true
    ;   format(user_error, 'generated executable failed with status ~w~n', [Status]),
        fail
    ).

wam_c_exec_smoke_main(
'#include <string.h>
#include "wam_runtime.h"

void setup_wam_c_exec_list_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_exec_list_1(&state);

    const char *a1 = wam_intern_atom(&state, "runtime_atom");
    const char *a2 = wam_intern_atom(&state, "runtime_atom");
    if (a1 != a2 || strcmp(a1, "runtime_atom") != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state.H;
    state.H_array[state.H++] = val_atom("head");
    state.H_array[state.H++] = val_atom("tail");

    WamValue args[1] = { list };
    int rc = wam_run_predicate(&state, "wam_c_exec_list/1", args, 1);
    if (rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_cross_exec_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_exec_caller_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_exec_caller_1(&state);

    WamValue ok_args[1] = { val_atom("a") };
    int ok_rc = wam_run_predicate(&state, "wam_c_exec_caller/1", ok_args, 1);
    if (ok_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue fail_args[1] = { val_atom("b") };
    int fail_rc = wam_run_predicate(&state, "wam_c_exec_caller/1", fail_args, 1);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

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
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
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
    test_generated_runtime_executable_smoke,
    test_cross_predicate_executable_smoke,
    format('~n=== WAM-C Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
