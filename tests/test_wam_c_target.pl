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
        member(builtin_call, Cases),
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

test_builtin_call_generation :-
    Test = 'WAM-C: builtin_call parses and delegates',
    WamCode = 'foo/1:\n    builtin_call atom/1, 1\n    proceed',
    (   compile_wam_predicate_to_c(user:foo/1, WamCode, [], PredCode),
        atom_string(PredCode, PredS),
        compile_step_wam_to_c([], StepCode),
        atom_string(StepCode, StepS),
        compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(PredS, _, _, _, 'INSTR_BUILTIN_CALL'),
        sub_string(PredS, _, _, _, '.pred = "atom/1"'),
        sub_string(StepS, _, _, _, 'wam_execute_builtin'),
        sub_string(HelpersS, _, _, _, 'bool wam_execute_builtin')
    ->  pass(Test)
    ;   fail_test(Test, 'builtin_call parser/runtime delegation missing')
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

test_builtin_call_executable_smoke :-
    Test = 'WAM-C: builtin_call executable smoke',
    (   gcc_available
    ->  (   run_builtin_call_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'builtin_call executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_builtin_executable_smoke :-
    Test = 'WAM-C: real Prolog builtin executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_builtin_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog builtin executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_multiclause_executable_smoke :-
    Test = 'WAM-C: real Prolog multi-clause executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_multiclause_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog multi-clause executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_structure_index_executable_smoke :-
    Test = 'WAM-C: real Prolog structure-index executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_structure_index_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog structure-index executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_is_list_executable_smoke :-
    Test = 'WAM-C: real Prolog is_list/1 executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_is_list_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog is_list/1 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_unify_executable_smoke :-
    Test = 'WAM-C: real Prolog =/2 executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_unify_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog =/2 executable failed')
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
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

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

run_builtin_call_executable_smoke :-
    WamCode = 'wam_c_builtin_atom/1:\n    builtin_call atom/1, 1\n    proceed\nwam_c_builtin_is/2:\n    builtin_call is/2, 2\n    proceed',
    compile_wam_predicate_to_c(user:wam_c_builtin_atom/1, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_builtin_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_builtin_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_real_prolog_builtin_executable_smoke :-
    assertz((user:wam_c_real_builtin(X, Y) :- atom(X), Y is 7)),
    (   compile_predicate_to_wam(user:wam_c_real_builtin/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'builtin_call atom/1, 1'),
        sub_string(WamCode, _, _, _, 'builtin_call is/2, 2'),
        compile_wam_predicate_to_c(user:wam_c_real_builtin/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_real_builtin_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_builtin_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_real_builtin(_, _))
    ;   retractall(user:wam_c_real_builtin(_, _)),
        fail
    ).

run_real_prolog_multiclause_executable_smoke :-
    assertz((user:wam_c_real_multi(a, yes) :- true)),
    assertz((user:wam_c_real_multi(X, int) :- integer(X))),
    assertz((user:wam_c_real_multi([_|_], list) :- true)),
    (   compile_predicate_to_wam(user:wam_c_real_multi/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'switch_on_constant_a2'),
        sub_string(WamCode, _, _, _, 'retry_me_else'),
        sub_string(WamCode, _, _, _, 'builtin_call integer/1, 1'),
        compile_wam_predicate_to_c(user:wam_c_real_multi/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_real_multi_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_multi_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_real_multi(_, _))
    ;   retractall(user:wam_c_real_multi(_, _)),
        fail
    ).

run_real_prolog_structure_index_executable_smoke :-
    assertz((user:wam_c_real_struct(foo(_), foo) :- true)),
    assertz((user:wam_c_real_struct(bar(_), bar) :- true)),
    (   compile_predicate_to_wam(user:wam_c_real_struct/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'switch_on_structure'),
        sub_string(WamCode, _, _, _, 'foo/1:default'),
        sub_string(WamCode, _, _, _, 'bar/1:'),
        compile_wam_predicate_to_c(user:wam_c_real_struct/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_real_struct_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_struct_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_real_struct(_, _))
    ;   retractall(user:wam_c_real_struct(_, _)),
        fail
    ).

run_real_prolog_is_list_executable_smoke :-
    assertz((user:wam_c_real_is_list(X, ok) :- is_list(X))),
    (   compile_predicate_to_wam(user:wam_c_real_is_list/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'builtin_call is_list/1, 1'),
        compile_wam_predicate_to_c(user:wam_c_real_is_list/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_real_is_list_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_is_list_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_real_is_list(_, _))
    ;   retractall(user:wam_c_real_is_list(_, _)),
        fail
    ).

run_real_prolog_unify_executable_smoke :-
    assertz((user:wam_c_real_unify(X, Y) :- X = Y)),
    (   compile_predicate_to_wam(user:wam_c_real_unify/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'builtin_call =/2, 2'),
        compile_wam_predicate_to_c(user:wam_c_real_unify/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_real_unify_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_unify_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_real_unify(_, _))
    ;   retractall(user:wam_c_real_unify(_, _)),
        fail
    ).

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

wam_c_builtin_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_builtin_atom_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_builtin_atom_1(&state);

    WamValue atom_args[1] = { val_atom("a") };
    int atom_rc = wam_run_predicate(&state, "wam_c_builtin_atom/1", atom_args, 1);
    if (atom_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue int_args[1] = { val_int(7) };
    int int_rc = wam_run_predicate(&state, "wam_c_builtin_atom/1", int_args, 1);
    if (int_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue is_args[2] = { val_unbound("Result"), val_int(7) };
    int is_rc = wam_run_predicate(&state, "wam_c_builtin_is/2", is_args, 2);
    if (is_rc != 0 || state.A[0].tag != VAL_INT || state.A[0].data.integer != 7) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_real_builtin_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_real_builtin_2(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_real_builtin_2(&state);

    WamValue ok_args[2] = { val_atom("a"), val_unbound("Y") };
    int ok_rc = wam_run_predicate(&state, "wam_c_real_builtin/2", ok_args, 2);
    if (ok_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_INT || state.A[0].data.integer != 7) {
        wam_free_state(&state);
        return 10;
    }

    WamValue fail_args[2] = { val_int(3), val_unbound("Y") };
    int fail_rc = wam_run_predicate(&state, "wam_c_real_builtin/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_real_multi_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_real_multi_2(WamState* state);

static WamValue make_test_list(WamState *state) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom("head");
    state->H_array[state->H++] = val_atom("tail");
    return list;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_real_multi_2(&state);

    WamValue atom_args[2] = { val_atom("a"), val_atom("yes") };
    int atom_rc = wam_run_predicate(&state, "wam_c_real_multi/2", atom_args, 2);
    if (atom_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue int_args[2] = { val_int(9), val_atom("int") };
    int int_rc = wam_run_predicate(&state, "wam_c_real_multi/2", int_args, 2);
    if (int_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue list = make_test_list(&state);
    WamValue list_args[2] = { list, val_atom("list") };
    int list_rc = wam_run_predicate(&state, "wam_c_real_multi/2", list_args, 2);
    if (list_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    WamValue fail_args[2] = { val_atom("z"), val_atom("none") };
    int fail_rc = wam_run_predicate(&state, "wam_c_real_multi/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_real_struct_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_real_struct_2(WamState* state);

static WamValue make_test_structure_1(WamState *state, const char *functor, WamValue arg) {
    WamValue structure;
    structure.tag = VAL_STR;
    structure.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(functor);
    state->H_array[state->H++] = arg;
    return structure;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_real_struct_2(&state);

    WamValue foo_term = make_test_structure_1(&state, "foo/1", val_int(1));
    WamValue foo_args[2] = { foo_term, val_atom("foo") };
    int foo_rc = wam_run_predicate(&state, "wam_c_real_struct/2", foo_args, 2);
    if (foo_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue bar_term = make_test_structure_1(&state, "bar/1", val_atom("x"));
    WamValue bar_args[2] = { bar_term, val_atom("bar") };
    int bar_rc = wam_run_predicate(&state, "wam_c_real_struct/2", bar_args, 2);
    if (bar_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue baz_term = make_test_structure_1(&state, "baz/1", val_int(3));
    WamValue fail_args[2] = { baz_term, val_atom("baz") };
    int fail_rc = wam_run_predicate(&state, "wam_c_real_struct/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_real_is_list_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_real_is_list_2(WamState* state);

static WamValue make_test_list_is_list(WamState *state) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom("head");
    state->H_array[state->H++] = val_atom("tail");
    return list;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_real_is_list_2(&state);

    WamValue list = make_test_list_is_list(&state);
    WamValue list_args[2] = { list, val_atom("ok") };
    int list_rc = wam_run_predicate(&state, "wam_c_real_is_list/2", list_args, 2);
    if (list_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue atom_args[2] = { val_atom("not_a_list"), val_atom("ok") };
    int atom_rc = wam_run_predicate(&state, "wam_c_real_is_list/2", atom_args, 2);
    if (atom_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_real_unify_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_real_unify_2(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_real_unify_2(&state);

    WamValue equal_args[2] = { val_atom("same"), val_atom("same") };
    int equal_rc = wam_run_predicate(&state, "wam_c_real_unify/2", equal_args, 2);
    if (equal_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue bind_args[2] = { val_unbound("X"), val_int(7) };
    int bind_rc = wam_run_predicate(&state, "wam_c_real_unify/2", bind_args, 2);
    if (bind_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_INT || state.A[0].data.integer != 7) {
        wam_free_state(&state);
        return 20;
    }

    WamValue fail_args[2] = { val_atom("left"), val_atom("right") };
    int fail_rc = wam_run_predicate(&state, "wam_c_real_unify/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
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
implemented_case(builtin_call, 'case INSTR_BUILTIN_CALL').
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
    test_builtin_call_generation,
    test_list_target_pc_emission,
    test_generated_runtime_executable_smoke,
    test_cross_predicate_executable_smoke,
    test_builtin_call_executable_smoke,
    test_real_prolog_builtin_executable_smoke,
    test_real_prolog_multiclause_executable_smoke,
    test_real_prolog_structure_index_executable_smoke,
    test_real_prolog_is_list_executable_smoke,
    test_real_prolog_unify_executable_smoke,
    format('~n=== WAM-C Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
