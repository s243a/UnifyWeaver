:- encoding(utf8).
% Test suite for WAM-to-C transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_c_target.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(filesex), [directory_file_path/3]).
:- use_module(library(readutil), [read_file_to_string/3]).

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
        member(put_list, Cases),
        member(set_variable, Cases),
        member(set_value, Cases),
        member(set_constant, Cases)
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
        member(call_foreign, Cases),
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
        sub_string(S, _, _, _, 'instr->as.switch_index.list_target_pc >= 0'),
        sub_string(S, _, _, _, 'state->P = instr->as.switch_index.list_target_pc')
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
        sub_string(StepS, _, _, _, 'resolve_predicate_hash(state, instr->as.pred.pred)')
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

test_call_foreign_generation :-
    Test = 'WAM-C: call_foreign parses and dispatches registered handlers',
    WamCode = 'foo/1:\n    call_foreign foo/1, 1\n    proceed',
    (   compile_wam_predicate_to_c(user:foo/1, WamCode, [], PredCode),
        atom_string(PredCode, PredS),
        compile_step_wam_to_c([], StepCode),
        atom_string(StepCode, StepS),
        compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        sub_string(PredS, _, _, _, 'INSTR_CALL_FOREIGN'),
        sub_string(PredS, _, _, _, '.pred = "foo/1"'),
        sub_string(StepS, _, _, _, 'wam_execute_foreign_predicate'),
        sub_string(HelpersS, _, _, _, 'bool wam_execute_foreign_predicate')
    ->  pass(Test)
    ;   fail_test(Test, 'call_foreign parser/runtime delegation missing')
    ).

test_category_ancestor_kernel_generation :-
    Test = 'WAM-C: category_ancestor native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_category_parent'),
        sub_string(S, _, _, _, 'void wam_register_category_ancestor_kernel'),
        sub_string(S, _, _, _, 'bool wam_category_ancestor_handler'),
        sub_string(S, _, _, _, 'wam_category_ancestor_dfs')
    ->  pass(Test)
    ;   fail_test(Test, 'category_ancestor native kernel helpers missing')
    ).

test_fact_source_generation :-
    Test = 'WAM-C: file FactSource helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_fact_source_init'),
        sub_string(S, _, _, _, 'bool wam_fact_source_load_tsv'),
        sub_string(S, _, _, _, 'bool wam_fact_source_load_lmdb'),
        sub_string(S, _, _, _, 'int wam_fact_source_lookup_arg1'),
        sub_string(S, _, _, _, 'bool wam_register_category_parent_fact_source')
    ->  pass(Test)
    ;   fail_test(Test, 'file FactSource helpers missing')
    ).

test_streaming_foreign_results_generation :-
    Test = 'WAM-C: streaming foreign result helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_int_results_init'),
        sub_string(S, _, _, _, 'bool wam_int_results_push'),
        sub_string(S, _, _, _, 'bool wam_collect_category_ancestor_hops'),
        sub_string(S, _, _, _, 'results.values[0]')
    ->  pass(Test)
    ;   fail_test(Test, 'streaming foreign result helpers missing')
    ).

test_kernel_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits category_ancestor setup',
    setup_wam_c_detector_category_ancestor,
    (   detect_kernels([user:category_ancestor/4], Detected),
        Detected = ['category_ancestor/4'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_category_ancestor_kernel(state, "category_ancestor/4", 10)')
    ->  cleanup_wam_c_detector_category_ancestor,
        pass(Test)
    ;   cleanup_wam_c_detector_category_ancestor,
        fail_test(Test, 'detected kernel setup missing')
    ).

test_kernel_detector_project_generation :-
    Test = 'WAM-C: project generation lowers detected kernel to foreign trampoline',
    setup_wam_c_detector_category_ancestor,
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_kernel_detector_project_~w', [Stamp]),
    (   write_wam_c_project([user:category_ancestor/4], [], ProjectDir),
        directory_file_path(ProjectDir, 'lib.c', LibPath),
        read_file_to_string(LibPath, LibCode, []),
        sub_string(LibCode, _, _, _, '#include "wam_runtime.h"'),
        sub_string(LibCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_string(LibCode, _, _, _, 'wam_register_category_ancestor_kernel(state, "category_ancestor/4", 10)'),
        sub_string(LibCode, _, _, _, 'INSTR_CALL_FOREIGN')
    ->  cleanup_wam_c_detector_category_ancestor,
        pass(Test)
    ;   cleanup_wam_c_detector_category_ancestor,
        fail_test(Test, 'generated project did not lower detected kernel')
    ).

test_lowered_fact_helper_generation :-
    Test = 'WAM-C: fact-only predicates can lower to native helper',
    assertz(user:wam_c_lowered_pair(a, b)),
    assertz(user:wam_c_lowered_pair(a, c)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_fact_project_~w', [Stamp]),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   write_wam_c_project([user:wam_c_lowered_pair/2], [lowered_helpers(true)], ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_lowered_pair_2'),
        sub_string(LibS, _, _, _, 'void setup_lowered_wam_c_helpers'),
        sub_string(LibS, _, _, _, 'wam_register_foreign_predicate(state, "wam_c_lowered_pair/2", 2'),
        sub_string(LibS, _, _, _, 'INSTR_CALL_FOREIGN'),
        sub_string(LibS, _, _, _, '.pred = "wam_c_lowered_pair/2"')
    ->  pass(Test)
    ;   fail_test(Test, 'lowered fact helper was not emitted')
    ),
    retractall(user:wam_c_lowered_pair(_, _)).

test_lowered_helper_planner_metadata :-
    Test = 'WAM-C: lowered helper planner reports routing decisions',
    assertz(user:wam_c_plan_fact(a, b)),
    assertz((user:wam_c_plan_rule(X) :- user:wam_c_plan_fact(X, _))),
    (   plan_wam_c_lowered_helpers([user:wam_c_plan_fact/2,
                                     user:wam_c_plan_rule/1,
                                     user:category_ancestor/4],
                                    [lowered_helpers(true)],
                                    ['category_ancestor/4'],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_plan_fact/2', _, lowered, fact_only([[a,b]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_plan_rule/1', _, rejected, non_fact_clause), Plans),
        member(wam_c_lowered_helper_plan('category_ancestor/4', _, interpreted, detected_kernel), Plans)
    ->  pass(Test)
    ;   fail_test(Test, 'planner did not classify lowered/rejected/interpreted predicates')
    ),
    retractall(user:wam_c_plan_fact(_, _)),
    retractall(user:wam_c_plan_rule(_)).

test_lowered_helper_plan_generation :-
    Test = 'WAM-C: generated project reports lowered helper plan',
    assertz(user:wam_c_plan_emit_fact(a, b)),
    assertz((user:wam_c_plan_emit_rule(X) :- user:wam_c_plan_emit_fact(X, _))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_plan_project_~w', [Stamp]),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   write_wam_c_project([user:wam_c_plan_emit_fact/2,
                             user:wam_c_plan_emit_rule/1],
                            [lowered_helpers(true), report_lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// WAM-C lowered helper plan'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_plan_emit_fact/2: fact_only'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_plan_emit_rule/1: non_fact_clause'),
        sub_string(LibS, _, _, _, 'INSTR_CALL_FOREIGN'),
        sub_string(LibS, _, _, _, 'setup_wam_c_plan_emit_rule_1')
    ->  pass(Test)
    ;   fail_test(Test, 'generated project did not include lowered helper plan metadata')
    ),
    retractall(user:wam_c_plan_emit_fact(_, _)),
    retractall(user:wam_c_plan_emit_rule(_)).

test_lowered_body_call_helper_generation :-
    Test = 'WAM-C: deterministic body-call predicates can lower to native helper',
    assertz(user:wam_c_body_fact(a, b)),
    assertz((user:wam_c_body_alias(X, Y) :- user:wam_c_body_fact(X, Y))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_body_project_~w', [Stamp]),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_body_fact/2,
                                     user:wam_c_body_alias/2],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_fact/2', _, lowered, fact_only([[a,b]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_alias/2', _, lowered, body_call('wam_c_body_fact/2', 2)), Plans),
        write_wam_c_project([user:wam_c_body_fact/2,
                             user:wam_c_body_alias/2],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - lowered wam_c_body_alias/2: body_call'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_body_alias_2'),
        sub_string(LibS, _, _, _, 'return wam_execute_foreign_predicate(state, "wam_c_body_fact/2", 2);'),
        sub_string(LibS, _, _, _, '.pred = "wam_c_body_alias/2"')
    ->  pass(Test)
    ;   fail_test(Test, 'lowered body-call helper was not emitted')
    ),
    retractall(user:wam_c_body_fact(_, _)),
    retractall(user:wam_c_body_alias(_, _)).

test_lowered_filtered_fact_helper_generation :-
    Test = 'WAM-C: guarded fact predicates can lower to filtered native helper',
    assertz(user:wam_c_filter_fact(a, keep)),
    assertz(user:wam_c_filter_fact(b, drop)),
    assertz(user:wam_c_filter_fact(c, keep)),
    assertz((user:wam_c_filter_keep(X) :- user:wam_c_filter_fact(X, keep))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_filter_project_~w', [Stamp]),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_filter_fact/2,
                                     user:wam_c_filter_keep/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_fact/2', _, lowered, fact_only([[a,keep],[b,drop],[c,keep]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_keep/1', _, lowered, filtered_fact('wam_c_filter_fact/2', [[a],[c]])), Plans),
        write_wam_c_project([user:wam_c_filter_fact/2,
                             user:wam_c_filter_keep/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - lowered wam_c_filter_keep/1: filtered_fact'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_filter_keep_1'),
        sub_string(LibS, _, _, _, 'val_atom("a")'),
        sub_string(LibS, _, _, _, 'val_atom("c")'),
        sub_string(LibS, _, _, _, '.pred = "wam_c_filter_keep/1"')
    ->  pass(Test)
    ;   fail_test(Test, 'lowered filtered fact helper was not emitted')
    ),
    retractall(user:wam_c_filter_fact(_, _)),
    retractall(user:wam_c_filter_keep(_)).

test_unsupported_instruction_fails_loudly :-
    Test = 'WAM-C: unsupported instructions fail loudly',
    BadWamCode = 'foo/1:\n    unknown_opcode A1\n    proceed',
    (   catch(wam_instruction_to_c_literal(unknown_opcode('A1'), _),
              error(wam_c_target_error(unsupported_instruction(unknown_opcode('A1'))), _),
              TermThrows = true),
        catch(compile_wam_predicate_to_c(user:foo/1, BadWamCode, [], _),
              error(wam_c_target_error(unsupported_instruction_tokens(["unknown_opcode", "A1"])), _),
              LineThrows = true),
        TermThrows == true,
        LineThrows == true
    ->  pass(Test)
    ;   fail_test(Test, 'unsupported instruction was silently emitted')
    ).

test_no_zero_instruction_fallback :-
    Test = 'WAM-C: unsupported pass-2 instructions never emit zero fallback',
    BadWamCode = 'foo/1:\n    definitely_unknown A1\n    proceed',
    (   catch(compile_wam_predicate_to_c(user:foo/1, BadWamCode, [], CCode),
              error(wam_c_target_error(unsupported_instruction_tokens(["definitely_unknown", "A1"])), _),
              Throws = true),
        Throws == true,
        \+ ( nonvar(CCode),
             atom_string(CCode, S),
             sub_string(S, _, _, _, '(Instruction){0}')
           )
    ->  pass(Test)
    ;   fail_test(Test, 'unsupported pass-2 instruction emitted {0}')
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

test_call_foreign_executable_smoke :-
    Test = 'WAM-C: call_foreign executable smoke',
    (   gcc_available
    ->  (   run_call_foreign_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'call_foreign executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_category_ancestor_kernel_executable_smoke :-
    Test = 'WAM-C: category_ancestor native kernel executable smoke',
    (   gcc_available
    ->  (   run_category_ancestor_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'category_ancestor native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_fact_source_executable_smoke :-
    Test = 'WAM-C: file FactSource executable smoke',
    (   gcc_available
    ->  (   run_fact_source_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'file FactSource executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_lmdb_fact_source_executable_smoke :-
    Test = 'WAM-C: LMDB FactSource executable smoke',
    (   gcc_available,
        lmdb_available
    ->  (   run_lmdb_fact_source_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'LMDB FactSource executable failed')
        )
    ;   format('[PASS] ~w (gcc or lmdb unavailable; skipped executable smoke)~n', [Test])
    ).

test_streaming_foreign_results_executable_smoke :-
    Test = 'WAM-C: streaming category_ancestor result executable smoke',
    (   gcc_available
    ->  (   run_streaming_foreign_results_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'streaming category_ancestor result executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_kernel_detector_executable_smoke :-
    Test = 'WAM-C: detected category_ancestor executable smoke',
    (   gcc_available
    ->  (   run_kernel_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected category_ancestor executable failed')
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

test_real_prolog_classic_recursive_executable_smoke :-
    Test = 'WAM-C: real Prolog classic recursive executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_classic_recursive_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog classic recursive executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_lowered_fact_helper_executable_smoke :-
    Test = 'WAM-C: lowered fact helper executable smoke',
    (   gcc_available
    ->  (   run_lowered_fact_helper_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'lowered fact helper executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_lowered_body_call_helper_executable_smoke :-
    Test = 'WAM-C: lowered body-call helper executable smoke',
    (   gcc_available
    ->  (   run_lowered_body_call_helper_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'lowered body-call helper executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_lowered_filtered_fact_helper_executable_smoke :-
    Test = 'WAM-C: lowered filtered fact helper executable smoke',
    (   gcc_available
    ->  (   run_lowered_filtered_fact_helper_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'lowered filtered fact helper executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_asan_memory_lifecycle_executable_smoke :-
    Test = 'WAM-C: ASAN memory lifecycle executable smoke',
    (   asan_available
    ->  (   run_asan_memory_lifecycle_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'ASAN memory lifecycle executable failed')
        )
    ;   format('[PASS] ~w (ASAN unavailable; skipped executable smoke)~n', [Test])
    ).

gcc_available :-
    catch(process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

asan_available :-
    gcc_available,
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_asan_probe_~w', [Stamp]),
    format(atom(SourcePath), '~w.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    format(atom(LogPath), '~w.log', [TmpBase]),
    write_text_file(SourcePath, 'int main(void) { return 0; }\n'),
    format(atom(CompileCmd),
           'gcc -std=c11 -fsanitize=address ~w -o ~w',
           [SourcePath, ExePath]),
    catch(shell(CompileCmd, CompileStatus), _, fail),
    CompileStatus =:= 0,
    format(atom(RunCmd),
           'ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 timeout 5 ~w > ~w 2>&1',
           [ExePath, LogPath]),
    catch(shell(RunCmd, RunStatus), _, fail),
    RunStatus =:= 0.

lmdb_available :-
    catch(process_create(path('pkg-config'), ['--exists', 'lmdb'],
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

run_call_foreign_executable_smoke :-
    WamCode = 'wam_c_foreign_emit/1:\n    call_foreign wam_c_foreign_emit/1, 1\n    proceed',
    compile_wam_predicate_to_c(user:wam_c_foreign_emit/1, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_foreign_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_foreign_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_category_ancestor_kernel_executable_smoke :-
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_category_ancestor_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_category_ancestor_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_fact_source_executable_smoke :-
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_fact_source_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(DataPath), '~w_edges.tsv', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    write_text_file(DataPath, 'leaf\tmid\nmid\troot\nleaf\tother\n'),
    wam_c_fact_source_smoke_main(DataPath, MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_lmdb_fact_source_executable_smoke :-
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_lmdb_fact_source_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(EnvPath), '~w_env', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_lmdb_fact_source_smoke_main(EnvPath, MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_lmdb(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_kernel_detector_executable_smoke :-
    setup_wam_c_detector_category_ancestor,
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_kernel_detector_smoke_~w', [Stamp]),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_kernel_detector_smoke', ExePath),
    (   write_wam_c_project([user:category_ancestor/4], [], ProjectDir),
        wam_c_kernel_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_category_ancestor
    ;   cleanup_wam_c_detector_category_ancestor,
        fail
    ).

run_streaming_foreign_results_executable_smoke :-
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(TmpBase), '/tmp/unifyweaver_wam_c_streaming_foreign_smoke_~w', [Stamp]),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_streaming_foreign_smoke_main(MainCode),
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

run_real_prolog_classic_recursive_executable_smoke :-
    assertz((user:wam_c_classic_fib(0, 0) :- true)),
    assertz((user:wam_c_classic_fib(1, 1) :- true)),
    assertz((user:wam_c_classic_fib(N, F) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        wam_c_classic_fib(N1, F1),
        wam_c_classic_fib(N2, F2),
        F is F1 + F2)),
    (   compile_predicate_to_wam(user:wam_c_classic_fib/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'retry_me_else'),
        sub_string(WamCode, _, _, _, 'trust_me'),
        sub_string(WamCode, _, _, _, 'builtin_call >/2, 2'),
        sub_string(WamCode, _, _, _, 'builtin_call is/2, 2'),
        sub_string(WamCode, _, _, _, 'call wam_c_classic_fib/2, 2'),
        compile_wam_predicate_to_c(user:wam_c_classic_fib/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_classic_fib_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_classic_fib_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_classic_fib(_, _))
    ;   retractall(user:wam_c_classic_fib(_, _)),
        fail
    ).

run_lowered_fact_helper_executable_smoke :-
    assertz(user:wam_c_lowered_pair(a, b)),
    assertz(user:wam_c_lowered_pair(a, c)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_fact_smoke_~w', [Stamp]),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_lowered_fact_smoke', ExePath),
    (   write_wam_c_project([user:wam_c_lowered_pair/2], [lowered_helpers(true)], ProjectDir),
        wam_c_lowered_fact_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_lowered_pair(_, _))
    ;   retractall(user:wam_c_lowered_pair(_, _)),
        fail
    ).

run_lowered_body_call_helper_executable_smoke :-
    assertz(user:wam_c_body_fact(a, b)),
    assertz(user:wam_c_body_fact(a, c)),
    assertz((user:wam_c_body_alias(X, Y) :- user:wam_c_body_fact(X, Y))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_body_smoke_~w', [Stamp]),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_lowered_body_smoke', ExePath),
    (   write_wam_c_project([user:wam_c_body_fact/2,
                             user:wam_c_body_alias/2],
                            [lowered_helpers(true)],
                            ProjectDir),
        wam_c_lowered_body_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_body_fact(_, _)),
        retractall(user:wam_c_body_alias(_, _))
    ;   retractall(user:wam_c_body_fact(_, _)),
        retractall(user:wam_c_body_alias(_, _)),
        fail
    ).

run_lowered_filtered_fact_helper_executable_smoke :-
    assertz(user:wam_c_filter_fact(a, keep)),
    assertz(user:wam_c_filter_fact(b, drop)),
    assertz(user:wam_c_filter_fact(c, keep)),
    assertz((user:wam_c_filter_keep(X) :- user:wam_c_filter_fact(X, keep))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    format(atom(ProjectDir), '/tmp/unifyweaver_wam_c_lowered_filter_smoke_~w', [Stamp]),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_lowered_filter_smoke', ExePath),
    (   write_wam_c_project([user:wam_c_filter_fact/2,
                             user:wam_c_filter_keep/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        wam_c_lowered_filter_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_filter_fact(_, _)),
        retractall(user:wam_c_filter_keep(_))
    ;   retractall(user:wam_c_filter_fact(_, _)),
        retractall(user:wam_c_filter_keep(_)),
        fail
    ).

run_asan_memory_lifecycle_executable_smoke :-
    assertz((user:wam_c_asan_term(a, atom) :- true)),
    assertz((user:wam_c_asan_term(foo(_), struct) :- true)),
    assertz((user:wam_c_asan_term([_|_], list) :- true)),
    (   compile_predicate_to_wam(user:wam_c_asan_term/2, [], TermWamCode),
        sub_string(TermWamCode, _, _, _, 'switch_on_term'),
        compile_wam_predicate_to_c(user:wam_c_asan_term/2, TermWamCode, [], TermPredCode),
        CategoryWamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
        compile_wam_predicate_to_c(user:category_ancestor/4, CategoryWamCode, [], CategoryPredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        format(atom(TmpBase), '/tmp/unifyweaver_wam_c_asan_lifecycle_smoke_~w', [Stamp]),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(DataPath), '~w_edges.tsv', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w~n~n~w', [TermPredCode, CategoryPredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        write_text_file(DataPath, 'leaf\tmid\nmid\troot\nleaf\tother\n'),
        wam_c_asan_lifecycle_smoke_main(DataPath, MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke(ExePath)
    ->  retractall(user:wam_c_asan_term(_, _))
    ;   retractall(user:wam_c_asan_term(_, _)),
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

compile_c_smoke_lmdb(RuntimePath, PredPath, MainPath, ExePath) :-
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    format(atom(Cmd),
           'gcc -std=c11 -Wall -Wextra -DWAM_C_ENABLE_LMDB -I ~w ~w ~w ~w -llmdb -o ~w',
           [IncludeDir, RuntimePath, PredPath, MainPath, ExePath]),
    shell(Cmd, Status),
    (   Status =:= 0
    ->  true
    ;   format(user_error, 'gcc lmdb smoke failed with status ~w~n', [Status]),
        fail
    ).

setup_wam_c_detector_category_ancestor :-
    cleanup_wam_c_detector_category_ancestor,
    assertz(user:max_depth(10)),
    assertz((user:category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited))),
    assertz((user:category_ancestor(Cat, Ancestor, Hops, Visited) :-
        max_depth(MaxD),
        length(Visited, D),
        D < MaxD,
        !,
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)).

cleanup_wam_c_detector_category_ancestor :-
    retractall(user:max_depth(_)),
    retractall(user:category_ancestor(_, _, _, _)).

run_c_smoke(ExePath) :-
    format(atom(LogPath), '~w.asan.log', [ExePath]),
    format(atom(Cmd),
           'ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 timeout 10 ~w > ~w 2>&1',
           [ExePath, LogPath]),
    shell(Cmd, Status),
    (   Status =:= 0
    ->  true
    ;   format(user_error, 'generated executable failed with status ~w (log: ~w)~n', [Status, LogPath]),
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

wam_c_foreign_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_foreign_emit_1(WamState* state);

static bool foreign_emit_ok(WamState *state, const char *pred, int arity) {
    if (strcmp(pred, "wam_c_foreign_emit/1") != 0 || arity != 1) return false;
    WamValue out = val_atom("foreign_ok");
    return wam_unify(state, &state->A[0], &out);
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_foreign_emit_1(&state);
    wam_register_foreign_predicate(&state, "wam_c_foreign_emit/1", 1, foreign_emit_ok);

    WamValue ok_args[1] = { val_unbound("Out") };
    int ok_rc = wam_run_predicate(&state, "wam_c_foreign_emit/1", ok_args, 1);
    if (ok_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_ATOM || strcmp(state.A[0].data.atom, "foreign_ok") != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamState missing;
    wam_state_init(&missing);
    setup_wam_c_foreign_emit_1(&missing);
    WamValue fail_args[1] = { val_unbound("Out") };
    int fail_rc = wam_run_predicate(&missing, "wam_c_foreign_emit/1", fail_args, 1);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        wam_free_state(&missing);
        return 20;
    }

    wam_free_state(&state);
    wam_free_state(&missing);
    return 0;
}
').

wam_c_category_ancestor_smoke_main(
'#include "wam_runtime.h"

void setup_category_ancestor_4(WamState* state);

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(atom);
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_category_ancestor_4(&state);
    wam_register_category_parent(&state, "leaf", "mid");
    wam_register_category_parent(&state, "mid", "root");
    wam_register_category_parent(&state, "leaf", "other");
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", 10);

    WamValue recursive_args[4] = {
        val_atom("leaf"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int recursive_rc = wam_run_predicate(&state, "category_ancestor/4", recursive_args, 4);
    if (recursive_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[4] = {
        val_atom("leaf"),
        val_atom("mid"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int direct_rc = wam_run_predicate(&state, "category_ancestor/4", direct_args, 4);
    if (direct_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 1) {
        wam_free_state(&state);
        return 20;
    }

    WamValue fail_args[4] = {
        val_atom("other"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "other")
    };
    int fail_rc = wam_run_predicate(&state, "category_ancestor/4", fail_args, 4);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_kernel_detector_smoke_main(
'#include "wam_runtime.h"

void setup_category_ancestor_4(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(atom);
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_category_ancestor_4(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_category_parent(&state, "leaf", "mid");
    wam_register_category_parent(&state, "mid", "root");

    WamValue args[4] = {
        val_atom("leaf"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int rc = wam_run_predicate(&state, "category_ancestor/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_fact_source_smoke_main(DataPath, MainCode) :-
    format(atom(MainCode),
'#include "wam_runtime.h"

void setup_category_ancestor_4(WamState* state);

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(atom);
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

int main(void) {
    WamState state;
    WamFactSource source;
    CategoryEdge matches[4];
    wam_state_init(&state);
    wam_fact_source_init(&source);
    setup_category_ancestor_4(&state);

    if (!wam_fact_source_load_tsv(&state, &source, "~w")) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 10;
    }

    int match_count = wam_fact_source_lookup_arg1(&source, "leaf", matches, 4);
    if (match_count != 2 ||
        strcmp(matches[0].child, "leaf") != 0 ||
        strcmp(matches[0].parent, "mid") != 0) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 20;
    }

    wam_register_category_parent_fact_source(&state, &source);
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", 10);

    WamValue args[4] = {
        val_atom("leaf"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int rc = wam_run_predicate(&state, "category_ancestor/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 30;
    }

    wam_free_state(&state);
    wam_fact_source_close(&source);
    return 0;
}
', [DataPath]).

wam_c_lmdb_fact_source_smoke_main(EnvPath, MainCode) :-
    format(atom(MainCode),
'#include "wam_runtime.h"
#include <sys/stat.h>

void setup_category_ancestor_4(WamState* state);

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(atom);
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

static int put_edge(MDB_env *env, MDB_dbi dbi, const char *child, const char *parent) {
    MDB_txn *txn = NULL;
    MDB_val key;
    MDB_val data;
    int rc = mdb_txn_begin(env, NULL, 0, &txn);
    if (rc != MDB_SUCCESS) return rc;
    key.mv_size = strlen(child);
    key.mv_data = (void *)child;
    data.mv_size = strlen(parent);
    data.mv_data = (void *)parent;
    rc = mdb_put(txn, dbi, &key, &data, 0);
    if (rc == MDB_SUCCESS) rc = mdb_txn_commit(txn);
    else mdb_txn_abort(txn);
    return rc;
}

static int seed_lmdb(const char *path) {
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = mkdir(path, 0777);
    (void)rc;
    rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) return rc;
    rc = mdb_env_set_maxdbs(env, 16);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_env_set_mapsize(env, 1048576);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_env_open(env, path, 0, 0664);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_txn_begin(env, NULL, 0, &txn);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_dbi_open(txn, NULL, MDB_CREATE | MDB_DUPSORT, &dbi);
    if (rc == MDB_SUCCESS) rc = mdb_txn_commit(txn);
    else { mdb_txn_abort(txn); mdb_env_close(env); return rc; }
    rc = put_edge(env, dbi, "leaf", "mid");
    if (rc == MDB_SUCCESS) rc = put_edge(env, dbi, "mid", "root");
    if (rc == MDB_SUCCESS) rc = put_edge(env, dbi, "leaf", "other");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
    return rc;
}

int main(void) {
    WamState state;
    WamFactSource source;
    CategoryEdge matches[4];
    if (seed_lmdb("~w") != MDB_SUCCESS) return 5;

    wam_state_init(&state);
    wam_fact_source_init(&source);
    setup_category_ancestor_4(&state);

    if (!wam_fact_source_load_lmdb(&state, &source, "~w", NULL)) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 10;
    }

    int match_count = wam_fact_source_lookup_arg1(&source, "leaf", matches, 4);
    if (match_count != 2 ||
        strcmp(matches[0].child, "leaf") != 0 ||
        strcmp(matches[0].parent, "mid") != 0) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 20;
    }

    wam_register_category_parent_fact_source(&state, &source);
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", 10);

    WamValue args[4] = {
        val_atom("leaf"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int run_rc = wam_run_predicate(&state, "category_ancestor/4", args, 4);
    if (run_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 30;
    }

    wam_free_state(&state);
    wam_fact_source_close(&source);
    return 0;
}
', [EnvPath, EnvPath]).

wam_c_streaming_foreign_smoke_main(
'#include "wam_runtime.h"

void setup_category_ancestor_4(WamState* state);

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(atom);
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

static bool has_value(WamIntResults *results, int value) {
    for (int i = 0; i < results->count; i++) {
        if (results->values[i] == value) return true;
    }
    return false;
}

int main(void) {
    WamState state;
    WamIntResults results;
    wam_state_init(&state);
    wam_int_results_init(&results);
    setup_category_ancestor_4(&state);

    wam_register_category_parent(&state, "leaf", "root");
    wam_register_category_parent(&state, "leaf", "mid");
    wam_register_category_parent(&state, "mid", "root");
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", 10);

    state.A[0] = val_atom("leaf");
    state.A[1] = val_atom("root");
    state.A[2] = val_unbound("Hops");
    state.A[3] = make_visited_singleton(&state, "leaf");

    if (!wam_collect_category_ancestor_hops(&state, &results) ||
        results.count != 2 ||
        !has_value(&results, 1) ||
        !has_value(&results, 2)) {
        wam_int_results_close(&results);
        wam_free_state(&state);
        return 10;
    }

    WamValue args[4] = {
        val_atom("leaf"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int rc = wam_run_predicate(&state, "category_ancestor/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 1) {
        wam_int_results_close(&results);
        wam_free_state(&state);
        return 20;
    }

    wam_int_results_close(&results);
    wam_free_state(&state);
    return 0;
}
').

wam_c_asan_lifecycle_smoke_main(DataPath, MainCode) :-
    format(atom(MainCode),
'#include "wam_runtime.h"

void setup_wam_c_asan_term_2(WamState* state);
void setup_category_ancestor_4(WamState* state);

static WamValue make_list_pair(WamState *state, const char *head, const char *tail) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(head);
    state->H_array[state->H++] = val_atom(tail);
    return list;
}

static WamValue make_unary_struct(WamState *state, const char *functor, const char *arg) {
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(functor);
    state->H_array[state->H++] = val_atom(arg);
    return term;
}

static bool expect_label(WamState *state, WamValue input, const char *label) {
    WamValue args[2] = { input, val_unbound("Kind") };
    int rc = wam_run_predicate(state, "wam_c_asan_term/2", args, 2);
    return rc == 0 &&
           state->P == WAM_HALT &&
           state->A[1].tag == VAL_ATOM &&
           strcmp(state->A[1].data.atom, label) == 0;
}

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    return make_list_pair(state, atom, "[]");
}

int main(void) {
    WamState state;
    WamFactSource source;
    CategoryEdge matches[4];
    wam_state_init(&state);
    wam_fact_source_init(&source);

    setup_wam_c_asan_term_2(&state);
    setup_wam_c_asan_term_2(&state);

    if (!expect_label(&state, val_atom("a"), "atom")) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 10;
    }
    if (!expect_label(&state, make_list_pair(&state, "head", "tail"), "list")) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 20;
    }
    if (!expect_label(&state, make_unary_struct(&state, "foo/1", "arg"), "struct")) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 30;
    }

    setup_category_ancestor_4(&state);
    if (!wam_fact_source_load_tsv(&state, &source, "~w")) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 40;
    }
    int match_count = wam_fact_source_lookup_arg1(&source, "leaf", matches, 4);
    if (match_count != 2) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 50;
    }
    wam_register_category_parent_fact_source(&state, &source);
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", 10);

    WamValue args[4] = {
        val_atom("leaf"),
        val_atom("root"),
        val_unbound("Hops"),
        make_visited_singleton(&state, "leaf")
    };
    int rc = wam_run_predicate(&state, "category_ancestor/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 60;
    }

    setup_wam_c_asan_term_2(&state);
    if (!expect_label(&state, val_atom("a"), "atom")) {
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 70;
    }

    wam_fact_source_close(&source);
    wam_free_state(&state);
    return 0;
}
', [DataPath]).

wam_c_lowered_fact_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_lowered_pair_2(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_lowered_pair_2(&state);
    setup_lowered_wam_c_helpers(&state);

    WamValue ok_args[2] = { val_atom("a"), val_unbound("Out") };
    int ok_rc = wam_run_predicate(&state, "wam_c_lowered_pair/2", ok_args, 2);
    if (ok_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "b") != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue ground_args[2] = { val_atom("a"), val_atom("c") };
    int ground_rc = wam_run_predicate(&state, "wam_c_lowered_pair/2", ground_args, 2);
    if (ground_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue fail_args[2] = { val_atom("z"), val_unbound("Out") };
    int fail_rc = wam_run_predicate(&state, "wam_c_lowered_pair/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_lowered_body_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_body_fact_2(WamState* state);
void setup_wam_c_body_alias_2(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_body_fact_2(&state);
    setup_wam_c_body_alias_2(&state);
    setup_lowered_wam_c_helpers(&state);

    WamValue ok_args[2] = { val_atom("a"), val_unbound("Out") };
    int ok_rc = wam_run_predicate(&state, "wam_c_body_alias/2", ok_args, 2);
    if (ok_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "b") != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue ground_args[2] = { val_atom("a"), val_atom("c") };
    int ground_rc = wam_run_predicate(&state, "wam_c_body_alias/2", ground_args, 2);
    if (ground_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue fail_args[2] = { val_atom("z"), val_unbound("Out") };
    int fail_rc = wam_run_predicate(&state, "wam_c_body_alias/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_lowered_filter_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_filter_fact_2(WamState* state);
void setup_wam_c_filter_keep_1(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_filter_fact_2(&state);
    setup_wam_c_filter_keep_1(&state);
    setup_lowered_wam_c_helpers(&state);

    WamValue ok_args[1] = { val_unbound("Out") };
    int ok_rc = wam_run_predicate(&state, "wam_c_filter_keep/1", ok_args, 1);
    if (ok_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_ATOM || strcmp(state.A[0].data.atom, "a") != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue ground_args[1] = { val_atom("c") };
    int ground_rc = wam_run_predicate(&state, "wam_c_filter_keep/1", ground_args, 1);
    if (ground_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue fail_args[1] = { val_atom("b") };
    int fail_rc = wam_run_predicate(&state, "wam_c_filter_keep/1", fail_args, 1);
    if (fail_rc != WAM_HALT) {
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

wam_c_classic_fib_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_classic_fib_2(WamState* state);

static int expect_fib(WamState *state, int n, int expected) {
    (void)state;
    WamState local;
    wam_state_init(&local);
    setup_wam_c_classic_fib_2(&local);
    WamValue args[2] = { val_int(n), val_unbound("F") };
    int rc = wam_run_predicate(&local, "wam_c_classic_fib/2", args, 2);
    WamValue *result = wam_deref_ptr(&local, &local.A[0]);
    int ok = rc == 0 &&
             local.P == WAM_HALT &&
             result->tag == VAL_INT &&
             result->data.integer == expected;
    wam_free_state(&local);
    return ok;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_classic_fib_2(&state);

    if (!expect_fib(&state, 0, 0)) {
        wam_free_state(&state);
        return 10;
    }
    if (!expect_fib(&state, 1, 1)) {
        wam_free_state(&state);
        return 20;
    }
    if (!expect_fib(&state, 6, 8)) {
        wam_free_state(&state);
        return 30;
    }

    WamValue fail_args[2] = { val_int(6), val_int(7) };
    int fail_rc = wam_run_predicate(&state, "wam_c_classic_fib/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
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
implemented_case(call_foreign, 'case INSTR_CALL_FOREIGN').
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
implemented_case(set_variable, 'case INSTR_SET_VARIABLE').
implemented_case(set_value, 'case INSTR_SET_VALUE').
implemented_case(set_constant, 'case INSTR_SET_CONSTANT').
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
    test_call_foreign_generation,
    test_category_ancestor_kernel_generation,
    test_fact_source_generation,
    test_streaming_foreign_results_generation,
    test_kernel_detector_setup_generation,
    test_kernel_detector_project_generation,
    test_lowered_fact_helper_generation,
    test_lowered_helper_planner_metadata,
    test_lowered_helper_plan_generation,
    test_lowered_body_call_helper_generation,
    test_lowered_filtered_fact_helper_generation,
    test_unsupported_instruction_fails_loudly,
    test_no_zero_instruction_fallback,
    test_list_target_pc_emission,
    test_generated_runtime_executable_smoke,
    test_cross_predicate_executable_smoke,
    test_builtin_call_executable_smoke,
    test_call_foreign_executable_smoke,
    test_category_ancestor_kernel_executable_smoke,
    test_fact_source_executable_smoke,
    test_lmdb_fact_source_executable_smoke,
    test_kernel_detector_executable_smoke,
    test_streaming_foreign_results_executable_smoke,
    test_real_prolog_builtin_executable_smoke,
    test_real_prolog_multiclause_executable_smoke,
    test_real_prolog_structure_index_executable_smoke,
    test_real_prolog_is_list_executable_smoke,
    test_real_prolog_unify_executable_smoke,
    test_real_prolog_classic_recursive_executable_smoke,
    test_lowered_fact_helper_executable_smoke,
    test_lowered_body_call_helper_executable_smoke,
    test_lowered_filtered_fact_helper_executable_smoke,
    test_asan_memory_lifecycle_executable_smoke,
    format('~n=== WAM-C Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
