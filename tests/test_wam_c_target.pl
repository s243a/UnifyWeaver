:- encoding(utf8).
% Test suite for WAM-to-C transpilation target
% Usage: swipl -g run_tests -t halt tests/test_wam_c_target.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('helpers/smoke_paths', [tmp_root/1]).
:- use_module(library(filesex), [directory_file_path/3]).
:- use_module(library(readutil), [read_file_to_string/3]).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

% Delegate to the shared smoke_paths helper so this test works on
% native Windows (no /tmp) and Termux (writable /data/data/.../tmp).
wam_c_temp_root(Root) :- tmp_root(Root).

wam_c_temp_path(Prefix, Stamp, Path) :-
    wam_c_temp_root(Root),
    format(atom(Path), '~w/~w_~w', [Root, Prefix, Stamp]).

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
        sub_string(S, _, _, _, 'resolve_predicate_hash'),
        sub_string(S, _, _, _, 'state->atom_table_size'),
        sub_string(S, _, _, _, 'free(state->atom_table)')
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

test_aggregate_instructions :-
    Test = 'WAM-C: aggregate instructions present',
    (   implemented_wam_c_cases(Cases),
        member(begin_aggregate, Cases),
        member(end_aggregate, Cases)
    ->  pass(Test)
    ;   fail_test(Test, 'missing aggregate instruction arms')
    ).

test_control_cut_jump_instructions :-
    Test = 'WAM-C: cut and jump control instructions present',
    (   implemented_wam_c_cases(Cases),
        member(get_level, Cases),
        member(cut, Cases),
        member(cut_ite, Cases),
        member(jump, Cases),
        compile_step_wam_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'wam_prune_choice_points(state, target_b)'),
        sub_string(S, _, _, _, 'state->P = target')
    ->  pass(Test)
    ;   fail_test(Test, 'missing cut/jump control instruction arms')
    ).

test_explicit_cut_uses_current_call_barrier :-
    Test = 'WAM-C: explicit cut prunes to current call barrier',
    (   compile_wam_helpers_to_c([], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'state->call_bases[state->call_base_top - 1]'),
        sub_string(S, _, _, _, 'wam_prune_choice_points(state, target_b)'),
        sub_string(S, _, _, _, 'state->call_bases[state->call_base_top] = base_b'),
        sub_string(S, _, _, _, 'state->call_base_preserve_choice[state->call_base_top] = false'),
        \+ sub_string(S, _, _, _, 'state->B = 0')
    ->  pass(Test)
    ;   fail_test(Test, 'explicit !/0 still clears all choice points instead of pruning to the current barrier')
    ).

test_control_instruction_parsing :-
    Test = 'WAM-C: cut and jump instructions parse to typed payloads',
    WamCode = 'wam_c_ctrl_parse/1:\n    get_level Y1\n    try_me_else L_else\n    cut Y1\n    cut_ite\n    jump L_done\nL_else:\n    trust_me\nL_done:\n    proceed',
    (   compile_wam_predicate_to_c(user:wam_c_ctrl_parse/1, WamCode, [], CCode),
        atom_string(CCode, S),
        sub_string(S, _, _, _, 'INSTR_GET_LEVEL'),
        sub_string(S, _, _, _, 'INSTR_CUT'),
        sub_string(S, _, _, _, 'INSTR_CUT_ITE'),
        sub_string(S, _, _, _, 'INSTR_JUMP'),
        sub_string(S, _, _, _, '.as.jump = { .target_pc = base_pc +')
    ->  pass(Test)
    ;   fail_test(Test, 'cut/jump instruction parsing missing typed payloads')
    ).

test_precise_ite_y_level_generation :-
    Test = 'WAM-C: precise if-then-else lowers to get_level/cut',
    assertz((user:wam_c_precise_codegen(X, R) :-
        (X = a -> R = then ; R = else))),
    (   compile_predicate_to_wam(user:wam_c_precise_codegen/2,
                                 [ite_use_y_level(true)],
                                 WamCode),
        sub_string(WamCode, _, _, _, 'get_level '),
        sub_string(WamCode, _, _, _, 'cut Y'),
        \+ sub_string(WamCode, _, _, _, 'cut_ite'),
        compile_wam_predicate_to_c(user:wam_c_precise_codegen/2,
                                   WamCode,
                                   [ite_use_y_level(true)],
                                   CCode),
        atom_string(CCode, S),
        sub_string(S, _, _, _, 'INSTR_GET_LEVEL'),
        sub_string(S, _, _, _, 'INSTR_CUT'),
        \+ sub_string(S, _, _, _, 'INSTR_CUT_ITE')
    ->  retractall(user:wam_c_precise_codegen(_, _)),
        pass(Test)
    ;   retractall(user:wam_c_precise_codegen(_, _)),
        fail_test(Test, 'precise ITE mode did not emit get_level/cut-only control flow')
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
        sub_string(PredS, _, _, _, 'wam_register_predicate_hash(state, "foo/1", base_pc + 0)'),
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
        sub_string(S, _, _, _, 'wam_state_category_child_range'),
        sub_string(S, _, _, _, 'wam_category_ancestor_dfs')
    ->  pass(Test)
    ;   fail_test(Test, 'category_ancestor native kernel helpers missing')
    ).

test_bidirectional_ancestor_kernel_generation :-
    Test = 'WAM-C: bidirectional_ancestor native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_bidirectional_ancestor_kernel'),
        sub_string(S, _, _, _, 'bool wam_bidirectional_ancestor_handler'),
        sub_string(S, _, _, _, 'wam_bidirectional_ancestor_dfs'),
        sub_string(S, _, _, _, 'void wam_attach_bidirectional_child_csr'),
        sub_string(S, _, _, _, 'void wam_register_category_id'),
        sub_string(S, _, _, _, 'wam_category_id_atom_index_find'),
        sub_string(S, _, _, _, 'wam_category_id_value_index_find'),
        sub_string(S, _, _, _, 'category_id_by_atom'),
        sub_string(S, _, _, _, 'WamBidirectionalDistanceMap'),
        sub_string(S, _, _, _, 'WamBidirectionalDistanceCacheEntry'),
        sub_string(S, _, _, _, 'wam_bidirectional_build_min_distances'),
        sub_string(S, _, _, _, 'wam_bidirectional_get_min_distances'),
        sub_string(S, _, _, _, 'wam_bidirectional_can_reach_root_within_budget'),
        sub_string(S, _, _, _, 'bool wam_category_min_parent_hops'),
        sub_string(S, _, _, _, 'bool wam_category_child_may_reach_root_within_budget'),
        sub_string(S, _, _, _, 'bidirectional_min_distance_cache'),
        sub_string(S, _, _, _, 'bidirectional_parent_step_cost')
    ->  pass(Test)
    ;   fail_test(Test, 'bidirectional_ancestor native kernel helpers missing')
    ).

test_transitive_closure_kernel_generation :-
    Test = 'WAM-C: transitive_closure2 native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_transitive_edge'),
        sub_string(S, _, _, _, 'void wam_register_transitive_closure_kernel'),
        sub_string(S, _, _, _, 'bool wam_transitive_closure_handler'),
        sub_string(S, _, _, _, 'wam_transitive_closure_dfs')
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_closure2 native kernel helpers missing')
    ).

test_transitive_distance_kernel_generation :-
    Test = 'WAM-C: transitive_distance3 native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_transitive_distance_kernel'),
        sub_string(S, _, _, _, 'bool wam_transitive_distance_handler'),
        sub_string(S, _, _, _, 'wam_transitive_distance_bfs')
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_distance3 native kernel helpers missing')
    ).

test_transitive_parent_distance_kernel_generation :-
    Test = 'WAM-C: transitive_parent_distance4 native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_transitive_parent_distance_kernel'),
        sub_string(S, _, _, _, 'bool wam_transitive_parent_distance_handler'),
        sub_string(S, _, _, _, 'wam_collect_transitive_parent_distance')
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_parent_distance4 native kernel helpers missing')
    ).

test_transitive_step_parent_distance_kernel_generation :-
    Test = 'WAM-C: transitive_step_parent_distance5 native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_transitive_step_parent_distance_kernel'),
        sub_string(S, _, _, _, 'bool wam_transitive_step_parent_distance_handler'),
        sub_string(S, _, _, _, 'wam_transitive_step_parent_distance_bfs')
    ->  pass(Test)
    ;   fail_test(Test, 'transitive_step_parent_distance5 native kernel helpers missing')
    ).

test_weighted_shortest_path_kernel_generation :-
    Test = 'WAM-C: weighted_shortest_path3 native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_weighted_edge'),
        sub_string(S, _, _, _, 'void wam_register_weighted_shortest_path_kernel'),
        sub_string(S, _, _, _, 'bool wam_weighted_shortest_path_handler'),
        sub_string(S, _, _, _, 'wam_weighted_shortest_path_dijkstra')
    ->  pass(Test)
    ;   fail_test(Test, 'weighted_shortest_path3 native kernel helpers missing')
    ).

test_astar_shortest_path_kernel_generation :-
    Test = 'WAM-C: astar_shortest_path4 native kernel helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_register_direct_distance_edge'),
        sub_string(S, _, _, _, 'void wam_register_astar_shortest_path_kernel'),
        sub_string(S, _, _, _, 'bool wam_astar_shortest_path_handler'),
        sub_string(S, _, _, _, 'wam_astar_shortest_path_search')
    ->  pass(Test)
    ;   fail_test(Test, 'astar_shortest_path4 native kernel helpers missing')
    ).

test_fact_source_generation :-
    Test = 'WAM-C: file FactSource helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_fact_source_init'),
        sub_string(S, _, _, _, 'bool wam_fact_source_load_tsv'),
        sub_string(S, _, _, _, 'bool wam_fact_source_load_lmdb'),
        sub_string(S, _, _, _, 'bool wam_fact_source_child_range'),
        sub_string(S, _, _, _, 'int wam_fact_source_lookup_arg1'),
        sub_string(S, _, _, _, 'bool wam_register_category_parent_fact_source'),
        sub_string(S, _, _, _, 'source->owner_state'),
        sub_string(S, _, _, _, 'memcpy(target, source->edges')
    ->  pass(Test)
    ;   fail_test(Test, 'file FactSource helpers missing')
    ).

test_reverse_csr_generation :-
    Test = 'WAM-C: reverse CSR pread helpers generated',
    (   compile_wam_runtime_to_c([], RuntimeCode),
        atom_string(RuntimeCode, S),
        sub_string(S, _, _, _, 'void wam_reverse_csr_init'),
        sub_string(S, _, _, _, 'bool wam_reverse_csr_load'),
        sub_string(S, _, _, _, 'bool wam_reverse_csr_load_pread_drop'),
        sub_string(S, _, _, _, 'bool wam_reverse_csr_load_direct_io'),
        sub_string(S, _, _, _, 'bool wam_reverse_csr_load_lmdb_offset'),
        sub_string(S, _, _, _, 'bool wam_reverse_csr_load_lmdb_offset_pread_drop'),
        sub_string(S, _, _, _, 'bool wam_reverse_csr_load_lmdb_offset_direct_io'),
        sub_string(S, _, _, _, 'int wam_reverse_csr_lookup_children'),
        sub_string(S, _, _, _, 'pread('),
        sub_string(S, _, _, _, 'O_DIRECT'),
        sub_string(S, _, _, _, 'posix_fadvise'),
        sub_string(S, _, _, _, 'wam_read_i32_le')
    ->  pass(Test)
    ;   fail_test(Test, 'reverse CSR helpers missing')
    ).

test_reverse_index_plan_none :-
    Test = 'WAM-C: reverse_index none plans no runtime child lookup',
    (   resolve_wam_c_reverse_index_plan(
            [reverse_index(none)],
            wam_c_reverse_index_plan(none, Capabilities)
        ),
        memberchk(planning(unneeded), Capabilities),
        memberchk(runtime_child_lookup(unavailable), Capabilities)
    ->  pass(Test)
    ;   fail_test(Test, 'reverse_index(none) did not plan as unavailable')
    ).

test_reverse_index_plan_csr_cost_model :-
    Test = 'WAM-C: reverse_index csr uses shared CSR cost model',
    (   resolve_wam_c_reverse_index_plan(
            [reverse_index(csr([
                index_backend(auto),
                expected_child_lookups_per_query(500),
                expected_query_count_per_artifact(100),
                sorted_array_lookup_ms_per_1000(4.418097),
                lmdb_offset_lookup_ms_per_1000(2.961144),
                sorted_array_build_seconds(0.331824),
                lmdb_offset_build_seconds(0.381317),
                lmdb_offset_bytes(2113536),
                available_memory_bytes(1073741824)
            ]))],
            wam_c_reverse_index_plan(csr(Resolved), Capabilities)
        ),
        memberchk(index_backend(lmdb_offset), Resolved),
        memberchk(io_policy(buffered_pread_drop), Resolved),
        memberchk(runtime_child_lookup(available), Capabilities),
        memberchk(runtime_api(wam_reverse_csr_lookup_children), Capabilities),
        memberchk(runtime_index_backend(lmdb_offset), Capabilities),
        memberchk(runtime_io(pread_drop), Capabilities),
        memberchk(runtime_requires(wam_c_enable_lmdb), Capabilities)
    ->  pass(Test)
    ;   fail_test(Test, 'CSR options were not normalized through the cost model')
    ).

test_reverse_index_plan_runtime_available_sorted_array :-
    Test = 'WAM-C: runtime sorted-array CSR artifact is available',
    (   resolve_wam_c_reverse_index_plan(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(sorted_array),
                io_policy(buffered_pread)
            ]))],
            wam_c_reverse_index_plan(artifact(Resolved), Capabilities)
        ),
        memberchk(phase(runtime_available), Resolved),
        memberchk(storage_kind(csr_pread_artifact), Resolved),
        memberchk(index_backend(sorted_array), Resolved),
        memberchk(runtime_child_lookup(available), Capabilities),
        memberchk(runtime_api(wam_reverse_csr_lookup_children), Capabilities),
        memberchk(runtime_io(pread), Capabilities)
    ->  pass(Test)
    ;   fail_test(Test, 'runtime sorted-array CSR artifact was not marked available')
    ).

test_reverse_index_plan_runtime_available_lmdb_offset :-
    Test = 'WAM-C: runtime lmdb-offset CSR artifact is available with LMDB',
    (   resolve_wam_c_reverse_index_plan(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(lmdb_offset),
                io_policy(buffered_pread)
            ]))],
            wam_c_reverse_index_plan(artifact(Resolved), Capabilities)
        ),
        memberchk(phase(runtime_available), Resolved),
        memberchk(storage_kind(csr_pread_artifact), Resolved),
        memberchk(index_backend(lmdb_offset), Resolved),
        memberchk(runtime_child_lookup(available), Capabilities),
        memberchk(runtime_index_backend(lmdb_offset), Capabilities),
        memberchk(runtime_requires(wam_c_enable_lmdb), Capabilities)
    ->  pass(Test)
    ;   fail_test(Test, 'runtime lmdb-offset CSR artifact was not marked available')
    ).

test_reverse_index_plan_runtime_direct_io_available :-
    Test = 'WAM-C: runtime direct_io CSR artifact is available when block size is declared',
    (   resolve_wam_c_reverse_index_plan(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(sorted_array),
                io_policy(direct_io),
                block_size_edges(1024)
            ]))],
            wam_c_reverse_index_plan(artifact(Resolved), Capabilities)
        ),
        memberchk(io_policy(direct_io), Resolved),
        memberchk(runtime_child_lookup(available), Capabilities),
        memberchk(runtime_io(direct_io), Capabilities)
    ->  pass(Test)
    ;   fail_test(Test, 'runtime direct_io CSR artifact should be marked available')
    ).

test_reverse_index_plan_runtime_buffered_pread_drop_available :-
    Test = 'WAM-C: runtime buffered_pread_drop CSR artifact is available',
    (   resolve_wam_c_reverse_index_plan(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(sorted_array),
                io_policy(buffered_pread_drop)
            ]))],
            wam_c_reverse_index_plan(artifact(Resolved), Capabilities)
        ),
        memberchk(io_policy(buffered_pread_drop), Resolved),
        memberchk(runtime_child_lookup(available), Capabilities),
        memberchk(runtime_io(pread_drop), Capabilities)
    ->  pass(Test)
    ;   fail_test(Test, 'runtime buffered_pread_drop CSR artifact should be marked available')
    ).

test_reverse_index_setup_generation :-
    Test = 'WAM-C: reverse_index artifact emits CSR setup lifecycle',
    (   generate_setup_reverse_index_c(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(sorted_array),
                io_policy(buffered_pread)
            ])),
             reverse_csr_index_path('/tmp/category_child.csr.idx'),
             reverse_csr_values_path('/tmp/category_child.csr.val'),
             category_id_map([orphan-20, child-30])],
            Code
        ),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'bool setup_wam_c_reverse_index_artifacts'),
        sub_string(S, _, _, _, 'wam_reverse_csr_load(bidirectional_child_csr, "/tmp/category_child.csr.idx", "/tmp/category_child.csr.val")'),
        sub_string(S, _, _, _, 'wam_register_category_id(state, "orphan", 20)'),
        sub_string(S, _, _, _, 'wam_register_category_id(state, "child", 30)'),
        sub_string(S, _, _, _, 'wam_attach_bidirectional_child_csr(state, bidirectional_child_csr)'),
        sub_string(S, _, _, _, 'void teardown_wam_c_reverse_index_artifacts')
    ->  pass(Test)
    ;   fail_test(Test, 'reverse_index artifact setup lifecycle missing')
    ).

test_reverse_index_setup_lmdb_offset_generation :-
    Test = 'WAM-C: reverse_index lmdb-offset setup preserves CSR load order',
    (   generate_setup_reverse_index_c(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(lmdb_offset),
                io_policy(buffered_pread)
            ])),
             reverse_csr_offset_lmdb_path('/tmp/category_child.offsets.lmdb'),
             reverse_csr_values_path('/tmp/category_child.csr.val'),
             reverse_csr_offset_lmdb_dbi(offsets)],
            Code
        ),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'wam_reverse_csr_load_lmdb_offset(bidirectional_child_csr, "/tmp/category_child.csr.val", "/tmp/category_child.offsets.lmdb", "offsets")')
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB-offset CSR setup load order changed')
    ).

test_reverse_index_setup_lmdb_offset_pread_drop_generation :-
    Test = 'WAM-C: reverse_index lmdb-offset setup emits buffered_pread_drop loader',
    (   generate_setup_reverse_index_c(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(lmdb_offset),
                io_policy(buffered_pread_drop)
            ])),
             reverse_csr_offset_lmdb_path('/tmp/category_child.offsets.lmdb'),
             reverse_csr_values_path('/tmp/category_child.csr.val'),
             reverse_csr_offset_lmdb_dbi(offsets)],
            Code
        ),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'wam_reverse_csr_load_lmdb_offset_pread_drop(bidirectional_child_csr, "/tmp/category_child.csr.val", "/tmp/category_child.offsets.lmdb", "offsets")')
    ->  pass(Test)
    ;   fail_test(Test, 'LMDB-offset buffered_pread_drop CSR setup loader missing')
    ).

test_reverse_index_setup_rejects_runtime_direct_io_without_block_size :-
    Test = 'WAM-C: reverse_index setup rejects direct_io without block size',
    (   catch((generate_setup_reverse_index_c(
                   [reverse_index(artifact([
                       storage_kind(csr_pread_artifact),
                       phase(runtime_available),
                       index_backend(sorted_array),
                       io_policy(direct_io)
                   ])),
                    reverse_csr_index_path('/tmp/category_child.csr.idx'),
                    reverse_csr_values_path('/tmp/category_child.csr.val')],
                   _Code
               ), fail),
              error(permission_error(use, csr_io_policy, direct_io), _),
              true)
    ->  pass(Test)
    ;   fail_test(Test, 'expected permission_error(use, csr_io_policy, direct_io)')
    ).

test_reverse_index_setup_direct_io_generation :-
    Test = 'WAM-C: reverse_index setup emits direct_io loader',
    (   generate_setup_reverse_index_c(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(sorted_array),
                io_policy(direct_io),
                block_size_edges(1024)
            ])),
             reverse_csr_index_path('/tmp/category_child.csr.idx'),
             reverse_csr_values_path('/tmp/category_child.csr.val')],
            Code
        ),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'wam_reverse_csr_load_direct_io(bidirectional_child_csr, "/tmp/category_child.csr.idx", "/tmp/category_child.csr.val", 1024)')
    ->  pass(Test)
    ;   fail_test(Test, 'direct_io CSR setup loader missing')
    ).

test_reverse_index_setup_buffered_pread_drop_generation :-
    Test = 'WAM-C: reverse_index setup emits buffered_pread_drop loader',
    (   generate_setup_reverse_index_c(
            [reverse_index(artifact([
                storage_kind(csr_pread_artifact),
                phase(runtime_available),
                index_backend(sorted_array),
                io_policy(buffered_pread_drop)
            ])),
             reverse_csr_index_path('/tmp/category_child.csr.idx'),
             reverse_csr_values_path('/tmp/category_child.csr.val')],
            Code
        ),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'wam_reverse_csr_load_pread_drop(bidirectional_child_csr, "/tmp/category_child.csr.idx", "/tmp/category_child.csr.val")')
    ->  pass(Test)
    ;   fail_test(Test, 'buffered_pread_drop CSR setup loader missing')
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

test_bidirectional_ancestor_setup_generation :-
    Test = 'WAM-C: bidirectional_ancestor setup preserves direction costs',
    Kernel = 'bidirectional_ancestor/5'-recursive_kernel(
        bidirectional_ancestor,
        bidirectional_ancestor/5,
        [max_depth(7), parent_step_cost(1.5), child_step_cost(4.0), cost_budget(12.0)]),
    (   generate_setup_detected_kernels_c([Kernel], SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_bidirectional_ancestor_kernel(state, "bidirectional_ancestor/5", 7, 1.5, 4.0, 12.0)')
    ->  pass(Test)
    ;   fail_test(Test, 'bidirectional_ancestor setup missing direction costs')
    ).

test_transitive_closure_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits transitive_closure2 setup',
    setup_wam_c_detector_transitive_closure,
    (   detect_kernels([user:tc_ancestor/2], Detected),
        Detected = ['tc_ancestor/2'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_transitive_closure_kernel(state, "tc_ancestor/2")')
    ->  cleanup_wam_c_detector_transitive_closure,
        pass(Test)
    ;   cleanup_wam_c_detector_transitive_closure,
        fail_test(Test, 'detected transitive_closure2 setup missing')
    ).

test_transitive_distance_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits transitive_distance3 setup',
    setup_wam_c_detector_transitive_distance,
    (   detect_kernels([user:tc_distance/3], Detected),
        Detected = ['tc_distance/3'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_transitive_distance_kernel(state, "tc_distance/3")')
    ->  cleanup_wam_c_detector_transitive_distance,
        pass(Test)
    ;   cleanup_wam_c_detector_transitive_distance,
        fail_test(Test, 'detected transitive_distance3 setup missing')
    ).

test_transitive_parent_distance_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits transitive_parent_distance4 setup',
    setup_wam_c_detector_transitive_parent_distance,
    (   detect_kernels([user:tc_parent_distance/4], Detected),
        Detected = ['tc_parent_distance/4'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_transitive_parent_distance_kernel(state, "tc_parent_distance/4")')
    ->  cleanup_wam_c_detector_transitive_parent_distance,
        pass(Test)
    ;   cleanup_wam_c_detector_transitive_parent_distance,
        fail_test(Test, 'detected transitive_parent_distance4 setup missing')
    ).

test_transitive_step_parent_distance_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits transitive_step_parent_distance5 setup',
    setup_wam_c_detector_transitive_step_parent_distance,
    (   detect_kernels([user:tc_step_parent_distance/5], Detected),
        Detected = ['tc_step_parent_distance/5'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_transitive_step_parent_distance_kernel(state, "tc_step_parent_distance/5")')
    ->  cleanup_wam_c_detector_transitive_step_parent_distance,
        pass(Test)
    ;   cleanup_wam_c_detector_transitive_step_parent_distance,
        fail_test(Test, 'detected transitive_step_parent_distance5 setup missing')
    ).

test_weighted_shortest_path_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits weighted_shortest_path3 setup',
    setup_wam_c_detector_weighted_shortest_path,
    (   detect_kernels([user:weighted_path/3], Detected),
        Detected = ['weighted_path/3'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_weighted_shortest_path_kernel(state, "weighted_path/3")')
    ->  cleanup_wam_c_detector_weighted_shortest_path,
        pass(Test)
    ;   cleanup_wam_c_detector_weighted_shortest_path,
        fail_test(Test, 'detected weighted_shortest_path3 setup missing')
    ).

test_astar_shortest_path_detector_setup_generation :-
    Test = 'WAM-C: shared kernel detector emits astar_shortest_path4 setup',
    setup_wam_c_detector_astar_shortest_path,
    (   detect_kernels([user:astar_path/4], Detected),
        Detected = ['astar_path/4'-_Kernel],
        generate_setup_detected_kernels_c(Detected, SetupCode),
        sub_atom(SetupCode, _, _, _, 'setup_detected_wam_c_kernels'),
        sub_atom(SetupCode, _, _, _, 'wam_register_astar_shortest_path_kernel(state, "astar_path/4")')
    ->  cleanup_wam_c_detector_astar_shortest_path,
        pass(Test)
    ;   cleanup_wam_c_detector_astar_shortest_path,
        fail_test(Test, 'detected astar_shortest_path4 setup missing')
    ).

test_kernel_detector_project_generation :-
    Test = 'WAM-C: project generation lowers detected kernel to foreign trampoline',
    setup_wam_c_detector_category_ancestor,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_kernel_detector_project', Stamp, ProjectDir),
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
    wam_c_temp_path('unifyweaver_wam_c_lowered_fact_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   write_wam_c_project([user:wam_c_lowered_pair/2], [lowered_helpers(true)], ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_lowered_pair_2'),
        sub_string(LibS, _, _, _, 'WamValue *cells[2];'),
        sub_string(LibS, _, _, _, 'if (!val_is_unbound(*cells[0]))'),
        sub_string(LibS, _, _, _, 'bucket = wam_hash_string(cells[0]->data.atom)'),
        sub_string(LibS, _, _, _, 'switch (bucket)'),
        sub_string(LibS, _, _, _, 'static const WamValue wam_c_lowered_wam_c_lowered_pair_2_rows[][2]'),
        sub_string(LibS, _, _, _, 'wam_c_lowered_wam_c_lowered_pair_2_scan_rows'),
        sub_string(LibS, _, _, _, '.data.atom = "a"'),
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
    assertz((user:wam_c_plan_alias(X, Y) :- user:wam_c_plan_fact(X, Y))),
    (   plan_wam_c_lowered_helpers([user:wam_c_plan_fact/2,
                                     user:wam_c_plan_alias/2,
                                     user:wam_c_plan_rule/1,
                                     user:category_ancestor/4],
                                    [lowered_helpers(true)],
                                    ['category_ancestor/4'],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_plan_fact/2', _, lowered, fact_only([[a,b]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_plan_alias/2', _, lowered, body_call('wam_c_plan_fact/2', 2)), Plans),
        member(wam_c_lowered_helper_plan('wam_c_plan_rule/1', _, lowered, body_call_projected('wam_c_plan_fact/2', 2, _)), Plans),
        member(wam_c_lowered_helper_plan('category_ancestor/4', _, interpreted, detected_kernel), Plans)
    ->  pass(Test)
    ;   fail_test(Test, 'planner did not classify lowered/rejected/interpreted predicates')
    ),
    retractall(user:wam_c_plan_fact(_, _)),
    retractall(user:wam_c_plan_alias(_, _)),
    retractall(user:wam_c_plan_rule(_)).

test_lowered_helper_plan_generation :-
    Test = 'WAM-C: generated project reports lowered helper plan',
    assertz(user:wam_c_plan_emit_fact(a, b)),
    assertz((user:wam_c_plan_emit_alias(X, Y) :- user:wam_c_plan_emit_fact(X, Y))),
    assertz((user:wam_c_plan_emit_rule(X) :- user:wam_c_plan_emit_fact(X, _))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_plan_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   write_wam_c_project([user:wam_c_plan_emit_fact/2,
                             user:wam_c_plan_emit_alias/2,
                             user:wam_c_plan_emit_rule/1],
                            [lowered_helpers(true), report_lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// WAM-C lowered helper plan'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_plan_emit_fact/2: fact_only'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_plan_emit_alias/2: body_call'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_plan_emit_rule/1: body_call_projected'),
        sub_string(LibS, _, _, _, 'INSTR_CALL_FOREIGN'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_plan_emit_alias_2'),
        sub_string(LibS, _, _, _, 'setup_wam_c_plan_emit_rule_1')
    ->  pass(Test)
    ;   fail_test(Test, 'generated project did not include lowered helper plan metadata')
    ),
    retractall(user:wam_c_plan_emit_fact(_, _)),
    retractall(user:wam_c_plan_emit_alias(_, _)),
    retractall(user:wam_c_plan_emit_rule(_)).

test_lowered_body_call_helper_generation :-
    Test = 'WAM-C: deterministic body-call predicates can lower to native helper',
    assertz(user:wam_c_body_fact(a, b)),
    assertz((user:wam_c_body_alias(X, Y) :- user:wam_c_body_fact(X, Y))),
    assertz((user:wam_c_body_projected(X, Y) :- user:wam_c_body_fact(Y, X))),
    assertz((user:wam_c_body_ignored_output(X) :- user:wam_c_body_fact(X, _Ignored))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_body_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_body_fact/2,
                                     user:wam_c_body_alias/2,
                                     user:wam_c_body_projected/2,
                                     user:wam_c_body_ignored_output/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_fact/2', _, lowered, fact_only([[a,b]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_alias/2', _, lowered, body_call('wam_c_body_fact/2', 2)), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projected/2', _, lowered, body_call_projected('wam_c_body_fact/2', 2, _)), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_ignored_output/1', _, lowered, body_call_projected('wam_c_body_fact/2', 2, _)), Plans),
        write_wam_c_project([user:wam_c_body_fact/2,
                             user:wam_c_body_alias/2,
                             user:wam_c_body_projected/2,
                             user:wam_c_body_ignored_output/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - lowered wam_c_body_alias/2: body_call'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_body_projected/2: body_call_projected'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_body_alias_2'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_body_projected_2'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_body_ignored_output_1'),
        sub_string(LibS, _, _, _, 'state->A[1] = val_unbound("_");'),
        sub_string(LibS, _, _, _, 'return wam_c_lowered_wam_c_body_fact_2(state, "wam_c_body_fact/2", 2);'),
        sub_string(LibS, _, _, _, 'wam_c_lowered_wam_c_body_fact_2(state, "wam_c_body_fact/2", 2);'),
        sub_string(LibS, _, _, _, '.pred = "wam_c_body_alias/2"')
    ->  pass(Test)
    ;   fail_test(Test, 'lowered body-call helper was not emitted')
    ),
    retractall(user:wam_c_body_fact(_, _)),
    retractall(user:wam_c_body_alias(_, _)),
    retractall(user:wam_c_body_projected(_, _)),
    retractall(user:wam_c_body_ignored_output(_)).

test_lowered_body_call_rejection_metadata :-
    Test = 'WAM-C: lowered helper planner explains unsupported body-call shapes',
    assertz(user:wam_c_body_reject_fact(a, b)),
    assertz((user:wam_c_body_reject_missing(X, Y) :-
                 user:wam_c_body_reject_fact(X, Y))),
    assertz((user:wam_c_body_reject_rule_callee(X, Y) :-
                 user:wam_c_body_reject_fact(X, Y))),
    assertz((user:wam_c_body_reject_rule_alias(X, Y) :-
                 user:wam_c_body_reject_rule_callee(X, Y))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_body_reject_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_body_reject_missing/2,
                                     user:wam_c_body_reject_rule_callee/2,
                                     user:wam_c_body_reject_rule_alias/2],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_reject_missing/2', _, rejected, body_call_callee_not_available), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_reject_rule_callee/2', _, rejected, body_call_callee_not_available), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_reject_rule_alias/2', _, rejected, body_call_callee_not_lowerable), Plans),
        write_wam_c_project([user:wam_c_body_reject_missing/2,
                             user:wam_c_body_reject_rule_callee/2,
                             user:wam_c_body_reject_rule_alias/2],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - rejected wam_c_body_reject_missing/2: body_call_callee_not_available'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_body_reject_rule_callee/2: body_call_callee_not_available'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_body_reject_rule_alias/2: body_call_callee_not_lowerable')
    ->  pass(Test)
    ;   fail_test(Test, 'planner did not report explicit body-call rejection reasons')
    ),
    retractall(user:wam_c_body_reject_fact(_, _)),
    retractall(user:wam_c_body_reject_missing(_, _)),
    retractall(user:wam_c_body_reject_rule_callee(_, _)),
    retractall(user:wam_c_body_reject_rule_alias(_, _)).

test_lowered_body_call_projection_rejection_metadata :-
    Test = 'WAM-C: lowered helper planner explains unsupported projection shapes',
    assertz(user:wam_c_body_projection_fact1(a)),
    assertz(user:wam_c_body_projection_fact2(a, a)),
    assertz(user:wam_c_body_projection_fact3(a, b, b)),
    assertz((user:wam_c_body_projection_omit(X, _Ignored) :-
                 user:wam_c_body_projection_fact1(X))),
    assertz((user:wam_c_body_projection_repeat(X) :-
                 user:wam_c_body_projection_fact2(X, X))),
    assertz((user:wam_c_body_projection_repeated_local(X) :-
                 user:wam_c_body_projection_fact3(X, Y, Y))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_body_projection_reject_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_body_projection_fact1/1,
                                     user:wam_c_body_projection_fact2/2,
                                     user:wam_c_body_projection_fact3/3,
                                     user:wam_c_body_projection_omit/2,
                                     user:wam_c_body_projection_repeat/1,
                                     user:wam_c_body_projection_repeated_local/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projection_fact1/1', _, lowered, fact_only([[a]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projection_fact2/2', _, lowered, fact_only([[a,a]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projection_fact3/3', _, lowered, fact_only([[a,b,b]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projection_omit/2', _, rejected, body_call_projection_omits_head_variable), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projection_repeat/1', _, lowered, filtered_fact('wam_c_body_projection_fact2/2', [[a]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_body_projection_repeated_local/1', _, rejected, body_call_projection_repeats_callee_local_variable), Plans),
        write_wam_c_project([user:wam_c_body_projection_fact1/1,
                             user:wam_c_body_projection_fact2/2,
                             user:wam_c_body_projection_fact3/3,
                             user:wam_c_body_projection_omit/2,
                             user:wam_c_body_projection_repeat/1,
                             user:wam_c_body_projection_repeated_local/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - rejected wam_c_body_projection_omit/2: body_call_projection_omits_head_variable'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_body_projection_repeat/1: filtered_fact'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_body_projection_repeated_local/1: body_call_projection_repeats_callee_local_variable')
    ->  pass(Test)
    ;   fail_test(Test, 'planner did not report explicit projection rejection reasons')
    ),
    retractall(user:wam_c_body_projection_fact1(_)),
    retractall(user:wam_c_body_projection_fact2(_, _)),
    retractall(user:wam_c_body_projection_fact3(_, _, _)),
    retractall(user:wam_c_body_projection_omit(_, _)),
    retractall(user:wam_c_body_projection_repeat(_)),
    retractall(user:wam_c_body_projection_repeated_local(_)).

test_lowered_filtered_fact_helper_generation :-
    Test = 'WAM-C: guarded fact predicates can lower to filtered native helper',
    assertz(user:wam_c_filter_fact(a, keep)),
    assertz(user:wam_c_filter_fact(b, drop)),
    assertz(user:wam_c_filter_fact(c, keep)),
    assertz((user:wam_c_filter_keep(X) :- user:wam_c_filter_fact(X, keep))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_filter_project', Stamp, ProjectDir),
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
        sub_string(LibS, _, _, _, 'if (!val_is_unbound(*cells[0]))'),
        sub_string(LibS, _, _, _, 'wam_c_lowered_wam_c_filter_keep_1_scan_rows'),
        sub_string(LibS, _, _, _, '.data.atom = "a"'),
        sub_string(LibS, _, _, _, '.data.atom = "c"'),
        sub_string(LibS, _, _, _, '.pred = "wam_c_filter_keep/1"')
    ->  pass(Test)
    ;   fail_test(Test, 'lowered filtered fact helper was not emitted')
    ),
    retractall(user:wam_c_filter_fact(_, _)),
    retractall(user:wam_c_filter_keep(_)).

test_lowered_comparison_filter_helper_generation :-
    Test = 'WAM-C: comparison-guarded fact predicates can lower to native helper',
    assertz(user:wam_c_filter_score(a, 1)),
    assertz(user:wam_c_filter_score(b, 2)),
    assertz(user:wam_c_filter_score(c, 3)),
    assertz((user:wam_c_filter_small(X) :-
                 user:wam_c_filter_score(X, N),
                 N =< 2)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_comparison_filter_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_filter_score/2,
                                     user:wam_c_filter_small/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_score/2', _, lowered, fact_only([[a,1],[b,2],[c,3]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_small/1', _, lowered, comparison_filtered_fact('wam_c_filter_score/2', [[a],[b]])), Plans),
        write_wam_c_project([user:wam_c_filter_score/2,
                             user:wam_c_filter_small/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - lowered wam_c_filter_small/1: comparison_filtered_fact'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_filter_small_1'),
        sub_string(LibS, _, _, _, '.data.atom = "a"'),
        sub_string(LibS, _, _, _, '.data.atom = "b"'),
        sub_string(LibS, _, _, _, '.pred = "wam_c_filter_small/1"')
    ->  pass(Test)
    ;   fail_test(Test, 'lowered comparison-filter helper was not emitted')
    ),
    retractall(user:wam_c_filter_score(_, _)),
    retractall(user:wam_c_filter_small(_)).

test_lowered_repeated_variable_filter_generation :-
    Test = 'WAM-C: repeated-variable filtered helpers preserve row constraints',
    assertz(user:wam_c_repeat_edge(a, a, keep)),
    assertz(user:wam_c_repeat_edge(a, b, keep)),
    assertz(user:wam_c_repeat_edge(c, c, drop)),
    assertz(user:wam_c_repeat_score(a, a, 1)),
    assertz(user:wam_c_repeat_score(a, b, 1)),
    assertz(user:wam_c_repeat_score(c, c, 3)),
    assertz((user:wam_c_repeat_keep(X) :- user:wam_c_repeat_edge(X, X, keep))),
    assertz((user:wam_c_repeat_small(X) :-
                 user:wam_c_repeat_score(X, X, Score),
                 Score =< 2)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_repeat_filter_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_repeat_edge/3,
                                     user:wam_c_repeat_score/3,
                                     user:wam_c_repeat_keep/1,
                                     user:wam_c_repeat_small/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_repeat_edge/3', _, lowered, fact_only([[a,a,keep],[a,b,keep],[c,c,drop]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_repeat_score/3', _, lowered, fact_only([[a,a,1],[a,b,1],[c,c,3]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_repeat_keep/1', _, lowered, filtered_fact('wam_c_repeat_edge/3', [[a]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_repeat_small/1', _, lowered, comparison_filtered_fact('wam_c_repeat_score/3', [[a]])), Plans),
        write_wam_c_project([user:wam_c_repeat_edge/3,
                             user:wam_c_repeat_score/3,
                             user:wam_c_repeat_keep/1,
                             user:wam_c_repeat_small/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - lowered wam_c_repeat_keep/1: filtered_fact'),
        sub_string(LibS, _, _, _, '// - lowered wam_c_repeat_small/1: comparison_filtered_fact'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_repeat_keep_1'),
        sub_string(LibS, _, _, _, 'static bool wam_c_lowered_wam_c_repeat_small_1'),
        sub_string(LibS, _, _, _, '.data.atom = "a"')
    ->  pass(Test)
    ;   fail_test(Test, 'repeated-variable filter helpers did not preserve row constraints')
    ),
    retractall(user:wam_c_repeat_edge(_, _, _)),
    retractall(user:wam_c_repeat_score(_, _, _)),
    retractall(user:wam_c_repeat_keep(_)),
    retractall(user:wam_c_repeat_small(_)).

test_lowered_filter_rejection_metadata :-
    Test = 'WAM-C: lowered helper planner explains unsupported filter shapes',
    assertz(user:wam_c_filter_reject_fact(a, 1, keep)),
    assertz(user:wam_c_filter_reject_fact(b, 2, drop)),
    assertz((user:wam_c_filter_reject_non_constant(X) :-
                 user:wam_c_filter_reject_fact(X, _IgnoredN, keep))),
    assertz((user:wam_c_filter_reject_no_match(X) :-
                 user:wam_c_filter_reject_fact(X, 999, keep))),
    assertz((user:wam_c_filter_reject_comparison(X) :- X > 1)),
    assertz((user:wam_c_filter_reject_builtin(X) :- atom(X))),
    assertz((user:wam_c_filter_reject_multi_goal(X) :-
                 user:wam_c_filter_reject_fact(X, _N, keep),
                 X = a,
                 atom(X))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_filter_reject_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_filter_reject_fact/3,
                                     user:wam_c_filter_reject_non_constant/1,
                                     user:wam_c_filter_reject_no_match/1,
                                     user:wam_c_filter_reject_comparison/1,
                                     user:wam_c_filter_reject_builtin/1,
                                     user:wam_c_filter_reject_multi_goal/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_reject_fact/3', _, lowered, fact_only([[a,1,keep],[b,2,drop]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_reject_non_constant/1', _, rejected, non_constant_filter_argument), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_reject_no_match/1', _, rejected, no_matching_filter_rows), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_reject_comparison/1', _, rejected, unsupported_comparison_guard), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_reject_builtin/1', _, rejected, unsupported_filter_callee), Plans),
        member(wam_c_lowered_helper_plan('wam_c_filter_reject_multi_goal/1', _, rejected, multi_goal_body), Plans),
        write_wam_c_project([user:wam_c_filter_reject_fact/3,
                             user:wam_c_filter_reject_non_constant/1,
                             user:wam_c_filter_reject_no_match/1,
                             user:wam_c_filter_reject_comparison/1,
                             user:wam_c_filter_reject_builtin/1,
                             user:wam_c_filter_reject_multi_goal/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - rejected wam_c_filter_reject_non_constant/1: non_constant_filter_argument'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_filter_reject_no_match/1: no_matching_filter_rows'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_filter_reject_comparison/1: unsupported_comparison_guard'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_filter_reject_builtin/1: unsupported_filter_callee'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_filter_reject_multi_goal/1: multi_goal_body')
    ->  pass(Test)
    ;   fail_test(Test, 'planner did not report explicit filter rejection reasons')
    ),
    retractall(user:wam_c_filter_reject_fact(_, _, _)),
    retractall(user:wam_c_filter_reject_non_constant(_)),
    retractall(user:wam_c_filter_reject_no_match(_)),
    retractall(user:wam_c_filter_reject_comparison(_)),
    retractall(user:wam_c_filter_reject_builtin(_)),
    retractall(user:wam_c_filter_reject_multi_goal(_)).

test_lowered_comparison_filter_rejection_metadata :-
    Test = 'WAM-C: lowered helper planner explains unsupported comparison filters',
    assertz(user:wam_c_cmp_reject_score(a, 1)),
    assertz(user:wam_c_cmp_reject_score(b, 2)),
    assertz(user:wam_c_cmp_reject_label(a, low)),
    assertz((user:wam_c_cmp_reject_unbound(X) :-
                 user:wam_c_cmp_reject_score(X, _N),
                 _Missing =< 2)),
    assertz((user:wam_c_cmp_reject_expr(X) :-
                 user:wam_c_cmp_reject_score(X, N),
                 N + 1 =< 3)),
    assertz((user:wam_c_cmp_reject_non_integer(X) :-
                 user:wam_c_cmp_reject_label(X, Label),
                 Label > 1)),
    assertz((user:wam_c_cmp_reject_no_match(X) :-
                 user:wam_c_cmp_reject_score(X, N),
                 N < 0)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_comparison_reject_project', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    (   plan_wam_c_lowered_helpers([user:wam_c_cmp_reject_score/2,
                                     user:wam_c_cmp_reject_label/2,
                                     user:wam_c_cmp_reject_unbound/1,
                                     user:wam_c_cmp_reject_expr/1,
                                     user:wam_c_cmp_reject_non_integer/1,
                                     user:wam_c_cmp_reject_no_match/1],
                                    [lowered_helpers(true)],
                                    [],
                                    Plans),
        member(wam_c_lowered_helper_plan('wam_c_cmp_reject_score/2', _, lowered, fact_only([[a,1],[b,2]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_cmp_reject_label/2', _, lowered, fact_only([[a,low]])), Plans),
        member(wam_c_lowered_helper_plan('wam_c_cmp_reject_unbound/1', _, rejected, comparison_guard_unbound_variable), Plans),
        member(wam_c_lowered_helper_plan('wam_c_cmp_reject_expr/1', _, rejected, unsupported_comparison_expression), Plans),
        member(wam_c_lowered_helper_plan('wam_c_cmp_reject_non_integer/1', _, rejected, non_integer_comparison_rows), Plans),
        member(wam_c_lowered_helper_plan('wam_c_cmp_reject_no_match/1', _, rejected, no_matching_comparison_rows), Plans),
        write_wam_c_project([user:wam_c_cmp_reject_score/2,
                             user:wam_c_cmp_reject_label/2,
                             user:wam_c_cmp_reject_unbound/1,
                             user:wam_c_cmp_reject_expr/1,
                             user:wam_c_cmp_reject_non_integer/1,
                             user:wam_c_cmp_reject_no_match/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        read_file_to_string(LibPath, LibS, []),
        sub_string(LibS, _, _, _, '// - rejected wam_c_cmp_reject_unbound/1: comparison_guard_unbound_variable'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_cmp_reject_expr/1: unsupported_comparison_expression'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_cmp_reject_non_integer/1: non_integer_comparison_rows'),
        sub_string(LibS, _, _, _, '// - rejected wam_c_cmp_reject_no_match/1: no_matching_comparison_rows')
    ->  pass(Test)
    ;   fail_test(Test, 'planner did not report explicit comparison-filter rejection reasons')
    ),
    retractall(user:wam_c_cmp_reject_score(_, _)),
    retractall(user:wam_c_cmp_reject_label(_, _)),
    retractall(user:wam_c_cmp_reject_unbound(_)),
    retractall(user:wam_c_cmp_reject_expr(_)),
    retractall(user:wam_c_cmp_reject_non_integer(_)),
    retractall(user:wam_c_cmp_reject_no_match(_)).

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
        sub_string(S, _, _, _, '.list_target_pc = base_pc + 1')
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

test_multi_predicate_setup_executable_smoke :-
    Test = 'WAM-C: multi-predicate setup executable smoke',
    (   gcc_available
    ->  (   run_multi_predicate_setup_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'multi-predicate setup executable failed')
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

test_bidirectional_ancestor_kernel_executable_smoke :-
    Test = 'WAM-C: bidirectional_ancestor native kernel executable smoke',
    (   gcc_available
    ->  (   run_bidirectional_ancestor_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'bidirectional_ancestor native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_bidirectional_ancestor_csr_child_lookup_executable_smoke :-
    Test = 'WAM-C: bidirectional_ancestor uses reverse CSR child lookup',
    (   gcc_available
    ->  (   run_bidirectional_ancestor_csr_child_lookup_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'bidirectional_ancestor reverse CSR child lookup failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_reverse_index_setup_executable_smoke :-
    Test = 'WAM-C: generated reverse_index CSR setup executable smoke',
    (   gcc_available
    ->  (   run_reverse_index_setup_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'generated reverse_index CSR setup executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_closure_kernel_executable_smoke :-
    Test = 'WAM-C: transitive_closure2 native kernel executable smoke',
    (   gcc_available
    ->  (   run_transitive_closure_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'transitive_closure2 native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_distance_kernel_executable_smoke :-
    Test = 'WAM-C: transitive_distance3 native kernel executable smoke',
    (   gcc_available
    ->  (   run_transitive_distance_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'transitive_distance3 native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_parent_distance_kernel_executable_smoke :-
    Test = 'WAM-C: transitive_parent_distance4 native kernel executable smoke',
    (   gcc_available
    ->  (   run_transitive_parent_distance_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'transitive_parent_distance4 native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_step_parent_distance_kernel_executable_smoke :-
    Test = 'WAM-C: transitive_step_parent_distance5 native kernel executable smoke',
    (   gcc_available
    ->  (   run_transitive_step_parent_distance_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'transitive_step_parent_distance5 native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_weighted_shortest_path_kernel_executable_smoke :-
    Test = 'WAM-C: weighted_shortest_path3 native kernel executable smoke',
    (   gcc_available
    ->  (   run_weighted_shortest_path_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'weighted_shortest_path3 native kernel executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_astar_shortest_path_kernel_executable_smoke :-
    Test = 'WAM-C: astar_shortest_path4 native kernel executable smoke',
    (   gcc_available
    ->  (   run_astar_shortest_path_kernel_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'astar_shortest_path4 native kernel executable failed')
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

test_reverse_csr_executable_smoke :-
    Test = 'WAM-C: reverse CSR executable smoke',
    (   gcc_available
    ->  (   run_reverse_csr_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'reverse CSR executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_reverse_csr_lmdb_offset_executable_smoke :-
    Test = 'WAM-C: reverse CSR LMDB offset executable smoke',
    (   gcc_available,
        lmdb_available
    ->  (   run_reverse_csr_lmdb_offset_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'reverse CSR LMDB offset executable failed')
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

test_transitive_closure_detector_executable_smoke :-
    Test = 'WAM-C: detected transitive_closure2 executable smoke',
    (   gcc_available
    ->  (   run_transitive_closure_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected transitive_closure2 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_distance_detector_executable_smoke :-
    Test = 'WAM-C: detected transitive_distance3 executable smoke',
    (   gcc_available
    ->  (   run_transitive_distance_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected transitive_distance3 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_parent_distance_detector_executable_smoke :-
    Test = 'WAM-C: detected transitive_parent_distance4 executable smoke',
    (   gcc_available
    ->  (   run_transitive_parent_distance_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected transitive_parent_distance4 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_transitive_step_parent_distance_detector_executable_smoke :-
    Test = 'WAM-C: detected transitive_step_parent_distance5 executable smoke',
    (   gcc_available
    ->  (   run_transitive_step_parent_distance_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected transitive_step_parent_distance5 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_weighted_shortest_path_detector_executable_smoke :-
    Test = 'WAM-C: detected weighted_shortest_path3 executable smoke',
    (   gcc_available
    ->  (   run_weighted_shortest_path_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected weighted_shortest_path3 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_astar_shortest_path_detector_executable_smoke :-
    Test = 'WAM-C: detected astar_shortest_path4 executable smoke',
    (   gcc_available
    ->  (   run_astar_shortest_path_detector_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'detected astar_shortest_path4 executable failed')
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

test_real_prolog_term_builtin_executable_smoke :-
    Test = 'WAM-C: real Prolog term builtin executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_term_builtin_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog term builtin executable failed')
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

test_real_prolog_control_executable_smoke :-
    Test = 'WAM-C: real Prolog negation and if-then-else executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_control_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog control executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_precise_ite_executable_smoke :-
    Test = 'WAM-C: real Prolog precise if-then-else executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_precise_ite_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog precise if-then-else executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_explicit_cut_executable_smoke :-
    Test = 'WAM-C: real Prolog explicit cut executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_explicit_cut_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog explicit cut executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_forall_executable_smoke :-
    Test = 'WAM-C: real Prolog forall/2 executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_forall_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog forall/2 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_findall_executable_smoke :-
    Test = 'WAM-C: real Prolog findall/3 executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_findall_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog findall/3 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_bagof_setof_executable_smoke :-
    Test = 'WAM-C: real Prolog bagof/3 and setof/3 executable smoke',
    (   gcc_available
    ->  (   run_real_prolog_bagof_setof_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog bagof/3 and setof/3 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_bagof_setof_witness_executable_smoke :-
    Test = 'WAM-C: real Prolog grouped bagof/3 and setof/3 witness smoke',
    (   gcc_available
    ->  (   run_real_prolog_bagof_setof_witness_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog grouped bagof/3 and setof/3 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_bagof_setof_existential_executable_smoke :-
    Test = 'WAM-C: real Prolog bagof/3 and setof/3 existential smoke',
    (   gcc_available
    ->  (   run_real_prolog_bagof_setof_existential_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog existential bagof/3 and setof/3 executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_bagof_setof_unbound_witness_groups_smoke :-
    Test = 'WAM-C: real Prolog bagof/3 and setof/3 unbound witness groups smoke',
    (   gcc_available
    ->  (   run_real_prolog_bagof_setof_unbound_witness_groups_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'real Prolog unbound witness groups executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_real_prolog_bagof_setof_meta_call_smoke :-
    Test = 'WAM-C: runtime bagof/3 and setof/3 meta-call smoke',
    (   gcc_available
    ->  (   run_real_prolog_bagof_setof_meta_call_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'runtime bagof/3 and setof/3 meta-call executable failed')
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

test_lowered_comparison_filter_helper_executable_smoke :-
    Test = 'WAM-C: lowered comparison-filter helper executable smoke',
    (   gcc_available
    ->  (   run_lowered_comparison_filter_helper_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'lowered comparison-filter helper executable failed')
        )
    ;   format('[PASS] ~w (gcc unavailable; skipped executable smoke)~n', [Test])
    ).

test_lowered_repeated_variable_filter_executable_smoke :-
    Test = 'WAM-C: lowered repeated-variable filter executable smoke',
    (   gcc_available
    ->  (   run_lowered_repeated_variable_filter_executable_smoke
        ->  pass(Test)
        ;   fail_test(Test, 'lowered repeated-variable filter executable failed')
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
    wam_c_temp_path('unifyweaver_wam_c_asan_probe', Stamp, TmpBase),
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
    wam_c_temp_path('unifyweaver_wam_c_exec_smoke', Stamp, TmpBase),
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
    wam_c_temp_path('unifyweaver_wam_c_cross_exec_smoke', Stamp, TmpBase),
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

run_multi_predicate_setup_executable_smoke :-
    FirstWamCode = 'wam_c_multi_first/1:\n    get_constant a, A1\n    proceed',
    SecondWamCode = 'wam_c_multi_second/1:\n    get_constant b, A1\n    proceed',
    compile_wam_predicate_to_c(user:wam_c_multi_first/1, FirstWamCode, [], FirstPredCode),
    compile_wam_predicate_to_c(user:wam_c_multi_second/1, SecondWamCode, [], SecondPredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_multi_setup_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w~n~n~w', [FirstPredCode, SecondPredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_multi_setup_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_builtin_call_executable_smoke :-
    WamCode = 'wam_c_builtin_atom/1:\n    builtin_call atom/1, 1\n    proceed\nwam_c_builtin_is/2:\n    builtin_call is/2, 2\n    proceed\nwam_c_builtin_functor/3:\n    builtin_call functor/3, 3\n    proceed\nwam_c_builtin_arg/3:\n    builtin_call arg/3, 3\n    proceed\nwam_c_builtin_atom_concat/3:\n    builtin_call atom_concat/3, 3\n    proceed',
    compile_wam_predicate_to_c(user:wam_c_builtin_atom/1, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_builtin_smoke', Stamp, TmpBase),
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
    wam_c_temp_path('unifyweaver_wam_c_foreign_smoke', Stamp, TmpBase),
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
    wam_c_temp_path('unifyweaver_wam_c_category_ancestor_smoke', Stamp, TmpBase),
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

run_bidirectional_ancestor_kernel_executable_smoke :-
    WamCode = 'bidirectional_ancestor/5:\n    call_foreign bidirectional_ancestor/5, 5\n    proceed',
    compile_wam_predicate_to_c(user:bidirectional_ancestor/5, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_bidirectional_ancestor_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_bidirectional_ancestor_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_bidirectional_ancestor_csr_child_lookup_executable_smoke :-
    WamCode = 'bidirectional_ancestor/5:\n    call_foreign bidirectional_ancestor/5, 5\n    proceed',
    compile_wam_predicate_to_c(user:bidirectional_ancestor/5, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_bidirectional_ancestor_csr_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(IndexPath), '~w.csr.idx', [TmpBase]),
    format(atom(ValuesPath), '~w.csr.val', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    write_binary_file(IndexPath, [
        20,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0,0,
        40,0,0,0, 1,0,0,0,0,0,0,0, 1,0,0,0
    ]),
    write_binary_file(ValuesPath, [
        30,0,0,0,
        30,0,0,0
    ]),
    wam_c_bidirectional_ancestor_csr_smoke_main(IndexPath, ValuesPath, MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_reverse_index_setup_executable_smoke :-
    WamCode = 'bidirectional_ancestor/5:\n    call_foreign bidirectional_ancestor/5, 5\n    proceed',
    compile_wam_predicate_to_c(user:bidirectional_ancestor/5, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_reverse_index_setup_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(IndexPath), '~w.csr.idx', [TmpBase]),
    format(atom(ValuesPath), '~w.csr.val', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    generate_setup_reverse_index_c(
        [reverse_index(artifact([
            storage_kind(csr_pread_artifact),
            phase(runtime_available),
            index_backend(sorted_array),
            io_policy(buffered_pread)
        ])),
         reverse_csr_index_path(IndexPath),
         reverse_csr_values_path(ValuesPath),
         category_id_map([orphan-20, child-30, root-40])],
        SetupCode
    ),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w~n~n~w',
           [SetupCode, PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    write_binary_file(IndexPath, [
        20,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0,0,
        40,0,0,0, 1,0,0,0,0,0,0,0, 1,0,0,0
    ]),
    write_binary_file(ValuesPath, [
        30,0,0,0,
        30,0,0,0
    ]),
    wam_c_reverse_index_setup_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_transitive_closure_kernel_executable_smoke :-
    WamCode = 'tc_ancestor/2:\n    call_foreign tc_ancestor/2, 2\n    proceed',
    compile_wam_predicate_to_c(user:tc_ancestor/2, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_transitive_closure_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_transitive_closure_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_transitive_distance_kernel_executable_smoke :-
    WamCode = 'tc_distance/3:\n    call_foreign tc_distance/3, 3\n    proceed',
    compile_wam_predicate_to_c(user:tc_distance/3, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_transitive_distance_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_transitive_distance_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_transitive_parent_distance_kernel_executable_smoke :-
    WamCode = 'tc_parent_distance/4:\n    call_foreign tc_parent_distance/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:tc_parent_distance/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_parent_distance_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_transitive_parent_distance_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_transitive_step_parent_distance_kernel_executable_smoke :-
    WamCode = 'tc_step_parent_distance/5:\n    call_foreign tc_step_parent_distance/5, 5\n    proceed',
    compile_wam_predicate_to_c(user:tc_step_parent_distance/5, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_step_parent_distance_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_transitive_step_parent_distance_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_weighted_shortest_path_kernel_executable_smoke :-
    WamCode = 'weighted_path/3:\n    call_foreign weighted_path/3, 3\n    proceed',
    compile_wam_predicate_to_c(user:weighted_path/3, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_weighted_shortest_path_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_weighted_shortest_path_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_astar_shortest_path_kernel_executable_smoke :-
    WamCode = 'astar_path/4:\n    call_foreign astar_path/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:astar_path/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_astar_shortest_path_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    wam_c_astar_shortest_path_smoke_main(MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_fact_source_executable_smoke :-
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_fact_source_smoke', Stamp, TmpBase),
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
    wam_c_temp_path('unifyweaver_wam_c_lmdb_fact_source_smoke', Stamp, TmpBase),
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

run_reverse_csr_executable_smoke :-
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_reverse_csr_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(IndexPath), '~w.csr.idx', [TmpBase]),
    format(atom(ValuesPath), '~w.csr.val', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    write_text_file(PredPath, '#include "wam_runtime.h"\n'),
    write_binary_file(IndexPath, [
        20,0,0,0, 0,0,0,0,0,0,0,0, 3,0,0,0,
        30,0,0,0, 3,0,0,0,0,0,0,0, 1,0,0,0
    ]),
    write_binary_file(ValuesPath, [
        10,0,0,0,
        12,0,0,0,
        13,0,0,0,
        11,0,0,0
    ]),
    wam_c_reverse_csr_smoke_main(IndexPath, ValuesPath, MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_reverse_csr_lmdb_offset_executable_smoke :-
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_reverse_csr_lmdb_offset_smoke', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(MainPath), '~w_main.c', [TmpBase]),
    format(atom(ValuesPath), '~w.csr.val', [TmpBase]),
    format(atom(OffsetEnvPath), '~w.offsets.lmdb', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    write_text_file(PredPath, '#include "wam_runtime.h"\n'),
    write_binary_file(ValuesPath, [
        10,0,0,0,
        12,0,0,0,
        13,0,0,0,
        11,0,0,0
    ]),
    wam_c_reverse_csr_lmdb_offset_smoke_main(ValuesPath, OffsetEnvPath, MainCode),
    write_text_file(MainPath, MainCode),
    compile_c_smoke_lmdb(RuntimePath, PredPath, MainPath, ExePath),
    run_c_smoke_plain(ExePath).

run_kernel_detector_executable_smoke :-
    setup_wam_c_detector_category_ancestor,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_kernel_detector_smoke', Stamp, ProjectDir),
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

run_transitive_closure_detector_executable_smoke :-
    setup_wam_c_detector_transitive_closure,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_transitive_detector_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_transitive_detector_smoke', ExePath),
    (   write_wam_c_project([user:tc_ancestor/2], [], ProjectDir),
        wam_c_transitive_closure_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_transitive_closure
    ;   cleanup_wam_c_detector_transitive_closure,
        fail
    ).

run_transitive_distance_detector_executable_smoke :-
    setup_wam_c_detector_transitive_distance,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_distance_detector_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_distance_detector_smoke', ExePath),
    (   write_wam_c_project([user:tc_distance/3], [], ProjectDir),
        wam_c_transitive_distance_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_transitive_distance
    ;   cleanup_wam_c_detector_transitive_distance,
        fail
    ).

run_transitive_parent_distance_detector_executable_smoke :-
    setup_wam_c_detector_transitive_parent_distance,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_parent_distance_detector_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_parent_distance_detector_smoke', ExePath),
    (   write_wam_c_project([user:tc_parent_distance/4], [], ProjectDir),
        wam_c_transitive_parent_distance_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_transitive_parent_distance
    ;   cleanup_wam_c_detector_transitive_parent_distance,
        fail
    ).

run_transitive_step_parent_distance_detector_executable_smoke :-
    setup_wam_c_detector_transitive_step_parent_distance,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_step_parent_distance_detector_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_step_parent_distance_detector_smoke', ExePath),
    (   write_wam_c_project([user:tc_step_parent_distance/5], [], ProjectDir),
        wam_c_transitive_step_parent_distance_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_transitive_step_parent_distance
    ;   cleanup_wam_c_detector_transitive_step_parent_distance,
        fail
    ).

run_weighted_shortest_path_detector_executable_smoke :-
    setup_wam_c_detector_weighted_shortest_path,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_weighted_shortest_path_detector_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_weighted_shortest_path_detector_smoke', ExePath),
    (   write_wam_c_project([user:weighted_path/3], [], ProjectDir),
        wam_c_weighted_shortest_path_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_weighted_shortest_path
    ;   cleanup_wam_c_detector_weighted_shortest_path,
        fail
    ).

run_astar_shortest_path_detector_executable_smoke :-
    setup_wam_c_detector_astar_shortest_path,
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_astar_shortest_path_detector_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_astar_shortest_path_detector_smoke', ExePath),
    (   write_wam_c_project([user:astar_path/4], [], ProjectDir),
        wam_c_astar_shortest_path_detector_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_detector_astar_shortest_path
    ;   cleanup_wam_c_detector_astar_shortest_path,
        fail
    ).

run_streaming_foreign_results_executable_smoke :-
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_streaming_foreign_smoke', Stamp, TmpBase),
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
        wam_c_temp_path('unifyweaver_wam_c_real_builtin_smoke', Stamp, TmpBase),
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

run_real_prolog_term_builtin_executable_smoke :-
    assertz((user:wam_c_real_term_builtins(T, Label) :-
        functor(T, Name, Arity),
        Arity = 2,
        N is 2,
        arg(N, T, Arg),
        atom_concat(Name, '_', Prefix),
        atom_concat(Prefix, Arg, Label))),
    (   compile_predicate_to_wam(user:wam_c_real_term_builtins/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'builtin_call functor/3, 3'),
        sub_string(WamCode, _, _, _, 'builtin_call arg/3, 3'),
        sub_string(WamCode, _, _, _, 'builtin_call atom_concat/3, 3'),
        compile_wam_predicate_to_c(user:wam_c_real_term_builtins/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_real_term_builtin_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_term_builtin_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_real_term_builtins(_, _))
    ;   retractall(user:wam_c_real_term_builtins(_, _)),
        fail
    ).

run_real_prolog_multiclause_executable_smoke :-
    assertz((user:wam_c_real_multi(a, yes) :- true)),
    assertz((user:wam_c_real_multi(X, int) :- integer(X))),
    assertz((user:wam_c_real_multi([_|_], list) :- true)),
    (   compile_predicate_to_wam(user:wam_c_real_multi/2, [], WamCode),
        sub_string(WamCode, _, _, _, 'switch_on_constant_fallthrough'),
        sub_string(WamCode, _, _, _, 'retry_me_else'),
        sub_string(WamCode, _, _, _, 'builtin_call integer/1, 1'),
        compile_wam_predicate_to_c(user:wam_c_real_multi/2, WamCode, [], PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_real_multi_smoke', Stamp, TmpBase),
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
        wam_c_temp_path('unifyweaver_wam_c_real_struct_smoke', Stamp, TmpBase),
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
        wam_c_temp_path('unifyweaver_wam_c_real_is_list_smoke', Stamp, TmpBase),
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
        wam_c_temp_path('unifyweaver_wam_c_real_unify_smoke', Stamp, TmpBase),
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

run_real_prolog_control_executable_smoke :-
    assertz((user:wam_c_ctrl_known(a) :- true)),
    assertz((user:wam_c_ctrl_neg(X, ok) :- \+ wam_c_ctrl_known(X))),
    assertz((user:wam_c_ctrl_if_known(X, yes) :-
        (wam_c_ctrl_known(X) -> true ; fail))),
    assertz((user:wam_c_ctrl_if_missing(X, no) :-
        (wam_c_ctrl_known(X) -> fail ; true))),
    (   compile_predicate_to_wam(user:wam_c_ctrl_known/1, [], WamKnown),
        compile_predicate_to_wam(user:wam_c_ctrl_neg/2, [], WamNeg),
        compile_predicate_to_wam(user:wam_c_ctrl_if_known/2, [], WamIfKnown),
        compile_predicate_to_wam(user:wam_c_ctrl_if_missing/2, [], WamIfMissing),
        sub_string(WamNeg, _, _, _, 'jump '),
        sub_string(WamIfKnown, _, _, _, 'cut_ite'),
        sub_string(WamIfKnown, _, _, _, 'jump '),
        sub_string(WamIfMissing, _, _, _, 'cut_ite'),
        compile_wam_predicate_to_c(user:wam_c_ctrl_known/1, WamKnown, [], KnownCode),
        compile_wam_predicate_to_c(user:wam_c_ctrl_neg/2, WamNeg, [], NegCode),
        compile_wam_predicate_to_c(user:wam_c_ctrl_if_known/2, WamIfKnown, [], IfKnownCode),
        compile_wam_predicate_to_c(user:wam_c_ctrl_if_missing/2, WamIfMissing, [], IfMissingCode),
        atomic_list_concat([KnownCode, NegCode, IfKnownCode, IfMissingCode], '\n\n', PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_real_control_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_real_control_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_real_control_smoke
    ;   cleanup_wam_c_real_control_smoke,
        fail
    ).

cleanup_wam_c_real_control_smoke :-
    retractall(user:wam_c_ctrl_known(_)),
    retractall(user:wam_c_ctrl_neg(_, _)),
    retractall(user:wam_c_ctrl_if_known(_, _)),
    retractall(user:wam_c_ctrl_if_missing(_, _)).

run_real_prolog_precise_ite_executable_smoke :-
    assertz((user:wam_c_precise_choice(a) :- true)),
    assertz((user:wam_c_precise_choice(a) :- true)),
    assertz((user:wam_c_precise_then(X, then) :-
        (wam_c_precise_choice(X) -> true ; fail))),
    assertz((user:wam_c_precise_else(X, else) :-
        (wam_c_precise_choice(X) -> fail ; true))),
    assertz((user:wam_c_precise_scope(X) :-
        (wam_c_precise_choice(X) -> R = then ; R = else),
        R = else)),
    Options = [ite_use_y_level(true)],
    (   compile_predicate_to_wam(user:wam_c_precise_choice/1, Options, WamChoice),
        compile_predicate_to_wam(user:wam_c_precise_then/2, Options, WamThen),
        compile_predicate_to_wam(user:wam_c_precise_else/2, Options, WamElse),
        compile_predicate_to_wam(user:wam_c_precise_scope/1, Options, WamScope),
        sub_string(WamThen, _, _, _, 'get_level '),
        sub_string(WamThen, _, _, _, 'cut Y'),
        \+ sub_string(WamThen, _, _, _, 'cut_ite'),
        sub_string(WamElse, _, _, _, 'get_level '),
        sub_string(WamElse, _, _, _, 'cut Y'),
        \+ sub_string(WamElse, _, _, _, 'cut_ite'),
        sub_string(WamScope, _, _, _, 'get_level '),
        sub_string(WamScope, _, _, _, 'cut Y'),
        \+ sub_string(WamScope, _, _, _, 'cut_ite'),
        compile_wam_predicate_to_c(user:wam_c_precise_choice/1, WamChoice, Options, ChoiceCode),
        compile_wam_predicate_to_c(user:wam_c_precise_then/2, WamThen, Options, ThenCode),
        compile_wam_predicate_to_c(user:wam_c_precise_else/2, WamElse, Options, ElseCode),
        compile_wam_predicate_to_c(user:wam_c_precise_scope/1, WamScope, Options, ScopeCode),
        atomic_list_concat([ChoiceCode, ThenCode, ElseCode, ScopeCode], '\n\n', PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_precise_ite_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_precise_ite_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_precise_ite_smoke
    ;   cleanup_wam_c_precise_ite_smoke,
        fail
    ).

cleanup_wam_c_precise_ite_smoke :-
    retractall(user:wam_c_precise_choice(_)),
    retractall(user:wam_c_precise_then(_, _)),
    retractall(user:wam_c_precise_else(_, _)),
    retractall(user:wam_c_precise_scope(_)).

run_real_prolog_explicit_cut_executable_smoke :-
    assertz((user:wam_c_cut_choice(a) :- true)),
    assertz((user:wam_c_cut_choice(b) :- true)),
    assertz((user:wam_c_inner_cut :- wam_c_cut_choice(_), !)),
    assertz((user:wam_c_outer_cut(ok) :- (wam_c_inner_cut, fail ; true))),
    (   compile_predicate_to_wam(user:wam_c_cut_choice/1, [], WamChoice),
        compile_predicate_to_wam(user:wam_c_inner_cut/0, [], WamInner),
        compile_predicate_to_wam(user:wam_c_outer_cut/1, [], WamOuter),
        sub_string(WamInner, _, _, _, 'builtin_call !/0, 0'),
        sub_string(WamOuter, _, _, _, 'try_me_else'),
        compile_wam_predicate_to_c(user:wam_c_cut_choice/1, WamChoice, [], ChoiceCode),
        compile_wam_predicate_to_c(user:wam_c_inner_cut/0, WamInner, [], InnerCode),
        compile_wam_predicate_to_c(user:wam_c_outer_cut/1, WamOuter, [], OuterCode),
        atomic_list_concat([ChoiceCode, InnerCode, OuterCode], '\n\n', PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_explicit_cut_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_explicit_cut_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_explicit_cut_smoke
    ;   cleanup_wam_c_explicit_cut_smoke,
        fail
    ).

cleanup_wam_c_explicit_cut_smoke :-
    retractall(user:wam_c_cut_choice(_)),
    retractall(user:wam_c_inner_cut),
    retractall(user:wam_c_outer_cut(_)).

run_real_prolog_forall_executable_smoke :-
    assertz((user:wam_c_forall_num(1) :- true)),
    assertz((user:wam_c_forall_num(2) :- true)),
    assertz((user:wam_c_forall_positive(1) :- true)),
    assertz((user:wam_c_forall_positive(2) :- true)),
    assertz((user:wam_c_forall_only_two(2) :- true)),
    assertz((user:wam_c_forall_empty(_) :- fail)),
    assertz((user:wam_c_forall_all(ok) :-
        forall(wam_c_forall_num(X), wam_c_forall_positive(X)))),
    assertz((user:wam_c_forall_fail(ok) :-
        forall(wam_c_forall_num(X), wam_c_forall_only_two(X)))),
    assertz((user:wam_c_forall_subset(ok) :-
        forall(wam_c_forall_only_two(X), wam_c_forall_positive(X)))),
    assertz((user:wam_c_forall_empty_ok(ok) :-
        forall(wam_c_forall_empty(X), wam_c_forall_positive(X)))),
    (   compile_predicate_to_wam(user:wam_c_forall_num/1, [], WamNum),
        compile_predicate_to_wam(user:wam_c_forall_positive/1, [], WamPositive),
        compile_predicate_to_wam(user:wam_c_forall_only_two/1, [], WamOnlyTwo),
        compile_predicate_to_wam(user:wam_c_forall_empty/1, [], WamEmpty),
        compile_predicate_to_wam(user:wam_c_forall_all/1, [], WamAll),
        compile_predicate_to_wam(user:wam_c_forall_fail/1, [], WamFail),
        compile_predicate_to_wam(user:wam_c_forall_subset/1, [], WamSubset),
        compile_predicate_to_wam(user:wam_c_forall_empty_ok/1, [], WamEmptyOk),
        sub_string(WamAll, _, _, _, 'cut_ite'),
        sub_string(WamFail, _, _, _, 'cut_ite'),
        sub_string(WamSubset, _, _, _, 'cut_ite'),
        sub_string(WamEmptyOk, _, _, _, 'cut_ite'),
        \+ sub_string(WamAll, _, _, _, 'builtin_call forall/2'),
        compile_wam_predicate_to_c(user:wam_c_forall_num/1, WamNum, [], NumCode),
        compile_wam_predicate_to_c(user:wam_c_forall_positive/1, WamPositive, [], PositiveCode),
        compile_wam_predicate_to_c(user:wam_c_forall_only_two/1, WamOnlyTwo, [], OnlyTwoCode),
        compile_wam_predicate_to_c(user:wam_c_forall_empty/1, WamEmpty, [], EmptyCode),
        compile_wam_predicate_to_c(user:wam_c_forall_all/1, WamAll, [], AllCode),
        compile_wam_predicate_to_c(user:wam_c_forall_fail/1, WamFail, [], FailCode),
        compile_wam_predicate_to_c(user:wam_c_forall_subset/1, WamSubset, [], SubsetCode),
        compile_wam_predicate_to_c(user:wam_c_forall_empty_ok/1, WamEmptyOk, [], EmptyOkCode),
        atomic_list_concat([NumCode, PositiveCode, OnlyTwoCode, EmptyCode,
                            AllCode, FailCode, SubsetCode, EmptyOkCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_forall_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_forall_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_forall_smoke
    ;   cleanup_wam_c_forall_smoke,
        fail
    ).

cleanup_wam_c_forall_smoke :-
    retractall(user:wam_c_forall_num(_)),
    retractall(user:wam_c_forall_positive(_)),
    retractall(user:wam_c_forall_only_two(_)),
    retractall(user:wam_c_forall_empty(_)),
    retractall(user:wam_c_forall_all(_)),
    retractall(user:wam_c_forall_fail(_)),
    retractall(user:wam_c_forall_subset(_)),
    retractall(user:wam_c_forall_empty_ok(_)).

run_real_prolog_findall_executable_smoke :-
    assertz((user:wam_c_findall_item(a) :- true)),
    assertz((user:wam_c_findall_item(b) :- true)),
    assertz((user:wam_c_findall_none(_) :- fail)),
    assertz((user:wam_c_findall_all(L) :-
        findall(X, wam_c_findall_item(X), L))),
    assertz((user:wam_c_findall_empty(L) :-
        findall(X, wam_c_findall_none(X), L))),
    assertz((user:wam_c_findall_nested_inner(L) :-
        findall(Y, wam_c_findall_item(Y), L))),
    assertz((user:wam_c_findall_nested_outer(L) :-
        findall(X, (wam_c_findall_item(X), wam_c_findall_nested_inner(_Ys)), L))),
    assertz((user:wam_c_findall_inline_nested(L) :-
        findall(X,
                (   wam_c_findall_item(X),
                    findall(Y, wam_c_findall_item(Y), Ys),
                    Ys = [a, b]
                ),
                L))),
    assertz((user:wam_c_findall_struct_template(L) :-
        findall(pair(X, [X]), wam_c_findall_item(X), L))),
    assertz((user:wam_c_findall_list_template(L) :-
        findall([X, X], wam_c_findall_item(X), L))),
    (   compile_predicate_to_wam(user:wam_c_findall_item/1, [], WamItem),
        compile_predicate_to_wam(user:wam_c_findall_none/1, [], WamNone),
        compile_predicate_to_wam(user:wam_c_findall_all/1, [], WamAll),
        compile_predicate_to_wam(user:wam_c_findall_empty/1, [], WamEmpty),
        compile_predicate_to_wam(user:wam_c_findall_nested_inner/1, [], WamNestedInner),
        compile_predicate_to_wam(user:wam_c_findall_nested_outer/1, [], WamNestedOuter),
        compile_predicate_to_wam(user:wam_c_findall_inline_nested/1, [], WamInlineNested),
        compile_predicate_to_wam(user:wam_c_findall_struct_template/1, [], WamStructTemplate),
        compile_predicate_to_wam(user:wam_c_findall_list_template/1, [], WamListTemplate),
        sub_string(WamAll, _, _, _, 'begin_aggregate collect'),
        sub_string(WamAll, _, _, _, 'end_aggregate'),
        sub_string(WamEmpty, _, _, _, 'begin_aggregate collect'),
        sub_string(WamEmpty, _, _, _, 'end_aggregate'),
        sub_string(WamNestedInner, _, _, _, 'begin_aggregate collect'),
        sub_string(WamNestedOuter, _, _, _, 'begin_aggregate collect'),
        sub_string(WamInlineNested, _, _, _, 'begin_aggregate collect'),
        \+ sub_string(WamInlineNested, _, _, _, 'call findall/3'),
        sub_string(WamStructTemplate, _, _, _, 'begin_aggregate collect'),
        sub_string(WamListTemplate, _, _, _, 'begin_aggregate collect'),
        \+ sub_string(WamAll, _, _, _, 'builtin_call findall/3'),
        compile_wam_predicate_to_c(user:wam_c_findall_item/1, WamItem, [], ItemCode),
        compile_wam_predicate_to_c(user:wam_c_findall_none/1, WamNone, [], NoneCode),
        compile_wam_predicate_to_c(user:wam_c_findall_all/1, WamAll, [], AllCode),
        compile_wam_predicate_to_c(user:wam_c_findall_empty/1, WamEmpty, [], EmptyCode),
        compile_wam_predicate_to_c(user:wam_c_findall_nested_inner/1, WamNestedInner, [], NestedInnerCode),
        compile_wam_predicate_to_c(user:wam_c_findall_nested_outer/1, WamNestedOuter, [], NestedOuterCode),
        compile_wam_predicate_to_c(user:wam_c_findall_inline_nested/1, WamInlineNested, [], InlineNestedCode),
        compile_wam_predicate_to_c(user:wam_c_findall_struct_template/1, WamStructTemplate, [], StructTemplateCode),
        compile_wam_predicate_to_c(user:wam_c_findall_list_template/1, WamListTemplate, [], ListTemplateCode),
        atomic_list_concat([ItemCode, NoneCode, AllCode, EmptyCode,
                            NestedInnerCode, NestedOuterCode,
                            InlineNestedCode,
                            StructTemplateCode, ListTemplateCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_findall_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_findall_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_findall_smoke
    ;   cleanup_wam_c_findall_smoke,
        fail
    ).

cleanup_wam_c_findall_smoke :-
    retractall(user:wam_c_findall_item(_)),
    retractall(user:wam_c_findall_none(_)),
    retractall(user:wam_c_findall_all(_)),
    retractall(user:wam_c_findall_empty(_)),
    retractall(user:wam_c_findall_nested_inner(_)),
    retractall(user:wam_c_findall_nested_outer(_)),
    retractall(user:wam_c_findall_inline_nested(_)),
    retractall(user:wam_c_findall_struct_template(_)),
    retractall(user:wam_c_findall_list_template(_)).

run_real_prolog_bagof_setof_executable_smoke :-
    Options = [inline_bagof_setof(true)],
    assertz((user:wam_c_bagset_item(b) :- true)),
    assertz((user:wam_c_bagset_item(a) :- true)),
    assertz((user:wam_c_bagset_none(_) :- fail)),
    assertz((user:wam_c_bagset_bag(L) :-
        bagof(X, (wam_c_bagset_item(X) ; wam_c_bagset_item(X)), L))),
    assertz((user:wam_c_bagset_set(L) :-
        setof(X, (wam_c_bagset_item(X) ; wam_c_bagset_item(X)), L))),
    assertz((user:wam_c_bagset_bag_empty(L) :-
        bagof(X, wam_c_bagset_none(X), L))),
    assertz((user:wam_c_bagset_set_empty(L) :-
        setof(X, wam_c_bagset_none(X), L))),
    (   compile_predicate_to_wam(user:wam_c_bagset_item/1, Options, WamItem),
        compile_predicate_to_wam(user:wam_c_bagset_none/1, Options, WamNone),
        compile_predicate_to_wam(user:wam_c_bagset_bag/1, Options, WamBag),
        compile_predicate_to_wam(user:wam_c_bagset_set/1, Options, WamSet),
        compile_predicate_to_wam(user:wam_c_bagset_bag_empty/1, Options, WamBagEmpty),
        compile_predicate_to_wam(user:wam_c_bagset_set_empty/1, Options, WamSetEmpty),
        sub_string(WamBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamBag, _, _, _, "''"),
        \+ sub_string(WamBag, _, _, _, 'call bagof/3'),
        sub_string(WamSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamSet, _, _, _, "''"),
        \+ sub_string(WamSet, _, _, _, 'call setof/3'),
        compile_wam_predicate_to_c(user:wam_c_bagset_item/1, WamItem, Options, ItemCode),
        compile_wam_predicate_to_c(user:wam_c_bagset_none/1, WamNone, Options, NoneCode),
        compile_wam_predicate_to_c(user:wam_c_bagset_bag/1, WamBag, Options, BagCode),
        compile_wam_predicate_to_c(user:wam_c_bagset_set/1, WamSet, Options, SetCode),
        compile_wam_predicate_to_c(user:wam_c_bagset_bag_empty/1, WamBagEmpty, Options, BagEmptyCode),
        compile_wam_predicate_to_c(user:wam_c_bagset_set_empty/1, WamSetEmpty, Options, SetEmptyCode),
        atomic_list_concat([ItemCode, NoneCode, BagCode, SetCode,
                            BagEmptyCode, SetEmptyCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_bagof_setof_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_bagof_setof_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_bagof_setof_smoke
    ;   cleanup_wam_c_bagof_setof_smoke,
        fail
    ).

cleanup_wam_c_bagof_setof_smoke :-
    retractall(user:wam_c_bagset_item(_)),
    retractall(user:wam_c_bagset_none(_)),
    retractall(user:wam_c_bagset_bag(_)),
    retractall(user:wam_c_bagset_set(_)),
    retractall(user:wam_c_bagset_bag_empty(_)),
    retractall(user:wam_c_bagset_set_empty(_)).

run_real_prolog_bagof_setof_witness_executable_smoke :-
    Options = [inline_bagof_setof(true)],
    assertz((user:wam_c_group_pair(b, red) :- true)),
    assertz((user:wam_c_group_pair(a, red) :- true)),
    assertz((user:wam_c_group_pair(c, blue) :- true)),
    assertz((user:wam_c_group_bag(Y, L) :-
        bagof(X, wam_c_group_pair(X, Y), L))),
    assertz((user:wam_c_group_set(Y, L) :-
        setof(X, wam_c_group_pair(X, Y), L))),
    (   compile_predicate_to_wam(user:wam_c_group_pair/2, Options, WamPair),
        compile_predicate_to_wam(user:wam_c_group_bag/2, Options, WamBag),
        compile_predicate_to_wam(user:wam_c_group_set/2, Options, WamSet),
        sub_string(WamBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamBag, _, _, _, "'Y"),
        \+ sub_string(WamBag, _, _, _, 'call bagof/3'),
        sub_string(WamSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamSet, _, _, _, "'Y"),
        \+ sub_string(WamSet, _, _, _, 'call setof/3'),
        compile_wam_predicate_to_c(user:wam_c_group_pair/2, WamPair, Options, PairCode),
        compile_wam_predicate_to_c(user:wam_c_group_bag/2, WamBag, Options, BagCode),
        compile_wam_predicate_to_c(user:wam_c_group_set/2, WamSet, Options, SetCode),
        atomic_list_concat([PairCode, BagCode, SetCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_bagof_setof_witness_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_bagof_setof_witness_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_bagof_setof_witness_smoke
    ;   cleanup_wam_c_bagof_setof_witness_smoke,
        fail
    ).

cleanup_wam_c_bagof_setof_witness_smoke :-
    retractall(user:wam_c_group_pair(_, _)),
    retractall(user:wam_c_group_bag(_, _)),
    retractall(user:wam_c_group_set(_, _)).

run_real_prolog_bagof_setof_existential_executable_smoke :-
    Options = [inline_bagof_setof(true)],
    assertz((user:wam_c_exist_pair(b, red) :- true)),
    assertz((user:wam_c_exist_pair(a, red) :- true)),
    assertz((user:wam_c_exist_pair(c, blue) :- true)),
    assertz((user:wam_c_exist_pair(b, blue) :- true)),
    assertz((user:wam_c_exist_bag(L) :-
        bagof(X, Y^wam_c_exist_pair(X, Y), L))),
    assertz((user:wam_c_exist_set(L) :-
        setof(X, Y^wam_c_exist_pair(X, Y), L))),
    (   compile_predicate_to_wam(user:wam_c_exist_pair/2, Options, WamPair),
        compile_predicate_to_wam(user:wam_c_exist_bag/1, Options, WamBag),
        compile_predicate_to_wam(user:wam_c_exist_set/1, Options, WamSet),
        sub_string(WamBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamBag, _, _, _, "''"),
        sub_string(WamBag, _, _, _, 'call wam_c_exist_pair/2'),
        \+ sub_string(WamBag, _, _, _, 'call ^/2'),
        \+ sub_string(WamBag, _, _, _, 'call bagof/3'),
        sub_string(WamSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamSet, _, _, _, "''"),
        sub_string(WamSet, _, _, _, 'call wam_c_exist_pair/2'),
        \+ sub_string(WamSet, _, _, _, 'call ^/2'),
        \+ sub_string(WamSet, _, _, _, 'call setof/3'),
        compile_wam_predicate_to_c(user:wam_c_exist_pair/2, WamPair, Options, PairCode),
        compile_wam_predicate_to_c(user:wam_c_exist_bag/1, WamBag, Options, BagCode),
        compile_wam_predicate_to_c(user:wam_c_exist_set/1, WamSet, Options, SetCode),
        atomic_list_concat([PairCode, BagCode, SetCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_bagof_setof_existential_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_bagof_setof_existential_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_bagof_setof_existential_smoke
    ;   cleanup_wam_c_bagof_setof_existential_smoke,
        fail
    ).

cleanup_wam_c_bagof_setof_existential_smoke :-
    retractall(user:wam_c_exist_pair(_, _)),
    retractall(user:wam_c_exist_bag(_)),
    retractall(user:wam_c_exist_set(_)).

run_real_prolog_bagof_setof_unbound_witness_groups_smoke :-
    Options = [inline_bagof_setof(true)],
    assertz((user:wam_c_uw_pair(b, red) :- true)),
    assertz((user:wam_c_uw_pair(a, red) :- true)),
    assertz((user:wam_c_uw_pair(c, blue) :- true)),
    assertz((user:wam_c_uw_pair(d, blue) :- true)),
    assertz((user:wam_c_uw_bag(Y, L) :-
        bagof(X, wam_c_uw_pair(X, Y), L))),
    assertz((user:wam_c_uw_set(Y, L) :-
        setof(X, wam_c_uw_pair(X, Y), L))),
    assertz((user:wam_c_uw_bag_groups_ok :-
        findall(Y-L, wam_c_uw_bag(Y, L), Groups),
        Groups = [red-[b, a], blue-[c, d]])),
    assertz((user:wam_c_uw_set_groups_ok :-
        findall(Y-L, wam_c_uw_set(Y, L), Groups),
        Groups = [red-[a, b], blue-[c, d]])),
    (   compile_predicate_to_wam(user:wam_c_uw_pair/2, Options, WamPair),
        compile_predicate_to_wam(user:wam_c_uw_bag/2, Options, WamBag),
        compile_predicate_to_wam(user:wam_c_uw_set/2, Options, WamSet),
        compile_predicate_to_wam(user:wam_c_uw_bag_groups_ok/0, Options, WamBagGroups),
        compile_predicate_to_wam(user:wam_c_uw_set_groups_ok/0, Options, WamSetGroups),
        sub_string(WamBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamBag, _, _, _, "'Y"),
        \+ sub_string(WamBag, _, _, _, 'call bagof/3'),
        sub_string(WamSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamSet, _, _, _, "'Y"),
        \+ sub_string(WamSet, _, _, _, 'call setof/3'),
        sub_string(WamBagGroups, _, _, _, 'begin_aggregate collect'),
        sub_string(WamBagGroups, _, _, _, 'call wam_c_uw_bag/2'),
        sub_string(WamSetGroups, _, _, _, 'begin_aggregate collect'),
        sub_string(WamSetGroups, _, _, _, 'call wam_c_uw_set/2'),
        compile_wam_predicate_to_c(user:wam_c_uw_pair/2, WamPair, Options, PairCode),
        compile_wam_predicate_to_c(user:wam_c_uw_bag/2, WamBag, Options, BagCode),
        compile_wam_predicate_to_c(user:wam_c_uw_set/2, WamSet, Options, SetCode),
        compile_wam_predicate_to_c(user:wam_c_uw_bag_groups_ok/0, WamBagGroups, Options, BagGroupsCode),
        compile_wam_predicate_to_c(user:wam_c_uw_set_groups_ok/0, WamSetGroups, Options, SetGroupsCode),
        atomic_list_concat([PairCode, BagCode, SetCode,
                            BagGroupsCode, SetGroupsCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_bagof_setof_unbound_witness_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_bagof_setof_unbound_witness_groups_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_bagof_setof_unbound_witness_groups_smoke
    ;   cleanup_wam_c_bagof_setof_unbound_witness_groups_smoke,
        fail
    ).

cleanup_wam_c_bagof_setof_unbound_witness_groups_smoke :-
    retractall(user:wam_c_uw_pair(_, _)),
    retractall(user:wam_c_uw_bag(_, _)),
    retractall(user:wam_c_uw_set(_, _)),
    retractall(user:wam_c_uw_bag_groups_ok),
    retractall(user:wam_c_uw_set_groups_ok).

run_real_prolog_bagof_setof_meta_call_smoke :-
    assertz((user:wam_c_meta_item(a) :- true)),
    assertz((user:wam_c_meta_item(b) :- true)),
    assertz((user:wam_c_meta_dup(b) :- true)),
    assertz((user:wam_c_meta_dup(a) :- true)),
    assertz((user:wam_c_meta_dup(b) :- true)),
    assertz((user:wam_c_meta_none(_) :- fail)),
    assertz((user:wam_c_meta_conj_item(a) :- true)),
    assertz((user:wam_c_meta_conj_item(b) :- true)),
    assertz((user:wam_c_meta_conj_item(c) :- true)),
    assertz((user:wam_c_meta_conj_keep(a) :- true)),
    assertz((user:wam_c_meta_conj_keep(c) :- true)),
    assertz((user:wam_c_meta_conj_dup(b) :- true)),
    assertz((user:wam_c_meta_conj_dup(a) :- true)),
    assertz((user:wam_c_meta_conj_dup(b) :- true)),
    assertz((user:wam_c_meta_conj_set_keep(a) :- true)),
    assertz((user:wam_c_meta_conj_set_keep(b) :- true)),
    assertz((user:wam_c_meta_disj_left(a) :- true)),
    assertz((user:wam_c_meta_disj_left(b) :- true)),
    assertz((user:wam_c_meta_disj_right(c) :- true)),
    assertz((user:wam_c_meta_disj_set_left(b) :- true)),
    assertz((user:wam_c_meta_disj_set_left(a) :- true)),
    assertz((user:wam_c_meta_disj_set_right(b) :- true)),
    assertz((user:wam_c_meta_disj_set_right(c) :- true)),
    assertz((user:wam_c_meta_ite_cond(a) :- true)),
    assertz((user:wam_c_meta_ite_cond(b) :- true)),
    assertz((user:wam_c_meta_ite_then(a) :- true)),
    assertz((user:wam_c_meta_ite_then(a) :- true)),
    assertz((user:wam_c_meta_ite_fail_cond(_) :- fail)),
    assertz((user:wam_c_meta_ite_else(c) :- true)),
    assertz((user:wam_c_meta_ite_else(d) :- true)),
    assertz((user:wam_c_meta_ite_set_then(b) :- true)),
    assertz((user:wam_c_meta_ite_set_then(a) :- true)),
    assertz((user:wam_c_meta_ite_set_then(b) :- true)),
    assertz((user:wam_c_meta_call_item(a) :- true)),
    assertz((user:wam_c_meta_call_item(b) :- true)),
    assertz((user:wam_c_meta_call_pair(a, k) :- true)),
    assertz((user:wam_c_meta_call_pair(b, k) :- true)),
    assertz((user:wam_c_meta_call_dup(b) :- true)),
    assertz((user:wam_c_meta_call_dup(a) :- true)),
    assertz((user:wam_c_meta_call_dup(b) :- true)),
    assertz((user:wam_c_meta_bag(L) :-
        bagof(X, wam_c_meta_item(X), L))),
    assertz((user:wam_c_meta_set(L) :-
        setof(X, wam_c_meta_dup(X), L))),
    assertz((user:wam_c_meta_empty_bag(L) :-
        bagof(X, wam_c_meta_none(X), L))),
    assertz((user:wam_c_meta_conj_bag(L) :-
        bagof(X,
              (wam_c_meta_conj_item(X), wam_c_meta_conj_keep(X)),
              L))),
    assertz((user:wam_c_meta_conj_set(L) :-
        setof(X,
              (wam_c_meta_conj_dup(X), wam_c_meta_conj_set_keep(X)),
              L))),
    assertz((user:wam_c_meta_disj_bag(L) :-
        bagof(X,
              (wam_c_meta_disj_left(X); wam_c_meta_disj_right(X)),
              L))),
    assertz((user:wam_c_meta_disj_set(L) :-
        setof(X,
              (wam_c_meta_disj_set_left(X); wam_c_meta_disj_set_right(X)),
              L))),
    assertz((user:wam_c_meta_ite_bag(L) :-
        bagof(X,
              (wam_c_meta_ite_cond(X)
               -> wam_c_meta_ite_then(X)
               ;  wam_c_meta_ite_else(X)),
              L))),
    assertz((user:wam_c_meta_ite_else_bag(L) :-
        bagof(X,
              (wam_c_meta_ite_fail_cond(X)
               -> wam_c_meta_ite_then(X)
               ;  wam_c_meta_ite_else(X)),
              L))),
    assertz((user:wam_c_meta_ite_set(L) :-
        setof(X,
              (true
               -> wam_c_meta_ite_set_then(X)
               ;  wam_c_meta_ite_else(X)),
              L))),
    assertz((user:wam_c_meta_call_bag(L) :-
        bagof(X, call(wam_c_meta_call_item, X), L))),
    assertz((user:wam_c_meta_call_pair_bag(L) :-
        bagof(X, call(wam_c_meta_call_pair, X, k), L))),
    assertz((user:wam_c_meta_call_partial_bag(L) :-
        bagof(X, call(wam_c_meta_call_pair(X), k), L))),
    assertz((user:wam_c_meta_call_set(L) :-
        setof(X, call(wam_c_meta_call_dup, X), L))),
    (   compile_predicate_to_wam(user:wam_c_meta_item/1, [], WamItem),
        compile_predicate_to_wam(user:wam_c_meta_dup/1, [], WamDup),
        compile_predicate_to_wam(user:wam_c_meta_none/1, [], WamNone),
        compile_predicate_to_wam(user:wam_c_meta_conj_item/1, [], WamConjItem),
        compile_predicate_to_wam(user:wam_c_meta_conj_keep/1, [], WamConjKeep),
        compile_predicate_to_wam(user:wam_c_meta_conj_dup/1, [], WamConjDup),
        compile_predicate_to_wam(user:wam_c_meta_conj_set_keep/1, [], WamConjSetKeep),
        compile_predicate_to_wam(user:wam_c_meta_disj_left/1, [], WamDisjLeft),
        compile_predicate_to_wam(user:wam_c_meta_disj_right/1, [], WamDisjRight),
        compile_predicate_to_wam(user:wam_c_meta_disj_set_left/1, [], WamDisjSetLeft),
        compile_predicate_to_wam(user:wam_c_meta_disj_set_right/1, [], WamDisjSetRight),
        compile_predicate_to_wam(user:wam_c_meta_ite_cond/1, [], WamIteCond),
        compile_predicate_to_wam(user:wam_c_meta_ite_then/1, [], WamIteThen),
        compile_predicate_to_wam(user:wam_c_meta_ite_fail_cond/1, [], WamIteFailCond),
        compile_predicate_to_wam(user:wam_c_meta_ite_else/1, [], WamIteElse),
        compile_predicate_to_wam(user:wam_c_meta_ite_set_then/1, [], WamIteSetThen),
        compile_predicate_to_wam(user:wam_c_meta_call_item/1, [], WamCallItem),
        compile_predicate_to_wam(user:wam_c_meta_call_pair/2, [], WamCallPair),
        compile_predicate_to_wam(user:wam_c_meta_call_dup/1, [], WamCallDup),
        compile_predicate_to_wam(user:wam_c_meta_bag/1, [], WamBag),
        compile_predicate_to_wam(user:wam_c_meta_set/1, [], WamSet),
        compile_predicate_to_wam(user:wam_c_meta_empty_bag/1, [], WamEmptyBag),
        compile_predicate_to_wam(user:wam_c_meta_conj_bag/1, [], WamConjBag),
        compile_predicate_to_wam(user:wam_c_meta_conj_set/1, [], WamConjSet),
        compile_predicate_to_wam(user:wam_c_meta_disj_bag/1, [], WamDisjBag),
        compile_predicate_to_wam(user:wam_c_meta_disj_set/1, [], WamDisjSet),
        compile_predicate_to_wam(user:wam_c_meta_ite_bag/1, [], WamIteBag),
        compile_predicate_to_wam(user:wam_c_meta_ite_else_bag/1, [], WamIteElseBag),
        compile_predicate_to_wam(user:wam_c_meta_ite_set/1, [], WamIteSet),
        compile_predicate_to_wam(user:wam_c_meta_call_bag/1, [], WamCallBag),
        compile_predicate_to_wam(user:wam_c_meta_call_pair_bag/1, [], WamCallPairBag),
        compile_predicate_to_wam(user:wam_c_meta_call_partial_bag/1, [], WamCallPartialBag),
        compile_predicate_to_wam(user:wam_c_meta_call_set/1, [], WamCallSet),
        sub_string(WamBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamSet, _, _, _, 'execute setof/3'),
        \+ sub_string(WamSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamConjBag, _, _, _, 'put_structure ,/2'),
        sub_string(WamConjBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamConjBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamConjSet, _, _, _, 'put_structure ,/2'),
        sub_string(WamConjSet, _, _, _, 'execute setof/3'),
        \+ sub_string(WamConjSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamDisjBag, _, _, _, 'put_structure ;/2'),
        sub_string(WamDisjBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamDisjBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamDisjSet, _, _, _, 'put_structure ;/2'),
        sub_string(WamDisjSet, _, _, _, 'execute setof/3'),
        \+ sub_string(WamDisjSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamIteBag, _, _, _, 'put_structure ;/2'),
        sub_string(WamIteBag, _, _, _, 'put_structure ->/2'),
        sub_string(WamIteBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamIteBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamIteElseBag, _, _, _, 'put_structure ->/2'),
        sub_string(WamIteElseBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamIteElseBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamIteSet, _, _, _, 'put_structure ->/2'),
        sub_string(WamIteSet, _, _, _, 'execute setof/3'),
        \+ sub_string(WamIteSet, _, _, _, 'begin_aggregate setof'),
        sub_string(WamCallBag, _, _, _, 'put_structure call/2'),
        sub_string(WamCallBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamCallBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamCallPairBag, _, _, _, 'put_structure call/3'),
        sub_string(WamCallPairBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamCallPairBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamCallPartialBag, _, _, _, 'put_structure call/2'),
        sub_string(WamCallPartialBag, _, _, _, 'put_structure wam_c_meta_call_pair/1'),
        sub_string(WamCallPartialBag, _, _, _, 'execute bagof/3'),
        \+ sub_string(WamCallPartialBag, _, _, _, 'begin_aggregate bagof'),
        sub_string(WamCallSet, _, _, _, 'put_structure call/2'),
        sub_string(WamCallSet, _, _, _, 'execute setof/3'),
        \+ sub_string(WamCallSet, _, _, _, 'begin_aggregate setof'),
        compile_wam_predicate_to_c(user:wam_c_meta_item/1, WamItem, [], ItemCode),
        compile_wam_predicate_to_c(user:wam_c_meta_dup/1, WamDup, [], DupCode),
        compile_wam_predicate_to_c(user:wam_c_meta_none/1, WamNone, [], NoneCode),
        compile_wam_predicate_to_c(user:wam_c_meta_conj_item/1, WamConjItem, [], ConjItemCode),
        compile_wam_predicate_to_c(user:wam_c_meta_conj_keep/1, WamConjKeep, [], ConjKeepCode),
        compile_wam_predicate_to_c(user:wam_c_meta_conj_dup/1, WamConjDup, [], ConjDupCode),
        compile_wam_predicate_to_c(user:wam_c_meta_conj_set_keep/1, WamConjSetKeep, [], ConjSetKeepCode),
        compile_wam_predicate_to_c(user:wam_c_meta_disj_left/1, WamDisjLeft, [], DisjLeftCode),
        compile_wam_predicate_to_c(user:wam_c_meta_disj_right/1, WamDisjRight, [], DisjRightCode),
        compile_wam_predicate_to_c(user:wam_c_meta_disj_set_left/1, WamDisjSetLeft, [], DisjSetLeftCode),
        compile_wam_predicate_to_c(user:wam_c_meta_disj_set_right/1, WamDisjSetRight, [], DisjSetRightCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_cond/1, WamIteCond, [], IteCondCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_then/1, WamIteThen, [], IteThenCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_fail_cond/1, WamIteFailCond, [], IteFailCondCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_else/1, WamIteElse, [], IteElseCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_set_then/1, WamIteSetThen, [], IteSetThenCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_item/1, WamCallItem, [], CallItemCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_pair/2, WamCallPair, [], CallPairCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_dup/1, WamCallDup, [], CallDupCode),
        compile_wam_predicate_to_c(user:wam_c_meta_bag/1, WamBag, [], BagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_set/1, WamSet, [], SetCode),
        compile_wam_predicate_to_c(user:wam_c_meta_empty_bag/1, WamEmptyBag, [], EmptyBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_conj_bag/1, WamConjBag, [], ConjBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_conj_set/1, WamConjSet, [], ConjSetCode),
        compile_wam_predicate_to_c(user:wam_c_meta_disj_bag/1, WamDisjBag, [], DisjBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_disj_set/1, WamDisjSet, [], DisjSetCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_bag/1, WamIteBag, [], IteBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_else_bag/1, WamIteElseBag, [], IteElseBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_ite_set/1, WamIteSet, [], IteSetCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_bag/1, WamCallBag, [], CallBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_pair_bag/1, WamCallPairBag, [], CallPairBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_partial_bag/1, WamCallPartialBag, [], CallPartialBagCode),
        compile_wam_predicate_to_c(user:wam_c_meta_call_set/1, WamCallSet, [], CallSetCode),
        atomic_list_concat([ItemCode, DupCode, NoneCode,
                            ConjItemCode, ConjKeepCode, ConjDupCode,
                            ConjSetKeepCode,
                            DisjLeftCode, DisjRightCode,
                            DisjSetLeftCode, DisjSetRightCode,
                            IteCondCode, IteThenCode, IteFailCondCode,
                            IteElseCode, IteSetThenCode,
                            CallItemCode, CallPairCode, CallDupCode,
                            BagCode, SetCode, EmptyBagCode,
                            ConjBagCode, ConjSetCode,
                            DisjBagCode, DisjSetCode,
                            IteBagCode, IteElseBagCode, IteSetCode,
                            CallBagCode, CallPairBagCode,
                            CallPartialBagCode, CallSetCode],
                           '\n\n',
                           PredCode),
        compile_wam_runtime_to_c([], RuntimeCode),
        get_time(Now),
        Stamp is round(Now * 1000000),
        wam_c_temp_path('unifyweaver_wam_c_bagof_setof_meta_smoke', Stamp, TmpBase),
        format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
        format(atom(PredPath), '~w_pred.c', [TmpBase]),
        format(atom(MainPath), '~w_main.c', [TmpBase]),
        format(atom(ExePath), '~w_bin', [TmpBase]),
        write_text_file(RuntimePath, RuntimeCode),
        format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
        write_text_file(PredPath, PredTranslationUnit),
        wam_c_bagof_setof_meta_call_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, PredPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  cleanup_wam_c_bagof_setof_meta_call_smoke
    ;   cleanup_wam_c_bagof_setof_meta_call_smoke,
        fail
    ).

cleanup_wam_c_bagof_setof_meta_call_smoke :-
    retractall(user:wam_c_meta_item(_)),
    retractall(user:wam_c_meta_dup(_)),
    retractall(user:wam_c_meta_none(_)),
    retractall(user:wam_c_meta_conj_item(_)),
    retractall(user:wam_c_meta_conj_keep(_)),
    retractall(user:wam_c_meta_conj_dup(_)),
    retractall(user:wam_c_meta_conj_set_keep(_)),
    retractall(user:wam_c_meta_disj_left(_)),
    retractall(user:wam_c_meta_disj_right(_)),
    retractall(user:wam_c_meta_disj_set_left(_)),
    retractall(user:wam_c_meta_disj_set_right(_)),
    retractall(user:wam_c_meta_ite_cond(_)),
    retractall(user:wam_c_meta_ite_then(_)),
    retractall(user:wam_c_meta_ite_fail_cond(_)),
    retractall(user:wam_c_meta_ite_else(_)),
    retractall(user:wam_c_meta_ite_set_then(_)),
    retractall(user:wam_c_meta_call_item(_)),
    retractall(user:wam_c_meta_call_pair(_, _)),
    retractall(user:wam_c_meta_call_dup(_)),
    retractall(user:wam_c_meta_bag(_)),
    retractall(user:wam_c_meta_set(_)),
    retractall(user:wam_c_meta_empty_bag(_)),
    retractall(user:wam_c_meta_conj_bag(_)),
    retractall(user:wam_c_meta_conj_set(_)),
    retractall(user:wam_c_meta_disj_bag(_)),
    retractall(user:wam_c_meta_disj_set(_)),
    retractall(user:wam_c_meta_ite_bag(_)),
    retractall(user:wam_c_meta_ite_else_bag(_)),
    retractall(user:wam_c_meta_ite_set(_)),
    retractall(user:wam_c_meta_call_bag(_)),
    retractall(user:wam_c_meta_call_pair_bag(_)),
    retractall(user:wam_c_meta_call_partial_bag(_)),
    retractall(user:wam_c_meta_call_set(_)).

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
        wam_c_temp_path('unifyweaver_wam_c_classic_fib_smoke', Stamp, TmpBase),
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
    wam_c_temp_path('unifyweaver_wam_c_lowered_fact_smoke', Stamp, ProjectDir),
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
    assertz(user:wam_c_body_fact(a, a)),
    assertz((user:wam_c_body_alias(X, Y) :- user:wam_c_body_fact(X, Y))),
    assertz((user:wam_c_body_projected(X, Y) :- user:wam_c_body_fact(Y, X))),
    assertz((user:wam_c_body_ignored_output(X) :- user:wam_c_body_fact(X, _Ignored))),
    assertz((user:wam_c_body_repeated_projection(X) :- user:wam_c_body_fact(X, X))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_body_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_lowered_body_smoke', ExePath),
    (   write_wam_c_project([user:wam_c_body_fact/2,
                             user:wam_c_body_alias/2,
                             user:wam_c_body_projected/2,
                             user:wam_c_body_ignored_output/1,
                             user:wam_c_body_repeated_projection/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        wam_c_lowered_body_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_body_fact(_, _)),
        retractall(user:wam_c_body_alias(_, _)),
        retractall(user:wam_c_body_projected(_, _)),
        retractall(user:wam_c_body_ignored_output(_)),
        retractall(user:wam_c_body_repeated_projection(_))
    ;   retractall(user:wam_c_body_fact(_, _)),
        retractall(user:wam_c_body_alias(_, _)),
        retractall(user:wam_c_body_projected(_, _)),
        retractall(user:wam_c_body_ignored_output(_)),
        retractall(user:wam_c_body_repeated_projection(_)),
        fail
    ).

run_lowered_filtered_fact_helper_executable_smoke :-
    assertz(user:wam_c_filter_fact(a, keep)),
    assertz(user:wam_c_filter_fact(b, drop)),
    assertz(user:wam_c_filter_fact(c, keep)),
    assertz((user:wam_c_filter_keep(X) :- user:wam_c_filter_fact(X, keep))),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_filter_smoke', Stamp, ProjectDir),
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

run_lowered_comparison_filter_helper_executable_smoke :-
    assertz(user:wam_c_filter_score(a, 1)),
    assertz(user:wam_c_filter_score(b, 2)),
    assertz(user:wam_c_filter_score(c, 3)),
    assertz((user:wam_c_filter_small(X) :-
                 user:wam_c_filter_score(X, N),
                 N =< 2)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_comparison_filter_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_lowered_comparison_filter_smoke', ExePath),
    (   write_wam_c_project([user:wam_c_filter_score/2,
                             user:wam_c_filter_small/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        wam_c_lowered_comparison_filter_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_filter_score(_, _)),
        retractall(user:wam_c_filter_small(_))
    ;   retractall(user:wam_c_filter_score(_, _)),
        retractall(user:wam_c_filter_small(_)),
        fail
    ).

run_lowered_repeated_variable_filter_executable_smoke :-
    assertz(user:wam_c_repeat_edge(a, a, keep)),
    assertz(user:wam_c_repeat_edge(a, b, keep)),
    assertz(user:wam_c_repeat_edge(c, c, drop)),
    assertz(user:wam_c_repeat_score(a, a, 1)),
    assertz(user:wam_c_repeat_score(a, b, 1)),
    assertz(user:wam_c_repeat_score(c, c, 3)),
    assertz((user:wam_c_repeat_keep(X) :- user:wam_c_repeat_edge(X, X, keep))),
    assertz((user:wam_c_repeat_small(X) :-
                 user:wam_c_repeat_score(X, X, Score),
                 Score =< 2)),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_lowered_repeat_filter_smoke', Stamp, ProjectDir),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    directory_file_path(ProjectDir, 'main.c', MainPath),
    directory_file_path(ProjectDir, 'wam_c_lowered_repeat_filter_smoke', ExePath),
    (   write_wam_c_project([user:wam_c_repeat_edge/3,
                             user:wam_c_repeat_score/3,
                             user:wam_c_repeat_keep/1,
                             user:wam_c_repeat_small/1],
                            [lowered_helpers(true)],
                            ProjectDir),
        wam_c_lowered_repeat_filter_smoke_main(MainCode),
        write_text_file(MainPath, MainCode),
        compile_c_smoke_plain(RuntimePath, LibPath, MainPath, ExePath),
        run_c_smoke_plain(ExePath)
    ->  retractall(user:wam_c_repeat_edge(_, _, _)),
        retractall(user:wam_c_repeat_score(_, _, _)),
        retractall(user:wam_c_repeat_keep(_)),
        retractall(user:wam_c_repeat_small(_))
    ;   retractall(user:wam_c_repeat_edge(_, _, _)),
        retractall(user:wam_c_repeat_score(_, _, _)),
        retractall(user:wam_c_repeat_keep(_)),
        retractall(user:wam_c_repeat_small(_)),
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
        wam_c_temp_path('unifyweaver_wam_c_asan_lifecycle_smoke', Stamp, TmpBase),
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

write_binary_file(Path, Bytes) :-
    setup_call_cleanup(
        open(Path, write, Stream, [type(binary)]),
        forall(member(Byte, Bytes), put_byte(Stream, Byte)),
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

setup_wam_c_detector_transitive_closure :-
    cleanup_wam_c_detector_transitive_closure,
    assertz((user:tc_parent(tom, bob))),
    assertz((user:tc_parent(bob, ann))),
    assertz((user:tc_ancestor(X, Y) :-
        tc_parent(X, Y))),
    assertz((user:tc_ancestor(X, Y) :-
        tc_parent(X, Z),
        tc_ancestor(Z, Y))).

cleanup_wam_c_detector_transitive_closure :-
    retractall(user:tc_parent(_, _)),
    retractall(user:tc_ancestor(_, _)).

setup_wam_c_detector_transitive_distance :-
    cleanup_wam_c_detector_transitive_distance,
    assertz((user:td_parent(tom, bob))),
    assertz((user:td_parent(bob, ann))),
    assertz((user:td_parent(bob, pat))),
    assertz((user:tc_distance(X, Y, 1) :-
        td_parent(X, Y))),
    assertz((user:tc_distance(X, Y, D) :-
        td_parent(X, Z),
        tc_distance(Z, Y, D0),
        D is D0 + 1)).

cleanup_wam_c_detector_transitive_distance :-
    retractall(user:td_parent(_, _)),
    retractall(user:tc_distance(_, _, _)).

setup_wam_c_detector_transitive_parent_distance :-
    cleanup_wam_c_detector_transitive_parent_distance,
    assertz((user:tpd_parent(tom, bob))),
    assertz((user:tpd_parent(bob, ann))),
    assertz((user:tpd_parent(bob, pat))),
    assertz((user:tc_parent_distance(X, Y, X, 1) :-
        tpd_parent(X, Y))),
    assertz((user:tc_parent_distance(X, Y, P, D) :-
        tpd_parent(X, Z),
        tc_parent_distance(Z, Y, P, D0),
        D is D0 + 1)).

cleanup_wam_c_detector_transitive_parent_distance :-
    retractall(user:tpd_parent(_, _)),
    retractall(user:tc_parent_distance(_, _, _, _)).

setup_wam_c_detector_transitive_step_parent_distance :-
    cleanup_wam_c_detector_transitive_step_parent_distance,
    assertz((user:tspd_parent(tom, bob))),
    assertz((user:tspd_parent(bob, ann))),
    assertz((user:tspd_parent(bob, pat))),
    assertz((user:tc_step_parent_distance(X, Y, Y, X, 1) :-
        tspd_parent(X, Y))),
    assertz((user:tc_step_parent_distance(X, Y, Step, Parent, D) :-
        tspd_parent(X, Step),
        tc_step_parent_distance(Step, Y, _Inner, Parent, D0),
        D is D0 + 1)).

cleanup_wam_c_detector_transitive_step_parent_distance :-
    retractall(user:tspd_parent(_, _)),
    retractall(user:tc_step_parent_distance(_, _, _, _, _)).

setup_wam_c_detector_weighted_shortest_path :-
    cleanup_wam_c_detector_weighted_shortest_path,
    assertz((user:test_weighted_edge(tom, bob, 5))),
    assertz((user:test_weighted_edge(tom, eve, 1))),
    assertz((user:test_weighted_edge(eve, ann, 1))),
    assertz((user:test_weighted_edge(bob, ann, 1))),
    assertz((user:weighted_path(X, Y, W) :-
        test_weighted_edge(X, Y, W))),
    assertz((user:weighted_path(X, Y, W) :-
        test_weighted_edge(X, Z, W0),
        weighted_path(Z, Y, W1),
        W is W0 + W1)).

cleanup_wam_c_detector_weighted_shortest_path :-
    retractall(user:test_weighted_edge(_, _, _)),
    retractall(user:weighted_path(_, _, _)).

setup_wam_c_detector_astar_shortest_path :-
    cleanup_wam_c_detector_astar_shortest_path,
    assertz((user:direct_dist_pred(test_direct_distance/3))),
    assertz((user:dimensionality(5))),
    assertz((user:test_astar_edge(tom, bob, 5))),
    assertz((user:test_astar_edge(tom, eve, 1))),
    assertz((user:test_astar_edge(eve, ann, 1))),
    assertz((user:test_astar_edge(bob, ann, 1))),
    assertz((user:test_direct_distance(tom, ann, 2))),
    assertz((user:test_direct_distance(eve, ann, 1))),
    assertz((user:test_direct_distance(bob, ann, 1))),
    assertz((user:astar_path(X, Y, _Dim, W) :-
        test_astar_edge(X, Y, W))),
    assertz((user:astar_path(X, Y, Dim, W) :-
        test_astar_edge(X, Z, W0),
        astar_path(Z, Y, Dim, W1),
        W is W0 + W1)).

cleanup_wam_c_detector_astar_shortest_path :-
    retractall(user:direct_dist_pred(_)),
    retractall(user:dimensionality(_)),
    retractall(user:test_astar_edge(_, _, _)),
    retractall(user:test_direct_distance(_, _, _)),
    retractall(user:astar_path(_, _, _, _)).

run_c_smoke(ExePath) :-
    format(atom(LogPath), '~w.asan.log', [ExePath]),
    run_c_smoke_once(ExePath, LogPath, Status),
    (   Status =:= 0
    ->  true
    ;   Status =:= 124
    ->  format(atom(RetryLogPath), '~w.asan.retry.log', [ExePath]),
        run_c_smoke_once(ExePath, RetryLogPath, RetryStatus),
        (   RetryStatus =:= 0
        ->  true
        ;   format(user_error, 'generated executable failed with status ~w (log: ~w)~n', [RetryStatus, RetryLogPath]),
            fail
        )
    ;   format(user_error, 'generated executable failed with status ~w (log: ~w)~n', [Status, LogPath]),
        fail
    ).

run_c_smoke_once(ExePath, LogPath, Status) :-
    format(atom(Cmd),
           'ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 timeout 10 ~w > /dev/null 2> ~w',
           [ExePath, LogPath]),
    shell(Cmd, Status).

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
    for (int i = 0; i < 1000; i++) {
        char atom[32];
        snprintf(atom, sizeof(atom), "runtime_atom_%d", i);
        if (wam_intern_atom(&state, atom) == NULL) {
            wam_free_state(&state);
            return 11;
        }
    }
    const char *a3 = wam_intern_atom(&state, "runtime_atom");
    if (a1 != a3 || state.atom_table_size <= WAM_INITIAL_ATOM_HASH_SIZE) {
        wam_free_state(&state);
        return 12;
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

wam_c_multi_setup_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_multi_first_1(WamState* state);
void setup_wam_c_multi_second_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_multi_first_1(&state);
    setup_wam_c_multi_second_1(&state);

    WamValue first_args[1] = { val_atom("a") };
    int first_rc = wam_run_predicate(&state, "wam_c_multi_first/1", first_args, 1);
    if (first_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue second_args[1] = { val_atom("b") };
    int second_rc = wam_run_predicate(&state, "wam_c_multi_second/1", second_args, 1);
    if (second_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue first_fail_args[1] = { val_atom("b") };
    int first_fail_rc = wam_run_predicate(&state, "wam_c_multi_first/1", first_fail_args, 1);
    if (first_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_builtin_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_builtin_atom_1(WamState* state);

static WamValue make_binary_struct(WamState *state, const char *functor, WamValue left, WamValue right) {
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(functor);
    state->H_array[state->H++] = left;
    state->H_array[state->H++] = right;
    return term;
}

static WamValue make_list_pair(WamState *state, WamValue head, WamValue tail) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = head;
    state->H_array[state->H++] = tail;
    return list;
}

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

    WamValue pair_term = make_binary_struct(&state, "pair/2", val_atom("left"), val_int(9));
    WamValue functor_read_args[3] = { pair_term, val_unbound("Name"), val_unbound("Arity") };
    int functor_read_rc = wam_run_predicate(&state, "wam_c_builtin_functor/3", functor_read_args, 3);
    if (functor_read_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "pair") != 0 ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 40;
    }

    WamValue functor_make_args[3] = { val_unbound("Term"), val_atom("edge"), val_int(2) };
    int functor_make_rc = wam_run_predicate(&state, "wam_c_builtin_functor/3", functor_make_args, 3);
    if (functor_make_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_STR ||
        state.H_array[state.A[0].data.ref_addr].tag != VAL_ATOM ||
        strcmp(state.H_array[state.A[0].data.ref_addr].data.atom, "edge/2") != 0) {
        wam_free_state(&state);
        return 50;
    }

    WamValue arg_struct_args[3] = { val_int(2), pair_term, val_unbound("Arg") };
    int arg_struct_rc = wam_run_predicate(&state, "wam_c_builtin_arg/3", arg_struct_args, 3);
    if (arg_struct_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 9) {
        wam_free_state(&state);
        return 60;
    }

    WamValue list_term = make_list_pair(&state, val_atom("head"), val_atom("tail"));
    WamValue arg_list_args[3] = { val_int(1), list_term, val_unbound("Head") };
    int arg_list_rc = wam_run_predicate(&state, "wam_c_builtin_arg/3", arg_list_args, 3);
    if (arg_list_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "head") != 0) {
        wam_free_state(&state);
        return 70;
    }

    WamValue arg_fail_args[3] = { val_int(3), list_term, val_unbound("Out") };
    int arg_fail_rc = wam_run_predicate(&state, "wam_c_builtin_arg/3", arg_fail_args, 3);
    if (arg_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 80;
    }

    WamValue concat_make_args[3] = { val_atom("left"), val_atom("_right"), val_unbound("Whole") };
    int concat_make_rc = wam_run_predicate(&state, "wam_c_builtin_atom_concat/3", concat_make_args, 3);
    if (concat_make_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "left_right") != 0) {
        wam_free_state(&state);
        return 90;
    }

    WamValue concat_prefix_args[3] = { val_unbound("Prefix"), val_atom("_right"), val_atom("left_right") };
    int concat_prefix_rc = wam_run_predicate(&state, "wam_c_builtin_atom_concat/3", concat_prefix_args, 3);
    if (concat_prefix_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_ATOM || strcmp(state.A[0].data.atom, "left") != 0) {
        wam_free_state(&state);
        return 100;
    }

    WamValue concat_suffix_args[3] = { val_atom("left_"), val_unbound("Suffix"), val_atom("left_right") };
    int concat_suffix_rc = wam_run_predicate(&state, "wam_c_builtin_atom_concat/3", concat_suffix_args, 3);
    if (concat_suffix_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "right") != 0) {
        wam_free_state(&state);
        return 110;
    }

    WamValue concat_fail_args[3] = { val_atom("left"), val_atom("_right"), val_atom("left_wrong") };
    int concat_fail_rc = wam_run_predicate(&state, "wam_c_builtin_atom_concat/3", concat_fail_args, 3);
    if (concat_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 120;
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

wam_c_bidirectional_ancestor_smoke_main(
'#include "wam_runtime.h"

void setup_bidirectional_ancestor_5(WamState* state);

static int run_query(WamState *state,
                     const char *cat,
                     const char *root,
                     int *total,
                     int *parents,
                     int *children) {
    WamValue args[5] = {
        val_atom(cat),
        val_atom(root),
        val_unbound("Total"),
        val_unbound("Parents"),
        val_unbound("Children")
    };
    int rc = wam_run_predicate(state, "bidirectional_ancestor/5", args, 5);
    if (rc != 0 || state->P != WAM_HALT) return 0;
    if (state->A[2].tag != VAL_INT ||
        state->A[3].tag != VAL_INT ||
        state->A[4].tag != VAL_INT) {
        return 0;
    }
    *total = state->A[2].data.integer;
    *parents = state->A[3].data.integer;
    *children = state->A[4].data.integer;
    return 1;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_bidirectional_ancestor_5(&state);
    wam_register_category_parent(&state, "orphan", "side");
    wam_register_category_parent(&state, "child", "orphan");
    wam_register_category_parent(&state, "child", "root");
    wam_register_bidirectional_ancestor_kernel(&state, "bidirectional_ancestor/5",
                                               4, 1.0, 2.0, 10.0);

    int total = -1;
    int parents = -1;
    int children = -1;
    if (!run_query(&state, "orphan", "root", &total, &parents, &children) ||
        total != 2 || parents != 1 || children != 1) {
        wam_free_state(&state);
        return 10;
    }
    if (state.bidirectional_min_distance_cache == NULL) {
        wam_free_state(&state);
        return 11;
    }
    int min_hops = -1;
    if (!wam_category_min_parent_hops(&state, "child", "root", &min_hops) ||
        min_hops != 1) {
        wam_free_state(&state);
        return 13;
    }
    if (wam_category_min_parent_hops(&state, "orphan", "root", &min_hops)) {
        wam_free_state(&state);
        return 14;
    }
    if (!wam_category_min_parent_hops(&state, "root", "root", &min_hops) ||
        min_hops != 0) {
        wam_free_state(&state);
        return 15;
    }
    wam_register_category_parent(&state, "fresh_child", "root");
    if (state.bidirectional_min_distance_cache != NULL) {
        wam_free_state(&state);
        return 12;
    }

    wam_register_bidirectional_ancestor_kernel(&state, "bidirectional_ancestor/5",
                                               4, 1.0, 2.0, 2.5);
    WamValue pruned_args[5] = {
        val_atom("orphan"),
        val_atom("root"),
        val_unbound("Total"),
        val_unbound("Parents"),
        val_unbound("Children")
    };
    int pruned_rc = wam_run_predicate(&state, "bidirectional_ancestor/5", pruned_args, 5);
    if (pruned_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_bidirectional_ancestor_csr_smoke_main(IndexPath, ValuesPath, MainCode) :-
    format(atom(MainCode),
'#include "wam_runtime.h"

void setup_bidirectional_ancestor_5(WamState* state);

int main(void) {
    WamState state;
    WamReverseCsrArtifact csr;
    wam_state_init(&state);
    wam_reverse_csr_init(&csr);
    setup_bidirectional_ancestor_5(&state);

    if (!wam_reverse_csr_load(&csr, "~w", "~w")) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 5;
    }

    wam_register_category_id(&state, "orphan", 20);
    wam_register_category_id(&state, "child", 30);
    wam_register_category_id(&state, "root", 40);
    for (int i = 0; i < 130; i++) {
        char atom[32];
        snprintf(atom, sizeof(atom), "extra_%d", i);
        wam_register_category_id(&state, atom, 1000 + i);
    }
    wam_register_category_parent(&state, "child", "root");
    wam_attach_bidirectional_child_csr(&state, &csr);
    wam_register_bidirectional_ancestor_kernel(&state, "bidirectional_ancestor/5",
                                               4, 1.0, 2.0, 10.0);

    WamValue args[5] = {
        val_atom("orphan"),
        val_atom("root"),
        val_unbound("Total"),
        val_unbound("Parents"),
        val_unbound("Children")
    };
    int rc = wam_run_predicate(&state, "bidirectional_ancestor/5", args, 5);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2 ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 1 ||
        state.A[4].tag != VAL_INT || state.A[4].data.integer != 1) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 10;
    }
    if (state.bidirectional_min_distance_cache == NULL) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 11;
    }
    int child_candidates = -1;
    if (!wam_category_child_may_reach_root_within_budget(
            &state, "orphan", "root", 8, 1, 1.0, 2.0, 10.0,
            &child_candidates) ||
        child_candidates != 1) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 13;
    }
    if (wam_category_child_may_reach_root_within_budget(
            &state, "child", "root", 8, 1, 1.0, 2.0, 10.0,
            &child_candidates) ||
        child_candidates != 0) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 14;
    }
    if (wam_category_child_may_reach_root_within_budget(
            &state, "orphan", "root", 8, 1, 1.0, 2.0, 1.5,
            &child_candidates) ||
        child_candidates != 0) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 15;
    }

    wam_attach_bidirectional_child_csr(&state, NULL);
    if (state.bidirectional_min_distance_cache != NULL) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 12;
    }
    WamValue fallback_args[5] = {
        val_atom("orphan"),
        val_atom("root"),
        val_unbound("Total"),
        val_unbound("Parents"),
        val_unbound("Children")
    };
    int fallback_rc = wam_run_predicate(&state, "bidirectional_ancestor/5", fallback_args, 5);
    if (fallback_rc != WAM_HALT) {
        wam_reverse_csr_close(&csr);
        wam_free_state(&state);
        return 20;
    }

    wam_reverse_csr_close(&csr);
    wam_free_state(&state);
    return 0;
}
', [IndexPath, ValuesPath]).

wam_c_reverse_index_setup_smoke_main(
'#include "wam_runtime.h"

void setup_bidirectional_ancestor_5(WamState* state);
bool setup_wam_c_reverse_index_artifacts(WamState* state,
                                         WamReverseCsrArtifact* bidirectional_child_csr);
void teardown_wam_c_reverse_index_artifacts(WamState* state,
                                            WamReverseCsrArtifact* bidirectional_child_csr);

int main(void) {
    WamState state;
    WamReverseCsrArtifact csr;
    wam_state_init(&state);
    setup_bidirectional_ancestor_5(&state);

    if (!setup_wam_c_reverse_index_artifacts(&state, &csr)) {
        wam_free_state(&state);
        return 5;
    }

    wam_register_category_parent(&state, "child", "root");
    wam_register_bidirectional_ancestor_kernel(&state, "bidirectional_ancestor/5",
                                               4, 1.0, 2.0, 10.0);

    WamValue args[5] = {
        val_atom("orphan"),
        val_atom("root"),
        val_unbound("Total"),
        val_unbound("Parents"),
        val_unbound("Children")
    };
    int rc = wam_run_predicate(&state, "bidirectional_ancestor/5", args, 5);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2 ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 1 ||
        state.A[4].tag != VAL_INT || state.A[4].data.integer != 1) {
        teardown_wam_c_reverse_index_artifacts(&state, &csr);
        wam_free_state(&state);
        return 10;
    }

    teardown_wam_c_reverse_index_artifacts(&state, &csr);
    WamValue fallback_args[5] = {
        val_atom("orphan"),
        val_atom("root"),
        val_unbound("Total"),
        val_unbound("Parents"),
        val_unbound("Children")
    };
    int fallback_rc = wam_run_predicate(&state, "bidirectional_ancestor/5", fallback_args, 5);
    if (fallback_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_closure_smoke_main(
'#include "wam_runtime.h"

void setup_tc_ancestor_2(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_ancestor_2(&state);
    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");
    wam_register_transitive_edge(&state, "bob", "pat");
    wam_register_transitive_closure_kernel(&state, "tc_ancestor/2");

    WamValue recursive_args[2] = {
        val_atom("tom"),
        val_atom("ann")
    };
    int recursive_rc = wam_run_predicate(&state, "tc_ancestor/2", recursive_args, 2);
    if (recursive_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[2] = {
        val_atom("tom"),
        val_atom("bob")
    };
    int direct_rc = wam_run_predicate(&state, "tc_ancestor/2", direct_args, 2);
    if (direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue output_args[2] = {
        val_atom("bob"),
        val_unbound("Target")
    };
    int output_rc = wam_run_predicate(&state, "tc_ancestor/2", output_args, 2);
    if (output_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "ann") != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue fail_args[2] = {
        val_atom("ann"),
        val_atom("tom")
    };
    int fail_rc = wam_run_predicate(&state, "tc_ancestor/2", fail_args, 2);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_distance_smoke_main(
'#include "wam_runtime.h"

void setup_tc_distance_3(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_distance_3(&state);
    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");
    wam_register_transitive_edge(&state, "bob", "pat");
    wam_register_transitive_distance_kernel(&state, "tc_distance/3");

    WamValue recursive_args[3] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Distance")
    };
    int recursive_rc = wam_run_predicate(&state, "tc_distance/3", recursive_args, 3);
    if (recursive_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[3] = {
        val_atom("tom"),
        val_atom("bob"),
        val_int(1)
    };
    int direct_rc = wam_run_predicate(&state, "tc_distance/3", direct_args, 3);
    if (direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue output_args[3] = {
        val_atom("bob"),
        val_unbound("Target"),
        val_unbound("Distance")
    };
    int output_rc = wam_run_predicate(&state, "tc_distance/3", output_args, 3);
    if (output_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "ann") != 0 ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 1) {
        wam_free_state(&state);
        return 30;
    }

    WamValue distance_filter_args[3] = {
        val_atom("tom"),
        val_unbound("Target"),
        val_int(2)
    };
    int distance_filter_rc = wam_run_predicate(&state, "tc_distance/3", distance_filter_args, 3);
    if (distance_filter_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "ann") != 0) {
        wam_free_state(&state);
        return 40;
    }

    WamValue fail_args[3] = {
        val_atom("ann"),
        val_atom("tom"),
        val_unbound("Distance")
    };
    int fail_rc = wam_run_predicate(&state, "tc_distance/3", fail_args, 3);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 50;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_parent_distance_smoke_main(
'#include "wam_runtime.h"

void setup_tc_parent_distance_4(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_parent_distance_4(&state);
    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");
    wam_register_transitive_edge(&state, "bob", "pat");
    wam_register_transitive_parent_distance_kernel(&state, "tc_parent_distance/4");

    WamValue recursive_args[4] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int recursive_rc = wam_run_predicate(&state, "tc_parent_distance/4", recursive_args, 4);
    if (recursive_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "bob") != 0 ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[4] = {
        val_atom("tom"),
        val_atom("bob"),
        val_atom("tom"),
        val_int(1)
    };
    int direct_rc = wam_run_predicate(&state, "tc_parent_distance/4", direct_args, 4);
    if (direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue output_args[4] = {
        val_atom("bob"),
        val_unbound("Target"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int output_rc = wam_run_predicate(&state, "tc_parent_distance/4", output_args, 4);
    if (output_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "ann") != 0 ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "bob") != 0 ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 1) {
        wam_free_state(&state);
        return 30;
    }

    WamValue fail_args[4] = {
        val_atom("ann"),
        val_atom("tom"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int fail_rc = wam_run_predicate(&state, "tc_parent_distance/4", fail_args, 4);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_step_parent_distance_smoke_main(
'#include "wam_runtime.h"

void setup_tc_step_parent_distance_5(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_step_parent_distance_5(&state);
    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");
    wam_register_transitive_edge(&state, "bob", "pat");
    wam_register_transitive_step_parent_distance_kernel(&state, "tc_step_parent_distance/5");

    WamValue recursive_args[5] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Step"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int recursive_rc = wam_run_predicate(&state, "tc_step_parent_distance/5", recursive_args, 5);
    if (recursive_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "bob") != 0 ||
        state.A[3].tag != VAL_ATOM || strcmp(state.A[3].data.atom, "bob") != 0 ||
        state.A[4].tag != VAL_INT || state.A[4].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[5] = {
        val_atom("tom"),
        val_atom("bob"),
        val_atom("bob"),
        val_atom("tom"),
        val_int(1)
    };
    int direct_rc = wam_run_predicate(&state, "tc_step_parent_distance/5", direct_args, 5);
    if (direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue output_args[5] = {
        val_atom("bob"),
        val_unbound("Target"),
        val_unbound("Step"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int output_rc = wam_run_predicate(&state, "tc_step_parent_distance/5", output_args, 5);
    if (output_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "ann") != 0 ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "ann") != 0 ||
        state.A[3].tag != VAL_ATOM || strcmp(state.A[3].data.atom, "bob") != 0 ||
        state.A[4].tag != VAL_INT || state.A[4].data.integer != 1) {
        wam_free_state(&state);
        return 30;
    }

    WamValue fail_args[5] = {
        val_atom("ann"),
        val_atom("tom"),
        val_unbound("Step"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int fail_rc = wam_run_predicate(&state, "tc_step_parent_distance/5", fail_args, 5);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_weighted_shortest_path_smoke_main(
'#include "wam_runtime.h"

void setup_weighted_path_3(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_weighted_path_3(&state);
    wam_register_weighted_edge(&state, "tom", "bob", 5);
    wam_register_weighted_edge(&state, "tom", "eve", 1);
    wam_register_weighted_edge(&state, "eve", "ann", 1);
    wam_register_weighted_edge(&state, "bob", "ann", 1);
    wam_register_weighted_edge(&state, "flo", "mid", 0.5);
    wam_register_weighted_edge(&state, "mid", "fin", 1.0);
    wam_register_weighted_shortest_path_kernel(&state, "weighted_path/3");

    WamValue shortest_args[3] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Weight")
    };
    int shortest_rc = wam_run_predicate(&state, "weighted_path/3", shortest_args, 3);
    if (shortest_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[3] = {
        val_atom("bob"),
        val_atom("ann"),
        val_int(1)
    };
    int direct_rc = wam_run_predicate(&state, "weighted_path/3", direct_args, 3);
    if (direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue output_args[3] = {
        val_atom("tom"),
        val_unbound("Target"),
        val_unbound("Weight")
    };
    int output_rc = wam_run_predicate(&state, "weighted_path/3", output_args, 3);
    if (output_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "eve") != 0 ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 1) {
        wam_free_state(&state);
        return 30;
    }

    WamValue float_args[3] = {
        val_atom("flo"),
        val_atom("fin"),
        val_unbound("Weight")
    };
    int float_rc = wam_run_predicate(&state, "weighted_path/3", float_args, 3);
    if (float_rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_FLOAT || state.A[2].data.floating != 1.5) {
        wam_free_state(&state);
        return 35;
    }

    WamValue float_direct_args[3] = {
        val_atom("flo"),
        val_atom("fin"),
        val_float(1.5)
    };
    int float_direct_rc = wam_run_predicate(&state, "weighted_path/3", float_direct_args, 3);
    if (float_direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 36;
    }

    WamValue fail_args[3] = {
        val_atom("ann"),
        val_atom("tom"),
        val_unbound("Weight")
    };
    int fail_rc = wam_run_predicate(&state, "weighted_path/3", fail_args, 3);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_astar_shortest_path_smoke_main(
'#include "wam_runtime.h"

void setup_astar_path_4(WamState* state);

static void register_astar_edges(WamState *state) {
    wam_register_weighted_edge(state, "tom", "bob", 5);
    wam_register_weighted_edge(state, "tom", "eve", 1);
    wam_register_weighted_edge(state, "eve", "ann", 1);
    wam_register_weighted_edge(state, "bob", "ann", 1);
    wam_register_weighted_edge(state, "flo", "mid", 0.5);
    wam_register_weighted_edge(state, "mid", "fin", 1.0);
    wam_register_direct_distance_edge(state, "tom", "ann", 2);
    wam_register_direct_distance_edge(state, "eve", "ann", 1);
    wam_register_direct_distance_edge(state, "bob", "ann", 1);
    wam_register_direct_distance_edge(state, "flo", "fin", 1.5);
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_astar_path_4(&state);
    register_astar_edges(&state);
    wam_register_astar_shortest_path_kernel(&state, "astar_path/4");

    WamValue shortest_args[4] = {
        val_atom("tom"),
        val_atom("ann"),
        val_int(5),
        val_unbound("Weight")
    };
    int shortest_rc = wam_run_predicate(&state, "astar_path/4", shortest_args, 4);
    if (shortest_rc != 0 || state.P != WAM_HALT ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    WamValue direct_args[4] = {
        val_atom("bob"),
        val_atom("ann"),
        val_int(5),
        val_int(1)
    };
    int direct_rc = wam_run_predicate(&state, "astar_path/4", direct_args, 4);
    if (direct_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue float_args[4] = {
        val_atom("flo"),
        val_atom("fin"),
        val_int(5),
        val_unbound("Weight")
    };
    int float_rc = wam_run_predicate(&state, "astar_path/4", float_args, 4);
    if (float_rc != 0 || state.P != WAM_HALT ||
        state.A[3].tag != VAL_FLOAT || state.A[3].data.floating != 1.5) {
        wam_free_state(&state);
        return 25;
    }

    WamValue bad_dim_args[4] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Dim"),
        val_unbound("Weight")
    };
    int bad_dim_rc = wam_run_predicate(&state, "astar_path/4", bad_dim_args, 4);
    if (bad_dim_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    WamValue fail_args[4] = {
        val_atom("ann"),
        val_atom("tom"),
        val_int(5),
        val_unbound("Weight")
    };
    int fail_rc = wam_run_predicate(&state, "astar_path/4", fail_args, 4);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
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

wam_c_transitive_closure_detector_smoke_main(
'#include "wam_runtime.h"

void setup_tc_ancestor_2(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_ancestor_2(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");

    WamValue args[2] = {
        val_atom("tom"),
        val_atom("ann")
    };
    int rc = wam_run_predicate(&state, "tc_ancestor/2", args, 2);
    if (rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_distance_detector_smoke_main(
'#include "wam_runtime.h"

void setup_tc_distance_3(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_distance_3(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");

    WamValue args[3] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Distance")
    };
    int rc = wam_run_predicate(&state, "tc_distance/3", args, 3);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_parent_distance_detector_smoke_main(
'#include "wam_runtime.h"

void setup_tc_parent_distance_4(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_parent_distance_4(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");

    WamValue args[4] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int rc = wam_run_predicate(&state, "tc_parent_distance/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "bob") != 0 ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_transitive_step_parent_distance_detector_smoke_main(
'#include "wam_runtime.h"

void setup_tc_step_parent_distance_5(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_tc_step_parent_distance_5(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_transitive_edge(&state, "tom", "bob");
    wam_register_transitive_edge(&state, "bob", "ann");

    WamValue args[5] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Step"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int rc = wam_run_predicate(&state, "tc_step_parent_distance/5", args, 5);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_ATOM || strcmp(state.A[2].data.atom, "bob") != 0 ||
        state.A[3].tag != VAL_ATOM || strcmp(state.A[3].data.atom, "bob") != 0 ||
        state.A[4].tag != VAL_INT || state.A[4].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_weighted_shortest_path_detector_smoke_main(
'#include "wam_runtime.h"

void setup_weighted_path_3(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_weighted_path_3(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_weighted_edge(&state, "tom", "bob", 5);
    wam_register_weighted_edge(&state, "tom", "eve", 1);
    wam_register_weighted_edge(&state, "eve", "ann", 1);
    wam_register_weighted_edge(&state, "bob", "ann", 1);

    WamValue args[3] = {
        val_atom("tom"),
        val_atom("ann"),
        val_unbound("Weight")
    };
    int rc = wam_run_predicate(&state, "weighted_path/3", args, 3);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[2].tag != VAL_INT || state.A[2].data.integer != 2) {
        wam_free_state(&state);
        return 10;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_astar_shortest_path_detector_smoke_main(
'#include "wam_runtime.h"

void setup_astar_path_4(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_astar_path_4(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_weighted_edge(&state, "tom", "bob", 5);
    wam_register_weighted_edge(&state, "tom", "eve", 1);
    wam_register_weighted_edge(&state, "eve", "ann", 1);
    wam_register_weighted_edge(&state, "bob", "ann", 1);
    wam_register_direct_distance_edge(&state, "tom", "ann", 2);
    wam_register_direct_distance_edge(&state, "eve", "ann", 1);
    wam_register_direct_distance_edge(&state, "bob", "ann", 1);

    WamValue args[4] = {
        val_atom("tom"),
        val_atom("ann"),
        val_int(5),
        val_unbound("Weight")
    };
    int rc = wam_run_predicate(&state, "astar_path/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT ||
        state.A[3].tag != VAL_INT || state.A[3].data.integer != 2) {
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
    if (state.category_edge_count != source.edge_count) {
        wam_free_state(&state);
        wam_fact_source_close(&source);
        return 25;
    }
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

wam_c_reverse_csr_smoke_main(IndexPath, ValuesPath, MainCode) :-
    format(atom(MainCode),
'#include "wam_runtime.h"

int main(void) {
    WamReverseCsrArtifact csr;
    int children[4] = {0, 0, 0, 0};
    int count = 0;

    wam_reverse_csr_init(&csr);
    if (!wam_reverse_csr_load(&csr, "~w", "~w")) {
        return 10;
    }

    count = wam_reverse_csr_lookup_children(&csr, 20, children, 4);
    if (count != 3 || children[0] != 10 || children[1] != 12 || children[2] != 13) {
        wam_reverse_csr_close(&csr);
        return 20;
    }

    children[0] = 0;
    children[1] = 0;
    count = wam_reverse_csr_lookup_children(&csr, 20, children, 2);
    if (count != 3 || children[0] != 10 || children[1] != 12) {
        wam_reverse_csr_close(&csr);
        return 30;
    }

    count = wam_reverse_csr_lookup_children(&csr, 30, children, 4);
    if (count != 1 || children[0] != 11) {
        wam_reverse_csr_close(&csr);
        return 40;
    }

    count = wam_reverse_csr_lookup_children(&csr, 99, children, 4);
    if (count != 0) {
        wam_reverse_csr_close(&csr);
        return 50;
    }

    wam_reverse_csr_close(&csr);
    return 0;
}
', [IndexPath, ValuesPath]).

wam_c_reverse_csr_lmdb_offset_smoke_main(ValuesPath, OffsetEnvPath, MainCode) :-
    format(atom(MainCode),
'#include "wam_runtime.h"
#include <sys/stat.h>

static void encode_i32(unsigned char *out, int value) {
    unsigned int v = (unsigned int)value;
    out[0] = (unsigned char)(v & 0xffU);
    out[1] = (unsigned char)((v >> 8) & 0xffU);
    out[2] = (unsigned char)((v >> 16) & 0xffU);
    out[3] = (unsigned char)((v >> 24) & 0xffU);
}

static void encode_offset_record(unsigned char *out, unsigned long long offset_edges, unsigned int child_count) {
    for (int i = 0; i < 8; i++) {
        out[i] = (unsigned char)((offset_edges >> (8 * i)) & 0xffULL);
    }
    out[8] = (unsigned char)(child_count & 0xffU);
    out[9] = (unsigned char)((child_count >> 8) & 0xffU);
    out[10] = (unsigned char)((child_count >> 16) & 0xffU);
    out[11] = (unsigned char)((child_count >> 24) & 0xffU);
}

static int put_offset(MDB_txn *txn, MDB_dbi dbi, int parent, unsigned long long offset_edges, unsigned int child_count) {
    unsigned char key_bytes[4];
    unsigned char data_bytes[12];
    MDB_val key;
    MDB_val data;
    encode_i32(key_bytes, parent);
    encode_offset_record(data_bytes, offset_edges, child_count);
    key.mv_size = sizeof(key_bytes);
    key.mv_data = key_bytes;
    data.mv_size = sizeof(data_bytes);
    data.mv_data = data_bytes;
    return mdb_put(txn, dbi, &key, &data, 0);
}

static int seed_offsets(const char *path) {
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = mkdir(path, 0777);
    (void)rc;
    rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) return rc;
    rc = mdb_env_set_maxdbs(env, 2);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_env_set_mapsize(env, 1048576);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_env_open(env, path, 0, 0664);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_txn_begin(env, NULL, 0, &txn);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return rc; }
    rc = mdb_dbi_open(txn, "offsets", MDB_CREATE, &dbi);
    if (rc == MDB_SUCCESS) rc = put_offset(txn, dbi, 20, 0, 3);
    if (rc == MDB_SUCCESS) rc = put_offset(txn, dbi, 30, 3, 1);
    if (rc == MDB_SUCCESS) rc = mdb_txn_commit(txn);
    else mdb_txn_abort(txn);
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
    return rc;
}

int main(void) {
    WamReverseCsrArtifact csr;
    int children[4] = {0, 0, 0, 0};
    int count = 0;

    if (seed_offsets("~w") != MDB_SUCCESS) return 5;

    wam_reverse_csr_init(&csr);
    if (!wam_reverse_csr_load_lmdb_offset(&csr, "~w", "~w", "offsets")) {
        return 10;
    }

    count = wam_reverse_csr_lookup_children(&csr, 20, children, 4);
    if (count != 3 || children[0] != 10 || children[1] != 12 || children[2] != 13) {
        wam_reverse_csr_close(&csr);
        return 20;
    }

    children[0] = 0;
    children[1] = 0;
    count = wam_reverse_csr_lookup_children(&csr, 20, children, 2);
    if (count != 3 || children[0] != 10 || children[1] != 12) {
        wam_reverse_csr_close(&csr);
        return 30;
    }

    count = wam_reverse_csr_lookup_children(&csr, 30, children, 4);
    if (count != 1 || children[0] != 11) {
        wam_reverse_csr_close(&csr);
        return 40;
    }

    count = wam_reverse_csr_lookup_children(&csr, 99, children, 4);
    if (count != 0) {
        wam_reverse_csr_close(&csr);
        return 50;
    }

    wam_reverse_csr_close(&csr);
    return 0;
}
', [OffsetEnvPath, ValuesPath, OffsetEnvPath]).

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
void setup_wam_c_body_projected_2(WamState* state);
void setup_wam_c_body_ignored_output_1(WamState* state);
void setup_wam_c_body_repeated_projection_1(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_lowered_wam_c_helpers(&state);

    setup_wam_c_body_alias_2(&state);
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

    setup_wam_c_body_projected_2(&state);
    WamValue projected_args[2] = { val_atom("b"), val_unbound("Projected") };
    int projected_rc = wam_run_predicate(&state, "wam_c_body_projected/2", projected_args, 2);
    if (projected_rc != 0 || state.P != WAM_HALT ||
        state.A[1].tag != VAL_ATOM || strcmp(state.A[1].data.atom, "a") != 0) {
        wam_free_state(&state);
        return 40;
    }

    WamValue projected_ground_args[2] = { val_atom("b"), val_atom("a") };
    int projected_ground_rc = wam_run_predicate(&state, "wam_c_body_projected/2", projected_ground_args, 2);
    if (projected_ground_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 50;
    }

    WamValue projected_fail_args[2] = { val_atom("b"), val_atom("c") };
    int projected_fail_rc = wam_run_predicate(&state, "wam_c_body_projected/2", projected_fail_args, 2);
    if (projected_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 60;
    }

    setup_wam_c_body_ignored_output_1(&state);
    WamValue ignored_args[1] = { val_atom("a") };
    int ignored_rc = wam_run_predicate(&state, "wam_c_body_ignored_output/1", ignored_args, 1);
    if (ignored_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 70;
    }

    WamValue ignored_fail_args[1] = { val_atom("z") };
    int ignored_fail_rc = wam_run_predicate(&state, "wam_c_body_ignored_output/1", ignored_fail_args, 1);
    if (ignored_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 80;
    }

    setup_wam_c_body_repeated_projection_1(&state);
    WamValue repeated_args[1] = { val_atom("a") };
    int repeated_rc = wam_run_predicate(&state, "wam_c_body_repeated_projection/1", repeated_args, 1);
    if (repeated_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 90;
    }

    WamValue repeated_fail_args[1] = { val_atom("b") };
    int repeated_fail_rc = wam_run_predicate(&state, "wam_c_body_repeated_projection/1", repeated_fail_args, 1);
    if (repeated_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 100;
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

wam_c_lowered_comparison_filter_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_filter_score_2(WamState* state);
void setup_wam_c_filter_small_1(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_filter_score_2(&state);
    setup_wam_c_filter_small_1(&state);
    setup_lowered_wam_c_helpers(&state);

    WamValue ok_args[1] = { val_unbound("Out") };
    int ok_rc = wam_run_predicate(&state, "wam_c_filter_small/1", ok_args, 1);
    if (ok_rc != 0 || state.P != WAM_HALT ||
        state.A[0].tag != VAL_ATOM || strcmp(state.A[0].data.atom, "a") != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue ground_args[1] = { val_atom("b") };
    int ground_rc = wam_run_predicate(&state, "wam_c_filter_small/1", ground_args, 1);
    if (ground_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue fail_args[1] = { val_atom("c") };
    int fail_rc = wam_run_predicate(&state, "wam_c_filter_small/1", fail_args, 1);
    if (fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_lowered_repeat_filter_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_repeat_edge_3(WamState* state);
void setup_wam_c_repeat_score_3(WamState* state);
void setup_wam_c_repeat_keep_1(WamState* state);
void setup_wam_c_repeat_small_1(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

static int expect_success(WamState* state, const char* pred, const char* atom) {
    WamValue args[1] = { val_atom(atom) };
    int rc = wam_run_predicate(state, pred, args, 1);
    return rc == 0 && state->P == WAM_HALT;
}

static int expect_failure(WamState* state, const char* pred, const char* atom) {
    WamValue args[1] = { val_atom(atom) };
    int rc = wam_run_predicate(state, pred, args, 1);
    return rc == WAM_HALT;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_repeat_edge_3(&state);
    setup_wam_c_repeat_score_3(&state);
    setup_wam_c_repeat_keep_1(&state);
    setup_wam_c_repeat_small_1(&state);
    setup_lowered_wam_c_helpers(&state);

    if (!expect_success(&state, "wam_c_repeat_keep/1", "a")) {
        wam_free_state(&state);
        return 10;
    }
    if (!expect_failure(&state, "wam_c_repeat_keep/1", "c")) {
        wam_free_state(&state);
        return 20;
    }
    if (!expect_success(&state, "wam_c_repeat_small/1", "a")) {
        wam_free_state(&state);
        return 30;
    }
    if (!expect_failure(&state, "wam_c_repeat_small/1", "c")) {
        wam_free_state(&state);
        return 40;
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

wam_c_real_term_builtin_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_real_term_builtins_2(WamState* state);

static WamValue make_binary_struct(WamState *state, const char *functor, WamValue left, WamValue right) {
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(functor);
    state->H_array[state->H++] = left;
    state->H_array[state->H++] = right;
    return term;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_real_term_builtins_2(&state);

    WamValue pair_term = make_binary_struct(&state, "pair/2", val_atom("left"), val_atom("right"));
    WamValue ok_args[2] = { pair_term, val_atom("pair_right") };
    int ok_rc = wam_run_predicate(&state, "wam_c_real_term_builtins/2", ok_args, 2);
    if (ok_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue unary_term = make_binary_struct(&state, "pair/2", val_atom("left"), val_atom("wrong"));
    WamValue fail_args[2] = { unary_term, val_atom("pair_right") };
    int fail_rc = wam_run_predicate(&state, "wam_c_real_term_builtins/2", fail_args, 2);
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

wam_c_real_control_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_ctrl_known_1(WamState* state);
void setup_wam_c_ctrl_neg_2(WamState* state);
void setup_wam_c_ctrl_if_known_2(WamState* state);
void setup_wam_c_ctrl_if_missing_2(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_ctrl_known_1(&state);
    setup_wam_c_ctrl_neg_2(&state);
    setup_wam_c_ctrl_if_known_2(&state);
    setup_wam_c_ctrl_if_missing_2(&state);

    WamValue neg_success_args[2] = { val_atom("b"), val_atom("ok") };
    int neg_success_rc = wam_run_predicate(&state, "wam_c_ctrl_neg/2", neg_success_args, 2);
    if (neg_success_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 10;
    }

    WamValue neg_fail_args[2] = { val_atom("a"), val_atom("ok") };
    int neg_fail_rc = wam_run_predicate(&state, "wam_c_ctrl_neg/2", neg_fail_args, 2);
    if (neg_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 20;
    }

    WamValue if_known_success_args[2] = { val_atom("a"), val_atom("yes") };
    int if_known_success_rc = wam_run_predicate(&state, "wam_c_ctrl_if_known/2", if_known_success_args, 2);
    if (if_known_success_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 30;
    }

    WamValue if_known_fail_args[2] = { val_atom("b"), val_atom("yes") };
    int if_known_fail_rc = wam_run_predicate(&state, "wam_c_ctrl_if_known/2", if_known_fail_args, 2);
    if (if_known_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    WamValue if_missing_success_args[2] = { val_atom("b"), val_atom("no") };
    int if_missing_success_rc = wam_run_predicate(&state, "wam_c_ctrl_if_missing/2", if_missing_success_args, 2);
    if (if_missing_success_rc != 0 || state.P != WAM_HALT) {
        wam_free_state(&state);
        return 50;
    }

    WamValue if_missing_fail_args[2] = { val_atom("a"), val_atom("no") };
    int if_missing_fail_rc = wam_run_predicate(&state, "wam_c_ctrl_if_missing/2", if_missing_fail_args, 2);
    if (if_missing_fail_rc != WAM_HALT) {
        wam_free_state(&state);
        return 60;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_precise_ite_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_precise_choice_1(WamState* state);
void setup_wam_c_precise_then_2(WamState* state);
void setup_wam_c_precise_else_2(WamState* state);
void setup_wam_c_precise_scope_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_precise_choice_1(&state);
    setup_wam_c_precise_then_2(&state);
    setup_wam_c_precise_else_2(&state);
    setup_wam_c_precise_scope_1(&state);

    WamValue then_success_args[2] = { val_atom("a"), val_atom("then") };
    int then_success_rc = wam_run_predicate(&state, "wam_c_precise_then/2", then_success_args, 2);
    if (then_success_rc != 0 || state.P != WAM_HALT || state.B != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue then_fail_args[2] = { val_atom("b"), val_atom("then") };
    int then_fail_rc = wam_run_predicate(&state, "wam_c_precise_then/2", then_fail_args, 2);
    if (then_fail_rc != WAM_HALT || state.B != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue else_success_args[2] = { val_atom("b"), val_atom("else") };
    int else_success_rc = wam_run_predicate(&state, "wam_c_precise_else/2", else_success_args, 2);
    if (else_success_rc != 0 || state.P != WAM_HALT || state.B != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue else_fail_args[2] = { val_atom("a"), val_atom("else") };
    int else_fail_rc = wam_run_predicate(&state, "wam_c_precise_else/2", else_fail_args, 2);
    if (else_fail_rc != WAM_HALT || state.B != 0) {
        wam_free_state(&state);
        return 40;
    }

    WamValue scope_fail_args[1] = { val_atom("a") };
    int scope_fail_rc = wam_run_predicate(&state, "wam_c_precise_scope/1", scope_fail_args, 1);
    if (scope_fail_rc != WAM_HALT || state.B != 0) {
        wam_free_state(&state);
        return 50;
    }

    WamValue scope_success_args[1] = { val_atom("b") };
    int scope_success_rc = wam_run_predicate(&state, "wam_c_precise_scope/1", scope_success_args, 1);
    if (scope_success_rc != 0 || state.P != WAM_HALT || state.B != 0) {
        wam_free_state(&state);
        return 60;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_explicit_cut_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_cut_choice_1(WamState* state);
void setup_wam_c_inner_cut_0(WamState* state);
void setup_wam_c_outer_cut_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_cut_choice_1(&state);
    setup_wam_c_inner_cut_0(&state);
    setup_wam_c_outer_cut_1(&state);

    int inner_rc = wam_run_predicate(&state, "wam_c_inner_cut/0", NULL, 0);
    if (inner_rc != 0 || state.P != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue outer_args[1] = { val_atom("ok") };
    int outer_rc = wam_run_predicate(&state, "wam_c_outer_cut/1", outer_args, 1);
    if (outer_rc != 0 || state.P != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue outer_fail_args[1] = { val_atom("bad") };
    int outer_fail_rc = wam_run_predicate(&state, "wam_c_outer_cut/1", outer_fail_args, 1);
    if (outer_fail_rc != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_forall_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_forall_num_1(WamState* state);
void setup_wam_c_forall_positive_1(WamState* state);
void setup_wam_c_forall_only_two_1(WamState* state);
void setup_wam_c_forall_empty_1(WamState* state);
void setup_wam_c_forall_all_1(WamState* state);
void setup_wam_c_forall_fail_1(WamState* state);
void setup_wam_c_forall_subset_1(WamState* state);
void setup_wam_c_forall_empty_ok_1(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_forall_num_1(&state);
    setup_wam_c_forall_positive_1(&state);
    setup_wam_c_forall_only_two_1(&state);
    setup_wam_c_forall_empty_1(&state);
    setup_wam_c_forall_all_1(&state);
    setup_wam_c_forall_fail_1(&state);
    setup_wam_c_forall_subset_1(&state);
    setup_wam_c_forall_empty_ok_1(&state);

    WamValue all_args[1] = { val_atom("ok") };
    int all_rc = wam_run_predicate(&state, "wam_c_forall_all/1", all_args, 1);
    if (all_rc != 0 || state.P != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue fail_args[1] = { val_atom("ok") };
    int fail_rc = wam_run_predicate(&state, "wam_c_forall_fail/1", fail_args, 1);
    if (fail_rc != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue subset_args[1] = { val_atom("ok") };
    int subset_rc = wam_run_predicate(&state, "wam_c_forall_subset/1", subset_args, 1);
    if (subset_rc != 0 || state.P != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue empty_args[1] = { val_atom("ok") };
    int empty_rc = wam_run_predicate(&state, "wam_c_forall_empty_ok/1", empty_args, 1);
    if (empty_rc != 0 || state.P != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 40;
    }

    WamValue bad_args[1] = { val_atom("bad") };
    int bad_rc = wam_run_predicate(&state, "wam_c_forall_all/1", bad_args, 1);
    if (bad_rc != WAM_HALT || state.B != 0 || state.call_base_top != 0) {
        wam_free_state(&state);
        return 50;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_findall_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_findall_item_1(WamState* state);
void setup_wam_c_findall_none_1(WamState* state);
void setup_wam_c_findall_all_1(WamState* state);
void setup_wam_c_findall_empty_1(WamState* state);
void setup_wam_c_findall_nested_inner_1(WamState* state);
void setup_wam_c_findall_nested_outer_1(WamState* state);
void setup_wam_c_findall_inline_nested_1(WamState* state);
void setup_wam_c_findall_struct_template_1(WamState* state);
void setup_wam_c_findall_list_template_1(WamState* state);

static bool expect_atom(WamState *state, WamValue value, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, &value);
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool expect_atom_slot(WamState *state, WamValue *slot, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, slot);
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool expect_atom_list2(WamState *state, WamValue value,
                              const char *first, const char *second) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_atom_slot(state, &state->H_array[tail1_addr], second) &&
           tail2->tag == VAL_ATOM &&
           strcmp(tail2->data.atom, "[]") == 0;
}

static bool expect_atom_list1(WamState *state, WamValue value,
                              const char *first) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           tail->tag == VAL_ATOM &&
           strcmp(tail->data.atom, "[]") == 0;
}

static bool expect_pair_atom_singleton(WamState *state, WamValue value,
                                       const char *atom) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag != VAL_STR) return false;
    WamValue *functor = &state->H_array[cell->data.ref_addr];
    if (functor->tag != VAL_ATOM || strcmp(functor->data.atom, "pair/2") != 0) {
        return false;
    }
    return expect_atom_slot(state, &state->H_array[cell->data.ref_addr + 1], atom) &&
           expect_atom_list1(state, state->H_array[cell->data.ref_addr + 2], atom);
}

static bool expect_pair_list2(WamState *state, WamValue value,
                              const char *first, const char *second) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    return expect_pair_atom_singleton(state, state->H_array[cell_addr], first) &&
           expect_pair_atom_singleton(state, state->H_array[tail1_addr], second) &&
           tail2->tag == VAL_ATOM &&
           strcmp(tail2->data.atom, "[]") == 0;
}

static bool expect_nested_atom_list2(WamState *state, WamValue value,
                                     const char *first, const char *second) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    return expect_atom_list2(state, state->H_array[cell_addr], first, first) &&
           expect_atom_list2(state, state->H_array[tail1_addr], second, second) &&
           tail2->tag == VAL_ATOM &&
           strcmp(tail2->data.atom, "[]") == 0;
}

static WamValue make_atom_list2(WamState *state,
                                const char *first,
                                const char *second) {
    WamValue tail = val_atom("[]");
    int base_second = state->H;
    state->H_array[state->H++] = val_atom(second);
    state->H_array[state->H++] = tail;
    tail.tag = VAL_LIST;
    tail.data.ref_addr = base_second;
    int base_first = state->H;
    state->H_array[state->H++] = val_atom(first);
    state->H_array[state->H++] = tail;
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = base_first;
    return list;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_findall_item_1(&state);
    setup_wam_c_findall_none_1(&state);
    setup_wam_c_findall_all_1(&state);
    setup_wam_c_findall_empty_1(&state);
    setup_wam_c_findall_nested_inner_1(&state);
    setup_wam_c_findall_nested_outer_1(&state);
    setup_wam_c_findall_inline_nested_1(&state);
    setup_wam_c_findall_struct_template_1(&state);
    setup_wam_c_findall_list_template_1(&state);

    WamValue all_args[1] = { val_unbound("All") };
    int all_rc = wam_run_predicate(&state, "wam_c_findall_all/1", all_args, 1);
    if (all_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue empty_args[1] = { val_unbound("Empty") };
    int empty_rc = wam_run_predicate(&state, "wam_c_findall_empty/1", empty_args, 1);
    if (empty_rc != 0 || state.P != WAM_HALT ||
        !expect_atom(&state, state.A[0], "[]") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue expected = make_atom_list2(&state, "a", "b");
    WamValue bound_args[1] = { expected };
    int bound_rc = wam_run_predicate(&state, "wam_c_findall_all/1", bound_args, 1);
    if (bound_rc != 0 || state.P != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue nested_args[1] = { val_unbound("Nested") };
    int nested_rc = wam_run_predicate(&state, "wam_c_findall_nested_outer/1", nested_args, 1);
    if (nested_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 40;
    }

    WamValue inline_nested_args[1] = { val_unbound("InlineNested") };
    int inline_nested_rc = wam_run_predicate(&state, "wam_c_findall_inline_nested/1", inline_nested_args, 1);
    if (inline_nested_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 45;
    }

    WamValue struct_args[1] = { val_unbound("Struct") };
    int struct_rc = wam_run_predicate(&state, "wam_c_findall_struct_template/1", struct_args, 1);
    if (struct_rc != 0 || state.P != WAM_HALT ||
        !expect_pair_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 50;
    }

    WamValue list_args[1] = { val_unbound("List") };
    int list_rc = wam_run_predicate(&state, "wam_c_findall_list_template/1", list_args, 1);
    if (list_rc != 0 || state.P != WAM_HALT ||
        !expect_nested_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 60;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_bagof_setof_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_bagset_item_1(WamState* state);
void setup_wam_c_bagset_none_1(WamState* state);
void setup_wam_c_bagset_bag_1(WamState* state);
void setup_wam_c_bagset_set_1(WamState* state);
void setup_wam_c_bagset_bag_empty_1(WamState* state);
void setup_wam_c_bagset_set_empty_1(WamState* state);

static bool expect_atom_slot(WamState *state, WamValue *slot, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, slot);
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool expect_nil(WamValue *cell) {
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, "[]") == 0;
}

static bool expect_atom_list2(WamState *state, WamValue value,
                              const char *first, const char *second) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_atom_slot(state, &state->H_array[tail1_addr], second) &&
           expect_nil(tail2);
}

static bool expect_atom_list4(WamState *state, WamValue value,
                              const char *first,
                              const char *second,
                              const char *third,
                              const char *fourth) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    int tail2_addr = wam_cons_head_addr(state, tail2);
    if (tail2_addr < 0) return false;
    WamValue *tail3 = wam_deref_ptr(state, &state->H_array[tail2_addr + 1]);
    int tail3_addr = wam_cons_head_addr(state, tail3);
    if (tail3_addr < 0) return false;
    WamValue *tail4 = wam_deref_ptr(state, &state->H_array[tail3_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_atom_slot(state, &state->H_array[tail1_addr], second) &&
           expect_atom_slot(state, &state->H_array[tail2_addr], third) &&
           expect_atom_slot(state, &state->H_array[tail3_addr], fourth) &&
           expect_nil(tail4);
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_bagset_item_1(&state);
    setup_wam_c_bagset_none_1(&state);
    setup_wam_c_bagset_bag_1(&state);
    setup_wam_c_bagset_set_1(&state);
    setup_wam_c_bagset_bag_empty_1(&state);
    setup_wam_c_bagset_set_empty_1(&state);

    WamValue bag_args[1] = { val_unbound("Bag") };
    int bag_rc = wam_run_predicate(&state, "wam_c_bagset_bag/1", bag_args, 1);
    if (bag_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list4(&state, state.A[0], "b", "a", "b", "a") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue set_args[1] = { val_unbound("Set") };
    int set_rc = wam_run_predicate(&state, "wam_c_bagset_set/1", set_args, 1);
    if (set_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue bag_empty_args[1] = { val_unbound("BagEmpty") };
    int bag_empty_rc = wam_run_predicate(&state, "wam_c_bagset_bag_empty/1", bag_empty_args, 1);
    if (bag_empty_rc != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue set_empty_args[1] = { val_unbound("SetEmpty") };
    int set_empty_rc = wam_run_predicate(&state, "wam_c_bagset_set_empty/1", set_empty_args, 1);
    if (set_empty_rc != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_bagof_setof_witness_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_group_pair_2(WamState* state);
void setup_wam_c_group_bag_2(WamState* state);
void setup_wam_c_group_set_2(WamState* state);

static bool expect_atom_slot(WamState *state, WamValue *slot, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, slot);
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool expect_nil(WamValue *cell) {
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, "[]") == 0;
}

static bool expect_atom_value(WamState *state, WamValue value, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, &value);
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool expect_atom_list1(WamState *state, WamValue value,
                              const char *first) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_nil(tail);
}

static bool expect_atom_list2(WamState *state, WamValue value,
                              const char *first, const char *second) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_atom_slot(state, &state->H_array[tail1_addr], second) &&
           expect_nil(tail2);
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_group_pair_2(&state);
    setup_wam_c_group_bag_2(&state);
    setup_wam_c_group_set_2(&state);

    WamValue bag_red_args[2] = { val_atom("red"), val_unbound("BagRed") };
    int bag_red_rc = wam_run_predicate(&state, "wam_c_group_bag/2", bag_red_args, 2);
    if (bag_red_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_value(&state, state.A[0], "red") ||
        !expect_atom_list2(&state, state.A[1], "b", "a") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue set_red_args[2] = { val_atom("red"), val_unbound("SetRed") };
    int set_red_rc = wam_run_predicate(&state, "wam_c_group_set/2", set_red_args, 2);
    if (set_red_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_value(&state, state.A[0], "red") ||
        !expect_atom_list2(&state, state.A[1], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue bag_blue_args[2] = { val_atom("blue"), val_unbound("BagBlue") };
    int bag_blue_rc = wam_run_predicate(&state, "wam_c_group_bag/2", bag_blue_args, 2);
    if (bag_blue_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_value(&state, state.A[0], "blue") ||
        !expect_atom_list1(&state, state.A[1], "c") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue bag_missing_args[2] = { val_atom("green"), val_unbound("BagMissing") };
    int bag_missing_rc = wam_run_predicate(&state, "wam_c_group_bag/2", bag_missing_args, 2);
    if (bag_missing_rc != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_bagof_setof_existential_smoke_main(
'#include "wam_runtime.h"

void setup_wam_c_exist_pair_2(WamState* state);
void setup_wam_c_exist_bag_1(WamState* state);
void setup_wam_c_exist_set_1(WamState* state);

static bool expect_atom_slot(WamState *state, WamValue *slot, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, slot);
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool expect_nil(WamValue *cell) {
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, "[]") == 0;
}

static bool expect_atom_list3(WamState *state, WamValue value,
                              const char *first,
                              const char *second,
                              const char *third) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    int tail2_addr = wam_cons_head_addr(state, tail2);
    if (tail2_addr < 0) return false;
    WamValue *tail3 = wam_deref_ptr(state, &state->H_array[tail2_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_atom_slot(state, &state->H_array[tail1_addr], second) &&
           expect_atom_slot(state, &state->H_array[tail2_addr], third) &&
           expect_nil(tail3);
}

static bool expect_atom_list4(WamState *state, WamValue value,
                              const char *first,
                              const char *second,
                              const char *third,
                              const char *fourth) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int cell_addr = wam_cons_head_addr(state, cell);
    if (cell_addr < 0) return false;
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[cell_addr + 1]);
    int tail1_addr = wam_cons_head_addr(state, tail1);
    if (tail1_addr < 0) return false;
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[tail1_addr + 1]);
    int tail2_addr = wam_cons_head_addr(state, tail2);
    if (tail2_addr < 0) return false;
    WamValue *tail3 = wam_deref_ptr(state, &state->H_array[tail2_addr + 1]);
    int tail3_addr = wam_cons_head_addr(state, tail3);
    if (tail3_addr < 0) return false;
    WamValue *tail4 = wam_deref_ptr(state, &state->H_array[tail3_addr + 1]);
    return expect_atom_slot(state, &state->H_array[cell_addr], first) &&
           expect_atom_slot(state, &state->H_array[tail1_addr], second) &&
           expect_atom_slot(state, &state->H_array[tail2_addr], third) &&
           expect_atom_slot(state, &state->H_array[tail3_addr], fourth) &&
           expect_nil(tail4);
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_exist_pair_2(&state);
    setup_wam_c_exist_bag_1(&state);
    setup_wam_c_exist_set_1(&state);

    WamValue bag_args[1] = { val_unbound("Bag") };
    int bag_rc = wam_run_predicate(&state, "wam_c_exist_bag/1", bag_args, 1);
    if (bag_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list4(&state, state.A[0], "b", "a", "c", "b") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue set_args[1] = { val_unbound("Set") };
    int set_rc = wam_run_predicate(&state, "wam_c_exist_set/1", set_args, 1);
    if (set_rc != 0 || state.P != WAM_HALT ||
        !expect_atom_list3(&state, state.A[0], "a", "b", "c") ||
        state.B != 0 || state.call_base_top != 0 || state.aggregate_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_bagof_setof_unbound_witness_groups_main(
'#include "wam_runtime.h"

void setup_wam_c_uw_pair_2(WamState* state);
void setup_wam_c_uw_bag_2(WamState* state);
void setup_wam_c_uw_set_2(WamState* state);
void setup_wam_c_uw_bag_groups_ok_0(WamState* state);
void setup_wam_c_uw_set_groups_ok_0(WamState* state);

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_uw_pair_2(&state);
    setup_wam_c_uw_bag_2(&state);
    setup_wam_c_uw_set_2(&state);
    setup_wam_c_uw_bag_groups_ok_0(&state);
    setup_wam_c_uw_set_groups_ok_0(&state);

    int bag_rc = wam_run_predicate(&state, "wam_c_uw_bag_groups_ok/0", NULL, 0);
    if (bag_rc != 0 || state.P != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    int set_rc = wam_run_predicate(&state, "wam_c_uw_set_groups_ok/0", NULL, 0);
    if (set_rc != 0 || state.P != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    wam_free_state(&state);
    return 0;
}
').

wam_c_bagof_setof_meta_call_main(
'#include "wam_runtime.h"

void setup_wam_c_meta_item_1(WamState* state);
void setup_wam_c_meta_dup_1(WamState* state);
void setup_wam_c_meta_none_1(WamState* state);
void setup_wam_c_meta_conj_item_1(WamState* state);
void setup_wam_c_meta_conj_keep_1(WamState* state);
void setup_wam_c_meta_conj_dup_1(WamState* state);
void setup_wam_c_meta_conj_set_keep_1(WamState* state);
void setup_wam_c_meta_disj_left_1(WamState* state);
void setup_wam_c_meta_disj_right_1(WamState* state);
void setup_wam_c_meta_disj_set_left_1(WamState* state);
void setup_wam_c_meta_disj_set_right_1(WamState* state);
void setup_wam_c_meta_ite_cond_1(WamState* state);
void setup_wam_c_meta_ite_then_1(WamState* state);
void setup_wam_c_meta_ite_fail_cond_1(WamState* state);
void setup_wam_c_meta_ite_else_1(WamState* state);
void setup_wam_c_meta_ite_set_then_1(WamState* state);
void setup_wam_c_meta_call_item_1(WamState* state);
void setup_wam_c_meta_call_pair_2(WamState* state);
void setup_wam_c_meta_call_dup_1(WamState* state);
void setup_wam_c_meta_bag_1(WamState* state);
void setup_wam_c_meta_set_1(WamState* state);
void setup_wam_c_meta_empty_bag_1(WamState* state);
void setup_wam_c_meta_conj_bag_1(WamState* state);
void setup_wam_c_meta_conj_set_1(WamState* state);
void setup_wam_c_meta_disj_bag_1(WamState* state);
void setup_wam_c_meta_disj_set_1(WamState* state);
void setup_wam_c_meta_ite_bag_1(WamState* state);
void setup_wam_c_meta_ite_else_bag_1(WamState* state);
void setup_wam_c_meta_ite_set_1(WamState* state);
void setup_wam_c_meta_call_bag_1(WamState* state);
void setup_wam_c_meta_call_pair_bag_1(WamState* state);
void setup_wam_c_meta_call_partial_bag_1(WamState* state);
void setup_wam_c_meta_call_set_1(WamState* state);

static bool wam_c_meta_expect_atom_list2(WamState *state, WamValue value,
                                         const char *first,
                                         const char *second) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag != VAL_LIST) return false;
    int first_addr = cell->data.ref_addr;
    WamValue *head1 = wam_deref_ptr(state, &state->H_array[first_addr]);
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[first_addr + 1]);
    if (head1->tag != VAL_ATOM || strcmp(head1->data.atom, first) != 0) return false;
    if (tail1->tag != VAL_LIST) return false;
    int second_addr = tail1->data.ref_addr;
    WamValue *head2 = wam_deref_ptr(state, &state->H_array[second_addr]);
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[second_addr + 1]);
    return head2->tag == VAL_ATOM &&
           strcmp(head2->data.atom, second) == 0 &&
           tail2->tag == VAL_ATOM &&
           strcmp(tail2->data.atom, "[]") == 0;
}

static bool wam_c_meta_expect_atom_list3(WamState *state, WamValue value,
                                         const char *first,
                                         const char *second,
                                         const char *third) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag != VAL_LIST) return false;
    int first_addr = cell->data.ref_addr;
    WamValue *head1 = wam_deref_ptr(state, &state->H_array[first_addr]);
    WamValue *tail1 = wam_deref_ptr(state, &state->H_array[first_addr + 1]);
    if (head1->tag != VAL_ATOM || strcmp(head1->data.atom, first) != 0) return false;
    if (tail1->tag != VAL_LIST) return false;
    int second_addr = tail1->data.ref_addr;
    WamValue *head2 = wam_deref_ptr(state, &state->H_array[second_addr]);
    WamValue *tail2 = wam_deref_ptr(state, &state->H_array[second_addr + 1]);
    if (head2->tag != VAL_ATOM || strcmp(head2->data.atom, second) != 0) return false;
    if (tail2->tag != VAL_LIST) return false;
    int third_addr = tail2->data.ref_addr;
    WamValue *head3 = wam_deref_ptr(state, &state->H_array[third_addr]);
    WamValue *tail3 = wam_deref_ptr(state, &state->H_array[third_addr + 1]);
    return head3->tag == VAL_ATOM &&
           strcmp(head3->data.atom, third) == 0 &&
           tail3->tag == VAL_ATOM &&
           strcmp(tail3->data.atom, "[]") == 0;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_meta_item_1(&state);
    setup_wam_c_meta_dup_1(&state);
    setup_wam_c_meta_none_1(&state);
    setup_wam_c_meta_conj_item_1(&state);
    setup_wam_c_meta_conj_keep_1(&state);
    setup_wam_c_meta_conj_dup_1(&state);
    setup_wam_c_meta_conj_set_keep_1(&state);
    setup_wam_c_meta_disj_left_1(&state);
    setup_wam_c_meta_disj_right_1(&state);
    setup_wam_c_meta_disj_set_left_1(&state);
    setup_wam_c_meta_disj_set_right_1(&state);
    setup_wam_c_meta_ite_cond_1(&state);
    setup_wam_c_meta_ite_then_1(&state);
    setup_wam_c_meta_ite_fail_cond_1(&state);
    setup_wam_c_meta_ite_else_1(&state);
    setup_wam_c_meta_ite_set_then_1(&state);
    setup_wam_c_meta_call_item_1(&state);
    setup_wam_c_meta_call_pair_2(&state);
    setup_wam_c_meta_call_dup_1(&state);
    setup_wam_c_meta_bag_1(&state);
    setup_wam_c_meta_set_1(&state);
    setup_wam_c_meta_empty_bag_1(&state);
    setup_wam_c_meta_conj_bag_1(&state);
    setup_wam_c_meta_conj_set_1(&state);
    setup_wam_c_meta_disj_bag_1(&state);
    setup_wam_c_meta_disj_set_1(&state);
    setup_wam_c_meta_ite_bag_1(&state);
    setup_wam_c_meta_ite_else_bag_1(&state);
    setup_wam_c_meta_ite_set_1(&state);
    setup_wam_c_meta_call_bag_1(&state);
    setup_wam_c_meta_call_pair_bag_1(&state);
    setup_wam_c_meta_call_partial_bag_1(&state);
    setup_wam_c_meta_call_set_1(&state);

    WamValue bag_args[1] = { val_unbound("Bag") };
    int bag_rc = wam_run_predicate(&state, "wam_c_meta_bag/1", bag_args, 1);
    if (bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 10;
    }

    WamValue set_args[1] = { val_unbound("Set") };
    int set_rc = wam_run_predicate(&state, "wam_c_meta_set/1", set_args, 1);
    if (set_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 20;
    }

    WamValue empty_args[1] = { val_unbound("Empty") };
    int empty_rc = wam_run_predicate(&state, "wam_c_meta_empty_bag/1",
                                     empty_args, 1);
    if (empty_rc != WAM_HALT ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 30;
    }

    WamValue conj_bag_args[1] = { val_unbound("ConjBag") };
    int conj_bag_rc = wam_run_predicate(&state, "wam_c_meta_conj_bag/1",
                                        conj_bag_args, 1);
    if (conj_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "c") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 40;
    }

    WamValue conj_set_args[1] = { val_unbound("ConjSet") };
    int conj_set_rc = wam_run_predicate(&state, "wam_c_meta_conj_set/1",
                                        conj_set_args, 1);
    if (conj_set_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 50;
    }

    WamValue disj_bag_args[1] = { val_unbound("DisjBag") };
    int disj_bag_rc = wam_run_predicate(&state, "wam_c_meta_disj_bag/1",
                                        disj_bag_args, 1);
    if (disj_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list3(&state, state.A[0], "a", "b", "c") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 60;
    }

    WamValue disj_set_args[1] = { val_unbound("DisjSet") };
    int disj_set_rc = wam_run_predicate(&state, "wam_c_meta_disj_set/1",
                                        disj_set_args, 1);
    if (disj_set_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list3(&state, state.A[0], "a", "b", "c") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 70;
    }

    WamValue ite_bag_args[1] = { val_unbound("IteBag") };
    int ite_bag_rc = wam_run_predicate(&state, "wam_c_meta_ite_bag/1",
                                       ite_bag_args, 1);
    if (ite_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "a") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 80;
    }

    WamValue ite_else_bag_args[1] = { val_unbound("IteElseBag") };
    int ite_else_bag_rc = wam_run_predicate(&state,
                                            "wam_c_meta_ite_else_bag/1",
                                            ite_else_bag_args, 1);
    if (ite_else_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "c", "d") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 90;
    }

    WamValue ite_set_args[1] = { val_unbound("IteSet") };
    int ite_set_rc = wam_run_predicate(&state, "wam_c_meta_ite_set/1",
                                       ite_set_args, 1);
    if (ite_set_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 100;
    }

    WamValue call_bag_args[1] = { val_unbound("CallBag") };
    int call_bag_rc = wam_run_predicate(&state, "wam_c_meta_call_bag/1",
                                        call_bag_args, 1);
    if (call_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 110;
    }

    WamValue call_pair_bag_args[1] = { val_unbound("CallPairBag") };
    int call_pair_bag_rc = wam_run_predicate(&state,
                                             "wam_c_meta_call_pair_bag/1",
                                             call_pair_bag_args, 1);
    if (call_pair_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 120;
    }

    WamValue call_partial_bag_args[1] = { val_unbound("CallPartialBag") };
    int call_partial_bag_rc =
        wam_run_predicate(&state, "wam_c_meta_call_partial_bag/1",
                          call_partial_bag_args, 1);
    if (call_partial_bag_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 130;
    }

    WamValue call_set_args[1] = { val_unbound("CallSet") };
    int call_set_rc = wam_run_predicate(&state, "wam_c_meta_call_set/1",
                                        call_set_args, 1);
    if (call_set_rc != 0 || state.P != WAM_HALT ||
        !wam_c_meta_expect_atom_list2(&state, state.A[0], "a", "b") ||
        state.B != 0 || state.call_base_top != 0 ||
        state.aggregate_top != 0 || state.aggregate_group_top != 0 ||
        state.conj_top != 0 || state.disj_top != 0 ||
        state.ite_top != 0) {
        wam_free_state(&state);
        return 140;
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
implemented_case(begin_aggregate, 'case INSTR_BEGIN_AGGREGATE').
implemented_case(end_aggregate, 'case INSTR_END_AGGREGATE').
implemented_case(try_me_else, 'case INSTR_TRY_ME_ELSE').
implemented_case(retry_me_else, 'case INSTR_RETRY_ME_ELSE').
implemented_case(trust_me, 'case INSTR_TRUST_ME').
implemented_case(get_level, 'case INSTR_GET_LEVEL').
implemented_case(cut, 'case INSTR_CUT').
implemented_case(cut_ite, 'case INSTR_CUT_ITE').
implemented_case(jump, 'case INSTR_JUMP').
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
    test_aggregate_instructions,
    test_control_cut_jump_instructions,
    test_explicit_cut_uses_current_call_barrier,
    test_control_instruction_parsing,
    test_precise_ite_y_level_generation,
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
    test_bidirectional_ancestor_kernel_generation,
    test_transitive_closure_kernel_generation,
    test_transitive_distance_kernel_generation,
    test_transitive_parent_distance_kernel_generation,
    test_transitive_step_parent_distance_kernel_generation,
    test_weighted_shortest_path_kernel_generation,
    test_astar_shortest_path_kernel_generation,
    test_fact_source_generation,
    test_reverse_csr_generation,
    test_reverse_index_plan_none,
    test_reverse_index_plan_csr_cost_model,
    test_reverse_index_plan_runtime_available_sorted_array,
    test_reverse_index_plan_runtime_available_lmdb_offset,
    test_reverse_index_plan_runtime_direct_io_available,
    test_reverse_index_plan_runtime_buffered_pread_drop_available,
    test_reverse_index_setup_generation,
    test_reverse_index_setup_lmdb_offset_generation,
    test_reverse_index_setup_lmdb_offset_pread_drop_generation,
    test_reverse_index_setup_rejects_runtime_direct_io_without_block_size,
    test_reverse_index_setup_direct_io_generation,
    test_reverse_index_setup_buffered_pread_drop_generation,
    test_streaming_foreign_results_generation,
    test_kernel_detector_setup_generation,
    test_bidirectional_ancestor_setup_generation,
    test_transitive_closure_detector_setup_generation,
    test_transitive_distance_detector_setup_generation,
    test_transitive_parent_distance_detector_setup_generation,
    test_transitive_step_parent_distance_detector_setup_generation,
    test_weighted_shortest_path_detector_setup_generation,
    test_astar_shortest_path_detector_setup_generation,
    test_kernel_detector_project_generation,
    test_lowered_fact_helper_generation,
    test_lowered_helper_planner_metadata,
    test_lowered_helper_plan_generation,
    test_lowered_body_call_helper_generation,
    test_lowered_body_call_rejection_metadata,
    test_lowered_body_call_projection_rejection_metadata,
    test_lowered_filtered_fact_helper_generation,
    test_lowered_comparison_filter_helper_generation,
    test_lowered_repeated_variable_filter_generation,
    test_lowered_filter_rejection_metadata,
    test_lowered_comparison_filter_rejection_metadata,
    test_unsupported_instruction_fails_loudly,
    test_no_zero_instruction_fallback,
    test_list_target_pc_emission,
    test_generated_runtime_executable_smoke,
    test_cross_predicate_executable_smoke,
    test_multi_predicate_setup_executable_smoke,
    test_builtin_call_executable_smoke,
    test_call_foreign_executable_smoke,
    test_category_ancestor_kernel_executable_smoke,
    test_bidirectional_ancestor_kernel_executable_smoke,
    test_bidirectional_ancestor_csr_child_lookup_executable_smoke,
    test_reverse_index_setup_executable_smoke,
    test_transitive_closure_kernel_executable_smoke,
    test_transitive_distance_kernel_executable_smoke,
    test_transitive_parent_distance_kernel_executable_smoke,
    test_transitive_step_parent_distance_kernel_executable_smoke,
    test_weighted_shortest_path_kernel_executable_smoke,
    test_astar_shortest_path_kernel_executable_smoke,
    test_fact_source_executable_smoke,
    test_lmdb_fact_source_executable_smoke,
    test_reverse_csr_executable_smoke,
    test_reverse_csr_lmdb_offset_executable_smoke,
    test_kernel_detector_executable_smoke,
    test_transitive_closure_detector_executable_smoke,
    test_transitive_distance_detector_executable_smoke,
    test_transitive_parent_distance_detector_executable_smoke,
    test_transitive_step_parent_distance_detector_executable_smoke,
    test_weighted_shortest_path_detector_executable_smoke,
    test_astar_shortest_path_detector_executable_smoke,
    test_streaming_foreign_results_executable_smoke,
    test_real_prolog_builtin_executable_smoke,
    test_real_prolog_term_builtin_executable_smoke,
    test_real_prolog_multiclause_executable_smoke,
    test_real_prolog_structure_index_executable_smoke,
    test_real_prolog_is_list_executable_smoke,
    test_real_prolog_unify_executable_smoke,
    test_real_prolog_control_executable_smoke,
    test_real_prolog_precise_ite_executable_smoke,
    test_real_prolog_explicit_cut_executable_smoke,
    test_real_prolog_forall_executable_smoke,
    test_real_prolog_findall_executable_smoke,
    test_real_prolog_bagof_setof_executable_smoke,
    test_real_prolog_bagof_setof_witness_executable_smoke,
    test_real_prolog_bagof_setof_existential_executable_smoke,
    test_real_prolog_bagof_setof_unbound_witness_groups_smoke,
    test_real_prolog_bagof_setof_meta_call_smoke,
    test_real_prolog_classic_recursive_executable_smoke,
    test_lowered_fact_helper_executable_smoke,
    test_lowered_body_call_helper_executable_smoke,
    test_lowered_filtered_fact_helper_executable_smoke,
    test_lowered_comparison_filter_helper_executable_smoke,
    test_lowered_repeated_variable_filter_executable_smoke,
    test_asan_memory_lifecycle_executable_smoke,
    format('~n=== WAM-C Target Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
