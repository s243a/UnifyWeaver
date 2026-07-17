:- encoding(utf8).
% test_wam_llvm_td3_stream_execution.pl
% End-to-end execution test for stream-mode transitive_distance3.
% When A2 is unbound (tag=6), td3 enumerates all reachable (target, distance)
% pairs via BFS and yields them through the paired foreign iterator.
%
% Graph: p -> q -> r -> s -> p, p -> t, with p -> q duplicated.
%
% Tests:
% Each case validates the exact target/distance pairing, duplicate
% suppression, Source re-entry through a nonempty cycle, and exhaustion.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic link/2.
link(p, q).
link(q, r).
link(r, s).
link(s, p).
link(p, t).
link(p, q).

:- dynamic reach/3.
reach(_, _, _) :- fail.

host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Raw), close(Out),
          process_wait(PID, exit(0))
        ), _, fail)
    -> split_string(Raw, "", "\n\r\t ", [S]), atom_string(Triple, S)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

extract_atom_id_for(Src, TablePattern, Labels, AtomIds) :-
    format(atom(StartMarker), '@~w = private constant', [TablePattern]),
    once(sub_string(Src, StartIdx, _, _, StartMarker)),
    sub_string(Src, StartIdx, _, 0, FromStart),
    once(sub_string(FromStart, AfterTypeIdx, 3, _, "] [")),
    BodyStart is AfterTypeIdx + 3,
    sub_string(FromStart, BodyStart, _, 0, FromBody),
    once(sub_string(FromBody, CloseIdx, 2, _, "\n]")),
    sub_string(FromBody, 0, CloseIdx, _, TableBody),
    re_foldl([Match, Acc, [FromId - ToId | Acc]]>>(
        get_dict(from, Match, FS), get_dict(to, Match, TS),
        number_string(FromId, FS), number_string(ToId, TS)
    ),
    "AtomFactPair \\{ i64 (?<from>\\d+), i64 (?<to>\\d+) \\}",
    TableBody, [], PairsRev, []),
    reverse(PairsRev, Pairs),
    zip_facts_to_ids(Labels, Pairs, Bindings),
    dict_pairs(AtomIds, atom_ids, Bindings).

zip_facts_to_ids(L, P, B) :- zip_loop(L, P, [], B).
zip_loop([], _, A, S) :- sort(1, @<, A, S).
zip_loop([F-T|LR], [FI-TI|PR], A, O) :- !,
    add_b(F, FI, A, A1), add_b(T, TI, A1, A2), zip_loop(LR, PR, A2, O).
zip_loop(_, _, A, S) :- sort(1, @<, A, S).
add_b(L, _, A, A) :- memberchk(L-_, A), !.
add_b(L, I, A, [L-I|A]).

extract_instr_count(Src, C) :-
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, C) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

run_stream_td3_case(Label, StartAtom, ExpectedPairs) :-
    length(ExpectedPairs, ExpectedCount),
    ( ExpectedCount =:= 0 -> NoResultsStatus = 0 ; NoResultsStatus = 2 ),
    format('  testing ~w: reach(~w, X, D) expected ~w~n',
        [Label, StartAtom, ExpectedPairs]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:reach/3],
        [ module_name('td3_stream'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              reach/3 - transitive_distance3 - [edge_pred(link/2)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_atom_id_for(Src, 'td3_inst_reach_0_edges',
        [p-q, q-r, r-s, s-p, p-t, p-q], AtomIds),
    td3_stream_start_id(StartAtom, AtomIds, StartId),
    ( StartAtom == outside_edge_table
    -> assertion(sub_string(Src, _, _, _,
           "br i1 %start_in_range, label %check_distance_filter, label %stream_fail_early"))
    ;  true
    ),
    td3_expected_pair_bit_ir(ExpectedPairs, AtomIds,
                             PairBitIR, ExpectedPairMask),
    extract_instr_count(Src, IC),
    extract_label_count(Src, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 0)

  ; Bound inputs may also arrive through Ref chains.
  %source_leaf = call i32 @wam_heap_push(%WamState* %vm, %Value %a1)
  %source_leaf_ref = call %Value @value_ref(i32 %source_leaf)
  %source_root = call i32 @wam_heap_push(%WamState* %vm, %Value %source_leaf_ref)
  %source_ref = call %Value @value_ref(i32 %source_root)

  ; Model normal compiled variables: each output is a two-deep Ref chain
  ; ending in its own unbound heap cell.
  %unbound_0 = insertvalue %Value undef, i32 6, 0
  %unbound = insertvalue %Value %unbound_0, i64 0, 1
  %target_leaf = call i32 @wam_heap_push(%WamState* %vm, %Value %unbound)
  %target_leaf_ref = call %Value @value_ref(i32 %target_leaf)
  %target_root = call i32 @wam_heap_push(%WamState* %vm, %Value %target_leaf_ref)
  %target_ref = call %Value @value_ref(i32 %target_root)
  %distance_leaf = call i32 @wam_heap_push(%WamState* %vm, %Value %unbound)
  %distance_leaf_ref = call %Value @value_ref(i32 %distance_leaf)
  %distance_root = call i32 @wam_heap_push(%WamState* %vm, %Value %distance_leaf_ref)
  %distance_ref = call %Value @value_ref(i32 %distance_root)

  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %source_ref)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %target_ref)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %distance_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %count_entry, label %no_results

count_entry:
  br label %count_loop

count_loop:
  %count = phi i32 [0, %count_entry], [%count_inc, %got_next]
  %seen_pairs = phi i64 [0, %count_entry], [%seen_pairs_next, %got_next]
  %pairs_valid = phi i1 [true, %count_entry], [%pairs_valid_next, %got_next]
  %target_value = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %distance_value = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %target_tag = extractvalue %Value %target_value, 0
  %distance_tag = extractvalue %Value %distance_value, 0
  %target_is_atom = icmp eq i32 %target_tag, 0
  %distance_is_int = icmp eq i32 %distance_tag, 1
  %types_ok = and i1 %target_is_atom, %distance_is_int
  %target = extractvalue %Value %target_value, 1
  %distance = extractvalue %Value %distance_value, 1
~w  %pair_known = icmp ne i64 %pair_bit, 0
  %pair_seen_bits = and i64 %seen_pairs, %pair_bit
  %pair_new = icmp eq i64 %pair_seen_bits, 0
  %known_and_new = and i1 %pair_known, %pair_new
  %pair_valid = and i1 %known_and_new, %types_ok
  %pairs_valid_next = and i1 %pairs_valid, %pair_valid
  %seen_pairs_next = or i64 %seen_pairs, %pair_bit
  %count_inc = add i32 %count, 1
  %bt_ok = call i1 @backtrack(%WamState* %vm)
  br i1 %bt_ok, label %got_next, label %done

got_next:
  br label %count_loop

done:
  %count_ok = icmp eq i32 %count_inc, ~w
  %pairs_ok = icmp eq i64 %seen_pairs_next, ~w
  %set_ok = and i1 %count_ok, %pairs_ok
  %all_ok = and i1 %set_ok, %pairs_valid_next
  ; Exhaustion must restore the pre-first-yield Ref snapshot and trail.
  %target_final_raw = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %distance_final_raw = call %Value @wam_get_reg(%WamState* %vm, i32 2)
  %target_final_tag = extractvalue %Value %target_final_raw, 0
  %distance_final_tag = extractvalue %Value %distance_final_raw, 0
  %target_ref_restored = icmp eq i32 %target_final_tag, 5
  %distance_ref_restored = icmp eq i32 %distance_final_tag, 5
  %refs_restored = and i1 %target_ref_restored, %distance_ref_restored
  %target_final = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %distance_final = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %target_unbound = call i1 @value_is_unbound(%Value %target_final)
  %distance_unbound = call i1 @value_is_unbound(%Value %distance_final)
  %vars_unbound = and i1 %target_unbound, %distance_unbound
  %cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %cpn = load i32, i32* %cpn_ptr
  %cp_empty = icmp eq i32 %cpn, 0
  %ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %ts = load i32, i32* %ts_ptr
  %trail_empty = icmp eq i32 %ts, 0
  %snapshot_ok_0 = and i1 %refs_restored, %vars_unbound
  %snapshot_ok_1 = and i1 %cp_empty, %trail_empty
  %snapshot_ok = and i1 %snapshot_ok_0, %snapshot_ok_1
  %alias_ok = call i1 @check_td3_alias()
  %runtime_ok = and i1 %all_ok, %snapshot_ok
  %contract_ok = and i1 %runtime_ok, %alias_ok
  %status = select i1 %contract_ok, i32 0, i32 1
  ret i32 %status

no_results:
  %no_alias_ok = call i1 @check_td3_alias()
  %no_status = select i1 %no_alias_ok, i32 ~w, i32 3
  ret i32 %no_status
}

; A2 and A3 sharing the same nested Ref chain cannot unify with the
; heterogeneous (Atom, Integer) tuple.  Failure must leave the alias
; unbound and must not leak the provisional foreign choice point/trail.
define i1 @check_td3_alias() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 0)
  %source_leaf = call i32 @wam_heap_push(%WamState* %vm, %Value %a1)
  %source_leaf_ref = call %Value @value_ref(i32 %source_leaf)
  %source_root = call i32 @wam_heap_push(%WamState* %vm, %Value %source_leaf_ref)
  %source_ref = call %Value @value_ref(i32 %source_root)
  %u0 = insertvalue %Value undef, i32 6, 0
  %u = insertvalue %Value %u0, i64 0, 1
  %leaf = call i32 @wam_heap_push(%WamState* %vm, %Value %u)
  %leaf_ref = call %Value @value_ref(i32 %leaf)
  %root = call i32 @wam_heap_push(%WamState* %vm, %Value %leaf_ref)
  %shared_ref = call %Value @value_ref(i32 %root)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %source_ref)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %shared_ref)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %shared_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  %failed = xor i1 %ok, true
  %target_after = call %Value @wam_get_reg_deref(%WamState* %vm, i32 1)
  %distance_after = call %Value @wam_get_reg_deref(%WamState* %vm, i32 2)
  %target_unbound = call i1 @value_is_unbound(%Value %target_after)
  %distance_unbound = call i1 @value_is_unbound(%Value %distance_after)
  %both_unbound = and i1 %target_unbound, %distance_unbound
  %target_raw = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %distance_raw = call %Value @wam_get_reg(%WamState* %vm, i32 2)
  %target_payload = extractvalue %Value %target_raw, 1
  %distance_payload = extractvalue %Value %distance_raw, 1
  %alias_preserved = icmp eq i64 %target_payload, %distance_payload
  %cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %cpn = load i32, i32* %cpn_ptr
  %cp_empty = icmp eq i32 %cpn, 0
  %ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %ts = load i32, i32* %ts_ptr
  %trail_empty = icmp eq i32 %ts, 0
  %state_ok_0 = and i1 %both_unbound, %alias_preserved
  %state_ok_1 = and i1 %cp_empty, %trail_empty
  %state_ok = and i1 %state_ok_0, %state_ok_1
  %all_ok = and i1 %failed, %state_ok
  ret i1 %all_ok
}
',
        [StartId, IC, IC, IC, LC, LC, PairBitIR,
         ExpectedCount, ExpectedPairMask,
         NoResultsStatus,
         StartId, IC, IC, IC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('    FAIL: llc exit=~w~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w -lm 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ( ExitCode =:= 0
    -> format('    PASS: ~w matched exact pair set~n', [Label])
    ;  format('    FAIL: ~w returned status ~w~n', [Label, ExitCode])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= 0).

td3_stream_start_id(StartAtom, AtomIds, StartId) :-
    ( get_dict(StartAtom, AtomIds, StartId)
    -> true
    ;  StartAtom == outside_edge_table,
       dict_pairs(AtomIds, _, Bindings),
       pairs_values(Bindings, EdgeAtomIds),
       max_list(EdgeAtomIds, EdgeMax),
       StartId is EdgeMax + 1
    ).

td3_expected_pair_bit_ir(Pairs, AtomIds, IR, ExpectedMask) :-
    td3_expected_pair_bit_ir_(Pairs, AtomIds, 0, "0", Lines, LastAcc),
    format(string(Final), '  %pair_bit = or i64 ~w, 0~n', [LastAcc]),
    append(Lines, [Final], AllLines),
    atomics_to_string(AllLines, '', IR),
    length(Pairs, Count),
    ExpectedMask is (1 << Count) - 1.

td3_expected_pair_bit_ir_([], _, _, LastAcc, [], LastAcc).
td3_expected_pair_bit_ir_([Atom-Distance|Rest], AtomIds, I, PrevAcc,
                          [Line|Lines], LastAcc) :-
    get_dict(Atom, AtomIds, AtomId),
    Bit is 1 << I,
    format(string(Line),
'  %target_match_~w = icmp eq i64 %target, ~w
  %distance_match_~w = icmp eq i64 %distance, ~w
  %pair_match_~w = and i1 %target_match_~w, %distance_match_~w
  %pair_bit_~w = select i1 %pair_match_~w, i64 ~w, i64 0
  %pair_bit_acc_~w = or i64 ~w, %pair_bit_~w
',
        [ I, AtomId, I, Distance, I, I, I, I, I, Bit,
          I, PrevAcc, I ]),
    format(string(NextAcc), '%pair_bit_acc_~w', [I]),
    I1 is I + 1,
    td3_expected_pair_bit_ir_(Rest, AtomIds, I1, NextAcc,
                              Lines, LastAcc).

test_td3_stream :-
    format('--- transitive_distance3 stream mode ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_stream_td3_case('cycle from p', p,
           [q-1, r-2, s-3, p-4, t-1]),
       run_stream_td3_case('cycle from r', r,
           [s-1, p-2, q-3, t-3, r-4]),
       run_stream_td3_case('cycle from s', s,
           [p-1, q-2, t-2, r-3, s-4]),
       run_stream_td3_case('out-of-table source', outside_edge_table, [])
    ;  format('  SKIP: clang or llc not found~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

test_all :-
    test_td3_stream.

:- initialization(test_all, main).
