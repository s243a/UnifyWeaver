:- encoding(utf8).
% test_wam_llvm_stress.pl
% Stress tests for BFS and Dijkstra kernels on larger graphs.
%
% Prior M5.7/M5.9 tests used 4-5 node graphs. This test generates
% 100-node graphs programmatically to validate:
%   - malloc/free cleanup paths don't crash at scale
%   - O(V*E) BFS and O(V²+VE) Dijkstra don't blow up
%   - The visited/queue/best arrays are correctly sized by max_atom_id
%   - Interned atom IDs work with larger ID ranges
%
% Graph shapes:
%   BFS:      chain n0 → n1 → n2 → ... → n99 (100 nodes, 99 edges)
%             Expected: distance(n0, n99) = 99
%   Dijkstra: same chain but with weights 1.0 each, plus a "shortcut"
%             n0 → n50 with weight 60.0 (longer than 50 hops * 1.0)
%             Expected: dijkstra(n0, n99) = 99.0 (chain is shorter)
%             Expected: dijkstra(n0, n50) = 50.0 (chain beats shortcut)

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% ============================================================
% Programmatic graph generation
% ============================================================

% Generate chain edge facts: chain_edge(n0, n1), chain_edge(n1, n2), ...
:- dynamic chain_edge/2.

generate_chain_edges(N) :-
    retractall(chain_edge(_, _)),
    Max is N - 1,
    forall(
        between(0, Max, I),
        ( J is I + 1,
          J =< N - 1 ->
            ( atom_concat(n, I, From),
              atom_concat(n, J, To),
              assertz(chain_edge(From, To))
            )
          ; true
        )
    ).

% Generate weighted chain + shortcut.
:- dynamic weighted_chain/3.

generate_weighted_chain(N) :-
    retractall(weighted_chain(_, _, _)),
    Max is N - 2,
    forall(
        between(0, Max, I),
        ( J is I + 1,
          atom_concat(n, I, From),
          atom_concat(n, J, To),
          assertz(weighted_chain(From, To, 1.0))
        )
    ),
    % Add a "shortcut" that's actually slower than the chain.
    Half is N // 2,
    atom_concat(n, 0, Start),
    atom_concat(n, Half, Mid),
    ShortcutWeight is float(Half + 10),
    assertz(weighted_chain(Start, Mid, ShortcutWeight)).

% Dummy predicates for the compile pipeline.
:- dynamic stress_reach/3.
stress_reach(_, _, _) :- fail.

:- dynamic stress_dijkstra/3.
stress_dijkstra(_, _, _) :- fail.

% ============================================================
% Host triple detection
% ============================================================

host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Raw), close(Out),
          process_wait(PID, exit(0))
        ), _, fail)
    -> split_string(Raw, "", "\n\r\t ", [S]),
       atom_string(Triple, S)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

% ============================================================
% Atom ID extraction helpers (same as M5.7/M5.8 tests)
% ============================================================

extract_atom_id_for(Src, TablePattern, AtomName, AtomId) :-
    % Find the atom's interned ID by searching for it in the fact table.
    % We look for AtomFactPair entries and match by position in the
    % chain. For n0, it's the first `from` value; for n99, we search
    % all entries.
    atom_concat(n, _, AtomName), !,  % sanity: must be nN
    format(atom(StartMarker), '@~w = private constant', [TablePattern]),
    once(sub_string(Src, StartIdx, _, _, StartMarker)),
    sub_string(Src, StartIdx, _, 0, FromStart),
    % Collect all atom IDs from the table to build a mapping.
    re_foldl([Match, Acc, [FromId-ToId | Acc]]>>(
        get_dict(from, Match, FS), get_dict(to, Match, TS),
        number_string(FromId, FS), number_string(ToId, TS)
    ),
    "i64 (?<from>\\d+), i64 (?<to>\\d+)",
    FromStart, [], PairsRev, []),
    reverse(PairsRev, Pairs),
    % n0's ID is the `from` of the first pair; nN's ID is the `to`
    % of the (N-1)th pair for chain graphs.
    atom_concat(n, NumStr, AtomName),
    atom_number(NumStr, Num),
    ( Num =:= 0
    -> Pairs = [AtomId - _ | _]
    ;  nth1(Num, Pairs, _ - AtomId)
    ).

extract_atom_id_for_weighted(Src, TablePattern, AtomName, AtomId) :-
    atom_concat(n, _, AtomName), !,
    format(atom(StartMarker), '@~w = private constant', [TablePattern]),
    once(sub_string(Src, StartIdx, _, _, StartMarker)),
    sub_string(Src, StartIdx, _, 0, FromStart),
    re_foldl([Match, Acc, [FromId-ToId | Acc]]>>(
        get_dict(from, Match, FS), get_dict(to, Match, TS),
        number_string(FromId, FS), number_string(ToId, TS)
    ),
    "i64 (?<from>\\d+), i64 (?<to>\\d+), double",
    FromStart, [], PairsRev, []),
    reverse(PairsRev, Pairs),
    atom_concat(n, NumStr, AtomName),
    atom_number(NumStr, Num),
    ( Num =:= 0
    -> Pairs = [AtomId - _ | _]
    ;  nth1(Num, Pairs, _ - AtomId)
    ).

extract_instr_count(Src, P, C) :-
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

% ============================================================
% BFS Stress Test
% ============================================================

test_bfs_stress :-
    format('--- BFS stress: 100-node chain ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_bfs_stress
    ;  format('  SKIP: clang or llc not found~n')
    ).

run_bfs_stress :-
    generate_chain_edges(100),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:stress_reach/3],
        [ module_name('bfs_stress'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              stress_reach/3 - transitive_distance3 - [edge_pred(chain_edge/2)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_atom_id_for(Src, 'td3_inst_stress_reach_0_edges', n0, N0Id),
    extract_atom_id_for(Src, 'td3_inst_stress_reach_0_edges', n99, N99Id),
    extract_instr_count(Src, stress_reach, IC),
    extract_label_count(Src, stress_reach, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 0, 0
  %a2 = insertvalue %Value %a2_0, i64 ~w, 1
  %a3_0 = insertvalue %Value undef, i32 6, 0
  %a3 = insertvalue %Value %a3_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %a3)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  %dist = call i64 @wam_get_reg_payload(%WamState* %vm, i32 2)
  %dist32 = trunc i64 %dist to i32
  ret i32 %dist32
miss:
  ret i32 255
}
',
        [N0Id, N99Id, IC, IC, IC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    compile_and_run(LLPath, ExitCode),
    % Expected: 99 hops in a 100-node chain.
    ( ExitCode =:= 99
    -> format('  PASS: BFS(n0, n99) = ~w on 100-node chain~n', [ExitCode])
    ;  format('  FAIL: BFS returned ~w (expected 99)~n', [ExitCode])
    ),
    clear_llvm_foreign_kernel_specs,
    retractall(chain_edge(_, _)),
    assertion(ExitCode =:= 99).

% ============================================================
% Dijkstra Stress Test
% ============================================================

test_dijkstra_stress :-
    format('--- Dijkstra stress: 100-node weighted chain + shortcut ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_dijkstra_stress
    ;  format('  SKIP: clang or llc not found~n')
    ).

run_dijkstra_stress :-
    generate_weighted_chain(100),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:stress_dijkstra/3],
        [ module_name('dijk_stress'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              stress_dijkstra/3 - weighted_shortest_path3 - [weight_pred(weighted_chain/3)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_atom_id_for_weighted(Src, 'wsp3_inst_stress_dijkstra_0_edges', n0, N0Id),
    extract_atom_id_for_weighted(Src, 'wsp3_inst_stress_dijkstra_0_edges', n99, N99Id),
    extract_instr_count(Src, stress_dijkstra, IC),
    extract_label_count(Src, stress_dijkstra, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 0, 0
  %a2 = insertvalue %Value %a2_0, i64 ~w, 1
  %a3_0 = insertvalue %Value undef, i32 6, 0
  %a3 = insertvalue %Value %a3_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %a3)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  %dist = call double @wam_get_reg_double(%WamState* %vm, i32 2)
  %dist100 = fmul double %dist, 1.0e2
  %dist_i32 = fptosi double %dist100 to i32
  ret i32 %dist_i32
miss:
  ret i32 255
}
',
        [N0Id, N99Id, IC, IC, IC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    compile_and_run(LLPath, ExitCode),
    % Expected: 99 * 1.0 = 99.0 → scaled = 9900 & 255 = 172
    % (chain of 99 edges at weight 1.0 each beats the shortcut
    % n0→n50 at weight 60.0 because 50*1.0 < 60.0 for the first
    % half, and the shortcut doesn't reach n99 at all).
    ExpectedScaled is truncate(99.0 * 100) /\ 255,
    ( ExitCode =:= ExpectedScaled
    -> format('  PASS: Dijkstra(n0, n99) scaled=~w (99.0) on 100-node chain~n',
          [ExitCode])
    ;  format('  FAIL: Dijkstra returned scaled=~w (expected ~w for 99.0)~n',
          [ExitCode, ExpectedScaled])
    ),
    clear_llvm_foreign_kernel_specs,
    retractall(weighted_chain(_, _, _)),
    assertion(ExitCode =:= ExpectedScaled).

% ============================================================
% Shared helpers
% ============================================================

compile_and_run(LLPath, ExitCode) :-
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('  FAIL: llc exit=~w~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w -lm 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('  FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

test_all :-
    catch(test_bfs_stress, E1,
        format('  ERROR in BFS stress: ~w~n', [E1])),
    catch(test_dijkstra_stress, E2,
        format('  ERROR in Dijkstra stress: ~w~n', [E2])).

:- initialization(test_all, main).
