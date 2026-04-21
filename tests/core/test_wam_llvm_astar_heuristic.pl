:- encoding(utf8).
% test_wam_llvm_astar_heuristic.pl
% Tests that A* with a non-zero admissible heuristic produces the
% correct shortest-path distance (same as Dijkstra since the
% heuristic is admissible — never overestimates).
%
% The heuristic is baked at compile time for a FIXED target atom
% specified via `heuristic_target(TargetAtom)` in the config. A
% `heuristic_pred(Name/3)` predicate supplies per-node estimates
% of the remaining distance to that target.
%
% Graph (same as M5.9/M5.10 — greedy ≠ optimal):
%
%     a --10--> b --1--> c --1--> d    (optimal three-hop, total 12)
%     a --100-> d                      (greedy direct, total 100)
%
% Heuristic (admissible, for target d):
%     h(a, d) = 3.0     (actual ≥ 12, so 3 is admissible)
%     h(b, d) = 2.0     (actual = 2, so exact)
%     h(c, d) = 1.0     (actual = 1, exact)
%     h(d, d) = 0.0     (exact)
%
% Expected: A*(a, d) = 12.0 (same as Dijkstra — heuristic only
%           affects exploration order, not the result)
%
% Additionally verifies that the heuristic array is NOT
% zeroinitializer (proving the heuristic_pred path was taken).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% Edge weights.
:- dynamic hw/3.
hw(a, b, 10.0).
hw(b, c, 1.0).
hw(c, d, 1.0).
hw(a, d, 100.0).

% Heuristic for target=d. Admissible: never overestimates.
:- dynamic h_to_d/3.
h_to_d(a, d, 3.0).
h_to_d(b, d, 2.0).
h_to_d(c, d, 1.0).
h_to_d(d, d, 0.0).

:- dynamic astar_h/3.
astar_h(_, _, _) :- fail.

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

extract_atom_ids_from_weighted(Src, TableName, Labels, AtomIds) :-
    format(atom(StartMarker), '@~w = private constant', [TableName]),
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
    "WeightedFact \\{ i64 (?<from>\\d+), i64 (?<to>\\d+), double",
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

extract_instr_count(Src, P, C) :-
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

test_astar_with_heuristic :-
    format('--- A* with non-zero admissible heuristic ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_heuristic_test
    ;  format('  SKIP: clang or llc not found~n')
    ).

run_heuristic_test :-
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:astar_h/3],
        [ module_name('astar_h_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              astar_h/3 - astar_shortest_path4 - [
                  weight_pred(hw/3),
                  heuristic_pred(h_to_d/3),
                  heuristic_target(d)
              ]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Structural: heuristic array is NOT zeroinitializer.
    ( sub_string(Src, _, _, _, 'astar4_inst_astar_h_0_heuristic')
    -> format('  PASS: heuristic global emitted~n')
    ;  format('  FAIL: heuristic global missing~n'), throw(no_heuristic)
    ),
    ( sub_string(Src, _, _, _, 'double 3.0')
    -> format('  PASS: non-zero heuristic value present (h(a,d)=3.0)~n')
    ;  format('  FAIL: heuristic appears to be all zeros~n')
    ),

    % Extract atom IDs and build driver.
    extract_atom_ids_from_weighted(Src, 'astar4_inst_astar_h_0_edges',
        [a-b, b-c, c-d, a-d], AtomIds),
    get_dict(a, AtomIds, AId),
    get_dict(d, AtomIds, DId),
    extract_instr_count(Src, astar_h, IC),
    extract_label_count(Src, astar_h, LC),
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
        [AId, DId, IC, IC, IC, LC, LC]),
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
    -> format('  FAIL: llc exit=~w~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w -lm 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('  FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ExpectedScaled is truncate(12.0 * 100) /\ 255,
    ( ExitCode =:= ExpectedScaled
    -> format('  PASS: A*(a,d) with heuristic = 12.0 (scaled=~w)~n', [ExitCode])
    ;  format('  FAIL: returned ~w (expected ~w for 12.0)~n',
          [ExitCode, ExpectedScaled])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= ExpectedScaled).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

test_all :-
    catch(test_astar_with_heuristic, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
