:- encoding(utf8).
% test_wam_llvm_astar_execution.pl
% M5.10: End-to-end execution test for astar_shortest_path4 via the
% @wam_astar_weighted_distance helper with a zero heuristic.
%
% With h=0 A* degenerates to Dijkstra, so the expected answers match
% M5.9's Dijkstra execution test. The value of this test is that it
% exercises the full A* pipeline end-to-end:
%
%   1. A zeroed heuristic [N x double] global is emitted alongside
%      the %WeightedFact edge table.
%   2. @wam_astar4_run passes both the edge table AND the heuristic
%      pointer to @wam_astar_weighted_distance.
%   3. The A* helper's min-scan computes f(n) = g(n) + h(n) where
%      h(n)=0, confirming the extra fadd doesn't corrupt the result.
%   4. The float bitcast return path (@wam_set_reg_double) is
%      validated for A* specifically.
%
% Test graph (same as M5.9 Dijkstra — greedy ≠ optimal):
%
%     a --10--> b --1--> c --1--> d    (three-hop, total 12)
%     a --100-> d                      (direct, total 100)
%
% Expected: A*(a, d) = 12.0 (same as Dijkstra)

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic astar_wedge/3.
astar_wedge(a, b, 10.0).
astar_wedge(b, c, 1.0).
astar_wedge(c, d, 1.0).
astar_wedge(a, d, 100.0).

:- dynamic my_astar/3.
my_astar(_, _, _) :- fail.

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
    format(atom(Pat), "@~w_code = private constant \\[(?<n>\\d+) x %Instruction\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    format(atom(Pat), "@~w_labels = private constant \\[(?<n>\\d+) x i32\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

test_astar_executes :-
    format('--- A* execution via llc + clang (zero heuristic) ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_astar_test
    ;  format('  SKIP: clang or llc not found~n')
    ).

run_astar_test :-
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_astar/3],
        [ module_name('astar_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              my_astar/3 - astar_shortest_path4 - [weight_pred(astar_wedge/3)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Structural checks.
    ( sub_string(Src, _, _, _, 'define i1 @wam_astar4_kernel_impl(%WamState* %vm, i32 %instance)')
    -> format('  PASS: concrete astar4 impl substituted~n')
    ;  format('  FAIL: astar4 impl missing~n'), throw(no_astar_impl)
    ),
    ( sub_string(Src, _, _, _, 'zeroinitializer')
    -> format('  PASS: zero heuristic array emitted~n')
    ;  format('  FAIL: heuristic array missing~n')
    ),
    ( sub_string(Src, _, _, _, 'call i1 @wam_astar4_run')
    -> format('  PASS: instance case calls @wam_astar4_run~n')
    ;  format('  FAIL: @wam_astar4_run call missing~n')
    ),

    % Extract atom IDs and build a driver main().
    extract_atom_ids_from_weighted(Src, 'astar4_inst_my_astar_0_edges',
        [a-b, b-c, c-d, a-d], AtomIds),
    get_dict(a, AtomIds, AId),
    get_dict(d, AtomIds, DId),
    extract_instr_count(Src, my_astar, IC),
    extract_label_count(Src, my_astar, LC),
    format(atom(DriverIR),
'; === M5.10 A* execution driver ===
define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 0, 0
  %a2 = insertvalue %Value %a2_0, i64 ~w, 1
  %a3_0 = insertvalue %Value undef, i32 6, 0
  %a3 = insertvalue %Value %a3_0, i64 0, 1

  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @my_astar_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @my_astar_labels, i32 0, i32 0),
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
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('  FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    % Expected: A*(a, d) = 12.0 → scaled = 1200 & 255 = 176
    ExpectedScaled is truncate(12.0 * 100) /\ 255,
    ( ExitCode =:= ExpectedScaled
    -> format('  PASS: A*(a,d) returned scaled=~w (distance=12.0, matching Dijkstra)~n',
          [ExitCode])
    ;  format('  FAIL: returned ~w (expected ~w for distance 12.0)~n',
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
    catch(test_astar_executes, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
