:- encoding(utf8).
% test_wam_llvm_wsp3_stream_execution.pl
% End-to-end execution test for stream-mode weighted_shortest_path3.
% When A2 is unbound (tag=6), wsp3 runs full Dijkstra and enumerates
% all reachable (target, best_distance) pairs via the paired iterator.
%
% Graph: a -3-> b -4-> c -5-> d, a -100-> d (direct but heavy)
%
% Tests:
%   1. Stream from 'a': 3 reachable nodes (b,c,d) = 3 pairs.
%   2. Stream from 'c': 1 reachable node (d) = 1 pair.
%   3. Stream from 'd': 0 reachable (sink) = 0 pairs.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic wedge/3.
wedge(a, b, 3.0).
wedge(b, c, 4.0).
wedge(c, d, 5.0).
wedge(a, d, 100.0).

:- dynamic wpath/3.
wpath(_, _, _) :- fail.

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

extract_instr_count(Src, P, C) :-
    format(atom(Pat), "@~w_code = private constant \\[(?<n>\\d+) x %Instruction\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    format(atom(Pat), "@~w_labels = private constant \\[(?<n>\\d+) x i32\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

% We need to find the atom ID for the start node. For wsp3 the edge table
% is %WeightedFact, so we parse differently.
extract_weighted_atom_ids(Src, TablePattern, Labels, AtomIds) :-
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
    "WeightedFact \\{ i64 (?<from>\\d+), i64 (?<to>\\d+), double [^}]+ \\}",
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

run_stream_wsp3_case(Label, StartAtom, Expected) :-
    format('  testing ~w: wpath(~w, X, D) expected ~w results~n',
        [Label, StartAtom, Expected]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:wpath/3],
        [ module_name('wsp3_stream'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              wpath/3 - weighted_shortest_path3 - [weight_pred(wedge/3)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_weighted_atom_ids(Src, 'wsp3_inst_wpath_0_edges',
        [a-b, b-c, c-d, a-d], AtomIds),
    get_dict(StartAtom, AtomIds, StartId),
    extract_instr_count(Src, wpath, IC),
    extract_label_count(Src, wpath, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @wpath_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @wpath_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %count_entry, label %no_results

count_entry:
  br label %count_loop

count_loop:
  %count = phi i32 [1, %count_entry], [%count_inc, %got_next]
  %bt_ok = call i1 @backtrack(%WamState* %vm)
  br i1 %bt_ok, label %got_next, label %done

got_next:
  %count_inc = add i32 %count, 1
  br label %count_loop

done:
  ret i32 %count

no_results:
  ret i32 0
}
',
        [StartId, IC, IC, IC, LC, LC]),
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
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ( ExitCode =:= Expected
    -> format('    PASS: ~w returned ~w~n', [Label, ExitCode])
    ;  format('    FAIL: ~w returned ~w (expected ~w)~n', [Label, ExitCode, Expected])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= Expected).

test_wsp3_stream :-
    format('--- weighted_shortest_path3 stream mode ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_stream_wsp3_case('3 from a', a, 3),
       run_stream_wsp3_case('1 from c', c, 1),
       run_stream_wsp3_case('0 from d', d, 0)
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
    catch(test_wsp3_stream, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
