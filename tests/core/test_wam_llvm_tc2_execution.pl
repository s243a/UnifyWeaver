:- encoding(utf8).
% test_wam_llvm_tc2_execution.pl
% End-to-end execution test for transitive_closure2 (boolean
% reachability). tc2 is the simplest kernel — it just checks whether
% a path exists from start to target, without computing the distance.
%
% Tests:
%   1. Reachable pair → exit 1 (true)
%   2. Unreachable pair → exit 0 (false)
%   3. Self-reachability → exit 1 (start == target fast path)

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic tc_edge/2.
tc_edge(a, b).
tc_edge(b, c).
tc_edge(c, d).

:- dynamic can_reach/2.
can_reach(_, _) :- fail.

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

extract_instr_count(Src, P, C) :-
    format(atom(Pat), "@~w_code = private constant \\[(?<n>\\d+) x %Instruction\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    format(atom(Pat), "@~w_labels = private constant \\[(?<n>\\d+) x i32\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

run_tc2_case(Label, StartAtom, TargetAtom, Expected) :-
    format('  testing ~w: can_reach(~w, ~w) expected ~w~n',
        [Label, StartAtom, TargetAtom, Expected]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:can_reach/2],
        [ module_name('tc2_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              can_reach/2 - transitive_closure2 - [edge_pred(tc_edge/2)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_atom_id_for(Src, 'tc2_inst_can_reach_0_edges',
        [a-b, b-c, c-d], AtomIds),
    get_dict(StartAtom, AtomIds, StartId),
    get_dict(TargetAtom, AtomIds, TargetId),
    extract_instr_count(Src, can_reach, IC),
    extract_label_count(Src, can_reach, LC),
    % tc2 is arity 2: A1=start, A2=target, no A3 result register.
    % main() returns 1 if run_loop succeeds (reachable), 0 if not.
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 0, 0
  %a2 = insertvalue %Value %a2_0, i64 ~w, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @can_reach_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @can_reach_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  %r = zext i1 %ok to i32
  ret i32 %r
}
',
        [StartId, TargetId, IC, IC, IC, LC, LC]),
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

test_tc2_executes :-
    format('--- transitive_closure2 execution ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_tc2_case('reachable a->d',    a, d, 1),
       run_tc2_case('direct edge a->b',  a, b, 1),
       run_tc2_case('self a->a',         a, a, 1)
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
    catch(test_tc2_executes, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
