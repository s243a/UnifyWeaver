:- encoding(utf8).
% test_wam_llvm_reach_execution.pl
% M5.7: Full-pipeline end-to-end execution test.
%
% This is the capstone test for M5 and M5.6. It exercises the entire
% foreign-lowering chain at runtime:
%
%   1. A td3-shaped `reach/3` Prolog predicate is registered via the
%      M5.6a options path (foreign_predicates/1).
%   2. write_wam_llvm_project compiles it to a module containing a
%      `call_foreign transitive_distance3, 3` instruction, a concrete
%      @wam_td3_kernel_impl body that calls @wam_bfs_atom_distance,
%      and an %AtomFactPair edge table.
%   3. The test appends a `main()` function to the module that
%      duplicates @reach's vm setup (same @reach_code and
%      @reach_labels globals), runs the WAM interpreter via
%      @run_loop, then reads the A3 register via
%      @wam_get_reg_payload to retrieve the computed distance.
%   4. The full module is compiled via llc → clang → native binary
%      and executed. The exit code carries the distance.
%
% The test graph uses deterministic atom IDs extracted from the
% generated fact table via post-compile inspection: we don't assume
% any particular intern_atom ordering, we just parse what the
% compile pipeline chose and feed those numeric IDs to main().

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic edge/2.
edge(p, q).
edge(q, r).
edge(r, s).
edge(p, t).

:- dynamic reach/3.
reach(_, _, _) :- fail.

host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, TripleStrRaw),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _, fail)
    -> split_string(TripleStrRaw, "", "\n\r\t ", [TripleStr]),
       atom_string(Triple, TripleStr)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

%% extract_atom_id(+Src, +AtomLabel, -Id)
%
%  Find the interned atom ID for a specific atom in the generated
%  module by searching for a comment we emit near the top of the
%  fact table. Since llvm_emit_atom_fact2_table doesn't currently
%  emit per-row comments, we instead search the file for a
%  %AtomFactPair entry whose position in the array tells us which
%  (from, to) pair it is. The fact order matches the edge/2 clause
%  order, so:
%
%     fact 0: p -> q    (first `from` = id of p, first `to` = id of q)
%     fact 1: q -> r
%     fact 2: r -> s
%     fact 3: p -> t
%
%  We parse out the four pairs and map letters to the first
%  occurrence's ID.
extract_atom_ids(Src, AtomIds) :-
    re_foldl([Match, Acc, [FromId - ToId | Acc]]>>(
        get_dict(from, Match, FromIdStr),
        get_dict(to, Match, ToIdStr),
        number_string(FromId, FromIdStr),
        number_string(ToId, ToIdStr)
    ),
    "AtomFactPair \\{ i64 (?<from>\\d+), i64 (?<to>\\d+) \\}",
    Src, [], PairsRev, []),
    reverse(PairsRev, Pairs),
    % Fact order matches edge/2 clause order:
    %   [p->q, q->r, r->s, p->t]
    Pairs = [P-Q, Q2-R, R2-S, P2-T],
    P = P2, Q = Q2, R = R2,  % sanity
    _ = S,  % suppress unused warning
    AtomIds = atom_ids{p:P, q:Q, r:R, s:S, t:T}.

%% extract_instr_count(+Src, -Count)
%  Reach has a call_foreign + proceed = 2 instructions. We parse it
%  from the `@reach_code = private constant [N x %Instruction]` line
%  so the test doesn't silently break if the compile pipeline changes.
extract_instr_count(Src, Count) :-
    re_matchsub("@reach_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
        Src, Match, []),
    get_dict(n, Match, NStr),
    number_string(Count, NStr).

extract_label_count(Src, Count) :-
    re_matchsub("@reach_labels = private constant \\[(?<n>\\d+) x i32\\]",
        Src, Match, []),
    get_dict(n, Match, NStr),
    number_string(Count, NStr).

%% build_reach_driver(+InstrCount, +LabelArraySize, +StartId, +TargetId, -DriverIR)
%
%  A main() that mirrors @reach's vm setup but then reads A3 back.
build_reach_driver(InstrCount, LabelArraySize, StartId, TargetId, DriverIR) :-
    format(atom(DriverIR),
'; === M5.7 reach execution driver ===
define i32 @main() {
entry:
  ; Build %Value atom for start.
  %a1.0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1.0, i64 ~w, 1

  ; Build %Value atom for target.
  %a2.0 = insertvalue %Value undef, i32 0, 0
  %a2 = insertvalue %Value %a2.0, i64 ~w, 1

  ; Build %Value unbound for result slot.
  %a3.0 = insertvalue %Value undef, i32 6, 0
  %a3 = insertvalue %Value %a3.0, i64 0, 1

  ; Create a fresh vm backed by reach_code / reach_labels.
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @reach_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @reach_labels, i32 0, i32 0),
      i32 0)

  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %a3)

  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss

hit:
  ; Read A3 (register index 2) payload back as the computed distance.
  %dist = call i64 @wam_get_reg_payload(%WamState* %vm, i32 2)
  %dist32 = trunc i64 %dist to i32
  ret i32 %dist32

miss:
  ret i32 255
}
',
    [StartId, TargetId,
     InstrCount, InstrCount, InstrCount,
     LabelArraySize, LabelArraySize]).

run_reach_for(Label, StartAtom, TargetAtom, Expected) :-
    format('  testing ~w: reach(~w, ~w, _) expected ~w~n',
        [Label, StartAtom, TargetAtom, Expected]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:reach/3],
        [ module_name('reach_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              reach/3 - transitive_distance3 - [edge_pred(edge/2)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, InstrCount),
    extract_label_count(Src, LabelArraySize),
    extract_atom_ids(Src, AtomIds),
    get_dict(StartAtom, AtomIds, StartId),
    get_dict(TargetAtom, AtomIds, TargetId),
    build_reach_driver(InstrCount, LabelArraySize, StartId, TargetId, DriverIR),
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
    -> format('    FAIL: llc failed (exit=~w)~n', [LlcExit]),
       ExitCode = -1
    ;  format(atom(ClangCmd),
           'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang link failed (exit=~w)~n', [ClangExit]),
          ExitCode = -1
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

test_reach_executes :-
    format('--- Full-pipeline foreign-lowered reach/3 execution ---~n'),
    ( process_which('clang'), process_which('llc')
    -> % Graph: p -> q -> r -> s, plus p -> t
       run_reach_for('three-hop path p->s',  p, s, 3),
       run_reach_for('two-hop path p->r',    p, r, 2),
       run_reach_for('direct edge p->q',     p, q, 1),
       run_reach_for('direct edge p->t',     p, t, 1),
       run_reach_for('self hit p->p',        p, p, 0)
    ;  format('  SKIP: clang or llc not found on PATH~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

test_all :-
    catch(test_reach_executes, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
