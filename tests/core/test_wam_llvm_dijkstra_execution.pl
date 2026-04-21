:- encoding(utf8).
% test_wam_llvm_dijkstra_execution.pl
% M5.9: End-to-end execution test for weighted_shortest_path3 via the
% @wam_dijkstra_weighted_distance helper.
%
% This is the first test that actually runs the M5.3 Dijkstra helper
% at runtime. Prior M5.3 tests only validated that the IR parses. M5.9
% ports the full M5.6+M5.8 compile-pipeline infrastructure to
% weighted_shortest_path3 (via the M5.9 @wam_wsp3_run helper, the
% weak default @wam_wsp3_kernel_impl, and the substitution path in
% substitute_foreign_kernel_impls) and then executes a compiled
% binary to verify it computes the right answer.
%
% Critical test design choice: the graph is constructed so that the
% GREEDY first-discovered path is NOT the optimal shortest path. If
% @wam_dijkstra_weighted_distance's extract-min logic is broken (e.g.
% it returned the first-reached target distance like BFS would), the
% test would catch it. The graph:
%
%     a --10--> b --1--> c --1--> d    (three-hop path, total 12)
%     a --100-> d                      (direct edge, total 100)
%
% Optimal shortest path from a to d is 12 (three hops), NOT 100
% (direct). BFS would pick the direct edge if evaluated by hop count,
% but Dijkstra's extract-min picks the lowest accumulated weight.
%
% A second test uses atoms with irrational-looking float weights to
% exercise the @wam_set_reg_double bitcast path (float payload via
% the M5.1 FFI bridge helper — first time this path is exercised at
% runtime).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% Weight predicate — the fact table Dijkstra scans. Three-hop path
% (a->b->c->d total weight 12) is cheaper than the direct edge (a->d
% weight 100), so Dijkstra must find 12.
:- dynamic wedge/3.
wedge(a, b, 10.0).
wedge(b, c, 1.0).
wedge(c, d, 1.0).
wedge(a, d, 100.0).

% A second weight predicate for the "simple direct edge" case.
:- dynamic wedge_simple/3.
wedge_simple(x, y, 2.5).
wedge_simple(y, z, 3.75).

:- dynamic wsp/3.
wsp(_, _, _) :- fail.

:- dynamic wsp_simple/3.
wsp_simple(_, _, _) :- fail.

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

%% extract_atom_ids_from_weighted(+Src, +TableName, +Labels, -AtomIds)
%
%  Parse a %WeightedFact global's body to recover the interned atom
%  IDs used for its from/to columns. Labels is the caller's known
%  fact-order list of (FromLabel-ToLabel) pairs. Weights are ignored.
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
        get_dict(from, Match, FromIdStr),
        get_dict(to, Match, ToIdStr),
        number_string(FromId, FromIdStr),
        number_string(ToId, ToIdStr)
    ),
    "WeightedFact \\{ i64 (?<from>\\d+), i64 (?<to>\\d+), double",
    TableBody, [], PairsRev, []),
    reverse(PairsRev, Pairs),
    zip_facts_to_ids(Labels, Pairs, Bindings),
    dict_pairs(AtomIds, atom_ids, Bindings).

zip_facts_to_ids(Labels, Pairs, Bindings) :-
    zip_facts_loop(Labels, Pairs, [], Bindings).

zip_facts_loop([], _, Acc, Sorted) :-
    sort(1, @<, Acc, Sorted).
zip_facts_loop([FromLabel-ToLabel | LRest], [FromId-ToId | PRest], Acc, Out) :- !,
    add_binding(FromLabel, FromId, Acc, Acc1),
    add_binding(ToLabel, ToId, Acc1, Acc2),
    zip_facts_loop(LRest, PRest, Acc2, Out).
zip_facts_loop(_, _, Acc, Sorted) :-
    sort(1, @<, Acc, Sorted).

add_binding(Label, _, Acc, Acc) :-
    memberchk(Label-_, Acc), !.
add_binding(Label, Id, Acc, [Label-Id | Acc]).

extract_instr_count(Src, PredName, Count) :-
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, Match, []),
    get_dict(n, Match, NStr),
    number_string(Count, NStr).

extract_label_count(Src, PredName, Count) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, Match, []),
    get_dict(n, Match, NStr),
    number_string(Count, NStr).

%% build_wsp_driver(+PredName, +InstrCount, +LabelArraySize,
%%                  +StartId, +TargetId, +ExpectedScaled, -DriverIR)
%
%  A main() that mirrors @wsp's vm setup but reads A3 back as a
%  double, multiplies by 100, converts to i32, and returns it. The
%  test caller compares the exit code to the expected integer value
%  (expected-distance * 100 truncated to int), which lets us check
%  double distances via a 0-255 exit code.
%
%  Example: true distance 12.0 → exit 1200 & 0xff = 176
%           true distance 2.5  → exit 250 & 0xff = 250
%
%  We pre-compute the expected scaled value in Prolog and compare
%  exactly. The `& 0xff` truncation means distances above 2.55 require
%  a different expected computation — the test handles that in
%  run_wsp_for.
build_wsp_driver(PredName, InstrCount, LabelArraySize, StartId, TargetId,
        DriverIR) :-
    format(atom(DriverIR),
'; === M5.9 wsp execution driver ===
define i32 @main() {
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
  ; Read A3 tag: must be 2 (Float) for success.
  %tag = call i32 @wam_get_reg_tag(%WamState* %vm, i32 2)
  %tag_ok = icmp eq i32 %tag, 2
  br i1 %tag_ok, label %read_double, label %wrong_tag

read_double:
  %dist = call double @wam_get_reg_double(%WamState* %vm, i32 2)
  ; Scale by 100 and truncate to i32 so we can return the value
  ; through the process exit code, which is unsigned 8-bit on most
  ; platforms but reads as i32 from the LLVM side.
  %dist100 = fmul double %dist, 1.0e2
  %dist_i32 = fptosi double %dist100 to i32
  ret i32 %dist_i32

wrong_tag:
  ; If the impl wrote the wrong tag, return a sentinel we can tell
  ; apart from any valid scaled distance.
  ret i32 254

miss:
  ret i32 255
}
',
        [StartId, TargetId,
         InstrCount, InstrCount, InstrCount,
         LabelArraySize, LabelArraySize]).

run_wsp_for(Label, PredIndicator, WeightPred, Facts, StartAtom, TargetAtom,
        ExpectedDist) :-
    format('  testing ~w: ~w(~w, ~w, _) expected ~w~n',
        [Label, PredIndicator, StartAtom, TargetAtom, ExpectedDist]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    PredIndicator = PredName/_,
    write_wam_llvm_project(
        [user:PredIndicator],
        [ module_name('wsp_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              PredIndicator - weighted_shortest_path3 - [weight_pred(WeightPred)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    % Find the instance-0 table name matching the predicate.
    sanitize_for_test(PredName, SanePred),
    format(atom(TableName), 'wsp3_inst_~w_0_edges', [SanePred]),
    extract_atom_ids_from_weighted(Src, TableName, Facts, AtomIds),
    get_dict(StartAtom, AtomIds, StartId),
    get_dict(TargetAtom, AtomIds, TargetId),
    extract_instr_count(Src, PredName, InstrCount),
    extract_label_count(Src, PredName, LabelArraySize),
    build_wsp_driver(PredName, InstrCount, LabelArraySize, StartId, TargetId,
        DriverIR),
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
    -> format('    FAIL: llc exit=~w~n', [LlcExit]),
       ExitCode = -1
    ;  format(atom(ClangCmd),
           'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang link exit=~w~n', [ClangExit]),
          ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    % Expected exit code = expected_distance * 100, truncated to 8 bits.
    ExpectedScaled is truncate(ExpectedDist * 100) /\ 255,
    ( ExitCode =:= ExpectedScaled
    -> format('    PASS: ~w returned scaled=~w (distance ≈ ~w)~n',
          [Label, ExitCode, ExpectedDist])
    ;  format('    FAIL: ~w returned ~w (expected scaled ~w for distance ~w)~n',
          [Label, ExitCode, ExpectedScaled, ExpectedDist])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= ExpectedScaled).

sanitize_for_test(Atom, Sane) :-
    atom_codes(Atom, Codes),
    maplist(sanitize_code_t, Codes, SaneCodes),
    atom_codes(Sane, SaneCodes).
sanitize_code_t(C, C) :-
    ( (C >= 0'a, C =< 0'z) ; (C >= 0'A, C =< 0'Z)
    ; (C >= 0'0, C =< 0'9) ; C =:= 0'_ ), !.
sanitize_code_t(_, 0'_).

test_dijkstra_executes :-
    format('--- Dijkstra execution via llc + clang ---~n'),
    ( process_which('clang'), process_which('llc')
    -> % Critical test: greedy direct edge (100) vs three-hop path (12).
       % Dijkstra must pick 12. If the kernel were BFS-shaped
       % (first-reached wins), it would return 100 and this would fail.
       run_wsp_for('three-hop beats direct (12 < 100)',
           wsp/3, wedge/3,
           [a-b, b-c, c-d, a-d],
           a, d, 12.0),
       % Sanity: direct-edge path returns its weight unchanged.
       run_wsp_for('single direct edge',
           wsp_simple/3, wedge_simple/3,
           [x-y, y-z],
           x, y, 2.5),
       % Two-hop through simple weights.
       run_wsp_for('two-hop float chain',
           wsp_simple/3, wedge_simple/3,
           [x-y, y-z],
           x, z, 6.25),
       % Self-hit fast path.
       run_wsp_for('self hit',
           wsp_simple/3, wedge_simple/3,
           [x-y, y-z],
           x, x, 0.0)
    ;  format('  SKIP: clang or llc not found~n')
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
    catch(test_dijkstra_executes, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
