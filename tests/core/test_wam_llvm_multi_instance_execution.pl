:- encoding(utf8).
% test_wam_llvm_multi_instance_execution.pl
% M5.8: End-to-end execution test for multi-instance foreign dispatch.
%
% Registers *two* td3 predicates that use *different* edge predicates
% (different graphs), compiles them together into one LLVM module,
% then executes each one and verifies that each gets the right answer
% for its own graph. This is the scenario that M5.6's "first spec wins"
% limitation silently got wrong, and that M5.8 fixes by assigning each
% predicate a unique instance_id and dispatching to its own edge table
% via the `switch i32 %instance` inside @wam_td3_kernel_impl.
%
% The two graphs:
%
%   people_edge/2:      alice -> bob -> carol -> dave
%                       alice -> eve
%   pipeline_edge/2:    src -> mid -> sink
%                       src -> side
%
% Expected results:
%   path_people(alice, dave, D)  -> D = 3
%   path_people(alice, eve, D)   -> D = 1
%   pipeline(src, sink, D)       -> D = 2
%   pipeline(src, side, D)       -> D = 1
%
% If the M5.6 first-spec-wins bug were still present, one of the
% kernels would call into the wrong table and return nonsense.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic people_edge/2.
people_edge(alice, bob).
people_edge(bob, carol).
people_edge(carol, dave).
people_edge(alice, eve).

:- dynamic pipeline_edge/2.
pipeline_edge(src, mid).
pipeline_edge(mid, sink).
pipeline_edge(src, side).

:- dynamic path_people/3.
path_people(_, _, _) :- fail.

:- dynamic pipeline/3.
pipeline(_, _, _) :- fail.

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

%% extract_atom_ids(+Src, +TableGlobalName, +Labels, -AtomIds).
%
%  Parse a specific %AtomFactPair global's body from the generated
%  module and pair each entry with the corresponding label from the
%  caller's known fact order. Returns a dict label:id.
%
%  We isolate the section of text starting right AFTER the array-type
%  declaration (which itself contains a `]`) and ending at the first
%  newline-followed-`]` — that's the initializer's closing bracket.
extract_atom_ids(Src, TableGlobalName, Labels, AtomIds) :-
    format(atom(StartMarkerA), '@~w = private constant', [TableGlobalName]),
    once(sub_string(Src, StartIdx, _, _, StartMarkerA)),
    sub_string(Src, StartIdx, _, 0, FromStart),
    % Skip past the type declaration's closing `]` and its
    % opening `[` for the initializer.
    once(sub_string(FromStart, AfterTypeIdx, 3, _, "] [")),
    BodyStart is AfterTypeIdx + 3,
    sub_string(FromStart, BodyStart, _, 0, FromBody),
    % The initializer's closing bracket is the first `\n]` in FromBody.
    once(sub_string(FromBody, CloseIdx, 2, _, "\n]")),
    sub_string(FromBody, 0, CloseIdx, _, TableBody),
    re_foldl([Match, Acc, [FromId - ToId | Acc]]>>(
        get_dict(from, Match, FromIdStr),
        get_dict(to, Match, ToIdStr),
        number_string(FromId, FromIdStr),
        number_string(ToId, ToIdStr)
    ),
    "AtomFactPair \\{ i64 (?<from>\\d+), i64 (?<to>\\d+) \\}",
    TableBody, [], PairsRev, []),
    reverse(PairsRev, Pairs),
    zip_facts_to_ids(Labels, Pairs, Bindings),
    dict_pairs(AtomIds, atom_ids, Bindings).

%% zip_facts_to_ids(+Labels, +Pairs, -Bindings).
%  Walk both lists in lockstep collecting first-seen bindings of
%  label -> id.
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
    format(atom(Pat),
        "@~w_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
        [PredName]),
    re_matchsub(Pat, Src, Match, []),
    get_dict(n, Match, NStr),
    number_string(Count, NStr).

extract_label_count(Src, PredName, Count) :-
    format(atom(Pat),
        "@~w_labels = private constant \\[(?<n>\\d+) x i32\\]",
        [PredName]),
    re_matchsub(Pat, Src, Match, []),
    get_dict(n, Match, NStr),
    number_string(Count, NStr).

%% build_multi_main(+Specs, -DriverIR)
%
%  Specs is a list of main_case(PredName, StartId, TargetId, Expected).
%  The generated main() runs each case in sequence and accumulates a
%  bitmap: bit N is set if case N returned its Expected value. We
%  return the bitmap as the process exit code (so all 4 cases correct
%  = exit 15 = 0b1111).
build_multi_main(Specs, InstrCounts, LabelCounts, DriverIR) :-
    build_multi_cases(Specs, InstrCounts, LabelCounts, 0, CasesIR),
    format(atom(DriverIR),
'; === M5.8 multi-instance execution driver ===
define i32 @main() {
entry:
  %ok_bitmap = alloca i32
  store i32 0, i32* %ok_bitmap
~w

  %final = load i32, i32* %ok_bitmap
  ret i32 %final
}
', [CasesIR]).

build_multi_cases([], _, _, _, '').
build_multi_cases([main_case(PredName, StartId, TargetId, Expected) | Rest],
        [InstrCount | IRest], [LabelCount | LRest], Index, Out) :-
    Bit is 1 << Index,
    % Build the case IR piece-by-piece. Each block uses simple
    % format/3 with few placeholders, which is easier to keep in
    % sync than one massive format string.
    format(atom(Header),
'  ; --- case ~w: ~w(~w, ~w) expect ~w ---~n',
        [Index, PredName, StartId, TargetId, Expected]),
    format(atom(Atoms),
'  %a1_~w_0 = insertvalue %Value undef, i32 0, 0
  %a1_~w = insertvalue %Value %a1_~w_0, i64 ~w, 1
  %a2_~w_0 = insertvalue %Value undef, i32 0, 0
  %a2_~w = insertvalue %Value %a2_~w_0, i64 ~w, 1
  %a3_~w_0 = insertvalue %Value undef, i32 6, 0
  %a3_~w = insertvalue %Value %a3_~w_0, i64 0, 1
',
        [Index,
         Index, Index, StartId,
         Index,
         Index, Index, TargetId,
         Index,
         Index, Index]),
    format(atom(VmSetup),
'  %vm_~w = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @~w_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @~w_labels, i32 0, i32 0),
      i32 0)
',
        [Index,
         InstrCount, InstrCount, PredName, InstrCount,
         LabelCount, LabelCount, PredName]),
    format(atom(RegSets),
'  call void @wam_set_reg(%WamState* %vm_~w, i32 0, %Value %a1_~w)
  call void @wam_set_reg(%WamState* %vm_~w, i32 1, %Value %a2_~w)
  call void @wam_set_reg(%WamState* %vm_~w, i32 2, %Value %a3_~w)
  %ok_~w = call i1 @run_loop(%WamState* %vm_~w)
  br i1 %ok_~w, label %hit_~w, label %skip_~w

hit_~w:
  %dist_~w = call i64 @wam_get_reg_payload(%WamState* %vm_~w, i32 2)
  %dist32_~w = trunc i64 %dist_~w to i32
  %match_~w = icmp eq i32 %dist32_~w, ~w
  br i1 %match_~w, label %setbit_~w, label %skip_~w

setbit_~w:
  %cur_~w = load i32, i32* %ok_bitmap
  %new_~w = or i32 %cur_~w, ~w
  store i32 %new_~w, i32* %ok_bitmap
  br label %skip_~w

skip_~w:
',
        [Index, Index,
         Index, Index,
         Index, Index,
         Index, Index,
         Index, Index, Index,
         Index,
         Index, Index,
         Index, Index,
         Index, Index, Expected,
         Index, Index, Index,
         Index,
         Index,
         Index, Index, Bit,
         Index,
         Index,
         Index]),
    atomic_list_concat([Header, Atoms, VmSetup, RegSets], CaseIR),
    NextIndex is Index + 1,
    build_multi_cases(Rest, IRest, LRest, NextIndex, RestIR),
    atom_concat(CaseIR, RestIR, Out).

test_multi_instance_runs :-
    format('--- two td3 predicates with different graphs coexist ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_multi_test
    ;  format('  SKIP: clang or llc not found~n')
    ).

run_multi_test :-
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:path_people/3, user:pipeline/3],
        [ module_name('multi_inst_test'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              path_people/3 - transitive_distance3 - [edge_pred(people_edge/2)],
              pipeline/3    - transitive_distance3 - [edge_pred(pipeline_edge/2)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Both instances registered.
    ( sub_string(Src, _, _, _, 'i32 0, label %inst_0'),
      sub_string(Src, _, _, _, 'i32 1, label %inst_1')
    -> format('  PASS: @wam_td3_kernel_impl switches on instance 0 and 1~n')
    ;  format('  FAIL: instance switch cases missing~n'),
       throw(switch_cases_missing)
    ),

    % Both edge tables emitted with distinct names.
    ( sub_string(Src, _, _, _, '@td3_inst_path_people_0_edges'),
      sub_string(Src, _, _, _, '@td3_inst_pipeline_1_edges')
    -> format('  PASS: both per-instance edge tables emitted~n')
    ;  format('  FAIL: edge table globals missing~n'),
       throw(edge_tables_missing)
    ),

    % Extract atom IDs from each table.
    extract_atom_ids(Src, 'td3_inst_path_people_0_edges',
        [alice-bob, bob-carol, carol-dave, alice-eve], PeopleIds),
    extract_atom_ids(Src, 'td3_inst_pipeline_1_edges',
        [src-mid, mid-sink, src-side], PipeIds),
    format('  PASS: atom IDs extracted: people=~w  pipeline=~w~n',
        [PeopleIds, PipeIds]),

    extract_instr_count(Src, path_people, PeopleInstr),
    extract_instr_count(Src, pipeline, PipeInstr),
    extract_label_count(Src, path_people, PeopleLabels),
    extract_label_count(Src, pipeline, PipeLabels),

    % Build a main() that runs 4 queries across the two predicates.
    get_dict(alice, PeopleIds, Alice),
    get_dict(dave,  PeopleIds, Dave),
    get_dict(eve,   PeopleIds, Eve),
    get_dict(src,   PipeIds, Src1),
    get_dict(sink,  PipeIds, Sink),
    get_dict(side,  PipeIds, Side),
    Cases = [
        main_case(path_people, Alice, Dave, 3),
        main_case(path_people, Alice, Eve,  1),
        main_case(pipeline,    Src1,  Sink, 2),
        main_case(pipeline,    Src1,  Side, 1)
    ],
    build_multi_main(Cases,
        [PeopleInstr, PeopleInstr, PipeInstr, PipeInstr],
        [PeopleLabels, PeopleLabels, PipeLabels, PipeLabels],
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
    -> format('  FAIL: llc failed exit=~w, see ~w.llc.err~n', [LlcExit, LLPath]),
       ExitCode = -1
    ;  format(atom(ClangCmd),
           'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('  FAIL: clang link failed exit=~w~n', [ClangExit]),
          ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    % Bit N set if case N returned its expected value. All 4 → 15.
    ( ExitCode =:= 15
    -> format('  PASS: all 4 cases correct (bitmap=~w)~n', [ExitCode])
    ;  format('  FAIL: bitmap=~w (expected 15 = 0b1111)~n', [ExitCode]),
       format('        bit0 (alice->dave=3): ~w~n', [ExitCode /\ 1]),
       format('        bit1 (alice->eve=1):  ~w~n', [(ExitCode >> 1) /\ 1]),
       format('        bit2 (src->sink=2):   ~w~n', [(ExitCode >> 2) /\ 1]),
       format('        bit3 (src->side=1):   ~w~n', [(ExitCode >> 3) /\ 1])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

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
    catch(test_multi_instance_runs, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
