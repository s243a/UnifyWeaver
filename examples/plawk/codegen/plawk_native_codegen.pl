% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_native_codegen, [
    plawk_program_native_driver_ir/3
]).

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target',
    [llvm_emit_atom_prefix_guard/5,
     llvm_emit_atom_field_eq_guard/7,
     llvm_emit_atom_field_slice/5]).

%% plawk_program_native_driver_ir(+Program, +InputPath, -DriverIR) is semidet.
%
%  Emit the first native Phase-2 PLAWK driver shape:
%
%      /^PREFIX/ { print $0 }
%      $N == "VALUE" { print $0 }
%      $N == "VALUE" { print $M, $K }
%      $N == "VALUE" { count++ } END { print count }
%      $N == "VALUE" { errors++; matches++ } END { print errors, matches }
%      $N == "ERROR" { errors++ } $N == "WARN" { warnings++ } END { print errors, warnings }
%      { counts[$1]++ } END { print counts["ERROR"], counts["WARN"] }
%      $N == "VALUE" { counts[$M]++ } END { print counts["KEY"] }
%      { total++; counts[$1]++ } END { print total, counts["ERROR"] }
%      { total++ } END { print "total", total }
%
%  The surrounding runtime still comes from write_wam_llvm_project/3. This
%  function emits the target-specific native main that streams the file, lowers
%  the deterministic guard, and prints matching records.
plawk_program_native_driver_ir(
    program([], [rule(Pattern, [print(Fields)])], []),
    InputPath,
    DriverIR
) :-
    plawk_pattern_guard_ir(Pattern, GuardGlobalIR-GuardCallIR),
    plawk_print_action_ir(Fields, PrintActionIR),
    format(atom(RecordIR),
'~w
  br i1 %is_match, label %print_line, label %continue_loop

print_line:
~w
  br label %continue_loop',
        [GuardCallIR, PrintActionIR]),
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_space = private constant [2 x i8] c" \\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
~w
',
        [GuardGlobalIR]),
    plawk_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, '', lowered_match, RecordIR, '',
            success, 'success:\n  ret i32 0'),
        DriverIR).

plawk_program_native_driver_ir(
    program([], Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_mixed_state_plan(Rules, PrintFields, MixedPlan),
    MixedPlan = mixed_plan(ScalarPlan, AssocPlan, _PlannedRules),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_mixed_rule_chain_ir(MixedPlan, RuleGlobalIR, RuleChainIR, RuleCount),
    plawk_state_loop_phi_ir(ScalarPlan, LoopPhiIR),
    plawk_mixed_scalar_next_phi_ir(ScalarPlan, RuleCount, NextPhiIR),
    plawk_mixed_end_print_ir(PrintFields, ScalarPlan, AssocPlan, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [StringGlobalIR, AssocGlobalIR, RuleGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  ret i32 0',
        [EndPrintIR]),
    plawk_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, lowered_mixed,
            RuleChainIR, NextPhiIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program([], Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_scalar_state_plan(Rules, PrintFields, StatePlan),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_scalar_rule_chain_ir(Rules, StatePlan, RuleGlobalIR, RuleChainIR, RuleCount),
    plawk_state_loop_phi_ir(StatePlan, LoopPhiIR),
    plawk_scalar_next_phi_ir(StatePlan, RuleCount, NextPhiIR),
    plawk_scalar_end_print_ir(PrintFields, StatePlan, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w', [StringGlobalIR, RuleGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  ret i32 0',
        [EndPrintIR]),
    plawk_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, LoopPhiIR, lowered_match, RuleChainIR,
            NextPhiIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program([], Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_assoc_runtime_count_plan(Rules, PrintFields, AssocPlan),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_end_print_ir(PrintFields, AssocPlan, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [StringGlobalIR, AssocGlobalIR, AssocRuleGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  ret i32 0',
        [EndPrintIR]),
    plawk_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, EntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', end_print, CloseOkIR),
        DriverIR).

plawk_i64_end_print_globals(SurfaceGlobals, RuntimeGlobals) :-
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_space = private constant [2 x i8] c" \\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
~w

',
        [SurfaceGlobals]).

% Shared native streaming skeleton. Surface-specific lowerers provide globals,
% loop-carried state phis, the per-record lowered block, continuation phis, and
% the close-success block; file open/read/eof/close stays backend infrastructure.
plawk_stream_driver_ir(
    InputPath,
    driver_blocks(RuntimeGlobals, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    plawk_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, '', LoopPhiIR, LoweredLabel, RecordIR,
            ContinueIR, CloseOkLabel, CloseOkIR),
        DriverIR).

plawk_stream_driver_ir(
    InputPath,
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    DriverIR
) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    format(atom(DriverIR),
'@.plawk_surface_path = private constant [~w x i8] c"~w\\00"
@.plawk_surface_eof = private constant [12 x i8] c"end_of_file\\00"
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_surface_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
~w
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
  %line = call %Value @wam_stream_read_line_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %fail_read, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %fail_line_tag

check_eof:
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %eof_s = getelementptr [12 x i8], [12 x i8]* @.plawk_surface_eof, i32 0, i32 0
  %eof_cmp = call i32 @strcmp(i8* %line_s, i8* %eof_s)
  %is_eof = icmp eq i32 %eof_cmp, 0
  br i1 %is_eof, label %close_stream, label %~w

~w:
~w

continue_loop:
~w
  br label %loop

close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %~w, label %fail_close

~w

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 12

fail_close:
  ret i32 16
}
',
        [BytesLen, PathBytes, RuntimeGlobals,
         BytesLen, BytesLen, PathLen, EntrySetupIR, LoopPhiIR, LoweredLabel,
         LoweredLabel, RecordIR, ContinueIR, CloseOkLabel, CloseOkIR]).

llvm_c_bytes([], '').
llvm_c_bytes([Code | Rest], Bytes) :-
    llvm_c_byte(Code, Byte),
    llvm_c_bytes(Rest, Tail),
    atom_concat(Byte, Tail, Bytes).

llvm_c_byte(Code, Byte) :-
    format(atom(Byte), '\\~|~`0t~16r~2+', [Code]).

% State plans keep recognized PLAWK state separate from the LLVM slot numbering.
% Associative arrays use a separate table plan because they are pointer state.
plawk_state_plan_slots(state_plan(Slots), Slots).

plawk_state_slot_count(StatePlan, Count) :-
    plawk_state_plan_slots(StatePlan, Slots),
    length(Slots, Count).

plawk_state_slot_index(StatePlan, Slot, Index) :-
    plawk_state_plan_slots(StatePlan, Slots),
    nth0(Index, Slots, Slot).

plawk_scalar_state_plan(Rules, PrintFields, state_plan(Slots)) :-
    findall(Name,
        ( member(rule(_Pattern, Actions), Rules),
          member(Action, Actions),
          plawk_scalar_increment_action(Action, Name)
        ),
        ActionVars),
    ActionVars \== [],
    findall(Name,
        ( member(Field, PrintFields),
          plawk_scalar_print_expr(Field, Name)
        ),
        PrintVars),
    append(ActionVars, PrintVars, Names0),
    sort(Names0, Names),
    maplist(plawk_scalar_state_slot, Names, Slots).

plawk_scalar_state_slot(Name, scalar_counter(Name)).

plawk_mixed_state_plan(Rules, PrintFields, mixed_plan(ScalarPlan, AssocPlan, PlannedRules)) :-
    plawk_mixed_scalar_state_plan(Rules, PrintFields, ScalarPlan),
    plawk_mixed_assoc_count_plan(Rules, PrintFields, AssocPlan),
    phrase(plawk_mixed_planned_rules(Rules, AssocPlan, 0), PlannedRules),
    PlannedRules \== [].

plawk_mixed_scalar_state_plan(Rules, PrintFields, state_plan(Slots)) :-
    findall(Name,
        ( member(rule(_Pattern, Actions), Rules),
          member(Action, Actions),
          plawk_scalar_increment_action(Action, Name)
        ),
        ActionVars),
    ActionVars \== [],
    findall(Name,
        ( member(Field, PrintFields),
          plawk_scalar_print_expr(Field, Name)
        ),
        PrintVars),
    append(ActionVars, PrintVars, Names0),
    sort(Names0, Names),
    maplist(plawk_scalar_state_slot, Names, Slots).

plawk_mixed_assoc_count_plan(Rules, PrintFields, assoc_plan(Tables, [])) :-
    findall(ArrayName,
        ( member(rule(_Pattern, Actions), Rules),
          member(Action, Actions),
          plawk_assoc_increment_action(Action, ArrayName-_KeyIndex)
        ),
        ActionArrays),
    ActionArrays \== [],
    findall(ArrayName,
        ( member(Field, PrintFields),
          plawk_assoc_print_array(Field, ArrayName)
        ),
        PrintArrays),
    PrintArrays \== [],
    append(ActionArrays, PrintArrays, ArrayNames0),
    sort(ArrayNames0, Tables).

plawk_mixed_planned_rules([], _AssocPlan, _Index) -->
    [].
plawk_mixed_planned_rules([rule(Pattern, Actions) | Rest], assoc_plan(Tables, Actions0), Index) -->
    { plawk_mixed_rule_actions(Actions, ScalarVars, AssocSpecs),
      ( ScalarVars == [], AssocSpecs == []
      -> HasActions = false,
         NextIndex = Index,
         PlannedAssocActions = []
      ;  HasActions = true,
         phrase(plawk_assoc_planned_actions(AssocSpecs, Tables, 0), PlannedAssocActions),
         NextIndex is Index + 1
      )
    },
    ( { HasActions == true }
    -> [mixed_rule(Index, Pattern, ScalarVars, PlannedAssocActions)]
    ;  []
    ),
    plawk_mixed_planned_rules(Rest, assoc_plan(Tables, Actions0), NextIndex).

plawk_mixed_rule_actions(Actions, ScalarVars, AssocSpecs) :-
    maplist(plawk_mixed_increment_action, Actions, TypedActions),
    findall(Name, member(scalar(Name), TypedActions), ScalarVars),
    findall(Spec, member(assoc(Spec), TypedActions), AssocSpecs).

plawk_mixed_increment_action(inc(var(Name)), scalar(Name)).
plawk_mixed_increment_action(inc_assoc(var(ArrayName), field(KeyIndex)), assoc(ArrayName-KeyIndex)) :-
    KeyIndex > 0.

plawk_mixed_rule_chain_ir(mixed_plan(ScalarPlan, _AssocPlan, Rules), GlobalIR, ChainIR, RuleCount) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    phrase(plawk_mixed_rule_chain_lines(Rules, ScalarPlan, 0), Pairs),
    pairs_keys_values(Pairs, GlobalParts, ChainParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_mixed_rule_chain_lines([], _ScalarPlan, _) -->
    [].
plawk_mixed_rule_chain_lines([mixed_rule(Index, Pattern, ScalarVars, AssocActions) | Rest], ScalarPlan, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'rule_~w_apply', [Index]),
      format(atom(DoneLabel), 'rule_~w_done', [Index]),
      plawk_mixed_scalar_rule_input_phi_ir(ScalarPlan, Index, InputPhiIR),
      plawk_mixed_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel,
          NextLabel, InputPhiIR, GuardGlobalIR-GuardIR),
      plawk_scalar_match_update_ir(ScalarPlan, ScalarVars, Index, ScalarUpdateIR),
      plawk_mixed_assoc_apply_ir(Index, AssocActions, DoneLabel, AssocApplyIR),
      ( Index =:= 0
      -> EntryIR = '  br label %rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      format(atom(RuleIR),
'~w~w

~w:
~w
~w

~w:
  br label %~w',
          [EntryIR, GuardIR, ApplyLabel, ScalarUpdateIR, AssocApplyIR,
           DoneLabel, NextLabel]),
      Pair = GuardGlobalIR-RuleIR
    },
    [Pair],
    plawk_mixed_rule_chain_lines(Rest, ScalarPlan, NextIndex).

plawk_mixed_rule_guard_ir(always, Index, RuleLabel, ApplyLabel, NextLabel,
    InputPhiIR, ''-IR) :-
    !,
    format(atom(MatchVar), 'rule_~w_is_match', [Index]),
    format(atom(IR),
'~w:
~w  %~w = icmp eq i1 true, true
  br i1 %~w, label %~w, label %~w',
        [RuleLabel, InputPhiIR, MatchVar, MatchVar, ApplyLabel, NextLabel]).
plawk_mixed_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel, NextLabel,
    InputPhiIR, GuardGlobalIR-IR) :-
    format(atom(MatchVar), 'rule_~w_is_match', [Index]),
    format(atom(GlobalBase), 'plawk_mixed_rule_~w', [Index]),
    format(atom(MatchValue), '%~w', [MatchVar]),
    plawk_pattern_guard_ir(Pattern, GlobalBase, MatchValue,
        GuardGlobalIR-GuardCallIR),
    format(atom(IR),
'~w:
~w~w
  br i1 %~w, label %~w, label %~w',
        [RuleLabel, InputPhiIR, GuardCallIR, MatchVar, ApplyLabel, NextLabel]).

plawk_mixed_assoc_apply_ir(_RuleIndex, [], DoneLabel, IR) :-
    !,
    format(atom(IR), '  br label %~w', [DoneLabel]).
plawk_mixed_assoc_apply_ir(RuleIndex, AssocActions, DoneLabel, IR) :-
    phrase(plawk_assoc_rule_action_lines(RuleIndex, AssocActions, DoneLabel), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_scalar_rule_input_phi_ir(_StatePlan, 0, '') :-
    !.
plawk_mixed_scalar_rule_input_phi_ir(StatePlan, RuleIndex, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_mixed_scalar_rule_input_phi_lines(Slots, RuleIndex, 0), Lines),
    atomic_list_concat(Lines, '\n', LinesIR),
    format(atom(IR), '~w~n', [LinesIR]).

plawk_mixed_scalar_rule_input_phi_lines([], _RuleIndex, _) -->
    [].
plawk_mixed_scalar_rule_input_phi_lines([_Slot | Rest], RuleIndex, SlotIndex) -->
    { PrevRuleIndex is RuleIndex - 1,
      plawk_scalar_rule_input_value(PrevRuleIndex, SlotIndex, PrevFalseValue),
      format(atom(Line),
          '  %rule_~w_in_slot_~w = phi i64 [~w, %rule_~w_match], [%rule_~w_slot_~w, %rule_~w_done]',
          [RuleIndex, SlotIndex, PrevFalseValue, PrevRuleIndex,
           PrevRuleIndex, SlotIndex, PrevRuleIndex]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_mixed_scalar_rule_input_phi_lines(Rest, RuleIndex, NextSlotIndex).

plawk_mixed_scalar_next_phi_ir(StatePlan, RuleCount, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_mixed_scalar_next_phi_lines(Slots, RuleCount, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_scalar_next_phi_lines([], _RuleCount, _) -->
    [].
plawk_mixed_scalar_next_phi_lines([_Slot | Rest], RuleCount, Index) -->
    { LastRuleIndex is RuleCount - 1,
      plawk_scalar_rule_input_value(LastRuleIndex, Index, FalseValue),
      format(atom(Line),
          '  %next_slot_~w = phi i64 [~w, %rule_~w_match], [%rule_~w_slot_~w, %rule_~w_done]',
          [Index, FalseValue, LastRuleIndex, LastRuleIndex, Index, LastRuleIndex]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_mixed_scalar_next_phi_lines(Rest, RuleCount, NextIndex).

plawk_assoc_runtime_count_plan(
    Rules,
    PrintFields,
    assoc_plan(Tables, PlannedRules)
) :-
    maplist(plawk_assoc_rule_action_specs, Rules, RuleSpecs),
    RuleSpecs \== [],
    findall(ArrayName,
        ( member(Field, PrintFields),
          plawk_assoc_print_array(Field, ArrayName)
        ),
        PrintArrays),
    PrintArrays \== [],
    findall(ArrayName,
        ( member(rule(_Pattern, ActionSpecs), RuleSpecs),
          member(ArrayName-_KeyIndex, ActionSpecs)
        ),
        ActionArrays),
    append(ActionArrays, PrintArrays, ArrayNames0),
    sort(ArrayNames0, Tables),
    phrase(plawk_assoc_planned_rules(RuleSpecs, Tables, 0), PlannedRules).

plawk_assoc_rule_action_specs(rule(Pattern, Actions), rule(Pattern, ActionSpecs)) :-
    maplist(plawk_assoc_increment_action, Actions, ActionSpecs),
    ActionSpecs \== [].

plawk_assoc_increment_action(inc_assoc(var(ArrayName), field(KeyIndex)), ArrayName-KeyIndex) :-
    KeyIndex > 0.

plawk_assoc_print_array(assoc(var(ArrayName), string(_Key)), ArrayName).

plawk_assoc_planned_rules([], _Tables, _Index) -->
    [].
plawk_assoc_planned_rules([rule(Pattern, ActionSpecs) | Rest], Tables, Index) -->
    { phrase(plawk_assoc_planned_actions(ActionSpecs, Tables, 0), PlannedActions),
      NextIndex is Index + 1
    },
    [assoc_rule(Index, Pattern, PlannedActions)],
    plawk_assoc_planned_rules(Rest, Tables, NextIndex).

plawk_assoc_planned_actions([], _Tables, _Index) -->
    [].
plawk_assoc_planned_actions([ArrayName-KeyIndex | Rest], Tables, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_action(Index, ArrayName, TableIndex, KeyIndex)],
    plawk_assoc_planned_actions(Rest, Tables, NextIndex).

plawk_assoc_entry_setup_ir(assoc_plan(Tables, _Actions), IR) :-
    phrase(plawk_assoc_entry_setup_lines(Tables, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_entry_setup_lines([], _) -->
    [].
plawk_assoc_entry_setup_lines([_ArrayName | Rest], Index) -->
    { format(atom(Line),
          '  %plawk_assoc_table_~w = call %WamAssocI64Table* @wam_assoc_i64_new(i64 4096)',
          [Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_assoc_entry_setup_lines(Rest, NextIndex).

plawk_assoc_rule_chain_ir(assoc_plan(_Tables, Rules), GlobalIR, ChainIR) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    phrase(plawk_assoc_rule_chain_lines(Rules, 0), Pairs),
    pairs_keys_values(Pairs, GlobalParts, ChainParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_assoc_rule_chain_lines([], _) -->
    [].
plawk_assoc_rule_chain_lines([assoc_rule(Index, Pattern, Actions) | Rest], Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'assoc_rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'assoc_rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'assoc_rule_~w_apply', [Index]),
      plawk_assoc_rule_apply_ir(Index, Actions, NextLabel, ApplyIR),
      ( Index =:= 0
      -> EntryIR = '  br label %assoc_rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      plawk_assoc_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel,
          NextLabel, EntryIR, GuardGlobalIR, BranchIR),
      format(atom(RuleIR),
'~w

~w:
~w',
          [BranchIR, ApplyLabel, ApplyIR]),
      Pair = GuardGlobalIR-RuleIR
    },
    [Pair],
    plawk_assoc_rule_chain_lines(Rest, NextIndex).

plawk_assoc_rule_guard_ir(always, _Index, RuleLabel, ApplyLabel, _NextLabel,
    EntryIR, '', IR) :-
    !,
    format(atom(IR),
'~w~w:
  br label %~w',
        [EntryIR, RuleLabel, ApplyLabel]).
plawk_assoc_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel, NextLabel,
    EntryIR, GuardGlobalIR, IR) :-
    format(atom(MatchVar), 'assoc_rule_~w_is_match', [Index]),
    format(atom(GlobalBase), 'plawk_assoc_rule_~w', [Index]),
    format(atom(MatchValue), '%~w', [MatchVar]),
    plawk_pattern_guard_ir(Pattern, GlobalBase, MatchValue,
        GuardGlobalIR-GuardCallIR),
    format(atom(IR),
'~w~w:
~w
  br i1 %~w, label %~w, label %~w',
        [EntryIR, RuleLabel, GuardCallIR, MatchVar, ApplyLabel, NextLabel]).

plawk_assoc_rule_apply_ir(RuleIndex, Actions, NextLabel, IR) :-
    phrase(plawk_assoc_rule_action_lines(RuleIndex, Actions, NextLabel), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_rule_action_lines(RuleIndex, Actions, NextLabel) -->
    { Actions = [_ | _],
      format(atom(FirstBranch), '  br label %assoc_rule_~w_action_0', [RuleIndex])
    },
    [FirstBranch, ''],
    plawk_assoc_rule_action_blocks(RuleIndex, Actions, NextLabel).

plawk_assoc_rule_action_blocks(_RuleIndex, [], _NextLabel) -->
    [].
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_action(Index, _ArrayName, TableIndex, KeyIndex) | Rest], NextLabel) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(HaveLabel), 'assoc_rule_~w_action_~w_have_key:',
          [RuleIndex, Index]),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_key',
          [RuleIndex, Index]),
      format(atom(Slice),
          '  %assoc_rule_~w_action_~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 32)',
          [RuleIndex, Index, KeyIndex]),
      format(atom(Ptr),
          '  %assoc_rule_~w_action_~w_key_ptr = extractvalue %WamSlice %assoc_rule_~w_action_~w_key_slice, 0',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Len),
          '  %assoc_rule_~w_action_~w_key_len = extractvalue %WamSlice %assoc_rule_~w_action_~w_key_slice, 1',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Missing),
          '  %assoc_rule_~w_action_~w_key_missing = icmp eq i8* %assoc_rule_~w_action_~w_key_ptr, null',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Branch),
          '  br i1 %assoc_rule_~w_action_~w_key_missing, label %~w, label %~w',
          [RuleIndex, Index, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %assoc_rule_~w_action_~w_key_id = call i64 @wam_intern_atom(i8* %assoc_rule_~w_action_~w_key_ptr, i64 %assoc_rule_~w_action_~w_key_len)',
          [RuleIndex, Index, RuleIndex, Index, RuleIndex, Index]),
      format(atom(Inc),
          '  %assoc_rule_~w_action_~w_count = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_rule_~w_action_~w_key_id, i64 1)',
          [RuleIndex, Index, TableIndex, RuleIndex, Index]),
      format(atom(Next), '  br label %~w', [ActionNextLabel])
    },
    [Label, Slice, Ptr, Len, Missing, Branch, '', HaveLabel, KeyId, Inc, Next, ''],
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel).

plawk_assoc_print_key_globals(PrintFields, GlobalIR) :-
    phrase(plawk_assoc_print_key_global_lines(PrintFields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_end_print_string_globals(PrintFields, GlobalIR) :-
    phrase(plawk_end_print_string_global_lines(PrintFields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_end_print_string_global_lines([], _) -->
    [].
plawk_end_print_string_global_lines([string(Value) | Rest], Index) -->
    { string_codes(Value, Codes),
      length(Codes, StringLen),
      BytesLen is StringLen + 1,
      llvm_c_bytes(Codes, Bytes),
      format(atom(Line),
          '@.plawk_end_print_string_~w = private constant [~w x i8] c"~w\\00"',
          [Index, BytesLen, Bytes]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_end_print_string_global_lines(Rest, NextIndex).
plawk_end_print_string_global_lines([Field | Rest], Index) -->
    { \+ Field = string(_),
      NextIndex is Index + 1
    },
    plawk_end_print_string_global_lines(Rest, NextIndex).

plawk_assoc_print_key_global_lines([], _) -->
    [].
plawk_assoc_print_key_global_lines([assoc(var(_ArrayName), string(Key)) | Rest], Index) -->
    { plawk_assoc_key_codes(Key, Codes),
      length(Codes, KeyLen),
      BytesLen is KeyLen + 1,
      llvm_c_bytes(Codes, Bytes),
      format(atom(Line),
          '@.plawk_assoc_print_key_~w = private constant [~w x i8] c"~w\\00"',
          [Index, BytesLen, Bytes]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_assoc_print_key_global_lines(Rest, NextIndex).
plawk_assoc_print_key_global_lines([Field | Rest], Index) -->
    { \+ Field = assoc(var(_), string(_)),
      NextIndex is Index + 1 },
    plawk_assoc_print_key_global_lines(Rest, NextIndex).

plawk_assoc_key_codes(Key, Codes) :-
    string(Key),
    !,
    string_codes(Key, Codes).
plawk_assoc_key_codes(Key, Codes) :-
    atom_codes(Key, Codes).

plawk_assoc_end_print_ir(PrintFields, AssocPlan, IR) :-
    phrase(plawk_assoc_end_print_lines(PrintFields, AssocPlan, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_end_print_lines([], AssocPlan, _) -->
    ['  %end_newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
     '  %printed_end_newline = call i32 (i8*, ...) @printf(i8* %end_newline_fmt)'],
    plawk_assoc_free_lines(AssocPlan).
plawk_assoc_end_print_lines([assoc(var(ArrayName), string(Key)) | Rest], AssocPlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    { plawk_assoc_key_codes(Key, Codes),
      length(Codes, KeyLen),
      BytesLen is KeyLen + 1,
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      format(atom(KeyPtr),
          '  %assoc_end_key_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_assoc_print_key_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(KeyId),
          '  %assoc_end_key_~w_id = call i64 @wam_intern_atom(i8* %assoc_end_key_~w_ptr, i64 ~w)',
          [PrintIndex, PrintIndex, KeyLen]),
      format(atom(Value),
          '  %assoc_end_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_end_key_~w_id)',
          [PrintIndex, TableIndex, PrintIndex]),
      format(atom(FmtPtr),
          '  %assoc_end_i64_fmt_~w = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
          [PrintIndex]),
      format(atom(PrintCall),
          '  %printed_assoc_end_i64_~w = call i32 (i8*, ...) @printf(i8* %assoc_end_i64_fmt_~w, i64 %assoc_end_value_~w)',
          [PrintIndex, PrintIndex, PrintIndex]),
      NextPrintIndex is PrintIndex + 1
    },
    [KeyPtr, KeyId, Value, FmtPtr, PrintCall],
    plawk_assoc_end_print_lines(Rest, AssocPlan, NextPrintIndex).
plawk_assoc_end_print_lines([string(Value) | Rest], AssocPlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_assoc_end_print_lines(Rest, AssocPlan, NextPrintIndex).

plawk_assoc_table_index(assoc_plan(Tables, _Actions), ArrayName, TableIndex) :-
    nth0(TableIndex, Tables, ArrayName).

plawk_assoc_free_lines(assoc_plan(Tables, _Actions)) -->
    plawk_assoc_free_lines(Tables, 0).

plawk_assoc_free_lines([], _) -->
    [].
plawk_assoc_free_lines([_ArrayName | Rest], Index) -->
    { format(atom(Line),
          '  call void @wam_assoc_i64_free(%WamAssocI64Table* %plawk_assoc_table_~w)',
          [Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_assoc_free_lines(Rest, NextIndex).

plawk_mixed_end_print_ir(PrintFields, ScalarPlan, AssocPlan, IR) :-
    phrase(plawk_mixed_end_print_lines(PrintFields, ScalarPlan, AssocPlan, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_end_print_lines([], _ScalarPlan, AssocPlan, _) -->
    ['  %end_newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
     '  %printed_end_newline = call i32 (i8*, ...) @printf(i8* %end_newline_fmt)'],
    plawk_assoc_free_lines(AssocPlan).
plawk_mixed_end_print_lines([var(Name) | Rest], ScalarPlan, AssocPlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    { plawk_state_slot_index(ScalarPlan, scalar_counter(Name), SlotIndex),
      format(atom(FmtPtr),
          '  %end_i64_fmt_~w = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
          [PrintIndex]),
      format(atom(PrintCall),
          '  %printed_end_i64_~w = call i32 (i8*, ...) @printf(i8* %end_i64_fmt_~w, i64 %slot_~w)',
          [PrintIndex, PrintIndex, SlotIndex]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, NextPrintIndex).
plawk_mixed_end_print_lines([assoc(var(ArrayName), string(Key)) | Rest], ScalarPlan, AssocPlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    { plawk_assoc_key_codes(Key, Codes),
      length(Codes, KeyLen),
      BytesLen is KeyLen + 1,
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      format(atom(KeyPtr),
          '  %assoc_end_key_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_assoc_print_key_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(KeyId),
          '  %assoc_end_key_~w_id = call i64 @wam_intern_atom(i8* %assoc_end_key_~w_ptr, i64 ~w)',
          [PrintIndex, PrintIndex, KeyLen]),
      format(atom(Value),
          '  %assoc_end_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_end_key_~w_id)',
          [PrintIndex, TableIndex, PrintIndex]),
      format(atom(FmtPtr),
          '  %assoc_end_i64_fmt_~w = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
          [PrintIndex]),
      format(atom(PrintCall),
          '  %printed_assoc_end_i64_~w = call i32 (i8*, ...) @printf(i8* %assoc_end_i64_fmt_~w, i64 %assoc_end_value_~w)',
          [PrintIndex, PrintIndex, PrintIndex]),
      NextPrintIndex is PrintIndex + 1
    },
    [KeyPtr, KeyId, Value, FmtPtr, PrintCall],
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, NextPrintIndex).
plawk_mixed_end_print_lines([string(Value) | Rest], ScalarPlan, AssocPlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, NextPrintIndex).

plawk_scalar_increment_action(inc(var(Name)), Name).

plawk_scalar_print_expr(var(Name), Name).

plawk_scalar_rule_chain_ir(Rules, StatePlan, GlobalIR, ChainIR, RuleCount) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    phrase(plawk_scalar_rule_chain_lines(Rules, StatePlan, 0), Pairs),
    pairs_keys_values(Pairs, GlobalParts, ChainParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_scalar_rule_chain_lines([], _StatePlan, _) -->
    [].
plawk_scalar_rule_chain_lines([rule(Pattern, Actions) | Rest], StatePlan, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'rule_~w_apply', [Index]),
      format(atom(MatchVar), 'rule_~w_is_match', [Index]),
      format(atom(GlobalBase), 'plawk_surface_rule_~w', [Index]),
      format(atom(MatchValue), '%~w', [MatchVar]),
      plawk_pattern_guard_ir(Pattern, GlobalBase, MatchValue,
          GuardGlobalIR-GuardCallIR),
      maplist(plawk_scalar_increment_action, Actions, ActionVars),
      plawk_scalar_rule_input_phi_ir(StatePlan, Index, InputPhiIR),
      plawk_scalar_match_update_ir(StatePlan, ActionVars, Index, MatchUpdateIR),
      ( Index =:= 0
      -> EntryIR = '  br label %rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      format(atom(BranchIR),
'~w~w:
~w~w
  br i1 %~w, label %~w, label %~w

~w:
~w
  br label %~w',
          [EntryIR, RuleLabel, InputPhiIR, GuardCallIR, MatchVar,
           ApplyLabel, NextLabel, ApplyLabel, MatchUpdateIR, NextLabel]),
      Pair = GuardGlobalIR-BranchIR
    },
    [Pair],
    plawk_scalar_rule_chain_lines(Rest, StatePlan, NextIndex).

plawk_scalar_rule_input_phi_ir(_StatePlan, 0, '') :-
    !.
plawk_scalar_rule_input_phi_ir(StatePlan, RuleIndex, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_rule_input_phi_lines(Slots, RuleIndex, 0), Lines),
    atomic_list_concat(Lines, '\n', LinesIR),
    format(atom(IR), '~w~n', [LinesIR]).

plawk_scalar_rule_input_phi_lines([], _RuleIndex, _) -->
    [].
plawk_scalar_rule_input_phi_lines([_Name | Rest], RuleIndex, SlotIndex) -->
    { PrevRuleIndex is RuleIndex - 1,
      plawk_scalar_rule_input_value(PrevRuleIndex, SlotIndex, PrevFalseValue),
      format(atom(Line),
          '  %rule_~w_in_slot_~w = phi i64 [~w, %rule_~w_match], [%rule_~w_slot_~w, %rule_~w_apply]',
          [RuleIndex, SlotIndex, PrevFalseValue, PrevRuleIndex,
           PrevRuleIndex, SlotIndex, PrevRuleIndex]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_scalar_rule_input_phi_lines(Rest, RuleIndex, NextSlotIndex).

plawk_scalar_rule_input_value(0, SlotIndex, Value) :-
    !,
    format(atom(Value), '%slot_~w', [SlotIndex]).
plawk_scalar_rule_input_value(RuleIndex, SlotIndex, Value) :-
    format(atom(Value), '%rule_~w_in_slot_~w', [RuleIndex, SlotIndex]).

plawk_scalar_rule_slot_input(0, SlotIndex, Value) :-
    !,
    format(atom(Value), '%slot_~w', [SlotIndex]).
plawk_scalar_rule_slot_input(RuleIndex, SlotIndex, Value) :-
    format(atom(Value), '%rule_~w_in_slot_~w', [RuleIndex, SlotIndex]).

plawk_state_loop_phi_ir(StatePlan, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_loop_phi_lines(Slots, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_loop_phi_lines([], _) -->
    [].
plawk_scalar_loop_phi_lines([_Slot | Rest], Index) -->
    { format(atom(Line),
          '  %slot_~w = phi i64 [0, %check_handle_value], [%next_slot_~w, %continue_loop]',
          [Index, Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_loop_phi_lines(Rest, NextIndex).

plawk_scalar_match_update_ir(StatePlan, ActionVars, RuleIndex, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_match_update_lines(Slots, ActionVars, RuleIndex, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_match_update_lines([], _ActionVars, _RuleIndex, _) -->
    [].
plawk_scalar_match_update_lines([scalar_counter(Name) | Rest], ActionVars, RuleIndex, SlotIndex) -->
    { plawk_scalar_increment_count(Name, ActionVars, Count),
      plawk_scalar_rule_slot_input(RuleIndex, SlotIndex, InputValue),
      format(atom(Line),
          '  %rule_~w_slot_~w = add i64 ~w, ~w',
          [RuleIndex, SlotIndex, InputValue, Count]),
      NextIndex is SlotIndex + 1
    },
    [Line],
    plawk_scalar_match_update_lines(Rest, ActionVars, RuleIndex, NextIndex).

plawk_scalar_increment_count(Name, ActionVars, Count) :-
    include(==(Name), ActionVars, Matches),
    length(Matches, Count).

plawk_scalar_next_phi_ir(StatePlan, RuleCount, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_next_phi_lines(Slots, RuleCount, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_next_phi_lines([], _RuleCount, _) -->
    [].
plawk_scalar_next_phi_lines([_Slot | Rest], RuleCount, Index) -->
    { LastRuleIndex is RuleCount - 1,
      plawk_scalar_rule_input_value(LastRuleIndex, Index, FalseValue),
      format(atom(Line),
          '  %next_slot_~w = phi i64 [~w, %rule_~w_match], [%rule_~w_slot_~w, %rule_~w_apply]',
          [Index, FalseValue, LastRuleIndex, LastRuleIndex, Index, LastRuleIndex]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_next_phi_lines(Rest, RuleCount, NextIndex).

plawk_scalar_end_print_ir(PrintFields, StatePlan, IR) :-
    phrase(plawk_scalar_end_print_lines(PrintFields, StatePlan, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_end_print_lines([], _StatePlan, _) -->
    ['  %end_newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
     '  %printed_end_newline = call i32 (i8*, ...) @printf(i8* %end_newline_fmt)'].
plawk_scalar_end_print_lines([var(Name) | Rest], StatePlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    { plawk_state_slot_index(StatePlan, scalar_counter(Name), SlotIndex),
      format(atom(FmtPtr),
          '  %end_i64_fmt_~w = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
          [PrintIndex]),
      format(atom(PrintCall),
          '  %printed_end_i64_~w = call i32 (i8*, ...) @printf(i8* %end_i64_fmt_~w, i64 %slot_~w)',
          [PrintIndex, PrintIndex, SlotIndex]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_scalar_end_print_lines(Rest, StatePlan, NextPrintIndex).
plawk_scalar_end_print_lines([string(Value) | Rest], StatePlan, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, NextPrintIndex).

plawk_end_string_print_lines(Value, PrintIndex) -->
    { string_codes(Value, Codes),
      length(Codes, StringLen),
      BytesLen is StringLen + 1,
      format(atom(StringPtr),
          '  %end_string_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_end_print_string_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(FmtPtr),
          '  %end_string_fmt_~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
          [PrintIndex]),
      format(atom(PrintCall),
          '  %printed_end_string_~w = call i32 (i8*, ...) @printf(i8* %end_string_fmt_~w, i8* %end_string_~w_ptr)',
          [PrintIndex, PrintIndex, PrintIndex])
    },
    [StringPtr, FmtPtr, PrintCall].

plawk_scalar_end_separator_lines(0) -->
    !,
    [].
plawk_scalar_end_separator_lines(PrintIndex) -->
    { format(atom(SpacePtr),
          '  %end_space_fmt_~w = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_space, i32 0, i32 0',
          [PrintIndex]),
      format(atom(SpaceCall),
          '  %printed_end_space_~w = call i32 (i8*, ...) @printf(i8* %end_space_fmt_~w)',
          [PrintIndex, PrintIndex])
    },
    [SpacePtr, SpaceCall].

plawk_pattern_guard_ir(always, GuardIR) :-
    GuardIR = ''-'  %is_match = icmp eq i1 true, true'.

plawk_pattern_guard_ir(prefix(Prefix), GuardIR) :-
    llvm_emit_atom_prefix_guard(plawk_surface_prefix, '%line', Prefix,
        '%is_match', GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GuardIR) :-
    llvm_emit_atom_field_eq_guard(plawk_surface_field_eq, '%line', Index, Value,
        32, '%is_match', GuardIR).

plawk_pattern_guard_ir(always, _GlobalBase, MatchValue, GuardIR) :-
    format(atom(GuardCallIR), '  ~w = icmp eq i1 true, true', [MatchValue]),
    GuardIR = ''-GuardCallIR.

plawk_pattern_guard_ir(prefix(Prefix), GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_atom_prefix_guard(GlobalBase, '%line', Prefix, MatchValue,
        GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_atom_field_eq_guard(GlobalBase, '%line', Index, Value, 32,
        MatchValue, GuardIR).

plawk_print_action_ir([field(0)], IR) :-
    !,
    IR = '  %fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_line, i32 0, i32 0
  %printed = call i32 (i8*, ...) @printf(i8* %fmt, i8* %line_s)'.
plawk_print_action_ir(Fields, IR) :-
    phrase(plawk_print_fields_ir(Fields, 0), Parts),
    atomic_list_concat(Parts, '\n', IR).

plawk_print_fields_ir([], _) -->
    ['  %newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
     '  %printed_newline = call i32 (i8*, ...) @printf(i8* %newline_fmt)'].
plawk_print_fields_ir([Field | Rest], Index) -->
    plawk_print_separator_ir(Index),
    plawk_print_field_ir(Field, Index),
    { NextIndex is Index + 1 },
    plawk_print_fields_ir(Rest, NextIndex).

plawk_print_separator_ir(0) -->
    !,
    [].
plawk_print_separator_ir(Index) -->
    { format(atom(SpacePtr),
          '  %space_fmt_~w = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_space, i32 0, i32 0',
          [Index]),
      format(atom(SpaceCall),
          '  %printed_space_~w = call i32 (i8*, ...) @printf(i8* %space_fmt_~w)',
          [Index, Index])
    },
    [SpacePtr, SpaceCall].

plawk_print_field_ir(field(0), Index) -->
    { format(atom(LineLen64),
          '  %line_len64_~w = call i64 @strlen(i8* %line_s)',
          [Index]),
      format(atom(LineLen),
          '  %line_len_~w = trunc i64 %line_len64_~w to i32',
          [Index, Index]),
      format(atom(FmtPtr),
          '  %line_fmt_~w = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
          [Index]),
      format(atom(PrintCall),
          '  %printed_line_~w = call i32 (i8*, ...) @printf(i8* %line_fmt_~w, i32 %line_len_~w, i8* %line_s)',
          [Index, Index, Index])
    },
    [LineLen64, LineLen, FmtPtr, PrintCall].
plawk_print_field_ir(field(FieldIndex), Index) -->
    { FieldIndex > 0,
      format(atom(Base), 'plawk_field_~w', [Index]),
      llvm_emit_atom_field_slice('%line', FieldIndex, 32, Base, SliceIR),
      format(atom(FmtPtr),
          '  %slice_fmt_~w = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
          [Index]),
      format(atom(PrintCall),
          '  %printed_slice_~w = call i32 (i8*, ...) @printf(i8* %slice_fmt_~w, i32 %~w_len, i8* %~w_ptr)',
          [Index, Index, Base, Base])
    },
    [SliceIR, FmtPtr, PrintCall].
