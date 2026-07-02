% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_native_codegen, [
    plawk_program_native_driver_ir/3,
    plawk_program_native_driver_ir/4,
    plawk_program_foreign_specs/3
]).

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target',
    [llvm_emit_atom_prefix_guard/5,
     llvm_emit_atom_field_eq_guard/7,
     llvm_emit_regex_field_match_guard/7,
     llvm_emit_atom_field_slice/5,
     llvm_emit_atom_field_count/4,
     llvm_emit_atom_field_length/5,
     llvm_emit_atom_field_subslice/7,
     llvm_emit_atom_field_index/7,
     llvm_emit_atom_field_i64_cmp_guard/7,
     llvm_emit_atom_field_i64_or_default/7,
     llvm_emit_c_string_global/5,
     llvm_emit_printf_i64/5,
     llvm_emit_printf_slice/6,
     llvm_emit_printf_string/5,
     llvm_emit_printf_string/6,
     llvm_emit_printf0/5,
     llvm_emit_stream_driver_ir/3,
     llvm_emit_ascii_case_slice_print/5]).

%% plawk_program_native_driver_ir(+Program, +InputPath, -DriverIR) is semidet.
%
%  Emit the first native Phase-2 PLAWK driver shape:
%
%      /^PREFIX/ { print $0 }
%      /LITERAL/ { print $0 }
%      $N == "VALUE" { print $0 }
%      $N == "VALUE" { print $M, $K }
%      $N == "VALUE" { count++ } END { print count }
%      $N == "VALUE" { errors++; matches++ } END { print errors, matches }
%      $N == "ERROR" { errors++ } $N == "WARN" { warnings++ } END { print errors, warnings }
%      { counts[$1]++ } END { print counts["ERROR"], counts["WARN"] }
%      $N == "VALUE" { counts[$M]++ } END { print counts["KEY"] }
%      { total++; counts[$1]++ } END { print total, counts["ERROR"] }
%      { total++ } END { print "total", total }
%      BEGIN { print "kind", "count" } { total++ } END { print "total", total }
%      BEGIN { FS = ":" } $1 == "ERROR" { counts[$2]++ } END { print counts["disk"] }
%      BEGIN { FS = ":"; OFS = "," } $1 == "ERROR" { print $2, $3 }
%      $1 == "ERROR" { printf "%s=%s\n", $2, $3 }
%      $3 > 100 { big++ } END { print big }
%      $1 == "ERROR" { print $3, int($3) }
%      $1 == "ERROR" { print int($3) + 1 }
%      $1 == "ERROR" { print int($3) - 1 }
%      $1 == "ERROR" { print NR - 1, NF + 1, length($0) - 3, index($2, "sk") + 1 }
%      $1 == "ERROR" { bytes += $3; last = $3 } END { print bytes, last }
%      $1 == "ERROR" { bytes += length($0); hits += 2 } END { print bytes, hits }
%      { last_pos = index($2, "sk") + 1; total_pos += index($0, "disk") - 1 } END { print last_pos, total_pos }
%      { adjusted += length($0) - 3; width = NF; fields += NF } END { print adjusted, width, fields }
%      { last = NR; prev = NR - 1; total += NR + 1 } END { print last, prev, total }
%      $1 == "ERROR" { hits++; break } { total++ } END { print hits, total }
%      $1 == "ERROR" { last_len = length($0); hits++ } END { print hits, last_len }
%      { if ($1 == "ERROR") { errors++ } else { warnings++ } } END { print errors, warnings }
%      { if ($1 == "ERROR") { print $2, $3 } else { counts[$1]++ } } END { print counts["WARN"] }
%      { if ($1 == "ERROR") { print "error", $2 } else { print "ok", $1 } } END { print "done" }
%
%  The surrounding runtime still comes from write_wam_llvm_project/3. This
%  function emits the target-specific native main that streams the file, lowers
%  the deterministic guard, and prints matching records.
plawk_program_native_driver_ir(
    program(BeginClauses, [rule(Pattern, [Action])], []),
    InputPath,
    DriverIR
) :-
    plawk_rule_body_print_action(Action),
    plawk_output_action_exprs(Action, Exprs),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_field_separator(BeginClauses, FieldSeparator),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GuardGlobalIR-GuardCallIR),
    plawk_print_record_counter_ir(Exprs, LoopPhiIR, RecordCounterIR),
    plawk_output_action_ir(Action, FieldSeparator, OutputSeparator, PrintGlobalIR-PrintActionIR),
    format(atom(RecordIR),
'~w
~w
  br i1 %is_match, label %print_line, label %continue_loop

print_line:
~w
  br label %continue_loop',
        [RecordCounterIR, GuardCallIR, PrintActionIR]),
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"
~w
~w
~w
',
        [BeginGlobalIR, GuardGlobalIR, PrintGlobalIR]),
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR, '',
            success, 'success:\n  ret i32 0'),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_mixed_state_plan(Rules, PrintFields, MixedPlan),
    MixedPlan = mixed_plan(ScalarPlan, AssocPlan, _PlannedRules),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_field_separator(BeginClauses, FieldSeparator),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_mixed_rule_chain_ir(MixedPlan, FieldSeparator, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(PrintFields, BodyPrintFields, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(ScalarPlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_mixed_rule_controls(MixedPlan, MixedRuleControls),
    plawk_mixed_scalar_next_phi_ir(ScalarPlan, RuleCount, MixedRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(ScalarPlan, RuleCount, MixedRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    plawk_mixed_end_print_ir(PrintFields, ScalarPlan, AssocPlan, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocGlobalIR, RuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w~w
  ret i32 0',
        [FinalStatePhiIR, EndPrintIR]),
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, LoopPhiIR, lowered_mixed,
            RecordIR, NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_scalar_state_plan(Rules, PrintFields, StatePlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_field_separator(BeginClauses, FieldSeparator),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_scalar_rule_chain_ir(Rules, StatePlan, FieldSeparator, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(PrintFields, BodyPrintFields, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(StatePlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_scalar_rule_controls(Rules, ScalarRuleControls),
    plawk_scalar_next_phi_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    plawk_scalar_end_print_ir(PrintFields, StatePlan, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, RuleGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w~w
  ret i32 0',
        [FinalStatePhiIR, EndPrintIR]),
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR,
            NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_assoc_runtime_count_plan(Rules, PrintFields, AssocPlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_field_separator(BeginClauses, FieldSeparator),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(PrintFields, BodyPrintFields, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_join_nonempty_ir([RecordCounterIR, AssocChainIR], RecordIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_assoc_end_print_ir(PrintFields, AssocPlan, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  ret i32 0',
        [EndPrintIR]),
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, RecordLoopPhiIR, lowered_assoc,
            RecordIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([for_in(var(LoopVar), var(ArrayName), BodyActions)])]),
    InputPath,
    DriverIR
) :-
    plawk_forin_end_plan(Rules, LoopVar, ArrayName, BodyActions, AssocPlan, PrintFields),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_field_separator(BeginClauses, FieldSeparator),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_print_ir(LoopVar, ArrayName, PrintFields, AssocPlan,
        OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

%% plawk_program_native_driver_ir(+Program, +InputPath, +Options, -DriverIR) is semidet.
%
%  Options-aware driver entry. Programs that call compiled Prolog
%  predicates (prolog_guard patterns / prolog_call expressions) need
%  Options to carry wam_vm(InstrCount, LabelCount) -- the counts
%  reported by wam_llvm_last_compile_counts/2 after
%  write_wam_llvm_project/3 compiled the predicates into the module.
%  The emitted support section holds a lazily created shared %WamState
%  and one wrapper function per called predicate; call sites marshal
%  arguments and invoke the wrappers, so the shared guard/expression
%  emitters need no VM plumbing. Programs with no foreign calls
%  delegate to plawk_program_native_driver_ir/3 unchanged.
plawk_program_native_driver_ir(Program, InputPath, Options, DriverIR) :-
    plawk_program_foreign_specs(Program, GuardSpecs, CallSpecs),
    (   GuardSpecs == [],
        CallSpecs == []
    ->  plawk_program_native_driver_ir(Program, InputPath, DriverIR)
    ;   memberchk(wam_vm(InstrCount, LabelCount), Options),
        plawk_foreign_support_ir(GuardSpecs, CallSpecs, InstrCount, LabelCount,
            SupportIR),
        plawk_program_native_driver_ir(Program, InputPath, MainIR),
        format(atom(DriverIR), '~w~n~n~w', [SupportIR, MainIR])
    ).

%% plawk_program_foreign_specs(+Program, -GuardSpecs, -CallSpecs)
%
%  Collect the deduplicated foreign predicate call shapes used by the
%  program. GuardSpecs are Name-NArgs pairs called as rule guards
%  (predicate arity NArgs); CallSpecs are Name-NArgs pairs called as
%  i64 expressions (predicate arity NArgs + 1 for the output).
plawk_program_foreign_specs(program(_BeginClauses, Rules, EndClauses),
        GuardSpecs, CallSpecs) :-
    findall(Name-NArgs,
        ( member(rule(Pattern, _Actions), Rules),
          plawk_pattern_prolog_guard(Pattern, Name, Args),
          length(Args, NArgs)
        ),
        GuardSpecs0),
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_prolog_call(Actions, Name, Args),
          length(Args, NArgs)
        ),
        CallSpecs0),
    findall(Name-NArgs,
        ( member(rule(_Pattern2, Actions2), Rules),
          member(if(CondPattern, _Then, _Else), Actions2),
          plawk_pattern_prolog_guard(CondPattern, Name, Args),
          length(Args, NArgs)
        ),
        CondGuardSpecs0),
    append(GuardSpecs0, CondGuardSpecs0, AllGuardSpecs),
    sort(AllGuardSpecs, GuardSpecs),
    sort(CallSpecs0, CallSpecs).

plawk_pattern_prolog_guard(prolog_guard(Name, Args), Name, Args).
plawk_pattern_prolog_guard(and_pat(Left, Right), Name, Args) :-
    ( plawk_pattern_prolog_guard(Left, Name, Args)
    ; plawk_pattern_prolog_guard(Right, Name, Args)
    ).
plawk_pattern_prolog_guard(or_pat(Left, Right), Name, Args) :-
    ( plawk_pattern_prolog_guard(Left, Name, Args)
    ; plawk_pattern_prolog_guard(Right, Name, Args)
    ).
plawk_pattern_prolog_guard(not_pat(Pattern), Name, Args) :-
    plawk_pattern_prolog_guard(Pattern, Name, Args).

plawk_actions_prolog_call(Actions, Name, Args) :-
    member(Action, Actions),
    plawk_action_prolog_call(Action, Name, Args).

plawk_action_prolog_call(add(_Var, Expr), Name, Args) :-
    plawk_expr_prolog_call(Expr, Name, Args).
plawk_action_prolog_call(set(_Var, Expr), Name, Args) :-
    plawk_expr_prolog_call(Expr, Name, Args).
plawk_action_prolog_call(print(Fields), Name, Args) :-
    member(Field, Fields),
    plawk_expr_prolog_call(Field, Name, Args).
plawk_action_prolog_call(printf(_Format, PrintfArgs), Name, Args) :-
    member(Field, PrintfArgs),
    plawk_expr_prolog_call(Field, Name, Args).
plawk_action_prolog_call(if(_Pattern, ThenActions, ElseActions), Name, Args) :-
    ( plawk_actions_prolog_call(ThenActions, Name, Args)
    ; plawk_actions_prolog_call(ElseActions, Name, Args)
    ).

plawk_expr_prolog_call(prolog_call(Name, Args), Name, Args).
plawk_expr_prolog_call(Expr, Name, Args) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_prolog_call(Left, Name, Args)
    ; plawk_expr_prolog_call(Right, Name, Args)
    ).

%% plawk_foreign_support_ir(+GuardSpecs, +CallSpecs, +InstrCount, +LabelCount, -IR)
%
%  Emit the shared foreign-call support section: a lazily initialized
%  process-wide %WamState plus one wrapper per called predicate. Guard
%  wrappers return run_loop's success directly. Call wrappers push one
%  unbound output cell, run the predicate, and return {value, ok};
%  failure or a non-integer binding yields {0, false}. Both wrappers
%  save and restore the VM heap top (WamState field 6) and rewind the
%  arena via @wam_cleanup, so per-record foreign calls run in constant
%  memory -- nothing WAM-side persists between plawk calls.
plawk_foreign_support_ir(GuardSpecs, CallSpecs, InstrCount, LabelCount, IR) :-
    format(atom(VmIR),
'@plawk_foreign_vm = internal global %WamState* null

define %WamState* @plawk_foreign_vm_get() {
entry:
  %cur = load %WamState*, %WamState** @plawk_foreign_vm
  %have = icmp ne %WamState* %cur, null
  br i1 %have, label %ret_cur, label %make

ret_cur:
  ret %WamState* %cur

make:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  store %WamState* %vm, %WamState** @plawk_foreign_vm
  ret %WamState* %vm
}',
        [InstrCount, InstrCount, InstrCount, LabelCount, LabelCount,
         LabelCount]),
    findall(GuardIR,
        ( member(Name-NArgs, GuardSpecs),
          plawk_foreign_guard_wrapper_ir(Name, NArgs, GuardIR)
        ),
        GuardIRs),
    findall(CallIR,
        ( member(Name-NArgs, CallSpecs),
          plawk_foreign_call_wrapper_ir(Name, NArgs, CallIR)
        ),
        CallIRs),
    append([[VmIR], GuardIRs, CallIRs], Parts),
    atomic_list_concat(Parts, '\n\n', IR).

plawk_foreign_wrapper_params(NArgs, ParamsIR) :-
    NArgs >= 1,
    NArgs1 is NArgs - 1,
    numlist(0, NArgs1, Ns),
    findall(Param,
        ( member(N, Ns),
          format(atom(Param), '%Value %a~w', [N])
        ),
        Params),
    atomic_list_concat(Params, ', ', ParamsIR).

plawk_foreign_set_reg_lines(NArgs, Lines) :-
    NArgs1 is NArgs - 1,
    numlist(0, NArgs1, Ns),
    findall(Line,
        ( member(N, Ns),
          format(atom(Line),
              '  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %a~w)',
              [N, N])
        ),
        Lines).

plawk_foreign_guard_wrapper_ir(Name, NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_foreign_set_reg_lines(NArgs, SetRegLines),
    atomic_list_concat(SetRegLines, '\n', SetRegIR),
    format(atom(IR),
'define i1 @plawk_foreign_guard_~w_~w(~w) {
entry:
  %vm = call %WamState* @plawk_foreign_vm_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %pc = load i32, i32* @~w_start_pc
  call void @wam_prepare_call(%WamState* %vm, i32 %pc)
~w
  %ok = call i1 @run_loop(%WamState* %vm)
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 %ok

fail:
  ret i1 false
}',
        [Name, NArgs, ParamsIR, Name, SetRegIR]).

plawk_foreign_call_wrapper_ir(Name, NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_foreign_set_reg_lines(NArgs, SetRegLines),
    atomic_list_concat(SetRegLines, '\n', SetRegIR),
    format(atom(IR),
'define { i64, i1 } @plawk_foreign_call_~w_~w(~w) {
entry:
  %vm = call %WamState* @plawk_foreign_vm_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %pc = load i32, i32* @~w_start_pc
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %pc)
~w
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail

read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_tag = extractvalue %Value %out, 0
  %out_is_int = icmp eq i32 %out_tag, 1
  br i1 %out_is_int, label %good, label %rewind_fail

good:
  %payload = extractvalue %Value %out, 1
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  %r0 = insertvalue { i64, i1 } undef, i64 %payload, 0
  %r1 = insertvalue { i64, i1 } %r0, i1 true, 1
  ret { i64, i1 } %r1

rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [Name, NArgs, ParamsIR, Name, SetRegIR, NArgs]).

%% plawk_forin_end_plan(+Rules, +LoopVar, +ArrayName, +BodyActions, -AssocPlan, -PrintFields)
%
%  Plan the first END for-in surface: rules are associative-count updates and
%  the loop body is one print whose fields are the loop key, associative
%  lookups keyed by the loop variable, or string literals.
plawk_forin_end_plan(Rules, LoopVar, ArrayName, BodyActions,
        assoc_plan(Tables, PlannedRules), PrintFields) :-
    BodyActions = [print(PrintFields)],
    PrintFields = [_ | _],
    maplist(plawk_forin_print_field(LoopVar), PrintFields),
    maplist(plawk_assoc_rule_action_specs, Rules, RuleSpecs),
    RuleSpecs \== [],
    findall(RuleArrayName,
        ( member(rule(_Pattern, ActionSpecs, _Control), RuleSpecs),
          member(RuleArrayName-_KeyIndex, ActionSpecs)
        ),
        ActionArrays),
    findall(LookupArrayName,
        member(assoc(var(LookupArrayName), var(LoopVar)), PrintFields),
        LookupArrays),
    append([ActionArrays, [ArrayName], LookupArrays], ArrayNames0),
    sort(ArrayNames0, Tables),
    phrase(plawk_assoc_planned_rules(RuleSpecs, Tables, 0), PlannedRules).

plawk_forin_print_field(LoopVar, var(LoopVar)).
plawk_forin_print_field(LoopVar, assoc(var(_ArrayName), var(LoopVar))).
plawk_forin_print_field(_LoopVar, string(_Value)).

%% plawk_forin_end_print_ir(+LoopVar, +ArrayName, +PrintFields, +AssocPlan,
%%     +OutputSeparator, -IR)
%
%  Emit the native END for-in loop: walk the iterated table's occupied slots
%  through @wam_assoc_i64_iter_next, print each record's fields, then free
%  every table and return.
plawk_forin_end_print_ir(LoopVar, ArrayName, PrintFields, AssocPlan,
        OutputSeparator, IR) :-
    plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
    phrase(plawk_forin_body_print_lines(PrintFields, LoopVar, ArrayName,
        TableIndex, AssocPlan, OutputSeparator, 0), BodyLines),
    atomic_list_concat(BodyLines, '\n', BodyIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(IR),
'  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %end_print], [%forin_next_idx, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  %forin_printed_newline = call i32 @putchar(i32 10)
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
~w
  ret i32 0',
        [TableIndex, TableIndex, BodyIR, FreeIR]).

plawk_forin_body_print_lines([], _LoopVar, _ArrayName, _TableIndex, _AssocPlan,
        _OutputSeparator, _) -->
    [].
plawk_forin_body_print_lines([var(LoopVar) | Rest], LoopVar, ArrayName,
        TableIndex, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    { format(atom(KeyString),
          '  %forin_key_s_~w = call i8* @wam_atom_to_string(i64 %forin_key_id)',
          [PrintIndex]),
      format(atom(FmtVar), 'forin_key_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'forin_printed_key_~w', [PrintIndex]),
      format(atom(PtrIR), '%forin_key_s_~w', [PrintIndex]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [KeyString, FmtPtr, PrintCall],
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        OutputSeparator, NextPrintIndex).
plawk_forin_body_print_lines([assoc(var(LookupArrayName), var(LoopVar)) | Rest],
        LoopVar, ArrayName, TableIndex, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    { (   LookupArrayName == ArrayName
      ->  format(atom(Value),
              '  %forin_value_~w = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
              [PrintIndex, TableIndex])
      ;   plawk_assoc_table_index(AssocPlan, LookupArrayName, LookupTableIndex),
          format(atom(Value),
              '  %forin_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_key_id)',
              [PrintIndex, LookupTableIndex])
      ),
      format(atom(FmtVar), 'forin_i64_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'forin_printed_i64_~w', [PrintIndex]),
      format(atom(ValueIR), '%forin_value_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [Value, FmtPtr, PrintCall],
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        OutputSeparator, NextPrintIndex).
plawk_forin_body_print_lines([string(Value) | Rest], LoopVar, ArrayName,
        TableIndex, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        OutputSeparator, NextPrintIndex).

plawk_forin_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_forin_separator_lines(PrintIndex, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %forin_printed_separator_~w = call i32 @putchar(i32 ~w)',
          [PrintIndex, OutputSeparator])
    },
    [SpaceCall].

plawk_combine_entry_ir('', IR, IR) :-
    !.
plawk_combine_entry_ir(IR, '', IR) :-
    !.
plawk_combine_entry_ir(FirstIR, SecondIR, CombinedIR) :-
    format(atom(CombinedIR), '~w~n~w', [FirstIR, SecondIR]).

plawk_i64_end_print_globals(SurfaceGlobals, RuntimeGlobals) :-
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"
~w

',
        [SurfaceGlobals]).

% State plans keep recognized PLAWK state separate from the LLVM slot numbering.
% Associative arrays use a separate table plan because they are pointer state.
plawk_state_plan_slots(state_plan(Slots), Slots).

plawk_state_slot_count(StatePlan, Count) :-
    plawk_state_plan_slots(StatePlan, Slots),
    length(Slots, Count).

plawk_state_slot_index(StatePlan, Slot, Index) :-
    plawk_state_plan_slots(StatePlan, Slots),
    nth0(Index, Slots, Slot).

plawk_trim_control_tails([], []).
plawk_trim_control_tails([next | _Rest], [next]) :-
    !.
plawk_trim_control_tails([break | _Rest], [break]) :-
    !.
plawk_trim_control_tails([if(Pattern, ThenActions, ElseActions) | Rest],
        [if(Pattern, TrimmedThenActions, TrimmedElseActions) | TrimmedRest]) :-
    !,
    plawk_trim_control_tails(ThenActions, TrimmedThenActions),
    plawk_trim_control_tails(ElseActions, TrimmedElseActions),
    plawk_trim_control_tails(Rest, TrimmedRest).
plawk_trim_control_tails([Action | Rest], [Action | TrimmedRest]) :-
    plawk_trim_control_tails(Rest, TrimmedRest).

plawk_split_terminal_control(Actions0, BodyActions, Control) :-
    plawk_trim_control_tails(Actions0, Actions),
    plawk_split_normalized_terminal_control(Actions, BodyActions, Control).

plawk_split_normalized_terminal_control(Actions, BodyActions, terminal_next) :-
    append(BodyActions, [next], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_terminal_control(Actions, BodyActions, terminal_break) :-
    append(BodyActions, [break], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_terminal_control(Actions, Actions, fallthrough) :-
    \+ plawk_actions_have_control(Actions).

plawk_actions_have_control(Actions) :-
    member(Action, Actions),
    plawk_action_has_control(Action).

plawk_action_has_control(next).
plawk_action_has_control(break).
plawk_action_has_control(if(_Pattern, ThenActions, ElseActions)) :-
    ( plawk_branch_actions_have_unsupported_control(ThenActions)
    ; plawk_branch_actions_have_unsupported_control(ElseActions)
    ).

plawk_branch_actions_have_unsupported_control(Actions) :-
    plawk_trim_control_tails(Actions, TrimmedActions),
    (   append(BodyActions, [next], TrimmedActions)
    ->  plawk_actions_have_control(BodyActions)
    ;   append(BodyActions, [break], TrimmedActions)
    ->  plawk_actions_have_control(BodyActions)
    ;   plawk_actions_have_control(TrimmedActions)
    ).

plawk_rule_target(fallthrough, NextLabel, NextLabel).
plawk_rule_target(terminal_next, _NextLabel, continue_loop).
plawk_rule_target(terminal_break, _NextLabel, break_close_stream).


plawk_controls_have_break(Controls) :-
    member(terminal_break, Controls).

plawk_assoc_break_close_ir(Controls, IR) :-
    (   plawk_controls_have_break(Controls)
    ->  IR = 'break_close_stream:
  %break_close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %break_close_ok, label %end_print, label %fail_close
'
    ;   IR = ''
    ).

plawk_break_close_ir(StatePlan, RuleCount, Controls, BranchControlExits, BreakPredKind, BreakCloseIR, FinalStatePhiIR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    (   plawk_controls_have_break(Controls)
    ;   member(branch_break(_Label, _Values), BranchControlExits)
    ),
    !,
    phrase(plawk_break_slot_phi_lines(Slots, RuleCount, Controls, BranchControlExits, BreakPredKind, 0), BreakSlotPhiLines),
    atomic_list_concat(BreakSlotPhiLines, '\n', BreakSlotPhiIR),
    format(atom(BreakCloseIR),
'break_close_stream:
~w
  %break_close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %break_close_ok, label %end_print, label %fail_close
',
        [BreakSlotPhiIR]),
    phrase(plawk_final_state_phi_lines(Slots, true, 0), FinalStatePhiLines),
    atomic_list_concat(FinalStatePhiLines, '\n', FinalStatePhiIR0),
    ( FinalStatePhiIR0 == ''
    -> FinalStatePhiIR = ''
    ;  format(atom(FinalStatePhiIR), '~w~n', [FinalStatePhiIR0])
    ).
plawk_break_close_ir(StatePlan, _RuleCount, _Controls, _BranchControlExits, _BreakPredKind, '', FinalStatePhiIR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_final_state_phi_lines(Slots, false, 0), FinalStatePhiLines),
    atomic_list_concat(FinalStatePhiLines, '\n', FinalStatePhiIR0),
    ( FinalStatePhiIR0 == ''
    -> FinalStatePhiIR = ''
    ;  format(atom(FinalStatePhiIR), '~w~n', [FinalStatePhiIR0])
    ).
plawk_break_slot_phi_lines([], _RuleCount, _Controls, _BranchControlExits, _BreakPredKind, _) -->
    [].
plawk_break_slot_phi_lines([_Slot | Rest], RuleCount, Controls, BranchControlExits, BreakPredKind, SlotIndex) -->
    { LastRuleIndex is RuleCount - 1,
      findall(Incoming,
          ( between(0, LastRuleIndex, RuleIndex),
            nth0(RuleIndex, Controls, terminal_break),
            plawk_break_predecessor_label(BreakPredKind, RuleIndex, PredLabel),
            format(atom(Incoming), '[%rule_~w_slot_~w, %~w]',
                [RuleIndex, SlotIndex, PredLabel])
          ),
          RuleIncomings),
      plawk_branch_break_phi_incomings(BranchControlExits, SlotIndex, BranchIncomings),
      append(RuleIncomings, BranchIncomings, Incomings),
      Incomings \== [],
      atomic_list_concat(Incomings, ', ', IncomingIR),
      format(atom(Line), '  %break_slot_~w = phi i64 ~w', [SlotIndex, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_break_slot_phi_lines(Rest, RuleCount, Controls, BranchControlExits, BreakPredKind, NextSlotIndex).

plawk_branch_break_phi_incomings([], _SlotIndex, []).
plawk_branch_break_phi_incomings([branch_break(Label, Values) | Rest], SlotIndex, [Incoming | Incomings]) :-
    !,
    nth0(SlotIndex, Values, Value),
    format(atom(Incoming), '[~w, %~w]', [Value, Label]),
    plawk_branch_break_phi_incomings(Rest, SlotIndex, Incomings).
plawk_branch_break_phi_incomings([_Exit | Rest], SlotIndex, Incomings) :-
    plawk_branch_break_phi_incomings(Rest, SlotIndex, Incomings).

plawk_break_predecessor_label(apply, RuleIndex, Label) :-
    format(atom(Label), 'rule_~w_apply', [RuleIndex]).
plawk_break_predecessor_label(done, RuleIndex, Label) :-
    format(atom(Label), 'rule_~w_done', [RuleIndex]).

plawk_final_state_phi_lines([], _HasBreak, _) -->
    [].
plawk_final_state_phi_lines([_Slot | Rest], HasBreak, SlotIndex) -->
    { format(atom(EofIncoming), '[%slot_~w, %close_stream]', [SlotIndex]),
      ( HasBreak == true
      -> format(atom(BreakIncoming), '[%break_slot_~w, %break_close_stream]', [SlotIndex]),
         Incomings = [EofIncoming, BreakIncoming]
      ;  Incomings = [EofIncoming]
      ),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      format(atom(Line), '  %final_slot_~w = phi i64 ~w', [SlotIndex, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_final_state_phi_lines(Rest, HasBreak, NextSlotIndex).

plawk_scalar_rule_controls(Rules, Controls) :-
    findall(Control,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_split_terminal_control(Actions, _BodyActions, Control)
        ),
        Controls).

plawk_scalar_state_plan(Rules, PrintFields, state_plan(Slots)) :-
    findall(Name,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_trim_control_tails(Actions, ReachableActions),
          member(Action, ReachableActions),
          plawk_scalar_update_action_name(Action, Name)
        ),
        ActionVars),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    ( ActionVars \== [] ; BodyPrintFields \== [] ),
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
    PlannedRules \== [],
    (   plawk_state_slot_count(ScalarPlan, ScalarCount),
        ScalarCount > 0
    ;   plawk_planned_rules_have_conditionals(PlannedRules)
    ).

plawk_mixed_scalar_state_plan(Rules, PrintFields, state_plan(Slots)) :-
    findall(Name,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_trim_control_tails(Actions, ReachableActions),
          member(Action, ReachableActions),
          plawk_scalar_update_action_name(Action, Name)
        ),
        ActionVars),
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
          plawk_assoc_increment_spec_in_actions(Actions, ArrayName-_KeyIndex)
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

plawk_planned_rules_have_conditionals(PlannedRules) :-
    member(mixed_rule(_Index, _Pattern, Actions, _AssocActions, _Control), PlannedRules),
    plawk_actions_have_conditional(Actions).

plawk_actions_have_conditional(Actions) :-
    member(Action, Actions),
    plawk_action_has_conditional(Action).

plawk_action_has_conditional(if(_Pattern, _ThenActions, _ElseActions)).

plawk_rules_body_print_fields(Rules, Fields) :-
    findall(Field,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_actions_body_print_field(Actions, Field)
        ),
        Fields).

plawk_rules_scalar_update_exprs(Rules, Exprs) :-
    findall(Expr,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_actions_scalar_update_expr(Actions, Expr)
        ),
        Exprs).

plawk_actions_scalar_update_expr(Actions, Expr) :-
    plawk_trim_control_tails(Actions, ReachableActions),
    member(Action, ReachableActions),
    plawk_action_scalar_update_expr(Action, Expr).

plawk_action_scalar_update_expr(if(_Pattern, ThenActions, ElseActions), Expr) :-
    !,
    (   plawk_actions_scalar_update_expr(ThenActions, Expr)
    ;   plawk_actions_scalar_update_expr(ElseActions, Expr)
    ).
plawk_action_scalar_update_expr(Action, Expr) :-
    plawk_scalar_action_update(Action, _Name, Operation),
    plawk_scalar_operation_expr(Operation, Expr).

plawk_scalar_operation_expr(add(Expr), Expr).
plawk_scalar_operation_expr(set(Expr), Expr).

plawk_actions_body_print_field(Actions, Field) :-
    plawk_trim_control_tails(Actions, ReachableActions),
    member(Action, ReachableActions),
    plawk_action_body_print_field(Action, Field).

plawk_action_body_print_field(print(Fields), Field) :-
    member(Field, Fields).
plawk_action_body_print_field(printf(string(_Format), Args), Field) :-
    member(Field, Args).
plawk_action_body_print_field(if(_Pattern, ThenActions, ElseActions), Field) :-
    (   plawk_actions_body_print_field(ThenActions, Field)
    ;   plawk_actions_body_print_field(ElseActions, Field)
    ).

plawk_mixed_planned_rules([], _AssocPlan, _Index) -->
    [].
plawk_mixed_planned_rules([rule(Pattern, Actions) | Rest], assoc_plan(Tables, Actions0), Index) -->
    { plawk_split_terminal_control(Actions, BodyActions, Control),
      plawk_mixed_rule_actions(BodyActions, PlannedActions),
      plawk_assoc_increment_specs_in_actions(BodyActions, AssocSpecs),
      ( PlannedActions == [], AssocSpecs == [], Control == fallthrough
      -> HasActions = false,
         NextIndex = Index,
         PlannedAssocActions = []
      ;  HasActions = true,
         phrase(plawk_assoc_planned_actions(AssocSpecs, Tables, 0), PlannedAssocActions),
         NextIndex is Index + 1
      )
    },
    ( { HasActions == true }
    -> [mixed_rule(Index, Pattern, PlannedActions, PlannedAssocActions, Control)]
    ;  []
    ),
    plawk_mixed_planned_rules(Rest, assoc_plan(Tables, Actions0), NextIndex).

plawk_mixed_rule_actions([], []).
plawk_mixed_rule_actions([Action | Rest], [Action | PlannedRest]) :-
    plawk_mixed_update_action(Action),
    plawk_mixed_rule_actions(Rest, PlannedRest).

plawk_mixed_update_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_mixed_update_action(Action) :-
    plawk_assoc_update_action(Action).
plawk_mixed_update_action(Action) :-
    plawk_rule_body_print_action(Action).
plawk_mixed_update_action(if(_Pattern, ThenActions, ElseActions)) :-
    append(ThenActions, ElseActions, Actions),
    Actions \== [],
    plawk_mixed_branch_body_actions(ThenActions),
    plawk_mixed_branch_body_actions(ElseActions).

plawk_mixed_branch_body_actions(Actions) :-
    plawk_split_branch_control(Actions, BodyActions, _Control),
    maplist(plawk_mixed_update_action, BodyActions).

plawk_assoc_update_action(inc_assoc(var(_ArrayName), field(KeyIndex))) :-
    KeyIndex > 0.

plawk_scalar_conditional_action(if(_Pattern, ThenActions, ElseActions)) :-
    append(ThenActions, ElseActions, Actions),
    Actions \== [],
    plawk_scalar_branch_body_actions(ThenActions),
    plawk_scalar_branch_body_actions(ElseActions).

plawk_scalar_plain_update_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).

plawk_split_branch_control(Actions0, BodyActions, Control) :-
    plawk_trim_control_tails(Actions0, Actions),
    plawk_split_normalized_branch_control(Actions, BodyActions, Control).

plawk_split_normalized_branch_control(Actions, BodyActions, branch_next) :-
    append(BodyActions, [next], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_branch_control(Actions, BodyActions, branch_break) :-
    append(BodyActions, [break], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_branch_control(Actions, Actions, fallthrough) :-
    \+ plawk_actions_have_control(Actions).

plawk_assoc_increment_spec_in_actions(Actions, Spec) :-
    plawk_trim_control_tails(Actions, ReachableActions),
    member(Action, ReachableActions),
    plawk_assoc_increment_spec_in_action(Action, Spec).

plawk_assoc_increment_specs_in_actions(Actions, Specs) :-
    findall(Spec, plawk_assoc_increment_spec_in_actions(Actions, Spec), Specs).

plawk_assoc_increment_spec_in_action(Action, Spec) :-
    plawk_assoc_increment_action(Action, Spec).
plawk_assoc_increment_spec_in_action(if(_Pattern, ThenActions, ElseActions), Spec) :-
    ( plawk_assoc_increment_spec_in_actions(ThenActions, Spec)
    ; plawk_assoc_increment_spec_in_actions(ElseActions, Spec)
    ).

plawk_mixed_rule_chain_ir(mixed_plan(ScalarPlan, AssocPlan, Rules), FieldSeparator, OutputSeparator,
        GlobalIR, ChainIR, RuleCount, BranchNextExits) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    plawk_mixed_rule_controls(mixed_plan(ScalarPlan, AssocPlan, Rules), Controls),
    phrase(plawk_mixed_rule_chain_lines(Rules, Controls, ScalarPlan, AssocPlan, FieldSeparator, OutputSeparator, 0), Parts),
    plawk_rule_chain_parts(Parts, GlobalParts, ChainParts, BranchNextExits),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_mixed_rule_controls(mixed_plan(_ScalarPlan, _AssocPlan, Rules), Controls) :-
    findall(Control, member(mixed_rule(_Index, _Pattern, _ScalarActions, _AssocActions, Control), Rules), Controls).

plawk_rule_chain_parts([], [], [], []).
plawk_rule_chain_parts([rule_chain_part(GlobalIR, ChainIR, Exits) | Rest],
        [GlobalIR | GlobalParts], [ChainIR | ChainParts], AllExits) :-
    plawk_rule_chain_parts(Rest, GlobalParts, ChainParts, RestExits),
    append(Exits, RestExits, AllExits).

plawk_mixed_rule_chain_lines([], _Controls, _ScalarPlan, _AssocPlan, _FieldSeparator, _OutputSeparator, _) -->
    [].
plawk_mixed_rule_chain_lines([mixed_rule(Index, Pattern, Actions, _AssocActions, Control) | Rest], Controls, ScalarPlan, AssocPlan, FieldSeparator, OutputSeparator, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'rule_~w_match', [NextIndex])
      ),
      plawk_rule_target(Control, NextLabel, RuleTargetLabel),
      format(atom(RuleLabel), 'rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'rule_~w_apply', [Index]),
      format(atom(DoneLabel), 'rule_~w_done', [Index]),
      plawk_mixed_scalar_rule_input_phi_ir(ScalarPlan, Index, Controls, InputPhiIR),
      plawk_mixed_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel,
          NextLabel, InputPhiIR, FieldSeparator, GuardGlobalIR-GuardIR),
      plawk_native_match_update_ir(ScalarPlan, AssocPlan, Actions, FieldSeparator, OutputSeparator, Index,
          BranchNextExits, ActionGlobalIR-ActionIR),
      ( Index =:= 0
      -> EntryIR = '  br label %rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      format(atom(RuleIR),
'~w~w

~w:
~w
  br label %~w

~w:
  br label %~w',
          [EntryIR, GuardIR, ApplyLabel, ActionIR, DoneLabel, DoneLabel, RuleTargetLabel]),
      format(atom(CombinedGlobalIR), '~w~n~w', [GuardGlobalIR, ActionGlobalIR]),
      Part = rule_chain_part(CombinedGlobalIR, RuleIR, BranchNextExits)
    },
    [Part],
    plawk_mixed_rule_chain_lines(Rest, Controls, ScalarPlan, AssocPlan, FieldSeparator, OutputSeparator, NextIndex).

plawk_mixed_rule_guard_ir(always, Index, RuleLabel, ApplyLabel, NextLabel,
    InputPhiIR, _FieldSeparator, ''-IR) :-
    !,
    format(atom(MatchVar), 'rule_~w_is_match', [Index]),
    format(atom(IR),
'~w:
~w  %~w = icmp eq i1 true, true
  br i1 %~w, label %~w, label %~w',
        [RuleLabel, InputPhiIR, MatchVar, MatchVar, ApplyLabel, NextLabel]).
plawk_mixed_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel, NextLabel,
    InputPhiIR, FieldSeparator, GuardGlobalIR-IR) :-
    format(atom(MatchVar), 'rule_~w_is_match', [Index]),
    format(atom(GlobalBase), 'plawk_mixed_rule_~w', [Index]),
    format(atom(MatchValue), '%~w', [MatchVar]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, MatchValue,
        GuardGlobalIR-GuardCallIR),
    format(atom(IR),
'~w:
~w~w
  br i1 %~w, label %~w, label %~w',
        [RuleLabel, InputPhiIR, GuardCallIR, MatchVar, ApplyLabel, NextLabel]).

plawk_mixed_scalar_rule_input_phi_ir(_StatePlan, 0, _Controls, '') :-
    !.
plawk_mixed_scalar_rule_input_phi_ir(StatePlan, RuleIndex, Controls, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_mixed_scalar_rule_input_phi_lines(Slots, RuleIndex, Controls, 0), Lines),
    atomic_list_concat(Lines, '\n', LinesIR),
    format(atom(IR), '~w~n', [LinesIR]).

plawk_mixed_scalar_rule_input_phi_lines([], _RuleIndex, _Controls, _) -->
    [].
plawk_mixed_scalar_rule_input_phi_lines([_Slot | Rest], RuleIndex, Controls, SlotIndex) -->
    { PrevRuleIndex is RuleIndex - 1,
      plawk_scalar_rule_input_value(PrevRuleIndex, SlotIndex, PrevFalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [PrevFalseValue, PrevRuleIndex]),
      (   plawk_terminal_control_skips_next_rule(Controls, PrevRuleIndex)
      ->  Incomings = [FalseIncoming]
      ;   format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
              [PrevRuleIndex, SlotIndex, PrevRuleIndex]),
          Incomings = [FalseIncoming, ApplyIncoming]
      ),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      format(atom(Line), '  %rule_~w_in_slot_~w = phi i64 ~w',
          [RuleIndex, SlotIndex, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_mixed_scalar_rule_input_phi_lines(Rest, RuleIndex, Controls, NextSlotIndex).

plawk_mixed_scalar_next_phi_ir(StatePlan, RuleCount, Controls, BranchNextExits, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_mixed_scalar_next_phi_lines(Slots, RuleCount, Controls, BranchNextExits, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_scalar_next_phi_lines([], _RuleCount, _Controls, _BranchNextExits, _) -->
    [].
plawk_mixed_scalar_next_phi_lines([_Slot | Rest], RuleCount, Controls, BranchNextExits, Index) -->
    { LastRuleIndex is RuleCount - 1,
      plawk_scalar_rule_input_value(LastRuleIndex, Index, FalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [FalseValue, LastRuleIndex]),
      findall(ApplyIncoming,
          ( between(0, LastRuleIndex, RuleIndex),
            ( ( RuleIndex =:= LastRuleIndex,
                \+ nth0(RuleIndex, Controls, terminal_break)
              )
            ; nth0(RuleIndex, Controls, terminal_next)
            ),
            format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
                [RuleIndex, Index, RuleIndex])
          ),
          ApplyIncomings),
      plawk_branch_next_phi_incomings(BranchNextExits, Index, BranchNextIncomings),
      append([FalseIncoming | ApplyIncomings], BranchNextIncomings, Incomings),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      format(atom(Line), '  %next_slot_~w = phi i64 ~w', [Index, IncomingIR]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_mixed_scalar_next_phi_lines(Rest, RuleCount, Controls, BranchNextExits, NextIndex).

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
        ( member(rule(_Pattern, ActionSpecs, _Control), RuleSpecs),
          member(ArrayName-_KeyIndex, ActionSpecs)
        ),
        ActionArrays),
    append(ActionArrays, PrintArrays, ArrayNames0),
    sort(ArrayNames0, Tables),
    phrase(plawk_assoc_planned_rules(RuleSpecs, Tables, 0), PlannedRules).

plawk_assoc_rule_controls(assoc_plan(_Tables, Rules), Controls) :-
    findall(Control, member(assoc_rule(_Index, _Pattern, _Actions, Control), Rules), Controls).

plawk_assoc_rule_action_specs(rule(Pattern, Actions), rule(Pattern, ActionSpecs, Control)) :-
    plawk_split_terminal_control(Actions, BodyActions, Control),
    ( BodyActions == []
    -> ActionSpecs = []
    ;  maplist(plawk_assoc_increment_action, BodyActions, ActionSpecs)
    ),
    ( ActionSpecs \== [] ; memberchk(Control, [terminal_next, terminal_break]) ).

plawk_assoc_increment_action(inc_assoc(var(ArrayName), field(KeyIndex)), ArrayName-KeyIndex) :-
    KeyIndex > 0.

plawk_assoc_print_array(assoc(var(ArrayName), string(_Key)), ArrayName).

plawk_assoc_planned_rules([], _Tables, _Index) -->
    [].
plawk_assoc_planned_rules([rule(Pattern, ActionSpecs, Control) | Rest], Tables, Index) -->
    { phrase(plawk_assoc_planned_actions(ActionSpecs, Tables, 0), PlannedActions),
      NextIndex is Index + 1
    },
    [assoc_rule(Index, Pattern, PlannedActions, Control)],
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

plawk_assoc_rule_chain_ir(assoc_plan(_Tables, Rules), FieldSeparator, GlobalIR, ChainIR) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    phrase(plawk_assoc_rule_chain_lines(Rules, FieldSeparator, 0), Pairs),
    pairs_keys_values(Pairs, GlobalParts, ChainParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_assoc_rule_chain_lines([], _FieldSeparator, _) -->
    [].
plawk_assoc_rule_chain_lines([assoc_rule(Index, Pattern, Actions, Control) | Rest], FieldSeparator, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'assoc_rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'assoc_rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'assoc_rule_~w_apply', [Index]),
      plawk_rule_target(Control, NextLabel, RuleTargetLabel),
      plawk_assoc_rule_apply_ir(Index, Actions, RuleTargetLabel, FieldSeparator, ApplyIR),
      ( Index =:= 0
      -> EntryIR = '  br label %assoc_rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      plawk_assoc_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel,
          NextLabel, EntryIR, FieldSeparator, GuardGlobalIR, BranchIR),
      format(atom(RuleIR),
'~w

~w:
~w',
          [BranchIR, ApplyLabel, ApplyIR]),
      Pair = GuardGlobalIR-RuleIR
    },
    [Pair],
    plawk_assoc_rule_chain_lines(Rest, FieldSeparator, NextIndex).

plawk_assoc_rule_guard_ir(always, _Index, RuleLabel, ApplyLabel, _NextLabel,
    EntryIR, _FieldSeparator, '', IR) :-
    !,
    format(atom(IR),
'~w~w:
  br label %~w',
        [EntryIR, RuleLabel, ApplyLabel]).
plawk_assoc_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel, NextLabel,
    EntryIR, FieldSeparator, GuardGlobalIR, IR) :-
    format(atom(MatchVar), 'assoc_rule_~w_is_match', [Index]),
    format(atom(GlobalBase), 'plawk_assoc_rule_~w', [Index]),
    format(atom(MatchValue), '%~w', [MatchVar]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, MatchValue,
        GuardGlobalIR-GuardCallIR),
    format(atom(IR),
'~w~w:
~w
  br i1 %~w, label %~w, label %~w',
        [EntryIR, RuleLabel, GuardCallIR, MatchVar, ApplyLabel, NextLabel]).

plawk_assoc_rule_apply_ir(_RuleIndex, [], NextLabel, _FieldSeparator, IR) :-
    !,
    format(atom(IR), '  br label %~w', [NextLabel]).
plawk_assoc_rule_apply_ir(RuleIndex, Actions, NextLabel, FieldSeparator, IR) :-
    phrase(plawk_assoc_rule_action_lines(RuleIndex, Actions, NextLabel, FieldSeparator), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_rule_action_lines(RuleIndex, Actions, NextLabel, FieldSeparator) -->
    { Actions = [_ | _],
      format(atom(FirstBranch), '  br label %assoc_rule_~w_action_0', [RuleIndex])
    },
    [FirstBranch, ''],
    plawk_assoc_rule_action_blocks(RuleIndex, Actions, NextLabel, FieldSeparator).

plawk_assoc_rule_action_blocks(_RuleIndex, [], _NextLabel, _FieldSeparator) -->
    [].
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_action(Index, _ArrayName, TableIndex, KeyIndex) | Rest], NextLabel, FieldSeparator) -->
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
          '  %assoc_rule_~w_action_~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
          [RuleIndex, Index, KeyIndex, FieldSeparator]),
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
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).

plawk_field_separator(BeginClauses, FieldSeparator) :-
    (   member(begin(Actions), BeginClauses),
        member(set(var('FS'), string(Value)), Actions)
    ->  string_codes(Value, [FieldSeparator])
    ;   FieldSeparator = 32
    ).

plawk_output_separator(BeginClauses, OutputSeparator) :-
    (   member(begin(Actions), BeginClauses),
        member(set(var('OFS'), string(Value)), Actions)
    ->  string_codes(Value, [OutputSeparator])
    ;   OutputSeparator = 32
    ).

plawk_begin_print_string_globals(BeginClauses, GlobalIR) :-
    plawk_begin_print_fields(BeginClauses, Fields),
    phrase(plawk_begin_print_string_global_lines(Fields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_begin_print_fields([], []).
plawk_begin_print_fields([begin(Actions)], Fields) :-
    (   member(print(Fields), Actions)
    ->  true
    ;   Fields = []
    ).

plawk_begin_print_string_global_lines([], _) -->
    [].
plawk_begin_print_string_global_lines([string(Value) | Rest], Index) -->
    { format(atom(GlobalName), 'plawk_begin_print_string_~w', [Index]),
      llvm_emit_c_string_global(GlobalName, Value, Line, _StringLen, _BytesLen),
      NextIndex is Index + 1
    },
    [Line],
    plawk_begin_print_string_global_lines(Rest, NextIndex).
plawk_begin_print_string_global_lines([_Field | Rest], Index) -->
    { NextIndex is Index + 1 },
    plawk_begin_print_string_global_lines(Rest, NextIndex).

plawk_begin_print_ir([], _OutputSeparator, '') :-
    !.
plawk_begin_print_ir([begin(Actions)], OutputSeparator, IR) :-
    member(print(Fields), Actions),
    !,
    maplist(plawk_begin_print_field, Fields),
    phrase(plawk_begin_print_lines(Fields, OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).
plawk_begin_print_ir([begin(_Actions)], _OutputSeparator, '').

plawk_begin_print_field(string(_)).

plawk_begin_print_lines([], _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          begin_newline_fmt, printed_begin_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].
plawk_begin_print_lines([string(Value) | Rest], OutputSeparator, Index) -->
    plawk_begin_separator_lines(Index, OutputSeparator),
    plawk_begin_string_print_lines(Value, Index),
    { NextIndex is Index + 1 },
    plawk_begin_print_lines(Rest, OutputSeparator, NextIndex).

plawk_begin_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_begin_separator_lines(Index, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %printed_begin_separator_~w = call i32 @putchar(i32 ~w)',
          [Index, OutputSeparator])
    },
    [SpaceCall].

plawk_begin_string_print_lines(Value, Index) -->
    { string_codes(Value, Codes),
      length(Codes, StringLen),
      BytesLen is StringLen + 1,
      format(atom(StringPtr),
          '  %begin_string_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_begin_print_string_~w, i32 0, i32 0',
          [Index, BytesLen, BytesLen, Index]),
      format(atom(FmtVar), 'begin_string_fmt_~w', [Index]),
      format(atom(PrintVar), 'printed_begin_string_~w', [Index]),
      format(atom(PtrIR), '%begin_string_~w_ptr', [Index]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR,
          [FmtPtr, PrintCall])
    },
    [StringPtr],
    [FmtPtr, PrintCall].

plawk_assoc_print_key_globals(PrintFields, GlobalIR) :-
    phrase(plawk_assoc_print_key_global_lines(PrintFields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_end_print_string_globals(PrintFields, GlobalIR) :-
    phrase(plawk_end_print_string_global_lines(PrintFields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_end_print_string_global_lines([], _) -->
    [].
plawk_end_print_string_global_lines([string(Value) | Rest], Index) -->
    { format(atom(GlobalName), 'plawk_end_print_string_~w', [Index]),
      llvm_emit_c_string_global(GlobalName, Value, Line, _StringLen, _BytesLen),
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
      format(atom(GlobalName), 'plawk_assoc_print_key_~w', [Index]),
      llvm_emit_c_string_global(GlobalName, Codes, Line, _KeyLen, _BytesLen),
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

plawk_assoc_end_print_ir(PrintFields, AssocPlan, OutputSeparator, IR) :-
    phrase(plawk_assoc_end_print_lines(PrintFields, AssocPlan, OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_end_print_lines([], AssocPlan, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          end_newline_fmt, printed_end_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall],
    plawk_assoc_free_lines(AssocPlan).
plawk_assoc_end_print_lines([assoc(var(ArrayName), string(Key)) | Rest], AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
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
      format(atom(FmtVar), 'assoc_end_i64_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_assoc_end_i64_~w', [PrintIndex]),
      format(atom(ValueIR), '%assoc_end_value_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [KeyPtr, KeyId, Value, FmtPtr, PrintCall],
    plawk_assoc_end_print_lines(Rest, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_assoc_end_print_lines([string(Value) | Rest], AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_assoc_end_print_lines(Rest, AssocPlan, OutputSeparator, NextPrintIndex).

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

plawk_mixed_end_print_ir(PrintFields, ScalarPlan, AssocPlan, OutputSeparator, IR) :-
    phrase(plawk_mixed_end_print_lines(PrintFields, ScalarPlan, AssocPlan, OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_end_print_lines([], _ScalarPlan, AssocPlan, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          end_newline_fmt, printed_end_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall],
    plawk_assoc_free_lines(AssocPlan).
plawk_mixed_end_print_lines([var(Name) | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    { plawk_state_slot_index(ScalarPlan, scalar_counter(Name), SlotIndex),
      format(atom(FmtVar), 'end_i64_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_i64_~w', [PrintIndex]),
      format(atom(ValueIR), '%final_slot_~w', [SlotIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([assoc(var(ArrayName), string(Key)) | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
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
      format(atom(FmtVar), 'assoc_end_i64_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_assoc_end_i64_~w', [PrintIndex]),
      format(atom(ValueIR), '%assoc_end_value_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [KeyPtr, KeyId, Value, FmtPtr, PrintCall],
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([special('NR') | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_nr_print_lines(PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([Expr | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    { plawk_end_scalar_expr(Expr) },
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_expr_print_lines(Expr, ScalarPlan, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([string(Value) | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).

plawk_scalar_print_expr(var(Name), Name).
plawk_scalar_print_expr(Expr, Name) :-
    plawk_end_scalar_expr(Expr),
    plawk_expr_scalar_read_name(Expr, Name).

plawk_scalar_rule_chain_ir(Rules, StatePlan, FieldSeparator, OutputSeparator,
        GlobalIR, ChainIR, RuleCount, BranchNextExits) :-
    plawk_scalar_planned_rules(Rules, PlannedRules, Controls),
    length(PlannedRules, RuleCount),
    RuleCount > 0,
    phrase(plawk_scalar_rule_chain_lines(PlannedRules, Controls, StatePlan, FieldSeparator, OutputSeparator, 0), Parts),
    plawk_rule_chain_parts(Parts, GlobalParts, ChainParts, BranchNextExits),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_scalar_planned_rules(Rules, PlannedRules, Controls) :-
    phrase(plawk_scalar_planned_rule_lines(Rules, 0), PlannedRules),
    findall(Control,
        member(scalar_rule(_Index, _Pattern, _Actions, Control), PlannedRules),
        Controls).

plawk_scalar_planned_rule_lines([], _Index) -->
    [].
plawk_scalar_planned_rule_lines([rule(Pattern, Actions) | Rest], Index) -->
    { plawk_split_terminal_control(Actions, BodyActions, Control),
      NextIndex is Index + 1 },
    [scalar_rule(Index, Pattern, BodyActions, Control)],
    plawk_scalar_planned_rule_lines(Rest, NextIndex).

plawk_scalar_rule_chain_lines([], _Controls, _StatePlan, _FieldSeparator, _OutputSeparator, _) -->
    [].
plawk_scalar_rule_chain_lines([scalar_rule(Index, Pattern, Actions, Control) | Rest], Controls, StatePlan, FieldSeparator, OutputSeparator, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'rule_~w_apply', [Index]),
      format(atom(DoneLabel), 'rule_~w_done', [Index]),
      format(atom(MatchVar), 'rule_~w_is_match', [Index]),
      format(atom(GlobalBase), 'plawk_surface_rule_~w', [Index]),
      format(atom(MatchValue), '%~w', [MatchVar]),
      plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, MatchValue,
          GuardGlobalIR-GuardCallIR),
      plawk_rule_target(Control, NextLabel, RuleTargetLabel),
      maplist(plawk_scalar_rule_body_action, Actions),
      BodyActions = Actions,
      plawk_scalar_rule_input_phi_ir(StatePlan, Index, Controls, InputPhiIR),
      plawk_scalar_match_update_ir(StatePlan, BodyActions, FieldSeparator, OutputSeparator, Index,
          BranchNextExits, MatchUpdateGlobalIR-MatchUpdateIR),
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
  br label %~w

~w:
  br label %~w',
          [EntryIR, RuleLabel, InputPhiIR, GuardCallIR, MatchVar,
           ApplyLabel, NextLabel, ApplyLabel, MatchUpdateIR, DoneLabel,
           DoneLabel, RuleTargetLabel]),
      format(atom(CombinedGlobalIR), '~w~n~w', [GuardGlobalIR, MatchUpdateGlobalIR]),
      Part = rule_chain_part(CombinedGlobalIR, BranchIR, BranchNextExits)
    },
    [Part],
    plawk_scalar_rule_chain_lines(Rest, Controls, StatePlan, FieldSeparator, OutputSeparator, NextIndex).

plawk_scalar_rule_input_phi_ir(_StatePlan, 0, _Controls, '') :-
    !.
plawk_scalar_rule_input_phi_ir(StatePlan, RuleIndex, Controls, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_rule_input_phi_lines(Slots, RuleIndex, Controls, 0), Lines),
    atomic_list_concat(Lines, '\n', LinesIR),
    format(atom(IR), '~w~n', [LinesIR]).

plawk_scalar_rule_input_phi_lines([], _RuleIndex, _Controls, _) -->
    [].
plawk_scalar_rule_input_phi_lines([_Name | Rest], RuleIndex, Controls, SlotIndex) -->
    { PrevRuleIndex is RuleIndex - 1,
      plawk_scalar_rule_input_value(PrevRuleIndex, SlotIndex, PrevFalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [PrevFalseValue, PrevRuleIndex]),
      (   plawk_terminal_control_skips_next_rule(Controls, PrevRuleIndex)
      ->  Incomings = [FalseIncoming]
      ;   format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
              [PrevRuleIndex, SlotIndex, PrevRuleIndex]),
          Incomings = [FalseIncoming, ApplyIncoming]
      ),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      format(atom(Line), '  %rule_~w_in_slot_~w = phi i64 ~w',
          [RuleIndex, SlotIndex, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_scalar_rule_input_phi_lines(Rest, RuleIndex, Controls, NextSlotIndex).

plawk_terminal_control_skips_next_rule(Controls, RuleIndex) :-
    nth0(RuleIndex, Controls, Control),
    memberchk(Control, [terminal_next, terminal_break]).

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

plawk_scalar_match_update_ir(StatePlan, Actions, FieldSeparator, OutputSeparator, RuleIndex, NextExits, GlobalIR-IR) :-
    plawk_native_match_update_ir(StatePlan, none, Actions, FieldSeparator, OutputSeparator, RuleIndex, NextExits, GlobalIR-IR).

plawk_native_match_update_ir(StatePlan, AssocPlan, Actions, FieldSeparator, OutputSeparator, RuleIndex, NextExits, GlobalIR-IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_initial_slot_values(RuleIndex, Slots, 0), InitialValues),
    format(atom(Prefix), 'rule_~w_body', [RuleIndex]),
    phrase(plawk_scalar_action_sequence_pairs(Actions, Slots, AssocPlan, FieldSeparator, OutputSeparator,
        Prefix, Prefix, RuleIndex, 0, InitialValues, FinalValues, _NextOpIndex, _ExitLabel, NextExits), Pairs0),
    phrase(plawk_scalar_final_slot_pairs(FinalValues, RuleIndex, 0), FinalPairs),
    append(Pairs0, FinalPairs, Pairs),
    pairs_keys_values(Pairs, GlobalParts, LineParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(LineParts, '\n', IR).

plawk_scalar_initial_slot_values(_RuleIndex, [], _) -->
    [].
plawk_scalar_initial_slot_values(RuleIndex, [_Slot | Rest], SlotIndex) -->
    { plawk_scalar_rule_slot_input(RuleIndex, SlotIndex, Value),
      NextIndex is SlotIndex + 1
    },
    [Value],
    plawk_scalar_initial_slot_values(RuleIndex, Rest, NextIndex).

plawk_scalar_final_slot_pairs([], _RuleIndex, _) -->
    [].
plawk_scalar_final_slot_pairs([Value | Rest], RuleIndex, SlotIndex) -->
    { format(atom(Line), '  %rule_~w_slot_~w = add i64 ~w, 0',
          [RuleIndex, SlotIndex, Value]),
      NextSlotIndex is SlotIndex + 1
    },
    [''-Line],
    plawk_scalar_final_slot_pairs(Rest, RuleIndex, NextSlotIndex).

plawk_scalar_update_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_scalar_update_action(Action) :-
    plawk_scalar_conditional_action(Action).

plawk_scalar_rule_body_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_scalar_rule_body_action(Action) :-
    plawk_rule_body_print_action(Action).
plawk_scalar_rule_body_action(if(_Pattern, ThenActions, ElseActions)) :-
    append(ThenActions, ElseActions, Actions),
    Actions \== [],
    plawk_scalar_branch_body_actions(ThenActions),
    plawk_scalar_branch_body_actions(ElseActions).

plawk_scalar_branch_body_actions(Actions) :-
    plawk_split_branch_control(Actions, BodyActions, _Control),
    maplist(plawk_scalar_rule_body_plain_action, BodyActions).

plawk_scalar_rule_body_plain_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_scalar_rule_body_plain_action(Action) :-
    plawk_rule_body_print_action(Action).
% else-if chains nest an if inside a branch body; the sequence walker
% lowers nested ifs recursively, so validation recurses the same way.
plawk_scalar_rule_body_plain_action(if(Pattern, ThenActions, ElseActions)) :-
    plawk_scalar_rule_body_action(if(Pattern, ThenActions, ElseActions)).

plawk_rule_body_print_action(print(Fields)) :-
    Fields = [_ | _],
    maplist(plawk_rule_body_print_field, Fields).
plawk_rule_body_print_action(printf(string(Format), Args)) :-
    string(Format),
    maplist(plawk_rule_body_print_field, Args).

plawk_rule_body_print_field(field(_)).
plawk_rule_body_print_field(string(_)).
plawk_rule_body_print_field(special('NR')).
plawk_rule_body_print_field(special('NF')).
plawk_rule_body_print_field(int(field(_))).
plawk_rule_body_print_field(Expr) :-
    plawk_i64_general_binary_expr(Expr).
plawk_rule_body_print_field(Expr) :-
    plawk_prolog_call_expr(Expr).
plawk_rule_body_print_field(Expr) :-
    plawk_f64_print_expr(Expr).
plawk_rule_body_print_field(length(field(_))).
plawk_rule_body_print_field(substr(field(_), _Start, _Len)).
plawk_rule_body_print_field(index(field(_), string(_))).
plawk_rule_body_print_field(tolower(field(_))).
plawk_rule_body_print_field(toupper(field(_))).

%% plawk_i64_general_binary_expr(+Expr) is semidet.
%
%  Recognize a native i64 binary expression tree whose leaves are i64
%  primaries, integer literals, or bare numeric field coercions.
plawk_i64_general_binary_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_i64_operand_expr(Left),
    plawk_i64_operand_expr(Right).

plawk_i64_operand_expr(int(Value)) :-
    integer(Value).
plawk_i64_operand_expr(field(FieldIndex)) :-
    integer(FieldIndex),
    FieldIndex >= 0.
plawk_i64_operand_expr(Expr) :-
    plawk_i64_binary_primary_expr(Expr).
plawk_i64_operand_expr(prolog_call(Name, Args)) :-
    plawk_prolog_call_expr(prolog_call(Name, Args)).
plawk_i64_operand_expr(Expr) :-
    plawk_i64_general_binary_expr(Expr).

plawk_prolog_call_expr(prolog_call(Name, Args)) :-
    atom(Name),
    Args = [_ | _],
    maplist(plawk_foreign_arg, Args).

plawk_foreign_arg(field(Index)) :-
    integer(Index),
    Index >= 0.
plawk_foreign_arg(string(String)) :-
    string(String).
plawk_foreign_arg(int(Value)) :-
    integer(Value).

%% plawk_i64_scalar_read_binary_expr(+Expr) is semidet.
%
%  Like plawk_i64_general_binary_expr but operands may also be scalar
%  variable reads (var/1). Only usable where the emitter has the current
%  slot values to substitute: scalar update expressions in rule bodies.
plawk_i64_scalar_read_binary_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_i64_scalar_read_operand_expr(Left),
    plawk_i64_scalar_read_operand_expr(Right).

plawk_i64_scalar_read_operand_expr(var(Name)) :-
    atom(Name).
plawk_i64_scalar_read_operand_expr(Expr) :-
    plawk_i64_operand_expr(Expr).
plawk_i64_scalar_read_operand_expr(Expr) :-
    plawk_i64_scalar_read_binary_expr(Expr).

%% plawk_end_scalar_expr(+Expr) is semidet.
%
%  END-position i64 expression: a binary tree whose leaves are integer
%  literals, scalar variables (final slot values), or NR (the final
%  record count). Fields, NF, length etc. are meaningless after the
%  stream closes and are rejected.
plawk_end_scalar_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_end_scalar_operand_expr(Left),
    plawk_end_scalar_operand_expr(Right).

plawk_end_scalar_operand_expr(int(Value)) :-
    integer(Value).
plawk_end_scalar_operand_expr(var(Name)) :-
    atom(Name).
plawk_end_scalar_operand_expr(special('NR')).
plawk_end_scalar_operand_expr(Expr) :-
    plawk_end_scalar_expr(Expr).

%% plawk_substitute_scalar_reads(+Expr0, +Slots, +Values, -Expr)
%
%  Replace var(Name) leaves with ssa(ValueIR) using the current scalar
%  slot values, so the shared i64 emitters never see a variable read.
plawk_substitute_operation_reads(add(Expr0), Slots, Values, add(Expr)) :-
    !,
    plawk_substitute_scalar_reads(Expr0, Slots, Values, Expr).
plawk_substitute_operation_reads(set(Expr0), Slots, Values, set(Expr)) :-
    !,
    plawk_substitute_scalar_reads(Expr0, Slots, Values, Expr).
plawk_substitute_operation_reads(Operation, _Slots, _Values, Operation).

plawk_substitute_scalar_reads(var(Name), Slots, Values, ssa(Value)) :-
    !,
    nth0(SlotIndex, Slots, scalar_counter(Name)),
    nth0(SlotIndex, Values, Value).
plawk_substitute_scalar_reads(Expr0, Slots, Values, Expr) :-
    plawk_i64_binary_expr(Expr0, _LLVMOp, _NamePart, Left0, Right0),
    !,
    plawk_substitute_scalar_reads(Left0, Slots, Values, Left),
    plawk_substitute_scalar_reads(Right0, Slots, Values, Right),
    Expr0 =.. [Functor, _, _],
    Expr =.. [Functor, Left, Right].
plawk_substitute_scalar_reads(Expr, _Slots, _Values, Expr).

%% plawk_substitute_end_reads(+Expr0, +StatePlan, -Expr)
%
%  END-position substitution: var(Name) becomes the final slot value and
%  NR becomes %plawk_nr, the loop-head record phi, which dominates
%  end_print via close_stream / break_close_stream.
plawk_substitute_end_reads(var(Name), StatePlan, ssa(Value)) :-
    !,
    plawk_state_slot_index(StatePlan, scalar_counter(Name), SlotIndex),
    format(atom(Value), '%final_slot_~w', [SlotIndex]).
plawk_substitute_end_reads(special('NR'), _StatePlan, ssa('%plawk_nr')) :-
    !.
plawk_substitute_end_reads(Expr0, StatePlan, Expr) :-
    plawk_i64_binary_expr(Expr0, _LLVMOp, _NamePart, Left0, Right0),
    !,
    plawk_substitute_end_reads(Left0, StatePlan, Left),
    plawk_substitute_end_reads(Right0, StatePlan, Right),
    Expr0 =.. [Functor, _, _],
    Expr =.. [Functor, Left, Right].
plawk_substitute_end_reads(Expr, _StatePlan, Expr).

plawk_i64_binary_primary_expr(special('NR')).
plawk_i64_binary_primary_expr(special('NF')).
plawk_i64_binary_primary_expr(int(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_binary_primary_expr(length(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_binary_primary_expr(index(field(FieldIndex), string(Needle))) :-
    FieldIndex >= 0,
    string(Needle).

plawk_i64_scalar_primary_expr(special('NR')).
plawk_i64_scalar_primary_expr(special('NF')).
plawk_i64_scalar_primary_expr(int(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_scalar_primary_expr(length(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_scalar_primary_expr(index(field(FieldIndex), string(Needle))) :-
    FieldIndex >= 0,
    string(Needle).

plawk_i64_binary_expr(add_i64(Left, Right), add, add, Left, Right).
plawk_i64_binary_expr(sub_i64(Left, Right), sub, sub, Left, Right).
plawk_i64_binary_expr(mul_i64(Left, Right), mul, mul, Left, Right).
plawk_i64_binary_expr(div_i64(Left, Right), sdiv, div, Left, Right).
plawk_i64_binary_expr(mod_i64(Left, Right), srem, mod, Left, Right).

plawk_i64_binary_print_kind(add, int_add).
plawk_i64_binary_print_kind(sub, int_sub).
plawk_i64_binary_print_kind(mul, int_mul).
plawk_i64_binary_print_kind(div, int_div).
plawk_i64_binary_print_kind(mod, int_mod).

% Reads count as slot names too: a variable read before any write gets a
% zero-initialized slot, matching awk's uninitialized-variable semantics.
plawk_scalar_update_action_name(Action, Name) :-
    plawk_scalar_action_update(Action, WriteName, Operation),
    (   Name = WriteName
    ;   plawk_scalar_operation_expr(Operation, Expr),
        plawk_expr_scalar_read_name(Expr, Name)
    ).
plawk_scalar_update_action_name(if(_Pattern, ThenActions, ElseActions), Name) :-
    ( member(Action, ThenActions)
    ; member(Action, ElseActions)
    ),
    plawk_scalar_update_action_name(Action, Name).

plawk_expr_scalar_read_name(var(Name), Name).
plawk_expr_scalar_read_name(Expr, Name) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_scalar_read_name(Left, Name)
    ; plawk_expr_scalar_read_name(Right, Name)
    ).

plawk_scalar_action_update(inc(var(Name)), Name, add(const(1))).
plawk_scalar_action_update(add(var(Name), int(Value)), Name, add(const(Value))) :-
    integer(Value),
    Value >= 0.
plawk_scalar_action_update(add(var(Name), length(field(FieldIndex))), Name, add(length(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(add(var(Name), field(FieldIndex)), Name, add(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(add(var(Name), int(field(FieldIndex))), Name, add(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(add(var(Name), Expr), Name, add(Expr)) :-
    plawk_i64_scalar_primary_expr(Expr).
plawk_scalar_action_update(add(var(Name), Expr), Name, add(Expr)) :-
    plawk_i64_scalar_read_binary_expr(Expr).
plawk_scalar_action_update(add(var(Name), var(Read)), Name, add(var(Read))) :-
    atom(Read).
plawk_scalar_action_update(add(var(Name), prolog_call(Pred, Args)), Name,
        add(prolog_call(Pred, Args))) :-
    plawk_prolog_call_expr(prolog_call(Pred, Args)).
plawk_scalar_action_update(set(var(Name), int(Value)), Name, set(const(Value))) :-
    integer(Value),
    Value >= 0.
plawk_scalar_action_update(set(var(Name), length(field(FieldIndex))), Name, set(length(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(set(var(Name), field(FieldIndex)), Name, set(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(set(var(Name), int(field(FieldIndex))), Name, set(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(set(var(Name), Expr), Name, set(Expr)) :-
    plawk_i64_scalar_primary_expr(Expr).
plawk_scalar_action_update(set(var(Name), Expr), Name, set(Expr)) :-
    plawk_i64_scalar_read_binary_expr(Expr).
plawk_scalar_action_update(set(var(Name), var(Read)), Name, set(var(Read))) :-
    atom(Read).
plawk_scalar_action_update(set(var(Name), prolog_call(Pred, Args)), Name,
        set(prolog_call(Pred, Args))) :-
    plawk_prolog_call_expr(prolog_call(Pred, Args)).

plawk_scalar_action_sequence_pairs([], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, CurrentLabel, []) -->
    [].
plawk_scalar_action_sequence_pairs([Action | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_scalar_action_update(Action, Name, Operation0),
      nth0(SlotIndex, Slots, scalar_counter(Name)),
      nth0(SlotIndex, Values0, InputValue),
      plawk_substitute_operation_reads(Operation0, Slots, Values0, Operation),
      plawk_scalar_update_operation_ir(Operation, FieldSeparator, Prefix, SlotIndex,
          OpIndex, InputValue, NextValue, Pair),
      replace_nth0(SlotIndex, Values0, NextValue, Values1),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values1, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([Action | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_assoc_increment_action(Action, ArrayName-KeyIndex),
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      plawk_assoc_update_operation_ir(Prefix, OpIndex, TableIndex, KeyIndex,
          FieldSeparator, Pair, AssocExitLabel),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, AssocExitLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([print(Fields) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_rule_body_print_action(print(Fields)),
      format(atom(PrintPrefix), '~w_print_~w', [Prefix, OpIndex]),
      plawk_prefixed_print_action_ir(Fields, FieldSeparator, OutputSeparator, PrintPrefix, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([printf(string(Format), Args) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_rule_body_print_action(printf(string(Format), Args)),
      format(atom(PrintPrefix), '~w_printf_~w', [Prefix, OpIndex]),
      plawk_prefixed_printf_action_ir(Format, Args, FieldSeparator, PrintPrefix, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([next], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, none, [branch_next(CurrentLabel, Values)]) -->
    [].
plawk_scalar_action_sequence_pairs([break], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, break, [branch_break(CurrentLabel, Values)]) -->
    [].
plawk_scalar_action_sequence_pairs([if(Pattern, ThenActions, ElseActions) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex, OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(GlobalBase), '~w_if_~w', [Prefix, OpIndex]),
      format(atom(CondValue), '%~w_if_~w_cond', [Prefix, OpIndex]),
      plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, CondValue, GuardGlobalIR-GuardIR),
      format(atom(ThenLabel), '~w_if_~w_then', [Prefix, OpIndex]),
      format(atom(ElseLabel), '~w_if_~w_else', [Prefix, OpIndex]),
      format(atom(DoneLabel), '~w_if_~w_done', [Prefix, OpIndex]),
      phrase(plawk_scalar_action_sequence_pairs(ThenActions, Slots, AssocPlan, FieldSeparator, OutputSeparator,
          ThenLabel, ThenLabel, RuleIndex, 0, Values0, ThenValues, _ThenOpIndex, ThenExitLabel, ThenNextExits), ThenPairs),
      phrase(plawk_scalar_action_sequence_pairs(ElseActions, Slots, AssocPlan, FieldSeparator, OutputSeparator,
          ElseLabel, ElseLabel, RuleIndex, 0, Values0, ElseValues, _ElseOpIndex, ElseExitLabel, ElseNextExits), ElsePairs),
      pairs_keys_values(ThenPairs, ThenGlobalParts, ThenLineParts),
      pairs_keys_values(ElsePairs, ElseGlobalParts, ElseLineParts),
      atomic_list_concat(ThenGlobalParts, '\n', ThenGlobalIR),
      atomic_list_concat(ElseGlobalParts, '\n', ElseGlobalIR),
      atomic_list_concat(ThenLineParts, '\n', ThenIR),
      atomic_list_concat(ElseLineParts, '\n', ElseIR),
      plawk_branch_to_done_ir(ThenExitLabel, DoneLabel, ThenDoneIR),
      plawk_branch_to_done_ir(ElseExitLabel, DoneLabel, ElseDoneIR),
      plawk_scalar_if_join_pairs(ThenExitLabel, ThenValues, ElseExitLabel, ElseValues,
          Prefix, OpIndex, PhiPairs),
      pairs_keys_values(PhiPairs, _PhiGlobalParts, PhiLineParts),
      atomic_list_concat(PhiLineParts, '\n', PhiIR),
      pairs_keys(PhiPairs, Values1),
      format(atom(IR),
'~w
  br i1 ~w, label %~w, label %~w

~w:
~w
~w

~w:
~w
~w

~w:
~w',
          [GuardIR, CondValue, ThenLabel, ElseLabel,
           ThenLabel, ThenIR, ThenDoneIR,
           ElseLabel, ElseIR, ElseDoneIR,
           DoneLabel, PhiIR]),
      atomic_list_concat([GuardGlobalIR, ThenGlobalIR, ElseGlobalIR], '\n', GlobalIR),
      append(ThenNextExits, ElseNextExits, BranchNextExits),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, DoneLabel, RuleIndex,
        NextOpIndex, Values1, Values, FinalOpIndex, ExitLabel, RestNextExits),
    { append(BranchNextExits, RestNextExits, NextExits) }.

plawk_scalar_update_operation_ir(add(Expr), FieldSeparator, Prefix, SlotIndex,
        OpIndex, InputValue, NextValue, GlobalIR-IR) :-
    plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR),
    format(atom(NextValue), '%~w_slot_~w_op_~w', [Prefix, SlotIndex, OpIndex]),
    format(atom(AddLine), '  ~w = add i64 ~w, ~w',
        [NextValue, InputValue, ValueIR]),
    plawk_join_nonempty_ir([SetupIR, AddLine], IR).
plawk_scalar_update_operation_ir(set(Expr), FieldSeparator, Prefix, SlotIndex,
        OpIndex, _InputValue, NextValue, GlobalIR-IR) :-
    plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR),
    format(atom(NextValue), '%~w_slot_~w_op_~w', [Prefix, SlotIndex, OpIndex]),
    format(atom(SetLine), '  ~w = add i64 0, ~w', [NextValue, ValueIR]),
    plawk_join_nonempty_ir([SetupIR, SetLine], IR).

plawk_scalar_numeric_expr_ir(ssa(Value), _FieldSeparator, _Prefix, _SlotIndex,
        _OpIndex, Value, '', '') :-
    atom(Value).
plawk_scalar_numeric_expr_ir(prolog_call(Name, Args), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(CallBase), '~w_slot_~w_op_~w_prolog_call',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(prolog_call(Name, Args), FieldSeparator, CallBase,
        CallBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(const(Value), _FieldSeparator, _Prefix, _SlotIndex,
        _OpIndex, ValueIR, GlobalIR, IR) :-
    plawk_i64_expr_ir_parts(const(Value), 0, scalar_const, scalar_const_global,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(length(FieldIndex), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(LengthBase), '~w_slot_~w_op_~w_len', [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(length(FieldIndex), FieldSeparator, LengthBase, LengthBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(field_i64(FieldIndex), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(ParseBase), '~w_slot_~w_op_~w_field_i64',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(field_i64(FieldIndex), FieldSeparator, ParseBase, ParseBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    plawk_i64_scalar_primary_expr(Expr),
    format(atom(PrimaryBase), '~w_slot_~w_op_~w_i64_primary',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(Expr, FieldSeparator, PrimaryBase, PrimaryBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, NamePart, _Left, _Right),
    format(atom(BinaryBase), '~w_slot_~w_op_~w_i64_~w',
        [Prefix, SlotIndex, OpIndex, NamePart]),
    plawk_i64_expr_ir_parts(Expr, FieldSeparator, BinaryBase,
        BinaryBase, ValueIR, GlobalIR, IR).

plawk_i64_expr_ir_parts(Expr, FieldSeparator, Base, GlobalBase, ValueIR, IR) :-
    plawk_i64_expr_ir_parts(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalIR, SetupIR),
    plawk_join_nonempty_ir([GlobalIR, SetupIR], IR).

plawk_i64_expr_ir_parts(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalIR, SetupIR) :-
    plawk_i64_expr_ir(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(SetupParts, '\n', SetupIR).

plawk_i64_expr_ir(const(Value), _FieldSeparator, _Base, _GlobalBase, ValueIR, [], []) :-
    integer(Value),
    format(atom(ValueIR), '~w', [Value]).
plawk_i64_expr_ir(int(Value), _FieldSeparator, _Base, _GlobalBase, ValueIR, [], []) :-
    integer(Value),
    format(atom(ValueIR), '~w', [Value]).
plawk_i64_expr_ir(ssa(Value), _FieldSeparator, _Base, _GlobalBase, Value, [], []) :-
    atom(Value).
plawk_i64_expr_ir(prolog_call(Name, Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { i64, i1 } @plawk_foreign_call_~w_~w(~w)',
        [Base, Name, NArgs, CallArgsIR]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, i64 %~w_val, i64 0', [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(ArgSetupParts, [ResIR, ValIR, OkIR, SelIR], SetupParts).
plawk_i64_expr_ir(field(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    integer(FieldIndex),
    plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(nr, _FieldSeparator, _Base, _GlobalBase, '%current_nr', [], []).
plawk_i64_expr_ir(nf, FieldSeparator, Base, _GlobalBase, ValueIR, [], [CountIR]) :-
    llvm_emit_atom_field_count('%line', FieldSeparator, Base, CountIR),
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(special('NR'), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(nr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(special('NF'), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(nf, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(length(field(FieldIndex)), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(length(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(length(FieldIndex), FieldSeparator, Base, _GlobalBase, ValueIR, [], [LengthIR]) :-
    llvm_emit_atom_field_length('%line', FieldIndex, FieldSeparator, Base, LengthIR),
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(int(field(FieldIndex)), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, _GlobalBase, ValueIR, [], [ParseIR]) :-
    format(atom(ValueIR), '%~w_value_or_default', [Base]),
    llvm_emit_atom_field_i64_or_default('%line', FieldIndex, FieldSeparator, 0,
        Base, ValueIR, ParseIR).
plawk_i64_expr_ir(index(field(FieldIndex), string(Needle)), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(index(FieldIndex, Needle), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_binary_expr(Expr, LLVMOp, _NamePart, Left, Right),
    format(atom(LeftBase), '~w_lhs', [Base]),
    format(atom(LeftGlobalBase), '~w_lhs', [GlobalBase]),
    plawk_i64_expr_ir(Left, FieldSeparator, LeftBase, LeftGlobalBase,
        LeftValueIR, LeftGlobalParts, LeftSetupParts),
    format(atom(RightBase), '~w_rhs', [Base]),
    format(atom(RightGlobalBase), '~w_rhs', [GlobalBase]),
    plawk_i64_expr_ir(Right, FieldSeparator, RightBase, RightGlobalBase,
        RightValueIR, RightGlobalParts, RightSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    plawk_i64_binary_op_lines(LLVMOp, Base, LeftValueIR, RightValueIR, OpLines),
    append(LeftGlobalParts, RightGlobalParts, GlobalParts),
    append([LeftSetupParts, RightSetupParts, OpLines], SetupParts).
plawk_i64_expr_ir(index(FieldIndex, Needle), FieldSeparator, Base, GlobalBase,
        ValueIR, [GlobalIR], [CallIR]) :-
    llvm_emit_atom_field_index(GlobalBase, '%line', FieldIndex, Needle, FieldSeparator,
        Base, GlobalIR-CallIR),
    format(atom(ValueIR), '%~w', [Base]).

%% plawk_i64_binary_op_lines(+LLVMOp, +Base, +LeftIR, +RightIR, -Lines)
%
%  add/sub/mul emit one instruction. sdiv/srem are guarded so awk-side
%  division stays defined: a zero divisor yields 0, and the
%  INT64_MIN / -1 overflow case divides by 1 instead, wrapping to
%  INT64_MIN for `/` and 0 for `%`.
plawk_i64_binary_op_lines(sdiv, Base, LeftIR, RightIR, Lines) :-
    !,
    plawk_i64_guarded_div_lines(sdiv, Base, LeftIR, RightIR, Lines).
plawk_i64_binary_op_lines(srem, Base, LeftIR, RightIR, Lines) :-
    !,
    plawk_i64_guarded_div_lines(srem, Base, LeftIR, RightIR, Lines).
plawk_i64_binary_op_lines(LLVMOp, Base, LeftIR, RightIR, [Line]) :-
    format(atom(Line), '  %~w = ~w i64 ~w, ~w',
        [Base, LLVMOp, LeftIR, RightIR]).

%% plawk_expr_is_double(+Expr) is semidet.
%
%  An expression tree is double-typed when any leaf is a float literal
%  or a float($N) coercion; i64 leaves in a double tree promote via
%  sitofp at emission time.
plawk_expr_is_double(float_const(_Mantissa, _Denominator)).
plawk_expr_is_double(float_field(_Index)).
plawk_expr_is_double(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_is_double(Left)
    ; plawk_expr_is_double(Right)
    ).

%% plawk_f64_print_expr(+Expr) is semidet.
%
%  Valid double print expression: double-typed with recognizable leaves.
plawk_f64_print_expr(Expr) :-
    plawk_expr_is_double(Expr),
    plawk_f64_operand_expr(Expr).

plawk_f64_operand_expr(float_const(Mantissa, Denominator)) :-
    integer(Mantissa),
    integer(Denominator),
    Denominator > 0.
plawk_f64_operand_expr(float_field(Index)) :-
    integer(Index),
    Index >= 0.
plawk_f64_operand_expr(Expr) :-
    plawk_i64_operand_expr(Expr).
plawk_f64_operand_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_f64_operand_expr(Left),
    plawk_f64_operand_expr(Right).

%% plawk_f64_expr_ir(+Expr, +FieldSeparator, +Base, +GlobalBase, -ValueIR,
%%     -GlobalParts, -SetupParts)
%
%  Double expression emitter. Float literals emit as an exact integer
%  ratio (fdiv of two exactly representable doubles gives the correctly
%  rounded value, matching strtod). i64 subtrees emit through the i64
%  emitter and promote with sitofp; IEEE semantics apply to / (no
%  divide-by-zero guard: x/0.0 is inf, 0.0/0.0 is nan, as in awk).
plawk_f64_expr_ir(float_const(Mantissa, Denominator), _FieldSeparator, Base,
        _GlobalBase, ValueIR, [], [ConstIR]) :-
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(ConstIR), '  ~w = fdiv double ~w.0, ~w.0',
        [ValueIR, Mantissa, Denominator]).
plawk_f64_expr_ir(float_field(Index), FieldSeparator, Base, _GlobalBase,
        ValueIR, [], [CallIR]) :-
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(CallIR),
        '  ~w = call double @wam_atom_field_f64_value(%Value %line, i64 ~w, i8 ~w)',
        [ValueIR, Index, FieldSeparator]).
plawk_f64_expr_ir(Expr, FieldSeparator, Base, GlobalBase, ValueIR,
        GlobalParts, SetupParts) :-
    plawk_expr_is_double(Expr),
    plawk_i64_binary_expr(Expr, LLVMOp, _NamePart, Left, Right),
    !,
    plawk_f64_llvm_op(LLVMOp, F64Op),
    format(atom(LeftBase), '~w_lhs', [Base]),
    format(atom(LeftGlobalBase), '~w_lhs', [GlobalBase]),
    plawk_f64_expr_ir(Left, FieldSeparator, LeftBase, LeftGlobalBase,
        LeftValueIR, LeftGlobalParts, LeftSetupParts),
    format(atom(RightBase), '~w_rhs', [Base]),
    format(atom(RightGlobalBase), '~w_rhs', [GlobalBase]),
    plawk_f64_expr_ir(Right, FieldSeparator, RightBase, RightGlobalBase,
        RightValueIR, RightGlobalParts, RightSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(OpIR), '  ~w = ~w double ~w, ~w',
        [ValueIR, F64Op, LeftValueIR, RightValueIR]),
    append(LeftGlobalParts, RightGlobalParts, GlobalParts),
    append([LeftSetupParts, RightSetupParts, [OpIR]], SetupParts).
plawk_f64_expr_ir(Expr, FieldSeparator, Base, GlobalBase, ValueIR,
        GlobalParts, SetupParts) :-
    % i64-typed subtree in a double context: emit as i64, then promote.
    format(atom(IntBase), '~w_int', [Base]),
    format(atom(IntGlobalBase), '~w_int', [GlobalBase]),
    plawk_i64_expr_ir(Expr, FieldSeparator, IntBase, IntGlobalBase,
        IntValueIR, GlobalParts, IntSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(PromoteIR), '  ~w = sitofp i64 ~w to double',
        [ValueIR, IntValueIR]),
    append(IntSetupParts, [PromoteIR], SetupParts).

plawk_f64_llvm_op(add, fadd).
plawk_f64_llvm_op(sub, fsub).
plawk_f64_llvm_op(mul, fmul).
plawk_f64_llvm_op(sdiv, fdiv).
plawk_f64_llvm_op(srem, frem).

plawk_i64_guarded_div_lines(LLVMOp, Base, LeftIR, RightIR, Lines) :-
    format(atom(DenZero),
        '  %~w_den_zero = icmp eq i64 ~w, 0', [Base, RightIR]),
    format(atom(LhsMin),
        '  %~w_lhs_min = icmp eq i64 ~w, -9223372036854775808', [Base, LeftIR]),
    format(atom(RhsNegOne),
        '  %~w_rhs_negone = icmp eq i64 ~w, -1', [Base, RightIR]),
    format(atom(Overflow),
        '  %~w_overflow = and i1 %~w_lhs_min, %~w_rhs_negone',
        [Base, Base, Base]),
    format(atom(DenBad),
        '  %~w_den_bad = or i1 %~w_den_zero, %~w_overflow', [Base, Base, Base]),
    format(atom(SafeDen),
        '  %~w_safe_den = select i1 %~w_den_bad, i64 1, i64 ~w',
        [Base, Base, RightIR]),
    format(atom(Raw),
        '  %~w_raw = ~w i64 ~w, %~w_safe_den', [Base, LLVMOp, LeftIR, Base]),
    format(atom(Result),
        '  %~w = select i1 %~w_den_zero, i64 0, i64 %~w_raw',
        [Base, Base, Base]),
    Lines = [DenZero, LhsMin, RhsNegOne, Overflow, DenBad, SafeDen, Raw, Result].

plawk_branch_to_done_ir(none, _DoneLabel, '  br label %continue_loop') :-
    !.
plawk_branch_to_done_ir(break, _DoneLabel, '  br label %break_close_stream') :-
    !.
plawk_branch_to_done_ir(ExitLabel, DoneLabel, IR) :-
    ExitLabel \== none,
    ExitLabel \== break,
    format(atom(IR), '  br label %~w', [DoneLabel]).

plawk_branch_terminal_exit(none).
plawk_branch_terminal_exit(break).

plawk_scalar_if_join_pairs(ThenExitLabel, _ThenValues, ElseExitLabel, _ElseValues, _Prefix, _OpIndex, _Pairs) :-
    plawk_branch_terminal_exit(ThenExitLabel),
    plawk_branch_terminal_exit(ElseExitLabel),
    !,
    fail.
plawk_scalar_if_join_pairs(ThenExitLabel, _ThenValues, _ElseExitLabel, ElseValues, _Prefix, _OpIndex, Pairs) :-
    plawk_branch_terminal_exit(ThenExitLabel),
    !,
    plawk_scalar_if_passthrough_pairs(ElseValues, Pairs).
plawk_scalar_if_join_pairs(_ThenExitLabel, ThenValues, ElseExitLabel, _ElseValues, _Prefix, _OpIndex, Pairs) :-
    plawk_branch_terminal_exit(ElseExitLabel),
    !,
    plawk_scalar_if_passthrough_pairs(ThenValues, Pairs).
plawk_scalar_if_join_pairs(ThenExitLabel, ThenValues, ElseExitLabel, ElseValues, Prefix, OpIndex, Pairs) :-
    phrase(plawk_scalar_if_phi_lines(ThenValues, ElseValues, Prefix, OpIndex,
        ThenExitLabel, ElseExitLabel, 0), Pairs).

plawk_scalar_if_passthrough_pairs([], []).
plawk_scalar_if_passthrough_pairs([Value | Rest], [Value-'' | Pairs]) :-
    plawk_scalar_if_passthrough_pairs(Rest, Pairs).

plawk_assoc_update_operation_ir(Prefix, OpIndex, TableIndex, KeyIndex,
        FieldSeparator, ''-IR, DoneLabel) :-
    format(atom(Label), '~w_assoc_~w', [Prefix, OpIndex]),
    format(atom(HaveLabel), '~w_assoc_~w_have_key', [Prefix, OpIndex]),
    format(atom(DoneLabel), '~w_assoc_~w_done', [Prefix, OpIndex]),
    format(atom(SliceValue), '%~w_assoc_~w_key_slice', [Prefix, OpIndex]),
    format(atom(KeyPtr), '%~w_assoc_~w_key_ptr', [Prefix, OpIndex]),
    format(atom(KeyLen), '%~w_assoc_~w_key_len', [Prefix, OpIndex]),
    format(atom(KeyMissing), '%~w_assoc_~w_key_missing', [Prefix, OpIndex]),
    format(atom(KeyId), '%~w_assoc_~w_key_id', [Prefix, OpIndex]),
    format(atom(CountValue), '%~w_assoc_~w_count', [Prefix, OpIndex]),
    format(atom(IR),
'  br label %~w

~w:
  ~w = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)
  ~w = extractvalue %WamSlice ~w, 0
  ~w = extractvalue %WamSlice ~w, 1
  ~w = icmp eq i8* ~w, null
  br i1 ~w, label %~w, label %~w

~w:
  ~w = call i64 @wam_intern_atom(i8* ~w, i64 ~w)
  ~w = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 ~w, i64 1)
  br label %~w

~w:',
        [Label,
         Label,
         SliceValue, KeyIndex, FieldSeparator,
         KeyPtr, SliceValue,
         KeyLen, SliceValue,
         KeyMissing, KeyPtr,
         KeyMissing, DoneLabel, HaveLabel,
         HaveLabel,
         KeyId, KeyPtr, KeyLen,
         CountValue, TableIndex, KeyId,
         DoneLabel,
         DoneLabel]).

plawk_scalar_if_phi_lines([], [], _Prefix, _OpIndex, _ThenLabel, _ElseLabel, _) -->
    [].
plawk_scalar_if_phi_lines([ThenValue | ThenRest], [ElseValue | ElseRest],
        Prefix, OpIndex, ThenLabel, ElseLabel, SlotIndex) -->
    { format(atom(PhiValue), '%~w_if_~w_slot_~w', [Prefix, OpIndex, SlotIndex]),
      format(atom(Line), '  ~w = phi i64 [~w, %~w], [~w, %~w]',
          [PhiValue, ThenValue, ThenLabel, ElseValue, ElseLabel]),
      NextSlotIndex is SlotIndex + 1
    },
    [PhiValue-Line],
    plawk_scalar_if_phi_lines(ThenRest, ElseRest, Prefix, OpIndex, ThenLabel, ElseLabel, NextSlotIndex).

replace_nth0(0, [_Old | Rest], Value, [Value | Rest]) :-
    !.
replace_nth0(Index, [Head | Rest], Value, [Head | NewRest]) :-
    Index > 0,
    NextIndex is Index - 1,
    replace_nth0(NextIndex, Rest, Value, NewRest).

plawk_scalar_next_phi_ir(StatePlan, RuleCount, Controls, BranchNextExits, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_next_phi_lines(Slots, RuleCount, Controls, BranchNextExits, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_next_phi_lines([], _RuleCount, _Controls, _BranchNextExits, _) -->
    [].
plawk_scalar_next_phi_lines([_Slot | Rest], RuleCount, Controls, BranchNextExits, Index) -->
    { LastRuleIndex is RuleCount - 1,
      plawk_scalar_rule_input_value(LastRuleIndex, Index, FalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [FalseValue, LastRuleIndex]),
      findall(ApplyIncoming,
          ( between(0, LastRuleIndex, RuleIndex),
            ( ( RuleIndex =:= LastRuleIndex,
                \+ nth0(RuleIndex, Controls, terminal_break)
              )
            ; nth0(RuleIndex, Controls, terminal_next)
            ),
            format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
                [RuleIndex, Index, RuleIndex])
          ),
          ApplyIncomings),
      plawk_branch_next_phi_incomings(BranchNextExits, Index, BranchNextIncomings),
      append([FalseIncoming | ApplyIncomings], BranchNextIncomings, Incomings),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      format(atom(Line), '  %next_slot_~w = phi i64 ~w', [Index, IncomingIR]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_next_phi_lines(Rest, RuleCount, Controls, BranchNextExits, NextIndex).

plawk_branch_next_phi_incomings([], _SlotIndex, []).
plawk_branch_next_phi_incomings([branch_next(Label, Values) | Rest], SlotIndex, [Incoming | Incomings]) :-
    !,
    nth0(SlotIndex, Values, Value),
    format(atom(Incoming), '[~w, %~w]', [Value, Label]),
    plawk_branch_next_phi_incomings(Rest, SlotIndex, Incomings).
plawk_branch_next_phi_incomings([_Exit | Rest], SlotIndex, Incomings) :-
    plawk_branch_next_phi_incomings(Rest, SlotIndex, Incomings).

plawk_scalar_end_print_ir(PrintFields, StatePlan, OutputSeparator, IR) :-
    phrase(plawk_scalar_end_print_lines(PrintFields, StatePlan, OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_end_print_lines([], _StatePlan, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          end_newline_fmt, printed_end_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].
plawk_scalar_end_print_lines([var(Name) | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    { plawk_state_slot_index(StatePlan, scalar_counter(Name), SlotIndex),
      format(atom(FmtVar), 'end_i64_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_i64_~w', [PrintIndex]),
      format(atom(ValueIR), '%final_slot_~w', [SlotIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
plawk_scalar_end_print_lines([special('NR') | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_nr_print_lines(PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
plawk_scalar_end_print_lines([Expr | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    { plawk_end_scalar_expr(Expr) },
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_expr_print_lines(Expr, StatePlan, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
plawk_scalar_end_print_lines([string(Value) | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).

plawk_end_nr_print_lines(PrintIndex) -->
    { format(atom(FmtVar), 'end_nr_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_nr_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, '%plawk_nr',
          [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].

plawk_end_expr_print_lines(Expr, StatePlan, PrintIndex) -->
    { plawk_substitute_end_reads(Expr, StatePlan, SubstitutedExpr),
      format(atom(Base), 'plawk_end_expr_~w', [PrintIndex]),
      plawk_i64_expr_ir(SubstitutedExpr, 32, Base, Base, ValueIR, [], SetupParts),
      format(atom(FmtVar), 'end_expr_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_expr_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      append(SetupParts, [FmtPtr, PrintCall], Lines)
    },
    plawk_emit_lines(Lines).

plawk_emit_lines([]) -->
    [].
plawk_emit_lines([Line | Rest]) -->
    [Line],
    plawk_emit_lines(Rest).

plawk_end_string_print_lines(Value, PrintIndex) -->
    { string_codes(Value, Codes),
      length(Codes, StringLen),
      BytesLen is StringLen + 1,
      format(atom(StringPtr),
          '  %end_string_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_end_print_string_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(FmtVar), 'end_string_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_string_~w', [PrintIndex]),
      format(atom(PtrIR), '%end_string_~w_ptr', [PrintIndex]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR,
          [FmtPtr, PrintCall])
    },
    [StringPtr, FmtPtr, PrintCall].

plawk_scalar_end_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %printed_end_separator_~w = call i32 @putchar(i32 ~w)',
          [PrintIndex, OutputSeparator])
    },
    [SpaceCall].

plawk_pattern_guard_ir(always, GuardIR) :-
    GuardIR = ''-'  %is_match = icmp eq i1 true, true'.

plawk_pattern_guard_ir(prefix(Prefix), GuardIR) :-
    llvm_emit_atom_prefix_guard(plawk_surface_prefix, '%line', Prefix,
        '%is_match', GuardIR).
plawk_pattern_guard_ir(contains(Needle), GuardIR) :-
    plawk_pattern_guard_ir(contains(Needle), 32, GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GuardIR) :-
    plawk_pattern_guard_ir(field_eq(Index, Value), 32, GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), GuardIR) :-
    plawk_pattern_guard_ir(field_cmp(Index, Op, Value), 32, GuardIR).

plawk_pattern_guard_ir(always, _FieldSeparator, GuardIR) :-
    plawk_pattern_guard_ir(always, GuardIR).
plawk_pattern_guard_ir(prefix(Prefix), _FieldSeparator, GuardIR) :-
    plawk_pattern_guard_ir(prefix(Prefix), GuardIR).
plawk_pattern_guard_ir(contains(Needle), FieldSeparator, GuardIR) :-
    plawk_literal_contains_guard_ir(plawk_surface_contains, Needle, FieldSeparator,
        '%is_match', GuardIR).
plawk_pattern_guard_ir(field_eq(Index, Value), FieldSeparator, GuardIR) :-
    llvm_emit_atom_field_eq_guard(plawk_surface_field_eq, '%line', Index, Value,
        FieldSeparator, '%is_match', GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), FieldSeparator, ''-GuardCallIR) :-
    plawk_field_cmp_op_code(Op, OpCode),
    llvm_emit_atom_field_i64_cmp_guard('%line', Index, OpCode, Value,
        FieldSeparator, '%is_match', GuardCallIR).
plawk_pattern_guard_ir(field_match(Index, Regex), FieldSeparator, GuardIR) :-
    llvm_emit_regex_field_match_guard(plawk_surface_regex, '%line', Index,
        Regex, FieldSeparator, '%is_match', GuardIR).
plawk_pattern_guard_ir(prolog_guard(Name, Args), FieldSeparator, GuardIR) :-
    plawk_foreign_guard_call_ir(Name, Args, FieldSeparator,
        plawk_surface_prolog_guard, '%is_match', GuardIR).
plawk_pattern_guard_ir(Pattern, FieldSeparator, GuardIR) :-
    plawk_combined_pattern(Pattern),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, plawk_surface_pattern,
        '%is_match', GuardIR).

plawk_pattern_guard_ir(always, _GlobalBase, MatchValue, GuardIR) :-
    format(atom(GuardCallIR), '  ~w = icmp eq i1 true, true', [MatchValue]),
    GuardIR = ''-GuardCallIR.

plawk_pattern_guard_ir(prefix(Prefix), GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_atom_prefix_guard(GlobalBase, '%line', Prefix, MatchValue,
        GuardIR).
plawk_pattern_guard_ir(contains(Needle), GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(contains(Needle), 32, GlobalBase, MatchValue, GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(field_eq(Index, Value), 32, GlobalBase, MatchValue,
        GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(field_cmp(Index, Op, Value), 32, GlobalBase,
        MatchValue, GuardIR).

plawk_pattern_guard_ir(always, _FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(always, GlobalBase, MatchValue, GuardIR).
plawk_pattern_guard_ir(prefix(Prefix), _FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(prefix(Prefix), GlobalBase, MatchValue, GuardIR).
plawk_pattern_guard_ir(contains(Needle), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_literal_contains_guard_ir(GlobalBase, Needle, FieldSeparator, MatchValue,
        GuardIR).
plawk_pattern_guard_ir(field_eq(Index, Value), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_atom_field_eq_guard(GlobalBase, '%line', Index, Value, FieldSeparator,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), FieldSeparator, _GlobalBase, MatchValue, ''-GuardCallIR) :-
    plawk_field_cmp_op_code(Op, OpCode),
    llvm_emit_atom_field_i64_cmp_guard('%line', Index, OpCode, Value,
        FieldSeparator, MatchValue, GuardCallIR).
plawk_pattern_guard_ir(field_match(Index, Regex), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_regex_field_match_guard(GlobalBase, '%line', Index, Regex,
        FieldSeparator, MatchValue, GuardIR).
plawk_pattern_guard_ir(prolog_guard(Name, Args), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_foreign_guard_call_ir(Name, Args, FieldSeparator, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(and_pat(Left, Right), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_binary_pattern_guard_ir(and, Left, Right, FieldSeparator, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(or_pat(Left, Right), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_binary_pattern_guard_ir(or, Left, Right, FieldSeparator, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(not_pat(Pattern), FieldSeparator, GlobalBase, MatchValue, GlobalIR-GuardCallIR) :-
    format(atom(InnerBase), '~w_n', [GlobalBase]),
    format(atom(InnerValue), '~w_n', [MatchValue]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, InnerBase, InnerValue,
        GlobalIR-InnerCallIR),
    format(atom(GuardCallIR),
'~w
  ~w = xor i1 ~w, true',
        [InnerCallIR, MatchValue, InnerValue]).

plawk_combined_pattern(and_pat(_Left, _Right)).
plawk_combined_pattern(or_pat(_Left, _Right)).
plawk_combined_pattern(not_pat(_Pattern)).

%% plawk_binary_pattern_guard_ir(+Op, +Left, +Right, +FieldSeparator,
%%     +GlobalBase, +MatchValue, -GuardIR)
%
%  Combine two pattern guards with a bitwise i1 op. The base guards are
%  side-effect-free straight-line checks, so evaluating both operands
%  keeps the combined guard a single block; awk's short-circuit order
%  is unobservable here.
plawk_binary_pattern_guard_ir(Op, Left, Right, FieldSeparator, GlobalBase,
        MatchValue, GlobalIR-GuardCallIR) :-
    format(atom(LeftBase), '~w_l', [GlobalBase]),
    format(atom(LeftValue), '~w_l', [MatchValue]),
    plawk_pattern_guard_ir(Left, FieldSeparator, LeftBase, LeftValue,
        LeftGlobalIR-LeftCallIR),
    format(atom(RightBase), '~w_r', [GlobalBase]),
    format(atom(RightValue), '~w_r', [MatchValue]),
    plawk_pattern_guard_ir(Right, FieldSeparator, RightBase, RightValue,
        RightGlobalIR-RightCallIR),
    plawk_join_nonempty_ir([LeftGlobalIR, RightGlobalIR], GlobalIR),
    format(atom(GuardCallIR),
'~w
~w
  ~w = ~w i1 ~w, ~w',
        [LeftCallIR, RightCallIR, MatchValue, Op, LeftValue, RightValue]).

%% plawk_foreign_args_ir(+Args, +FieldSeparator, +BasePrefix, -ArgValueIRs,
%%     -GlobalParts, -SetupParts)
%
%  Marshal plawk foreign-call arguments into %Value SSA names. field(0)
%  passes the record atom %line directly; positive fields intern the
%  projected slice (missing fields intern the empty atom); string
%  literals intern per-site globals; integers build integer values.
plawk_foreign_args_ir(Args, FieldSeparator, BasePrefix, ArgValueIRs,
        GlobalParts, SetupParts) :-
    plawk_foreign_args_ir(Args, FieldSeparator, BasePrefix, 0, ArgValueIRs,
        GlobalPartsNested, SetupPartsNested),
    append(GlobalPartsNested, GlobalParts),
    append(SetupPartsNested, SetupParts).

plawk_foreign_args_ir([], _FieldSeparator, _BasePrefix, _Index, [], [], []).
plawk_foreign_args_ir([Arg | Rest], FieldSeparator, BasePrefix, Index,
        [ArgValueIR | ArgValueIRs], [GlobalParts | GlobalPartsRest],
        [SetupParts | SetupPartsRest]) :-
    format(atom(ArgBase), '~w_a~w', [BasePrefix, Index]),
    plawk_foreign_arg_ir(Arg, FieldSeparator, ArgBase, ArgValueIR,
        GlobalParts, SetupParts),
    NextIndex is Index + 1,
    plawk_foreign_args_ir(Rest, FieldSeparator, BasePrefix, NextIndex,
        ArgValueIRs, GlobalPartsRest, SetupPartsRest).

plawk_foreign_arg_ir(field(0), _FieldSeparator, ArgBase, ArgValueIR,
        [], SetupParts) :-
    !,
    % The record Value is the transient line atom whose buffer mutates
    % on the next read; Prolog-side atom identity (X == 'ERROR') and
    % anything the predicate might persist need a real atom, so $0
    % interns the current line text. %line_s is the C string the
    % driver's EOF check already resolved.
    SafeBase = ArgBase,
    format(atom(LenIR),
        '  %~w_len = call i64 @strlen(i8* %line_s)', [SafeBase]),
    format(atom(InternIR),
        '  %~w_id = call i64 @wam_intern_atom(i8* %line_s, i64 %~w_len)',
        [SafeBase, SafeBase]),
    format(atom(Value0IR),
        '  %~w_v0 = insertvalue %Value undef, i32 0, 0', [SafeBase]),
    format(atom(ValueIR),
        '  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]),
    SetupParts = [LenIR, InternIR, Value0IR, ValueIR].
plawk_foreign_arg_ir(field(FieldIndex), FieldSeparator, ArgBase, ArgValueIR,
        [EmptyGlobalIR], SetupParts) :-
    integer(FieldIndex),
    FieldIndex > 0,
    !,
    SafeBase = ArgBase,
    format(atom(EmptyGlobalIR),
        '@.~w_empty = private constant [1 x i8] zeroinitializer', [SafeBase]),
    format(atom(SliceIR),
        '  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [SafeBase, FieldIndex, FieldSeparator]),
    format(atom(PtrIR),
        '  %~w_ptr = extractvalue %WamSlice %~w_slice, 0', [SafeBase, SafeBase]),
    format(atom(LenIR),
        '  %~w_len = extractvalue %WamSlice %~w_slice, 1', [SafeBase, SafeBase]),
    format(atom(NullIR),
        '  %~w_null = icmp eq i8* %~w_ptr, null', [SafeBase, SafeBase]),
    format(atom(SafePtrIR),
        '  %~w_safe_ptr = select i1 %~w_null, i8* getelementptr ([1 x i8], [1 x i8]* @.~w_empty, i32 0, i32 0), i8* %~w_ptr',
        [SafeBase, SafeBase, SafeBase, SafeBase]),
    format(atom(SafeLenIR),
        '  %~w_safe_len = select i1 %~w_null, i64 0, i64 %~w_len',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(InternIR),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_safe_ptr, i64 %~w_safe_len)',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(Value0IR),
        '  %~w_v0 = insertvalue %Value undef, i32 0, 0', [SafeBase]),
    format(atom(ValueIR),
        '  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]),
    SetupParts = [SliceIR, PtrIR, LenIR, NullIR, SafePtrIR, SafeLenIR,
        InternIR, Value0IR, ValueIR].
plawk_foreign_arg_ir(string(String), _FieldSeparator, ArgBase, ArgValueIR,
        [StringGlobalIR], SetupParts) :-
    !,
    SafeBase = ArgBase,
    format(atom(StringGlobalName), '~w_str', [SafeBase]),
    llvm_emit_c_string_global(StringGlobalName, String, StringGlobalIR,
        StringLen, BytesLen),
    format(atom(PtrIR),
        '  %~w_str_ptr = getelementptr [~w x i8], [~w x i8]* @.~w_str, i32 0, i32 0',
        [SafeBase, BytesLen, BytesLen, SafeBase]),
    format(atom(InternIR),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_str_ptr, i64 ~w)',
        [SafeBase, SafeBase, StringLen]),
    format(atom(Value0IR),
        '  %~w_v0 = insertvalue %Value undef, i32 0, 0', [SafeBase]),
    format(atom(ValueIR),
        '  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]),
    SetupParts = [PtrIR, InternIR, Value0IR, ValueIR].
plawk_foreign_arg_ir(int(Value), _FieldSeparator, ArgBase, ArgValueIR,
        [], [IntIR]) :-
    integer(Value),
    SafeBase = ArgBase,
    format(atom(IntIR),
        '  %~w_v = call %Value @value_integer(i64 ~w)', [SafeBase, Value]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]).

plawk_foreign_call_args_ir([], '').
plawk_foreign_call_args_ir(ArgValueIRs, IR) :-
    ArgValueIRs = [_ | _],
    findall(Part,
        ( member(ArgValueIR, ArgValueIRs),
          format(atom(Part), '%Value ~w', [ArgValueIR])
        ),
        Parts),
    atomic_list_concat(Parts, ', ', IR).

plawk_foreign_guard_call_ir(Name, Args, FieldSeparator, GlobalBase, MatchValue,
        GlobalIR-GuardCallIR) :-
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, SetupParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(CallLine),
        '  ~w = call i1 @plawk_foreign_guard_~w_~w(~w)',
        [MatchValue, Name, NArgs, CallArgsIR]),
    append(SetupParts, [CallLine], Lines),
    atomic_list_concat(Lines, '\n', GuardCallIR).

plawk_literal_contains_guard_ir(GlobalBase, Needle, FieldSeparator, MatchValue, GlobalIR-GuardCallIR) :-
    format(atom(IndexBase), '~w_contains_index', [GlobalBase]),
    llvm_emit_atom_field_index(GlobalBase, '%line', 0, Needle, FieldSeparator,
        IndexBase, GlobalIR-IndexCallIR),
    format(atom(GuardCallIR),
'~w
  ~w = icmp sgt i64 %~w, 0',
        [IndexCallIR, MatchValue, IndexBase]).

plawk_field_cmp_op_code(eq, 0).
plawk_field_cmp_op_code(ne, 1).
plawk_field_cmp_op_code(lt, 2).
plawk_field_cmp_op_code(le, 3).
plawk_field_cmp_op_code(gt, 4).
plawk_field_cmp_op_code(ge, 5).

plawk_print_record_counter_ir(Fields, LoopPhiIR, RecordCounterIR) :-
    (   plawk_fields_include_nr(Fields)
    ->  LoopPhiIR = '  %plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]',
        RecordCounterIR = '  %current_nr = add i64 %plawk_nr, 1'
    ;   LoopPhiIR = '',
        RecordCounterIR = ''
    ).

plawk_fields_include_nr(Fields) :-
    member(Field, Fields),
    plawk_expr_uses_nr(Field).

plawk_expr_uses_nr(special('NR')).
plawk_expr_uses_nr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_uses_nr(Left)
    ; plawk_expr_uses_nr(Right)
    ).

plawk_print_action_ir([field(0)], _FieldSeparator, _OutputSeparator, ''-IR) :-
    !,
    llvm_emit_printf_string(plawk_surface_print_line, 4, fmt, printed, '%line_s',
        [FmtPtr, PrintCall]),
    atomic_list_concat([FmtPtr, PrintCall], '\n', IR).
plawk_print_action_ir(Fields, FieldSeparator, OutputSeparator, GlobalIR-IR) :-
    phrase(plawk_print_fields_ir(Fields, FieldSeparator, OutputSeparator, 0), Pairs),
    plawk_print_ir_parts(Pairs, GlobalParts, BodyParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(BodyParts, '\n', IR).

plawk_output_action_exprs(print(Fields), Fields).
plawk_output_action_exprs(printf(string(_Format), Args), Args).

plawk_output_action_ir(print(Fields), FieldSeparator, OutputSeparator, Pair) :-
    plawk_print_action_ir(Fields, FieldSeparator, OutputSeparator, Pair).
plawk_output_action_ir(printf(string(Format), Args), FieldSeparator, _OutputSeparator, Pair) :-
    plawk_prefixed_printf_action_ir(Format, Args, FieldSeparator, plawk_printf, Pair).

plawk_prefixed_print_action_ir([field(0)], _FieldSeparator, _OutputSeparator, Prefix, ''-IR) :-
    !,
    format(atom(FmtVar), '~w_line_fmt', [Prefix]),
    format(atom(PrintVar), '~w_printed_line', [Prefix]),
    llvm_emit_printf_string(plawk_surface_print_line, 4, FmtVar, PrintVar, '%line_s',
        [FmtPtr, PrintCall]),
    atomic_list_concat([FmtPtr, PrintCall], '\n', IR).
plawk_prefixed_print_action_ir(Fields, FieldSeparator, OutputSeparator, Prefix, GlobalIR-IR) :-
    phrase(plawk_prefixed_print_fields_ir(Fields, FieldSeparator, OutputSeparator, Prefix, 0), Pairs),
    plawk_print_ir_parts(Pairs, GlobalParts, BodyParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(BodyParts, '\n', IR).

plawk_prefixed_printf_action_ir(Format, Args, FieldSeparator, Prefix, GlobalIR-IR) :-
    phrase(plawk_printf_arg_pairs(Args, FieldSeparator, Prefix, 0), ArgPairs),
    pairs_keys_values(ArgPairs, ArgGlobalParts, ArgInfoPairs),
    pairs_keys_values(ArgInfoPairs, ArgSetupParts, ArgCallArgLists),
    append(ArgCallArgLists, CallArgs),
    maplist(plawk_printf_call_arg_kind, CallArgs, ArgKinds),
    plawk_printf_rewrite_format(Format, ArgKinds, PrintfFormat),
    format(atom(FormatGlobal), '~w_fmt', [Prefix]),
    llvm_emit_c_string_global(FormatGlobal, PrintfFormat, FormatGlobalIR, _FormatLen, FormatBytesLen),
    format(atom(FmtPtrVar), '~w_fmt_ptr', [Prefix]),
    format(atom(FmtPtr),
        '  %~w = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [FmtPtrVar, FormatBytesLen, FormatBytesLen, FormatGlobal]),
    plawk_printf_call_args_ir(CallArgs, CallArgsIR),
    format(atom(PrintVar), '~w_printed', [Prefix]),
    (   CallArgsIR == ''
    ->  format(atom(PrintCall),
            '  %~w = call i32 (i8*, ...) @printf(i8* %~w)',
            [PrintVar, FmtPtrVar])
    ;   format(atom(PrintCall),
            '  %~w = call i32 (i8*, ...) @printf(i8* %~w, ~w)',
            [PrintVar, FmtPtrVar, CallArgsIR])
    ),
    plawk_join_nonempty_ir([FormatGlobalIR | ArgGlobalParts], GlobalIR),
    append(ArgSetupParts, [FmtPtr, PrintCall], BodyParts),
    atomic_list_concat(BodyParts, '\n', IR).

plawk_printf_arg_pairs([], _FieldSeparator, _Prefix, _Index) -->
    [].
plawk_printf_arg_pairs([Arg | Args], FieldSeparator, Prefix, Index) -->
    { plawk_emit_prefixed_print_expr_ir(Arg, FieldSeparator, Prefix, Index,
          Type, GlobalParts, SetupParts),
      plawk_printf_type_call_args(Type, CallArgs),
      plawk_join_nonempty_ir(GlobalParts, GlobalIR),
      plawk_join_nonempty_ir(SetupParts, SetupIR),
      NextIndex is Index + 1
    },
    [GlobalIR-(SetupIR-CallArgs)],
    plawk_printf_arg_pairs(Args, FieldSeparator, Prefix, NextIndex).

plawk_printf_type_call_args(i64(_FmtPrefix, _PrintPrefix, ValueIR), [i64(ValueIR)]).
plawk_printf_type_call_args(slice(_FmtPrefix, _PrintPrefix, LenIR, PtrIR), [slice_len(LenIR), slice_ptr(PtrIR)]).
plawk_printf_type_call_args(string(_Base, PtrIR), [string_ptr(PtrIR)]).
plawk_printf_type_call_args(f64(_FmtPrefix, _PrintPrefix, ValueIR), [f64(ValueIR)]).

plawk_printf_call_arg_kind(i64(_ValueIR), i64).
plawk_printf_call_arg_kind(slice_len(_LenIR), slice_len).
plawk_printf_call_arg_kind(slice_ptr(_PtrIR), slice_ptr).
plawk_printf_call_arg_kind(string_ptr(_PtrIR), string).
plawk_printf_call_arg_kind(f64(_ValueIR), f64).

plawk_printf_call_args_ir([], '') :-
    !.
plawk_printf_call_args_ir(CallArgs, IR) :-
    maplist(plawk_printf_call_arg_ir, CallArgs, Parts),
    atomic_list_concat(Parts, ', ', IR).

plawk_printf_call_arg_ir(i64(ValueIR), IR) :-
    format(atom(IR), 'i64 ~w', [ValueIR]).
plawk_printf_call_arg_ir(slice_len(LenIR), IR) :-
    format(atom(IR), 'i32 ~w', [LenIR]).
plawk_printf_call_arg_ir(slice_ptr(PtrIR), IR) :-
    format(atom(IR), 'i8* ~w', [PtrIR]).
plawk_printf_call_arg_ir(string_ptr(PtrIR), IR) :-
    format(atom(IR), 'i8* ~w', [PtrIR]).
plawk_printf_call_arg_ir(f64(ValueIR), IR) :-
    format(atom(IR), 'double ~w', [ValueIR]).

plawk_printf_rewrite_format(Format, ArgKinds, RewrittenFormat) :-
    string_codes(Format, Codes),
    plawk_printf_rewrite_codes(Codes, ArgKinds, RewrittenCodes),
    string_codes(RewrittenFormat, RewrittenCodes).

plawk_printf_rewrite_codes([], [], []).
plawk_printf_rewrite_codes([Code | Rest], Kinds, [Code | RewrittenRest]) :-
    Code =\= 0'%,
    !,
    plawk_printf_rewrite_codes(Rest, Kinds, RewrittenRest).
plawk_printf_rewrite_codes([0'%, 0'% | Rest], Kinds, [0'%, 0'% | RewrittenRest]) :-
    !,
    plawk_printf_rewrite_codes(Rest, Kinds, RewrittenRest).
plawk_printf_rewrite_codes([0'% | Rest], [i64 | Kinds], [0'%, 0'l, 0'd | RewrittenRest]) :-
    plawk_printf_i64_spec_codes(Rest, RestAfterSpec),
    !,
    plawk_printf_rewrite_codes(RestAfterSpec, Kinds, RewrittenRest).
plawk_printf_rewrite_codes([0'%, 0's | Rest], [string | Kinds], [0'%, 0's | RewrittenRest]) :-
    !,
    plawk_printf_rewrite_codes(Rest, Kinds, RewrittenRest).
plawk_printf_rewrite_codes([0'%, 0's | Rest], [slice_len, slice_ptr | Kinds],
        [0'%, 0'., 0'*, 0's | RewrittenRest]) :-
    !,
    plawk_printf_rewrite_codes(Rest, Kinds, RewrittenRest).
plawk_printf_rewrite_codes([0'% | Rest], [f64 | Kinds], [0'% | RewrittenSpec]) :-
    plawk_printf_f64_spec_codes(Rest, SpecCodes, RestAfterSpec),
    !,
    append(SpecCodes, RewrittenRest, RewrittenSpec),
    plawk_printf_rewrite_codes(RestAfterSpec, Kinds, RewrittenRest).

% %f / %g / %e with an optional .N precision pass through unchanged for
% double arguments (C varargs receive the double directly).
plawk_printf_f64_spec_codes([0'., Digit | Rest0], [0'., Digit | SpecRest], Rest) :-
    code_type(Digit, digit),
    !,
    plawk_printf_f64_precision_codes(Rest0, SpecRest, Rest).
plawk_printf_f64_spec_codes([Conv | Rest], [Conv], Rest) :-
    plawk_printf_f64_conversion_code(Conv).

plawk_printf_f64_precision_codes([Digit | Rest0], [Digit | SpecRest], Rest) :-
    code_type(Digit, digit),
    !,
    plawk_printf_f64_precision_codes(Rest0, SpecRest, Rest).
plawk_printf_f64_precision_codes([Conv | Rest], [Conv], Rest) :-
    plawk_printf_f64_conversion_code(Conv).

plawk_printf_f64_conversion_code(0'f).
plawk_printf_f64_conversion_code(0'g).
plawk_printf_f64_conversion_code(0'e).

plawk_printf_i64_spec_codes([0'l, 0'd | Rest], Rest).
plawk_printf_i64_spec_codes([0'd | Rest], Rest).
plawk_printf_i64_spec_codes([0'i | Rest], Rest).

plawk_print_ir_parts([], [], []).
plawk_print_ir_parts([GlobalIR-BodyIR | Parts], [GlobalIR | GlobalParts], [BodyIR | BodyParts]) :-
    plawk_print_ir_parts(Parts, GlobalParts, BodyParts).

plawk_join_nonempty_ir(Parts, IR) :-
    include(plawk_nonempty_ir, Parts, NonEmptyParts),
    atomic_list_concat(NonEmptyParts, '\n', IR).

plawk_nonempty_ir(IR) :-
    IR \== ''.

plawk_print_fields_ir([], _FieldSeparator, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          newline_fmt, printed_newline, [FmtPtr, PrintCall])
    },
    [''-FmtPtr, ''-PrintCall].
plawk_print_fields_ir([Field | Rest], FieldSeparator, OutputSeparator, Index) -->
    plawk_print_separator_ir(Index, OutputSeparator),
    plawk_print_field_ir(Field, FieldSeparator, Index),
    { NextIndex is Index + 1 },
    plawk_print_fields_ir(Rest, FieldSeparator, OutputSeparator, NextIndex).

plawk_prefixed_print_fields_ir([], _FieldSeparator, _OutputSeparator, Prefix, _) -->
    { format(atom(FmtVar), '~w_newline_fmt', [Prefix]),
      format(atom(PrintVar), '~w_printed_newline', [Prefix]),
      llvm_emit_printf0(plawk_surface_print_newline, 2, FmtVar, PrintVar,
          [FmtPtr, PrintCall])
    },
    [''-FmtPtr, ''-PrintCall].
plawk_prefixed_print_fields_ir([Field | Rest], FieldSeparator, OutputSeparator, Prefix, Index) -->
    plawk_prefixed_print_separator_ir(Index, OutputSeparator, Prefix),
    plawk_prefixed_print_field_ir(Field, FieldSeparator, Prefix, Index),
    { NextIndex is Index + 1 },
    plawk_prefixed_print_fields_ir(Rest, FieldSeparator, OutputSeparator, Prefix, NextIndex).

plawk_print_separator_ir(0, _OutputSeparator) -->
    !,
    [].
plawk_print_separator_ir(Index, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %printed_separator_~w = call i32 @putchar(i32 ~w)',
          [Index, OutputSeparator])
    },
    [''-SpaceCall].

plawk_prefixed_print_separator_ir(0, _OutputSeparator, _Prefix) -->
    !,
    [].
plawk_prefixed_print_separator_ir(Index, OutputSeparator, Prefix) -->
    { format(atom(SpaceCall),
          '  %~w_printed_separator_~w = call i32 @putchar(i32 ~w)',
          [Prefix, Index, OutputSeparator])
    },
    [''-SpaceCall].

plawk_print_field_ir(Field, FieldSeparator, Index) -->
    { plawk_emit_print_expr_ir(Field, FieldSeparator, Index, Type, GlobalParts, SetupParts),
      plawk_print_expr_output_ir(Type, Index, PrintParts),
      plawk_join_nonempty_ir(GlobalParts, GlobalIR),
      append(SetupParts, PrintParts, BodyParts),
      atomic_list_concat(BodyParts, '\n', BodyIR)
    },
    [GlobalIR-BodyIR].

plawk_prefixed_print_field_ir(Field, FieldSeparator, Prefix, Index) -->
    { plawk_emit_prefixed_print_expr_ir(Field, FieldSeparator, Prefix, Index,
          Type, GlobalParts, SetupParts),
      plawk_prefixed_print_expr_output_ir(Type, Prefix, Index, PrintParts),
      plawk_join_nonempty_ir(GlobalParts, GlobalIR),
      append(SetupParts, PrintParts, BodyParts),
      atomic_list_concat(BodyParts, '\n', BodyIR)
    },
    [GlobalIR-BodyIR].

plawk_emit_print_expr_ir(Field, FieldSeparator, Index, Type, GlobalParts, SetupParts) :-
    plawk_emit_print_expr_for_context(Field, FieldSeparator, print_context(normal, '', Index),
        Type, GlobalParts, SetupParts).

plawk_emit_prefixed_print_expr_ir(Field, FieldSeparator, Prefix, Index,
        Type, GlobalParts, SetupParts) :-
    plawk_emit_print_expr_for_context(Field, FieldSeparator,
        print_context(prefixed, Prefix, Index), Type, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(special('NR'), _FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_output_names(Context, nr, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(nr, 0, nr, nr, ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(string(Value), _FieldSeparator, Context,
        string(Base, PtrIR), [GlobalIR], [StringPtr]) :-
    plawk_print_expr_value_base(Context, string, Base),
    llvm_emit_c_string_global(Base, Value, GlobalIR, _StringLen, BytesLen),
    format(atom(StringPtr),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [Base, BytesLen, BytesLen, Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_emit_print_expr_for_context(special('NF'), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, nf, Base),
    plawk_print_expr_output_names(Context, nf, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(nf, FieldSeparator, Base, Base, ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(int(field(FieldIndex)), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, int, Base),
    plawk_print_expr_output_names(Context, int, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(Expr, FieldSeparator, Context,
        f64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_f64_print_expr(Expr),
    !,
    plawk_print_expr_value_base(Context, f64, Base),
    plawk_print_expr_output_names(Context, f64, FmtPrefix, PrintPrefix),
    plawk_f64_expr_ir(Expr, FieldSeparator, Base, Base, ValueIR,
        GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(Expr, FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, NamePart, _Left, _Right),
    plawk_i64_binary_print_kind(NamePart, Kind),
    plawk_print_expr_value_base(Context, Kind, Base),
    plawk_print_expr_output_names(Context, Kind, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(Expr, FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(prolog_call(Name, Args), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_prolog_call_expr(prolog_call(Name, Args)),
    plawk_print_expr_value_base(Context, prolog_call, Base),
    plawk_print_expr_output_names(Context, prolog_call, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(prolog_call(Name, Args), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(length(field(FieldIndex)), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, length, Base),
    plawk_print_expr_output_names(Context, length, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(length(FieldIndex), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(substr(field(FieldIndex), Start, Len), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), [], [SliceIR]) :-
    plawk_print_expr_value_base(Context, substr, Base),
    plawk_print_expr_output_names(Context, substr, FmtPrefix, PrintPrefix),
    llvm_emit_atom_field_subslice('%line', FieldIndex, FieldSeparator, Start, Len, Base, SliceIR),
    format(atom(LenIR), '%~w_len', [Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_emit_print_expr_for_context(index(field(FieldIndex), string(Needle)), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, index, Base),
    plawk_print_expr_value_base(Context, index_needle, GlobalBase),
    plawk_print_expr_output_names(Context, index, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(index(FieldIndex, Needle), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(tolower(field(FieldIndex)), FieldSeparator, Context,
        case_slice(lower, LowerBase, LenIR, PtrIR), [], SetupParts) :-
    plawk_print_expr_value_base(Context, tolower, LowerBase),
    plawk_emit_case_source_slice_ir(FieldIndex, FieldSeparator, LowerBase, LenIR, PtrIR,
        SetupParts).

plawk_emit_print_expr_for_context(toupper(field(FieldIndex)), FieldSeparator, Context,
        case_slice(upper, UpperBase, LenIR, PtrIR), [], SetupParts) :-
    plawk_print_expr_value_base(Context, toupper, UpperBase),
    plawk_emit_case_source_slice_ir(FieldIndex, FieldSeparator, UpperBase, LenIR, PtrIR,
        SetupParts).

plawk_emit_print_expr_for_context(field(0), _FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, '%line_s'), [], [LineLen64, LineLen]) :-
    plawk_print_expr_output_names(Context, line, FmtPrefix, PrintPrefix),
    plawk_emit_print_line_length_ir(Context, LenIR, LineLen64, LineLen).

plawk_emit_print_expr_for_context(field(FieldIndex), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), [], [SliceIR]) :-
    FieldIndex > 0,
    plawk_print_expr_value_base(Context, field, Base),
    plawk_print_expr_output_names(Context, field, FmtPrefix, PrintPrefix),
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, Base, SliceIR),
    format(atom(LenIR), '%~w_len', [Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_print_expr_value_base(print_context(normal, _Prefix, Index), Kind, Base) :-
    plawk_normal_print_expr_value_base(Kind, Index, Base).
plawk_print_expr_value_base(print_context(prefixed, Prefix, Index), Kind, Base) :-
    format(atom(Base), '~w_~w_~w', [Prefix, Kind, Index]).

plawk_normal_print_expr_value_base(nf, Index, Base) :-
    format(atom(Base), 'plawk_nf_~w', [Index]).
plawk_normal_print_expr_value_base(int, Index, Base) :-
    format(atom(Base), 'plawk_int_~w', [Index]).
plawk_normal_print_expr_value_base(int_add, Index, Base) :-
    format(atom(Base), 'plawk_int_add_~w', [Index]).
plawk_normal_print_expr_value_base(int_sub, Index, Base) :-
    format(atom(Base), 'plawk_int_sub_~w', [Index]).
plawk_normal_print_expr_value_base(int_mul, Index, Base) :-
    format(atom(Base), 'plawk_int_mul_~w', [Index]).
plawk_normal_print_expr_value_base(int_div, Index, Base) :-
    format(atom(Base), 'plawk_int_div_~w', [Index]).
plawk_normal_print_expr_value_base(int_mod, Index, Base) :-
    format(atom(Base), 'plawk_int_mod_~w', [Index]).
plawk_normal_print_expr_value_base(prolog_call, Index, Base) :-
    format(atom(Base), 'plawk_prolog_call_~w', [Index]).
plawk_normal_print_expr_value_base(f64, Index, Base) :-
    format(atom(Base), 'plawk_f64_~w', [Index]).
plawk_normal_print_expr_value_base(length, Index, Base) :-
    format(atom(Base), 'plawk_length_~w', [Index]).
plawk_normal_print_expr_value_base(substr, Index, Base) :-
    format(atom(Base), 'plawk_substr_~w', [Index]).
plawk_normal_print_expr_value_base(index, Index, Base) :-
    format(atom(Base), 'plawk_index_~w', [Index]).
plawk_normal_print_expr_value_base(index_needle, Index, Base) :-
    format(atom(Base), 'plawk_index_needle_~w', [Index]).
plawk_normal_print_expr_value_base(tolower, Index, Base) :-
    format(atom(Base), 'plawk_tolower_~w', [Index]).
plawk_normal_print_expr_value_base(toupper, Index, Base) :-
    format(atom(Base), 'plawk_toupper_~w', [Index]).
plawk_normal_print_expr_value_base(field, Index, Base) :-
    format(atom(Base), 'plawk_field_~w', [Index]).
plawk_normal_print_expr_value_base(string, Index, Base) :-
    format(atom(Base), 'plawk_string_~w', [Index]).

plawk_emit_print_line_length_ir(print_context(normal, _Prefix, Index), LenIR, LineLen64, LineLen) :-
    format(atom(LineLen64),
        '  %line_len64_~w = call i64 @strlen(i8* %line_s)',
        [Index]),
    format(atom(LineLen),
        '  %line_len_~w = trunc i64 %line_len64_~w to i32',
        [Index, Index]),
    format(atom(LenIR), '%line_len_~w', [Index]).
plawk_emit_print_line_length_ir(print_context(prefixed, Prefix, Index), LenIR, LineLen64, LineLen) :-
    format(atom(Base), '~w_line_~w', [Prefix, Index]),
    format(atom(LineLen64),
        '  %~w_len64 = call i64 @strlen(i8* %line_s)',
        [Base]),
    format(atom(LineLen),
        '  %~w_len = trunc i64 %~w_len64 to i32',
        [Base, Base]),
    format(atom(LenIR), '%~w_len', [Base]).

plawk_print_expr_output_names(print_context(normal, _Prefix, _Index), field, slice, slice) :-
    !.
plawk_print_expr_output_names(print_context(normal, _Prefix, _Index), Kind, Kind, Kind).
plawk_print_expr_output_names(print_context(prefixed, Prefix, Index), Kind, Base, Base) :-
    format(atom(Base), '~w_~w_~w', [Prefix, Kind, Index]).

plawk_emit_case_source_slice_ir(0, _FieldSeparator, Base, LenIR, '%line_s', [LineLen64]) :-
    format(atom(LineLen64),
        '  %~w_len64 = call i64 @strlen(i8* %line_s)',
        [Base]),
    format(atom(LenIR), '%~w_len64', [Base]).
plawk_emit_case_source_slice_ir(FieldIndex, FieldSeparator, Base, LenIR, PtrIR, [SliceIR]) :-
    FieldIndex > 0,
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, Base, SliceIR),
    format(atom(LenIR), '%~w_len64', [Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_print_expr_output_ir(i64(FmtPrefix, PrintPrefix, ValueIR), Index, Parts) :-
    format(atom(FmtVar), '~w_fmt_~w', [FmtPrefix, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [PrintPrefix, Index]),
    llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR, Parts).

plawk_print_expr_output_ir(slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), Index, Parts) :-
    format(atom(FmtVar), '~w_fmt_~w', [FmtPrefix, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [PrintPrefix, Index]),
    llvm_emit_printf_slice(plawk_surface_print_slice, FmtVar, PrintVar, LenIR, PtrIR,
        Parts).

plawk_print_expr_output_ir(f64(FmtPrefix, PrintPrefix, ValueIR), Index, [FmtPtr, PrintCall]) :-
    format(atom(FmtVar), '~w_fmt_~w', [FmtPrefix, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [PrintPrefix, Index]),
    format(atom(FmtPtr),
        '  %~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
        [FmtVar]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w, double ~w)',
        [PrintVar, FmtVar, ValueIR]).

plawk_print_expr_output_ir(case_slice(Mode, PrintBase, LenIR, PtrIR), _Index, [PrintCall]) :-
    llvm_emit_ascii_case_slice_print(Mode, PtrIR, LenIR, PrintBase, PrintCall).

plawk_prefixed_print_expr_output_ir(i64(FmtPrefix, PrintPrefix, ValueIR), _Prefix, Index, Parts) :-
    plawk_print_expr_output_ir(i64(FmtPrefix, PrintPrefix, ValueIR), Index, Parts).

plawk_prefixed_print_expr_output_ir(slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), _Prefix, Index, Parts) :-
    plawk_print_expr_output_ir(slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), Index, Parts).

plawk_prefixed_print_expr_output_ir(f64(FmtPrefix, PrintPrefix, ValueIR), _Prefix, Index, Parts) :-
    plawk_print_expr_output_ir(f64(FmtPrefix, PrintPrefix, ValueIR), Index, Parts).

plawk_prefixed_print_expr_output_ir(case_slice(Mode, PrintBase, LenIR, PtrIR), _Prefix, _Index, [PrintCall]) :-
    llvm_emit_ascii_case_slice_print(Mode, PtrIR, LenIR, PrintBase, PrintCall).

plawk_prefixed_print_expr_output_ir(string(Base, PtrIR), _Prefix, Index, Parts) :-
    format(atom(FmtVar), '~w_fmt_~w', [Base, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [Base, Index]),
    llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR, Parts).
